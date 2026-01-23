import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy 
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from options import parse_args
from data_processing.dataloader_manager import gen_lgv_ds, gen_test_dl, gen_cbgru_dl, gen_client_ds, gen_valid_dl
from data_processing.preprocessing import compute_global_clusters, coordinate_sys_noise_clusters
from models.ClassiFilerNet import ClassiFilerNet
from models.CGE_Variants import CGEVariant
from trainers.server import LGV_server
from trainers.client import Fed_LGV_client, Fed_Avg_client
from global_test import global_test
import random
import time
import concurrent.futures


def train_warmup_client(client_id, args, global_model, criterion, dataset):
    """
    热身阶段单个客户端训练函数
    """
    client = Fed_Avg_client(args,
                        criterion,
                        None,
                        dataset)
    # 下发全局模型参数
    client.model = copy.deepcopy(global_model)
    client.train()
    
    weights = copy.deepcopy(client.get_parameters())
    num_samples = client.result['sample']
    result = client.result
    
    del client
    torch.cuda.empty_cache()
    gc.collect()
    
    return client_id, weights, num_samples, result


def train_lgv_client(client_id, client, global_model, global_weight):
    """
    LGV正式训练阶段单个客户端训练函数
    """
    if hasattr(client, 'model'):
        del client.model
        
    # 1. 接收全局模型
    client.model = copy.deepcopy(global_model)
    client.global_weight = global_weight
    
    # 2. 更新全局视图 (Global View)
    # 关键步骤：利用当前全局模型提取特征，动态更新 KNN 概率和一致性
    client.get_global_feature_global_knn_labels()
    
    # 3. 本地训练 (Local Training)
    # 融合标签 -> 生成伪标签 -> 加权训练
    client.train()
    
    weights = copy.deepcopy(client.get_parameters())
    num_samples = client.result['sample']
    result = client.result
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return client_id, weights, num_samples, result


if __name__ == '__main__':
    args = parse_args()
    input_size, time_steps = 100, 300

    if args.diff == True:
        noise_rates = random.sample([0.2, 0.2, 0.3, 0.3], 4)
    else:
        noise_rates = [args.noise_rate] * 4

    # -------------------------------------------------------------------------
    # 系统性噪声协调 (Systemic Noise Coordination)
    # -------------------------------------------------------------------------
    # 如果启用了系统性噪声 (sys_noise)，我们希望不同客户端的噪声模式是“错开”的。
    # 方案：预先分配簇 ID 给每个客户端。
    # 逻辑已封装在 coordinate_sys_noise_clusters 中，包含基于分布的优化分配。
    
    assigned_clusters_dict, global_cluster_map = coordinate_sys_noise_clusters(
        args.client_num, 
        args.vul, 
        args.noise_type, 
        n_clusters=args.n_clusters, 
        seed=int(args.seed)
    )

    train_ds = list()
    for i in range(args.client_num):
        ds = gen_client_ds(
            args.model_type, 
            i, 
            args.vul, 
            args.noise_type, 
            noise_rates[i], 
            args.random_noise, 
            args.num_neigh,
            assigned_clusters=assigned_clusters_dict, 
            global_cluster_map=global_cluster_map,
            n_clusters=args.n_clusters,
            seed=int(args.seed)
        )
        train_ds.append(ds)
    
    # Collect noise labels from warm-up datasets to reuse in LGV datasets
    generated_noise_labels = [ds.labels for ds in train_ds]
    
    # initialize Server
    # -------------------------------------------------------------------------
    # 初始化服务器 (Server Initialization)
    # -------------------------------------------------------------------------
    # 服务器负责维护全局模型 (Global Model) 并协调各客户端的训练。
    # - model_type: 支持 'CBGRU' 或 'CGE' 等不同模型架构。
    # - global_weight: 控制 LGV 算法中全局视图概率的权重。
    # 动态调整类别权重 (Class Weighting)
    # 根据漏洞类型 (args.vul) 设置不同的权重策略：
    # 1. reentrancy (重入漏洞): 数据分布可能较均衡，或者需要轻微的权重调整，使用 [1.0, 1.0] (不加权) 或 [1.0, 1.2]。
    # 2. timestamp (时间戳依赖): 存在严重的漏报 (High FNR)，需要大幅提高 Positive 权重，使用 [1.0, 2.0]。
    # 3. 其他类型: 默认使用 [1.0, 1.0] 或 [1.0, 1.5] 作为保守策略。
    
    if args.vul == 'reentrancy':
        class_weights = torch.tensor([1.0, 1.0]).to(args.device) # 重入漏洞暂时不加权，或者微调
    elif args.vul == 'timestamp':
        class_weights = torch.tensor([1.2, 1.5]).to(args.device) # 稍微增加Negative权重以控制误报，降低Positive权重以减少FPR=1.0
    else:
        class_weights = torch.tensor([1.0, 1.0]).to(args.device) # 默认情况

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    if args.model_type == "CBGRU":
        global_model = ClassiFilerNet(input_size, time_steps)
        reduction = 'none'
    elif args.model_type == "CGE":
        global_model = CGEVariant()
        reduction = 'mean'
    global_model = global_model.to(args.device)
    server = LGV_server(
        args,
        global_model,
        args.device,
        criterion,
        args.global_weight,
        run_timestamp=time.strftime("%Y%m%d_%H%M%S")
    )

    #  Warm up
    # -------------------------------------------------------------------------
    # 阶段 1: 热身训练 (Warm-up Phase)
    # -------------------------------------------------------------------------
    # 在正式启用 Fed_LGV 逻辑之前，先使用标准的 FedAvg 算法进行若干轮预训练。
    # 目的：
    # 1. 让全局模型快速收敛到一个合理的初始状态。
    # 2. 为后续 LGV 阶段提取有效的全局特征奠定基础（如果模型完全随机，提取的特征就没有意义）。
    for epoch in range(args.warm_up_epoch):
        print(f"Warm Up Epoch {epoch}: ")
        server.initialize_epoch_updates(epoch)

        # 并行热身训练
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for client_id in range(args.client_num):
                futures.append(executor.submit(train_warmup_client, client_id, args, server.global_model, criterion, train_ds[client_id]))
            
            for future in futures:
                client_id, weights, num_samples, result = future.result()
                # 上传更新
                server.save_train_updates(
                    weights,
                    num_samples,
                    result
                )
                print(f"client:{client_id}")
                print(f"loss is {result['loss']}")

        # 聚合参数 (FedAvg Aggregation)
        server.average_weights()
    
    # initialize dataset
    # 重新初始化数据集，为正式的 Fed_LGV 训练做准备
    # gen_lgv_ds 会生成支持图特征/模式特征读取的专用数据集
    
    # -------------------------------------------------------------------------
    # 系统性噪声协调 (Systemic Noise Coordination)
    # -------------------------------------------------------------------------
    # 如果启用了系统性噪声 (sys_noise)，我们希望不同客户端的噪声模式是“错开”的。
    # 方案：预先分配簇 ID 给每个客户端。
    # 逻辑已封装在 coordinate_sys_noise_clusters 中，包含基于分布的优化分配。
    
    # assigned_clusters_dict, global_cluster_map = coordinate_sys_noise_clusters(
    #     args.client_num, 
    #     args.vul, 
    #     args.noise_type, 
    #     n_clusters=args.n_clusters, 
    #     seed=int(args.seed)
    # )

    train_ds = list()
    for i in range(args.client_num):
        ds = gen_lgv_ds(
            i, 
            args.vul, 
            args.noise_type, 
            args.noise_rate, 
            args.random_noise, 
            args.num_neigh, 
            args.model_type, 
            assigned_clusters=assigned_clusters_dict, 
            global_cluster_map=global_cluster_map,
            n_clusters=args.n_clusters,
            seed=int(args.seed),
            predefined_labels=generated_noise_labels[i]
        )
        train_ds.append(ds)
    print(args.random_noise)
    # test_dl 是真正的测试集 (Test Set)
    test_dl = gen_test_dl(args.model_type, args.vul)
    # valid_dl 是验证集 (Validation Set)，用于辅助调参或早停（目前代码中未使用，预留）
    valid_dl = gen_valid_dl(args.model_type, args.vul)

    # initialize Client
    # -------------------------------------------------------------------------
    # 初始化 LGV 客户端
    # -------------------------------------------------------------------------
    # 每个客户端被实例化为 Fed_LGV_client，并进行本地视角的初始化。
    clients = []
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")

    for i in range(args.client_num):
        client = Fed_LGV_client(
            args,
            nn.CrossEntropyLoss(weight=class_weights, reduction=reduction),
            copy.deepcopy(server.global_model),
            train_ds[i],
            i,
            server.global_weight,
            run_timestamp=run_timestamp
        )
        # 计算本地静态视图 (Local View)
        # 基于预训练特征运行 KNN，生成初始的概率分布和一致性，作为先验知识。
        client.get_local_knn_labels(args.vul, args.noise_type, args.noise_rate)
        # client.get_global_knn_labels(args.vul, args.noise_type, args.noise_rate)
        # 生成缩减版数据集，用于后续快速 KNN 检索
        client.gen_reduced_ds()
        clients.append(client)
        print(f"Generate Client {i}!")


    # Train Stage
    # -------------------------------------------------------------------------
    # 阶段 2: 正式训练 (Fed_LGV Training Phase)
    # -------------------------------------------------------------------------
    # 核心循环：
    # 1. 下发模型：服务器分发最新的全局模型给客户端。
    # 2. 全局视图更新：客户端利用全局模型提取特征，运行 KNN 更新标签概率和一致性。
    # 3. 本地训练：结合本地和全局视图生成伪标签，并使用一致性加权 Loss 训练模型。
    # 4. 聚合：服务器聚合客户端上传的参数。
    candidates = [i for i in range(args.client_num)]
    for epoch in range(args.epoch):
        print(f"Epoch {epoch}:")
        server.initialize_epoch_updates(epoch)

        # 并行正式训练
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # for client_id in selected_indices:
            for client_id in range(args.client_num):
                # 提交任务
                futures.append(executor.submit(train_lgv_client, client_id, clients[client_id], server.global_model, server.global_weight))
            
            for future in futures:
                client_id, weights, num_samples, result = future.result()
                # 4. 上传更新
                server.save_train_updates(
                    weights,
                    num_samples,
                    result
                )
                print(f"client:{client_id}")
                print(f"loss is {result['loss']}")
        
        # 5. 服务器聚合 (Aggregation)
        server.average_weights()
        if epoch % 5 == 0:
            # 使用验证集 (valid_dl) 而不是测试集来调整全局权重
            server.autotune_gr(valid_dl)
    
    global_test(server.global_model, test_dl, criterion, args, f"{args.num_neigh}neigh_{args.global_weight}_{args.lab_name}", run_timestamp=run_timestamp)
        
    
