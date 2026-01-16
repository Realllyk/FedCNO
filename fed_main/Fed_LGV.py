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
from data_processing.dataloader_manager import gen_lgv_ds, gen_valid_dl, gen_cbgru_dl, gen_client_ds
from models.ClassiFilerNet import ClassiFilerNet
from models.CGE_Variants import CGEVariant
from trainers.server import LGV_server
from trainers.client import Fed_LGV_client, Fed_Avg_client
from global_test import global_test
import random
import time


if __name__ == '__main__':
    args = parse_args()
    input_size, time_steps = 100, 300

    if args.diff == True:
        noise_rates = random.sample([0.2, 0.2, 0.3, 0.3], 4)
    else:
        noise_rates = [args.noise_rate] * 4

    train_ds = list()
    for i in range(args.client_num):
        ds = gen_client_ds(args.model_type, i, args.vul, args.noise_type, noise_rates[i], args.random_noise, args.num_neigh)
        train_ds.append(ds)

    
    # initialize Server
    # -------------------------------------------------------------------------
    # 初始化服务器 (Server Initialization)
    # -------------------------------------------------------------------------
    # 服务器负责维护全局模型 (Global Model) 并协调各客户端的训练。
    # - model_type: 支持 'CBGRU' 或 'CGE' 等不同模型架构。
    # - global_weight: 控制 LGV 算法中全局视图概率的权重。
    class_weights = torch.tensor([2.0, 1.0]).to(args.device)
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
        args.global_weight
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

        for client_id in range(args.client_num):
            # 使用普通的 Fed_Avg_client 进行热身，不涉及 KNN 和标签修正
            client = Fed_Avg_client(args,
                                criterion,
                                None,
                                train_ds[i])
            # 下发全局模型参数
            client.model = copy.deepcopy(server.global_model)
            client.train()
            # 上传更新
            server.save_train_updates(
                copy.deepcopy(client.get_parameters()),
                client.result['sample'],
                client.result
            )
            print(f"client:{client_id}")
            client.print_loss()
            del client
            torch.cuda.empty_cache()
            gc.collect()
        
        # 聚合参数 (FedAvg Aggregation)
        server.average_weights()
    
    # initialize dataset
    # 重新初始化数据集，为正式的 Fed_LGV 训练做准备
    # gen_lgv_ds 会生成支持图特征/模式特征读取的专用数据集
    train_ds = list()
    for i in range(args.client_num):
        ds = gen_lgv_ds(i, args.vul, args.noise_type, args.noise_rate, args.random_noise, args.num_neigh, args.model_type)
        train_ds.append(ds)
    print(args.random_noise)
    test_dl = gen_valid_dl(args.model_type, args.vul)

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

        # sample clients
        # selected_indices = np.random.choice(candidates, int(args.client_num*args.sample_rate), replace=False).tolist()

        # for client_id in selected_indices:
        for client_id in range(args.client_num):
            client = clients[client_id]
            del client.model
            # 1. 接收全局模型
            client.model = copy.deepcopy(server.global_model)
            client.global_weight = server.global_weight
            
            # 2. 更新全局视图 (Global View)
            # 关键步骤：利用当前全局模型提取特征，动态更新 KNN 概率和一致性
            # client.get_global_knn_labels(args.vul, args.noise_type, args.noise_rate)
            # client.get_global_prob_labels(args.vul)
            # client.get_global_feature_knn_labels()
            client.get_global_feature_global_knn_labels()
            
            # 3. 本地训练 (Local Training)
            # 融合标签 -> 生成伪标签 -> 加权训练
            client.train()
            
            # 4. 上传更新
            server.save_train_updates(
                copy.deepcopy(client.get_parameters()),
                client.result['sample'],
                client.result
            )
            print(f"client:{client_id}")
            client.print_loss()
            torch.cuda.empty_cache()
            gc.collect()
        
        # 5. 服务器聚合 (Aggregation)
        server.average_weights()
        # if epoch % 5 == 0:
        #     server.autotune_gr(test_dl)
    
    global_test(server.global_model, test_dl, criterion, args, f"{args.num_neigh}neigh_{args.global_weight}_{args.lab_name}", run_timestamp=run_timestamp)
        
    
