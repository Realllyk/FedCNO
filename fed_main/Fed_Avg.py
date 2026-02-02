import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import gc
import torch
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F
from options import parse_args
from data_processing.dataloader_manager import gen_cbgru_dl, gen_cbgru_valid_dl, gen_client_ds, gen_valid_dl
from data_processing.preprocessing import coordinate_sys_noise_clusters
from models.ClassiFilerNet import ClassiFilerNet
from models.CGE_Variants import CGEVariant
from trainers.server import Server
from trainers.client import Fed_Avg_client
from global_test import global_test
import random
import concurrent.futures


def train_one_client(client_id, args, global_model, criterion, dataset):
    """
    单个客户端训练函数
    """
    client = Fed_Avg_client(args,
                        criterion,
                        None,
                        dataset)
    # print(f"Create Client {client_id}!")
    # 深拷贝全局模型，确保线程安全，每个客户端拥有独立的模型副本
    client.model = copy.deepcopy(global_model)
    client.train()
    
    # 获取训练后的参数权重
    weights = copy.deepcopy(client.get_parameters())
    num_samples = client.result['sample']
    result = client.result
    loss = client.result['loss']
    
    # 清理内存
    del client
    torch.cuda.empty_cache()
    gc.collect()
    
    return client_id, weights, num_samples, result, loss


if __name__ == '__main__':
    args = parse_args()
    INPUT_SIZE, TIME_STAMP = 100, 300

    # dataloader_dict = dict()
    # dataloader_dict['train'] = list()
    if args.diff == True:
        noise_rates = random.sample([0.2, 0.2, 0.3, 0.3], 4)
    else:
        noise_rates = [args.noise_rate] * 4

    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(int(args.seed))
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # -------------------------------------------------------------------------
    # 系统性噪声协调 (Systemic Noise Coordination)
    # -------------------------------------------------------------------------
    assigned_clusters_dict, global_cluster_map = coordinate_sys_noise_clusters(
        args.client_num, 
        args.vul, 
        args.noise_type, 
        model_type=args.model_type,
        n_clusters=args.n_clusters, 
        seed=int(args.seed),
        data_dir=args.data_dir
    )

    train_ds = list()
    for i in range(args.client_num):
        # train_dl, INPUT_SIZE, TIME_STAMP = gen_cbgru_dl(i, args.vul, args.noise_type, args.noise_rate, args.batch)
        # dataloader_dict['train'].append(train_dl)
        # ds = gen_cbgru_ds(i, args.vul, args.noise_type, args.noise_rate, args.random_noise, args.num_neigh)
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
            seed=int(args.seed),
            data_dir=args.data_dir
        )
        train_ds.append(ds)

        # DEBUG: Check if noise is applied
        if args.noise_type != 'non_noise':
            import pandas as pd
            if args.model_type == 'CBGRU':
                client_dir = os.path.join(args.data_dir, f"graduate_client_split/cbgru/{args.vul}/client_{i}/")
            elif args.model_type == 'CGE':
                client_dir = os.path.join(args.data_dir, f"graduate_client_split/cge/{args.vul}/client_{i}/")
            else:
                client_dir = os.path.join(args.data_dir, f"graduate_client_split/{args.vul}/client_{i}/")
            labels_path = os.path.join(client_dir, f"label_train.csv")
            if os.path.exists(labels_path):
                clean_labels = pd.read_csv(labels_path, header=None).iloc[:, 0].values
                noise_labels = np.array(ds.labels)
                diff = np.sum(clean_labels != noise_labels)
                print(f"[DEBUG] Client {i}: Noise Rate={noise_rates[i]}, Clean vs Noisy Diff={diff}/{len(clean_labels)} ({diff/len(clean_labels):.4f})")
            else:
                print(f"[DEBUG] Client {i}: Label file not found at {labels_path}")

    criterion = nn.CrossEntropyLoss()

    if args.model_type == "CBGRU":
        global_model = ClassiFilerNet(INPUT_SIZE, TIME_STAMP)
    elif args.model_type == "CGE":
        global_model = CGEVariant()
    global_model = global_model.to(args.device)

    server = Server(
        args,
        global_model,
        args.device,
        criterion
    )

    # client_list = list()
    # for i in range(args.client_num):
    #     model = ClassiFilerNet(INPUT_SIZE, TIME_STAMP)
    #     model = model.to(args.device)

    # test_dl = gen_cbgru_valid_dl(args.vul)
    test_dl = gen_valid_dl(args.model_type, args.vul, data_dir=args.data_dir)
    
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Initialize clients list (Commented out as it is unused in parallel training)
    # clients = []
    # for client_id in range(args.client_num):
    #     client = Fed_Avg_client(args,
    #                         criterion,
    #                         copy.deepcopy(server.global_model),
    #                         train_ds[client_id],
    #                         client_id=client_id,
    #                         run_timestamp=run_timestamp)
    #     clients.append(client)
    
    for epoch in range(args.epoch):
        server.initialize_epoch_updates(epoch)

        # Parallel training
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            for client_id in range(args.client_num):
                # Correctly using client_id to access the corresponding dataset
                futures.append(executor.submit(train_one_client, client_id, args, server.global_model, criterion, train_ds[client_id]))
            
            for future in futures:
                client_id, weights, num_samples, result, loss = future.result()
                server.save_train_updates(
                        weights,
                        num_samples,
                        result
                )
                print(f"client:{client_id}")
                print(f"loss is {loss}")

        server.average_weights()
        # CBGRU_test(server.global_model, test_dl, criterion, args, 'Fed_CBGRU')
        print(epoch)
    
    
    global_test(server.global_model, test_dl, criterion, args, args.lab_name, run_timestamp=run_timestamp)
