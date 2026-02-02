import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import gc
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from options import parse_args
from data_processing.dataloader_manager import gen_client_ds, gen_valid_dl
from data_processing.preprocessing import coordinate_sys_noise_clusters
from models.ClassiFilerNet import ClassiFilerNet
from models.CGE_Variants import CGEVariant
from trainers.server import ARFL_Server
from trainers.client import Fed_ARFL_client
from global_test import global_test
import random


if __name__ == '__main__':
    args = parse_args()
    INPUT_SIZE, TIME_STAMP = 100, 300
    criterion = nn.CrossEntropyLoss()
    if args.device != "cpu":
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

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

    clients = list()
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

    for i in range(args.client_num):
        client = Fed_ARFL_client(
            args,
            criterion,
            None,
            train_ds[i],
            1.,
        )
        clients.append(client)
    total_num_samples = sum([c.num_train_samples for c in clients])

    if args.model_type == "CBGRU":
        global_model = ClassiFilerNet(INPUT_SIZE, TIME_STAMP)
    elif args.model_type == "CGE":
        global_model = CGEVariant()
    global_model = global_model.to(device)
    server = ARFL_Server(
        args,
        global_model,
        criterion,
        args.seed,
        clients,
        total_num_samples
    )

    for c in clients:
        c.model = copy.deepcopy(global_model)
        c.test()
        # 初始化测试后释放模型以节省显存
        del c.model
        c.model = None

    for epoch in range(args.epoch):
        print(f"Epoch {epoch} Training:------------------")
        server.initialize_epoch_updates(epoch)
        server.sample_clients(epoch)

        # 优化：只为选中的客户端分配模型副本，避免不必要的显存占用和 deepcopy 开销
        for c in server.selected_clients:
            if c.model is not None:
                del c.model
            c.model = copy.deepcopy(server.global_model)
        
        for i, c in enumerate(server.selected_clients):
            c.train()
            print(f"Selected Client {i} Train Loss: {c.result['loss']}")

        server.average_weights()
        server.update_alpha()
        
        # 这一轮结束了，释放模型以节省显存
        for c in server.selected_clients:
            if c.model is not None:
                del c.model
                c.model = None
    
    # test_dl = gen_cbgru_valid_dl(args.vul, 0, args.batch)
    test_dl = gen_valid_dl(args.model_type, args.vul, args.data_dir)
    global_test(server.global_model, test_dl, criterion, args, args.lab_name)



        

        

        