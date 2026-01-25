import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import gc
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import concurrent.futures
from options import parse_args
from data_processing.dataloader_manager import gen_cbgru_valid_dl, gen_client_ds, gen_valid_dl
from data_processing.preprocessing import coordinate_sys_noise_clusters
from trainers.server import CLC_Server
from trainers.client import Fed_CLC_client
from models.ClassiFilerNet import ClassiFilerNet
from models.CGE_Variants import CGEVariant
from global_test import global_test
import random


def train_client_warmup(client, global_model):
    client.model = copy.deepcopy(global_model)
    client.train()
    weights = copy.deepcopy(client.get_parameters())
    num_samples = client.result['sample']
    result = client.result
    
    # Clean up
    torch.cuda.empty_cache()
    gc.collect()
    
    return client.client_id, weights, num_samples, result

def client_send_conf(client):
    conf, classnum = client.sendconf()
    return client.client_id, conf, classnum

def train_client_holdout(client, global_model, conf_score):
    client.model = copy.deepcopy(global_model)
    client.data_holdout(conf_score)
    # Ensure client uses the filtered dataloader
    if hasattr(client, 'data_loader'):
        client.dataloader = client.data_loader
    client.train()
    weights = copy.deepcopy(client.get_parameters())
    num_samples = client.result['sample']
    result = client.result
    
    # Clean up
    torch.cuda.empty_cache()
    gc.collect()
    
    return client.client_id, weights, num_samples, result

def prepare_client_correct(client, conf_score):
    client.data_holdout(conf_score)
    client.data_correct()
    return client.client_id

def train_client_correct(client, global_model):
    client.model = copy.deepcopy(global_model)
    # Ensure client uses the corrected dataloader
    if hasattr(client, 'data_loader'):
        client.dataloader = client.data_loader
    client.train()
    weights = copy.deepcopy(client.get_parameters())
    num_samples = client.result['sample']
    result = client.result
    
    # Clean up
    torch.cuda.empty_cache()
    gc.collect()
    
    return client.client_id, weights, num_samples, result


if __name__ == "__main__":
    args = parse_args()
    INPUT_SIZE, TIME_STAMP = 100, 300

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
        n_clusters=args.n_clusters, 
        seed=int(args.seed),
        data_dir=args.data_dir
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
            seed=int(args.seed),
            data_dir=args.data_dir
        )
        train_ds.append(ds)
        
        # DEBUG: Check if noise is applied
        if args.noise_type != 'non_noise':
            import pandas as pd
            client_dir = os.path.join(args.data_dir, f"graduate_client_split/{args.vul}/client_{i}/")
            labels_path = os.path.join(client_dir, f"label_train.csv")
            if os.path.exists(labels_path):
                clean_labels = pd.read_csv(labels_path, header=None).iloc[:, 0].values
                noise_labels = np.array(ds.labels)
                diff = np.sum(clean_labels != noise_labels)
                print(f"[DEBUG] Client {i}: Noise Rate={noise_rates[i]}, Clean vs Noisy Diff={diff}/{len(clean_labels)} ({diff/len(clean_labels):.4f})")
            else:
                print(f"[DEBUG] Client {i}: Label file not found at {labels_path}")

    test_dl = gen_valid_dl(args.model_type, args.vul, data_dir=args.data_dir)
    
    criterion = nn.CrossEntropyLoss()
    
    if args.model_type == "CBGRU":
        global_model = ClassiFilerNet(INPUT_SIZE, TIME_STAMP)
    elif args.model_type == "CGE":
        global_model = CGEVariant()
    global_model = global_model.to(args.device)

    server = CLC_Server(args, global_model, args.device, criterion)
    
    clients = []
    tao = 0.1
    for i in range(args.client_num):
        client = Fed_CLC_client(
            args,
            criterion,
            copy.deepcopy(server.global_model),
            train_ds[i],
            i,
            tao
        )
        clients.append(client)

    # Warmup Stage
    print("Warmup Stage...")
    server.initialize_epoch_updates(-1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(train_client_warmup, clients[i], server.global_model) for i in range(args.client_num)]
        for future in concurrent.futures.as_completed(futures):
            cid, weights, num_samples, result = future.result()
            server.save_train_updates(weights, num_samples, result)
            print(f"client:{cid} warmup done")
            clients[cid].print_loss()
    server.average_weights()

    # Holdout Stage
    print("Holdout Stage...")
    for epoch in range(args.first_epochs):
        server.initialize_epoch_updates(epoch)
        
        confs = [None] * args.client_num
        classnums = [None] * args.client_num
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(client_send_conf, clients[i]) for i in range(args.client_num)]
            for future in concurrent.futures.as_completed(futures):
                cid, conf, classnum = future.result()
                confs[cid] = conf
                classnums[cid] = classnum
        
        server.receiveconf(confs, classnums)
        conf_score = server.conf_agg()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(train_client_holdout, clients[i], server.global_model, conf_score) for i in range(args.client_num)]
            for future in concurrent.futures.as_completed(futures):
                cid, weights, num_samples, result = future.result()
                server.save_train_updates(weights, num_samples, result)
                print(f"client:{cid} holdout epoch {epoch} done")
                clients[cid].print_loss()
        
        server.average_weights()

    # Correct Stage
    print("Correct Stage...")
    correct_done = False
    for epoch in range(args.first_epochs, args.first_epochs+args.last_epochs):
        server.initialize_epoch_updates(epoch)
        
        if not correct_done:
            confs = [None] * args.client_num
            classnums = [None] * args.client_num
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(client_send_conf, clients[i]) for i in range(args.client_num)]
                for future in concurrent.futures.as_completed(futures):
                    cid, conf, classnum = future.result()
                    confs[cid] = conf
                    classnums[cid] = classnum
            
            server.receiveconf(confs, classnums)
            conf_score = server.conf_agg()

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(prepare_client_correct, clients[i], conf_score) for i in range(args.client_num)]
                for future in concurrent.futures.as_completed(futures):
                    pass
            
            correct_done = True
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(train_client_correct, clients[i], server.global_model) for i in range(args.client_num)]
            for future in concurrent.futures.as_completed(futures):
                cid, weights, num_samples, result = future.result()
                server.save_train_updates(weights, num_samples, result)
                print(f"client:{cid} correct epoch {epoch} done")
                clients[cid].print_loss()
        
        server.average_weights()

    global_test(server.global_model, test_dl, criterion, args, args.lab_name)
