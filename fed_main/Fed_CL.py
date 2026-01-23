import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from options import parse_args
from data_processing.dataloader_manager import gen_cbgru_valid_dl, gen_client_ds, gen_valid_dl
from data_processing.preprocessing import coordinate_sys_noise_clusters
from trainers.server import CLC_Server
from trainers.client import Fed_CLC_client
from models.ClassiFilerNet import ClassiFilerNet
from models.CGE_Variants import CGEVariant
from global_test import global_test
import random
import concurrent.futures
import time
import numpy as np


def train_one_client_clc(client_id, args, global_model, dataset, tao, conf_score=None, stage='warmup'):
    """
    Function to train one client in parallel.
    Mimics the logic inside CLC.warmup and CLC.holdout_stage loops.
    """
    criterion = nn.CrossEntropyLoss()
    client = Fed_CLC_client(
        args,
        criterion,
        None, # Model will be set below
        dataset,
        client_id,
        tao
    )
    
    # Deepcopy global model
    client.model = copy.deepcopy(global_model)
    
    if stage == 'holdout':
        if conf_score is not None:
            # We must compute confidence first to generate sfm_Mat
            client.confidence()
            client.data_holdout(conf_score)
    elif stage == 'correct':
        if conf_score is not None:
            # We must compute confidence first to generate sfm_Mat
            client.confidence()
            client.data_holdout(conf_score)
            client.data_correct()

    client.train()
    
    weights = copy.deepcopy(client.get_parameters())
    num_samples = client.result['sample']
    result = client.result
    loss = client.result['loss']
    
    del client
    torch.cuda.empty_cache()
    gc.collect()
    
    return client_id, weights, num_samples, result, loss


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
        seed=int(args.seed)
    )

    datasets_dict = dict()
    datasets_dict['train'] = list()
    for i in range(args.client_num):
        train_ds = gen_client_ds(
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
        datasets_dict['train'].append(train_ds)

    test_dl = gen_valid_dl(args.model_type, args.vul, data_dir=args.data_dir)
    
    criterion = nn.CrossEntropyLoss()
    
    # Initialize Global Model
    if args.model_type == "CBGRU":
        global_model = ClassiFilerNet(INPUT_SIZE, TIME_STAMP)
    elif args.model_type == "CGE":
        global_model = CGEVariant()
    global_model = global_model.to(args.device)

    # Initialize Server
    server = CLC_Server(args, global_model, args.device, criterion)

    tao = 0.1
    
    # -------------------------------------------------------------------------
    # Warmup Stage
    # -------------------------------------------------------------------------
    print("Starting Warmup Stage")
    server.initialize_epoch_updates(-1)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in range(args.client_num):
            futures.append(executor.submit(train_one_client_clc, i, args, server.global_model, datasets_dict['train'][i], tao, None, 'warmup'))
        
        for future in futures:
            client_id, weights, num_samples, result, loss = future.result()
            server.save_train_updates(weights, num_samples, result)
            print(f"client:{client_id}")
            print(f"Loss: {loss}")

    server.average_weights()

    # -------------------------------------------------------------------------
    # Holdout Stage
    # -------------------------------------------------------------------------
    print("Starting Holdout Stage")
    for epoch in range(args.first_epochs):
        server.initialize_epoch_updates(epoch)
        
        # 1. Gather Confidence
        confs = []
        classnums = []
        for ix in range(args.client_num):
            # Create temp client for sendconf
            temp_client = Fed_CLC_client(args, criterion, copy.deepcopy(server.global_model), datasets_dict['train'][ix], ix, tao)
            conf, classnum = temp_client.sendconf()
            confs.append(conf)
            classnums.append(classnum)
            del temp_client
            
        server.receiveconf(confs, classnums)
        conf_score = server.conf_agg()
        
        # 2. Parallel Training
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for ix in range(args.client_num):
                futures.append(executor.submit(train_one_client_clc, ix, args, server.global_model, datasets_dict['train'][ix], tao, conf_score, 'holdout'))
            
            for future in futures:
                client_id, weights, num_samples, result, loss = future.result()
                server.save_train_updates(weights, num_samples, result)
                print(f"client:{client_id}")
                print(f"Loss: {loss}")
            
        server.average_weights()
        print(epoch)

    # clc.correct_stage() is omitted as it was commented out in original file.

    global_test(server.global_model, test_dl, criterion, args, args.lab_name)
