import sys
import os
# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy 
import gc
import numpy as np
import torch
import torch.nn as nn
from options import parse_args
from data_processing.dataloader_manager import gen_crd_ds, gen_test_dl, gen_valid_dl
from data_processing.preprocessing import coordinate_sys_noise_clusters
from models.ClassiFilerNet import ClassiFilerNet
from models.CGE_Variants import CGEVariant
from trainers.server import CRD_server
from trainers.client import Fed_CRD_client
from global_test import global_test
import random
import time
import concurrent.futures


def train_crd_client(client_id, client, global_model, ema_model, criterion):
    """
    Train one client for FedCRD
    """
    # 1. Update local model with global parameters (for training)
    # Important: Deepcopy to ensure independent training
    client.model = copy.deepcopy(global_model)
    
    # Keep a copy of EMA model for consistency calculation (theta_ema^t)
    # This provides a more stable anchor than the current global model
    ema_model_copy = copy.deepcopy(ema_model)
    ema_model_copy.eval()
    for param in ema_model_copy.parameters():
        param.requires_grad = False
        
    # 2. Local Training
    # This updates client.model to theta_k^t
    client.train()
    
    # 3. Compute Update Delta and Reliability
    local_params = client.get_parameters()
    # Delta is still computed against the *current* global model (what we started with)
    # because we want to know the update direction relative to theta^t
    global_params = global_model.state_dict()
    
    # Calculate delta = theta_k^t - theta^t
    delta = {}
    for k in local_params.keys():
        delta[k] = local_params[k] - global_params[k]
        
    # Calculate Reliability q_k^t using EMA model
    q_k, num_samples = client.get_consistency_stats(ema_model_copy)
    
    result = client.result
    loss = result.get('loss', 0.0)
    
    # Clean up
    del ema_model_copy
    torch.cuda.empty_cache()
    gc.collect()
    
    return client_id, delta, q_k, num_samples, result, loss


if __name__ == '__main__':
    args = parse_args()
    input_size, time_steps = 100, 300
    
    print(f"Starting FedCRD with {args.vul}, Noise: {args.noise_type} ({args.noise_rate})")
    print(f"Training on device: {args.device}")

    # Setup Random Seeds
    if args.diff == True:
        noise_rates = random.sample([0.2, 0.2, 0.3, 0.3], 4)
    else:
        noise_rates = [args.noise_rate] * 4

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(int(args.seed))
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Systemic Noise Coordination
    assigned_clusters_dict, global_cluster_map = coordinate_sys_noise_clusters(
        args.client_num, 
        args.vul, 
        args.noise_type, 
        model_type=args.model_type,
        n_clusters=args.n_clusters, 
        seed=int(args.seed),
        data_dir=args.data_dir
    )

    # Data Loading
    print("Generating Datasets...")
    train_ds = list()
    for i in range(args.client_num):
        ds = gen_crd_ds(
            i, 
            args.vul, 
            args.noise_type, 
            noise_rates[i], # Use client-specific noise rate
            args.random_noise, 
            args.num_neigh, 
            args.model_type, 
            assigned_clusters=assigned_clusters_dict, 
            global_cluster_map=global_cluster_map,
            n_clusters=args.n_clusters,
            seed=int(args.seed),
            data_dir=args.data_dir
        )
        train_ds.append(ds)

    # Model & Server Init
    if args.vul == 'reentrancy':
        class_weights = torch.tensor([1.0, 1.0]).to(args.device)
    elif args.vul == 'timestamp':
        class_weights = torch.tensor([1.2, 1.5]).to(args.device)
    else:
        class_weights = torch.tensor([1.0, 1.0]).to(args.device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    if args.model_type == "CBGRU":
        global_model = ClassiFilerNet(input_size, time_steps)
    elif args.model_type == "CGE":
        global_model = CGEVariant()
    global_model = global_model.to(args.device)
    
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    server = CRD_server(
        args,
        global_model,
        args.device,
        criterion
    )

    test_dl = gen_test_dl(args.model_type, args.vul, data_dir=args.data_dir)
    valid_dl = gen_valid_dl(args.model_type, args.vul, data_dir=args.data_dir)

    # Client Init
    print("Initializing Clients...")
    clients = []
    for i in range(args.client_num):
        client = Fed_CRD_client(
            args,
            criterion, 
            copy.deepcopy(server.global_model),
            train_ds[i],
            i,
            run_timestamp=run_timestamp
        )
        # Initialize Static KNN Neighborhood
        client.init_knn_neighborhood()
        clients.append(client)
        
    print("Initialization Complete. Starting Training...")

    # Training Loop
    for epoch in range(args.epoch):
        print(f"\n--- Epoch {epoch} ---")
        server.initialize_epoch_updates(epoch) 

        updates_list = [] # Store (client_id, delta, q_k, n_k)
        
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for client_id in range(args.client_num):
                futures.append(executor.submit(train_crd_client, client_id, clients[client_id], server.global_model, server.ema_model, criterion))
            
            for future in futures:
                client_id, delta, q_k, num_samples, result, loss = future.result()
                updates_list.append((client_id, delta, q_k, num_samples))
                
                print(f"Client {client_id}: Loss={loss:.4f}, Reliability(q_k)={q_k:.4f}")
                
                # Log to server for record keeping (optional)
                server.save_train_updates(delta, num_samples, result)
                
        # Server Aggregation (FedCRD Logic)
        server.aggregate(updates_list)
        
        # Validation
        if epoch % 5 == 0 or epoch == args.epoch - 1:
             print(f"Validation at Epoch {epoch}...")
             global_test(
                server.global_model, 
                valid_dl, 
                criterion, 
                args, 
                f"Fed_CRD_{args.vul}", 
                run_timestamp=run_timestamp,
                save_result=True,
                tag='valid',
                epoch=epoch
            )
            
    # Final Test
    print("\n--- Final Testing ---")
    global_test(server.global_model, test_dl, criterion, args, f"Fed_CRD_{args.vul}_Final", run_timestamp=run_timestamp)
