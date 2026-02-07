import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import gc
import torch
import numpy as np
import time
import torch.nn as nn
import random
import concurrent.futures
import argparse
import torch.nn.functional as F

# Import original modules
from options import parse_args
from data_processing.dataloader_manager import gen_client_ds, gen_valid_dl
from data_processing.preprocessing import coordinate_sys_noise_clusters
from models.ClassiFilerNet import ClassiFilerNet
from models.CGE_Variants import CGEVariant
from trainers.server_elc import ELC_Server
from trainers.client_elc import Fed_ELC_client
from global_test import global_test

def add_elc_args(args):
    # Add FedELC specific arguments
    # We can't easily modify the namespace returned by parse_args() if it's not extensible,
    # but Python objects are dynamic.
    if not hasattr(args, 'epoch_of_stage1'):
        args.epoch_of_stage1 = 20
    if not hasattr(args, 'lambda_pencil'):
        args.lambda_pencil = 1000
    if not hasattr(args, 'alpha_pencil'):
        args.alpha_pencil = 0.5
    if not hasattr(args, 'beta_pencil'):
        args.beta_pencil = 0.2
    if not hasattr(args, 'K_pencil'):
        args.K_pencil = 10
    return args

def get_client_stats(client_id, args, global_model, dataset, cls_num_list=None):
    """
    Compute class-wise loss using the global model for GMM splitting
    """
    client = Fed_ELC_client(args,
                        None, # criterion not needed for eval if we define it inside or use default
                        global_model, # Use global model
                        dataset,
                        client_id=client_id,
                        cls_num_list=cls_num_list)
    
    # We need to compute class-wise loss
    # Fed_ELC_client has get_class_wise_loss method
    class_wise_loss = client.get_class_wise_loss()
    
    del client
    torch.cuda.empty_cache()
    # gc.collect() # Optional, might slow down if called too often
    
    return client_id, class_wise_loss

def train_one_client_elc(client_id, args, global_model, criterion, dataset, is_noisy, stage, soft_labels=None, cls_num_list=None):
    """
    Client training function for FedELC
    """
    client = Fed_ELC_client(args,
                        criterion,
                        None,
                        dataset,
                        client_id=client_id,
                        cls_num_list=cls_num_list)
    
    client.model = copy.deepcopy(global_model)
    
    updated_soft_labels = None
    if stage == 1:
        client.train_stage1()
    elif stage == 2:
        if is_noisy:
            # ELC Training
            updated_soft_labels = client.train_stage2(soft_labels, 
                                                      K_pencil=args.K_pencil,
                                                      lambda_pencil=args.lambda_pencil,
                                                      alpha=args.alpha_pencil,
                                                      beta=args.beta_pencil)
        else:
            # Standard Training for clean clients
            client.train_stage1()
            
    # Always compute class-wise loss for GMM (only needed at end of stage 1, but harmless to compute)
    # To save compute, only do it if requested. 
    # Let's add a flag 'compute_gmm_stats' to arguments or infer from epoch
    
    weights = copy.deepcopy(client.get_parameters())
    num_samples = client.result['sample']
    result = client.result
    loss = client.result['loss']
    
    del client
    torch.cuda.empty_cache()
    gc.collect()
    
    return client_id, weights, num_samples, result, loss, updated_soft_labels

if __name__ == '__main__':
    args = parse_args()
    args = add_elc_args(args)
    
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

    train_ds = list()
    # Store class counts for LogitAdjust
    client_cls_counts = {}
    inferred_num_classes = None
    
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
        
        # Calculate class counts for this client
        # Assuming ds.labels exists and is list/array
        if hasattr(ds, 'labels'):
            labels = np.array(ds.labels)
            num_classes = int(labels.max()) + 1 if labels.size > 0 else 2
            if inferred_num_classes is None:
                inferred_num_classes = num_classes
            else:
                inferred_num_classes = max(inferred_num_classes, num_classes)
            counts = np.bincount(labels, minlength=num_classes)
            client_cls_counts[i] = counts
        else:
             client_cls_counts[i] = None

    if inferred_num_classes is not None and not hasattr(args, 'num_classes'):
        args.num_classes = inferred_num_classes

    criterion = nn.CrossEntropyLoss()

    if args.model_type == "CBGRU":
        global_model = ClassiFilerNet(INPUT_SIZE, TIME_STAMP)
    elif args.model_type == "CGE":
        global_model = CGEVariant()
    global_model = global_model.to(args.device)

    server = ELC_Server(
        args,
        global_model,
        args.device,
        criterion
    )

    test_dl = gen_valid_dl(args.model_type, args.vul, data_dir=args.data_dir)
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # State management
    client_soft_labels = {} # {client_id: tensor}
    
    # Initialize soft labels for all clients (will be used in Stage 2)
    # We need to do this based on dataset size.
    # We can init them when switching to Stage 2.
    
    for epoch in range(args.epoch):
        server.initialize_epoch_updates(epoch)
        
        # Stage selection
        stage = 1 if epoch < args.epoch_of_stage1 else 2

        # Perform GMM split once at the transition epoch
        if epoch == args.epoch_of_stage1 and not server.clean_clients and not server.noisy_clients:
            print(">>> Transitioning to Stage 2: Performing GMM Split...")
            epoch_client_losses = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures_stats = []
                for client_id in range(args.client_num):
                    futures_stats.append(executor.submit(
                        get_client_stats,
                        client_id,
                        args,
                        server.global_model,
                        train_ds[client_id],
                        client_cls_counts.get(client_id)
                    ))

                for future in futures_stats:
                    cid, cw_loss = future.result()
                    epoch_client_losses[cid] = cw_loss

            clean_c, noisy_c = server.gmm_split(epoch_client_losses)
            print(f"Clean clients: {clean_c}")
            print(f"Noisy clients: {noisy_c}")

            # Initialize soft labels for noisy clients
            for client_id in noisy_c:
                ds = train_ds[client_id]
                labels = torch.tensor(ds.labels).long()
                num_classes = getattr(args, 'num_classes', int(labels.max().item()) + 1 if labels.numel() > 0 else 2)
                y_onehot = F.one_hot(labels, num_classes).float()
                client_soft_labels[client_id] = y_onehot * args.K_pencil

        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            for client_id in range(args.client_num):
                is_noisy = (client_id in server.noisy_clients) if stage == 2 else False
                
                # Prepare soft labels if needed
                soft_label_in = None
                if stage == 2 and is_noisy:
                    if client_id not in client_soft_labels:
                        # Initialize soft labels if first time
                        # One-hot encoded original labels * K
                        # We need access to dataset labels.
                        ds = train_ds[client_id]
                        labels = torch.tensor(ds.labels).long()
                        # num_classes = labels.max().item() + 1
                        num_classes = 10 # Default
                        
                        y_onehot = F.one_hot(labels, num_classes).float()
                        client_soft_labels[client_id] = y_onehot * args.K_pencil
                        
                    soft_label_in = client_soft_labels[client_id]
                
                futures.append(executor.submit(train_one_client_elc, 
                                               client_id, 
                                               args, 
                                               server.global_model, 
                                               criterion, 
                                               train_ds[client_id], 
                                               is_noisy, 
                                               stage, 
                                               soft_label_in,
                                               client_cls_counts.get(client_id)))
            
            for future in futures:
                cid, weights, num_samples, result, loss, updated_soft = future.result()
                
                server.save_train_updates(weights, num_samples, result, cid)
                
                if updated_soft is not None:
                    client_soft_labels[cid] = updated_soft
                    
                print(f"client:{cid} loss:{loss}")

        # Standard aggregation to match baseline
        server.average_weights()

        print(f"Epoch {epoch} finished. Stage {stage}")

    global_test(server.global_model, test_dl, criterion, args, args.lab_name, run_timestamp=run_timestamp)
