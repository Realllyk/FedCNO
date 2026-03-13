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
from data_processing.dataloader_manager import gen_lgv_ds, gen_valid_dl
from data_processing.preprocessing import coordinate_sys_noise_clusters
from models.model_factory import build_model
from trainers.server import LGV_server
from trainers.client import Fed_LGV_client
from global_test import global_test
import time
import random


if __name__ == '__main__':
    args = parse_args()
    if args.model_type == "MANDO" and args.vul != "tod":
        raise ValueError("MANDO only supports --vul tod in this project.")
    input_size, time_steps = 100, 300
    if args.diff == True:
        noise_rates = random.sample([0.2, 0.2, 0.3, 0.3], 4)
    else:
        noise_rates = [args.noise_rate] * 4

    assigned_clusters_dict, global_cluster_map = coordinate_sys_noise_clusters(
        args.client_num,
        args.vul,
        args.noise_type,
        model_type=args.model_type,
        n_clusters=args.n_clusters,
        seed=int(args.seed),
        data_dir=args.data_dir
    )

    # initialize dataset
    train_ds = list()
    for i in range(args.client_num):
        ds = gen_lgv_ds(
            i,
            args.vul,
            args.noise_type,
            noise_rates[i],
            args.num_neigh,
            args.model_type,
            assigned_clusters=assigned_clusters_dict,
            global_cluster_map=global_cluster_map,
            n_clusters=args.n_clusters,
            seed=int(args.seed),
            data_dir=args.data_dir
        )
        train_ds.append(ds)
    test_dl = gen_valid_dl(args.model_type, args.vul, data_dir=args.data_dir)
    
    # initialize Server
    criterion = nn.CrossEntropyLoss()
    global_model = build_model(args, input_size, time_steps)
    global_model = global_model.to(args.device)
    server = LGV_server(
        args,
        global_model,
        args.device,
        criterion,
        args.global_weight
    )
    
    # initialize Client
    clients = []
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")

    for i in range(args.client_num):
        client = Fed_LGV_client(
            args,
            nn.CrossEntropyLoss(reduction='none'),
            copy.deepcopy(server.global_model),
            train_ds[i],
            i,
            server.global_weight,
            run_timestamp=run_timestamp
        )
        client.get_local_knn_labels(args.vul, args.noise_type, args.noise_rate)
        # client.get_global_knn_labels(args.vul, args.noise_type, args.noise_rate)
        clients.append(client)
        print(f"Generate Client {i}!")

    #  Warm up
    for epoch in range(args.warm_up_epoch):
        print(f"Warm Up Epoch {epoch}: ")
        server.initialize_epoch_updates(epoch)

        for client_id in range(args.client_num):
            client = clients[client_id]
            del client.model
            client.model = copy.deepcopy(server.global_model)
            client.global_weight = server.global_weight
            client.warmup_train()
            server.save_train_updates(
                copy.deepcopy(client.get_parameters()),
                client.result['sample'],
                client.result
            )
            print(f"client:{client_id}")
            client.print_loss()
            torch.cuda.empty_cache()
            gc.collect()
        
        server.average_weights()

    # Train Stage
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
            client.model = copy.deepcopy(server.global_model)
            client.global_weight = server.global_weight
            client.get_global_knn_labels(args.vul, args.noise_type, args.noise_rate)
            # client.get_global_prob_labels(args.vul)
            client.train()
            server.save_train_updates(
                copy.deepcopy(client.get_parameters()),
                client.result['sample'],
                client.result
            )
            print(f"client:{client_id}")
            client.print_loss()
            torch.cuda.empty_cache()
            gc.collect()
        
        server.average_weights()
        # if epoch % 5 == 0:
        #     server.autotune_gr(test_dl)
    
    global_test(server.global_model, test_dl, criterion, args, 'Fed_LGV', run_timestamp=run_timestamp)
        
    


