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
    criterion = nn.CrossEntropyLoss()
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
    for epoch in range(args.warm_up_epoch):
        print(f"Warm Up Epoch {epoch}: ")
        server.initialize_epoch_updates(epoch)

        for client_id in range(args.client_num):
            client = Fed_Avg_client(args,
                                criterion,
                                None,
                                train_ds[i])
            client.model = copy.deepcopy(server.global_model)
            client.train()
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
        
        server.average_weights()
    
    # initialize dataset
    train_ds = list()
    for i in range(args.client_num):
        ds = gen_lgv_ds(i, args.vul, args.noise_type, args.noise_rate, args.random_noise, args.num_neigh, args.model_type)
        train_ds.append(ds)
    print(args.random_noise)
    test_dl = gen_valid_dl(args.model_type, args.vul)

    # initialize Client
    clients = []
    for i in range(args.client_num):
        client = Fed_LGV_client(
            args,
            nn.CrossEntropyLoss(reduction=reduction),
            copy.deepcopy(server.global_model),
            train_ds[i],
            i,
            server.global_weight
        )
        client.get_local_knn_labels(args.vul, args.noise_type, args.noise_rate)
        # client.get_global_knn_labels(args.vul, args.noise_type, args.noise_rate)
        client.gen_reduced_ds()
        clients.append(client)
        print(f"Generate Client {i}!")


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
            # client.get_global_knn_labels(args.vul, args.noise_type, args.noise_rate)
            # client.get_global_prob_labels(args.vul)
            # client.get_global_feature_knn_labels()
            client.get_global_feature_glbal_knn_labels()
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
    
    global_test(server.global_model, test_dl, criterion, args, f"{args.num_neigh}neigh_{args.global_weight}_{args.lab_name}")
        
    
