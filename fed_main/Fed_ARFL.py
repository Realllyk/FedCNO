import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from options import parse_args
from data_processing.dataloader_manager import gen_client_ds, gen_valid_dl
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

    clients = list()
    train_ds = list()
    for i in range(args.client_num):
        ds = gen_client_ds(args.model_type, i, args.vul, args.noise_type, noise_rates[i], args.random_noise, args.num_neigh)
        train_ds.append(ds)

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

    for epoch in range(args.epoch):
        print(f"Epoch {epoch} Training:------------------")
        server.initialize_epoch_updates(epoch)
        server.sample_clients(epoch)

        for c in clients:
            if c.model != None:
                del c.model
            c.model = copy.deepcopy(server.global_model)
        
        for i, c in enumerate(server.selected_clients):
            c.train()
            print(f"Selected Client {i} Train Loss: {c.result['loss']}")

        server.average_weights()
        server.update_alpha()
    
    # test_dl = gen_cbgru_valid_dl(args.vul, 0, args.batch)
    test_dl = gen_valid_dl(args.model_type, args.vul)
    global_test(server.global_model, test_dl, criterion, args, args.lab_name)



        

        

        