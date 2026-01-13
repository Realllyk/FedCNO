import copy
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from options import parse_args
from data_processing.dataloader_manager import gen_cbgru_dl, gen_cbgru_valid_dl, gen_client_ds, gen_valid_dl
from models.ClassiFilerNet import ClassiFilerNet
from models.CGE_Variants import CGEVariant
from trainers.server import Server
from trainers.client import Fed_Avg_client
from global_test import global_test
import random


if __name__ == '__main__':
    args = parse_args()
    INPUT_SIZE, TIME_STAMP = 100, 300

    # dataloader_dict = dict()
    # dataloader_dict['train'] = list()
    if args.diff == True:
        noise_rates = random.sample([0.2, 0.2, 0.3, 0.3], 4)
    else:
        noise_rates = [args.noise_rate] * 4


    train_ds = list()
    for i in range(args.client_num):
        # train_dl, INPUT_SIZE, TIME_STAMP = gen_cbgru_dl(i, args.vul, args.noise_type, args.noise_rate, args.batch)
        # dataloader_dict['train'].append(train_dl)
        # ds = gen_cbgru_ds(i, args.vul, args.noise_type, args.noise_rate, args.random_noise, args.num_neigh)
        ds = gen_client_ds(args.model_type, i, args.vul, args.noise_type, noise_rates[i], args.random_noise, args.num_neigh)
        train_ds.append(ds)

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
    test_dl = gen_valid_dl(args.model_type, args.vul)
    for epoch in range(args.epoch):
        server.initialize_epoch_updates(epoch)

        for client_id in range(args.client_num):
            client = Fed_Avg_client(args,
                                criterion,
                                None,
                                train_ds[i])
            # print(f"Create Client {client_id}!")
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
        # CBGRU_test(server.global_model, test_dl, criterion, args, 'Fed_CBGRU')
        print(epoch)
    
    
    global_test(server.global_model, test_dl, criterion, args, args.lab_name)
        
    
