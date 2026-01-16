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
from trainers.CLC import CLC
from global_test import global_test
import random


if __name__ == "__main__":
    args = parse_args()
    if args.diff == True:
        noise_rates = random.sample([0.2, 0.2, 0.3, 0.3], 4)
    else:
        noise_rates = [args.noise_rate] * 4

    datasets_dict = dict()
    datasets_dict['train'] = list()
    for i in range(args.client_num):
        # train_ds = gen_cbgru_ds(i, args.vul, args.noise_type, args.noise_rate, args.random_noise, args.num_neigh)
        train_ds = gen_client_ds(args.model_type, i, args.vul, args.noise_type, noise_rates[i], args.random_noise, args.num_neigh)
        datasets_dict['train'].append(train_ds)
    # test_dl = gen_cbgru_valid_dl(args.vul)
    test_dl = gen_valid_dl(args.model_type, args.vul)
    
    criterion = nn.CrossEntropyLoss()
    clc = CLC(args, 100, 300, datasets_dict['train'], 0.1)
    clc.holdout_stage()
    # clc.correct_stage()

    global_test(clc.server.global_model, test_dl, criterion, args, args.lab_name)
