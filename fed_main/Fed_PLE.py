import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gc
import copy
import random
import pandas as pd
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from models.ClassiFilerNet import ClassiFilerNet
from models.LCN import LCN
from trainers.client import Fed_PLE_client
from trainers.server import Server
from data_processing.dataloader_manager import gen_cbgru_dl, gen_cbgru_client_pure_dl, gen_cbgru_valid_dl, gen_cbgru_client_noise_dl, gen_cbgru_client_valid_dl
from options import parse_args
from global_test import global_test


def flip_values(names_path, labels_path, noise_rate):
    names = []
    with open(names_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            names.append(line.strip())

    with open(labels_path, 'rb') as file:
        df = pd.read_csv(labels_path, header=None)
        labels = df.iloc[:, 0].values

    name_set = set()
    unique_names = []
    unique_labels = []
    for i, name in enumerate(names):
        if name not in name_set:
            unique_labels.append(labels[i])
            unique_names.append(name)
            name_set.add(name)

    for i in range(len(unique_labels)):
        if random.random() < noise_rate:
            unique_labels[i] = 1-unique_labels[i]
    
    name_label = dict()
    for i in range(len(unique_names)):
        name_label[unique_names[i]] = unique_labels[i]
    
    noise_labels = []
    for name in names:
        noise_labels.append(name_label[name])
    
    return noise_labels


# 需要调整ClassiFilerNet和input_channel，需要读取ClassiFilerNet的中间输出作为outer_model的输入
if __name__ == "__main__":
    args = parse_args()
    criterion = nn.CrossEntropyLoss()
    if args.device != "cpu":
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # 根据client_id生成不同的dataloader
    input_size, time_stamp = 0, 0
    dataloader_dict = dict()
    dataloader_dict["noise"] = list()
    dataloader_dict["valid"] = list()
    dataloader_dict["pure"] = list()
    noise_labels = list()
    for i in range(args.client_num):
        noise_dl, input_size, time_stamp = gen_cbgru_dl(client_id=i, vul=args.vul, noise_type=args.noise_type, noise_rate=args.noise_rate, batch=args.batch, shuffle=False, data_dir=args.data_dir)
        dataloader_dict["noise"].append(noise_dl)
        pure_dl = gen_cbgru_client_pure_dl(client_id=i, vul = args.vul, noise_type=args.noise_type, noise_rate=args.noise_rate, batch=args.batch, data_dir=args.data_dir)
        dataloader_dict["pure"].append(pure_dl)

        word2vec_dir = os.path.join(args.data_dir, f"cbgru_data/{args.vul}/word2vec")
        fastText_dir = os.path.join(args.data_dir, f"cbgru_data/{args.vul}/FastText")
        client_dir = os.path.join(args.data_dir, f"graduate_client_split/{args.vul}/client_{i}")
        names_path = os.path.join(client_dir, "contract_name_train.txt")
        labels_path = os.path.join(client_dir, "label_train.csv")
        noise_labels.append(flip_values(names_path, labels_path, args.noise_rate))
        
    
    # 初始化Server
    global_model = ClassiFilerNet(input_size, time_stamp)
    global_model = global_model.to(device)
    server = Server(
        args,
        global_model,
        device,
        criterion
    )

    for i in range(args.client_num):
        valid_dl = gen_cbgru_client_valid_dl(i, args.vul, args.batch, noise_labels[i], args.valid_frac, data_dir=args.data_dir)
        dataloader_dict["valid"].append(valid_dl)

    # 初始化Client
    client_list = list()
    for i in range(args.client_num):
        inner_model = copy.deepcopy(server.global_model)
        outer_model = LCN(in_channels=args.input_channels)
        inner_model, outer_model = inner_model.to(device), outer_model.to(device)
        client = Fed_PLE_client(
            args,
            criterion,
            device,
            inner_model,
            outer_model,
            None,
            dataloader_dict["pure"][i],
            dataloader_dict["valid"][i]
        )
        client_list.append(client)

    # 训练部分
    for epoch in range(args.epoch):
        server.initialize_epoch_updates(epoch)
        
        for client_id in range(args.client_num):
            global_labels = []
            client = client_list[client_id]
            client.inner_model = copy.deepcopy(server.global_model)

            # 生成全局预测标签
            with torch.no_grad():
                server.global_model.eval()
                for x1, x2, _ in dataloader_dict["noise"][client_id]:
                    x1, x2 = x1.to(device), x2.to(device)
                    outputs = server.global_model(x1, x2)
                    outputs = F.softmax(outputs, dim=-1)
                    labels = torch.argmax(outputs, dim=-1)
                    global_labels.append(labels)

                    del x1, x2, outputs, labels
                    torch.cuda.empty_cache()
                    gc.collect()

            conc_labels = torch.cat(global_labels, dim = 0)
            noise_dl = gen_cbgru_client_noise_dl(client_id, args.vul, args.noise_type, conc_labels, args.noise_rate, args.batch, noise_labels[client_id], data_dir=args.data_dir)
            client.noise_dataloader = noise_dl

            # 本地训练并保存训练结果
            client.meta_train()
            server.save_train_updates(
                copy.deepcopy(client.get_inner_parameters()),
                client.result["sample"],
                client.result
            )
            print(f"client:{client_id} train")
            client.print_loss()

            # if epoch % 4 == 0:
            #     for client_id in range(args.client_num):
            #         client = client_list[client_id]
            #         client.validation()
            #         server.save_val_updates(client.result)
            #     server.save_best_model()
        
        server.average_weights()
        print(epoch)
    
    # server.global_model.load_state_dict(server.best_model)
    test_dl = gen_cbgru_valid_dl(args.vul, id=0, batch=args.batch, data_dir=args.data_dir)
    global_test(server.global_model, test_dl, criterion, args, 'test_Fed_CBGRU_PLE')
            



