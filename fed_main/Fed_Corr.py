import sys
import os
import copy
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from sklearn.mixture import GaussianMixture
import torch.nn as nn
from scipy.spatial.distance import cdist

from options import parse_args
from trainers.client import Fed_Corr_client
from trainers.server import Server
from models.ClassiFilerNet import ClassiFilerNet
from models.CGE_Variants import CGEVariant
from data_processing.dataloader_manager import gen_whole_dataset, gen_valid_dl

from global_test import global_test


def lid_term(X, batch, k=20):
    eps = 1e-6
    X = np.asarray(X, dtype=np.float32)

    batch = np.asarray(batch, dtype=np.float32)
    f = lambda v: - k / (np.sum(np.log(v / (v[-1]+eps)))+eps)
    distances = cdist(X, batch)

    # get the closest k neighbours
    sort_indices = np.apply_along_axis(np.argsort, axis=1, arr=distances)[:, 1:k + 1]
    m, n = sort_indices.shape
    idx = np.ogrid[:m, :n]
    idx[1] = sort_indices
    # sorted matrix
    distances_ = distances[tuple(idx)]
    lids = np.apply_along_axis(f, axis=1, arr=distances_)
    return lids


def get_output(dataloader, model, args, criterion):
        model.eval()
        with torch.no_grad():
            for i, (x1, x2, y) in enumerate(dataloader):
                x1, x2, y = x1.to(args.device), x2.to(args.device), y.to(args.device)
                y = y.long()

                outputs = model(x1, x2)
                outputs = F.softmax(outputs, dim=1)

                loss = criterion(outputs, y)
                if i == 0:
                    output_whole = np.array(outputs.cpu())
                    loss_whole = np.array(loss.cpu())
                else:
                    output_whole = np.concatenate((output_whole, outputs.cpu()), axis=0)
                    loss_whole = np.concatenate((loss_whole, loss.cpu()), axis=0)

        return output_whole, loss_whole


if __name__ == "__main__":
    args = parse_args()
    INPUT_SIZE, TIME_STAMP = 100, 300
    # set random seed
    torch.manual_seed(args.corr_seed)
    torch.cuda.manual_seed(args.corr_seed)
    torch.cuda.manual_seed_all(args.corr_seed)
    np.random.seed(args.corr_seed)
    random.seed(args.corr_seed)

    # get dataset
    # whole_ds, input_size, time_step, data_indices = gen_whole_dataset(args.client_num, args.vul, args.noise_type, args.noise_rate)
    if args.diff == True:
        noise_rates = random.sample([0.2, 0.2, 0.3, 0.3], 4)
    else:
        noise_rates = [args.noise_rate] * 4

    whole_ds,  data_indices = gen_whole_dataset(args.model_type, args.client_num, args.vul, args.noise_type, noise_rates, args.num_neigh)
    criterion = nn.CrossEntropyLoss(reduction='none')
    LID_accumulative_client = np.zeros(args.client_num)

    # set Server
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

    for iteration in range(args.iteration1):
        LID_whole = np.zeros(len(whole_ds))
        loss_whole = np.zeros(len(whole_ds))
        LID_client = np.zeros(args.client_num)
        loss_accumulative_whole = np.zeros(len(whole_ds))

        if iteration == 0:
            mu_list = np.zeros(args.client_num)
        else:
            mu_list = estimated_noisy_level
        
        prob = [1 / args.client_num] * args.client_num

        for epoch in range(int(1/args.sample_rate)):
            server.initialize_epoch_updates(epoch)
            idxs_users = np.random.choice(range(args.client_num), int(args.client_num * args.sample_rate), p=prob)

            for idx in idxs_users:
                prob[idx] = 0
                if sum(prob) > 0:
                    prob = [prob[i] / sum(prob) for i in range(len(prob))]
                
                sample_idx = np.array(data_indices[idx])
                dataset_client = Subset(whole_ds, sample_idx)
                # dl_client = DataLoader(dataset_client, batch_size=args.batch, shuffle=False)

                mu_i = mu_list[idx]
                client = Fed_Corr_client(
                    args,
                    criterion,
                    copy.deepcopy(server.global_model),
                    dataset_client
                )
                client.train()
                server.save_train_updates(
                    copy.deepcopy(client.get_parameters()),
                    client.result['sample'],
                    client.result
                )

                local_output, loss = get_output(client.dataloader, client.model, args, criterion)
                LID_local = list(lid_term(local_output, local_output))
                LID_whole[sample_idx] = LID_local
                loss_whole[sample_idx] = loss
                LID_client[idx] = np.mean(LID_local)
            
            server.average_weights()

        LID_accumulative_client = LID_accumulative_client + np.array(LID_client)
        loss_accumulative_whole = loss_accumulative_whole + np.array(loss_whole)

        # Apply Gaussian Mixture Model to LID
        gmm_LID_accumulative = GaussianMixture(n_components=2, random_state=args.corr_seed).fit(
            np.array(LID_accumulative_client).reshape(-1, 1))
        labels_LID_accumulative = gmm_LID_accumulative.predict(np.array(LID_accumulative_client).reshape(-1, 1))
        clean_label = np.argsort(gmm_LID_accumulative.means_[:, 0])[0]

        noisy_set = np.where(labels_LID_accumulative != clean_label)[0]
        clean_set = np.where(labels_LID_accumulative == clean_label)[0]

        estimated_noisy_level = np.zeros(args.client_num)

        for client_id in noisy_set:
            sample_idx = np.array(list(data_indices[client_id]))

            loss = np.array(loss_accumulative_whole[sample_idx])
            gmm_loss = GaussianMixture(n_components=2, random_state=args.corr_seed).fit(np.array(loss).reshape(-1, 1))
            labels_loss = gmm_loss.predict(np.array(loss).reshape(-1, 1))
            gmm_clean_label_loss = np.argsort(gmm_loss.means_[:, 0])[0]

            pred_n = np.where(labels_loss.flatten() != gmm_clean_label_loss)[0]
            y_train_noisy_new = np.array(whole_ds.labels)

        if args.correction:
            for idx in noisy_set:
                sample_idx = np.array(list(data_indices[idx]))
                dataset_client = Subset(whole_ds, sample_idx)
                # dl_client = DataLoader(dataset_client, batch_size=args.batch, shuffle=False)
                client = Fed_Corr_client(
                    args,
                    criterion,
                    copy.deepcopy(server.global_model),
                    dataset_client
                )
                loss = np.array(loss_accumulative_whole[sample_idx])
                local_output, _ = get_output(client.dataloader, client.model, args, criterion)
                relabel_idx = (-loss).argsort()[:int(len(sample_idx) * estimated_noisy_level[idx] * args.relabel_ratio)]
                relabel_idx = list(set(np.where(np.max(local_output, axis=1) > args.confidence_thres)[0]) & set(relabel_idx))
                
                y_train_noisy_new = np.array(whole_ds.labels)
                y_train_noisy_new[sample_idx[relabel_idx]] = np.argmax(local_output, axis=1)[relabel_idx]
                whole_ds.labels = y_train_noisy_new

    # reset the beta
    args.beta = 0
    
    #---------------------------------- second stage training ----------------------------------
    if args.fine_tuning:
        selected_clean_idx = np.where(estimated_noisy_level <= args.clean_set_thres)[0]

        prob = np.zeros(args.client_num)
        prob[selected_clean_idx] = 1/len(selected_clean_idx)
        m = max(int(args.frac2 * args.client_num), 1)
        m = min(m, len(selected_clean_idx))

        for rnd in range(args.rounds1):
            w_locals, loss_locals = [], []
            server.initialize_epoch_updates(rnd)
            idxs_users = np.random.choice(range(args.client_num), m, replace=False, p=prob)
            for idx in idxs_users:
                sample_idx = np.array(data_indices[idx])
                dataset_client = Subset(whole_ds, sample_idx)
                # dl_client = DataLoader(dataset_client, batch_size=args.batch, shuffle=False)
                client = Fed_Corr_client(
                    args,
                    criterion,
                    copy.deepcopy(server.global_model),
                    dataset_client
                )
                
                client.train()
                server.save_train_updates(
                    copy.deepcopy(client.get_parameters()),
                    client.result['sample'],
                    client.result
                )
            
            server.average_weights()
        
        if args.correction:
            relabel_idx_whole = []
            for idx in noisy_set:
                sample_idx = np.array(data_indices[idx])
                dataset_client = Subset(whole_ds, sample_idx)
                dl_client = DataLoader(dataset_client, batch_size=args.batch, shuffle=False)
                glob_output, _ = get_output(dl_client, server.global_model, args, criterion)
                y_predicted = np.argmax(glob_output, axis=1)
                relabel_idx = np.where(np.max(glob_output, axis=1) > args.confidence_thres)[0]
                y_train_noisy_new = np.array(whole_ds.labels)
                y_train_noisy_new[sample_idx[relabel_idx]] = y_predicted[relabel_idx]
                whole_ds.labels = y_train_noisy_new

# ---------------------------- third stage training -------------------------------
    m = max(int(args.sample_rate * args.client_num), 1)
    prob = [1/args.client_num for i in range(args.client_num)]
    print("----------------------STAGE 3--------------------------------")
    # test_dl = gen_cbgru_valid_dl(args.vul, 0, args.batch)
    test_dl = gen_valid_dl(args.model_type, args.vul)

    for rnd in range(args.rounds2):
        idxs_users = np.random.choice(range(args.client_num), m, replace=False, p = prob)
        server.initialize_epoch_updates(rnd)
        print(f"epoch {rnd}:")
        for idx in idxs_users:
            sample_idx = np.array(data_indices[idx])
            dataset_client = Subset(whole_ds, sample_idx)
            # dl_client = DataLoader(dataset_client, batch_size=args.batch, shuffle=False)
            client = Fed_Corr_client(
                args,
                criterion,
                copy.deepcopy(server.global_model),
                dataset_client
            )
            client.train()
            server.save_train_updates(
                copy.deepcopy(client.get_parameters()),
                client.result['sample'],
                client.result
            )
            del client
        server.average_weights()
    
    global_test(server.global_model, test_dl, criterion, args, args.lab_name, 'none')


        



                
        
    
                


            





    


