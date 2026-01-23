import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import numpy as np
import random
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from sklearn.mixture import GaussianMixture
import torch.nn as nn
from scipy.spatial.distance import cdist
import concurrent.futures

from options import parse_args
from trainers.client import Fed_Corr_client
from trainers.server import Server
from models.ClassiFilerNet import ClassiFilerNet
from models.CGE_Variants import CGEVariant
from data_processing.dataloader_manager import gen_whole_dataset, gen_valid_dl

from global_test import global_test


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def lid_term(X, batch, k=20):
    eps = 1e-6
    X = np.asarray(X, dtype=np.float32)

    batch = np.asarray(batch, dtype=np.float32)
    f = lambda v: - k / (np.sum(np.log(v / (v[-1]+eps)))+eps)
    
    # 将 cdist 计算转移到 GPU (如果数据量大)
    # 但考虑到 LID 计算主要是 pairwise distance，sklearn/scipy 在 CPU 上优化较好
    # 这里保持 CPU 计算，但可以通过多进程并行处理多个客户端的 LID
    distances = cdist(X, batch)

    # get the closest k neighbours
    sort_indices = np.apply_along_axis(np.argsort, axis=1, arr=distances)[:, 1:k + 1]
    m, n = sort_indices.shape
    idx = list(np.ogrid[:m, :n])
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

# -----------------------------------------------------------------------------
# 并行任务函数封装
# -----------------------------------------------------------------------------

def train_fed_corr_client(idx, args, global_model, criterion, dataset_client, run_timestamp, global_round_counter):
    """
    第一阶段训练任务：
    1. 本地训练
    2. 计算本地输出和Loss
    3. 计算LID (LID计算是CPU密集型，适合在进程/线程中异步执行，但要注意GIL)
    """
    # 设置随机种子以尽可能保证复现性
    setup_seed(args.corr_seed + idx + global_round_counter)
    
    client = Fed_Corr_client(
        args,
        criterion,
        copy.deepcopy(global_model),
        dataset_client,
        client_id=idx,
        run_timestamp=run_timestamp,
        global_round=global_round_counter
    )
    client.train()
    
    # 提取需要的训练结果，释放client内存
    weights = copy.deepcopy(client.get_parameters())
    num_samples = client.result['sample']
    train_result = client.result
    
    # 注意：原代码使用 client.dataloader (shuffle=True) 进行 get_output，
    # 这会导致输出的顺序与 sample_idx 不对应，从而导致 LID_whole 和 loss_whole 赋值错位。
    # 这里修正为使用 shuffle=False 的 dataloader 进行评估。
    eval_dl = DataLoader(dataset_client, batch_size=args.batch, shuffle=False)
    local_output, loss = get_output(eval_dl, client.model, args, criterion)
    
    # 释放显存
    del client
    # torch.cuda.empty_cache() # 单GPU多线程场景下避免频繁清理显存导致同步阻塞
    
    # LID 计算 (CPU密集型)
    # 在这里计算是为了利用多线程/进程的并行性，虽然Python线程有GIL，但numpy部分操作释放GIL
    LID_local = list(lid_term(local_output, local_output))
    
    return idx, weights, num_samples, train_result, local_output, loss, LID_local


def train_simple_client(idx, args, global_model, criterion, dataset_client, run_timestamp, global_round_counter):
    """
    普通训练任务（第二、三阶段使用）：仅训练和返回权重
    """
    # 设置随机种子以尽可能保证复现性
    setup_seed(args.corr_seed + idx + global_round_counter)

    client = Fed_Corr_client(
        args,
        criterion,
        copy.deepcopy(global_model),
        dataset_client,
        client_id=idx,
        run_timestamp=run_timestamp,
        global_round=global_round_counter
    )
    client.train()
    
    weights = copy.deepcopy(client.get_parameters())
    num_samples = client.result['sample']
    train_result = client.result
    
    del client
    # torch.cuda.empty_cache() # 单GPU多线程场景下避免频繁清理显存导致同步阻塞
    
    return idx, weights, num_samples, train_result


if __name__ == "__main__":
    args = parse_args()
    INPUT_SIZE, TIME_STAMP = 100, 300
    # set random seed
    setup_seed(args.corr_seed)

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

    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    global_round_counter = 0

    # -------------------------------------------------------------------------
    # 第一阶段 (Stage 1): 初始训练与 LID 累积
    # -------------------------------------------------------------------------
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

            # 更新概率，避免重复采样 (逻辑保持原样)
            for idx in idxs_users:
                prob[idx] = 0
            if sum(prob) > 0:
                prob = [prob[i] / sum(prob) for i in range(len(prob))]
            
            # 并行训练 + LID计算
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                for idx in idxs_users:
                    sample_idx = np.array(data_indices[idx])
                    dataset_client = Subset(whole_ds, sample_idx)
                    futures.append(executor.submit(
                        train_fed_corr_client, 
                        idx, args, server.global_model, criterion, dataset_client, run_timestamp, global_round_counter
                    ))
                
                for future in futures:
                    idx, weights, num_samples, train_result, local_output, loss, LID_local = future.result()
                    
                    # 1. 保存训练结果
                    server.save_train_updates(weights, num_samples, train_result)
                    
                    # 2. 处理 LID 和 Loss 统计
                    sample_idx = np.array(data_indices[idx])
                    LID_whole[sample_idx] = LID_local
                    loss_whole[sample_idx] = loss
                    LID_client[idx] = np.mean(LID_local)
            
            server.average_weights()
            global_round_counter += 1

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
            # y_train_noisy_new = np.array(whole_ds.labels) # 此行原代码未被使用

        if args.correction:
            # 这里的修正逻辑也可以考虑并行，但因为它不是训练过程，主要是推理和数据操作，且只针对 noisy_set
            # 暂时保持串行，避免逻辑过于复杂，或者可以简单并行化推理部分
            for idx in noisy_set:
                sample_idx = np.array(list(data_indices[idx]))
                dataset_client = Subset(whole_ds, sample_idx)
                # dl_client = DataLoader(dataset_client, batch_size=args.batch, shuffle=False)
                client = Fed_Corr_client(
                    args,
                    criterion,
                    copy.deepcopy(server.global_model),
                    dataset_client,
                    client_id=idx,
                    run_timestamp=run_timestamp,
                    global_round=global_round_counter
                )
                loss = np.array(loss_accumulative_whole[sample_idx])
                # 这里也需要修正顺序问题
                eval_dl = DataLoader(dataset_client, batch_size=args.batch, shuffle=False)
                local_output, _ = get_output(eval_dl, client.model, args, criterion)
                
                relabel_idx = (-loss).argsort()[:int(len(sample_idx) * estimated_noisy_level[idx] * args.relabel_ratio)]
                relabel_idx = list(set(np.where(np.max(local_output, axis=1) > args.confidence_thres)[0]) & set(relabel_idx))
                
                y_train_noisy_new = np.array(whole_ds.labels)
                y_train_noisy_new[sample_idx[relabel_idx]] = np.argmax(local_output, axis=1)[relabel_idx]
                whole_ds.labels = y_train_noisy_new

    # reset the beta
    args.beta = 0
    
    #---------------------------------- second stage training ----------------------------------
    # 阶段 2 (Stage 2): 基于清洗后的数据微调 (Fine-tuning)
    # -------------------------------------------------------------------------
    if args.fine_tuning:
        selected_clean_idx = np.where(estimated_noisy_level <= args.clean_set_thres)[0]

        prob = np.zeros(args.client_num)
        if len(selected_clean_idx) > 0:
            prob[selected_clean_idx] = 1/len(selected_clean_idx)
        else:
             # Fallback if no clean clients found
            prob = [1/args.client_num] * args.client_num
            
        m = max(int(args.frac2 * args.client_num), 1)
        m = min(m, len(selected_clean_idx)) if len(selected_clean_idx) > 0 else m

        for rnd in range(args.rounds1):
            server.initialize_epoch_updates(rnd)
            idxs_users = np.random.choice(range(args.client_num), m, replace=False, p=prob)
            
            # 并行微调训练
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                for idx in idxs_users:
                    sample_idx = np.array(data_indices[idx])
                    dataset_client = Subset(whole_ds, sample_idx)
                    futures.append(executor.submit(
                        train_simple_client,
                        idx, args, server.global_model, criterion, dataset_client, run_timestamp, global_round_counter
                    ))
                
                for future in futures:
                    idx, weights, num_samples, train_result = future.result()
                    server.save_train_updates(weights, num_samples, train_result)
            
            server.average_weights()
            global_round_counter += 1
        
        if args.correction:
            # 全局修正逻辑
            # 这里涉及全量数据标签修改，暂时保持串行逻辑以确保数据一致性
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
    # 阶段 3 (Stage 3): 最终训练
    # -------------------------------------------------------------------------
    m = max(int(args.sample_rate * args.client_num), 1)
    prob = [1/args.client_num for i in range(args.client_num)]
    print("----------------------STAGE 3--------------------------------")
    # test_dl = gen_cbgru_valid_dl(args.vul, 0, args.batch)
    test_dl = gen_valid_dl(args.model_type, args.vul)

    for rnd in range(args.rounds2):
        idxs_users = np.random.choice(range(args.client_num), m, replace=False, p = prob)
        server.initialize_epoch_updates(rnd)
        print(f"epoch {rnd}:")
        
        # 并行最终训练
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for idx in idxs_users:
                sample_idx = np.array(data_indices[idx])
                dataset_client = Subset(whole_ds, sample_idx)
                futures.append(executor.submit(
                    train_simple_client,
                    idx, args, server.global_model, criterion, dataset_client, run_timestamp, global_round_counter
                ))
            
            for future in futures:
                idx, weights, num_samples, train_result = future.result()
                server.save_train_updates(weights, num_samples, train_result)

        server.average_weights()
        global_round_counter += 1
    
    global_test(server.global_model, test_dl, criterion, args, args.lab_name, 'none')
