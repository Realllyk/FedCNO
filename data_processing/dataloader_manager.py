import torch
import numpy as np
import random
import pandas as pd
import os
from torch.utils.data import DataLoader,TensorDataset, Subset
from data_processing.CustomerDataset import CustomerDataset
from data_processing.CbgruDataset import CbgruDataset
from data_processing.CgeDataset import CgeDataset
from data_processing.LgvCgeDataset import LgvCgeDataset
from data_processing.PleDataset import NoiseDataset, CBGruDataset
from data_processing.preprocessing import get_cbgru_feature, relabel_with_pretrained_knn, read_pretrain_feature, reduced_name_labels, gen_noise_labels, gen_sys_noise_labels, get_graph_feature, get_pattern_feature, gen_sys_noise_labels_kmeans, compute_global_clusters
# from data_processing.cbgru_dataset import CbgruDataset


# 生成cbgru使用的dataloader
def gen_cbgru_dl(client_id, vul, noise_type, noise_rate, batch = 16, shuffle=True, random_noise = False, data_dir=None):
    if data_dir is None:
        raise ValueError("data_dir must be provided")
    # word2vec_dir = f"/root/autodl-tmp/data/cbgru_data/{vul}/word2vec"
    # fastText_dir = f"/root/autodl-tmp/data/cbgru_data/{vul}/FastText"
    # client_dir = f"/root/autodl-tmp/data/client_split/{vul}/client_{client_id}/"
    # names_path = os.path.join(client_dir, "cbgru_contract_name_train.txt")
    # labels_path = os.path.join(client_dir, f"{noise_type}_label_train_{noise_rate*100:03.0f}.csv")

    # ds = CBGruDataset(word2vec_dir, fastText_dir, labels_path, names_path)

    word2vec_dir = os.path.join(data_dir, f"cbgru_data/{vul}/word2vec")
    fastText_dir = os.path.join(data_dir, f"cbgru_data/{vul}/FastText")
    client_dir = os.path.join(data_dir, f"client_split/{vul}/client_{client_id}/")
    names_path = os.path.join(client_dir, "cbgru_contract_name_train.txt")
    if noise_type == 'fn_noise':
        labels_path = os.path.join(client_dir, f"non_noise_label_train_000.csv")
    else:
        labels_path = os.path.join(client_dir, f"{noise_type}_label_train_{noise_rate*100:03.0f}.csv")

    ds = CustomerDataset(word2vec_dir, fastText_dir, labels_path, names_path)
    # 生成随机
    if random_noise:
        labels_path = os.path.join(client_dir, "non_noise_label_train_000.csv")
        noise_labels = gen_noise_labels(names_path, labels_path, noise_type, noise_rate)
        ds.labels = noise_labels
    dl = DataLoader(ds, batch_size=batch,shuffle=shuffle)
    
    return dl, 100, 300


def gen_cbgru_client_valid_dl(client_id, vul, batch, noise_labels, frac, data_dir=None):
    if data_dir is None:
        raise ValueError("data_dir must be provided")
    word2vec_dir = os.path.join(data_dir, f"cbgru_data/{vul}/word2vec")
    fastText_dir = os.path.join(data_dir, f"cbgru_data/{vul}/FastText")
    client_dir = os.path.join(data_dir, f"client_split/{vul}/client_{client_id}")
    # word2vec_dir = f"./new_dataset/cbgru_data/{vul}/word2vec"
    # fastText_dir = f"./new_dataset/cbgru_data/{vul}/FastText"
    # client_dir = f"./new_dataset/client_split/{vul}/client_{client_id}/"
    names_path = os.path.join(client_dir, "cbgru_contract_name_train.txt")
    labels_path = os.path.join(client_dir, f"non_noise_label_train_000.csv")

    ds = CBGruDataset(word2vec_dir, fastText_dir, labels_path, names_path)
    ds.labels = torch.tensor(np.array(noise_labels))
    n = len(ds)
    sub_size = int(n * frac)
    indices = np.random.permutation(n)[:sub_size]
    sub_ds = Subset(ds, indices)
    dl = DataLoader(sub_ds, batch_size=batch,shuffle=False)
     
    return dl


def gen_cbgru_valid_dl(vul, id=0, batch=16, data_dir=None):
    if data_dir is None:
        raise ValueError("data_dir must be provided")
    embeddings = ['word2vec', 'FastText']
    file_paths = []
    for emb in embeddings:
        file_paths.append(os.path.join(data_dir, f'cbgru_data/{vul}/cbgru_valid_dataset_{emb}_fragment_vectors.pkl'))
        # file_paths.append(f'../merge_sc_dataset/client_split/{vul}/client_{id}/cbgru_non_noise_{0.05*100:03.0f}_{emb}_fragment_vectors.pkl')

    x1, y = get_cbgru_feature(file_paths[0])
    x2, _ = get_cbgru_feature(file_paths[1])
    
    x1 = torch.tensor(x1.reshape(-1, 1, 100, 300), dtype=torch.float32)
    x2 = torch.tensor(x2, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    y = y.flatten().long()

    ds = TensorDataset(x1, x2, y)
    dl = DataLoader(ds,batch_size=batch)
    
    return dl

    
def gen_whole_dataset(model_type, client_num, vul, noise_type, noise_rates, num_neigh=0, assigned_clusters=None, global_cluster_map=None, n_clusters=20, seed=42, data_dir=None):
    if data_dir is None:
        raise ValueError("data_dir must be provided")
    if(model_type == "CBGRU"):
        return gen_cbgru_whole_dataset(client_num, vul, noise_type, noise_rates, num_neigh, assigned_clusters, global_cluster_map, n_clusters, seed, data_dir)
    else:
        return gen_cge_whole_dataset(client_num, vul, noise_type, noise_rates, num_neigh, assigned_clusters, global_cluster_map, n_clusters, seed, data_dir)


def gen_cbgru_whole_dataset(client_num, vul, noise_type, noise_rates, num_neigh=0, assigned_clusters=None, global_cluster_map=None, n_clusters=20, seed=42, data_dir=None):
    if data_dir is None:
        raise ValueError("data_dir must be provided")
    word2vec_dir = os.path.join(data_dir, f"cbgru_data/{vul}/word2vec")
    fastText_dir = os.path.join(data_dir, f"cbgru_data/{vul}/FastText")
    all_names = []
    all_labels = []
    # names_path = f"./data/4_client_split/{vul}/client_0/contract_name_train.txt"
    # labels_path = f"./data/4_client_split/{vul}/client_0/label_train.csv"
    names_path = os.path.join(data_dir, f"graduate_client_split/{vul}/client_0/contract_name_train.txt")
    labels_path = os.path.join(data_dir, f"graduate_client_split/{vul}/client_0/label_train.csv")
    ds = CBGruDataset(word2vec_dir, fastText_dir, labels_path, names_path)
    data_indices = []
    offset = 0
    
    for client_id in range(client_num):
        # client_dir = f"./data/4_client_split/{vul}/client_{client_id}/"
        client_dir = os.path.join(data_dir, f"graduate_client_split/{vul}/client_{client_id}/")
        names_path = os.path.join(client_dir, "contract_name_train.txt")
        labels_path = os.path.join(client_dir, f"label_train.csv")
        names = []
        with open(names_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                name = line.strip()
                name = name.split('.')[0]
                names.append(name)
        n_data = len(names)
        bound = offset+n_data
        all_names.extend(names)

        if noise_type == 'non_noise' or noise_type == 'fn_noise':
            noise_labels = gen_noise_labels(names_path, labels_path, noise_type, noise_rates[client_id])
        elif noise_type == 'sys_noise':
            pre_feature_dir = os.path.join(data_dir, f"pretrain_feature/{vul}")
            # noise_labels = gen_sys_noise_labels(names_path, labels_path, pre_feature_dir, 'sys_noise', noise_rates[client_id], num_neigh)
            # 使用基于 KMeans 的新系统性噪声生成方法
            cluster_indices = assigned_clusters[client_id] if assigned_clusters else None
            noise_labels = gen_sys_noise_labels_kmeans(names_path, labels_path, pre_feature_dir, noise_rates[client_id], n_clusters=n_clusters, seed=seed, assigned_cluster_indices=cluster_indices, global_cluster_map=global_cluster_map)
        all_labels.extend(noise_labels)

        data_indices.append(list(range(offset, bound)))
    
    ds.names = all_names
    ds.labels = all_labels
        
    return ds, data_indices


def gen_cge_whole_dataset(client_num, vul, noise_type, noise_rate, num_neigh=0, assigned_clusters=None, global_cluster_map=None, n_clusters=20, seed=42, data_dir=None):
    if data_dir is None:
        raise ValueError("data_dir must be provided")
    graph_dir = os.path.join(data_dir, f'cge_data/{vul}/graph_feature')
    pattern_dir = os.path.join(data_dir, f'cge_data/{vul}/pattern_feature')
    all_names = []
    all_labels = []
    # names_path = f"./data/4_client_split/{vul}/client_0/contract_name_train.txt"
    # labels_path = f"./data/4_client_split/{vul}/client_0/label_train.csv"
    names_path = os.path.join(data_dir, f"graduate_client_split/{vul}/client_0/contract_name_train.txt")
    labels_path = os.path.join(data_dir, f"graduate_client_split/{vul}/client_0/label_train.csv")
    ds = CgeDataset(graph_dir, pattern_dir, labels_path, names_path)
    data_indices = []
    offset = 0
    
    for client_id in range(client_num):
        # client_dir = f"./data/4_client_split/{vul}/client_{client_id}/"
        client_dir = os.path.join(data_dir, f"graduate_client_split/{vul}/client_{client_id}/")
        names_path = os.path.join(client_dir, "contract_name_train.txt")
        labels_path = os.path.join(client_dir, f"label_train.csv")
        names = []
        with open(names_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                name = line.strip()
                name = name.split('.')[0]
                names.append(name)
        n_data = len(names)
        bound = offset+n_data
        all_names.extend(names)

        if noise_type == 'non_noise' or noise_type == 'fn_noise':
            noise_labels = gen_noise_labels(names_path, labels_path, noise_type, noise_rate)
        elif noise_type == 'sys_noise':
            pre_feature_dir = os.path.join(data_dir, f"pretrain_feature/{vul}")
            # noise_labels = gen_sys_noise_labels(names_path, labels_path, pre_feature_dir, 'sys_noise', noise_rate, num_neigh)
            # 使用基于 KMeans 的新系统性噪声生成方法
            cluster_indices = assigned_clusters[client_id] if assigned_clusters else None
            noise_labels = gen_sys_noise_labels_kmeans(names_path, labels_path, pre_feature_dir, noise_rate, n_clusters=n_clusters, seed=seed, assigned_cluster_indices=cluster_indices, global_cluster_map=global_cluster_map)
        all_labels.extend(noise_labels)

        data_indices.append(list(range(offset, bound)))
    
    ds.names = all_names
    ds.labels = all_labels
        
    return ds, data_indices


def gen_knn_dl(client_id, vul, noise_type, noise_rate, batch, num_neigh):
    embeddings = ['word2vec', 'FastText']
    file_paths = []

    if noise_type == 'noise':
        for emb in embeddings:
            file_paths.append(f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_non_noise_{noise_rate*100:03.0f}_{emb}_fragment_vectors.pkl')
    elif noise_type == 'fn_noise':
        # Todo
        for emb in embeddings:
            file_paths.append(f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_fn_noise_{noise_rate*100:03.0f}_{emb}_fragment_vectors.pkl')
    # 无噪声数据集（与下面的gen_cbgru_client_pure_dl不同，下面的函数用于挑选纯净数据集）
    elif noise_type == 'pure':
        for emb in embeddings:
            file_paths.append(f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_pure_{emb}_fragment_vectors.pkl')
    
    x1, y = get_cbgru_feature(file_paths[0])
    x2, _ = get_cbgru_feature(file_paths[1])
    input_size, time_steps = x1.shape[1], x1.shape[2]

    # Run KNN relable
    feature_dir = f'../merge_sc_dataset/source_code/{vul}/pretrain_feature'
    name_path = f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/cbgru_contract_name_train.txt'
    relabel_path = f'../merge_sc_dataset/client_split/{vul}/client_{client_id}/{noise_type}_{noise_rate}_relabel.txt'
    if not os.path.exists(relabel_path):
        reduced_names, reduced_labels = reduced_name_labels(name_path, y)
        features = read_pretrain_feature(reduced_names, feature_dir)
        relabels = relabel_with_pretrained_knn(reduced_labels, features, 2, 'uniform', num_neigh, 0.15)
        
        name_relabels = dict()
        for i, name in enumerate(reduced_names):
            name_relabels[name] = relabels[i]

        relabels = list()
        # read name list, map labels
        with open(name_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                name = line.strip()
                relabels.append(name_relabels[name])

    x1 = torch.tensor(x1.reshape(-1, 1, 100, 300), dtype=torch.float32)
    x2 = torch.tensor(x2, dtype=torch.float32)
    print(x1.shape, len(relabels))
    y = torch.tensor(relabels)
    y = y.flatten().long()

    ds = TensorDataset(x1, x2, y)
    dl = DataLoader(ds, batch, shuffle=True)
    return dl, input_size, time_steps
    

# 文件生成好的噪声标签
def gen_lgv_ds(client_id, vul, noise_type, noise_rate, random_noise, num_neigh, model_type, assigned_clusters=None, global_cluster_map=None, n_clusters=20, seed=42, predefined_labels=None, data_dir=None):
    if data_dir is None:
        raise ValueError("data_dir must be provided")
    if model_type == "CBGRU":
        ds = gen_lgv_cbgru_ds(client_id, vul, noise_type, noise_rate, random_noise, num_neigh, assigned_clusters, global_cluster_map, n_clusters, seed, predefined_labels, data_dir)
    elif model_type == "CGE":
        ds = gen_lgv_cge_ds(client_id, vul, noise_type, noise_rate, random_noise, num_neigh, assigned_clusters, global_cluster_map, n_clusters, seed, predefined_labels, data_dir)
        
    return ds
    

def gen_lgv_cbgru_ds(client_id, vul, noise_type, noise_rate, random_noise, num_neigh, assigned_clusters=None, global_cluster_map=None, n_clusters=20, seed=42, predefined_labels=None, data_dir=None):
    if data_dir is None:
        raise ValueError("data_dir must be provided")
    word2vec_dir = os.path.join(data_dir, f"cbgru_data/{vul}/word2vec")
    fastText_dir = os.path.join(data_dir, f"cbgru_data/{vul}/FastText")
    # client_dir = f"./data/client_split/{vul}/client_{client_id}/"
    # names_path = os.path.join(client_dir, "cbgru_contract_name_train.txt")
    # client_dir = f"./data/4_client_split/{vul}/client_{client_id}/"
    client_dir = os.path.join(data_dir, f"graduate_client_split/{vul}/client_{client_id}/")
    names_path = os.path.join(client_dir, "contract_name_train.txt")
    labels_path = os.path.join(client_dir, f"label_train.csv")
    ds = CustomerDataset(word2vec_dir, fastText_dir, labels_path, names_path)

    if random_noise:
        if predefined_labels is not None:
            noise_labels = predefined_labels
        else:
            if noise_type == 'non_noise' or noise_type == 'fn_noise':
                noise_labels = gen_noise_labels(names_path, labels_path, noise_type, noise_rate)
            elif noise_type == 'sys_noise':
                pre_feature_dir = os.path.join(data_dir, f"pretrain_feature/{vul}")
                # noise_labels = gen_sys_noise_labels(names_path, labels_path, pre_feature_dir, 'sys_noise', noise_rate, num_neigh)
                # 使用基于 KMeans 的新系统性噪声生成方法
                cluster_indices = assigned_clusters[client_id] if assigned_clusters else None
                noise_labels = gen_sys_noise_labels_kmeans(names_path, labels_path, pre_feature_dir, noise_rate, n_clusters=n_clusters, seed=seed, assigned_cluster_indices=cluster_indices, global_cluster_map=global_cluster_map)
        ds.labels = noise_labels

    return ds


def gen_lgv_cge_ds(client_id, vul, noise_type, noise_rate, random_noise, num_neigh, assigned_clusters=None, global_cluster_map=None, n_clusters=20, seed=42, predefined_labels=None, data_dir=None):
    if data_dir is None:
        raise ValueError("data_dir must be provided")
    graph_dir = os.path.join(data_dir, f'cge_data/{vul}/graph_feature')
    pattern_dir = os.path.join(data_dir, f'cge_data/{vul}/pattern_feature')
    # client_dir = f"./data/4_client_split/{vul}/client_{client_id}/"
    client_dir = os.path.join(data_dir, f"graduate_client_split/{vul}/client_{client_id}/")
    names_path = os.path.join(client_dir, "contract_name_train.txt")
    labels_path = os.path.join(client_dir, f"label_train.csv")
    ds = LgvCgeDataset(graph_dir, pattern_dir, labels_path, names_path)
    
    if predefined_labels is not None:
        noise_labels = predefined_labels
    else:
        if noise_type == 'non_noise' or noise_type == 'fn_noise':
            noise_labels = gen_noise_labels(names_path, labels_path, noise_type, noise_rate)
        elif noise_type == 'sys_noise':
            pre_feature_dir = os.path.join(data_dir, f"pretrain_feature/{vul}")
            # noise_labels = gen_sys_noise_labels(names_path, labels_path, pre_feature_dir, 'sys_noise', noise_rate, num_neigh)
            # 使用基于 KMeans 的新系统性噪声生成方法
            cluster_indices = assigned_clusters[client_id] if assigned_clusters else None
            noise_labels = gen_sys_noise_labels_kmeans(names_path, labels_path, pre_feature_dir, noise_rate, n_clusters=n_clusters, seed=seed, assigned_cluster_indices=cluster_indices, global_cluster_map=global_cluster_map)
    ds.labels = noise_labels
    return ds


def gen_client_ds(model_type, client_id, vul, noise_type, noise_rate, random_noise, num_neigh, assigned_clusters=None, global_cluster_map=None, n_clusters=20, seed=42, data_dir=None):
    if data_dir is None:
        raise ValueError("data_dir must be provided")
    if model_type == 'CBGRU':
        ds = gen_cbgru_client_ds(client_id, vul, noise_type, noise_rate, random_noise, num_neigh, assigned_clusters, global_cluster_map, n_clusters, seed, data_dir)
    elif model_type == 'CGE':
        ds = gen_cge_client_ds(client_id, vul, noise_type, noise_rate, num_neigh, assigned_clusters, global_cluster_map, n_clusters, seed, data_dir)
    return ds


def gen_cbgru_client_ds(client_id, vul, noise_type, noise_rate, random_noise, num_neigh, assigned_clusters=None, global_cluster_map=None, n_clusters=20, seed=42, data_dir=None):
    if data_dir is None:
        raise ValueError("data_dir must be provided")
    word2vec_dir = os.path.join(data_dir, f"cbgru_data/{vul}/word2vec")
    fastText_dir = os.path.join(data_dir, f"cbgru_data/{vul}/FastText")
    # client_dir = f"./data/client_split/{vul}/client_{client_id}/"
    # names_path = os.path.join(client_dir, "cbgru_contract_name_train.txt")
    # if noise_type == 'fn_noise' or noise_type == 'sys_noise':
    #     labels_path = os.path.join(client_dir, f"non_noise_label_train_000.csv")
    # elif noise_type == 'non_noise':
    #     labels_path = os.path.join(client_dir, f"{noise_type}_label_train_{noise_rate*100:03.0f}.csv")
    # client_dir = f"./data/4_client_split/{vul}/client_{client_id}/"
    # client_dir = f"./data/graduate_client_split/{vul}/cbgru/client_{client_id}/"
    client_dir = os.path.join(data_dir, f"graduate_client_split/{vul}/client_{client_id}/")
    names_path = os.path.join(client_dir, "contract_name_train.txt")
    labels_path = os.path.join(client_dir, f"label_train.csv")
    
    ds = CbgruDataset(word2vec_dir, fastText_dir, labels_path, names_path)
    if noise_type == 'non_noise' or noise_type == 'fn_noise':
        noise_labels = gen_noise_labels(names_path, labels_path, noise_type, noise_rate)
    elif noise_type == 'sys_noise':
        pre_feature_dir = os.path.join(data_dir, f"pretrain_feature/{vul}")
        # noise_labels = gen_sys_noise_labels(names_path, labels_path, pre_feature_dir, 'sys_noise', noise_rate, num_neigh)
        # 使用基于 KMeans 的新系统性噪声生成方法
        cluster_indices = assigned_clusters[client_id] if assigned_clusters else None
        noise_labels = gen_sys_noise_labels_kmeans(names_path, labels_path, pre_feature_dir, noise_rate, n_clusters=n_clusters, seed=seed, assigned_cluster_indices=cluster_indices, global_cluster_map=global_cluster_map)
    ds.labels = noise_labels
        
    return ds


def gen_cbgru_test_ds(vul, data_dir=None):
    if data_dir is None:
        raise ValueError("data_dir must be provided")
    word2vec_dir = os.path.join(data_dir, f"cbgru_data/{vul}/word2vec")
    fastText_dir = os.path.join(data_dir, f"cbgru_data/{vul}/FastText")
    # names_path = f'./data/4_client_split/{vul}/contract_name_test.txt'
    # labels_path = f'./data/4_client_split/{vul}/label_test.csv'
    names_path = os.path.join(data_dir, f"graduate_client_split/{vul}/contract_name_test.txt")
    labels_path = os.path.join(data_dir, f"graduate_client_split/{vul}/label_test.csv")
    ds = CbgruDataset(word2vec_dir, fastText_dir, labels_path, names_path)
    return ds


def gen_cge_test_ds(vul, data_dir=None):
    if data_dir is None:
        raise ValueError("data_dir must be provided")
    graph_dir = os.path.join(data_dir, f'cge_data/{vul}/graph_feature')
    pattern_dir = os.path.join(data_dir, f'cge_data/{vul}/pattern_feature')
    # names_path = f'./data/4_client_split/{vul}/contract_name_test.txt'
    # labels_path = f'./data/4_client_split/{vul}/label_test.csv'
    names_path = os.path.join(data_dir, f"graduate_client_split/{vul}/contract_name_test.txt")
    labels_path = os.path.join(data_dir, f"graduate_client_split/{vul}/label_test.csv")
    ds = CgeDataset(graph_dir, pattern_dir, labels_path, names_path)
    return ds


def gen_test_dl(model_type, vul, data_dir=None):
    if model_type == "CBGRU":
        ds = gen_cbgru_test_ds(vul, data_dir=data_dir)
    elif model_type == "CGE":
        ds = gen_cge_test_ds(vul, data_dir=data_dir)

    dl = DataLoader(ds)
    return dl


def gen_cge_client_ds(client_id, vul, noise_type, noise_rate, num_neigh, assigned_clusters=None, global_cluster_map=None, n_clusters=20, seed=42, data_dir=None):
    if data_dir is None:
        raise ValueError("data_dir must be provided")
    graph_dir = os.path.join(data_dir, f'cge_data/{vul}/graph_feature')
    pattern_dir = os.path.join(data_dir, f'cge_data/{vul}/pattern_feature')
    # client_dir = f"./data/4_client_split/{vul}/client_{client_id}/"
    client_dir = os.path.join(data_dir, f"graduate_client_split/{vul}/client_{client_id}/")
    names_path = os.path.join(client_dir, "contract_name_train.txt")
    labels_path = os.path.join(client_dir, f"label_train.csv")
    ds = CgeDataset(graph_dir, pattern_dir, labels_path, names_path)
    
    if noise_type == 'non_noise' or noise_type == 'fn_noise':
        noise_labels = gen_noise_labels(names_path, labels_path, noise_type, noise_rate)
    elif noise_type == 'sys_noise':
        pre_feature_dir = os.path.join(data_dir, f"pretrain_feature/{vul}")
        # noise_labels = gen_sys_noise_labels(names_path, labels_path, pre_feature_dir, 'sys_noise', noise_rate, num_neigh)
        # 使用基于 KMeans 的新系统性噪声生成方法
        cluster_indices = assigned_clusters[client_id] if assigned_clusters else None
        noise_labels = gen_sys_noise_labels_kmeans(names_path, labels_path, pre_feature_dir, noise_rate, n_clusters=n_clusters, seed=seed, assigned_cluster_indices=cluster_indices, global_cluster_map=global_cluster_map)
    ds.labels = noise_labels
    return ds


def gen_valid_ds(model_type, vul, data_dir=None):
    if data_dir is None:
        raise ValueError("data_dir must be provided")
    if model_type == "CBGRU":
        word2vec_dir = os.path.join(data_dir, f"cbgru_data/{vul}/word2vec")
        fastText_dir = os.path.join(data_dir, f"cbgru_data/{vul}/FastText")
        names_path = os.path.join(data_dir, f"graduate_client_split/{vul}/contract_name_valid.txt")
        labels_path = os.path.join(data_dir, f"graduate_client_split/{vul}/label_valid.csv")
        ds = CbgruDataset(word2vec_dir, fastText_dir, labels_path, names_path)
    elif model_type == "CGE":
        graph_dir = os.path.join(data_dir, f'cge_data/{vul}/graph_feature')
        pattern_dir = os.path.join(data_dir, f'cge_data/{vul}/pattern_feature')
        names_path = os.path.join(data_dir, f"graduate_client_split/{vul}/contract_name_valid.txt")
        labels_path = os.path.join(data_dir, f"graduate_client_split/{vul}/label_valid.csv")
        ds = CgeDataset(graph_dir, pattern_dir, labels_path, names_path)
    return ds


def gen_valid_dl(model_type, vul, data_dir=None):
    ds = gen_valid_ds(model_type, vul, data_dir=data_dir)
    dl = DataLoader(ds)
    return dl
        

#  ---------------------------------test---------------------------------------
def gen_cbgru_client_pure_dl(client_id, vul, noise_type, noise_rate, batch=16, data_dir="./"):
    word2vec_dir = os.path.join(data_dir, f"cbgru_data/{vul}/word2vec")
    fastText_dir = os.path.join(data_dir, f"cbgru_data/{vul}/FastText")
    client_dir = os.path.join(data_dir, f"client_split/{vul}/client_{client_id}/")
    names_path = os.path.join(client_dir, "cbgru_contract_name_train.txt")
    labels_path = os.path.join(client_dir, f"{noise_type}_label_train_000.csv")

    ds = CBGruDataset(word2vec_dir, fastText_dir, labels_path, names_path)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    
    return dl

def gen_cbgru_client_noise_dl(client_id, vul, noise_type, global_labels, noise_rate, batch, noise_labels, data_dir=None):
    if data_dir is None:
        raise ValueError("data_dir must be provided")
    word2vec_dir = os.path.join(data_dir, f"cbgru_data/{vul}/word2vec")
    fastText_dir = os.path.join(data_dir, f"cbgru_data/{vul}/FastText")
    client_dir = os.path.join(data_dir, f"client_split/{vul}/client_{client_id}")
    names_path = os.path.join(client_dir, "cbgru_contract_name_train.txt")
    # labels_path = os.path.join(client_dir, f"{noise_type}_label_train_{noise_rate*100:03.0f}.csv")
    labels_path = os.path.join(client_dir, f"{noise_type}_label_train_000.csv")

    ds = NoiseDataset(word2vec_dir, fastText_dir, labels_path, names_path, global_labels)
    ds.labels = torch.tensor(np.array(noise_labels))
    dl = DataLoader(ds, batch, shuffle=True)

    return dl