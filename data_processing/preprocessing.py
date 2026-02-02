import os
import json
import torch
import random
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from models.KMeansTorch import KMeansTorch


def vec2one(input):
    if input.is_cuda:
        input = input.cpu()
    input = input.numpy()
    one_hot_labels = np.zeros(input.shape[0], 2)
    one_hot_labels[np.arange(input.shape[0]), input.flattern()] = 1
    one_hot_labels = torch.from_numpy(one_hot_labels).to('cuda')
    return one_hot_labels


def get_cbgru_feature(file_path):
    try:
        df = pd.read_pickle(file_path)
        # print(f"{file_path}加载成功！")
    except FileNotFoundError:
        print(f"错误：{file_path}不存在！")

    x_train = np.stack(df.iloc[:, 0].values)
    labels = df.iloc[:,1].values

    return x_train, labels

 
def relabel_with_pretrained_knn(labels, features, num_classses, weights='uniform', num_neighbors=10, noise_theshold=0.15):
    # Initialize
    _labels = np.array(labels, dtype=np.int64)
    knn = KNeighborsClassifier(n_neighbors=num_neighbors, weights=weights, n_jobs=1)
    knn.fit(features, _labels)
    # knn.classes_ = np.arange(2)
    predictions = np.squeeze(knn.predict(features).astype(np.int64))
    probabilities = knn.predict_proba(features)

    # get K neighbours for each sample
    # compute agreement
    agreement_ratrios = []
    distances, indices = knn.kneighbors(features)
    for idx, neighbors in enumerate(indices):
        neighbor_labels = _labels[neighbors]
        agreement_ratrio = np.mean(neighbor_labels == predictions[idx])
        agreement_ratrios.append(agreement_ratrio)

    # Estimate label noise
    # est_noise_lvl = (predictions!=_labels).astype(np.int64).mean()
    return predictions.astype(np.float32), probabilities, agreement_ratrios, indices


def read_pretrain_feature(names, feature_dir):
    features = list()
    for name in names:
        name = name.split('.')[0]
        feature_path = os.path.join(feature_dir, f"{name}.json")
        with open(feature_path, 'r') as file:
            feature = np.array(json.load(file))
            feature = feature.reshape(-1)
        features.append(feature)
    
    features = np.array(features)
    return features


def reduced_name_labels(name_path, labels):
    reduced_names = list()
    reduced_labels = list()
    name_set = set()
    
    with open(name_path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            name = line.strip()
            if name not in name_set:
                name_set.add(name)
                reduced_names.append(name)
                reduced_labels.append(labels[i])

    return reduced_names, reduced_labels


def gen_noise_labels(names_path, labels_path, noise_type, noise_rate):
    names = []
    with open(names_path, 'r') as f:
        lines = f.readlines()
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

    if noise_type == 'non_noise':
        print("non_noise")

    num_to_flip = int(len(unique_labels) * noise_rate)
    indices_to_flip = random.sample(range(len(unique_labels)), num_to_flip)
    flipped_labels = unique_labels[:]
    for idx in indices_to_flip:
        flipped_labels[idx] = 1 - flipped_labels[idx]
    # for i in range(len(unique_labels)):
        # if noise_type == 'fn_noise':
        #     if unique_labels[i] == 1:
        #         if random.random() < noise_rate:
        #             unique_labels[i] = 1-unique_labels[i]
        

    name_label = dict()
    for i in range(len(unique_names)):
        # name_label[unique_names[i]] = unique_labels[i]
        name_label[unique_names[i]] = flipped_labels[i]
    
    noise_labels = []
    for name in names:
        noise_labels.append(name_label[name])
    
    return noise_labels


def gen_sys_noise_labels(names_path, labels_path, pre_feature_dir, noise_type, noise_rate, n_neigh):
    names = []
    with open(names_path, 'r') as f:
        lines = f.readlines()
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
    pre_features = read_pretrain_feature(unique_names, pre_feature_dir)
    _labels = np.array(labels, dtype=np.int64)
    knn = KNeighborsClassifier(n_neighbors=n_neigh, weights='uniform', n_jobs=1)
    knn.fit(pre_features, unique_labels)

    num_samples = len(unique_labels)
    num_noisy_samples = int(num_samples * noise_rate)
    noisy_sample_indices = np.random.choice(num_samples, num_noisy_samples, replace=False)

    modified_samples = set()
    for idx in noisy_sample_indices:
        knn_outputs = knn.predict(pre_features[idx].reshape(1, -1))
        # print("================knn outputs: ===================", knn_outputs)
        noisy_label = 1-knn_outputs[0]
        if idx not in modified_samples:
            unique_labels[idx] = noisy_label
            modified_samples.add(idx)
        
        distances, indices = knn.kneighbors([pre_features[idx]])
        for neighbor_idx in indices[0][1:]:  # 排除自己
            if neighbor_idx not in modified_samples:  
                unique_labels[neighbor_idx] = noisy_label
                modified_samples.add(neighbor_idx)

        if len(modified_samples) >= num_samples * noise_rate:
            break
    
    name_label = dict()
    for i in range(len(unique_names)):
        name_label[unique_names[i]] = unique_labels[i]
    
    noise_labels = []
    for name in names:
        noise_labels.append(name_label[name])
    
    return noise_labels


def compute_global_clusters(all_name_paths, pre_feature_dir, n_clusters=20, seed=42):
    """
    读取所有客户端的合约名称，加载其特征，进行全局聚类。
    返回:
        cluster_map (dict): {contract_name: cluster_id}
    """
    unique_names = set()
    for path in all_name_paths:
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            for line in f:
                unique_names.add(line.strip())
    
    unique_names_list = list(unique_names)
    if not unique_names_list:
        return {}
        
    # Read features
    # pre_features: shape (n_samples, n_features)
    pre_features = read_pretrain_feature(unique_names_list, pre_feature_dir)
    
    # Standardize
    scaler = StandardScaler()
    Zs = scaler.fit_transform(pre_features)
    
    # Global KMeans
    if torch.cuda.is_available():
        print(f"  [Info] Using GPU (KMeansTorch) for global clustering.")
        kmeans = KMeansTorch(n_clusters=n_clusters, seed=seed, device='cuda')
    else:
        print(f"  [Info] Using CPU (sklearn.cluster.KMeans) for global clustering.")
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        
    cluster_ids = kmeans.fit_predict(Zs)
    
    # Create map
    cluster_map = {}
    for name, cid in zip(unique_names_list, cluster_ids):
        cluster_map[name] = cid
        
    return cluster_map


def coordinate_sys_noise_clusters(client_num, vul, noise_type, model_type='CBGRU', n_clusters=20, seed=42, data_dir='./data/'):
    """
    协调系统性噪声的簇分配。
    1. 执行全局聚类。
    2. 统计每个客户端在各个簇中的样本分布。
    3. 优化分配簇 ID，支持共享以保证噪声覆盖。
    
    Returns:
        assigned_clusters_dict (dict): {client_id: [cluster_id, ...]}
        global_cluster_map (dict): {contract_name: cluster_id}
    """
    if noise_type != 'sys_noise':
        return None, None

    print("Computing Global Clusters for Systemic Noise...")
    # 收集所有客户端的 contract_name 文件路径
    all_name_paths = []
    for i in range(client_num):
        # 假设路径结构如下 (需根据实际情况调整，保持与 Fed_LGV.py 一致):
        # client_dir = f"./data/graduate_client_split/{vul}/client_{client_id}/"
        if model_type == "CBGRU":
            client_dir = os.path.join(data_dir, f"graduate_client_split/cbgru/{vul}/client_{i}/")
        elif model_type == "CGE":
            client_dir = os.path.join(data_dir, f"graduate_client_split/cge/{vul}/client_{i}/")
        else:
            client_dir = os.path.join(data_dir, f"graduate_client_split/{vul}/client_{i}/")
            
        names_path = os.path.join(client_dir, "contract_name_train.txt")
        all_name_paths.append(names_path)
        
    pre_feature_dir = os.path.join(data_dir, f"pretrain_feature/{vul}")
    global_cluster_map = compute_global_clusters(all_name_paths, pre_feature_dir, n_clusters=n_clusters, seed=seed)
    print(f"Global clustering completed. Total unique samples: {len(global_cluster_map)}")

    # 2. 分配簇给客户端 (优化版：基于分布 + 允许共享)
    
    # 2.1 统计分布：计算每个客户端在每个全局簇中的样本数量
    client_cluster_counts = {client_id: {c: 0 for c in range(n_clusters)} for client_id in range(client_num)}
    
    for client_id in range(client_num):
        names_path = all_name_paths[client_id]
        if os.path.exists(names_path):
            with open(names_path, 'r') as f:
                for line in f:
                    name = line.strip()
                    if name in global_cluster_map:
                        c_id = global_cluster_map[name]
                        client_cluster_counts[client_id][c_id] += 1

    # 2.2 确定每个客户端的“候选簇”
    candidate_clusters = {client_id: [] for client_id in range(client_num)}
    for client_id in range(client_num):
        for c_id, count in client_cluster_counts[client_id].items():
            if count > 0: # 只要有样本就可以作为候选
                candidate_clusters[client_id].append((c_id, count))
        # 按样本数从大到小排序，优先分配样本多的簇
        candidate_clusters[client_id].sort(key=lambda x: x[1], reverse=True)
        # 只保留簇ID
        candidate_clusters[client_id] = [x[0] for x in candidate_clusters[client_id]]

    # 2.3 贪心分配
    assigned_clusters_dict = {client_id: [] for client_id in range(client_num)}
    unassigned_clusters = set(range(n_clusters))
    
    # 理想情况下每个客户端分到的簇数量
    target_clusters_per_client = int(np.ceil(n_clusters / client_num))
    
    # 第一轮：尝试分配互斥的簇 (Exclusive Assignment)
    for client_id in range(client_num):
        needed = target_clusters_per_client
        for c_id in candidate_clusters[client_id]:
            if len(assigned_clusters_dict[client_id]) >= needed:
                break
            if c_id in unassigned_clusters:
                assigned_clusters_dict[client_id].append(c_id)
                unassigned_clusters.remove(c_id)
    
    # 2.4 补全分配 (Shared Assignment)
    for client_id in range(client_num):
        current_count = len(assigned_clusters_dict[client_id])
        if current_count < target_clusters_per_client:
            needed = target_clusters_per_client - current_count
            # 从它的候选簇中找，即使已经被分配过
            for c_id in candidate_clusters[client_id]:
                if needed <= 0:
                    break
                if c_id not in assigned_clusters_dict[client_id]:
                    assigned_clusters_dict[client_id].append(c_id)
                    needed -= 1
    
    print(f"Systemic Noise Cluster Assignment (Optimized): {assigned_clusters_dict}")
    # 打印一下实际分配的覆盖情况
    cluster_usage = {c: 0 for c in range(n_clusters)}
    for c_list in assigned_clusters_dict.values():
        for c in c_list:
            cluster_usage[c] += 1
    print(f"Cluster Usage Counts: {cluster_usage}")
    
    return assigned_clusters_dict, global_cluster_map


def gen_sys_noise_labels_kmeans(names_path, labels_path, pre_feature_dir, noise_rate, n_clusters=20, seed=42, assigned_cluster_indices=None, global_cluster_map=None):
    """
    基于 KMeans 聚类的系统性标签噪声生成 (方案A)
    在聚类簇内将少数类翻转为多数类，模拟系统性错误
    
    Args:
        assigned_cluster_indices (list, optional): 指定要添加噪声的簇 ID 列表。
                                                 如果提供，将只从这些簇中选择样本进行翻转。
                                                 用于在联邦学习中协调不同客户端的噪声分布。
        global_cluster_map (dict, optional): {name: cluster_id} 全局聚类结果。
                                           如果提供，将直接使用该映射获取簇 ID，不再进行本地聚类。
    """
    names = []
    with open(names_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            names.append(line.strip())
    
    with open(labels_path, 'rb') as file:
        df = pd.read_csv(labels_path, header=None)
        labels = df.iloc[:, 0].values
    
    name_set = set()
    unique_names = []
    unique_labels = []
    # 保持原有逻辑，获取去重后的 names 和 labels (对应 unique_names 和 unique_labels)
    for i, name in enumerate(names):
        if name not in name_set:
            unique_labels.append(labels[i])
            unique_names.append(name)
            name_set.add(name)

    # 转为 numpy
    yk = np.array(unique_labels, dtype=np.int64)
    n = len(yk)
    
    # 目标翻转数
    m_target = int(np.floor(noise_rate * n))
    
    # 1. 获取簇 ID (Global vs Local)
    if global_cluster_map is not None:
        # 使用全局聚类结果
        cluster_ids = []
        for name in unique_names:
            if name in global_cluster_map:
                cluster_ids.append(global_cluster_map[name])
            else:
                # Fallback: 如果名字不在全局映射中 (理论上不应发生)，给个 -1 或随机
                cluster_ids.append(-1) 
        cluster_ids = np.array(cluster_ids)
        
        # 此时不需要读取特征和做本地聚类
    else:
        # 本地聚类 (Fallback)
        # 读取特征 (CodeBERT embedding)
        pre_features = read_pretrain_feature(unique_names, pre_feature_dir)
        Zk = pre_features
        
        # 标准化 + 聚类
        scaler = StandardScaler()
        Zs = scaler.fit_transform(Zk)
        
        if torch.cuda.is_available():
            print(f"Using device: cuda (KMeansTorch)")
            kmeans = KMeansTorch(n_clusters=n_clusters, seed=seed, device='cuda')
        else:
            print(f"Using device: cpu (sklearn.cluster.KMeans)")
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
            
        cluster_ids = kmeans.fit_predict(Zs) # shape (n,)

    # 2. 统计每个簇的翻转候选信息 (Flip Candidates)
    # 策略变更：为了更符合噪声理念并解决样本不足问题，我们将簇内的“多数类”样本翻转为“少数类”标签。
    # 这模拟了在某类样本聚集的区域（特征空间）中，大量样本被错误标注为该区域的主导类（但实际上可能是另一类，或者就是强行加噪）。
    # 这样可以利用多数类样本的丰富性，避免“无可翻转”的问题。
    flip_candidates_info = []
    
    # 确定分配的簇集合
    assigned_set = set()
    if assigned_cluster_indices is not None:
        assigned_set = set(c % n_clusters for c in assigned_cluster_indices)

    # 遍历所有簇，而不仅仅是分配的簇
    # 允许“借用”其他簇的样本来满足噪声率，优先使用分配的簇
    for c in range(n_clusters):
        idx_c = np.where(cluster_ids == c)[0]
        if len(idx_c) == 0:
            continue
            
        # 统计簇内标签分布
        labels_c = yk[idx_c]
        count0 = np.sum(labels_c == 0)
        count1 = np.sum(labels_c == 1)
        
        # 确定多数类（Source for flipping）和少数类（Target label）
        if count0 >= count1:
            maj_label = 0
            min_label = 1
            majority_count = count0
        else:
            maj_label = 1
            min_label = 0
            majority_count = count1
            
        # 找出该簇中属于多数类的样本下标
        # 这些样本是“候选翻转对象” (Source: Majority -> Target: Minority)
        majority_indices_c = idx_c[labels_c == maj_label]
        
        if len(majority_indices_c) > 0:
            flip_candidates_info.append({
                'cluster_id': c,
                'candidate_indices': majority_indices_c,
                'candidate_count': len(majority_indices_c),
                'target_label': min_label, # 翻转后的目标标签
                'is_assigned': (c in assigned_set) if assigned_cluster_indices is not None else True
            })

    # 3. 排序
    # 优先级 1: 是分配给该客户端的簇 (True > False)
    # 优先级 2: 候选样本数量多 (大 > 小)
    flip_candidates_info.sort(key=lambda x: (x['is_assigned'], x['candidate_count']), reverse=True)

    # 4. 收集候选翻转样本，直到满足 m_target
    total_available = sum(info['candidate_count'] for info in flip_candidates_info)
    
    # 贪心选择簇 (Priority 1: Majority -> Minority)
    current_count = 0
    selected_clusters = set()
    chosen_candidates = []
    
    # 先尽可能多地收集多数类样本
    for info in flip_candidates_info:
        if current_count >= m_target:
            break
            
        c_idxs = info['candidate_indices']
        needed = m_target - current_count
        
        if len(c_idxs) <= needed:
            # 当前簇全取
            for idx in c_idxs:
                chosen_candidates.append({'index': idx, 'target_label': info['target_label'], 'cluster_id': info['cluster_id']})
            current_count += len(c_idxs)
            selected_clusters.add(info['cluster_id'])
        else:
            # 当前簇取一部分 (随机)
            rng = np.random.RandomState(seed + info['cluster_id']) 
            selected_c_idxs = rng.choice(c_idxs, size=needed, replace=False)
            for idx in selected_c_idxs:
                chosen_candidates.append({'index': idx, 'target_label': info['target_label'], 'cluster_id': info['cluster_id']})
            current_count += needed
            selected_clusters.add(info['cluster_id'])

    # 5. Fallback: 如果数量不够，从剩余样本中随机选择进行翻转 (Minority -> Majority)
    if current_count < m_target:
        print(f"[Warning] 多数类样本不足以生成 {noise_rate} 的系统性噪声。")
        print(f"  目标: {m_target}, 当前: {current_count}, 缺口: {m_target - current_count}")
        print("  启用 Fallback: 随机翻转剩余样本以满足噪声率。")
        
        # 找出所有已经被选中的样本索引
        chosen_indices = set(item['index'] for item in chosen_candidates)
        
        # 在目标簇范围内寻找所有未被选中的样本
        # (限制在 assigned_cluster_indices 内，如果没有指定则全量)
        available_indices = []
        
        if assigned_cluster_indices is not None:
            # 1. 尝试从分配的簇中找
            valid_clusters = set([c % n_clusters for c in assigned_cluster_indices])
            for idx, c_id in enumerate(cluster_ids):
                if c_id in valid_clusters and idx not in chosen_indices:
                    available_indices.append(idx)
            
            # 2. 如果不够，解除簇限制 (Global Fallback)
            if len(available_indices) < (m_target - current_count):
                print(f"  [Info] 分配簇内样本不足 (可用: {len(available_indices)}). 解除簇限制，在全局搜索...")
                available_indices = []
                for idx in range(n):
                    if idx not in chosen_indices:
                        available_indices.append(idx)
        else:
            # 全局找
            for idx in range(n):
                if idx not in chosen_indices:
                    available_indices.append(idx)
            
        shortage = m_target - current_count
        if len(available_indices) < shortage:
            print(f"[Error] 即使启用 Fallback，样本总数仍不足以满足噪声率！(可用: {len(available_indices)}, 需要: {shortage})")
            shortage = len(available_indices) # 尽力而为
            
        if shortage > 0:
            rng = np.random.RandomState(seed + 999) # 全局 fallback 随机种子
            fallback_indices = rng.choice(available_indices, size=shortage, replace=False)
            
            for idx in fallback_indices:
                # 翻转逻辑：0->1, 1->0
                original_label = yk[idx]
                target_label = 1 - original_label
                c_id = cluster_ids[idx]
                chosen_candidates.append({'index': idx, 'target_label': target_label, 'cluster_id': c_id})
                selected_clusters.add(c_id)
            current_count += shortage

    # 6. 执行改标
    noisy_labels = yk.copy()
    flip_record = [] # 用于调试
    
    for item in chosen_candidates:
        idx = item['index']
        target = item['target_label']
        original = noisy_labels[idx]
        
        noisy_labels[idx] = target
        flip_record.append({
            'index': idx,
            'original': original,
            'new': target,
            'cluster': item['cluster_id']
        })
        
    # 调试输出
    print(f"--- Systemic Noise Generation (KMeans) ---")
    print(f"  Noise Rate: {noise_rate}, Target Flips: {m_target}, Actual Flips: {len(flip_record)}")
    print(f"  Clusters used: {len(selected_clusters)} (IDs: {list(selected_clusters)})")
    if assigned_cluster_indices is not None:
         print(f"  (Assigned Pool: {assigned_cluster_indices})")
    
    flip_0_to_1 = sum(1 for x in flip_record if x['original'] == 0 and x['new'] == 1)
    flip_1_to_0 = sum(1 for x in flip_record if x['original'] == 1 and x['new'] == 0)
    print(f"  Flips 0->1: {flip_0_to_1}, Flips 1->0: {flip_1_to_0}")
    print("------------------------------------------")

    # 7. 映射回原始 names (处理重复 name 的情况)
    name_label_map = dict()
    for i in range(len(unique_names)):
        name_label_map[unique_names[i]] = noisy_labels[i]
    
    final_noise_labels = []
    for name in names:
        final_noise_labels.append(name_label_map[name])
    
    return final_noise_labels


def get_pattern_feature(vul, graph_path):
    train_total_name_path = os.path.join(graph_path, f'contract_name_train.txt')
    valid_total_name_path = f'./merge_sc_dataset/graph_feature/{vul}/contract_name_valid.txt'
    pattern_feature_path = f"./merge_sc_dataset/pattern_feature/original_pattern_feature/{vul}/"

    final_pattern_feature_train = []  # pattern feature train
    pattern_feature_train_label_path = f"./merge_sc_dataset/pattern_feature/feature_by_zeropadding/{vul}/label_by_extractor_train.txt"
    
    final_pattern_feature_valid = []  # pattern feature valid
    pattern_feature_test_label_path = f"./merge_sc_dataset/pattern_feature/feature_by_zeropadding/{vul}/label_by_extractor_valid.txt"

    f_train = open(train_total_name_path, 'r')
    lines = f_train.readlines()
    for line in lines:
        line = line.strip('\n').split('.')[0]
        tmp_feature = np.loadtxt(pattern_feature_path + line + '.txt')
        final_pattern_feature_train.append(tmp_feature)

    f_test = open(valid_total_name_path, 'r')
    lines = f_test.readlines()
    for line in lines:
        line = line.strip('\n').split('.')[0]
        tmp_feature = np.loadtxt(pattern_feature_path + line + '.txt')
        final_pattern_feature_valid.append(tmp_feature)

    # labels of extractor definition
    label_by_extractor_train = []
    f_train_label_extractor = open(pattern_feature_train_label_path, 'r')
    labels = f_train_label_extractor.readlines()
    for label in labels:
        label_by_extractor_train.append(int(label.strip('\n')))

    label_by_extractor_valid = []
    f_test_label_extractor = open(pattern_feature_test_label_path, 'r')
    labels = f_test_label_extractor.readlines()
    for label in labels:
        label_by_extractor_valid.append(int(label.strip('\n')))

    for i in range(len(final_pattern_feature_train)):
        final_pattern_feature_train[i] = final_pattern_feature_train[i].tolist()

    for i in range(len(final_pattern_feature_valid)):
        final_pattern_feature_valid[i] = final_pattern_feature_valid[i].tolist()
    
    # tranfrom list or numpy array to tensor
    final_pattern_feature_train = torch.tensor(final_pattern_feature_train, dtype=torch.float32)
    final_pattern_feature_valid = torch.tensor(final_pattern_feature_valid, dtype=torch.float32)
    label_by_extractor_train = torch.tensor(label_by_extractor_train, dtype=torch.float32)
    label_by_extractor_train = label_by_extractor_train.reshape(-1,1)
    label_by_extractor_valid = torch.tensor(label_by_extractor_valid, dtype=torch.float32)
    label_by_extractor_valid = label_by_extractor_valid.reshape(-1, 1)

    return final_pattern_feature_train, final_pattern_feature_valid, label_by_extractor_train, label_by_extractor_valid


def get_graph_feature(vul, noise_type, graph_path, noise_rate=0.05):
    graph_feature_train_data_path = os.path.join(graph_path, 'train_feature.txt')

    graph_feature_test_data_path = f"./merge_sc_dataset/graph_feature/{vul}/valid_feature.txt"
    graph_feature_test_label_path = f"./merge_sc_dataset/graph_feature/{vul}/label_by_experts_valid.txt"

    train_total_name_path = os.path.join(graph_path, f'contract_name_train.txt')
    graph_feature_train_label_path = os.path.join(graph_path, 'label_by_experts_train.txt')
    label_by_experts_train = flip_values(train_total_name_path, graph_feature_train_label_path, noise_rate, noise_type)

    label_by_experts_valid = []
    f_test_label_expert = open(graph_feature_test_label_path, 'r')
    labels = f_test_label_expert.readlines()
    for label in labels:
        label_by_experts_valid.append(int(label.strip('\n')))

    print(f'Reading Graph Features: {graph_feature_train_data_path}...')
    graph_feature_train = np.loadtxt(graph_feature_train_data_path).tolist()  # graph feature train
    graph_feature_test = np.loadtxt(graph_feature_test_data_path, delimiter=", ").tolist()  # graph feature test

    for i in range(len(graph_feature_train)):
        graph_feature_train[i] = [graph_feature_train[i]]

    for i in range(len(graph_feature_test)):
        graph_feature_test[i] = [graph_feature_test[i]]

    # compute class_weight
    label_by_experts_train = np.array(label_by_experts_train)
    class_weights = compute_class_weight('balanced', classes=np.unique(label_by_experts_train), y=label_by_experts_train)
    if len(class_weights == 1): 
        pos_weight = 0
    else:
        pos_weight = class_weights[1]/class_weights[0]

    # tranfrom list or numpy array to tensor
    graph_feature_train = torch.tensor(graph_feature_train, dtype=torch.float32)
    graph_feature_test = torch.tensor(graph_feature_test, dtype=torch.float32)
    label_by_experts_train = torch.tensor(label_by_experts_train, dtype=torch.float32)
    label_by_experts_train = label_by_experts_train.reshape(-1, 1)
    label_by_experts_valid = torch.tensor(label_by_experts_valid, dtype=torch.float32)
    label_by_experts_valid = label_by_experts_valid.reshape(-1, 1)

    return graph_feature_train, graph_feature_test, label_by_experts_train, label_by_experts_valid, pos_weight