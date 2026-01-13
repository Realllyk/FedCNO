import os
import json
import torch
import random
import numpy as np
import torch
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


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
    return predictions.astype(np.float32), probabilities, agreement_ratrios,


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

    if noise_type == 'non_nosie':
        print("non_noise")

    num_to_flip = int(len(unique_labels) * noise_rate)
    indices_to_flip = random.sample(range(len(labels)), num_to_flip)
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