import numpy as np
import os
from collections import Counter


# 这里orgin_labels存储的是noise
if __name__ == "__main__":
    vuls = ["timestamp", "reentrancy"]
    client_num = 8
    for vul in vuls:
        name_path = f"../../merge_sc_dataset/graph_feature/{vul}/contract_name_train.txt"
        pure_label_path = f"../../merge_sc_dataset/graph_feature/{vul}/label_by_experts_train.txt"
        feature_path = f"../../merge_sc_dataset/graph_feature/{vul}/train_feature.txt"
        output_dir = f"../../merge_sc_dataset/client_split/{vul}"

        # 加载纯净标签
        pure_labels = []
        with open(pure_label_path, 'r') as f:
            labels = f.readlines()
            for label in labels:
                pure_labels.append(int(label.strip('\n')))
        print(len(pure_labels))

        # 创建记载contract_name的set，便于分为不同的client
        name_set = set()
        origin_name = []
        with open(name_path, 'r') as f:
            contracts = f.readlines()
            for index, con_name in enumerate(contracts):
                con_name = con_name.strip('\n')
                origin_name.append(con_name)
                if(con_name not in name_set):
                    name_set.add(con_name)

        # 加载图特征（不同合约的模式特征写在不同名字的文件中，不需要做分割，后续根据名字去读就好了）
        graph_feature_train = np.loadtxt(feature_path)
        
        # 将name_set分到不同的client中
        client_set = [set() for _ in range(client_num)]
        name_nums = len(name_set)
        client_size = name_nums // client_num
        print(name_nums, client_size)
        for index, name in enumerate(name_set):
            client_index = index // client_size
            if client_index < client_num:
                client_set[client_index].add(name)
            else:
                client_set[-1].add(name)

        # 遍历name，并根据client_set来保存相应信息
        for i in range(client_num):
            client_dir = os.path.join(output_dir, f'client_{i}')
            os.makedirs(client_dir, exist_ok=True)
            output_name = os.path.join(client_dir, 'contract_name_train.txt')
            output_pure_label = os.path.join(client_dir, 'label_by_experts_train.txt')
            output_feature = os.path.join(client_dir, 'train_feature.txt')

            names = []
            n_labels = []
            p_labels = []
            features = []
            for idx, name in enumerate(origin_name):
                if name in client_set[i]:
                    names.append(name)
                    p_labels.append(pure_labels[idx])
                    features.append(graph_feature_train[idx])
            
            np.savetxt(output_name, names, fmt='%s', delimiter=' ')
            np.savetxt(output_pure_label, p_labels, '%s', delimiter=' ')
            np.savetxt(output_feature, features, fmt = '%s', delimiter=' ')



    