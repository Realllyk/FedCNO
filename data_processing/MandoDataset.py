import os
import pandas as pd
import torch
from torch.utils.data import Dataset


def _read_labels(labels_path):
    # 与现有工程保持一致: 单列 CSV，无表头
    df = pd.read_csv(labels_path, header=None)
    return df.iloc[:, 0].values


def _read_names(names_path):
    # 每行一个合约名；若含扩展名则只保留主名
    names = []
    with open(names_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip().split(".")[0]
            if name:
                names.append(name)
    return names


def _default_graph(feature_dim=80):
    # 图缺失/损坏时的兜底样本，避免训练直接崩溃
    return {"node_features": torch.zeros((1, feature_dim), dtype=torch.float32)}


class MandoDataset(Dataset):
    def __init__(self, graph_dir, labels_path, names_path, feature_dim=80):
        # graph_dir 下要求存在 {contract_id}.pt
        self.graph_dir = graph_dir
        self.labels = _read_labels(labels_path)
        self.names = _read_names(names_path)
        self.feature_dim = feature_dim

    def __len__(self):
        return len(self.labels)

    def _load_graph(self, name):
        graph_path = os.path.join(self.graph_dir, f"{name}.pt")
        if not os.path.exists(graph_path):
            return _default_graph(self.feature_dim)
        data = torch.load(graph_path, map_location="cpu")
        if isinstance(data, dict) and "node_features" in data:
            return {"node_features": data["node_features"].float()}
        # 文件结构不符合预期时也回退到默认图
        return _default_graph(self.feature_dim)

    def __getitem__(self, idx):
        # 返回结构与主工程统一: (x1, x2, y)
        # 其中 x1 是图字典，x2 是占位特征（目前未使用）
        graph_item = self._load_graph(self.names[idx])
        aux = torch.zeros(1, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return graph_item, aux, label
