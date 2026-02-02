import os
import pandas as pd
import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class CgeDataset(Dataset):
    def __init__(self, graph_dir, pattern_dir, labels_path, names_path, preload=True):
        labels = []
        with open(labels_path, 'rb') as file:
            df = pd.read_csv(labels_path, header=None)
            labels = df.iloc[:, 0].values
        self.labels = labels

        names = []
        with open(names_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                name = line.strip()
                name = name.split('.')[0]
                names.append(name)
        self.names = names

        self.graph_dir = graph_dir
        self.pattern_dir = pattern_dir
        
        self.preload = preload
        self.data_cache = []
        if self.preload:
            print(f"Preloading {len(self.names)} samples into memory...")
            for i in range(len(self.names)):
                self.data_cache.append(self.load_item(i))
            print("Preloading complete.")

    def load_item(self, idx):
        graph_path = os.path.join(self.graph_dir, f"{self.names[idx]}.txt")
        pattern_path = os.path.join(self.pattern_dir, f"{self.names[idx]}.txt")
        with open(graph_path, 'r') as f:
            graph_feature = np.loadtxt(graph_path).tolist()
            graph_feature = [graph_feature]
            graph_feature = torch.tensor(graph_feature, dtype=torch.float32)
        with open(pattern_path, 'r') as f:
            pattern_feature = np.loadtxt(pattern_path).tolist()
            pattern_feature = torch.tensor(pattern_feature, dtype=torch.float32)
        return graph_feature, pattern_feature

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.preload:
            graph_feature, pattern_feature = self.data_cache[idx]
        else:
            graph_feature, pattern_feature = self.load_item(idx)
            
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return graph_feature, pattern_feature, label
        