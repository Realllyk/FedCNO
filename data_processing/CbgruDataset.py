import numpy as np
import os
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class CbgruDataset(Dataset):
    def __init__(self, word2vec_dir, fastText_dir, labels_path, name_path):
        labels = []
        with open(labels_path, 'rb') as file:
            df = pd.read_csv(labels_path, header=None)
            labels = df.iloc[:, 0].values
        self.labels = labels

        names = []
        with open(name_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                name = line.strip()
                name = name.split('.')[0]
                names.append(name)
        self.names = names

        self.word2vec_dir = word2vec_dir
        self.fastText_dir = fastText_dir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        word2vec_path = os.path.join(self.word2vec_dir, f"{self.names[idx]}.pkl")
        fastText_path = os.path.join(self.fastText_dir, f"{self.names[idx]}.pkl")
        try:
            with open(word2vec_path, 'rb') as f:
                word2vec = pickle.load(f)
                if isinstance(word2vec, list):
                    print(f"[DEBUG] Found list type in word2vec file: {word2vec_path}")
                # if isinstance(word2vec, list):
                #     # print(f"DEBUG: Data at {word2vec_path} is a LIST, not numpy array.")
                #     word2vec = np.array(word2vec)
                word2vec = torch.tensor(word2vec.reshape(1, 100, 300), dtype=torch.float32)
            with open(fastText_path, 'rb') as f:
                fastText = pickle.load(f)
                if isinstance(fastText, list):
                     print(f"[DEBUG] Found list type in fastText file: {fastText_path}")
                # if isinstance(fastText, list):
                #     fastText = np.array(fastText)
                fastText = torch.tensor(fastText, dtype=torch.float32)
            label = torch.tensor(self.labels[idx], dtype=torch.long)
        except Exception as e:
            print(f"Error loading data for index {idx}, name: {self.names[idx]}")
            print(f"Word2Vec Path: {word2vec_path}")
            print(f"FastText Path: {fastText_path}")
            print(f"Error details: {e}")
            raise e

        return word2vec, fastText, label

