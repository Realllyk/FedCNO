import torch
from torch.utils.data import Dataset

class CbgruDataset(Dataset):
    def __init__(self, x1, x2, y):
        self.x1 = x1
        self.x2 = x2
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.y[idx]