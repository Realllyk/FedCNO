import torch
import torch.nn as nn

class Lable_Embedding(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc1(in_channels, 128)
    
    def forward(self, pse_labels):
        x = self.fc1(pse_labels)
        return x