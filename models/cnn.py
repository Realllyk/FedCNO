"""
CNN的pytorch实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from options import parse_args


args = parse_args()


class CNN(nn.Module):
    def __init__(self, INPUT_SIZE, TIME_STEPS):
        super().__init__()
        self.con1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5,stride=(1,1), padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.con2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding='same')
        # 原来keras论文没有指定stride，默认与kernal相同
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        x = F.relu(self.con1(x))
        x = self.pool1(x)
        x = F.relu(self.con2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return x
    
    @property
    def output_features(self):
        return 64*25*75
    
def get_cnn(INPUT_SIZE, TIME_STEPS):
    return CNN(INPUT_SIZE, TIME_STEPS)
