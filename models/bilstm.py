"""
blstm的pytorch实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from options import parse_args


args = parse_args()


class bilstm(nn.Module):
    def __init__(self, INPUT_SIZE, TIME_STEP, hidden_dim=300, dropout=args.dropout):
        # Pytorch中不需要指定TIME_STEP
        # 这里作者好像搞错了INPUT_SIZE和TIME_STEP的顺序，先姑且用着TIME_STEP
        super().__init__()
        self.input_size = TIME_STEP
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_size = self.input_size,
                            hidden_size = self.hidden_dim,
                            num_layers = 1,
                            batch_first = True,
                            bidirectional = True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.hidden_dim*2, hidden_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        x = F.relu(lstm_out)
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x
    
    @property
    def output_features(self):
        return 300
    

def get_bilstm(INPUT_SIZE, TIME_STEP):
    return bilstm(INPUT_SIZE, TIME_STEP)

        
