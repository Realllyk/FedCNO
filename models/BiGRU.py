"""
BiGRU的pytorch实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from options import parse_args


args = parse_args()


class BiGRU(nn.Module):
    def __init__(self, INPUT_SIZE, TIME_STEPS):
        super().__init__()
        # Pytorch中不需要指定TIME_STEP
        # 这里作者好像搞错了INPUT_SIZE和TIME_STEP的顺序，先姑且用着TIME_STEP
        self.bigru = nn.GRU(TIME_STEPS, 300, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(args.dropout)
        # 双向GRU的输出是2倍hidden_size
        self.dense = nn.Linear(300*2, 300)

    def forward(self, x):
        # keras中默认return_sequences=False，即只返回最后一个时间步的隐藏状态
        # 下面是在pytorch中模拟这一过程
        self.bigru.flatten_parameters()

        gru_output, _ = self.bigru(x)
        gru_output = gru_output[:, -1, :]
        x = F.relu(gru_output)
        x = self.dropout(x)
        x = self.dense(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

    @property
    def output_features(self):
        return 300

def get_bigru(INPUT_SIZE, TIME_STEPS):
    return BiGRU(INPUT_SIZE, TIME_STEPS)

