import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cnn import get_cnn
from models.BiGRU import get_bigru
from models.bilstm import get_bilstm
from options import parse_args


args = parse_args()


class ClassiFilerNet(nn.Module):
    def __init__(self, INPUT_SIZE, TIME_STEPS):
        super().__init__()
        self.inter_outputs = None

        self.net1 = self.get_net(args.cbgru_net1, INPUT_SIZE, TIME_STEPS)
        self.net2 = self.get_net(args.cbgru_net2, INPUT_SIZE, TIME_STEPS)
        
        self.fc1 = nn.Linear(self.net1.output_features+self.net2.output_features, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, 2)

        self.fc2.register_forward_hook(self.get_intermediate_outputs)
        
    def forward(self, x1, x2):
        out1 = self.net1(x1)
        out2 = self.net2(x2)

        merged = torch.cat((out1, out2), dim=1)

        x = F.relu(self.fc1(merged))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def get_net(self, net_type:str, INPUT_SIZE, TIME_STEPS):
        res_net = None
        if net_type == 'cnn':
            res_net = get_cnn(INPUT_SIZE, TIME_STEPS)
        elif net_type == 'bilstm':
            res_net = get_bilstm(INPUT_SIZE, TIME_STEPS)
        elif net_type == 'bigru':
            res_net = get_bigru(INPUT_SIZE, TIME_STEPS)
        else:
            print(f"Wrong Net Type: {net_type}!")

        return res_net
    
    def get_intermediate_outputs(self, module, input, output):
        self.inter_outputs = output.detach()