import torch
import torch.nn as nn
import torch.nn.functional as F

class CGEVariant(nn.Module):
    def __init__(self):
        super().__init__()
        # graph training
        self.inter_outputs = None
        self.con1 = nn.Conv1d(1, 200, kernel_size=3, stride=1, padding='same')
        self.pool1 = nn.MaxPool1d(1, 1)
        # pattern training
        self.con2 = nn.Conv1d(3, 200, kernel_size=3, stride=1, padding='same')
        self.pool2 = nn.MaxPool1d(3, 3)
        # joint traning
        # self.fc1 = nn.Linear(200*256+200*1, 10)
        # self.fc2 = nn.Linear(10, 1)
        self.fc1 = nn.Linear(200*256+200*1, 300)
        self.fc2 = nn.Linear(300, 2)

        self.fc2.register_forward_hook(self.get_intermediate_outputs)


    def forward(self, graph_train, pattern_train):
        # graph_training
        graph_train = F.relu(self.con1(graph_train))
        graph_train = self.pool1(graph_train)

        # pattern_traning
        pattern_train = F.relu(self.con2(pattern_train))
        pattern_train = self.pool2(pattern_train)

        # flatten
        graph_train = torch.flatten(graph_train, 1)
        pattern_train = torch.flatten(pattern_train, 1)
        mergevec = torch.cat((graph_train, pattern_train), dim=1)

        # joint training
        joint_feature = F.relu(self.fc1(mergevec))
        predict = self.fc2(joint_feature)
        # joint_feature = F.relu(self.fc1(mergevec))
        # joint_feature = F.relu(self.fc2(joint_feature))
        # predict = self.fc3(joint_feature)
        
        return predict
    
    # Get h(x) from main model
    def get_intermediate_outputs(self, module, input, output):
        self.inter_outputs = output.detach()
