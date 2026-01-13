import numpy as np
import copy
import torch.nn as nn
from trainers.server import CLC_Server
from trainers.client import Fed_CLC_client
from models.ClassiFilerNet import ClassiFilerNet
from models.CGE_Variants import CGEVariant

class CLC:
    def __init__(self, args, input_size, time_step, datasets, tao=0.1):
        self.args = args
        self.tao = tao
        if args.model_type == "CBGRU":
            self.model = ClassiFilerNet(input_size, time_step)
        elif args.model_type == "CGE":
            self.model = CGEVariant()
        self.model = self.model.to(args.device)
        self.clients = []
        self.server = CLC_Server(args, self.model, args.device, criterion = nn.CrossEntropyLoss())
        for i in range(args.client_num):
            self.clients.append(
                Fed_CLC_client(
                    args,
                    nn.CrossEntropyLoss(),
                    copy.deepcopy(self.server.global_model
                    ),
                    datasets[i],
                    i,
                    self.tao
                )
            )
            
        self.warmup()

    def warmup(self):
        Keep_size = [0]*self.args.client_num
        
        self.server.initialize_epoch_updates(-1)
        for i in range(self.args.client_num):
            client = self.clients[i]
            client.model = copy.deepcopy(self.server.global_model)
            client.train()
            self.server.save_train_updates(
                copy.deepcopy(client.get_parameters()),
                client.result['sample'],
                client.result
            )
            print(f"client:{i}")
            client.print_loss()

        self.server.average_weights()

    def holdout_stage(self):
        Keep_size = [0]*self.args.client_num
        
        for epoch in range(self.args.first_epochs):
            
            self.server.initialize_epoch_updates(epoch)
            confs = []
            classnums = []
            for ix in range(self.args.client_num):
                conf, classnum = self.clients[ix].sendconf()
                confs.append(conf)
                classnums.append(classnum)
            self.server.receiveconf(confs, classnums)
            conf_score = self.server.conf_agg()
            for ix in range(self.args.client_num):
                client = self.clients[ix]
                client.model = copy.deepcopy(self.server.global_model)
                client.data_holdout(conf_score)
                client.train()
                self.server.save_train_updates(
                    copy.deepcopy(client.get_parameters()),
                    client.result['sample'],
                    client.result
                )
                print(f"client:{ix}")
                client.print_loss()
            
            self.server.average_weights()
            # test (Not complete)

    def correct_stage(self):
        Keep_size = [0] * self.args.client_num
        model_param = [[] for i in range(self.args.client_num)]
        correct_done = False

        for epoch in range(self.args.first_epochs, self.args.first_epochs+self.args.last_epochs):
            self.server.initialize_epoch_updates(epoch)
            if not correct_done:
                confs = []
                classnums = []
                for ix in range(self.args.client_num):
                    conf, classnum = self.clients[ix].sendconf()
                    confs.append(conf)
                    classnums.append(classnum)
                self.server.receiveconf(confs, classnums)
                conf_score = self.server.conf_agg()
                for ix in range(self.args.client_num):
                    self.clients[ix].data_holdout(conf_score)
                    self.clients[ix].data_correct()
                correct_done=True
            
            for ix in range(self.args.client_num):
                client = self.clients[ix]
                client.model = copy.deepcopy(self.server.global_model)
                client.train()
                self.server.save_train_updates(
                    copy.deepcopy(client.get_parameters()),
                    client.result['sample'],
                    client.result
                )
                print(f"client:{ix}")
                client.print_loss()
            
            self.server.average_weights()
            # test (Not complete)
