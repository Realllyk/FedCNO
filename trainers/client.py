import gc
import torch
import numpy as np
import copy
import time
import os
import sys
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.utils.tensorboard import SummaryWriter
from data_processing.preprocessing import vec2one, reduced_name_labels, read_pretrain_feature, relabel_with_pretrained_knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


class Fed_Avg_client(object):
    def __init__(
      self,
      args,
      criterion,
      model,
      dataset,
      client_id=0,
      run_timestamp=None
    ):
        self.args = args
        self.criterion = criterion
        self.model = model
        self.dataset = dataset
        self.device = args.device
        self.client_id = client_id

        # Initialize TensorBoard and log file
        lab_name = getattr(self.args, 'lab_name', 'default')
        if run_timestamp is None:
            import time
            run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        model_type = getattr(self.args, 'model_type', 'unknown_model')
        noise_type = getattr(self.args, 'noise_type', 'unknown_noise')
        noise_rate = getattr(self.args, 'noise_rate', 0.0)
        vul = getattr(self.args, 'vul', 'unknown_vul')
        
        # runs/lab_name/model_type/noise_type/noise_rate/vul/timestamp/client_id
        log_dir = f"./runs/{lab_name}/{model_type}/{noise_type}/{noise_rate}/{vul}/{run_timestamp}/client_{self.client_id}"
        self.tb_writer = SummaryWriter(log_dir=log_dir)
        
        os.makedirs(log_dir, exist_ok=True)
        self.log_file_path = os.path.join(log_dir, "loss_log.txt")
        # Check if file exists to append or write header
        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, "w") as f:
                f.write("Global_Step,Epoch,Batch_Loss\n")
            
        self.tb_global_step = 0

    def get_parameters(self):
        return self.model.state_dict()
    
    def print_loss(self):
        print(f"loss is {self.result['loss']}")

    def cross_validation(self, dl_1, dl_2):
        model_1 = copy.deepcopy(self.model)
        model_2 = copy.deepcopy(self.model)
        opt_1 = torch.optim.Adam(model_1.parameters(), lr=self.args.inner_lr)
        opt_2 = torch.optim.Adam(model_2.parameters(), lr=self.args.inner_lr)

        model_1.train()
        for e in range(20):
            for x1, x2, y in dl_1:
                opt_1.zero_grad()
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                pred = model_1(x1, x2)
                y = y.long().flatten()
                loss = self.criterion(pred, y)
                loss.backward()
                opt_1.step()
        
        model_2.train()
        for e in range(20):
            for x1, x2, y in dl_2:
                opt_2.zero_grad()
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                pred = model_2(x1, x2)
                y = y.long().flatten()
                loss = self.criterion(pred, y)
                loss.backward()
                opt_2.step()
        
        model_1.eval()
        model_2.eval()
        cv_x1 = list()
        cv_x2 = list()
        cv_y = list()
        with torch.no_grad():
            for x1, x2, y in dl_2:
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                outputs = model_1(x1, x2)
                outputs = F.softmax(outputs, dim=-1)
                preds = torch.argmax(outputs, dim=-1).long()
                y = y.long().flatten()
                indices = torch.nonzero(torch.eq(preds, y), as_tuple=False).squeeze(dim=1)
                if indices.numel() != 0:
                    cv_x1.append(x1[indices])
                    cv_x2.append(x2[indices])
                    cv_y.append(y[indices])

            for x1, x2, y in dl_1:
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                outputs = model_2(x1, x2)
                outputs = F.softmax(outputs, dim=-1)
                preds = torch.argmax(outputs, dim=-1).long()
                y = y.long().flatten()
                indices = torch.nonzero(torch.eq(preds, y), as_tuple=False).squeeze(dim=1)
                if indices.numel() != 0:
                    cv_x1.append(x1[indices])
                    cv_x2.append(x2[indices])
                    cv_y.append(y[indices])
        
        pure_x1 = torch.cat(cv_x1, dim=0)
        pure_x2 = torch.cat(cv_x2, dim=0)
        pure_y = torch.cat(cv_y, dim=0)
        pure_ds = TensorDataset(pure_x1, pure_x2, pure_y)
        pure_dl = DataLoader(dataset=pure_ds, batch_size=self.args.batch, shuffle=True)
        return pure_dl
           
    def train(self):
        dataloader = DataLoader(self.dataset, batch_size=self.args.batch, shuffle=True, pin_memory=True)
        if self.args.model_type == "CBGRU":
            lr = self.args.cbgru_local_lr
        elif self.args.model_type == "CGE":
            lr = self.args.cge_local_lr
        # Add weight decay for regularization
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.args.weight_decay)

        self.result = dict()
        device = self.args.device
        self.result['sample'] = len(self.dataset)

        self.model.train()
        for epoch in range(self.args.cbgru_local_epoch):
            self.result['loss'] = 0
            for x1, x2, y in dataloader:
                optimizer.zero_grad()
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                outputs = self.model(x1, x2)
                y = y.flatten().long()
                loss = self.criterion(outputs, y)
                self.result['loss'] = self.result['loss'] + loss.item()
                loss.backward()
                # Gradient Clipping:
                # - reentrancy: 使用极其严格的裁剪 (1.0) 以强力防止震荡和 NaN
                # - timestamp: 使用极其严格的裁剪 (1.0) 以强力防止震荡
                clip_value = 1.0 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_value)
                optimizer.step()

                # del x1, x2, y, outputs, loss
                # torch.cuda.empty_cache()
                # gc.collect()
            
            # Record Average Loss
            avg_loss = self.result['loss'] / len(dataloader)
            self.tb_writer.add_scalar("loss/train", avg_loss, self.tb_global_step)
            
            # Log loss to text file
            with open(self.log_file_path, "a") as f:
                f.write(f"{self.tb_global_step},{epoch},{avg_loss}\n")
                
            self.tb_global_step += 1


class Fed_PLE_client(object):
    def __init__(
        self,
        args,
        criterion,
        device,
        inner_model,
        outer_model,
        noise_dataloader,
        pure_dataloader,
        valid_dataloader,
    ):
        self.args = args
        self.criterion = criterion
        self.device = device
        self.inner_model = inner_model
        self.outer_model = outer_model
        self.noise_dataloader = noise_dataloader
        self.pure_dataloader = pure_dataloader
        self.valid_dataloader = valid_dataloader

    def get_inner_parameters(self):
        return self.inner_model.state_dict()
    
    def get_outer_parameters(self):
        return self.outer_model.state_dict()
    
    def get_all_parameters(self):
        return self.get_inner_parameters(), self.get_outer_parameters()
    
    def print_loss(self):
        print(f"outer_loss is {self.result['outer_loss']}")

    def meta_train(self):
        torch.autograd.set_detect_anomaly(True)

        inner_model_copy = copy.deepcopy(self.inner_model)
        # outer_optimizer = torch.optim.SGD(self.outer_model.parameters(), lr=self.args.outer_lr)
        # inner_optimizer = torch.optim.SGD(self.inner_model.parameters(), lr=self.args.cbgru_local_lr)
        # inner_copy_opt = torch.optim.SGD(inner_model_copy.parameters(), lr=self.args.cbgru_local_lr)
        outer_optimizer = torch.optim.Adam(self.outer_model.parameters(), lr=self.args.outer_lr)
        inner_optimizer = torch.optim.Adam(self.inner_model.parameters(), lr=self.args.cbgru_local_lr)
        inner_copy_opt = torch.optim.Adam(inner_model_copy.parameters(), lr=self.args.cbgru_local_lr)
        
        self.result = dict()
        self.result['sample'] = len(self.noise_dataloader)
        for epoch in range(self.args.local_epoch):
            outer_loss_total = torch.tensor(0., device=self.device)

            for e in range(1):
                self.result['outer_loss'] = torch.tensor(0., device=self.device)
                for (x1, x2, noise_labels, global_labels), (x1_pure, x2_pure, pure_labels) in zip(self.noise_dataloader, self.pure_dataloader):
                    outer_optimizer.zero_grad()

                    x1, x2 = x1.to(self.device), x2.to(self.device)
                    noise_labels, global_labels = noise_labels.to(self.device), global_labels.to(self.device)
                    x1_pure, x2_pure, pure_labels = x1_pure.to(self.device), x2_pure.to(self.device), pure_labels.to(self.device)

                    inner_model_copy.train()
                    self.outer_model.eval()
                    inner_optimizer.zero_grad()
                    predictions = inner_model_copy(x1, x2)
                    predictions = F.softmax(predictions, dim=-1)

                    # 使用钩子函数获取的中间输出
                    h_x = inner_model_copy.inter_outputs
                    h_x.requires_grad = True
                    gl_one_hot = F.one_hot(global_labels.long().flatten(),num_classes=2)
                    gl_one_hot = gl_one_hot.unsqueeze(1)
                    nl_one_hot = F.one_hot(noise_labels.long().flatten(),num_classes=2)
                    nl_one_hot = nl_one_hot.unsqueeze(1)
                    cat_labels = torch.cat((gl_one_hot, nl_one_hot), dim=1)
                    cat_labels = cat_labels.float()
                    cat_labels.requires_grad = True
                    outer_outputs = self.outer_model(h_x, cat_labels)
                    outer_outputs = torch.squeeze(outer_outputs, dim=1)
                    outer_outputs = torch.softmax(outer_outputs, dim=-1)

                    # 内循环
                    inner_loss = nn.functional.kl_div(predictions, outer_outputs, reduction='batchmean')
                    inner_loss.backward()
                    inner_copy_opt.step()
                
                    # inner_model_copy.eval()
                    inner_model_copy.train()
                    self.outer_model.train()
                    updated_predictions = inner_model_copy(x1_pure, x2_pure)
                    pure_labels = pure_labels.long().flatten()
                    outer_loss = self.criterion(updated_predictions, pure_labels)
                    self.result['outer_loss'] = self.result['outer_loss'] + outer_loss.item()
                    
                    outer_loss.backward()
                    outer_optimizer.step()
                    
                    # del x1, x2, noise_labels, global_labels
                    # del h_x, gl_one_hot, nl_one_hot, cat_labels
                    # del outer_outputs, inner_loss, outer_loss
                    # torch.cuda.empty_cache()
                    # gc.collect()

            for e in range(1):
                for x1, x2, noise_labels, global_labels in self.noise_dataloader:
                    outer_optimizer.zero_grad()
                    x1, x2 =x1.to(self.device), x2.to(self.device)
                    noise_labels, global_labels = noise_labels.to(self.device), global_labels.to(self.device)

                    self.inner_model.train()
                    self.outer_model.eval()
                    inner_optimizer.zero_grad()
                    predictions = self.inner_model(x1, x2)
                    predictions = F.log_softmax(predictions, dim=-1)

                    h_x = self.inner_model.inter_outputs
                    h_x.requires_grad = True
                    gl_one_hot = F.one_hot(global_labels.long().flatten(),num_classes=2)
                    gl_one_hot = gl_one_hot.unsqueeze(1)
                    nl_one_hot = F.one_hot(noise_labels.long().flatten(),num_classes=2)
                    nl_one_hot = nl_one_hot.unsqueeze(1)
                    cat_labels = torch.cat((gl_one_hot, nl_one_hot), dim=1)
                    cat_labels = cat_labels.float()
                    cat_labels.requires_grad = True
                    outer_outputs = self.outer_model(h_x, cat_labels)
                    outer_outputs = torch.squeeze(outer_outputs, dim=1)
                    outer_outputs = torch.softmax(outer_outputs, dim=-1)

                    inner_loss = nn.functional.kl_div(predictions, outer_outputs, reduction='batchmean')
                    inner_loss.backward()
                    inner_optimizer.step()

                    # del x1, x2, noise_labels, global_labels
                    # del h_x, gl_one_hot, nl_one_hot, cat_labels
                    # del outer_outputs, inner_loss
                    # torch.cuda.empty_cache()
                    # gc.collect()

    def validation(self):
        # validation_dl = gen_cbgru_valid_dl(self.args.vul)
        validation_dl = gen_valid_dl(self.args.model_type, self.args.vul, data_dir=self.args.data_dir)
        self.inner_model.eval()
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for x1, x2, y in self.valid_dataloader:
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                softmax = nn.Softmax(dim=1)
                outputs = self.inner_model(x1, x2)
                pred = torch.argmax(softmax(outputs), dim=-1)
                all_predictions.extend(pred.flatten().tolist())
                all_targets.extend(y.flatten().tolist())

                # del x1, x2, y
                # torch.cuda.empty_cache()
                # gc.collect()

            tn, fp, fn, tp = confusion_matrix(all_targets, all_predictions).ravel()
            self.result['Recall(TPR)'] = tp / (tp + fn)
            self.result['Precision'] = tp / (tp + fp)
            self.result['F1 score'] = (2 * self.result['Precision'] * self.result['Recall(TPR)']) / (self.result['Precision'] + self.result['Recall(TPR)'])

    def cross_validation(self):
        model_1 = copy.deepcopy(self.model)
        model_2 = copy.deepcopy(self.model)
        

class Fed_ARFL_client(object):
    def __init__(
        self,
        args,
        criterion,
        model,
        dataset,
        weight,
    ):
        self.args = args
        self.device = args.device
        self.criterion = criterion
        self.model = model
        self.dataset = dataset
        self.weight = weight
        self.num_train_samples = len(dataset)
    
    def train(self):
        if self.args.model_type == "CBGRU":
            lr = self.args.cbgru_local_lr
        elif self.args.model_type == "CGE":
            lr = self.args.cge_local_lr
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.result = dict()
        device = self.device
        dataloader = DataLoader(self.dataset, batch_size=self.args.batch, shuffle=True, pin_memory=True)

        for epoch in range(self.args.local_epoch):
            self.model.train()
            self.result['loss'] = 0
            for x1, x2 ,y in dataloader:
                optimizer.zero_grad()
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                outputs = self.model(x1, x2)
                y = y.flatten().long()

                loss = self.criterion(outputs, y)

                self.result['loss'] = self.result['loss'] + loss.item()
                loss.backward()
                optimizer.step()
                
                # del x1, x2, y, outputs, loss
                # torch.cuda.empty_cache()
                # gc.collect()
    
    def test(self):
        device = self.device
        
        with torch.no_grad():
            # self.result['test_loss'] = 0
            self.test_loss = 0
            self.model.eval()
            dataloader = DataLoader(self.dataset, batch_size=self.args.batch, shuffle=True, pin_memory=True)
            for x1, x2, y in dataloader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                outputs = self.model(x1, x2)

                y = y.flatten().long()
                loss = self.criterion(outputs, y)
                # self.result['test_loss'] = self.result['test_loss'] + loss.item()
                self.test_loss += loss.item()
                # del x1, x2, y, outputs, loss
                # torch.cuda.empty_cache()
                # gc.collect()
        self.test_loss = self.test_loss / len(dataloader)

    def get_model_parameters(self):
        return self.model.state_dict()

    def get_test_loss(self):
        # return self.result['test_loss']
        return self.test_loss
    
    def set_weight(self, weight):
        self.weight = weight


class Fed_Corr_client(Fed_Avg_client):
    def __init__(
        self,
        args,
        criterion,
        model,
        dataset,
        client_id=None,
        run_timestamp=None,
        global_round=0
    ):
        super().__init__(
            args,
            criterion,
            model,
            dataset
        )
        self.dataloader = DataLoader(self.dataset, batch_size=self.args.batch, shuffle=True, pin_memory=True)
        self.client_id = client_id
        self.global_round = global_round
        
        # Setup logging
        if self.client_id is not None:
            lab_name = getattr(self.args, 'lab_name', 'default')
            if run_timestamp is None:
                run_timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            model_type = getattr(self.args, 'model_type', 'unknown_model')
            noise_type = getattr(self.args, 'noise_type', 'unknown_noise')
            noise_rate = getattr(self.args, 'noise_rate', 0.0)
            vul = getattr(self.args, 'vul', 'unknown_vul')
            
            log_dir = f"./runs/{lab_name}/{model_type}/{noise_type}/{noise_rate}/{vul}/{run_timestamp}/client_{self.client_id}"
            os.makedirs(log_dir, exist_ok=True)
            self.log_file_path = os.path.join(log_dir, "loss_log.txt")
            
            self.tb_writer = SummaryWriter(log_dir=log_dir)
            
            # Initialize global step based on global round and local epochs
            # Assuming cbgru_local_epoch is used for training
            local_epochs = getattr(self.args, 'cbgru_local_epoch', 1)
            self.tb_global_step = self.global_round * local_epochs
            
            # Create file with header if it doesn't exist
            if not os.path.exists(self.log_file_path):
                with open(self.log_file_path, "w") as f:
                    f.write("Epoch,Batch_Loss\n")
        

    def train(self):
        if self.args.model_type == "CBGRU":
            lr = self.args.cbgru_local_lr
        elif self.args.model_type == "CGE":
            lr = self.args.cge_local_lr
        # Add weight decay for regularization
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.args.weight_decay)

        self.result = dict()
        device = self.args.device
        self.result['sample'] = len(self.dataloader)

        self.model.train()
        for epoch in range(self.args.cbgru_local_epoch):
            self.result['loss'] = 0
            for x1, x2, y in self.dataloader:
                optimizer.zero_grad()
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                outputs = self.model(x1, x2)
                y = y.flatten().long()
                loss = self.criterion(outputs, y)
                loss = loss.mean()
                self.result['loss'] = self.result['loss'] + loss.item()
                loss.backward()
                optimizer.step()

                # del x1, x2, y, outputs, loss
                # torch.cuda.empty_cache()
                # gc.collect()

            # Log loss
            if hasattr(self, 'client_id') and self.client_id is not None:
                self.tb_writer.add_scalar("loss/train", self.result['loss'], self.tb_global_step)
                with open(self.log_file_path, "a") as f:
                    f.write(f"{self.tb_global_step},{epoch},{self.result['loss']}\n")
                self.tb_global_step += 1

    
    def get_output(self):
        self.model.eval()
        with torch.no_grad():
            for i, (x1, x2, y) in enumerate(self.dataloader):
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                y = y.long()

                outputs = self.model(x1, x2)
                outputs = F.softmax(outputs, dim=1)

                loss = self.criterion(outputs, y)
                if i == 0:
                    outputs_whole = np.array(outputs.cpu())
                    loss_whole = np.array(loss.cpu())
                else:
                    output_whole = np.concatenate((output_whole, outputs.cpu()), axis=0)
                    loss_whole = np.concatenate((loss_whole, loss.cpu()), axis=0)

        return output_whole, loss_whole
        

class Fed_LGV_client(object):
    def __init__(
        self,
        args,
        criterion,
        model,
        dataset,
        client_id,
        global_weight,
        run_timestamp=None
    ):
        self.args = args
        self.criterion = criterion
        self.model = model
        self.dataset = dataset
        self.client_id = client_id
        self.device = args.device
        self.global_weight = global_weight
        
        # FedCNO: 初始化 fixed_global_model
        # 注意：这里的 model 是客户端初始化时传入的初始模型
        # 在 train() 开始前，我们应该手动更新它，以确保它始终是本轮最新的全局模型
        self.fixed_global_model = copy.deepcopy(model)
        self.fixed_global_model.eval() 
        for param in self.fixed_global_model.parameters():
            param.requires_grad = False
             
        # Use a safe log_dir even if args lacks lab_name
        lab_name = getattr(self.args, 'lab_name', 'default')
        # Use formatted timestamp to separate runs while keeping history
        if run_timestamp is None:
            import time
            run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # New log path format: runs/lab_name/model_type/noise_type/noise_rate/vul/timestamp/client_id
        # args.model_type, args.noise_type, args.noise_rate should be available in args
        # Ensure compatibility if some args are missing
        model_type = getattr(self.args, 'model_type', 'unknown_model')
        noise_type = getattr(self.args, 'noise_type', 'unknown_noise')
        noise_rate = getattr(self.args, 'noise_rate', 0.0)
        vul = getattr(self.args, 'vul', 'unknown_vul')
        
        self.tb_writer = SummaryWriter(log_dir=f"./runs/{lab_name}/{model_type}/{noise_type}/{noise_rate}/{vul}/{run_timestamp}/client_{self.client_id}")
        
        # Also create a text log file in the same directory
        log_dir = f"./runs/{lab_name}/{model_type}/{noise_type}/{noise_rate}/{vul}/{run_timestamp}/client_{self.client_id}"
        os.makedirs(log_dir, exist_ok=True)
        self.log_file_path = os.path.join(log_dir, "loss_log.txt")
        # Initialize log file with header
        with open(self.log_file_path, "w") as f:
            f.write("Global_Step,Epoch,Batch_Loss\n")
            
        self.tb_global_step = 0


    def get_local_knn_labels(self, vul, noise_type, noise_rate):
        # -------------------------------------------------------------------------
        # 初始化：生成本地视图 (Local View)
        # -------------------------------------------------------------------------
        # 该方法在训练开始前调用一次，利用本地的静态预训练特征（如 Word2Vec/FastText）
        # 来建立初始的标签概率分布和一致性基准。
        
        pre_feature_dir = os.path.join(self.args.data_dir, f"pretrain_feature/{vul}")
        
        # 根据 model_type 选择路径，确保与数据集加载路径一致
        if self.args.model_type == 'CBGRU':
            name_path = os.path.join(self.args.data_dir, f"graduate_client_split/cbgru/{vul}/client_{self.client_id}/contract_name_train.txt")
        elif self.args.model_type == 'CGE':
            name_path = os.path.join(self.args.data_dir, f"graduate_client_split/cge/{vul}/client_{self.client_id}/contract_name_train.txt")
        else:
            name_path = os.path.join(self.args.data_dir, f"graduate_client_split/{vul}/client_{self.client_id}/contract_name_train.txt")
            
        labels = self.dataset.labels

        # 读取数据和预训练特征
        reduced_names, reduced_labels = reduced_name_labels(name_path, labels)
        # pre_features = read_pretrain_feature(reduced_names, pre_feature_dir)
        
        # 使用 dataset 本身的特征 (x1, x2) 替代预训练特征
        print(f"Client {self.client_id}: Extracting features from dataset for KNN...")
        reduced_features = []
        name_set = set()
        
        # 遍历数据集提取去重后的特征
        # 注意：这里假设 dataset.names 的顺序与 reduced_name_labels 处理文件的顺序一致
        for i in range(len(self.dataset)):
            name = self.dataset.names[i]
            if name not in name_set:
                name_set.add(name)
                
                # 获取特征 (x1, x2)
                # dataset[i] 返回 (x1, x2, label, ratio) 或 (x1, x2, label)
                data_item = self.dataset[i]
                x1 = data_item[0]
                x2 = data_item[1]
                
                # Flatten 并拼接特征
                # x1: (1, 100, 300) -> Flatten
                # x2: (300,) -> Flatten
                f1 = x1.view(-1).numpy()
                f2 = x2.view(-1).numpy()
                feature = np.concatenate([f1, f2])
                
                reduced_features.append(feature)
        
        reduced_features = np.array(reduced_features)
        
        # 运行 KNN 获取初始概率和一致性
        # - prob_relabels: 本地特征视角下的标签概率。
        # - agreement_ratios: 本地特征视角下的样本一致性。
        relabels, prob_relabels, agreement_ratios, indices = relabel_with_pretrained_knn(reduced_labels, reduced_features, 2, 'uniform', self.args.num_neigh, 0.15)
        
        # 保存静态特征 KNN 的邻居索引，供后续 Global View 构建使用
        self.reduced_knn_indices = indices
        
        # Reduced probabilities, need to fullfill for the whole ds
        name_agr = dict()
        name_prob = dict()
        for i, name in enumerate(reduced_names):
            name_agr[name] = agreement_ratios[i]
            name_prob[name] = prob_relabels[i]

        full_agr = list()
        full_prob = list()
        with open(name_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                name = line.strip()
                full_agr.append(name_agr[name])
                full_prob.append(name_prob[name])
        
        # 保存本地概率分布，这部分在后续训练中保持静态，作为先验知识 (Prior Knowledge)
        full_prob = np.array(full_prob, dtype=np.float32)
        self.local_prob_labels = torch.tensor(full_prob, dtype=torch.float32).to(self.device)
        
        # 初始化数据集的一致性比率
        self.dataset.set_ag_rt(full_agr)

    # 使用全局模型直接生成概率标签
    def get_global_prob_labels(self, vul):
        dl = DataLoader(self.dataset, batch_size=self.args.batch, shuffle=False, pin_memory=True)
        gl_probs = []
        with torch.no_grad():
            self.model.eval()
            for x1, x2, y, _ in dl:
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                outputs = self.model(x1, x2)
                outputs = F.softmax(outputs, dim=1)
                gl_probs.append(outputs)
                
        all_preds = torch.cat(gl_probs, dim=0).to(self.device)
        self.global_prob_labels = torch.cat(gl_probs, dim=0)

    # 使用全局模型标签来进行投票
    def get_global_knn_labels(self, vul, noise_type, noise_rate):
        pre_feature_dir = os.path.join(self.args.data_dir, f"pretrain_feature/{vul}")
        
        # 根据 model_type 选择路径，确保与数据集加载路径一致
        if self.args.model_type == 'CBGRU':
            name_path = os.path.join(self.args.data_dir, f"graduate_client_split/cbgru/{vul}/client_{self.client_id}/contract_name_train.txt")
        elif self.args.model_type == 'CGE':
            name_path = os.path.join(self.args.data_dir, f"graduate_client_split/cge/{vul}/client_{self.client_id}/contract_name_train.txt")
        else:
            name_path = os.path.join(self.args.data_dir, f"graduate_client_split/{vul}/client_{self.client_id}/contract_name_train.txt")

        # Before local training, new local model is global model
        dl = DataLoader(self.dataset, batch_size=self.args.batch, shuffle=False, pin_memory=True)
        gl_labels = []
        with torch.no_grad():
            self.model.eval()
            for x1, x2, y, _ in dl:
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                outputs = self.model(x1, x2)
                outputs = F.softmax(outputs, dim=1)
                pred = torch.argmax(outputs, dim=-1)
                gl_labels.extend(pred.flatten().tolist())
                
        # print(len(gl_labels), gl_labels)
        reduced_names, reduced_labels = reduced_name_labels(name_path, gl_labels)
        pre_features = read_pretrain_feature(reduced_names, pre_feature_dir)
        relabels, prob_relabels, _, _ = relabel_with_pretrained_knn(reduced_labels, pre_features, 2, 'uniform', self.args.num_neigh, 0.15)
        # print(relabels)

        # Reduced probabilities, need to fullfill for the whole ds
        name_prob = dict()
        for i, name in enumerate(reduced_names):
            name_prob[name] = prob_relabels[i]
        
        full_prob = list()
        with open(name_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                name = line.strip()
                full_prob.append(name_prob[name])

        full_prob = np.array(full_prob, dtype=np.float32)
        self.global_prob_labels = torch.tensor(full_prob, dtype=torch.float32).to(self.args.device)

    def gen_reduced_ds(self):
        names = self.dataset.names
        labels = self.dataset.labels

        name_set = set()
        reduced_names = list()
        reduced_labels = list()
        for i, name in enumerate(names):
            if name not in name_set:
                name_set.add(name)
                reduced_names.append(name)
                reduced_labels.append(labels[i])

        reduced_ds = copy.deepcopy(self.dataset)
        reduced_ds.names = reduced_names
        reduced_ds.labels = reduced_labels
        reduced_ds.agreement_ratio = [1.0 for _ in range(len(reduced_ds.labels))]
        self.reduced_ds = reduced_ds
        
    # 使用全局模型生成的特征来进行knn
    def get_global_feature_knn_labels(self):
        dl = DataLoader(self.reduced_ds, batch_size=self.args.batch, shuffle=False, pin_memory=True)
        outputs_list = []
        with torch.no_grad():
            self.model.eval()
            for x1, x2, y, _ in dl:
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                outputs = self.model(x1, x2)
                # outputs_list.append(outputs)
                outputs_list.append(self.model.inter_outputs)
                
        conc_outputs = torch.cat(outputs_list, dim=0)
        conc_outputs = conc_outputs.cpu().numpy()
        relabels, prob_relabels, _, _ = relabel_with_pretrained_knn(self.reduced_ds.labels, conc_outputs, 2, 'uniform', self.args.num_neigh, 0.15)

        name_prob = dict()
        for i, name in enumerate(self.reduced_ds.names):
            name_prob[name] = prob_relabels[i]
        
        full_prob = list()
        for name in self.dataset.names:
            full_prob.append(name_prob[name])

        full_prob = np.array(full_prob, dtype=np.float32)
        self.global_prob_labels = torch.tensor(full_prob, dtype=torch.float32).to(self.args.device)

    # 使用全局模型，并使用全局模型生成的特征合和标签一起进行knn
    def get_global_feature_global_knn_labels(self):
        # Update reduced dataset with latest labels before KNN
        self.gen_reduced_ds()
        
        # ---------------------------------------------------------------------
        # 修改：Global View 构建逻辑更新
        # 1. 邻居选择：复用 Local View 中的静态特征 KNN 邻居 (self.reduced_knn_indices)
        # 2. 证据生成：使用当前全局模型对邻居样本输出的 Softmax 概率进行平均
        # ---------------------------------------------------------------------

        if not hasattr(self, 'reduced_knn_indices'):
             # Fallback if not initialized (should not happen in correct flow)
             print("Warning: reduced_knn_indices not found, falling back to dynamic KNN (which is not implemented in this experimental version)")
             return 

        # 1. 对 reduced_ds 进行全量推理，获取每个样本的当前模型预测概率
        dl = DataLoader(self.reduced_ds, batch_size=self.args.batch, shuffle=False, pin_memory=True)
        all_probs_list = []
        
        with torch.no_grad():
            self.model.eval()
            for x1, x2, y, _ in dl:
                x1, x2 = x1.to(self.device), x2.to(self.device)
                outputs = self.model(x1, x2)
                probs = F.softmax(outputs, dim=1) # (B, C)
                all_probs_list.append(probs)
        
        # 拼接所有批次结果 -> (M, C)
        all_probs = torch.cat(all_probs_list, dim=0)
        
        # 2. 获取静态邻居索引
        # self.reduced_knn_indices 是一个 list of lists 或者 numpy array (M, K)
        # 确保它是 tensor 以便索引
        knn_indices = torch.tensor(self.reduced_knn_indices, dtype=torch.long, device=self.device)
        
        # 3. 查表获取邻居概率
        # neighbor_probs: (M, K, C)
        neighbor_probs = all_probs[knn_indices]
        
        # ---------------------------------------------------------------------
        # 修改：引入基于熵的动态加权 (Entropy-based Dynamic Weighting)
        # ---------------------------------------------------------------------
        
        # 计算每个邻居的熵 (Entropy): H(p) = -sum(p * log(p))
        # (M, K)
        neighbor_entropy = -torch.sum(neighbor_probs * torch.log(neighbor_probs + 1e-8), dim=2)
        
        # 计算权重：熵越小 (越确信)，权重越大
        # 使用 Softmax(-Entropy) 是一种平滑且鲁棒的加权方式 (相当于 Temperature=1.0)
        # (M, K)
        neighbor_weights = F.softmax(-neighbor_entropy, dim=1)
        
        # 扩展权重维度以匹配概率矩阵: (M, K, 1)
        neighbor_weights = neighbor_weights.unsqueeze(2)
        
        # 加权平均得到 Global View
        # (M, C)
        global_view_probs = torch.sum(neighbor_probs * neighbor_weights, dim=1)
        
        # ---------------------------------------------------------------------
        
        # 转为 numpy 以便后续映射
        prob_relabels = global_view_probs.cpu().numpy()

        name_prob = dict()
        # name_agr = dict()
        for i, name in enumerate(self.reduced_ds.names):
            name_prob[name] = prob_relabels[i]
            # name_agr[name] = agreement_ratios[i]

        full_prob = list()
        # full_agr = list()
        for name in self.dataset.names:
            full_prob.append(name_prob[name])
            # full_agr.append(name_agr[name])

        full_prob = np.array(full_prob, dtype=np.float32)
        self.global_prob_labels = torch.tensor(full_prob, dtype=torch.float32).to(self.args.device)
        
        # Update agreement ratio for the dataset
        # self.dataset.set_ag_rt(full_agr)

    def warmup_train(self):
        dl = DataLoader(self.dataset, batch_size=self.args.batch, shuffle=True, pin_memory=True)
        if self.args.model_type == "CBGRU":
            lr = self.args.cbgru_local_lr
        elif self.args.model_type == "CGE":
            lr = self.args.cge_local_lr
        # Add weight decay for regularization
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.args.weight_decay)

        self.result = dict()
        device = self.args.device
        self.result['sample'] = len(self.dataset)

        self.model.train()
        for epoch in range(self.args.cbgru_local_epoch):
            self.result['loss'] = 0
            for x1, x2, y, agr in dl:
                optimizer.zero_grad()
                x1, x2, y= x1.to(device), x2.to(device), y.to(device)
                outputs = self.model(x1, x2)
                y = y.flatten().long()
                loss = self.criterion(outputs, y)
                # loss = loss.mean()
                self.result['loss'] = self.result['loss'] + loss.item()
                loss.backward()
                # Gradient Clipping:
                # - reentrancy: 使用极其严格的裁剪 (1.0) 以强力防止震荡和 NaN
                # - timestamp: 使用极其严格的裁剪 (1.0) 以强力防止震荡
                clip_value = 1.0 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_value)
                optimizer.step()

                # del x1, x2, y, outputs, loss
                # torch.cuda.empty_cache()
                # gc.collect()
                
    def train(self):
        # ---------------- FedCNO: 动态 alpha 计算与数据集标签更新 ----------------
        
        # 关键修正：在每轮本地训练开始前，必须更新 fixed_global_model 为当前最新的本地模型
        # (因为在 Fed_LGV.py 中，client.model 已经在每轮开始时被重置为 server.global_model)
        # 这样 fixed_global_model 才能代表本轮的"全局视图"
        self.fixed_global_model.load_state_dict(self.model.state_dict())
        self.fixed_global_model.eval()
        
        # 在每轮训练开始前，遍历整个数据集，计算每个样本的不确定性和动态融合权重 alpha
        # 并据此更新 dataset.labels (伪标签)
        
        # 1. 计算全局不确定性与动态 alpha
        # 为了计算整个数据集的不确定性，我们需要遍历一遍数据
        # 使用 self.fixed_global_model (本轮固定的全局视图)
        
        # 创建一个不打乱的 DataLoader 以便按顺序获取不确定性
        eval_dl = DataLoader(self.dataset, batch_size=self.args.batch, shuffle=False, pin_memory=True)
        all_uncertainties = []
        
        with torch.no_grad():
            self.fixed_global_model.eval()
            for x1, x2, _, _ in eval_dl:
                x1, x2 = x1.to(self.device), x2.to(self.device)
                global_logits = self.fixed_global_model(x1, x2)
                global_probs = F.softmax(global_logits, dim=1) # (B, C)
                
                # 计算 Entropy: u = - sum(p * log(p)) / log(C)
                num_classes = global_probs.shape[1]
                entropy = -torch.sum(global_probs * torch.log(global_probs + 1e-8), dim=1)
                max_entropy = np.log(num_classes)
                uncertainty = entropy / max_entropy # (B,)
                all_uncertainties.append(uncertainty.cpu())
                
        # 拼接所有不确定性分数
        all_uncertainties = torch.cat(all_uncertainties, dim=0) # (N,)
        
        # 2. 计算动态权重 alpha (公式 3 & 5)
        # 逻辑修正 (Critical Fix for Systemic Noise):
        # 原逻辑: alpha = 1.0 - uncertainty 
        #   -> 意味着模型越确定 (Low Uncertainty)，越信任 Global View。
        #   -> 在 Systemic Noise 下，模型可能对错误标签非常自信 (Overconfidence)。
        #   -> 导致强行融合错误的 Global View，破坏 WarmUp 成果。
        #
        # 新逻辑: alpha = uncertainty
        #   -> 意味着模型越迷茫 (High Uncertainty)，才去参考 Global View。
        #   -> 如果模型很确定 (Low Uncertainty)，则坚持 Local View (WarmUp 结果)，不受 Global 噪声干扰。
        # 优化: 使用平方衰减 (u^2) 而非线性 (u)。
        #   -> 进一步降低在"半懂不懂" (u~0.5) 时的干扰，只在极度迷茫 (u>0.8) 时才显著引入 Global View。
        alpha_raw = all_uncertainties ** 2 # 非线性衰减
        
        # 截断约束 alpha \in [alpha_min, alpha_max]
        # 修改原因：实验发现模型在全阴/全阳之间剧烈震荡，说明全局视图可能引入了过大的噪声或不稳定性。
        # 降低 alpha_max 以限制全局视图的最大权重，更多地依赖本地的一致性（特别是修复了本地KNN特征后）。
        # 进一步修改：将 alpha_max 与 server 端的 self.global_weight 绑定，
        # 使 autotune_gr 的动态调整能够真正影响客户端的融合策略。
        alpha_min = self.args.alpha_min
        # alpha_max = 0.5 
        alpha_max = self.global_weight # 动态绑定到 global_weight
        
        # 增加保护：防止 global_weight 越界或过大
        alpha_max = max(0.0, min(alpha_max, self.args.alpha_max))
        
        alpha = torch.clamp(alpha_raw, alpha_min, alpha_max)
        
        # 将 alpha 扩展维度以匹配 prob_labels: (N, 1)
        alpha = alpha.unsqueeze(1).to(self.device)
        
        # 3. 融合生成伪标签 (公式 6)
        # p_tilde = alpha * p_global + (1-alpha) * p_local
        # 注意：这里的 p_global 和 p_local 分别是 self.global_prob_labels 和 self.local_prob_labels
        # 它们已经在之前通过 get_global/local_knn_labels 计算并存储好了
        
        # 确保 global/local_prob_labels 在设备上
        if self.global_prob_labels.device != self.device:
            self.global_prob_labels = self.global_prob_labels.to(self.device)
        if self.local_prob_labels.device != self.device:
            self.local_prob_labels = self.local_prob_labels.to(self.device)
            
        with torch.no_grad():
            # 使用动态 alpha 进行融合
            prob_labels = alpha * self.global_prob_labels + (1 - alpha) * self.local_prob_labels
            prob_labels = F.softmax(prob_labels, dim=1)
            labels = torch.argmax(prob_labels, dim=-1) # (N,)
            
        # 4. 更新数据集标签
        self.dataset.labels = labels.detach().cpu().numpy()
        
        # ---------------------------------------------------------------------
        
        dl = DataLoader(self.dataset, batch_size=self.args.batch, shuffle=True, pin_memory=True)
        if self.args.model_type == "CBGRU":
            lr = self.args.cbgru_local_lr
        elif self.args.model_type == "CGE":
            lr = self.args.cge_local_lr
        # Add weight decay for regularization
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.args.weight_decay)

        self.result = dict()
        device = self.args.device
        self.result['sample'] = len(self.dataset)

        self.model.train()
        for epoch in range(self.args.cbgru_local_epoch):
            self.result['loss'] = 0
            for x1, x2, y, agr in dl:
                optimizer.zero_grad()
                x1, x2, y, agr = x1.to(device), x2.to(device), y.to(device), agr.to(device)
                outputs = self.model(x1, x2)
                y = y.flatten().long()
                loss = self.criterion(outputs, y)

                # compute weights
                _, predictions = torch.max(outputs, 1)
                correct_predictions = (predictions==y)
                weights = torch.ones_like(y, dtype=torch.float32)
                weights += agr * (~correct_predictions).float()
                weights -= 0.5 * agr * correct_predictions.float()
                if self.args.consistency_score == True:
                    weighted_losses = weights*loss
                else:
                    weighted_losses = loss
                loss = weighted_losses.mean()

                self.result['loss'] = self.result['loss'] + loss.item()
                loss.backward()
                # Gradient Clipping:
                # - reentrancy: 使用默认或较宽松的裁剪 (10)
                # - timestamp: 使用极其严格的裁剪 (1.0) 以强力防止震荡
                clip_value = 1.0 if getattr(self.args, 'vul', '') == 'timestamp' else 10
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_value)
                optimizer.step()

                # del x1, x2, y, outputs, loss
                # torch.cuda.empty_cache()
                # gc.collect()

            # Record Average Loss
            avg_loss = self.result['loss'] / len(dl)
            self.tb_writer.add_scalar("loss/train", avg_loss, self.tb_global_step)
            
            # Log loss to text file
            with open(self.log_file_path, "a") as f:
                f.write(f"{self.tb_global_step},{epoch},{avg_loss}\n")
                
            self.tb_global_step += 1
    
    def get_parameters(self):
        return self.model.state_dict()
    
    def print_loss(self):
        print(f"loss is {self.result['loss']}")


class Fed_LGV_GKNN_client(Fed_LGV_client):
    def __init__(self, args, criterion, model, dataset, client_id, global_weight, run_timestamp=None):
        super().__init__(args, criterion, model, dataset, client_id, global_weight, run_timestamp)

    def get_global_knn_labels(self, vul, noise_type, noise_rate):
        pass


class Fed_CRD_client(Fed_Avg_client):
    def __init__(self, args, criterion, model, dataset, client_id, run_timestamp=None):
        super().__init__(args, criterion, model, dataset, client_id, run_timestamp)
        self.reduced_knn_indices = None
        self.local_probs = None # To store local distribution
        self.pi_loc = None
        self.h_loc = None
        self.reduced_names = None
        self.reduced_labels = None

    def init_knn_neighborhood(self):
        """
        Build static KNN neighborhood using pre-trained features.
        This follows the logic in Fed_LGV_client to ensure consistency.
        """
        vul = getattr(self.args, 'vul', 'unknown_vul')
        
        # We need to extract features similar to Fed_LGV_client.get_local_knn_labels
        # Note: This logic assumes we can access dataset features directly or via file paths
        
        # 1. Extract reduced features (deduplicated by name)
        reduced_features = []
        name_set = set()
        
        # We also need reduced labels for local consistency calculation
        reduced_labels = []
        reduced_names = []
        
        # Iterate dataset to extract features
        # Assuming dataset structure is consistent with what Fed_LGV expects
        print(f"Client {self.client_id}: Initializing KNN neighborhood...")
        for i in range(len(self.dataset)):
            name = self.dataset.names[i]
            if name not in name_set:
                name_set.add(name)
                reduced_names.append(name)
                reduced_labels.append(self.dataset.labels[i])
                
                # Extract feature (x1, x2)
                # Handle different dataset return formats
                data_item = self.dataset[i]
                x1 = data_item[0]
                x2 = data_item[1]
                
                f1 = x1.view(-1).numpy()
                f2 = x2.view(-1).numpy()
                feature = np.concatenate([f1, f2])
                reduced_features.append(feature)
        
        reduced_features = np.array(reduced_features)
        
        # 2. Run KNN to get indices
        # We reuse relabel_with_pretrained_knn but we are interested in indices
        # Note: relabel_with_pretrained_knn returns (relabels, prob_relabels, agreement_ratios, indices)
        # We use a dummy K=2 here, but args.num_neigh is what matters for neighborhood size
        
        _, _, _, indices = relabel_with_pretrained_knn(
            reduced_labels, 
            reduced_features, 
            2, # num_classes (dummy)
            'uniform', 
            self.args.num_neigh, 
            0.15 # threshold (dummy)
        )
        
        self.reduced_knn_indices = indices
        self.reduced_names = reduced_names
        self.reduced_labels = np.array(reduced_labels)
        
        # 3. Pre-calculate Local Distribution (pi_loc) and Local Homogeneity (h_loc)
        # pi_loc: Distribution of labels in the neighborhood
        
        M = len(reduced_names)
        K = self.args.num_neigh
        num_classes = 2 # Assuming binary classification
        
        # Get neighbor labels: (M, K)
        neighbor_labels = self.reduced_labels[self.reduced_knn_indices]
        
        # Calculate distribution: (M, C)
        pi_loc = np.zeros((M, num_classes))
        for i in range(M):
            for k in range(K):
                label = int(neighbor_labels[i, k])
                if 0 <= label < num_classes:
                    pi_loc[i, label] += 1
        pi_loc = pi_loc / K
        
        self.pi_loc = torch.tensor(pi_loc, dtype=torch.float32, device=self.device)
        self.h_loc = torch.max(self.pi_loc, dim=1)[0] # (M,)
        print(f"Client {self.client_id}: KNN neighborhood initialized.")

    def get_consistency_stats(self, global_model):
        """
        Compute consistency statistics: pi_glob, h_glob, delta, q_k
        """
        if self.reduced_knn_indices is None:
            self.init_knn_neighborhood()
            
        # 1. Compute Global Distribution (pi_glob) using Global Model
        # We need to run inference on the reduced dataset
        
        # Create a mapping from name to index in original dataset
        name_to_idx = {name: i for i, name in enumerate(self.dataset.names)}
        indices = [name_to_idx[name] for name in self.reduced_names]
        
        # Create Subset
        reduced_ds = Subset(self.dataset, indices)
        dl = DataLoader(reduced_ds, batch_size=self.args.batch, shuffle=False, pin_memory=True)
        
        all_probs = []
        global_model.eval()
        with torch.no_grad():
            for batch in dl: 
                 # Unpack dynamically
                 if len(batch) == 3:
                     x1, x2, _ = batch
                 elif len(batch) >= 4:
                     x1, x2, _, _ = batch[0:4]
                     
                 x1, x2 = x1.to(self.device), x2.to(self.device)
                 outputs = global_model(x1, x2)
                 probs = F.softmax(outputs, dim=1)
                 all_probs.append(probs)
        
        # (M, C)
        all_probs = torch.cat(all_probs, dim=0)
        
        # Now compute pi_glob based on neighbors
        # neighbor_probs: (M, K, C)
        knn_indices = torch.tensor(self.reduced_knn_indices, dtype=torch.long, device=self.device)
        neighbor_probs = all_probs[knn_indices]
        
        # pi_glob = mean of neighbor probabilities
        # (M, C)
        pi_glob = torch.mean(neighbor_probs, dim=1)
        h_glob = torch.max(pi_glob, dim=1)[0] # (M,)
        
        # 2. Compute Discrepancy (delta)
        # JS Divergence between pi_loc and pi_glob
        # JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), M = 0.5*(P+Q)
        
        m = 0.5 * (self.pi_loc + pi_glob)
        kl_loc = F.kl_div(torch.log(self.pi_loc + 1e-8), m, reduction='none').sum(dim=1)
        kl_glob = F.kl_div(torch.log(pi_glob + 1e-8), m, reduction='none').sum(dim=1)
        js_div = 0.5 * (kl_loc + kl_glob)
        
        # delta = h_loc * h_glob * js_div
        delta = self.h_loc * h_glob * js_div # (M,)
        
        # 3. Client-level Ambiguity (d_tilde)
        # Average over all samples
        d_tilde = torch.mean(delta).item()
        
        # 4. Reliability (q_k)
        q_loc = torch.mean(self.h_loc).item()
        q_glob = torch.mean(h_glob).item()
        
        lambda_val = getattr(self.args, 'lambda_crd', 2.0) # Hyperparameter, default 2.0
        
        q_k = min(q_loc, q_glob) * np.exp(-lambda_val * d_tilde)
        
        return q_k, len(self.dataset)

    def train(self):
        """
        Custom train loop for Fed_CRD to handle dataset unpacking correctly.
        FedCRD datasets (reused from LGV) might return extra values (agr, index), 
        but we only need x1, x2, y for standard CrossEntropy training.
        """
        dataloader = DataLoader(self.dataset, batch_size=self.args.batch, shuffle=True, pin_memory=True)
        if self.args.model_type == "CBGRU":
            lr = self.args.cbgru_local_lr
        elif self.args.model_type == "CGE":
            lr = self.args.cge_local_lr
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.args.weight_decay)

        self.result = dict()
        device = self.args.device
        self.result['sample'] = len(self.dataset)

        self.model.train()
        for epoch in range(self.args.cbgru_local_epoch):
            self.result['loss'] = 0
            for batch in dataloader:
                # Dynamically unpack based on length
                if len(batch) == 3:
                    x1, x2, y = batch
                elif len(batch) >= 4:
                    x1, x2, y = batch[0], batch[1], batch[2]
                
                optimizer.zero_grad()
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                
                outputs = self.model(x1, x2)
                y = y.flatten().long()
                
                loss = self.criterion(outputs, y)
                self.result['loss'] = self.result['loss'] + loss.item()
                loss.backward()
                
                # Gradient Clipping
                clip_value = 1.0 if getattr(self.args, 'vul', '') == 'timestamp' else 10
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_value)
                
                optimizer.step()

                # del x1, x2, y, outputs, loss
                # torch.cuda.empty_cache()
                # gc.collect()
            
            # Record Average Loss
            avg_loss = self.result['loss'] / len(dataloader)
            self.tb_writer.add_scalar("loss/train", avg_loss, self.tb_global_step)
            
            # Log loss to text file
            with open(self.log_file_path, "a") as f:
                f.write(f"{self.tb_global_step},{epoch},{avg_loss}\n")
                
            self.tb_global_step += 1


class Fed_CLC_client(object):
    def __init__(
        self, 
        args,
        criterion,
        model,
        dataset,
        client_id,
        tao
    ):
        self.args = args
        self.criterion = criterion
        self.model = model
        self.dataset = dataset
        self.client_id = client_id
        self.tao = tao
        self.dataloader = DataLoader(self.dataset, batch_size=args.batch, shuffle=True, pin_memory=True)

    def get_parameters(self):
        return self.model.state_dict()
    
    def print_loss(self):
        print(f"loss is {self.result['loss']}")

    def train(self):
        if self.args.model_type == "CBGRU":
            lr = self.args.cbgru_local_lr
        elif self.args.model_type == "CGE":
            lr = self.args.cge_local_lr
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.result = dict()
        device = self.args.device
        self.result['sample'] = len(self.dataloader)

        self.model.train()
        for epoch in range(self.args.cbgru_local_epoch):
            self.result['loss'] = 0
            for x1, x2, y in self.dataloader:
                optimizer.zero_grad()
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                outputs = self.model(x1, x2)
                y = y.flatten().long()
                loss = self.criterion(outputs, y)
                self.result['loss'] = self.result['loss'] + loss.item()
                loss.backward()
                optimizer.step()

                # del x1, x2, y, outputs, loss
                # torch.cuda.empty_cache()
                # gc.collect()

    def sendconf(self):
        confListU, class_nums = self.confidence()
        sys.stdout.write('\r')
        sys.stdout.write('User = [%d/%d]  | confidence is computed        '
                         % (self.client_id, self.args.client_num))
        sys.stdout.flush()
        return confListU, class_nums

    def data_holdout(self, conf_score):
        r = self.sfm_Mat.shape[0]
        delta_sort = {}
        naive_num = 0
        self.keys = []
        self.sudo_labels = []
        for idx in range(r):
            if idx == 26:
                debug = True
            softmax = self.sfm_Mat[idx]

            maxPro_Naive = -1
            preIndex_Naive = -1
            maxPro = -1
            preIndex = -1

            for j in range(self.args.num_classes):
                if softmax[j] > maxPro_Naive:
                    preIndex_Naive = j
                    maxPro_Naive = softmax[j]

                if softmax[j] > conf_score[j]:
                    if softmax[j] > maxPro:
                        maxPro = softmax[j]
                        preIndex = j

            label = int(softmax[-1])
            margin = maxPro_Naive - softmax[label]

            if preIndex == -1:
                preIndex = preIndex_Naive
                maxPro = maxPro_Naive
                naive_num += 1
            elif preIndex != label:
                delta_sort[idx] = margin
            
            self.sudo_labels.append(preIndex)
        
        delta_sorted = sorted(delta_sort.items(), key=lambda delta_sort: delta_sort[1], reverse=True)  # 降序
        reserve = []

        for (k, v) in delta_sorted:
            if v > self.tao:
                self.keys.append(k)

        # 对于没有被放入到keys中的样本,保留下来
        for idx in range(r):
            if idx not in self.keys:
                reserve.append(idx)

        for idx in range(r):
            if idx not in self.keys:
                reserve.append(idx)
        names = np.array(self.dataset.names)
        labels = np.array(self.dataset.labels)
        names = names[reserve]
        labels = labels[reserve]
        self.avai_dataset = copy.deepcopy(self.dataset)
        self.avai_dataset.names = names
        self.avai_dataset.labels = labels
        self.data_loader = DataLoader(self.avai_dataset, batch_size=self.args.batch, shuffle=True)

    def confidence(self):
        outputSofma = self.outputSof()
        self.sfm_Mat = outputSofma  # Store it here
        r = outputSofma.shape[0]
        c = outputSofma.shape[1]
        prob_everyclass = [[] for i in range(c - 1)]
        class_nums = []
        confList = []

        for i in range(r):
            oriL = outputSofma[i][c - 1]
            oriL = int(oriL)
            pro = outputSofma[i, oriL]
            prob_everyclass[oriL].append(pro)

        for i in range(c - 1):
            confList.append(round(np.mean(prob_everyclass[i], axis=0), 3))
            class_nums.append(len(prob_everyclass[i]))
        self.sfm_Mat = outputSofma

        return confList, class_nums

    def outputSof(self):
        dataset = self.dataset
        s = dataset.labels
        s = np.array(s)

        self.model.eval()
        device = self.args.device
        val_loader = DataLoader(dataset, batch_size=self.args.batch, shuffle=False, pin_memory=True)
        outputs_list = []
        with torch.no_grad():
            for x1, x2, labels in val_loader:
                x1, x2 = x1.to(device), x2.to(device)
                batch_out = self.model(x1, x2)
                outputs_list.append(batch_out.cpu()) # Move to CPU immediately

        outputs = torch.cat(outputs_list, dim=0)
        
        psx_cv = F.softmax(outputs, dim=1)
        
        psx = psx_cv.numpy().reshape((-1, self.args.num_classes))
        s = s.reshape([s.size, 1])
        sfm_Mat = np.hstack((psx, s))

        return sfm_Mat

    def data_correct(self):
        self.avai_dataset.labels = np.array(self.sudo_labels)[self.reserve]
        self.data_loader = DataLoader(self.avai_dataset, batch_size=self.args.batch, shuffle=True, pin_memory=True)





class Fed_Ablation_client(Fed_PLE_client):
    def __init__(self, args, criterion, device, inner_model, outer_model, noise_dataloader, pure_dataloader):
        super().__init__(args, criterion, device, inner_model, outer_model, noise_dataloader, pure_dataloader)

    def print_loss(self):
        print(f"lcn_loss is {self.result['lcn_loss']}")
        print(f"classifier_loss is {self.result['classifier_loss']}")

    def train(self):
        torch.autograd.set_detect_anomaly(True)

        inner_model_copy = copy.deepcopy(self.inner_model)

        outer_optimizer = torch.optim.Adam(self.outer_model.parameters(), lr=self.args.outer_lr)
        inner_optimizer = torch.optim.Adam(self.inner_model.parameters(), lr=self.args.cbgru_local_lr)
        inner_copy_opt = torch.optim.Adam(inner_model_copy.parameters(), lr=self.args.cbgru_local_lr)
        
        self.result = dict()
        self.result['sample'] = len(self.noise_dataloader)
        for epoch in range(self.args.local_epoch):
            outer_loss_total = torch.tensor(0., device=self.device)

            # 训练概率标签模型
            for e in range(1):
                self.result['lcn_loss'] = torch.tensor(0., device=self.device)
                for (x1, x2, noise_labels, global_labels), (x1_pure, x2_pure, pure_labels) in zip(self.noise_dataloader, self.pure_dataloader):
                    outer_optimizer.zero_grad()
                    
                    x1, x2 = x1.to(self.device), x2.to(self.device)
                    noise_labels, global_labels = noise_labels.to(self.device), global_labels.to(self.device)
                    x1_pure, x2_pure, pure_labels = x1_pure.to(self.device), x2_pure.to(self.device), pure_labels.to(self.device)

                    inner_model_copy.eval()
                    self.outer_model.train()
                    predictions = inner_model_copy(x1, x2)
                    predictions = F.softmax(predictions, dim=-1)

                    # 使用钩子函数获取的中间输出
                    h_x = inner_model_copy.inter_outputs
                    h_x.requires_grad = True
                    gl_one_hot = F.one_hot(global_labels.long().flatten(),num_classes=2)
                    gl_one_hot = gl_one_hot.unsqueeze(1)
                    nl_one_hot = F.one_hot(noise_labels.long().flatten(),num_classes=2)
                    nl_one_hot = nl_one_hot.unsqueeze(1)
                    cat_labels = torch.cat((gl_one_hot, nl_one_hot), dim=1)
                    cat_labels = cat_labels.float()
                    cat_labels.requires_grad = True
                    outer_outputs = self.outer_model(h_x, cat_labels)
                    outer_outputs = torch.squeeze(outer_outputs, dim=1)
                    outer_outputs = torch.softmax(outer_outputs, dim=-1)

                    loss = self.criterion(outer_outputs, pure_labels)
                    self.result['lcn_loss'] = self.result['lcn_loss'] + loss.item()

                    loss.backward()
                    outer_optimizer.step()

                    del x1, x2, noise_labels, global_labels
                    del h_x, gl_one_hot, nl_one_hot, cat_labels
                    del outer_outputs
                    torch.cuda.empty_cache()
                    gc.collect()

            for e in range(1):
                self.result['classifier_loss'] = torch.tensor(0., device=self.device)
                for (x1, x2, noise_labels, global_labels), (x1_pure, x2_pure, pure_labels) in zip(self.noise_dataloader, self.pure_dataloader):
                    outer_optimizer.zero_grad()

                    x1, x2 = x1.to(self.device), x2.to(self.device)
                    noise_labels, global_labels = noise_labels.to(self.device), global_labels.to(self.device)
                    x1_pure, x2_pure, pure_labels = x1_pure.to(self.device), x2_pure.to(self.device), pure_labels.to(self.device)

                    self.inner_model.train()
                    self.outer_model.eval()
                    inner_optimizer.zero_grad()
                    predictions = self.inner_model(x1, x2)
                    predictions = F.softmax(predictions, dim=-1)

                    # 使用钩子函数获取的中间输出
                    h_x = self.inner_model.inter_outputs
                    h_x.requires_grad = True
                    gl_one_hot = F.one_hot(global_labels.long().flatten(),num_classes=2)
                    gl_one_hot = gl_one_hot.unsqueeze(1)
                    nl_one_hot = F.one_hot(noise_labels.long().flatten(),num_classes=2)
                    nl_one_hot = nl_one_hot.unsqueeze(1)
                    cat_labels = torch.cat((gl_one_hot, nl_one_hot), dim=1)
                    cat_labels = cat_labels.float()
                    cat_labels.requires_grad = True
                    outer_outputs = self.outer_model(h_x, cat_labels)
                    outer_outputs = torch.squeeze(outer_outputs, dim=1)
                    outer_outputs = torch.softmax(outer_outputs, dim=-1)

                    # 内循环
                    inner_loss = nn.functional.kl_div(predictions, outer_outputs, reduction='batchmean')
                    inner_loss.backward()
                    inner_optimizer.step()
                
                    self.result['classifier_loss'] = self.result['classifier_loss'] + inner_loss.item()
                    
                    # del x1, x2, noise_labels, global_labels
                    # del h_x, gl_one_hot, nl_one_hot, cat_labels
                    # del outer_outputs, inner_loss
                    # torch.cuda.empty_cache()
                    # gc.collect()
