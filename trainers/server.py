import torch
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from trainers.evaluation import Evaluation
from sklearn.metrics import f1_score
import os
import time


class Server(object):
    def __init__(
        self,
        args,
        model,
        device,
        criterion
    ):
        self.args = args
        self.global_model = model
        self.device = device
        self.criterion = criterion
        self.result_dict = dict()

    def _initialize_global_optimizer(self):
        global_optimizer = torch.optim.SGD(
            self.global_model.parameters(),
            lr= self.args.global_learning_rate,
            momentum=0.9,
            weight_decay=0.0
        )
        return global_optimizer
    
    def sample_clients(self, num_of_clients, sample_rate = 0.5):
        pass

    def initialize_epoch_updates(self, epoch):
        self.epoch = epoch
        self.model_updates = list()
        self.num_samples_list = list()
        self.val_F1 = list()
        self.result_dict[self.epoch] = dict()
        self.result_dict[self.epoch]['train'] = list()
        self.result_dict[self.epoch]['dev'] = list()
        self.result_dict[self.epoch]['test'] = list()
    
    def get_paramerters(self):
        return self.global_model.state_dict()
    
    def save_val_updates(
        self,
        result:dict
    ):
        self.result_dict[self.epoch]['val'].append(result)
        self.val_F1.append(result['F1 score'])

    def save_train_updates(
        self,
        model_updates: dict,
        num_sample: int,
        result: dict
    ):
        self.model_updates.append(model_updates)
        self.num_samples_list.append(num_sample)
        self.result_dict[self.epoch]['train'].append(result)

    def average_weights(self):
        if len(self.num_samples_list) == 0:
            return
        print(self.num_samples_list)
        total_num_samples = np.sum(self.num_samples_list)
        total_client_num = len(self.num_samples_list)
        w_avg = copy.deepcopy(self.model_updates[0])

        for key in w_avg.keys():
            w_avg[key] = self.model_updates[0][key] * (self.num_samples_list[0]/total_num_samples)
            # w_avg[key] = self.model_updates[0][key] * (1.0/total_client_num)
        for key in w_avg.keys():
            for i in range(1, len(self.model_updates)):
                w_avg[key] += torch.div(self.model_updates[i][key]*self.num_samples_list[i], total_num_samples)
                # w_avg[key] += torch.div(self.model_updates[i][key], total_client_num)
        
        self.global_model.load_state_dict(copy.deepcopy(w_avg))
    

class ARFL_Server(Server):
    def __init__(
        self,
        args,
        model,
        criterion,
        seed,
        clients,
        total_num_samples
    ):
        super().__init__(args, model, args.device, criterion)
        self.client_num = args.client_num
        # self.weights = np.ones(self.client_num, dtype=np.float64)
        self.clients = clients
        self.seed = seed
        self.total_num_samples = total_num_samples
        self.reg_weight = self.total_num_samples if args.reg_weight is None else args.reg_weight * self.total_num_samples

    def sample_clients(self, my_round):
        # np.random.seed(self.seed*1000 + float(my_round))
        candidates = [i for i in range(self.client_num)]
        print(candidates)
        while True:
            selected_indices = np.random.choice(candidates, int(self.client_num*self.args.sample_rate), replace=False).tolist()
            if sum([self.clients[c].weight for c in selected_indices]) != 0:
                break
        # self.selected_clients = self.clients[selected_indices]
        self.selected_clients = list()
        for idx in selected_indices:
            self.selected_clients.append(self.clients[idx])

        print(f"Selected Clients in Round{my_round}: {selected_indices}")

    def average_weights(self):
        weights = [c.weight for c in self.selected_clients]
        if sum(weights) >= 0:
            nor_weights = np.array(weights) / np.sum(weights)
            # w_avg = copy.deepcopy(self.model_updates[self.sample_clients[0]])
            first_model = self.selected_clients[0].get_model_parameters()
            w_avg = copy.deepcopy(first_model)
            for key in w_avg.keys():
                # w_avg[key] = self.model_updates[self.sample_clients[0]][key] * nor_weights[0]
                w_avg[key] = first_model[key] * nor_weights[0]

            for key in w_avg.keys():
                for i in range(1, len(self.selected_clients)):
                    client = self.selected_clients[i]
                    client_parameters = client.get_model_parameters()
                    w_avg[key] += client_parameters[key] * nor_weights[i]

            self.global_model.load_state_dict(copy.deepcopy(w_avg))
        else:
            print("All weights sum up is 0")

    def update_alpha(self):
        for c in self.selected_clients:
            c.test()
        idxs = [x for x, _ in sorted(enumerate(self.clients), key=lambda x: x[1].get_test_loss())]
        print(idxs)
        eta_optimal = self.clients[idxs[0]].get_test_loss() + self.reg_weight
        for p in range(0, len(idxs)):
            eta = (sum([self.clients[i].num_train_samples * self.clients[i].get_test_loss() for i in idxs[:p+1]]) + self.reg_weight) / sum([self.clients[i].num_train_samples for i in idxs[:p+1]])

            if eta - self.clients[idxs[p]].get_test_loss() < 0:
                break
            else:
                eta_optimal = eta
        weights = [c.num_train_samples * max(eta_optimal - c.get_test_loss(), 0) / self.reg_weight for c in self.clients]
        for i, c in enumerate(self.clients):
            w = c.num_train_samples * max(eta_optimal - c.get_test_loss(), 0) / self.reg_weight
            c.set_weight(w)
        return weights, np.dot(weights, [c.get_test_loss() for c in self.clients]) + self.reg_weight * np.sum([w**2 / c.num_train_samples for w, c in zip(weights, self.clients)]) / 2


class LGV_server(Server):
    def __init__(
        self,
        args,
        model,
        device,
        criterion,
        global_weight,
        run_timestamp=None
    ):
        super().__init__(args, model, device, criterion)
        self.global_weight = global_weight
        self.previous_f1 = None
        self.run_timestamp = run_timestamp if run_timestamp else time.strftime("%Y%m%d_%H%M%S")
    
    def autotune_gr(self, valid_dl):
        all_predictions = []
        all_targets = []
        total_loss = 0
        self.global_model.eval()
        
        # 构建与 Client 一致的日志路径: runs/{lab_name}/{noise_type}/{noise_rate}/{vul}/{timestamp}/valid
        # 注意：Client 是在 runs/.../{timestamp}/client_{id}
        # Server 验证集日志放在 runs/.../{timestamp}/valid
        
        # 1. 构建基础路径
        # 假设 lab_name, noise_type 等都在 args 中
        # 路径结构: runs/Fed_LGV/CBGRU/sys_noise/0.3/reentrancy/20260122_164552/valid
        
        base_dir = os.path.join(
            "runs",
            self.args.lab_name,
            self.args.model_type,
            self.args.noise_type,
            str(self.args.noise_rate),
            self.args.vul,
            self.run_timestamp
        )
        
        valid_log_dir = os.path.join(base_dir, "valid")
        
        if not os.path.exists(valid_log_dir):
            os.makedirs(valid_log_dir)
        
        log_file_path = os.path.join(valid_log_dir, "loss_log.txt")
        
        # 如果文件不存在，写入表头
        if not os.path.exists(log_file_path):
            with open(log_file_path, "w") as f:
                f.write("Timestamp,Validation_Loss,Macro_F1\n")
        
        with torch.no_grad():
            for x1, x2, y in valid_dl:
                x1, x2, y = x1.to(self.args.device), x2.to(self.args.device), y.to(self.args.device)
                y = y.flatten().long()
                outputs = self.global_model(x1, x2)

                loss = self.criterion(outputs, y)
                total_loss += loss.item()
                softmax = nn.Softmax(dim=1)
                pred = torch.argmax(softmax(outputs), dim=-1)
                all_predictions.extend(pred.flatten().tolist())
                all_targets.extend(y.flatten().tolist())
            
            torch.cuda.empty_cache()
            
        # 计算平均 loss
        avg_loss = total_loss / len(valid_dl)
        print(f"Validation Loss: {avg_loss}")
        
        # 记录 loss 到文件
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        
        from sklearn.metrics import f1_score
        current_f1 = f1_score(all_targets, all_predictions, average='macro')
        
        with open(log_file_path, "a") as f:
            f.write(f"{current_time},{avg_loss},{current_f1}\n")
        
        if self.previous_f1 != None:
            # 逻辑修正：
            # 如果 F1 上升 (current > previous)，说明当前方向正确或模型变强，
            # 我们应该保持信心，或者适度增加 global_weight 以利用更强的全局模型（前提是没到上限）。
            # 如果 F1 下降 (current < previous)，说明全局模型可能引入了噪声，或者权重过大，
            # 应该降低 global_weight 以回退到更安全的本地视图。
            
            if current_f1 > self.previous_f1:
                # 性能提升，尝试稍微增加权重（奖励），利用更好的全局模型
                # 但不要加太快，防止震荡
                self.global_weight += self.args.adjustment_factor
            elif current_f1 < self.previous_f1:
                # 性能下降，降低权重（惩罚），减少全局噪声影响
                self.global_weight -= self.args.adjustment_factor
                
            # 边界约束：防止权重过小或过大
            self.global_weight = max(0.1, min(self.global_weight, 0.75))
            
            print(f"Auto-tuning: F1 {self.previous_f1:.4f} -> {current_f1:.4f}, New Global Weight: {self.global_weight:.4f}")
            
        self.previous_f1 = current_f1

class CLC_Server(Server):
    def __init__(
        self,
        args,
        model,
        device,
        criterion
    ):
        super().__init__(args, model, args.device, criterion)
        self.class_nums_each = [[] for i in range(args.client_num)]
        self.conflist_each = [[] for i in range(args.client_num)]

    def receiveconf(self, confs, classnums):
        for ix in range(self.args.client_num):
            self.conflist_each[ix] = confs[ix]
            self.class_nums_each[ix] = classnums[ix]
    
    def conf_agg(self):
        conf_score = [0] * self.args.num_classes
        conf_wt = [[0] * self.args.client_num for i in range(self.args.num_classes)]
        class_nums = np.array(self.class_nums_each)
        sum_col = class_nums.sum(axis=0)
        for ix in range(self.args.client_num):
            for i in range(self.args.num_classes):
                denom = sum_col[i]
                nom = self.class_nums_each[ix][i]
                w = nom / denom
                conf_wt[i][ix] = w

            if ix == self.args.client_num - 1:

                for i in range(self.args.num_classes):
                    for j in range(self.args.client_num):
                        conf_score[i] += conf_wt[i][j] * self.conflist_each[j][i]
        return conf_score


class CRD_server(Server):
    def __init__(
        self,
        args,
        model,
        device,
        criterion
    ):
        super().__init__(args, model, device, criterion)
        # CRD specific params
        self.lambda_crd = getattr(args, 'lambda_crd', 2.0)
        self.alpha_crd = getattr(args, 'alpha_crd', 0.5)
        self.C_min = getattr(args, 'C_min', 0.5)
        self.C_max = getattr(args, 'C_max', 2.0)

    def aggregate(self, updates_list):
        """
        updates_list: list of (client_id, delta_state_dict, q_k, n_k)
        """
        if not updates_list:
            return
            
        # 1. Compute Reference Update (delta_bar)
        num_clients = len(updates_list)
        
        # Initialize delta_bar with zeros
        first_delta = updates_list[0][1]
        delta_bar = {k: torch.zeros_like(v).float() for k, v in first_delta.items()}
        
        for _, delta, _, _ in updates_list:
            for k, v in delta.items():
                delta_bar[k] += v.float()
                
        for k in delta_bar.keys():
            delta_bar[k] /= num_clients
            
        # Flatten delta_bar for cosine similarity calculation
        def flatten(state_dict):
            return torch.cat([v.flatten().float() for v in state_dict.values()])
            
        delta_bar_vec = flatten(delta_bar).to(self.device)
        norm_delta_bar = torch.norm(delta_bar_vec) + 1e-8
        
        # 2. Process each client
        processed_updates = [] # list of (hat_delta, omega)
        total_weight = 0.0
        
        for client_id, delta, q_k, n_k in updates_list:
            delta_vec = flatten(delta).to(self.device)
            norm_delta = torch.norm(delta_vec) + 1e-8
            
            # Direction Consistency (s_k)
            cos_sim = torch.dot(delta_vec, delta_bar_vec) / (norm_delta * norm_delta_bar)
            s_k = cos_sim.item()
            
            # Magnitude Consistency (m_k)
            ratio = norm_delta / norm_delta_bar
            m_k = torch.exp(-torch.abs(torch.log(ratio))).item()
            
            # Training Process Signal (r_k)
            r_k = s_k * m_k
            
            # Corrected Reliability (q_tilde)
            q_tilde = q_k + self.alpha_crd * r_k
            q_tilde = max(0.0, min(1.0, q_tilde)) # clip to [0, 1]
            
            # Adaptive Clipping Threshold (C_k)
            C_k = self.C_min + (self.C_max - self.C_min) * q_tilde
            
            # Perform Clipping (hat_delta)
            scaling_factor = min(1.0, C_k / (norm_delta.item() + 1e-8))
            
            hat_delta = {k: v * scaling_factor for k, v in delta.items()}
            
            # Aggregation Weight (omega_k)
            omega_k = n_k * q_tilde
            
            processed_updates.append((hat_delta, omega_k))
            total_weight += omega_k
            
        # 3. Aggregate
        if total_weight == 0:
            print("Total weight is 0, skipping update")
            return
            
        # Initialize global update with zeros
        global_update = {k: torch.zeros_like(v).float() for k, v in first_delta.items()}
        
        for hat_delta, omega in processed_updates:
            normalized_weight = omega / total_weight
            for k, v in hat_delta.items():
                global_update[k] += v.to(self.device) * normalized_weight
                
        # Apply update to global model
        current_params = self.global_model.state_dict()
        new_params = copy.deepcopy(current_params)
        
        for k in new_params.keys():
            if k in global_update:
                new_params[k] = current_params[k].float() + global_update[k].float()
                
        self.global_model.load_state_dict(new_params)
        print(f"Aggregated {len(processed_updates)} updates with total weight {total_weight:.4f}")
    