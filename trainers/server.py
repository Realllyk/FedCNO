import torch
import copy
import numpy as np
import math
import torch
import torch.nn as nn
import torch.functional as F
from trainers.evaluation import Evaluation
from sklearn.metrics import f1_score
import os
import time


def _move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_device(v, device) for v in obj)
    return obj


def _unpack_batch(batch):
    if len(batch) == 3:
        return batch[0], batch[1], batch[2]
    if len(batch) == 4:
        return batch[0], batch[1], batch[2]
    raise ValueError(f"Unexpected batch length={len(batch)} in server autotune")


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
            for batch in valid_dl:
                x1, x2, y = _unpack_batch(batch)
                x1 = _move_to_device(x1, self.args.device)
                x2 = _move_to_device(x2, self.args.device)
                y = _move_to_device(y, self.args.device)
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
        self.lambda_agg = getattr(args, 'lambda_agg', 2.0)
        self.alpha_crd = getattr(args, 'alpha_crd', 0.5)
        # Legacy placeholders kept intentionally for ablation history.
        # Current thesis-aligned implementation below does not use C_min/C_max.
        self.C_min = getattr(args, 'C_min', 0.5)
        self.C_max = getattr(args, 'C_max', 2.0)
        
        # EMA Model Init
        self.ema_beta = getattr(args, 'ema_beta', 0.9)
        self.ema_model = copy.deepcopy(model)
        for param in self.ema_model.parameters():
            param.requires_grad = False
        self.diag_logger = None
        self.diag_meta = {}

    def update_ema_model(self):
        """
        Update EMA model: theta_ema = beta * theta_ema + (1-beta) * theta
        """
        beta = self.ema_beta
        with torch.no_grad():
            for param_ema, param_global in zip(self.ema_model.parameters(), self.global_model.parameters()):
                param_ema.data.mul_(beta).add_(param_global.data, alpha=1 - beta)

    def aggregate(self, updates_list):
        """
        updates_list: list of (client_id, delta_state_dict, q_k, n_k)
        """
        if not updates_list:
            return

        # Legacy implementation (kept for reference, DO NOT DELETE):
        # - relative magnitude consistency m_k = exp(-|log(||delta_k||/||delta_bar||)|)
        # - corrected reliability q_tilde clipped to [0, 1]
        # - clipping threshold C_k = C_min + (C_max - C_min) * q_tilde
        #
        # num_clients = len(updates_list)
        # first_delta = updates_list[0][1]
        # delta_bar = {k: torch.zeros_like(v).float() for k, v in first_delta.items()}
        # for _, delta, _, _ in updates_list:
        #     for k, v in delta.items():
        #         delta_bar[k] += v.float()
        # for k in delta_bar.keys():
        #     delta_bar[k] /= num_clients
        # def flatten(state_dict):
        #     return torch.cat([v.flatten().float() for v in state_dict.values()])
        # delta_bar_vec = flatten(delta_bar).to(self.device)
        # norm_delta_bar = torch.norm(delta_bar_vec) + 1e-8
        # processed_updates = []
        # total_weight = 0.0
        # for client_id, delta, q_k, n_k in updates_list:
        #     delta_vec = flatten(delta).to(self.device)
        #     norm_delta = torch.norm(delta_vec) + 1e-8
        #     cos_sim = torch.dot(delta_vec, delta_bar_vec) / (norm_delta * norm_delta_bar)
        #     s_k = cos_sim.item()
        #     ratio = norm_delta / norm_delta_bar
        #     m_k = torch.exp(-torch.abs(torch.log(ratio))).item()
        #     r_k = s_k * m_k
        #     q_tilde = q_k + self.alpha_crd * r_k
        #     q_tilde = max(0.0, min(1.0, q_tilde))
        #     C_k = self.C_min + (self.C_max - self.C_min) * q_tilde
        #     scaling_factor = min(1.0, C_k / (norm_delta.item() + 1e-8))
        #     hat_delta = {k: v * scaling_factor for k, v in delta.items()}
        #     omega_k = n_k * q_tilde
        #     processed_updates.append((hat_delta, omega_k))
        #     total_weight += omega_k
        # if total_weight == 0:
        #     return
        # global_update = {k: torch.zeros_like(v).float() for k, v in first_delta.items()}
        # for hat_delta, omega in processed_updates:
        #     normalized_weight = omega / total_weight
        #     for k, v in hat_delta.items():
        #         global_update[k] += v.to(self.device) * normalized_weight

        # Thesis-aligned implementation:
        # (1) r_raw_k^t = cos(delta_k^t, delta_bar^t) * exp(-lambda_agg * ||delta_k^t||)
        # (2) r_hat_k^t = q_k^t + alpha * r_raw_k^t
        # (3) r_tilde_k^t = softplus(r_hat_k^t) / sum_u softplus(r_hat_u^t)
        # (4) tau^t is rho-quantile of ||delta_k^t|| over participating clients
        # (5) delta_k,clip^t = min(||delta_k^t||, tau^t) / (||delta_k^t|| + eps) * delta_k^t
        # (6) omega_k^t = n_k * r_tilde_k^t / sum_u n_u * r_tilde_u^t
        crd_eps = float(getattr(self.args, 'crd_eps', 1e-8))
        crd_rho = float(getattr(self.args, 'crd_rho', 0.7))
        crd_rho = max(0.0, min(1.0, crd_rho))
        crd_sigma = str(getattr(self.args, 'crd_sigma', 'softplus')).lower()
        if crd_sigma != 'softplus':
            print(f"[FedCRD] crd_sigma={crd_sigma} is unsupported, fallback to softplus.")

        num_clients = len(updates_list)
        first_delta = updates_list[0][1]

        def flatten(state_dict):
            return torch.cat([v.flatten().float() for v in state_dict.values()])

        # 1) Reference update delta_bar^t
        delta_bar = {k: torch.zeros_like(v).float() for k, v in first_delta.items()}
        for _, delta_k_t, _, _ in updates_list:
            for k, v in delta_k_t.items():
                delta_bar[k] += v.float()
        for k in delta_bar.keys():
            delta_bar[k] /= num_clients

        delta_bar_t_vec = flatten(delta_bar).to(self.device)
        norm_delta_bar_t = torch.norm(delta_bar_t_vec).item()

        # 2) Compute r_raw and r_hat for each client
        client_stats = []
        sigma_sum = 0.0
        for client_id, delta_k_t, q_k_t, n_k in updates_list:
            delta_k_t_vec = flatten(delta_k_t).to(self.device)
            norm_delta_k_t = torch.norm(delta_k_t_vec).item()

            denom = (norm_delta_k_t * norm_delta_bar_t) + crd_eps
            cos_sim = (torch.dot(delta_k_t_vec, delta_bar_t_vec).item()) / denom
            r_raw_k_t = cos_sim * np.exp(-self.lambda_agg * norm_delta_k_t)
            r_hat_k_t = q_k_t + self.alpha_crd * r_raw_k_t

            sigma_k_t = torch.nn.functional.softplus(
                torch.tensor(r_hat_k_t, dtype=torch.float32, device=self.device)
            ).item()
            sigma_sum += sigma_k_t

            client_stats.append(
                {
                    'client_id': client_id,
                    'delta_k_t': delta_k_t,
                    'n_k': n_k,
                    'q_k_t': q_k_t,
                    'norm_delta_k_t': norm_delta_k_t,
                    'cos_sim_k_t': cos_sim,
                    'r_raw_k_t': r_raw_k_t,
                    'r_hat_k_t': r_hat_k_t,
                    'sigma_k_t': sigma_k_t
                }
            )

        # 3) Positive and normalized reliability r_tilde
        sigma_denom = sigma_sum + crd_eps
        for stat in client_stats:
            stat['r_tilde_k_t'] = stat['sigma_k_t'] / sigma_denom

        # 4) Rho-quantile adaptive clipping threshold tau^t
        norms_sorted = sorted([stat['norm_delta_k_t'] for stat in client_stats])
        n_t = len(norms_sorted)
        quantile_rank = max(1, min(n_t, int(math.ceil(crd_rho * n_t))))
        tau_t = norms_sorted[quantile_rank - 1]

        # 5) Clip updates and 6) reliability-weighted aggregation
        omega_denom = sum([stat['n_k'] * stat['r_tilde_k_t'] for stat in client_stats]) + crd_eps
        global_update = {k: torch.zeros_like(v).float() for k, v in first_delta.items()}
        sum_omega = 0.0

        for stat in client_stats:
            norm_delta_k_t = stat['norm_delta_k_t']
            clip_scale = min(norm_delta_k_t, tau_t) / (norm_delta_k_t + crd_eps)
            delta_k_t_clip = {k: v * clip_scale for k, v in stat['delta_k_t'].items()}

            omega_k_t = (stat['n_k'] * stat['r_tilde_k_t']) / omega_denom
            stat['tau_t'] = tau_t
            stat['clip_scale_k_t'] = clip_scale
            stat['omega_k_t'] = omega_k_t
            sum_omega += omega_k_t

            for k, v in delta_k_t_clip.items():
                global_update[k] += v.to(self.device) * omega_k_t

        # Apply update to global model
        current_params = self.global_model.state_dict()
        new_params = copy.deepcopy(current_params)
        
        for k in new_params.keys():
            if k in global_update:
                new_params[k] = current_params[k].float() + global_update[k].float()
                
        self.global_model.load_state_dict(new_params)
        mean_r_tilde = float(np.mean([stat['r_tilde_k_t'] for stat in client_stats]))
        sum_r_tilde = float(np.sum([stat['r_tilde_k_t'] for stat in client_stats]))
        print(
            f"Aggregated {len(client_stats)} updates | "
            f"tau_t={tau_t:.6f} | mean_r_tilde={mean_r_tilde:.6f} | "
            f"sum_r_tilde={sum_r_tilde:.6f} | sum_omega={sum_omega:.6f}"
        )

        if self.diag_logger is not None:
            rows = []
            for stat in client_stats:
                rows.append(
                    {
                        'client_id': stat.get('client_id', -1),
                        'n_k': stat.get('n_k', 0),
                        'q_k_t': stat.get('q_k_t', 0.0),
                        'norm_delta_k_t': stat.get('norm_delta_k_t', 0.0),
                        'cos_sim_k_t': stat.get('cos_sim_k_t', 0.0),
                        'r_raw_k_t': stat.get('r_raw_k_t', 0.0),
                        'r_hat_k_t': stat.get('r_hat_k_t', 0.0),
                        'sigma_k_t': stat.get('sigma_k_t', 0.0),
                        'r_tilde_k_t': stat.get('r_tilde_k_t', 0.0),
                        'tau_t': stat.get('tau_t', tau_t),
                        'clip_scale_k_t': stat.get('clip_scale_k_t', 1.0),
                        'omega_k_t': stat.get('omega_k_t', 0.0),
                    }
                )
            self.diag_logger.log_round(self.epoch, self.diag_meta, rows)
        
        # Update EMA model after aggregation
        self.update_ema_model()
    
