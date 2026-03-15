import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy 
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from options import parse_args
from data_processing.dataloader_manager import gen_lgv_ds, gen_test_dl, gen_cbgru_dl, gen_client_ds, gen_valid_dl
from data_processing.preprocessing import compute_global_clusters, coordinate_sys_noise_clusters
from models.model_factory import build_model, get_local_epoch, get_local_lr
from trainers.server import LGV_server
from trainers.client import Fed_LGV_client, Fed_Avg_client, _unpack_batch, _move_to_device, lgv_mando_collate_fn
from global_test import global_test
import random
import time
import concurrent.futures
from collections import deque


def train_warmup_client(client_id, args, global_model, criterion, dataset, run_timestamp):
    """
    热身阶段单个客户端训练函数
    """
    warmup_args = copy.deepcopy(args)
    warmup_args.lab_name = 'Fed_Avg'
    client = Fed_Avg_client(warmup_args,
                        criterion,
                        None,
                        dataset,
                        client_id=client_id,
                        run_timestamp=run_timestamp)
    # 下发全局模型参数
    client.model = copy.deepcopy(global_model)
    client.train()
    
    weights = copy.deepcopy(client.get_parameters())
    num_samples = client.result['sample']
    result = client.result
    
    del client
    torch.cuda.empty_cache()
    gc.collect()
    
    return client_id, weights, num_samples, result


def train_lgv_client(client_id, client, global_model, global_weight):
    """
    LGV正式训练阶段单个客户端训练函数
    """
    if hasattr(client, 'model'):
        del client.model
        
    # 下发全局模型参数
    client.model = copy.deepcopy(global_model)
    client.global_weight = global_weight
    
    # 关键步骤：利用当前全局模型提取特征，动态更新 KNN 概率和一致性
    # 2. 更新全局视图 (Global View)
    client.gen_reduced_ds()
    client.get_global_feature_knn_labels()
    
    # 中文注释：该处逻辑与原实现保持一致。
    # 3. 本地训练 (Local Training)
    client.train()
    
    weights = copy.deepcopy(client.get_parameters())
    num_samples = client.result['sample']
    result = client.result
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return client_id, weights, num_samples, result


class LGVServerPosF1(LGV_server):
    """
    Tuned server for LGV:
    - Keep validation metric consistent with early-stop metric (positive-class F1).
    - Use the same positive-class F1 signal to auto-tune global_weight.
    """
    def autotune_gr(self, valid_dl):
        self.global_model.eval()
        total_loss = 0.0
        tp, fp, fn = 0, 0, 0

        base_dir = os.path.join(
            "runs",
            self.args.lab_name,
            self.args.model_type,
            self.args.noise_type,
            str(self.args.noise_rate),
            self.args.vul,
            self.run_timestamp,
        )
        valid_log_dir = os.path.join(base_dir, "valid")
        os.makedirs(valid_log_dir, exist_ok=True)
        log_file_path = os.path.join(valid_log_dir, "loss_log.txt")
        if not os.path.exists(log_file_path):
            with open(log_file_path, "w") as f:
                f.write("Timestamp,Validation_Loss,Positive_F1\n")

        with torch.no_grad():
            for batch in valid_dl:
                x1, x2, y, _ = _unpack_batch(batch)
                x1 = _move_to_device(x1, self.args.device)
                x2 = _move_to_device(x2, self.args.device)
                y = _move_to_device(y, self.args.device).flatten().long()
                outputs = self.global_model(x1, x2)
                loss = self.criterion(outputs, y)
                total_loss += loss.item()

                pred = torch.argmax(outputs, dim=1)
                tp += torch.sum((pred == 1) & (y == 1)).item()
                fp += torch.sum((pred == 1) & (y == 0)).item()
                fn += torch.sum((pred == 0) & (y == 1)).item()

        avg_loss = total_loss / len(valid_dl)
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        current_f1 = 2.0 * precision * recall / (precision + recall + 1e-12)

        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file_path, "a") as f:
            f.write(f"{current_time},{avg_loss},{current_f1}\n")

        if self.previous_f1 is not None:
            if current_f1 > self.previous_f1:
                self.global_weight += self.args.adjustment_factor
            elif current_f1 < self.previous_f1:
                self.global_weight -= self.args.adjustment_factor
            self.global_weight = max(0.1, min(self.global_weight, 0.75))
            print(
                f"Auto-tuning (positive F1): {self.previous_f1:.4f} -> {current_f1:.4f}, "
                f"New Global Weight: {self.global_weight:.4f}"
            )
        self.previous_f1 = current_f1


class FedLGVClientTuned(Fed_LGV_client):
    """
    Tuned LGV client:
    1) local epoch uses model-adaptive get_local_epoch(args), instead of fixed CBGRU epoch.
    2) pseudo-label update is confidence-gated to avoid low-confidence noisy relabeling.
    3) keep consistency-score weighted loss behavior from original LGV flow.
    """
    def train(self):
        # 1) Fix global snapshot for uncertainty estimation.
        self.fixed_global_model.load_state_dict(self.model.state_dict())
        self.fixed_global_model.eval()

        # 2) Estimate uncertainty and derive per-sample alpha.
        collate_fn = lgv_mando_collate_fn if self.args.model_type == "MANDO" else None
        eval_dl = DataLoader(
            self.dataset,
            batch_size=self.args.batch,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        all_uncertainties = []
        with torch.no_grad():
            for batch in eval_dl:
                x1, x2, _, _ = _unpack_batch(batch)
                x1 = _move_to_device(x1, self.device)
                x2 = _move_to_device(x2, self.device)
                global_logits = self.fixed_global_model(x1, x2)
                global_probs = F.softmax(global_logits, dim=1)
                entropy = -torch.sum(global_probs * torch.log(global_probs + 1e-8), dim=1)
                max_entropy = np.log(global_probs.shape[1])
                all_uncertainties.append((entropy / max_entropy).cpu())

        all_uncertainties = torch.cat(all_uncertainties, dim=0)
        alpha_raw = all_uncertainties ** 2
        alpha_min = self.args.alpha_min
        alpha_max = max(0.0, min(self.global_weight, self.args.alpha_max))
        alpha = torch.clamp(alpha_raw, alpha_min, alpha_max).unsqueeze(1).to(self.device)

        if self.global_prob_labels.device != self.device:
            self.global_prob_labels = self.global_prob_labels.to(self.device)
        if self.local_prob_labels.device != self.device:
            self.local_prob_labels = self.local_prob_labels.to(self.device)

        # 3) Pseudo-label fusion with confidence gate.
        #    Only update sample label when fused pseudo-label confidence >= threshold.
        #    This reduces unstable relabeling in noisy rounds.
        with torch.no_grad():
            fused_prob = alpha * self.global_prob_labels + (1.0 - alpha) * self.local_prob_labels
            fused_prob = F.softmax(fused_prob, dim=1)
            pseudo_labels = torch.argmax(fused_prob, dim=-1)
            confidence = torch.max(fused_prob, dim=1).values
            conf_th = float(getattr(self.args, "dshar_pseudo_threshold", 0.8))
            old_labels = torch.tensor(self.dataset.labels, dtype=torch.long, device=self.device)
            update_mask = confidence >= conf_th
            final_labels = torch.where(update_mask, pseudo_labels, old_labels)
            self.dataset.labels = final_labels.detach().cpu().numpy()

        # 4) Local training using model-adaptive local epoch.
        dl = DataLoader(
            self.dataset,
            batch_size=self.args.batch,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        lr = get_local_lr(self.args)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.args.weight_decay)

        # Keep acceptance ratio for diagnosis in logs/TensorBoard.
        self.result = {"sample": len(self.dataset), "pseudo_accept_rate": float(update_mask.float().mean().item())}
        self.model.train()
        for epoch in range(get_local_epoch(self.args)):
            self.result["loss"] = 0.0
            for batch in dl:
                optimizer.zero_grad()
                x1, x2, y, agr = _unpack_batch(batch)
                x1 = _move_to_device(x1, self.device)
                x2 = _move_to_device(x2, self.device)
                y = _move_to_device(y, self.device).flatten().long()
                agr = _move_to_device(agr, self.device) if agr is not None else torch.ones_like(y, dtype=torch.float32, device=self.device)

                outputs = self.model(x1, x2)
                loss = self.criterion(outputs, y)
                _, predictions = torch.max(outputs, 1)
                correct_predictions = predictions == y
                weights = torch.ones_like(y, dtype=torch.float32, device=self.device)
                weights += agr * (~correct_predictions).float()
                weights -= 0.5 * agr * correct_predictions.float()

                if self.args.consistency_score:
                    if loss.dim() == 0:
                        loss = loss * weights.mean()
                    else:
                        loss = (weights * loss).mean()
                elif loss.dim() > 0:
                    loss = loss.mean()

                self.result["loss"] += loss.item()
                loss.backward()
                clip_value = 1.0 if getattr(self.args, "vul", "") == "timestamp" else 10
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_value)
                optimizer.step()

            avg_loss = self.result["loss"] / len(dl)
            self.tb_writer.add_scalar("loss/train", avg_loss, self.tb_global_step)
            self.tb_writer.add_scalar("pseudo/accept_rate", self.result["pseudo_accept_rate"], self.tb_global_step)
            with open(self.log_file_path, "a") as f:
                f.write(f"{self.tb_global_step},{epoch},{avg_loss}\n")
            self.tb_global_step += 1


if __name__ == '__main__':
    args = parse_args()
    if args.model_type == "MANDO" and args.vul != "tod":
        raise ValueError("MANDO only supports --vul tod in this project.")
    input_size, time_steps = 100, 300

    if args.diff == True:
        noise_rates = random.sample([0.2, 0.2, 0.3, 0.3], 4)
    else:
        noise_rates = [args.noise_rate] * 4

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # 方案：预先分配簇 ID 给每个客户端。
    # 逻辑已封装在 coordinate_sys_noise_clusters 中，包含基于分布的优化分配。
    
    assigned_clusters_dict, global_cluster_map = coordinate_sys_noise_clusters(
        args.client_num, 
        args.vul, 
        args.noise_type, 
        model_type=args.model_type,
        n_clusters=args.n_clusters, 
        seed=int(args.seed),
        data_dir=args.data_dir
    )

    train_ds = list()
    for i in range(args.client_num):
        ds = gen_client_ds(
            args.model_type, 
            i, 
            args.vul, 
            args.noise_type, 
            noise_rates[i], 
            args.num_neigh,
            assigned_clusters=assigned_clusters_dict, 
            global_cluster_map=global_cluster_map,
            n_clusters=args.n_clusters,
            seed=int(args.seed),
            data_dir=args.data_dir
        )
        train_ds.append(ds)
    
    # Collect noise labels from warm-up datasets to reuse in LGV datasets
    generated_noise_labels = [ds.labels for ds in train_ds]
    
    # initialize Server
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # - model_type: 支持 'CBGRU' 或 'CGE' 等不同模型架构。
    # - global_weight: 控制 LGV 算法中全局视图概率的权重。
    # 动态调整类别权重 (Class Weighting)
    # 根据漏洞类型 (args.vul) 设置不同的权重策略：
    # 1. reentrancy (重入漏洞): 数据分布可能较均衡，或者需要轻微的权重调整，使用 [1.0, 1.0] (不加权) 或 [1.0, 1.2]。
    # 2. timestamp (时间戳依赖): 存在严重的漏报 (High FNR)，需要大幅提高 Positive 权重，使用 [1.0, 2.0]。
    # 3. 其他类型: 默认使用 [1.0, 1.0] 或 [1.0, 1.5] 作为保守策略。
    
    vul_label_stats = {
        'reentrancy': (656, 871),
        'timestamp': (187, 199),
        'tod': (717, 196),
    }
    if args.vul in vul_label_stats:
        n0, n1 = vul_label_stats[args.vul]
        total = float(n0 + n1)
        class_weights = torch.tensor(
            [total / (2.0 * n0), total / (2.0 * n1)],
            dtype=torch.float32,
            device=args.device
        )
    else:
        class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32, device=args.device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    if args.model_type == "CBGRU":
        reduction = 'none'
    else:
        reduction = 'mean'
    global_model = build_model(args, input_size, time_steps)
    global_model = global_model.to(args.device)
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    server = LGVServerPosF1(
        args,
        global_model,
        args.device,
        criterion,
        args.global_weight,
        run_timestamp=run_timestamp
    )

    # test_dl 是真正的测试集 (Test Set)
    test_dl = gen_test_dl(args.model_type, args.vul, data_dir=args.data_dir)
    # valid_dl 是验证集 (Validation Set)，用于辅助调参或早停（目前代码中未使用，预留）
    valid_dl = gen_valid_dl(args.model_type, args.vul, data_dir=args.data_dir)

    #  Warm up
    # -------------------------------------------------------------------------
    # 阶段 1: 热身训练 (Warm-up Phase)
    # -------------------------------------------------------------------------
    # 在正式启用 Fed_LGV 逻辑之前，先使用标准的 FedAvg 算法进行若干轮预训练。
    # 目的：
    # 1. 让全局模型快速收敛到一个合理的初始状态。
    # 2. 为后续 LGV 阶段提取有效的全局特征奠定基础（如果模型完全随机，提取的特征就没有意义）。
    warmup_valid_interval = max(1, int(getattr(args, "warmup_valid_interval", 1)))
    warmup_early_stop_patience = int(getattr(args, "warmup_early_stop_patience", 0))
    warmup_early_stop_min_delta = float(getattr(args, "warmup_early_stop_min_delta", 1e-4))
    warmup_min_epoch_for_early_stop = int(getattr(args, "warmup_min_epoch_for_early_stop", 0))
    warmup_valid_f1_smooth_window = max(1, int(getattr(args, "warmup_valid_f1_smooth_window", 1)))
    warmup_f1_history = deque(maxlen=warmup_valid_f1_smooth_window)
    warmup_best_val_f1 = -1.0
    warmup_best_val_f1_raw = -1.0
    warmup_best_epoch = -1
    warmup_no_improve_rounds = 0
    warmup_best_global_state = copy.deepcopy(server.global_model.state_dict())
    original_lab_name = args.lab_name
    args.lab_name = 'Fed_Avg'
    for epoch in range(args.warm_up_epoch):
        print(f"Warm Up Epoch {epoch}: ")
        server.initialize_epoch_updates(epoch)

        # 并行热身训练
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            for client_id in range(args.client_num):
                futures.append(executor.submit(train_warmup_client, client_id, args, server.global_model, criterion, train_ds[client_id], run_timestamp))
            
            for future in futures:
                client_id, weights, num_samples, result = future.result()
                # 上传更新
                server.save_train_updates(
                    weights,
                    num_samples,
                    result
                )
                print(f"client:{client_id}")
                print(f"loss is {result['loss']}")

        # 聚合参数 (FedAvg Aggregation)
        server.average_weights()

        if epoch % warmup_valid_interval == 0:
            print(f"\n--- WarmUp Validation at Epoch {epoch} ---")
            warmup_valid_result = global_test(
                server.global_model,
                valid_dl,
                criterion,
                args,
                "WarmUp_FedAvg",
                reduction='mean',
                run_timestamp=run_timestamp,
                save_result=True,
                tag='valid',
                epoch=epoch
            )
            current_warmup_f1 = warmup_valid_result['F1 score']
            warmup_f1_history.append(current_warmup_f1)
            smooth_warmup_f1 = sum(warmup_f1_history) / len(warmup_f1_history)
            print(
                f"[WARMUP_EARLY_STOP] raw_f1={current_warmup_f1:.6f}, "
                f"smooth_f1={smooth_warmup_f1:.6f}, window={warmup_valid_f1_smooth_window}"
            )
            if smooth_warmup_f1 > (warmup_best_val_f1 + warmup_early_stop_min_delta):
                warmup_best_val_f1 = smooth_warmup_f1
                warmup_best_val_f1_raw = current_warmup_f1
                warmup_best_epoch = epoch
                warmup_no_improve_rounds = 0
                warmup_best_global_state = copy.deepcopy(server.global_model.state_dict())
                print(
                    f"[WARMUP_EARLY_STOP] improved at epoch {epoch}, "
                    f"best_smooth_f1={warmup_best_val_f1:.6f}, best_raw_f1={warmup_best_val_f1_raw:.6f}"
                )
            else:
                warmup_no_improve_rounds += 1
                print(f"[WARMUP_EARLY_STOP] no improvement rounds: {warmup_no_improve_rounds}/{warmup_early_stop_patience}")
                if epoch < warmup_min_epoch_for_early_stop:
                    print(
                        f"[WARMUP_EARLY_STOP] early-stop gating active: "
                        f"epoch {epoch} < min_epoch {warmup_min_epoch_for_early_stop}"
                    )
                elif warmup_early_stop_patience > 0 and warmup_no_improve_rounds >= warmup_early_stop_patience:
                    print(f"[WARMUP_EARLY_STOP] triggered at epoch {epoch}, restoring best epoch {warmup_best_epoch}")
                    break
            print("-----------------------------------------\n")

    # WarmUp 阶段结束后的测试
    if warmup_best_epoch >= 0:
        server.global_model.load_state_dict(warmup_best_global_state)
        print(
            f"[WARMUP_EARLY_STOP] best warmup model restored from epoch {warmup_best_epoch} "
            f"(best_smooth_f1={warmup_best_val_f1:.6f}, best_raw_f1={warmup_best_val_f1_raw:.6f})"
        )
    else:
        print("[WARMUP_EARLY_STOP] no warmup validation checkpoint captured, using final warmup model.")

    # Optional warm-up test:
    # - Controlled by --run_warmup_global_test (default: False).
    # - When enabled, save one FedAvg-stage test result before entering LGV stage.
    if getattr(args, "run_warmup_global_test", False):
        print("\n--- WarmUp Phase Finished. Testing with Fed_Avg lab_name ---")
        global_test(
            server.global_model,
            test_dl,
            criterion,
            args,
            "WarmUp_FedAvg",
            reduction='mean',
            run_timestamp=run_timestamp,
            save_result=True,
            tag='test'
        )
    args.lab_name = original_lab_name
    print("-----------------------------------------------------------\n")
    if args.exit_after_warmup_test:
        print("[CONTROL] --exit_after_warmup_test enabled, exiting after warm-up test.")
        sys.exit(0)
    
    # initialize dataset
    # 重新初始化数据集，为正式的 Fed_LGV 训练做准备
    # gen_lgv_ds 会生成支持图特征/模式特征读取的专用数据集
    
    # -------------------------------------------------------------------------
    # 系统性噪声协调 (Systemic Noise Coordination)
    # -------------------------------------------------------------------------
    # 如果启用了系统性噪声 (sys_noise)，我们希望不同客户端的噪声模式是“错开”的。
    # 方案：预先分配簇 ID 给每个客户端。
    # 逻辑已封装在 coordinate_sys_noise_clusters 中，包含基于分布的优化分配。
    
    # assigned_clusters_dict, global_cluster_map = coordinate_sys_noise_clusters(
    #     args.client_num, 
    #     args.vul, 
    #     args.noise_type, 
    #     n_clusters=args.n_clusters, 
    #     seed=int(args.seed)
    # )

    train_ds = list()
    for i in range(args.client_num):
        ds = gen_lgv_ds(
            i, 
            args.vul, 
            args.noise_type, 
            args.noise_rate, 
            args.num_neigh, 
            args.model_type, 
            assigned_clusters=assigned_clusters_dict, 
            global_cluster_map=global_cluster_map,
            n_clusters=args.n_clusters,
            seed=int(args.seed),
            predefined_labels=generated_noise_labels[i],
            data_dir=args.data_dir
        )
        train_ds.append(ds)
    
    # initialize Client
    # -------------------------------------------------------------------------
    # 初始化 LGV 客户端
    # -------------------------------------------------------------------------
    # 每个客户端被实例化为 Fed_LGV_client，并进行本地视角的初始化。
    clients = []

    for i in range(args.client_num):
        client = FedLGVClientTuned(
            args,
            nn.CrossEntropyLoss(weight=class_weights, reduction=reduction),
            copy.deepcopy(server.global_model),
            train_ds[i],
            i,
            server.global_weight,
            run_timestamp=run_timestamp
        )
        # 计算本地静态视图 (Local View)
        # 基于预训练特征运行 KNN，生成初始的概率分布和一致性，作为先验知识。
        client.get_local_knn_labels(args.vul, args.noise_type, args.noise_rate)
        # client.get_global_knn_labels(args.vul, args.noise_type, args.noise_rate)
        # 生成缩减版数据集，用于后续快速 KNN 检索
        client.gen_reduced_ds()
        clients.append(client)
        print(f"Generate Client {i}!")


    # Train Stage
    # -------------------------------------------------------------------------
    # 阶段 2: 正式训练 (Fed_LGV Training Phase)
    # -------------------------------------------------------------------------
    # 核心循环：
    # 1. 下发模型：服务器分发最新的全局模型给客户端。
    # 2. 全局视图更新：客户端利用全局模型提取特征，运行 KNN 更新标签概率和一致性。
    # 3. 本地训练：结合本地和全局视图生成伪标签，并使用一致性加权 Loss 训练模型。
    # 4. 聚合：服务器聚合客户端上传的参数。
    candidates = [i for i in range(args.client_num)]
    valid_interval = max(1, int(getattr(args, "lgv_valid_interval", 1)))
    early_stop_patience = int(getattr(args, "lgv_early_stop_patience", 15))
    early_stop_min_delta = float(getattr(args, "lgv_early_stop_min_delta", 1e-4))
    min_epoch_for_early_stop = int(getattr(args, "lgv_min_epoch_for_early_stop", 5))
    valid_f1_smooth_window = max(1, int(getattr(args, "lgv_valid_f1_smooth_window", 3)))
    val_f1_history = deque(maxlen=valid_f1_smooth_window)
    best_val_f1 = -1.0
    best_val_f1_raw = -1.0
    best_epoch = -1
    no_improve_rounds = 0
    best_global_state = copy.deepcopy(server.global_model.state_dict())
    for epoch in range(args.epoch):
        print(f"Epoch {epoch}:")
        server.initialize_epoch_updates(epoch)

        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            for client_id in range(args.client_num):
                futures.append(executor.submit(train_lgv_client, client_id, clients[client_id], server.global_model, server.global_weight))

            for future in futures:
                client_id, weights, num_samples, result = future.result()
                server.save_train_updates(
                    weights,
                    num_samples,
                    result
                )
                print(f"client:{client_id}")
                print(f"loss is {result['loss']}")

        server.average_weights()
        if epoch % valid_interval == 0:
            server.autotune_gr(valid_dl)
            print(f"\n--- Validation at Epoch {epoch} ---")
            valid_result = global_test(
                server.global_model,
                valid_dl,
                criterion,
                args,
                f"{args.num_neigh}neigh_{args.global_weight}_{args.lab_name}",
                reduction=reduction,
                run_timestamp=run_timestamp,
                save_result=True,
                tag='valid',
                epoch=epoch
            )
            current_val_f1 = valid_result['F1 score']
            val_f1_history.append(current_val_f1)
            smooth_val_f1 = sum(val_f1_history) / len(val_f1_history)
            print(
                f"[EARLY_STOP] raw_f1={current_val_f1:.6f}, "
                f"smooth_f1={smooth_val_f1:.6f}, window={valid_f1_smooth_window}"
            )
            if smooth_val_f1 > (best_val_f1 + early_stop_min_delta):
                best_val_f1 = smooth_val_f1
                best_val_f1_raw = current_val_f1
                best_epoch = epoch
                no_improve_rounds = 0
                best_global_state = copy.deepcopy(server.global_model.state_dict())
                print(
                    f"[EARLY_STOP] improved at epoch {epoch}, "
                    f"best_smooth_f1={best_val_f1:.6f}, best_raw_f1={best_val_f1_raw:.6f}"
                )
            else:
                no_improve_rounds += 1
                print(f"[EARLY_STOP] no improvement rounds: {no_improve_rounds}/{early_stop_patience}")
                if epoch < min_epoch_for_early_stop:
                    print(
                        f"[EARLY_STOP] early-stop gating active: "
                        f"epoch {epoch} < min_epoch {min_epoch_for_early_stop}"
                    )
                elif early_stop_patience > 0 and no_improve_rounds >= early_stop_patience:
                    print(f"[EARLY_STOP] triggered at epoch {epoch}, restoring best epoch {best_epoch}")
                    break
            print("-------------------------------\n")

    if best_epoch >= 0:
        server.global_model.load_state_dict(best_global_state)
        print(
            f"[EARLY_STOP] best model restored from epoch {best_epoch} "
            f"(best_smooth_f1={best_val_f1:.6f}, best_raw_f1={best_val_f1_raw:.6f})"
        )
    else:
        best_epoch = args.epoch - 1
        print("[EARLY_STOP] no validation checkpoint captured, using final epoch model.")

    global_test(
        server.global_model,
        test_dl,
        criterion,
        args,
        f"{args.num_neigh}neigh_{args.global_weight}_{args.lab_name}",
        reduction=reduction,
        run_timestamp=run_timestamp,
        save_result=True,
        tag='test',
        epoch=best_epoch,
        extra_info={
            "best_valid_f1": best_val_f1,
            "best_valid_f1_raw": best_val_f1_raw,
            "early_stop_patience": early_stop_patience,
            "early_stop_min_delta": early_stop_min_delta,
            "valid_interval": valid_interval,
            "min_epoch_for_early_stop": min_epoch_for_early_stop,
            "valid_f1_smooth_window": valid_f1_smooth_window
        }
    )
        
    
