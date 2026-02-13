import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import gc
import time
import random
import concurrent.futures
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from options import parse_args
from data_processing.dataloader_manager import gen_client_ds, gen_valid_dl
from data_processing.preprocessing import coordinate_sys_noise_clusters
from models.ClassiFilerNet import ClassiFilerNet
from models.CGE_Variants import CGEVariant
from trainers.server import Server
from trainers.client import Fed_Avg_client
from trainers.client_dshar import Fed_DSHAR_client
from global_test import global_test


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_client_dataset_by_mr(dataset, mr_model, args):
    mr_model.eval()
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, pin_memory=True)
    all_conf = []

    with torch.no_grad():
        for x1, x2, _ in loader:
            x1 = x1.to(args.device)
            x2 = x2.to(args.device)
            logits = mr_model(x1, x2)
            probs = torch.softmax(logits, dim=1)
            confidence = torch.max(probs, dim=1)[0]
            all_conf.append(confidence.cpu())

    if len(all_conf) == 0:
        return [], []

    conf = torch.cat(all_conf, dim=0).numpy()
    n = len(conf)
    noisy_ratio = float(np.clip(args.dshar_split_ratio, 0.0, 1.0))
    noisy_count = int(round(n * noisy_ratio))
    noisy_count = min(max(noisy_count, 0), n)

    sorted_idx = np.argsort(conf)
    noisy_indices = sorted_idx[:noisy_count].tolist()
    clean_indices = sorted_idx[noisy_count:].tolist()
    return clean_indices, noisy_indices


def update_teacher_model(teacher_model, student_model, beta):
    with torch.no_grad():
        for t_param, s_param in zip(teacher_model.parameters(), student_model.parameters()):
            t_param.data.mul_(beta).add_(s_param.data, alpha=1.0 - beta)


def evaluate_model(model, dataloader, criterion, args):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for x1, x2, y in dataloader:
            x1 = x1.to(args.device)
            x2 = x2.to(args.device)
            y = y.to(args.device).flatten().long()
            outputs = model(x1, x2)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            pred = torch.argmax(torch.softmax(outputs, dim=1), dim=-1)
            all_predictions.extend(pred.flatten().tolist())
            all_targets.extend(y.flatten().tolist())

    avg_loss = total_loss / max(len(dataloader), 1)
    val_f1 = f1_score(all_targets, all_predictions, average="macro")
    return avg_loss, val_f1


def train_warmup_client(client_id, args, global_model, criterion, dataset, run_timestamp):
    client = Fed_Avg_client(
        args,
        criterion,
        None,
        dataset,
        client_id=client_id,
        run_timestamp=run_timestamp
    )
    client.model = copy.deepcopy(global_model)
    client.train()

    weights = copy.deepcopy(client.get_parameters())
    num_samples = client.result["sample"]
    result = copy.deepcopy(client.result)

    del client
    torch.cuda.empty_cache()
    gc.collect()
    return client_id, weights, num_samples, result


def train_dshar_client(
    client_id,
    args,
    global_model,
    teacher_model,
    dataset,
    split_map,
    run_timestamp
):
    client = Fed_DSHAR_client(
        args=args,
        model=copy.deepcopy(global_model),
        teacher_model=copy.deepcopy(teacher_model),
        dataset=dataset,
        client_id=client_id,
        run_timestamp=run_timestamp
    )

    clean_indices, noisy_indices = split_map[client_id]
    result = client.train(clean_indices=clean_indices, noisy_indices=noisy_indices)
    weights = copy.deepcopy(client.get_parameters())
    num_samples = result["sample"]

    del client
    torch.cuda.empty_cache()
    gc.collect()
    return client_id, weights, num_samples, result


if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        set_seed(int(args.seed))

    input_size, time_stamp = 100, 300
    if args.diff:
        noise_rates = random.sample([0.2, 0.2, 0.3, 0.3], 4)
    else:
        noise_rates = [args.noise_rate] * args.client_num

    assigned_clusters_dict, global_cluster_map = coordinate_sys_noise_clusters(
        args.client_num,
        args.vul,
        args.noise_type,
        model_type=args.model_type,
        n_clusters=args.n_clusters,
        seed=int(args.seed),
        data_dir=args.data_dir
    )

    train_ds = []
    for client_id in range(args.client_num):
        ds = gen_client_ds(
            args.model_type,
            client_id,
            args.vul,
            args.noise_type,
            noise_rates[client_id],
            args.random_noise,
            args.num_neigh,
            assigned_clusters=assigned_clusters_dict,
            global_cluster_map=global_cluster_map,
            n_clusters=args.n_clusters,
            seed=int(args.seed),
            data_dir=args.data_dir
        )
        train_ds.append(ds)

    if args.model_type == "CBGRU":
        global_model = ClassiFilerNet(input_size, time_stamp)
    else:
        global_model = CGEVariant()
    global_model = global_model.to(args.device)

    criterion = nn.CrossEntropyLoss()
    server = Server(args, global_model, args.device, criterion)
    valid_dl = gen_valid_dl(args.model_type, args.vul, data_dir=args.data_dir)
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")

    print(">>> FedDSHAR Warmup Stage")
    for epoch in range(args.dshar_warmup_epoch):
        print(f"Warmup Epoch {epoch}")
        server.initialize_epoch_updates(epoch)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for client_id in range(args.client_num):
                futures.append(executor.submit(
                    train_warmup_client,
                    client_id,
                    args,
                    server.global_model,
                    criterion,
                    train_ds[client_id],
                    run_timestamp
                ))
            for future in futures:
                client_id, weights, num_samples, result = future.result()
                server.save_train_updates(weights, num_samples, result)
                print(f"warmup client:{client_id} loss:{result['loss']}")
        server.average_weights()

    teacher_model = copy.deepcopy(server.global_model).to(args.device)
    teacher_model.eval()
    early_stop_patience = int(getattr(args, "dshar_early_stop_patience", 10))
    early_stop_min_delta = float(getattr(args, "dshar_early_stop_min_delta", 1e-4))
    best_val_f1 = -1.0
    best_epoch = -1
    no_improve_rounds = 0
    best_global_state = copy.deepcopy(server.global_model.state_dict())

    split_map = {}
    print(">>> FedDSHAR Dual-Strategy Stage")
    for epoch in range(args.epoch):
        server.initialize_epoch_updates(epoch)

        if epoch == 0 or (epoch % max(args.dshar_mr_refresh_interval, 1) == 0):
            split_map = {}
            for client_id in range(args.client_num):
                clean_idx, noisy_idx = split_client_dataset_by_mr(train_ds[client_id], teacher_model, args)
                split_map[client_id] = (clean_idx, noisy_idx)
                print(
                    f"epoch:{epoch} client:{client_id} split clean={len(clean_idx)} noisy={len(noisy_idx)}"
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for client_id in range(args.client_num):
                futures.append(executor.submit(
                    train_dshar_client,
                    client_id,
                    args,
                    server.global_model,
                    teacher_model,
                    train_ds[client_id],
                    split_map,
                    run_timestamp
                ))
            for future in futures:
                client_id, weights, num_samples, result = future.result()
                server.save_train_updates(weights, num_samples, result)
                print(
                    "client:{0} loss:{1:.6f} clean:{2} noisy:{3} pseudo:{4}".format(
                        client_id,
                        result["loss"],
                        result["clean_samples"],
                        result["noisy_samples"],
                        result["pseudo_selected"]
                    )
                )

        server.average_weights()
        update_teacher_model(teacher_model, server.global_model, args.dshar_ema_beta)

        val_loss, val_f1 = evaluate_model(server.global_model, valid_dl, criterion, args)
        print(f"[VALID] epoch:{epoch} loss:{val_loss:.6f} f1:{val_f1:.6f}")

        if val_f1 > (best_val_f1 + early_stop_min_delta):
            best_val_f1 = val_f1
            best_epoch = epoch
            no_improve_rounds = 0
            best_global_state = copy.deepcopy(server.global_model.state_dict())
            print(f"[EARLY_STOP] improved at epoch {epoch}, best_f1={best_val_f1:.6f}")
        else:
            no_improve_rounds += 1
            print(f"[EARLY_STOP] no improvement rounds: {no_improve_rounds}/{early_stop_patience}")
            if early_stop_patience > 0 and no_improve_rounds >= early_stop_patience:
                print(f"[EARLY_STOP] triggered at epoch {epoch}, restore best epoch {best_epoch}")
                break

    if best_epoch >= 0:
        server.global_model.load_state_dict(best_global_state)
        print(f"[EARLY_STOP] best model restored from epoch {best_epoch} (best_f1={best_val_f1:.6f})")

    global_test(server.global_model, valid_dl, criterion, args, args.lab_name, run_timestamp=run_timestamp)
