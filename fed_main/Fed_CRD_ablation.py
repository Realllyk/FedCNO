import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import concurrent.futures
import copy
import gc
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn

from data_processing.dataloader_manager import gen_crd_ds, gen_test_dl, gen_valid_dl
from data_processing.preprocessing import coordinate_sys_noise_clusters
from global_test import global_test
from models.CGE_Variants import CGEVariant
from models.ClassiFilerNet import ClassiFilerNet
from options import parse_args
from trainers.client import Fed_CRD_client
from trainers.client_ablate import Fed_CRD_client_NoAmb
from trainers.server import CRD_server
from trainers.server_ablate import CRD_server_NoCal_FedAvg, CRD_server_NoClip, CRD_server_OnlyClip


def sanitize_tag(text):
    if text is None:
        return "exp"
    clean = re.sub(r"[^A-Za-z0-9_\-]", "_", str(text).strip())
    clean = re.sub(r"_+", "_", clean)
    return clean or "exp"


def build_client(args):
    mode = str(getattr(args, "crd_ablation_mode", "full")).lower()
    if mode == "no_amb":
        return Fed_CRD_client_NoAmb
    return Fed_CRD_client


def build_server(args):
    mode = str(getattr(args, "crd_ablation_mode", "full")).lower()
    if mode in ("no_cal", "fedavg"):
        return CRD_server_NoCal_FedAvg
    if mode == "no_clip":
        return CRD_server_NoClip
    if mode == "only_clip":
        return CRD_server_OnlyClip
    return CRD_server


def train_crd_client(client_id, client, global_model, ema_model):
    client.model = copy.deepcopy(global_model)

    q_anchor = getattr(client.args, "crd_q_anchor", "global")
    anchor_model_copy = None
    if q_anchor == "ema":
        anchor_model_copy = copy.deepcopy(ema_model)
        anchor_model_copy.eval()
        for param in anchor_model_copy.parameters():
            param.requires_grad = False
        anchor_model = anchor_model_copy
    else:
        anchor_model = global_model

    client.train()

    local_params = client.get_parameters()
    global_params = global_model.state_dict()
    delta = {}
    for k in local_params.keys():
        delta[k] = local_params[k] - global_params[k]

    q_k, num_samples = client.get_consistency_stats(anchor_model)
    result = client.result
    loss = result.get("loss", 0.0)

    if anchor_model_copy is not None:
        del anchor_model_copy
    torch.cuda.empty_cache()
    gc.collect()

    return client_id, delta, q_k, num_samples, result, loss


if __name__ == "__main__":
    args = parse_args()
    input_size, time_steps = 100, 300

    mode = str(getattr(args, "crd_ablation_mode", "full")).lower()
    noamb_variant = str(getattr(args, "crd_noamb_variant", "soft")).lower()
    exp_tag = sanitize_tag(getattr(args, "exp_tag", "crd_abl"))
    seed = int(args.seed) if args.seed is not None else None

    print(
        f"Starting FedCRD Ablation | mode={mode} | noamb_variant={noamb_variant} | "
        f"vul={args.vul} | noise={args.noise_type}({args.noise_rate})"
    )
    print(f"Training on device: {args.device}")

    if args.diff is True:
        noise_rates = random.sample([0.2, 0.2, 0.3, 0.3], 4)
    else:
        noise_rates = [args.noise_rate] * args.client_num

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    assigned_clusters_dict, global_cluster_map = coordinate_sys_noise_clusters(
        args.client_num,
        args.vul,
        args.noise_type,
        model_type=args.model_type,
        n_clusters=args.n_clusters,
        seed=seed,
        data_dir=args.data_dir,
    )

    print("Generating Datasets...")
    train_ds = []
    for i in range(args.client_num):
        ds = gen_crd_ds(
            i,
            args.vul,
            args.noise_type,
            noise_rates[i],
            args.random_noise,
            args.num_neigh,
            args.model_type,
            assigned_clusters=assigned_clusters_dict,
            global_cluster_map=global_cluster_map,
            n_clusters=args.n_clusters,
            seed=seed,
            data_dir=args.data_dir,
        )
        train_ds.append(ds)

    if args.vul == "reentrancy":
        class_weights = torch.tensor([1.0, 1.0]).to(args.device)
    elif args.vul == "timestamp":
        class_weights = torch.tensor([1.2, 1.5]).to(args.device)
    else:
        class_weights = torch.tensor([1.0, 1.0]).to(args.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if args.model_type == "CBGRU":
        global_model = ClassiFilerNet(input_size, time_steps)
    elif args.model_type == "CGE":
        global_model = CGEVariant()
    else:
        raise ValueError(f"Unsupported model_type={args.model_type} for FedCRD ablation.")
    global_model = global_model.to(args.device)

    raw_timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_timestamp = f"{exp_tag}_{mode}_seed{seed}_{raw_timestamp}"
    if mode == "no_amb":
        run_timestamp = f"{exp_tag}_{mode}_{noamb_variant}_seed{seed}_{raw_timestamp}"

    server_cls = build_server(args)
    client_cls = build_client(args)
    server = server_cls(args, global_model, args.device, criterion)

    print(f"Using server={server_cls.__name__}, client={client_cls.__name__}, run={run_timestamp}")

    test_dl = gen_test_dl(args.model_type, args.vul, data_dir=args.data_dir)
    valid_dl = gen_valid_dl(args.model_type, args.vul, data_dir=args.data_dir)

    print("Initializing Clients...")
    clients = []
    for i in range(args.client_num):
        client = client_cls(
            args,
            criterion,
            copy.deepcopy(server.global_model),
            train_ds[i],
            i,
            run_timestamp=run_timestamp,
        )
        client.init_knn_neighborhood()
        clients.append(client)

    print("Initialization Complete. Starting Training...")

    for epoch in range(args.epoch):
        print(f"\n--- Epoch {epoch} ---")
        server.initialize_epoch_updates(epoch)
        updates_list = []

        max_workers = int(getattr(args, "num_workers", 4))
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for client_id in range(args.client_num):
                futures.append(
                    executor.submit(
                        train_crd_client,
                        client_id,
                        clients[client_id],
                        server.global_model,
                        server.ema_model,
                    )
                )

            for future in futures:
                client_id, delta, q_k, num_samples, result, loss = future.result()
                updates_list.append((client_id, delta, q_k, num_samples))
                print(f"Client {client_id}: Loss={loss:.4f}, Reliability(q_k)={q_k:.4f}")
                server.save_train_updates(delta, num_samples, result)

        server.aggregate(updates_list)

        if epoch % 5 == 0 or epoch == args.epoch - 1:
            global_test(
                server.global_model,
                valid_dl,
                criterion,
                args,
                f"Fed_CRD_ablation_{mode}_{args.vul}",
                run_timestamp=run_timestamp,
                save_result=True,
                tag="valid",
                epoch=epoch,
            )

    print("\n--- Final Testing ---")
    global_test(
        server.global_model,
        test_dl,
        criterion,
        args,
        f"Fed_CRD_ablation_{mode}_{args.vul}_Final",
        run_timestamp=run_timestamp,
        save_result=True,
        tag="test",
    )
