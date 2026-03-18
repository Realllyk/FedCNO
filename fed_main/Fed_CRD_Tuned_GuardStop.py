import sys
import os
# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy 
import gc
import numpy as np
import torch
import torch.nn as nn
from options import parse_args
from data_processing.dataloader_manager import gen_crd_ds, gen_test_dl, gen_valid_dl
from data_processing.preprocessing import coordinate_sys_noise_clusters
from models.model_factory import build_model
from trainers.server import CRD_server
from trainers.client import Fed_CRD_client, Fed_Avg_client
from global_test import global_test
import random
import time
import concurrent.futures
import json
from pathlib import Path
from sklearn.metrics import confusion_matrix
from utils.crd_diag_logger import CRDDiagLogger


def _env_flag(name, default=False):
    raw = os.environ.get(name, None)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name, default):
    raw = os.environ.get(name, None)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_float(name, default):
    raw = os.environ.get(name, None)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def train_warmup_client(client_id, args, global_model, criterion, dataset, run_timestamp):
    """
    Train one client for warm-up stage (FedAvg style).
    """
    warmup_args = copy.deepcopy(args)
    warmup_args.lab_name = 'Fed_Avg'
    client = Fed_Avg_client(
        warmup_args,
        criterion,
        None,
        dataset,
        client_id=client_id,
        run_timestamp=run_timestamp
    )
    client.model = copy.deepcopy(global_model)
    client.train()

    weights = copy.deepcopy(client.get_parameters())
    num_samples = client.result['sample']
    result = client.result

    del client
    torch.cuda.empty_cache()
    gc.collect()

    return client_id, weights, num_samples, result


def train_crd_client(client_id, client, global_model, ema_model, criterion):
    """
    Train one client for FedCRD
    """
    # 1. Update local model with global parameters (for training)
    # Important: Deepcopy to ensure independent training
    client.model = copy.deepcopy(global_model)
    
    # Select anchor model for q_k calculation.
    # Default is global_model to align with the thesis algorithm description.
    q_anchor = getattr(client.args, 'crd_q_anchor', 'global')
    anchor_model_copy = None
    if q_anchor == 'ema':
        anchor_model_copy = copy.deepcopy(ema_model)
        anchor_model_copy.eval()
        for param in anchor_model_copy.parameters():
            param.requires_grad = False
        anchor_model = anchor_model_copy
    else:
        anchor_model = global_model
        
    # 2. Local Training
    # This updates client.model to theta_k^t
    client.train()
    
    # 3. Compute Update Delta and Reliability
    local_params = client.get_parameters()
    # Delta is still computed against the *current* global model (what we started with)
    # because we want to know the update direction relative to theta^t
    global_params = global_model.state_dict()
    
    # Calculate delta = theta_k^t - theta^t
    delta = {}
    for k in local_params.keys():
        delta[k] = local_params[k] - global_params[k]
        
    # Calculate Reliability q_k^t using selected anchor model
    q_k, num_samples = client.get_consistency_stats(anchor_model)
    
    result = client.result
    loss = result.get('loss', 0.0)
    
    # Clean up
    if anchor_model_copy is not None:
        del anchor_model_copy
    torch.cuda.empty_cache()
    gc.collect()
    
    return client_id, delta, q_k, num_samples, result, loss


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
    if len(batch) >= 4:
        return batch[0], batch[1], batch[2]
    raise ValueError(f"Unexpected batch length={len(batch)} in threshold eval")


def _safe_div(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _binary_eval_with_threshold(model, dataloader, criterion, args, threshold):
    """
    Evaluate binary classifier by thresholding positive-class probability.
    """
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            x1, x2, y = _unpack_batch(batch)
            x1 = _move_to_device(x1, args.device)
            x2 = _move_to_device(x2, args.device)
            y = _move_to_device(y, args.device).flatten().long()

            logits = model(x1, x2)
            loss = criterion(logits, y)
            total_loss += float(loss.item())

            probs = torch.softmax(logits, dim=1)
            pos_prob = probs[:, 1]
            pred = (pos_prob >= threshold).long()
            all_predictions.extend(pred.flatten().tolist())
            all_targets.extend(y.flatten().tolist())

    tn, fp, fn, tp = confusion_matrix(all_targets, all_predictions, labels=[0, 1]).ravel()
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    avg_loss = _safe_div(total_loss, len(dataloader))
    return {
        "threshold": float(threshold),
        "Averge Loss": float(avg_loss),
        "Accuracy": float(accuracy),
        "False positive rate(FPR)": float(_safe_div(fp, fp + tn)),
        "False negative rate(FNR)": float(_safe_div(fn, fn + tp)),
        "Recall(TPR)": float(recall),
        "Precision": float(precision),
        "F1 score": float(f1),
    }


def _parse_threshold_grid(args):
    """
    Parse threshold list from options.py argument --binary_threshold_grid.
    """
    raw = str(getattr(args, "binary_threshold_grid", "0.5"))
    thresholds = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        val = float(part)
        thresholds.append(min(0.99, max(0.01, val)))
    if not thresholds:
        thresholds = [0.5]
    # Keep deterministic ordering and de-dup for reproducibility.
    return sorted(set(thresholds))


def _save_threshold_result(args, run_timestamp, result_dict):
    """
    Save threshold-tuned test result without touching existing result schema.
    """
    noise_rate_str = str(args.noise_rate)
    result_path = Path(os.path.realpath(__file__)).parents[1].joinpath(
        "graduate_final_result",
        args.lab_name,
        args.model_type,
        args.noise_type,
        noise_rate_str
    )
    Path.mkdir(result_path, parents=True, exist_ok=True)

    base_file_name = f"{args.vul}_result_threshold.json"
    if args.noise_type == "fn_noise":
        base_file_name = f"fn_{base_file_name}"
    elif args.noise_type == "diff_noise":
        base_file_name = f"diff_{base_file_name}"
    elif args.noise_type == "sys_noise":
        base_file_name = f"sys_{base_file_name}"

    file_path = result_path.joinpath(base_file_name)
    if file_path.exists():
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                data = [data]
        except Exception:
            data = []
    else:
        data = []

    payload = {
        "tag": "test_threshold",
        "time": run_timestamp,
        "hparams": {k: v for k, v in vars(args).items()},
    }
    payload.update(result_dict)
    data.append(payload)
    file_path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")


if __name__ == '__main__':
    args = parse_args()
    if args.model_type == "MANDO" and args.vul != "tod":
        raise ValueError("MANDO only supports --vul tod in this project.")
    input_size, time_steps = 100, 300
    
    print(f"Starting FedCRD-Tuned with {args.vul}, Noise: {args.noise_type} ({args.noise_rate})")
    print(f"Training on device: {args.device}")
    print(
        f"[TUNED_CONFIG] lambda_q={float(getattr(args, 'lambda_q', 2.0))}, "
        f"lambda_agg={float(getattr(args, 'lambda_agg', 2.0))}, "
        f"workers={args.num_workers}, "
        f"crd_valid_interval={max(1, int(getattr(args, 'crd_valid_interval', 1)))}, "
        f"crd_patience={int(getattr(args, 'crd_early_stop_patience', 0))}, "
        f"crd_min_delta={float(getattr(args, 'crd_early_stop_min_delta', 1e-4))}"
    )

    # Guard-stop rule (env driven) to avoid premature manual interruption:
    # stop only when valid has stagnated and optimization signals flatten.
    guard_enabled = _env_flag("CRD_GUARD_ENABLED", False)
    guard_min_epoch = _env_int("CRD_GUARD_MIN_EPOCH", 12)
    guard_no_improve_rounds = _env_int("CRD_GUARD_NO_IMPROVE", 8)
    guard_window = max(2, _env_int("CRD_GUARD_WINDOW", 5))
    guard_min_loss_drop = _env_float("CRD_GUARD_MIN_LOSS_DROP", 0.003)
    guard_min_omega_spread_change = _env_float("CRD_GUARD_MIN_OMEGA_SPREAD_CHANGE", 0.003)
    print(
        f"[CRD_GUARD] enabled={guard_enabled}, min_epoch={guard_min_epoch}, "
        f"no_improve={guard_no_improve_rounds}, window={guard_window}, "
        f"min_loss_drop={guard_min_loss_drop}, min_omega_change={guard_min_omega_spread_change}"
    )

    # Setup Random Seeds
    if args.diff == True:
        noise_rates = random.sample([0.2, 0.2, 0.3, 0.3], 4)
    else:
        noise_rates = [args.noise_rate] * 4

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(int(args.seed))
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Systemic Noise Coordination
    assigned_clusters_dict, global_cluster_map = coordinate_sys_noise_clusters(
        args.client_num, 
        args.vul, 
        args.noise_type, 
        model_type=args.model_type,
        n_clusters=args.n_clusters, 
        seed=int(args.seed),
        data_dir=args.data_dir
    )

    # Data Loading
    print("Generating Datasets...")
    train_ds = list()
    for i in range(args.client_num):
        ds = gen_crd_ds(
            i, 
            args.vul, 
            args.noise_type, 
            noise_rates[i], # Use client-specific noise rate
                        args.num_neigh, 
            args.model_type, 
            assigned_clusters=assigned_clusters_dict, 
            global_cluster_map=global_cluster_map,
            n_clusters=args.n_clusters,
            seed=int(args.seed),
            data_dir=args.data_dir
        )
        train_ds.append(ds)

    # Model & Server Init
    if args.vul == 'reentrancy':
        class_weights = torch.tensor([1.0, 1.0]).to(args.device)
    elif args.vul == 'timestamp':
        class_weights = torch.tensor([1.2, 1.5]).to(args.device)
    else:
        class_weights = torch.tensor([1.0, 1.0]).to(args.device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    global_model = build_model(args, input_size, time_steps)
    global_model = global_model.to(args.device)
    
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    server = CRD_server(
        args,
        global_model,
        args.device,
        criterion
    )
    run_tag = os.path.join(
        args.vul,
        args.model_type,
        args.noise_type,
        f"rate_{args.noise_rate}",
        f"seed_{int(args.seed)}"
    )
    if getattr(args, "save_crd_diag", False):
        server.diag_logger = CRDDiagLogger(
            log_dir=os.path.join("result", "crd_rq1_diag"),
            run_tag=run_tag
        )
    else:
        server.diag_logger = None
    server.diag_meta = {
        "noise_type": args.noise_type,
        "noise_rate": args.noise_rate,
        "vul": args.vul,
        "model_type": args.model_type,
        "seed": int(args.seed),
    }

    test_dl = gen_test_dl(args.model_type, args.vul, data_dir=args.data_dir)
    valid_dl = gen_valid_dl(args.model_type, args.vul, data_dir=args.data_dir)

    # Warm-up Stage (reference: Fed_LGV.py)
    warmup_valid_interval = max(1, int(getattr(args, "warmup_valid_interval", 1)))
    warmup_early_stop_patience = int(getattr(args, "warmup_early_stop_patience", 0))
    warmup_early_stop_min_delta = float(getattr(args, "warmup_early_stop_min_delta", 1e-4))
    warmup_best_val_f1 = -1.0
    warmup_best_epoch = -1
    warmup_no_improve_rounds = 0
    warmup_best_global_state = copy.deepcopy(server.global_model.state_dict())
    original_lab_name = args.lab_name
    args.lab_name = 'Fed_Avg'

    for epoch in range(args.warm_up_epoch):
        print(f"Warm Up Epoch {epoch}: ")
        server.initialize_epoch_updates(epoch)

        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            for client_id in range(args.client_num):
                futures.append(
                    executor.submit(
                        train_warmup_client,
                        client_id,
                        args,
                        server.global_model,
                        criterion,
                        train_ds[client_id],
                        run_timestamp
                    )
                )

            for future in futures:
                client_id, weights, num_samples, result = future.result()
                server.save_train_updates(weights, num_samples, result)
                print(f"warmup client:{client_id}")
                print(f"loss is {result['loss']}")

        server.average_weights()

        if epoch % warmup_valid_interval == 0:
            print(f"\n--- WarmUp Validation at Epoch {epoch} ---")
            warmup_valid_result = global_test(
                server.global_model,
                valid_dl,
                criterion,
                args,
                "WarmUp_FedAvg_CRD",
                run_timestamp=run_timestamp,
                save_result=False,
                tag='valid',
                epoch=epoch
            )
            current_warmup_f1 = warmup_valid_result['F1 score']
            if current_warmup_f1 > (warmup_best_val_f1 + warmup_early_stop_min_delta):
                warmup_best_val_f1 = current_warmup_f1
                warmup_best_epoch = epoch
                warmup_no_improve_rounds = 0
                warmup_best_global_state = copy.deepcopy(server.global_model.state_dict())
                print(f"[WARMUP_EARLY_STOP] improved at epoch {epoch}, best_f1={warmup_best_val_f1:.6f}")
            else:
                warmup_no_improve_rounds += 1
                print(f"[WARMUP_EARLY_STOP] no improvement rounds: {warmup_no_improve_rounds}/{warmup_early_stop_patience}")
                if warmup_early_stop_patience > 0 and warmup_no_improve_rounds >= warmup_early_stop_patience:
                    print(f"[WARMUP_EARLY_STOP] triggered at epoch {epoch}, restoring best epoch {warmup_best_epoch}")
                    break
            print("-----------------------------------------\n")

    if warmup_best_epoch >= 0:
        server.global_model.load_state_dict(warmup_best_global_state)
        print(f"[WARMUP_EARLY_STOP] best warmup model restored from epoch {warmup_best_epoch} (best_f1={warmup_best_val_f1:.6f})")
    else:
        print("[WARMUP_EARLY_STOP] no warmup validation checkpoint captured, using final warmup model.")

    args.lab_name = original_lab_name
    print("-----------------------------------------------------------\n")
    if args.exit_after_warmup_test:
        print("[CONTROL] --exit_after_warmup_test enabled, exiting after warm-up stage.")
        sys.exit(0)

    # Keep EMA anchor consistent with warm-up updated global model.
    server.ema_model.load_state_dict(copy.deepcopy(server.global_model.state_dict()))

    # Client Init
    print("Initializing Clients...")
    clients = []
    for i in range(args.client_num):
        client = Fed_CRD_client(
            args,
            criterion, 
            copy.deepcopy(server.global_model),
            train_ds[i],
            i,
            run_timestamp=run_timestamp
        )
        # Initialize Static KNN Neighborhood
        client.init_knn_neighborhood()
        clients.append(client)
        
    print("Initialization Complete. Starting Training...")

    # CRD stage early-stop (FedCRD-specific args)
    crd_valid_interval = max(1, int(getattr(args, "crd_valid_interval", 1)))
    crd_early_stop_patience = int(getattr(args, "crd_early_stop_patience", 0))
    crd_early_stop_min_delta = float(getattr(args, "crd_early_stop_min_delta", 1e-4))
    crd_best_val_f1 = -1.0
    crd_best_epoch = -1
    crd_no_improve_rounds = 0
    crd_best_global_state = copy.deepcopy(server.global_model.state_dict())
    crd_epoch_avg_loss_hist = []
    crd_omega_spread_hist = []

    # Training Loop
    for epoch in range(args.epoch):
        print(f"\n--- Epoch {epoch} ---")
        server.initialize_epoch_updates(epoch) 

        updates_list = [] # Store (client_id, delta, q_k, n_k)
        epoch_client_losses = []
        
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            for client_id in range(args.client_num):
                futures.append(executor.submit(train_crd_client, client_id, clients[client_id], server.global_model, server.ema_model, criterion))
            
            for future in futures:
                client_id, delta, q_k, num_samples, result, loss = future.result()
                updates_list.append((client_id, delta, q_k, num_samples))
                epoch_client_losses.append(float(loss))
                
                print(f"Client {client_id}: Loss={loss:.4f}, Reliability(q_k)={q_k:.4f}")
                
                # Log to server for record keeping (optional)
                server.save_train_updates(delta, num_samples, result)
                
        # Server Aggregation (FedCRD Logic)
        server.aggregate(updates_list)
        if getattr(server, 'last_client_stats', None):
            print("[CRD_RELIABILITY] client-wise q_k / r_tilde / omega")
            for stat in sorted(server.last_client_stats, key=lambda x: x.get('client_id', -1)):
                print(
                    f"  client={int(stat['client_id'])} "
                    f"q_k={float(stat['q_k_t']):.4f} "
                    f"r_tilde={float(stat['r_tilde_k_t']):.4f} "
                    f"omega={float(stat['omega_k_t']):.4f}"
                )
        if epoch_client_losses:
            epoch_avg_loss = float(np.mean(epoch_client_losses))
            crd_epoch_avg_loss_hist.append(epoch_avg_loss)
            print(f"[CRD_LOSS] epoch={epoch} avg_client_loss={epoch_avg_loss:.6f}")
        if getattr(server, 'last_client_stats', None):
            omegas = [float(s.get('omega_k_t', 0.0)) for s in server.last_client_stats]
            omega_spread = (max(omegas) - min(omegas)) if omegas else 0.0
            crd_omega_spread_hist.append(float(omega_spread))
            print(f"[CRD_RELIABILITY] epoch={epoch} omega_spread={omega_spread:.6f}")
        
        # Validation
        if epoch % crd_valid_interval == 0 or epoch == args.epoch - 1:
             print(f"Validation at Epoch {epoch}...")
             crd_valid_result = global_test(
                server.global_model, 
                valid_dl, 
                criterion, 
                args, 
                f"Fed_CRD_{args.vul}", 
                run_timestamp=run_timestamp,
                save_result=True,
                tag='valid',
                epoch=epoch
            )
             current_crd_f1 = crd_valid_result.get('F1 score', float('nan'))
             if current_crd_f1 is not None and not np.isnan(current_crd_f1):
                 if current_crd_f1 > (crd_best_val_f1 + crd_early_stop_min_delta):
                     crd_best_val_f1 = current_crd_f1
                     crd_best_epoch = epoch
                     crd_no_improve_rounds = 0
                     crd_best_global_state = copy.deepcopy(server.global_model.state_dict())
                     print(f"[CRD_EARLY_STOP] improved at epoch {epoch}, best_f1={crd_best_val_f1:.6f}")
                 else:
                     crd_no_improve_rounds += 1
                     print(
                         f"[CRD_EARLY_STOP] no improvement rounds: "
                         f"{crd_no_improve_rounds}/{crd_early_stop_patience}"
                     )

                     # Guard-stop: valid stagnation + weak optimization signal + stable reliability spread.
                     if (
                         guard_enabled
                         and epoch >= guard_min_epoch
                         and crd_no_improve_rounds >= guard_no_improve_rounds
                         and len(crd_epoch_avg_loss_hist) >= guard_window
                         and len(crd_omega_spread_hist) >= guard_window
                     ):
                         recent_loss_drop = crd_epoch_avg_loss_hist[-guard_window] - crd_epoch_avg_loss_hist[-1]
                         recent_omega_change = abs(
                             crd_omega_spread_hist[-1] - crd_omega_spread_hist[-guard_window]
                         )
                         print(
                             f"[CRD_GUARD] loss_drop({guard_window})={recent_loss_drop:.6f}, "
                             f"omega_change({guard_window})={recent_omega_change:.6f}"
                         )
                         if (
                             recent_loss_drop < guard_min_loss_drop
                             and recent_omega_change < guard_min_omega_spread_change
                         ):
                             print(
                                 f"[CRD_GUARD] triggered at epoch {epoch}: "
                                 f"valid stagnation + low loss gain + low reliability change."
                             )
                             break

                     if crd_early_stop_patience > 0 and crd_no_improve_rounds >= crd_early_stop_patience:
                         print(
                             f"[CRD_EARLY_STOP] triggered at epoch {epoch}, "
                             f"restoring best epoch {crd_best_epoch}"
                         )
                         break

    # Keep both checkpoints for comparison:
    # - last_epoch model (before restore)
    # - best_valid model (after restore)
    crd_last_global_state = copy.deepcopy(server.global_model.state_dict())

    print("\n--- Final Testing (Last Epoch) ---")
    server.global_model.load_state_dict(crd_last_global_state)
    global_test(
        server.global_model,
        test_dl,
        criterion,
        args,
        f"Fed_CRD_Tuned",
        run_timestamp=run_timestamp,
        tag='test_last',
        extra_info={
            'checkpoint': 'last_epoch',
            'best_epoch': int(crd_best_epoch),
            'best_valid_f1': float(crd_best_val_f1),
        }
    )

    if crd_best_epoch >= 0:
        server.global_model.load_state_dict(crd_best_global_state)
        print(
            f"[CRD_EARLY_STOP] best CRD model restored from epoch {crd_best_epoch} "
            f"(best_f1={crd_best_val_f1:.6f})"
        )
    else:
        print("[CRD_EARLY_STOP] no CRD validation checkpoint captured, using final CRD model.")
            
    print("\n--- Final Testing (Best Valid Restore) ---")
    global_test(
        server.global_model,
        test_dl,
        criterion,
        args,
        f"Fed_CRD_Tuned",
        run_timestamp=run_timestamp,
        tag='test_best',
        extra_info={
            'checkpoint': 'best_valid_restore' if crd_best_epoch >= 0 else 'last_epoch_fallback',
            'best_epoch': int(crd_best_epoch),
            'best_valid_f1': float(crd_best_val_f1),
        }
    )

    # Optional threshold tuning:
    # 1) select best threshold on validation set by F1
    # 2) evaluate test set with selected threshold
    if getattr(args, "binary_threshold_tune", False):
        threshold_grid = _parse_threshold_grid(args)
        print(f"[THRESH_TUNE] threshold grid: {threshold_grid}")
        best_threshold = 0.5
        best_valid_result = None
        best_valid_f1 = -1.0
        for threshold in threshold_grid:
            valid_result = _binary_eval_with_threshold(
                server.global_model,
                valid_dl,
                criterion,
                args,
                threshold
            )
            if valid_result["F1 score"] > best_valid_f1:
                best_valid_f1 = valid_result["F1 score"]
                best_valid_result = valid_result
                best_threshold = threshold

        print(
            f"[THRESH_TUNE] selected threshold={best_threshold:.3f}, "
            f"valid_f1={best_valid_f1:.6f}, valid_precision={best_valid_result['Precision']:.6f}, "
            f"valid_recall={best_valid_result['Recall(TPR)']:.6f}"
        )

        threshold_test_result = _binary_eval_with_threshold(
            server.global_model,
            test_dl,
            criterion,
            args,
            best_threshold
        )
        print(
            f"[THRESH_TUNE][TEST] threshold={best_threshold:.3f}, "
            f"f1={threshold_test_result['F1 score']:.6f}, "
            f"precision={threshold_test_result['Precision']:.6f}, "
            f"recall={threshold_test_result['Recall(TPR)']:.6f}"
        )
        _save_threshold_result(args, run_timestamp, threshold_test_result)
