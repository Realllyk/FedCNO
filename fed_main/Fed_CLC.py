import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import gc
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import concurrent.futures
import time
from options import parse_args
from data_processing.dataloader_manager import gen_cbgru_valid_dl, gen_client_ds, gen_valid_dl
from data_processing.preprocessing import coordinate_sys_noise_clusters
from trainers.server import CLC_Server
from trainers.client import Fed_CLC_client
from models.model_factory import build_model
from global_test import global_test
import random


def train_client_warmup(client, global_model):
    client.model = copy.deepcopy(global_model)
    client.train()
    
    # Move weights to CPU
    weights = client.get_parameters()
    weights = {k: v.cpu() for k, v in weights.items()}
    
    num_samples = client.result['sample']
    result = client.result
    
    # Clean up
    del client.model
    if hasattr(client, 'tb_writer'):
        client.tb_writer.close()
        
    torch.cuda.empty_cache()
    gc.collect()
    
    return client.client_id, weights, num_samples, result

def client_send_conf(client, global_model):
    client.model = copy.deepcopy(global_model)
    conf, classnum = client.sendconf()
    del client.model
    torch.cuda.empty_cache()
    gc.collect()
    return client.client_id, conf, classnum

def train_client_holdout(client, global_model, conf_score):
    client.model = copy.deepcopy(global_model)
    client.data_holdout(conf_score)
    # Ensure client uses the filtered dataloader
    if hasattr(client, 'data_loader'):
        client.dataloader = client.data_loader
    client.train()
    
    # Move weights to CPU
    weights = client.get_parameters()
    weights = {k: v.cpu() for k, v in weights.items()}
    
    num_samples = client.result['sample']
    result = client.result
    
    # Clean up
    del client.model
    if hasattr(client, 'avai_dataset'):
        del client.avai_dataset
        
    torch.cuda.empty_cache()
    gc.collect()
    
    return client.client_id, weights, num_samples, result

def prepare_client_correct(client, conf_score):
    client.data_holdout(conf_score)
    client.data_correct()
    
    # Clean up intermediate data if possible, though data_correct might set avai_dataset which is needed for training
    # We only clean what's strictly temporary
    if hasattr(client, 'sfm_Mat'):
        del client.sfm_Mat
    # torch.cuda.empty_cache()
    
    return client.client_id

def train_client_correct(client, global_model):
    client.model = copy.deepcopy(global_model)
    # Ensure client uses the corrected dataloader
    if hasattr(client, 'data_loader'):
        client.dataloader = client.data_loader
    client.train()
    
    # Move weights to CPU
    weights = client.get_parameters()
    weights = {k: v.cpu() for k, v in weights.items()}
    
    num_samples = client.result['sample']
    result = client.result
    
    # Clean up
    del client.model
    if hasattr(client, 'avai_dataset'):
        del client.avai_dataset
        
    # torch.cuda.empty_cache()
    # gc.collect()
    
    return client.client_id, weights, num_samples, result


if __name__ == "__main__":
    args = parse_args()
    if args.model_type == "MANDO" and args.vul != "tod":
        raise ValueError("MANDO only supports --vul tod in this project.")
    INPUT_SIZE, TIME_STAMP = 100, 300

    if args.diff == True:
        noise_rates = random.sample([0.2, 0.2, 0.3, 0.3], 4)
    else:
        noise_rates = [args.noise_rate] * 4
    
    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(int(args.seed))
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # -------------------------------------------------------------------------
    # 绯荤粺鎬у櫔澹板崗璋?(Systemic Noise Coordination)
    # -------------------------------------------------------------------------
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
        
        # DEBUG: Check if noise is applied
        if args.noise_type != 'non_noise':
            import pandas as pd
            if args.model_type == 'CBGRU':
                client_dir = os.path.join(args.data_dir, f"graduate_client_split/cbgru/{args.vul}/client_{i}/")
            elif args.model_type == 'CGE':
                client_dir = os.path.join(args.data_dir, f"graduate_client_split/cge/{args.vul}/client_{i}/")
            elif args.model_type == 'MANDO':
                client_dir = os.path.join(args.data_dir, f"graduate_client_split/mando/{args.vul}/client_{i}/")
            else:
                client_dir = os.path.join(args.data_dir, f"graduate_client_split/{args.vul}/client_{i}/")
            labels_path = os.path.join(client_dir, f"label_train.csv")
            if os.path.exists(labels_path):
                clean_labels = pd.read_csv(labels_path, header=None).iloc[:, 0].values
                noise_labels = np.array(ds.labels)
                diff = np.sum(clean_labels != noise_labels)
                print(f"[DEBUG] Client {i}: Noise Rate={noise_rates[i]}, Clean vs Noisy Diff={diff}/{len(clean_labels)} ({diff/len(clean_labels):.4f})")
            else:
                print(f"[DEBUG] Client {i}: Label file not found at {labels_path}")

    test_dl = gen_valid_dl(args.model_type, args.vul, data_dir=args.data_dir)
    valid_dl = test_dl
    
    criterion = nn.CrossEntropyLoss()
    
    global_model = build_model(args, INPUT_SIZE, TIME_STAMP)
    global_model = global_model.to(args.device)

    server = CLC_Server(args, global_model, args.device, criterion)
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")

    valid_interval = max(1, int(getattr(args, "clc_valid_interval", getattr(args, "lgv_valid_interval", 1))))
    early_stop_patience = int(getattr(args, "clc_early_stop_patience", getattr(args, "lgv_early_stop_patience", 15)))
    early_stop_min_delta = float(getattr(args, "clc_early_stop_min_delta", getattr(args, "lgv_early_stop_min_delta", 1e-4)))
    best_val_f1 = -1.0
    best_epoch = -1
    no_improve_rounds = 0
    best_global_state = copy.deepcopy(server.global_model.state_dict())
    early_stop_triggered = False
    
    clients = []
    tao = 0.1
    for i in range(args.client_num):
        client = Fed_CLC_client(
            args,
            criterion,
            copy.deepcopy(server.global_model),
            train_ds[i],
            i,
            tao
        )
        clients.append(client)

    # Warmup Stage
    print("Warmup Stage...")
    server.initialize_epoch_updates(-1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(train_client_warmup, clients[i], server.global_model) for i in range(args.client_num)]
        for future in concurrent.futures.as_completed(futures):
            cid, weights, num_samples, result = future.result()
            server.save_train_updates(weights, num_samples, result)
            print(f"client:{cid} warmup done")
            clients[cid].print_loss()
    server.average_weights()

    # Holdout Stage
    print("Holdout Stage...")
    for epoch in range(args.first_epochs):
        server.initialize_epoch_updates(epoch)
        
        confs = [None] * args.client_num
        classnums = [None] * args.client_num
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(client_send_conf, clients[i], server.global_model) for i in range(args.client_num)]
            for future in concurrent.futures.as_completed(futures):
                cid, conf, classnum = future.result()
                confs[cid] = conf
                classnums[cid] = classnum
        
        server.receiveconf(confs, classnums)
        print() # Clear the progress line
        conf_score = server.conf_agg()

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(train_client_holdout, clients[i], server.global_model, conf_score) for i in range(args.client_num)]
            for future in concurrent.futures.as_completed(futures):
                cid, weights, num_samples, result = future.result()
                server.save_train_updates(weights, num_samples, result)
                print(f"client:{cid} holdout epoch {epoch} done")
                clients[cid].print_loss()
        
        server.average_weights()
        if epoch % valid_interval == 0:
            print(f"\n--- Validation at Epoch {epoch} ---")
            valid_result = global_test(
                server.global_model,
                valid_dl,
                criterion,
                args,
                args.lab_name,
                run_timestamp=run_timestamp,
                save_result=True,
                tag='valid',
                epoch=epoch
            )
            current_val_f1 = valid_result['F1 score']
            if current_val_f1 > (best_val_f1 + early_stop_min_delta):
                best_val_f1 = current_val_f1
                best_epoch = epoch
                no_improve_rounds = 0
                best_global_state = copy.deepcopy(server.global_model.state_dict())
                print(f"[EARLY_STOP] improved at epoch {epoch}, best_f1={best_val_f1:.6f}")
            else:
                no_improve_rounds += 1
                print(f"[EARLY_STOP] no improvement rounds: {no_improve_rounds}/{early_stop_patience}")
                if early_stop_patience > 0 and no_improve_rounds >= early_stop_patience:
                    print(f"[EARLY_STOP] triggered at epoch {epoch}, restoring best epoch {best_epoch}")
                    early_stop_triggered = True
                    break
            print("-------------------------------\n")
    if early_stop_triggered:
        print("[EARLY_STOP] stop remaining training stages.")

    # Correct Stage
    print("Correct Stage...")
    correct_done = False
    if not early_stop_triggered:
        for epoch in range(args.first_epochs, args.first_epochs+args.last_epochs):
            server.initialize_epoch_updates(epoch)
        
            if not correct_done:
                confs = [None] * args.client_num
                classnums = [None] * args.client_num
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                    futures = [executor.submit(client_send_conf, clients[i], server.global_model) for i in range(args.client_num)]
                    for future in concurrent.futures.as_completed(futures):
                        cid, conf, classnum = future.result()
                        confs[cid] = conf
                        classnums[cid] = classnum
            
                server.receiveconf(confs, classnums)
                print() # Clear the progress line
                conf_score = server.conf_agg()

                with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                    futures = [executor.submit(prepare_client_correct, clients[i], conf_score) for i in range(args.client_num)]
                    for future in concurrent.futures.as_completed(futures):
                        pass
            
                correct_done = True
        
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [executor.submit(train_client_correct, clients[i], server.global_model) for i in range(args.client_num)]
                for future in concurrent.futures.as_completed(futures):
                    cid, weights, num_samples, result = future.result()
                    server.save_train_updates(weights, num_samples, result)
                    print(f"client:{cid} correct epoch {epoch} done")
                    clients[cid].print_loss()
        
            server.average_weights()
            if epoch % valid_interval == 0:
                print(f"\n--- Validation at Epoch {epoch} ---")
                valid_result = global_test(
                    server.global_model,
                    valid_dl,
                    criterion,
                    args,
                    args.lab_name,
                    run_timestamp=run_timestamp,
                    save_result=True,
                    tag='valid',
                    epoch=epoch
                )
                current_val_f1 = valid_result['F1 score']
                if current_val_f1 > (best_val_f1 + early_stop_min_delta):
                    best_val_f1 = current_val_f1
                    best_epoch = epoch
                    no_improve_rounds = 0
                    best_global_state = copy.deepcopy(server.global_model.state_dict())
                    print(f"[EARLY_STOP] improved at epoch {epoch}, best_f1={best_val_f1:.6f}")
                else:
                    no_improve_rounds += 1
                    print(f"[EARLY_STOP] no improvement rounds: {no_improve_rounds}/{early_stop_patience}")
                    if early_stop_patience > 0 and no_improve_rounds >= early_stop_patience:
                        print(f"[EARLY_STOP] triggered at epoch {epoch}, restoring best epoch {best_epoch}")
                        early_stop_triggered = True
                        break
                print("-------------------------------\n")
        if early_stop_triggered:
            print("[EARLY_STOP] training terminated in Correct Stage.")

    if best_epoch >= 0:
        server.global_model.load_state_dict(best_global_state)
        print(f"[EARLY_STOP] best model restored from epoch {best_epoch} (best_f1={best_val_f1:.6f})")
    else:
        best_epoch = args.first_epochs + args.last_epochs - 1
        print("[EARLY_STOP] no validation checkpoint captured, using final epoch model.")

    global_test(
        server.global_model,
        test_dl,
        criterion,
        args,
        args.lab_name,
        run_timestamp=run_timestamp,
        save_result=True,
        tag='test',
        epoch=best_epoch,
        extra_info={
            "best_valid_f1": best_val_f1,
            "early_stop_patience": early_stop_patience,
            "early_stop_min_delta": early_stop_min_delta,
            "valid_interval": valid_interval
        }
    )


