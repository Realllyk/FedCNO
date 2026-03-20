import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import gc
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from options import parse_args
from data_processing.dataloader_manager import gen_client_ds, gen_valid_dl
from data_processing.preprocessing import coordinate_sys_noise_clusters
from models.model_factory import build_model
from trainers.server import ARFL_Server
from trainers.client import Fed_ARFL_client
from global_test import global_test
import random
import time


if __name__ == '__main__':
    args = parse_args()
    if args.model_type == "MANDO" and args.vul != "tod":
        raise ValueError("MANDO only supports --vul tod in this project.")
    INPUT_SIZE, TIME_STAMP = 100, 300
    criterion = nn.CrossEntropyLoss()
    if args.device != "cpu":
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

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

    assigned_clusters_dict, global_cluster_map = coordinate_sys_noise_clusters(
        args.client_num, 
        args.vul, 
        args.noise_type, 
        model_type=args.model_type,
        n_clusters=args.n_clusters, 
        seed=int(args.seed),
        data_dir=args.data_dir
    )

    clients = list()
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

    for i in range(args.client_num):
        client = Fed_ARFL_client(
            args,
            criterion,
            None,
            train_ds[i],
            1.,
        )
        clients.append(client)
    total_num_samples = sum([c.num_train_samples for c in clients])

    global_model = build_model(args, INPUT_SIZE, TIME_STAMP)
    global_model = global_model.to(device)
    server = ARFL_Server(
        args,
        global_model,
        criterion,
        args.seed,
        clients,
        total_num_samples
    )
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    valid_interval = max(1, int(getattr(args, "arfl_valid_interval", getattr(args, "lgv_valid_interval", 1))))
    early_stop_patience = int(getattr(args, "arfl_early_stop_patience", getattr(args, "lgv_early_stop_patience", 15)))
    early_stop_min_delta = float(getattr(args, "arfl_early_stop_min_delta", getattr(args, "lgv_early_stop_min_delta", 1e-4)))
    best_val_f1 = -1.0
    best_epoch = -1
    no_improve_rounds = 0
    best_global_state = copy.deepcopy(server.global_model.state_dict())

    for c in clients:
        c.model = copy.deepcopy(global_model)
        c.test()

    valid_dl = gen_valid_dl(args.model_type, args.vul, data_dir=args.data_dir)
    for epoch in range(args.epoch):
        print(f"Epoch {epoch} Training:------------------")
        server.initialize_epoch_updates(epoch)
        server.sample_clients(epoch)

        for c in clients:
            if c.model != None:
                del c.model
            c.model = copy.deepcopy(server.global_model)
        
        for i, c in enumerate(server.selected_clients):
            c.train()
            print(f"Selected Client {i} Train Loss: {c.result['loss']}")
            
            # Clean up
            torch.cuda.empty_cache()
            gc.collect()

        server.average_weights()
        server.update_alpha()
        
        # Clean up after update_alpha (which calls test())
        torch.cuda.empty_cache()
        gc.collect()
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
                    break
            print("-------------------------------\n")
    
    if best_epoch >= 0:
        server.global_model.load_state_dict(best_global_state)
        print(f"[EARLY_STOP] best model restored from epoch {best_epoch} (best_f1={best_val_f1:.6f})")
    else:
        best_epoch = args.epoch - 1
        print("[EARLY_STOP] no validation checkpoint captured, using final epoch model.")
    
    # test_dl = gen_cbgru_valid_dl(args.vul, 0, args.batch)
    test_dl = gen_valid_dl(args.model_type, args.vul, data_dir=args.data_dir)
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



        

        

        


