import torch
import torch.nn as nn
import json
import os
import time
from pathlib import Path
from sklearn.metrics import confusion_matrix


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
    raise ValueError(f"Unexpected batch length={len(batch)} in global_test")


def _serialize_hparams(args):
    hparams = {}
    for k, v in vars(args).items():
        try:
            json.dumps(v)
            hparams[k] = v
        except TypeError:
            hparams[k] = str(v)
    return hparams


def _load_result_list(result_file_path):
    if not os.path.exists(result_file_path):
        return []
    try:
        with open(result_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, UnicodeDecodeError):
        corrupt_path = f"{result_file_path}.corrupt.{int(time.time())}"
        try:
            os.replace(result_file_path, corrupt_path)
            print(f"[WARN] Corrupted result json moved to: {corrupt_path}")
        except OSError:
            pass
        return []


def _atomic_dump_json(result_file_path, data):
    tmp_path = f"{result_file_path}.tmp"
    with open(tmp_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    os.replace(tmp_path, result_file_path)


def global_test(
    model,
    dataloader,
    criterion,
    args,
    method,
    reduction='mean',
    run_timestamp=None,
    save_result=True,
    tag='test',
    epoch=None,
    extra_info=None
):
    all_predictions = []
    all_targets = []
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x1, x2, y = _unpack_batch(batch)
            x1 = _move_to_device(x1, args.device)
            x2 = _move_to_device(x2, args.device)
            y = _move_to_device(y, args.device)
            y = y.flatten().long()
            outputs = model(x1, x2)

            loss = criterion(outputs, y)
            if reduction == 'none':
                loss = loss.mean()
            total_loss += loss.item()
            softmax = nn.Softmax(dim=1)
            pred = torch.argmax(softmax(outputs), dim=-1)
            all_predictions.extend(pred.flatten().tolist())
            all_targets.extend(y.flatten().tolist())
            
            torch.cuda.empty_cache()
    
    avg_loss = total_loss / len(dataloader)
    print(f"[{tag.upper()}] Averge Loss: {avg_loss}")
    tn, fp, fn, tp = confusion_matrix(all_targets, all_predictions).ravel()
    result_dict = dict()
    
    result_dict['tag'] = tag
    if epoch is not None:
        result_dict['epoch'] = epoch
    if extra_info is not None:
        result_dict['extra_info'] = extra_info

    # Add timestamp to result_dict if provided
    if run_timestamp:
        result_dict['time'] = run_timestamp
    else:
        # Fallback to current time if not provided, though run_timestamp is preferred for consistency
        result_dict['time'] = time.strftime("%Y%m%d_%H%M%S")
        
    result_dict['Accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    result_dict['False positive rate(FPR)'] = fp / (fp + tn)
    result_dict['False negative rate(FNR)'] = fn / (fn + tp)
    result_dict['Recall(TPR)'] = tp / (tp + fn)
    result_dict['Precision'] = tp / (tp + fp)
    result_dict['F1 score'] = (2 * result_dict['Precision'] * result_dict['Recall(TPR)']) / (result_dict['Precision'] + result_dict['Recall(TPR)'])
    result_dict['hparams'] = _serialize_hparams(args)
    
    print(f"[{tag.upper()}] Accuracy: ", result_dict['Accuracy'])
    print(f"[{tag.upper()}] False positive rate(FPR): ", result_dict['False positive rate(FPR)'])
    print(f"[{tag.upper()}] False negative rate(FNR): ", result_dict['False negative rate(FNR)'])
    print(f"[{tag.upper()}] Recall(TPR): ", result_dict['Recall(TPR)'])
    print(f"[{tag.upper()}] Precision: ", result_dict['Precision'])
    print(f"[{tag.upper()}] F1 score: ", result_dict['F1 score'])
    
    if not save_result:
        return result_dict

    # result_path = Path(os.path.realpath(__file__)).parents[0].joinpath(
    #     'merge_result',
    #     str(args.noise_rate),
    #     f"{method}_{args.cbgru_net1}_{args.cbgru_net2}",
    # )
    # Old result path logic (commented out)
    # result_path = Path(os.path.realpath(__file__)).parents[0].joinpath(
    #     '4_client_result',
    #     str(args.noise_rate),
    #     args.model_type,
    #     method,
    # )
    
    # New result path: graduate_result/labName/model_type/noise_type/noise_rate
    lab_name = getattr(args, 'lab_name', 'default_lab')
    noise_rate_str = str(args.noise_rate)
    
    # Handle pure noise type specifically if needed, otherwise use args.noise_type
    # Assuming 'pure' noise type might not use noise_rate, but keeping structure consistent
    current_noise_type = args.noise_type
    
    result_root = getattr(args, 'result_root', 'graduate_final_result')
    result_path = Path(os.path.realpath(__file__)).parents[0].joinpath(
        result_root,
        lab_name,
        args.model_type,
        current_noise_type,
        noise_rate_str
    )
    
    Path.mkdir(result_path, parents=True, exist_ok=True)
    
    # Filename construction based on vulnerability and validation fraction
    if args.valid_frac == 1.0:
        file_name = f'{args.vul}_result.json'
    else:
        file_name = f'{args.vul}_test_{args.valid_frac}_result.json'
        
    # Special handling for different noise types in filename (if still needed)
    # The previous logic prefixed filenames with 'fn_', 'diff_', 'sys_'.
    # Since we now separate by directory (noise_type), we might not strictly need the prefix,
    # but keeping it for clarity if the user wants to maintain file naming conventions within the folder.
    # However, the user request specified folder structure clearly, but not filename changes.
    # Let's keep the filename simple as requested implicitly by the folder structure, 
    # OR maintain the prefixes to avoid confusion if multiple runs end up in the same folder 
    # (though directory structure seems to separate them well).
    
    # Let's stick to the previous filename prefixes to be safe and consistent with previous logic,
    # just in case 'noise_type' argument doesn't capture everything or for backward compatibility of reading.
    if args.noise_type == 'fn_noise':
        file_name = f'fn_{file_name}'
    elif args.noise_type == 'diff_noise':
        file_name = f'diff_{file_name}'
    elif args.noise_type == 'sys_noise':
        file_name = f'sys_{file_name}'
    
    if tag == 'valid':
        file_name = file_name.replace('.json', '_valid.json')
    
    result_file_path = result_path.joinpath(file_name)
        
    data = _load_result_list(result_file_path)
    data.append(result_dict)
    _atomic_dump_json(result_file_path, data)

    return result_dict
            
            





    
