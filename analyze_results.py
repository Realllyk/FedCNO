import json
import os
import numpy as np

# Define paths
json_dir = r'd:\PythonProject\Fed_LGV\Fed_LGV\graduate_result\Fed_LGV\CBGRU\non_noise\0.3'
runs_root = r'd:\PythonProject\Fed_LGV\Fed_LGV\runs\Fed_LGV\CBGRU\non_noise\0.3'

reentrancy_json = os.path.join(json_dir, 'reentrancy_result.json')
timestamp_json = os.path.join(json_dir, 'timestamp_result.json')

# Helper to read JSON
def get_last_n_results(json_path, n=10):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return []
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data[-n:]

# Helper to analyze loss logs
def analyze_loss(time_str, vul_type):
    # Path construction: runs/Fed_LGV/CBGRU/non_noise/0.3/{vul}/{timestamp}/client_{id}/loss_log.txt
    base_log_dir = os.path.join(runs_root, vul_type, time_str)
    
    if not os.path.exists(base_log_dir):
        return 'Log dir not found: ' + base_log_dir
    
    client_losses = {}
    for client_id in range(4): # Assuming 4 clients
        log_file = os.path.join(base_log_dir, f'client_{client_id}', 'loss_log.txt')
        if os.path.exists(log_file):
            losses = []
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()[1:] # Skip header
                    for line in lines:
                        try:
                            # Format: Global_Step,Epoch,Batch_Loss
                            parts = line.strip().split(',')
                            if len(parts) >= 3:
                                loss = float(parts[2])
                                losses.append(loss)
                        except ValueError:
                            continue
            except Exception as e:
                print(f"Error reading {log_file}: {e}")
                
            if losses:
                client_losses[f'client_{client_id}'] = {
                    'initial': losses[0],
                    'final': losses[-1],
                    'min': np.min(losses),
                    'max': np.max(losses),
                    'mean': np.mean(losses),
                    'std': np.std(losses),
                    'trend': 'decreasing' if losses[-1] < losses[0] else 'increasing'
                }
    return client_losses

def print_analysis(title, results, vul_type):
    print(f'\n--- {title} Analysis ---')
    for res in results:
        time_str = res.get('time', 'Unknown')
        f1 = res.get('F1 score', 'N/A')
        acc = res.get('Accuracy', 'N/A')
        recall = res.get('Recall(TPR)', 'N/A')
        fpr = res.get('False positive rate(FPR)', 'N/A')
        
        print(f'Time: {time_str}')
        print(f'  Metrics: F1={f1}, Acc={acc}, Recall={recall}, FPR={fpr}')
        
        loss_analysis = analyze_loss(time_str, vul_type)
        if isinstance(loss_analysis, dict):
            if not loss_analysis:
                 print('  Logs: No client logs found (empty dict).')
            for client, stats in loss_analysis.items():
                 print(f"  {client}: Start={stats['initial']:.4f}, End={stats['final']:.4f}, Min={stats['min']:.4f}, Trend={stats['trend']}")
        else:
            print(f'  Logs: {loss_analysis}')

# Run analysis
reentrancy_results = get_last_n_results(reentrancy_json)
timestamp_results = get_last_n_results(timestamp_json)

print_analysis('Reentrancy', reentrancy_results, 'reentrancy')
print_analysis('Timestamp', timestamp_results, 'timestamp')
