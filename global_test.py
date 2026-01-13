import torch
import torch.nn as nn
import json
import os
from pathlib import Path
from sklearn.metrics import confusion_matrix


def global_test(model, dataloader, criterion, args, method, reduction='mean'):
    all_predictions = []
    all_targets = []
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for x1, x2, y in dataloader:
            x1, x2, y = x1.to(args.device), x2.to(args.device), y.to(args.device)
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
    print(f"Averge Loss: {avg_loss}")
    tn, fp, fn, tp = confusion_matrix(all_targets, all_predictions).ravel()
    result_dict = dict()
    result_dict['Accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    result_dict['False positive rate(FPR)'] = fp / (fp + tn)
    result_dict['False negative rate(FNR)'] = fn / (fn + tp)
    result_dict['Recall(TPR)'] = tp / (tp + fn)
    result_dict['Precision'] = tp / (tp + fp)
    result_dict['F1 score'] = (2 * result_dict['Precision'] * result_dict['Recall(TPR)']) / (result_dict['Precision'] + result_dict['Recall(TPR)'])\
    
    # result_path = Path(os.path.realpath(__file__)).parents[0].joinpath(
    #     'merge_result',
    #     str(args.noise_rate),
    #     f"{method}_{args.cbgru_net1}_{args.cbgru_net2}",
    # )
    result_path = Path(os.path.realpath(__file__)).parents[0].joinpath(
        '4_client_result',
        str(args.noise_rate),
        args.model_type,
        method,
    )
    
    Path.mkdir(result_path, parents=True, exist_ok=True)
    if args.noise_type == 'non_noise':
        if args.valid_frac == 1.0:
            result_file_path = result_path.joinpath(f'{args.vul}_result.json')
        else:
            result_file_path = result_path.joinpath(f'{args.vul}_test_{args.valid_frac}_result.json')
    elif args.noise_type == 'fn_noise':
        if args.valid_frac == 1.0:
            result_file_path = result_path.joinpath(f'fn_{args.vul}_result.json')
        else:
            result_file_path = result_path.joinpath(f'fn_{args.vul}_test_{args.valid_frac}_result.json')
    elif args.noise_type == 'diff_noise':
        result_file_path = result_path.joinpath(f'diff_{args.vul}_result.json')
    elif args.noise_type == 'sys_noise':
        result_file_path = result_path.joinpath(f'sys_{args.vul}_result.json')
    else:
        # result_path = Path(os.path.realpath(__file__)).parents[0].joinpath(
        #     'merge_result',
        #     'pure',
        #     f"{method}_{args.cbgru_net1}_{args.cbgru_net2}",
        # )
        result_path = Path(os.path.realpath(__file__)).parents[0].joinpath(
            '4_client_result',
            'pure',
            f"{method}_{args.cbgru_net1}_{args.cbgru_net2}",
        )
        os.makedirs(result_path, exist_ok=True)
        result_file_path = result_path.joinpath(f'{args.vul}_result.json')
        
    if os.path.exists(result_file_path):
        with open(result_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if type(data) is dict:
                data = [data]
            data.append(result_dict)
        with open(result_file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    else:
        data = [result_dict]
        file = open(str(result_file_path), "w")
        json.dump(data, file, ensure_ascii=False, indent=4)
        file.close()

    print("Accuracy: ", result_dict['Accuracy'])
    print('False positive rate(FPR): ', result_dict['False positive rate(FPR)'])
    print('False negative rate(FNR): ', result_dict['False negative rate(FNR)'])
    print('Recall(TPR): ', result_dict['Recall(TPR)'])
    print('Precision: ', result_dict['Precision'])
    print('F1 score: ', result_dict['F1 score'])
            
            





    
