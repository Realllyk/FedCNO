import numpy as np
import torch as nn
from sklearn.metrics import accuracy_score, confusion_matrix

class Evaluation(object):
    def __init__(self):
        self.truth_list = list()
        self.pred_list = list()
        self.loss_list = list()

    def update_result_list(
        self,
        labels: nn.tensor,
        outputs: nn.tensor,
        loss
    ):
        preds = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        labels = labels.detach().cpu().numpy()

        for index, pred in enumerate(preds):
            self.pred_list.append(pred)
            self.truth_list.append(labels[index])
        self.loss_list.append(loss.item())
        
    def results_summary(self):
        result_dict = dict()
        tn, fp, fn, tp = confusion_matrix(np.array(self.truth_list), np.array(self.pred_list))
        result_dict['acc'] = (tn+tp)/(tn+fp+fn+tp)
        result_dict['pre'] = tp/(tp+fp)
        result_dict['recall'] = tp/(tp+fn)
        result_dict['loss'] = np.mean(self.loss_list)
        result_dict['sample'] = len(self.truth_list)
        return result_dict

        