import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_processing.mando_collate import lgv_mando_collate_fn
from models.model_factory import get_local_epoch, get_local_lr
from trainers.client import Fed_LGV_client, _move_to_device, _unpack_batch


def _set_agreement_ratio(dataset, size):
    if hasattr(dataset, 'set_ag_rt'):
        dataset.set_ag_rt([1.0 for _ in range(size)])


class FedLGV_NoLocalClient(Fed_LGV_client):
    def get_local_knn_labels(self, vul, noise_type, noise_rate):
        labels = torch.as_tensor(self.dataset.labels, dtype=torch.long)
        num_classes = int(getattr(self.args, 'num_classes', 2))
        labels = torch.clamp(labels, min=0, max=num_classes - 1)

        local_prob = F.one_hot(labels, num_classes=num_classes).float()
        self.local_prob_labels = local_prob.to(self.device)

        _set_agreement_ratio(self.dataset, len(self.dataset.labels))

        self.gen_reduced_ds()
        reduced_size = len(self.reduced_ds.labels)
        neigh = max(1, int(getattr(self.args, 'num_neigh', 1)))
        base_idx = np.arange(reduced_size, dtype=np.int64).reshape(-1, 1)
        self.reduced_knn_indices = np.repeat(base_idx, repeats=neigh, axis=1)


class FedLGV_NoGlobalClient(Fed_LGV_client):
    def get_global_feature_global_knn_labels(self):
        if self.local_prob_labels.device != self.device:
            self.local_prob_labels = self.local_prob_labels.to(self.device)
        self.global_prob_labels = self.local_prob_labels.clone()


class FedLGV_NoUncAlphaClient(Fed_LGV_client):
    def train(self):
        self.fixed_global_model.load_state_dict(self.model.state_dict())
        self.fixed_global_model.eval()

        alpha_scalar = float(self.global_weight)
        alpha_scalar = max(float(self.args.alpha_min), min(alpha_scalar, float(self.args.alpha_max)))

        if self.global_prob_labels.device != self.device:
            self.global_prob_labels = self.global_prob_labels.to(self.device)
        if self.local_prob_labels.device != self.device:
            self.local_prob_labels = self.local_prob_labels.to(self.device)

        alpha = torch.full(
            (self.local_prob_labels.shape[0], 1),
            fill_value=alpha_scalar,
            dtype=self.local_prob_labels.dtype,
            device=self.device,
        )

        with torch.no_grad():
            prob_labels = alpha * self.global_prob_labels + (1 - alpha) * self.local_prob_labels
            prob_labels = F.softmax(prob_labels, dim=1)
            labels = torch.argmax(prob_labels, dim=-1)

        self.dataset.labels = labels.detach().cpu().numpy()

        collate_fn = lgv_mando_collate_fn if self.args.model_type == "MANDO" else None
        dl = DataLoader(self.dataset, batch_size=self.args.batch, shuffle=True, pin_memory=True, collate_fn=collate_fn)
        lr = get_local_lr(self.args)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.args.weight_decay)

        self.result = dict()
        device = self.args.device
        self.result['sample'] = len(self.dataset)

        self.model.train()
        for epoch in range(get_local_epoch(self.args)):
            self.result['loss'] = 0
            for batch in dl:
                optimizer.zero_grad()
                x1, x2, y, agr = _unpack_batch(batch)
                x1 = _move_to_device(x1, device)
                x2 = _move_to_device(x2, device)
                y = _move_to_device(y, device)
                if agr is None:
                    agr = torch.ones_like(y, dtype=torch.float32, device=device)
                else:
                    agr = _move_to_device(agr, device)
                outputs = self.model(x1, x2)
                y = y.flatten().long()
                loss = self.criterion(outputs, y)

                _, predictions = torch.max(outputs, 1)
                correct_predictions = (predictions == y)
                weights = torch.ones_like(y, dtype=torch.float32)
                weights += agr * (~correct_predictions).float()
                weights -= 0.5 * agr * correct_predictions.float()
                if self.args.consistency_score:
                    weighted_losses = weights * loss
                else:
                    weighted_losses = loss
                loss = weighted_losses.mean()

                self.result['loss'] += loss.item()
                loss.backward()
                clip_value = 1.0 if getattr(self.args, 'vul', '') == 'timestamp' else 10
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_value)
                optimizer.step()

            avg_loss = self.result['loss'] / len(dl)
            self.tb_writer.add_scalar("loss/train", avg_loss, self.tb_global_step)
            with open(self.log_file_path, "a") as f:
                f.write(f"{self.tb_global_step},{epoch},{avg_loss}\n")
            self.tb_global_step += 1


class FedLGV_NoLocalNoGlobalClient(FedLGV_NoLocalClient, FedLGV_NoGlobalClient):
    def get_local_knn_labels(self, vul, noise_type, noise_rate):
        return FedLGV_NoLocalClient.get_local_knn_labels(self, vul, noise_type, noise_rate)

    def get_global_feature_global_knn_labels(self):
        return FedLGV_NoGlobalClient.get_global_feature_global_knn_labels(self)


class FedLGV_NoGlobalNoUncAlphaClient(FedLGV_NoUncAlphaClient, FedLGV_NoGlobalClient):
    def get_global_feature_global_knn_labels(self):
        return FedLGV_NoGlobalClient.get_global_feature_global_knn_labels(self)
