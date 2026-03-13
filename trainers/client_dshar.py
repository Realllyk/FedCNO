import copy
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from data_processing.mando_collate import mando_collate_fn
from models.model_factory import get_local_epoch, get_local_lr


class _IndexedSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        x1, x2, y = self.dataset[original_idx]
        return x1, x2, y


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


class Fed_DSHAR_client(object):
    def __init__(
        self,
        args,
        model,
        teacher_model,
        dataset,
        client_id=0,
        run_timestamp=None
    ):
        self.args = args
        self.model = model
        self.teacher_model = teacher_model
        self.dataset = dataset
        self.client_id = client_id
        self.device = args.device
        self.result = {}

        if run_timestamp is None:
            run_timestamp = time.strftime("%Y%m%d_%H%M%S")

        log_dir = os.path.join(
            "runs",
            args.lab_name,
            args.model_type,
            args.noise_type,
            str(args.noise_rate),
            args.vul,
            run_timestamp,
            f"client_{self.client_id}"
        )
        os.makedirs(log_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=log_dir)
        self.tb_global_step = 0

        labels = np.array(self.dataset.labels, dtype=np.int64)
        num_classes = max(getattr(args, "num_classes", 2), int(labels.max()) + 1 if labels.size > 0 else 2)
        counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
        priors = counts / max(counts.sum(), 1.0)
        priors = np.clip(priors, 1e-6, 1.0)
        self.logit_bias = torch.tensor(
            args.dshar_la_tau * np.log(priors),
            dtype=torch.float32,
            device=self.device
        )

    def get_parameters(self):
        return self.model.state_dict()

    def _augment(self, x):
        if isinstance(x, dict):
            out = dict(x)
            node_feat = out.get("node_features", None)
            if node_feat is None:
                return out
            if self.args.dshar_aug_noise_std > 0:
                node_feat = node_feat + torch.randn_like(node_feat) * self.args.dshar_aug_noise_std
            if self.args.dshar_aug_mask_ratio > 0:
                mask = (torch.rand_like(node_feat) > self.args.dshar_aug_mask_ratio).float()
                node_feat = node_feat * mask
            out["node_features"] = node_feat
            return out
        if self.args.dshar_aug_noise_std > 0:
            x = x + torch.randn_like(x) * self.args.dshar_aug_noise_std
        if self.args.dshar_aug_mask_ratio > 0:
            mask = (torch.rand_like(x) > self.args.dshar_aug_mask_ratio).float()
            x = x * mask
        return x

    def _train_clean_step(self, optimizer, clean_loader):
        total_loss = 0.0
        total_batches = 0

        for x1, x2, y in clean_loader:
            x1 = _move_to_device(x1, self.device)
            x2 = _move_to_device(x2, self.device)
            y = _move_to_device(y, self.device).flatten().long()

            aug1_x1 = self._augment(x1)
            aug1_x2 = self._augment(x2)
            aug2_x1 = self._augment(x1)
            aug2_x2 = self._augment(x2)

            optimizer.zero_grad()
            logits_1 = self.model(aug1_x1, aug1_x2)
            logits_2 = self.model(aug2_x1, aug2_x2)

            ce_loss = F.cross_entropy(logits_1, y)

            prob_1 = F.softmax(logits_1, dim=1)
            prob_2 = F.softmax(logits_2, dim=1)
            mean_prob = 0.5 * (prob_1.mean(dim=0) + prob_2.mean(dim=0))
            diversity_loss = torch.sum(mean_prob * torch.log(mean_prob + 1e-8))

            loss = ce_loss + self.args.dshar_lambda_div * diversity_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        if total_batches == 0:
            return 0.0
        return total_loss / total_batches

    def _train_noisy_step(self, optimizer, noisy_loader):
        total_loss = 0.0
        total_batches = 0
        total_selected = 0

        self.teacher_model.eval()

        for x1, x2, _ in noisy_loader:
            x1 = _move_to_device(x1, self.device)
            x2 = _move_to_device(x2, self.device)

            with torch.no_grad():
                teacher_logits = self.teacher_model(x1, x2)
                teacher_probs = F.softmax(teacher_logits, dim=1)
                confidence, pseudo_labels = torch.max(teacher_probs, dim=1)
                mask = confidence >= self.args.dshar_pseudo_threshold

            if mask.sum().item() == 0:
                continue

            optimizer.zero_grad()
            student_logits = self.model(x1, x2) + self.logit_bias.unsqueeze(0)
            loss = F.cross_entropy(student_logits[mask], pseudo_labels[mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1
            total_selected += int(mask.sum().item())

        avg_loss = 0.0 if total_batches == 0 else total_loss / total_batches
        return avg_loss, total_selected

    def train(self, clean_indices, noisy_indices):
        lr = get_local_lr(self.args)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.args.weight_decay)
        self.model.to(self.device)
        self.model.train()
        collate_fn = mando_collate_fn if self.args.model_type == "MANDO" else None

        clean_loader = DataLoader(
            _IndexedSubset(self.dataset, clean_indices),
            batch_size=self.args.batch,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn
        ) if len(clean_indices) > 0 else None

        noisy_loader = DataLoader(
            _IndexedSubset(self.dataset, noisy_indices),
            batch_size=self.args.batch,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn
        ) if len(noisy_indices) > 0 else None

        round_loss = 0.0
        clean_loss_meter = 0.0
        noisy_loss_meter = 0.0
        pseudo_selected = 0

        for _ in range(get_local_epoch(self.args)):
            clean_loss = 0.0
            noisy_loss = 0.0
            selected = 0

            if clean_loader is not None:
                clean_loss = self._train_clean_step(optimizer, clean_loader)

            if noisy_loader is not None:
                noisy_loss, selected = self._train_noisy_step(optimizer, noisy_loader)

            epoch_loss = self.args.dshar_w_clean * clean_loss + self.args.dshar_w_noisy * noisy_loss
            round_loss += epoch_loss
            clean_loss_meter += clean_loss
            noisy_loss_meter += noisy_loss
            pseudo_selected += selected

            self.tb_writer.add_scalar("loss/clean", clean_loss, self.tb_global_step)
            self.tb_writer.add_scalar("loss/noisy", noisy_loss, self.tb_global_step)
            self.tb_writer.add_scalar("loss/total", epoch_loss, self.tb_global_step)
            self.tb_writer.add_scalar("pseudo/selected", selected, self.tb_global_step)
            self.tb_global_step += 1

        denom = max(get_local_epoch(self.args), 1)
        self.result = {
            "loss": round_loss / denom,
            "clean_loss": clean_loss_meter / denom,
            "noisy_loss": noisy_loss_meter / denom,
            "pseudo_selected": pseudo_selected,
            "clean_samples": len(clean_indices),
            "noisy_samples": len(noisy_indices),
            "sample": len(self.dataset),
        }

        return copy.deepcopy(self.result)
