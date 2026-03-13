import copy
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from trainers.client import Fed_Avg_client
from utils.fedelc_losses import LogitAdjust, pencil_loss
from data_processing.mando_collate import mando_collate_fn


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


def _get_collate_fn(args):
    return mando_collate_fn if args.model_type == "MANDO" else None


def _indexed_mando_collate_fn(batch):
    base_batch = [(item[0], item[1], item[2]) for item in batch]
    x1, x2, y = mando_collate_fn(base_batch)
    idx = torch.tensor([item[3] for item in batch], dtype=torch.long)
    return x1, x2, y, idx

class Fed_ELC_client(Fed_Avg_client):
    def __init__(self, args, criterion, model, dataset, client_id=0, run_timestamp=None, cls_num_list=None):
        super().__init__(args, criterion, model, dataset, client_id, run_timestamp)
        self.cls_num_list = cls_num_list
        
        # Initialize LogitAdjust loss if class counts are provided
        if self.cls_num_list is not None:
            self.criterion_la = LogitAdjust(self.cls_num_list)
            self.criterion_la.m_list = self.criterion_la.m_list.to(self.device)

    def train_stage1(self):
        """
        Warm-up stage using Logit Adjustment
        """
        self.model.train()
        self.model.to(self.device)
        
        # Use LogitAdjust if available, else standard criterion
        criterion = self.criterion_la if hasattr(self, 'criterion_la') else self.criterion
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.outer_lr, momentum=0.9, weight_decay=5e-4)
        
        # Create DataLoader (assuming dataset is TensorDataset or similar)
        # Note: Fed_Avg_client structure implies dataset handling might need adaptation
        # Here we assume self.dataset is iterable or we create a loader
        train_loader = DataLoader(
            self.dataset,
            batch_size=self.args.batch,
            shuffle=True,
            collate_fn=_get_collate_fn(self.args)
        )
        
        epoch_loss = []
        for epoch in range(self.args.local_epoch):
            batch_loss = []
            for batch_idx, batch in enumerate(train_loader):
                # Unpack batch dynamically
                if len(batch) == 2:
                    x, y = batch
                    x_in = _move_to_device(x, self.device)
                else:
                    x1, x2, y = batch[0], batch[1], batch[2]
                    x_in = [_move_to_device(x1, self.device), _move_to_device(x2, self.device)]
                y = _move_to_device(y, self.device).long()
                
                optimizer.zero_grad()
                
                # Forward
                # Check model signature
                if isinstance(x_in, list):
                    output = self.model(*x_in)
                else:
                    output = self.model(x_in)
                    
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        self.result = {'loss': sum(epoch_loss)/len(epoch_loss), 'sample': len(self.dataset)}

    def get_class_wise_loss(self):
        """
        Calculate class-wise loss for GMM splitting
        """
        self.model.eval()
        self.model.to(self.device)
        
        # Need to iterate and sum loss per class
        # Infer num_classes from args or dataset
        if hasattr(self.args, 'num_classes'):
            num_classes = self.args.num_classes
        else:
            if hasattr(self.dataset, 'labels') and len(self.dataset.labels) > 0:
                num_classes = int(np.max(self.dataset.labels)) + 1
            else:
                num_classes = 2
            
        loss_per_class = np.zeros(num_classes)
        count_per_class = np.zeros(num_classes)
        
        train_loader = DataLoader(
            self.dataset,
            batch_size=self.args.batch,
            shuffle=False,
            collate_fn=_get_collate_fn(self.args)
        )
        
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        
        with torch.no_grad():
            for batch in train_loader:
                if len(batch) == 2:
                    x, y = batch
                    x_in = _move_to_device(x, self.device)
                else:
                    x1, x2, y = batch[0], batch[1], batch[2]
                    x_in = [_move_to_device(x1, self.device), _move_to_device(x2, self.device)]
                y = _move_to_device(y, self.device).long()

                if isinstance(x_in, list):
                    output = self.model(*x_in)
                else:
                    output = self.model(x_in)
                
                losses = criterion(output, y)
                
                for i in range(len(y)):
                    label = y[i].item()
                    loss_per_class[label] += losses[i].item()
                    count_per_class[label] += 1
                    
        # Avoid division by zero
        count_per_class[count_per_class == 0] = 1
        return loss_per_class / count_per_class

    def train_stage2(self, soft_labels, K_pencil=10, lambda_pencil=1000, alpha=0.5, beta=0.2):
        """
        Correction stage using PENCIL for noisy clients
        soft_labels: Tensor of shape (N, C)
        """
        self.model.train()
        self.model.to(self.device)
        
        # Convert dataset to loader that returns indices to map back to soft_labels
        class IndexedDataset(torch.utils.data.Dataset):
            def __init__(self, dataset):
                self.dataset = dataset
            def __len__(self):
                return len(self.dataset)
            def __getitem__(self, idx):
                data = self.dataset[idx]
                return data + (idx,) # (x, y, idx)
        
        indexed_ds = IndexedDataset(self.dataset)
        collate_fn = _indexed_mando_collate_fn if self.args.model_type == "MANDO" else None
        train_loader = torch.utils.data.DataLoader(
            indexed_ds,
            batch_size=self.args.batch,
            shuffle=True,
            collate_fn=collate_fn
        )

        # y_tilde is updated by gradient on labels (PENCIL)
        y_tilde = soft_labels.clone().detach().cpu()

        optimizer_model = torch.optim.SGD(self.model.parameters(), lr=self.args.outer_lr, momentum=0.9, weight_decay=5e-4)

        epoch_loss = []
        
        for epoch in range(self.args.local_epoch):
            batch_loss = []
            labels_grad = torch.zeros_like(y_tilde)
            for batch in train_loader:
                if len(batch) == 3:
                    x, y, idx = batch
                    x_in = _move_to_device(x, self.device)
                else:
                    x1, x2, y, idx = batch
                    x_in = [_move_to_device(x1, self.device), _move_to_device(x2, self.device)]

                y = _move_to_device(y, self.device).long()
                idx = _move_to_device(idx, self.device)
                idx_cpu = idx.detach().cpu()
                
                optimizer_model.zero_grad()
                
                if isinstance(x_in, list):
                    output = self.model(*x_in)
                else:
                    output = self.model(x_in)
                
                labels_update = y_tilde[idx_cpu].to(self.device)
                labels_update.requires_grad_(True)

                # PENCIL Loss aligned with baseline
                loss = pencil_loss(output, labels_update, y, alpha=alpha, beta=beta)
                
                loss.backward()

                labels_grad[idx_cpu] = labels_update.grad.detach().cpu()

                optimizer_model.step()

                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

            # Update y_tilde after each local epoch
            with torch.no_grad():
                y_tilde = y_tilde - lambda_pencil * labels_grad

        self.result = {'loss': sum(epoch_loss)/len(epoch_loss), 'sample': len(self.dataset)}
        
        # Refine soft labels: merge estimated labels and model predictions
        self.model.eval()
        with torch.no_grad():
            eval_loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.args.batch,
                shuffle=False,
                collate_fn=_get_collate_fn(self.args)
            )
            
            all_outputs = []
            for batch in eval_loader:
                if len(batch) == 2:
                    x, y = batch
                    x_in = _move_to_device(x, self.device)
                else:
                    x1, x2, y = batch[0], batch[1], batch[2]
                    x_in = [_move_to_device(x1, self.device), _move_to_device(x2, self.device)]

                if isinstance(x_in, list):
                    output = self.model(*x_in)
                else:
                    output = self.model(x_in)
                all_outputs.append(F.softmax(output, dim=1))
                
            all_outputs = torch.cat(all_outputs, dim=0)

            # Estimated labels from y_tilde
            estimated_labels = F.softmax(y_tilde, dim=1)

            # Merge and rescale by K_pencil
            merged_labels = (estimated_labels + all_outputs.cpu()) / 2
            y_refined = merged_labels * K_pencil

        # Return updated soft labels
        return y_refined.detach().cpu()
