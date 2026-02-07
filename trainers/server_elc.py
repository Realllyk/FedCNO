import copy
import torch
import numpy as np
from trainers.server import Server
from sklearn.mixture import GaussianMixture

class ELC_Server(Server):
    def __init__(self, args, model, device, criterion):
        super().__init__(args, model, device, criterion)
        self.clean_clients = []
        self.noisy_clients = []
        self.client_ids_in_updates = [] # Track client IDs for current epoch updates

    def initialize_epoch_updates(self, epoch):
        super().initialize_epoch_updates(epoch)
        self.client_ids_in_updates = []

    def save_train_updates(self, model_updates, num_sample, result, client_id):
        # Override to store client_id
        super().save_train_updates(model_updates, num_sample, result)
        self.client_ids_in_updates.append(client_id)

    def gmm_split(self, client_losses):
        """
        Split clients into clean and noisy sets based on class-wise losses
        client_losses: dict {client_id: loss_vector (np.array)}
        """
        loss_matrix = []
        ids = []
        for cid, loss_vec in client_losses.items():
            loss_matrix.append(loss_vec)
            ids.append(cid)
            
        loss_matrix = np.array(loss_matrix)
        
        # Normalize per-class (column-wise) to [0, 1] range as in FedELC
        # Handle columns with constant values (max == min) to avoid division by zero
        min_vals = loss_matrix.min(axis=0)
        max_vals = loss_matrix.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0 # Prevent division by zero
        
        loss_matrix_norm = (loss_matrix - min_vals) / range_vals
        
        # Fill NaNs with min (though we shouldn't have NaNs if logic is correct)
        loss_matrix_norm = np.nan_to_num(loss_matrix_norm)

        # FedELC uses voting from multiple GMM seeds (0-9)
        # Noisy cluster is the one with larger mean loss (sum of means).
        vote = []
        for seed in range(10):
            gmm = GaussianMixture(n_components=2, random_state=seed).fit(loss_matrix_norm)
            means = gmm.means_
            noisy_cluster_idx = np.argmax(means.sum(axis=1))
            labels = gmm.predict(loss_matrix_norm)
            noisy_clients = [ids[i] for i, label in enumerate(labels) if label == noisy_cluster_idx]
            vote.append(tuple(sorted(noisy_clients)))

        # Pick the most frequent vote
        counts = {v: vote.count(v) for v in vote}
        noisy_final = max(counts, key=counts.get)

        self.noisy_clients = list(noisy_final)
        self.clean_clients = list(set(ids) - set(self.noisy_clients))

        return self.clean_clients, self.noisy_clients

    def distance_aware_aggregation(self):
        """
        Aggregate models using distance-aware weights for noisy clients
        Matches DaAgg in FedELC/fl_models/fed.py
        """
        if not self.model_updates:
            return
            
        # Identify clean and noisy indices in current round updates
        clean_indices = []
        noisy_indices = []
        
        cid_to_idx = {cid: i for i, cid in enumerate(self.client_ids_in_updates)}
        
        for cid in self.client_ids_in_updates:
            if cid in self.clean_clients:
                clean_indices.append(cid_to_idx[cid])
            elif cid in self.noisy_clients:
                noisy_indices.append(cid_to_idx[cid])
                
        # Fallback to standard avg if mix is not present
        if not clean_indices or not noisy_indices:
            self.average_weights()
            return

        def get_model_dist(w1, w2):
            dist_total = 0.0
            for k in w1.keys():
                if 'num_batches_tracked' in k: continue
                if 'running' in k: continue
                if w1[k].dtype in [torch.float32, torch.float64]:
                    # Use cpu() to avoid gpu memory buildup if needed, but here we likely on same device
                    d = torch.norm(w1[k] - w2[k]).item()
                    dist_total += d
            return dist_total

        # Calculate base weights (based on sample size)
        total_samples = sum(self.num_samples_list)
        base_weights = np.array(self.num_samples_list) / total_samples
        
        # Calculate distances for noisy clients
        # Distance to NEAREST clean client (DaAgg logic)
        distances = np.zeros(len(self.model_updates))
        
        for n_idx in noisy_indices:
            n_w = self.model_updates[n_idx]
            dists = []
            
            for c_idx in clean_indices:
                c_w = self.model_updates[c_idx]
                dists.append(get_model_dist(n_w, c_w))
            
            distances[n_idx] = min(dists)
            
        # Normalize distances
        max_dist = distances.max()
        if max_dist > 0:
            distances = distances / max_dist
            
        # Adjust weights: w' = w * exp(-distance)
        # Clean clients have distance 0, so exp(0)=1
        adjusted_weights = base_weights * np.exp(-distances)
        
        # Re-normalize
        if adjusted_weights.sum() > 0:
            adjusted_weights = adjusted_weights / adjusted_weights.sum()
        else:
            adjusted_weights = base_weights # Fallback
        
        # Aggregate
        w_avg = copy.deepcopy(self.model_updates[0])
        # Zero out
        for key in w_avg.keys():
            w_avg[key] = torch.zeros_like(w_avg[key], dtype=torch.float)
            
        for i, cid in enumerate(self.client_ids_in_updates):
            w = adjusted_weights[i]
            update = self.model_updates[i]
            for key in w_avg.keys():
                 w_avg[key] += update[key] * w
                
        self.global_model.load_state_dict(w_avg)
