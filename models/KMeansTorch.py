import torch
import numpy as np

class KMeansTorch:
    """
    A simple KMeans implementation using PyTorch for GPU acceleration.
    API follows sklearn.cluster.KMeans style.
    """
    def __init__(self, n_clusters=20, max_iter=300, tol=1e-4, device='cuda', seed=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.seed = seed
        self.cluster_centers_ = None

    def fit_predict(self, X):
        """
        Compute cluster centers and predict cluster index for each sample.
        Args:
            X: numpy array or torch tensor, shape (n_samples, n_features)
        Returns:
            labels: numpy array, shape (n_samples,)
        """
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float()
        elif isinstance(X, torch.Tensor):
            X_tensor = X.float()
        else:
            raise ValueError("Input must be numpy array or torch tensor")
            
        # Robust device handling
        target_device = torch.device(self.device)
        if target_device.type == 'cuda' and not torch.cuda.is_available():
            target_device = torch.device('cpu')
            
        X_tensor = X_tensor.to(target_device)
        n_samples, n_features = X_tensor.shape
        
        # Set seed for reproducibility
        torch.manual_seed(self.seed)
        if target_device.type == 'cuda':
            torch.cuda.manual_seed_all(self.seed)
        
        # Initialize centroids randomly from data points
        # Use torch.randperm to select random indices
        random_indices = torch.randperm(n_samples, device=target_device)[:self.n_clusters]
        self.cluster_centers_ = X_tensor[random_indices].clone()
        
        prev_labels = None
        
        for i in range(self.max_iter):
            # Compute distances: ||X - C||
            # torch.cdist computes euclidean distance
            # X: (N, D), Centers: (K, D) -> Dist: (N, K)
            distances = torch.cdist(X_tensor, self.cluster_centers_)
            
            # Assign clusters
            labels = torch.argmin(distances, dim=1)
            
            # Check convergence by labels
            if prev_labels is not None and torch.equal(labels, prev_labels):
                break
            prev_labels = labels.clone()
            
            # Update centroids
            new_centroids = torch.zeros_like(self.cluster_centers_)
            
            # Vectorized update? 
            # A loop is safer for empty clusters handling and usually K is small (e.g. 20)
            for k in range(self.n_clusters):
                mask = (labels == k)
                if mask.sum() > 0:
                    new_centroids[k] = X_tensor[mask].mean(dim=0)
                else:
                    # Handle empty cluster: keep old centroid or re-init?
                    # Sklearn keeps old or re-inits. Here we keep old to avoid drift/NaN.
                    # Or better: pick a random point far from current centers (complex).
                    # Simple: keep old.
                    new_centroids[k] = self.cluster_centers_[k]
            
            # Check convergence by center shift
            center_shift = torch.sum((self.cluster_centers_ - new_centroids) ** 2)
            self.cluster_centers_ = new_centroids
            
            if center_shift < self.tol:
                break
                
        return labels.cpu().numpy()
