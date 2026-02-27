import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from trainers.client import Fed_CRD_client


class Fed_CRD_client_NoAmb(Fed_CRD_client):
    """
    FedCRD ablation client for removing ambiguity penalty in q_k.

    - soft variant: q_k = min(q_loc, q_glob)
    - const variant: q_k = args.crd_const_q
    """

    def get_consistency_stats(self, anchor_model):
        if self.reduced_knn_indices is None:
            self.init_knn_neighborhood()

        variant = str(getattr(self.args, "crd_noamb_variant", "soft")).lower()
        if variant == "const":
            q_const = float(getattr(self.args, "crd_const_q", 1.0))
            return q_const, len(self.dataset)

        # Reuse the same q_loc/q_glob calculation path as Fed_CRD_client,
        # but skip JS-based ambiguity decay exp(-lambda * d_tilde).
        name_to_idx = {name: i for i, name in enumerate(self.dataset.names)}
        indices = [name_to_idx[name] for name in self.reduced_names]
        reduced_ds = Subset(self.dataset, indices)
        dl = DataLoader(
            reduced_ds,
            batch_size=self.args.batch,
            shuffle=False,
            pin_memory=True,
        )

        all_probs = []
        anchor_model.eval()
        with torch.no_grad():
            for batch in dl:
                if len(batch) == 3:
                    x1, x2, _ = batch
                elif len(batch) >= 4:
                    x1, x2, _, _ = batch[0:4]
                else:
                    raise ValueError(f"Unexpected batch length={len(batch)}")

                x1, x2 = x1.to(self.device), x2.to(self.device)
                outputs = anchor_model(x1, x2)
                probs = F.softmax(outputs, dim=1)
                all_probs.append(probs)

        all_probs = torch.cat(all_probs, dim=0)
        knn_indices = torch.tensor(
            self.reduced_knn_indices, dtype=torch.long, device=self.device
        )
        neighbor_probs = all_probs[knn_indices]
        pi_glob = torch.mean(neighbor_probs, dim=1)
        h_glob = torch.max(pi_glob, dim=1)[0]

        q_loc = torch.mean(self.h_loc).item()
        q_glob = torch.mean(h_glob).item()
        q_k = min(q_loc, q_glob)

        return q_k, len(self.dataset)
