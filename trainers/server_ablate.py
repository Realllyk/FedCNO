import copy
import math

import numpy as np
import torch

from trainers.server import CRD_server


class CRD_server_NoCal_FedAvg(CRD_server):
    """
    No calibration ablation:
    omega_k = n_k / sum_u n_u
    """

    def aggregate(self, updates_list):
        if not updates_list:
            return

        crd_eps = float(getattr(self.args, "crd_eps", 1e-8))
        first_delta = updates_list[0][1]

        n_sum = sum([n_k for _, _, _, n_k in updates_list]) + crd_eps
        global_update = {k: torch.zeros_like(v).float() for k, v in first_delta.items()}

        for _, delta_k_t, _, n_k in updates_list:
            omega_k_t = n_k / n_sum
            for k, v in delta_k_t.items():
                global_update[k] += v.to(self.device) * omega_k_t

        current_params = self.global_model.state_dict()
        new_params = copy.deepcopy(current_params)
        for k in new_params.keys():
            if k in global_update:
                new_params[k] = current_params[k].float() + global_update[k].float()
        self.global_model.load_state_dict(new_params)

        print(f"[FedCRD-Ablation][NoCal] Aggregated {len(updates_list)} updates | sum_omega=1.000000")
        self.update_ema_model()


class CRD_server_NoClip(CRD_server):
    """
    No quantile clipping ablation:
    keep reliability calibration, but delta_clip = delta.
    """

    def aggregate(self, updates_list):
        if not updates_list:
            return

        crd_eps = float(getattr(self.args, "crd_eps", 1e-8))
        crd_sigma = str(getattr(self.args, "crd_sigma", "softplus")).lower()
        if crd_sigma != "softplus":
            print(f"[FedCRD] crd_sigma={crd_sigma} is unsupported, fallback to softplus.")

        num_clients = len(updates_list)
        first_delta = updates_list[0][1]

        def flatten(state_dict):
            return torch.cat([v.flatten().float() for v in state_dict.values()])

        delta_bar = {k: torch.zeros_like(v).float() for k, v in first_delta.items()}
        for _, delta_k_t, _, _ in updates_list:
            for k, v in delta_k_t.items():
                delta_bar[k] += v.float()
        for k in delta_bar.keys():
            delta_bar[k] /= num_clients

        delta_bar_t_vec = flatten(delta_bar).to(self.device)
        norm_delta_bar_t = torch.norm(delta_bar_t_vec).item()

        client_stats = []
        sigma_sum = 0.0
        for client_id, delta_k_t, q_k_t, n_k in updates_list:
            delta_k_t_vec = flatten(delta_k_t).to(self.device)
            norm_delta_k_t = torch.norm(delta_k_t_vec).item()

            denom = (norm_delta_k_t * norm_delta_bar_t) + crd_eps
            cos_sim = (torch.dot(delta_k_t_vec, delta_bar_t_vec).item()) / denom
            r_raw_k_t = cos_sim * np.exp(-self.lambda_agg * norm_delta_k_t)
            r_hat_k_t = q_k_t + self.alpha_crd * r_raw_k_t

            sigma_k_t = torch.nn.functional.softplus(
                torch.tensor(r_hat_k_t, dtype=torch.float32, device=self.device)
            ).item()
            sigma_sum += sigma_k_t

            client_stats.append(
                {
                    "client_id": client_id,
                    "delta_k_t": delta_k_t,
                    "n_k": n_k,
                    "norm_delta_k_t": norm_delta_k_t,
                    "r_raw_k_t": r_raw_k_t,
                    "r_hat_k_t": r_hat_k_t,
                    "sigma_k_t": sigma_k_t,
                }
            )

        sigma_denom = sigma_sum + crd_eps
        for stat in client_stats:
            stat["r_tilde_k_t"] = stat["sigma_k_t"] / sigma_denom

        omega_denom = sum([stat["n_k"] * stat["r_tilde_k_t"] for stat in client_stats]) + crd_eps
        global_update = {k: torch.zeros_like(v).float() for k, v in first_delta.items()}
        sum_omega = 0.0

        for stat in client_stats:
            omega_k_t = (stat["n_k"] * stat["r_tilde_k_t"]) / omega_denom
            sum_omega += omega_k_t

            # NoClip: use raw update directly.
            for k, v in stat["delta_k_t"].items():
                global_update[k] += v.to(self.device) * omega_k_t

        current_params = self.global_model.state_dict()
        new_params = copy.deepcopy(current_params)
        for k in new_params.keys():
            if k in global_update:
                new_params[k] = current_params[k].float() + global_update[k].float()
        self.global_model.load_state_dict(new_params)

        mean_r_tilde = float(np.mean([stat["r_tilde_k_t"] for stat in client_stats]))
        sum_r_tilde = float(np.sum([stat["r_tilde_k_t"] for stat in client_stats]))
        print(
            f"[FedCRD-Ablation][NoClip] Aggregated {len(client_stats)} updates | "
            f"tau_t=disabled | mean_r_tilde={mean_r_tilde:.6f} | "
            f"sum_r_tilde={sum_r_tilde:.6f} | sum_omega={sum_omega:.6f}"
        )
        self.update_ema_model()


class CRD_server_OnlyClip(CRD_server):
    """
    Reserved extension:
    sample-count weighting + quantile clipping only.
    """

    def aggregate(self, updates_list):
        if not updates_list:
            return

        crd_eps = float(getattr(self.args, "crd_eps", 1e-8))
        crd_rho = float(getattr(self.args, "crd_rho", 0.7))
        crd_rho = max(0.0, min(1.0, crd_rho))
        first_delta = updates_list[0][1]

        def flatten(state_dict):
            return torch.cat([v.flatten().float() for v in state_dict.values()])

        norms = []
        for _, delta_k_t, _, _ in updates_list:
            norms.append(torch.norm(flatten(delta_k_t).to(self.device)).item())
        norms_sorted = sorted(norms)
        n_t = len(norms_sorted)
        quantile_rank = max(1, min(n_t, int(math.ceil(crd_rho * n_t))))
        tau_t = norms_sorted[quantile_rank - 1]

        n_sum = sum([n_k for _, _, _, n_k in updates_list]) + crd_eps
        global_update = {k: torch.zeros_like(v).float() for k, v in first_delta.items()}
        sum_omega = 0.0
        for _, delta_k_t, _, n_k in updates_list:
            norm_delta_k_t = torch.norm(flatten(delta_k_t).to(self.device)).item()
            clip_scale = min(norm_delta_k_t, tau_t) / (norm_delta_k_t + crd_eps)
            omega_k_t = n_k / n_sum
            sum_omega += omega_k_t
            for k, v in delta_k_t.items():
                global_update[k] += (v * clip_scale).to(self.device) * omega_k_t

        current_params = self.global_model.state_dict()
        new_params = copy.deepcopy(current_params)
        for k in new_params.keys():
            if k in global_update:
                new_params[k] = current_params[k].float() + global_update[k].float()
        self.global_model.load_state_dict(new_params)
        print(
            f"[FedCRD-Ablation][OnlyClip] Aggregated {len(updates_list)} updates | "
            f"tau_t={tau_t:.6f} | sum_omega={sum_omega:.6f}"
        )
        self.update_ema_model()
