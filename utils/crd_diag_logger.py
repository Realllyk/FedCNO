import csv
import os


class CRDDiagLogger(object):
    def __init__(self, log_dir, run_tag):
        self.log_dir = os.path.join(log_dir, run_tag)
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, "crd_diag.csv")
        self.fieldnames = [
            "round",
            "client_id",
            "noise_type",
            "noise_rate",
            "vul",
            "model_type",
            "seed",
            "n_k",
            "q_k_t",
            "norm_delta_k_t",
            "cos_sim_k_t",
            "r_raw_k_t",
            "r_hat_k_t",
            "sigma_k_t",
            "r_tilde_k_t",
            "tau_t",
            "clip_scale_k_t",
            "omega_k_t",
        ]

    def log_round(self, round_idx, meta_dict, client_rows):
        if not client_rows:
            return

        file_exists = os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not file_exists:
                writer.writeheader()

            for row in client_rows:
                merged = {
                    "round": round_idx,
                    "noise_type": meta_dict.get("noise_type", ""),
                    "noise_rate": meta_dict.get("noise_rate", ""),
                    "vul": meta_dict.get("vul", ""),
                    "model_type": meta_dict.get("model_type", ""),
                    "seed": meta_dict.get("seed", ""),
                }
                merged.update(row)
                writer.writerow({k: merged.get(k, "") for k in self.fieldnames})
