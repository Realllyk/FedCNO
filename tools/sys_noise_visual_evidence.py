import os
import sys
import json
import math
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
try:
    import seaborn as sns
except ImportError:
    sns = None

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from data_processing.preprocessing import (
    coordinate_sys_noise_clusters,
    resolve_pretrain_feature_dir,
    read_pretrain_feature,
)


def parse_args():
    parser = argparse.ArgumentParser("Build visual evidence for sys_noise.")
    parser.add_argument("--vul", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="CBGRU", choices=["CBGRU", "CGE", "MANDO"])
    parser.add_argument("--noise_rate", type=float, required=True)
    parser.add_argument("--noise_type", type=str, default="sys_noise")
    parser.add_argument("--client_num", type=int, default=4)
    parser.add_argument("--n_clusters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--method", type=str, default="tsne", choices=["tsne", "umap"])
    parser.add_argument("--perplexity", type=float, default=30.0)
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def model_split_prefix(model_type: str) -> str:
    if model_type == "CBGRU":
        return "cbgru"
    if model_type == "CGE":
        return "cge"
    if model_type == "MANDO":
        return "mando"
    return model_type.lower()


def client_dir_for(model_type: str, vul: str, client_id: int, data_dir: str) -> str:
    prefix = model_split_prefix(model_type)
    return os.path.join(data_dir, f"graduate_client_split/{prefix}/{vul}/client_{client_id}")


def read_names_labels(names_path: str, labels_path: str) -> Tuple[List[str], np.ndarray]:
    with open(names_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f.readlines()]
    labels = pd.read_csv(labels_path, header=None).iloc[:, 0].to_numpy(dtype=np.int64)
    return names, labels


def unique_by_name(names: List[str], labels: np.ndarray) -> Tuple[List[str], np.ndarray]:
    unique_names = []
    unique_labels = []
    seen = set()
    for i, name in enumerate(names):
        if name not in seen:
            seen.add(name)
            unique_names.append(name)
            unique_labels.append(int(labels[i]))
    return unique_names, np.array(unique_labels, dtype=np.int64)


def build_client_cluster_counts(
    client_num: int,
    model_type: str,
    vul: str,
    data_dir: str,
    global_cluster_map: Dict[str, int],
    n_clusters: int,
) -> Dict[int, Dict[int, int]]:
    counts = {cid: {c: 0 for c in range(n_clusters)} for cid in range(client_num)}
    for cid in range(client_num):
        names_path = os.path.join(client_dir_for(model_type, vul, cid, data_dir), "contract_name_train.txt")
        if not os.path.exists(names_path):
            continue
        with open(names_path, "r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name in global_cluster_map:
                    c = int(global_cluster_map[name])
                    if 0 <= c < n_clusters:
                        counts[cid][c] += 1
    return counts


def traced_sys_noise_on_unique(
    unique_names: List[str],
    y_clean: np.ndarray,
    noise_rate: float,
    n_clusters: int,
    seed: int,
    assigned_cluster_indices: List[int],
    global_cluster_map: Dict[str, int],
):
    n = len(unique_names)
    m_target = int(math.floor(noise_rate * n))

    cluster_ids = np.array([int(global_cluster_map.get(name, -1)) for name in unique_names], dtype=np.int64)

    assigned_set = set()
    if assigned_cluster_indices is not None:
        assigned_set = set(int(c) % n_clusters for c in assigned_cluster_indices)

    flip_candidates_info = []
    for c in range(n_clusters):
        idx_c = np.where(cluster_ids == c)[0]
        if len(idx_c) == 0:
            continue

        labels_c = y_clean[idx_c]
        count0 = int(np.sum(labels_c == 0))
        count1 = int(np.sum(labels_c == 1))

        if count0 >= count1:
            maj_label, min_label = 0, 1
        else:
            maj_label, min_label = 1, 0

        majority_indices = idx_c[labels_c == maj_label]
        if len(majority_indices) > 0:
            flip_candidates_info.append(
                {
                    "cluster_id": c,
                    "candidate_indices": majority_indices,
                    "candidate_count": len(majority_indices),
                    "target_label": min_label,
                    "is_assigned": (c in assigned_set) if assigned_cluster_indices is not None else True,
                }
            )

    flip_candidates_info.sort(key=lambda x: (x["is_assigned"], x["candidate_count"]), reverse=True)

    chosen_candidates = []
    current_count = 0

    for info in flip_candidates_info:
        if current_count >= m_target:
            break

        c_idxs = info["candidate_indices"]
        needed = m_target - current_count
        if len(c_idxs) <= needed:
            for idx in c_idxs:
                chosen_candidates.append(
                    {
                        "index": int(idx),
                        "target_label": int(info["target_label"]),
                        "cluster_id": int(info["cluster_id"]),
                        "flip_source": "greedy",
                    }
                )
            current_count += len(c_idxs)
        else:
            rng = np.random.RandomState(seed + int(info["cluster_id"]))
            selected = rng.choice(c_idxs, size=needed, replace=False)
            for idx in selected:
                chosen_candidates.append(
                    {
                        "index": int(idx),
                        "target_label": int(info["target_label"]),
                        "cluster_id": int(info["cluster_id"]),
                        "flip_source": "greedy",
                    }
                )
            current_count += needed

    if current_count < m_target:
        chosen_indices = set(item["index"] for item in chosen_candidates)
        available = []

        if assigned_cluster_indices is not None:
            valid_clusters = set(int(c) % n_clusters for c in assigned_cluster_indices)
            for idx, c_id in enumerate(cluster_ids):
                if int(c_id) in valid_clusters and idx not in chosen_indices:
                    available.append(idx)

            if len(available) < (m_target - current_count):
                available = [idx for idx in range(n) if idx not in chosen_indices]
        else:
            available = [idx for idx in range(n) if idx not in chosen_indices]

        shortage = min(m_target - current_count, len(available))
        if shortage > 0:
            rng = np.random.RandomState(seed + 999)
            fallback_indices = rng.choice(available, size=shortage, replace=False)
            for idx in fallback_indices:
                chosen_candidates.append(
                    {
                        "index": int(idx),
                        "target_label": int(1 - y_clean[idx]),
                        "cluster_id": int(cluster_ids[idx]),
                        "flip_source": "fallback",
                    }
                )
            current_count += shortage

    y_noisy = y_clean.copy()
    flip_info = {}
    for item in chosen_candidates:
        idx = item["index"]
        y_noisy[idx] = item["target_label"]
        flip_info[idx] = item

    return cluster_ids, y_noisy, flip_info, m_target


def project_2d(features: np.ndarray, method: str, seed: int, perplexity: float):
    if method == "umap":
        try:
            import umap

            reducer = umap.UMAP(random_state=seed, n_components=2)
            coords = reducer.fit_transform(features)
            return coords
        except Exception:
            print("[Warn] UMAP unavailable; fallback to t-SNE.")

    p = min(perplexity, max(5.0, len(features) - 1.0))
    model = TSNE(n_components=2, random_state=seed, perplexity=p, init="pca", learning_rate="auto")
    return model.fit_transform(features)


def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def concentration_metrics(df: pd.DataFrame, flip_col: str, assigned_col: str = "is_assigned_cluster") -> Dict[str, float]:
    flips = df[df[flip_col] == 1]
    n_flips = len(flips)
    if n_flips == 0:
        return {"flips": 0, "top3_coverage": 0.0, "assigned_coverage": 0.0, "flip_entropy": 0.0}

    cluster_counts = flips["cluster_id"].value_counts()
    top3 = int(cluster_counts.head(3).sum())
    probs = (cluster_counts / cluster_counts.sum()).to_numpy(dtype=float)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())

    if assigned_col in flips.columns:
        assigned_cov = float(flips[assigned_col].mean())
    else:
        assigned_cov = 0.0

    return {
        "flips": int(n_flips),
        "top3_coverage": float(top3 / n_flips),
        "assigned_coverage": assigned_cov,
        "flip_entropy": entropy,
    }


def main():
    args = parse_args()
    if args.noise_type != "sys_noise":
        raise ValueError("This script is designed for --noise_type sys_noise.")

    rate_tag = f"{args.noise_rate:.3f}".rstrip("0").rstrip(".")
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join("results", "sys_noise_viz", args.vul, f"seed_{args.seed}", f"rate_{rate_tag}")
    ensure_dir(out_dir)

    assigned_clusters_dict, global_cluster_map = coordinate_sys_noise_clusters(
        args.client_num,
        args.vul,
        args.noise_type,
        model_type=args.model_type,
        n_clusters=args.n_clusters,
        seed=int(args.seed),
        data_dir=args.data_dir,
    )
    if assigned_clusters_dict is None or global_cluster_map is None:
        raise RuntimeError("sys_noise cluster coordination failed.")

    client_cluster_counts = build_client_cluster_counts(
        args.client_num,
        args.model_type,
        args.vul,
        args.data_dir,
        global_cluster_map,
        args.n_clusters,
    )

    save_json(os.path.join(out_dir, "global_cluster_map.json"), {k: int(v) for k, v in global_cluster_map.items()})
    save_json(
        os.path.join(out_dir, "assigned_clusters_dict.json"),
        {str(k): [int(x) for x in v] for k, v in assigned_clusters_dict.items()},
    )
    save_json(
        os.path.join(out_dir, "client_cluster_counts.json"),
        {str(k): {str(c): int(n) for c, n in vv.items()} for k, vv in client_cluster_counts.items()},
    )

    pre_feature_dir = resolve_pretrain_feature_dir(args.data_dir, args.vul)
    all_rows = []

    for client_id in range(args.client_num):
        c_dir = client_dir_for(args.model_type, args.vul, client_id, args.data_dir)
        names_path = os.path.join(c_dir, "contract_name_train.txt")
        labels_path = os.path.join(c_dir, "label_train.csv")
        if not (os.path.exists(names_path) and os.path.exists(labels_path)):
            continue

        names, labels = read_names_labels(names_path, labels_path)
        unique_names, y_clean = unique_by_name(names, labels)

        assigned = assigned_clusters_dict.get(client_id, [])
        cluster_ids, y_noisy, flip_info, m_target = traced_sys_noise_on_unique(
            unique_names,
            y_clean,
            args.noise_rate,
            args.n_clusters,
            int(args.seed),
            assigned,
            global_cluster_map,
        )

        rng = np.random.RandomState(int(args.seed) + 10000 + client_id)
        is_flipped_random = np.zeros(len(unique_names), dtype=np.int64)
        n_flip_random = min(int((y_clean != y_noisy).sum()), len(unique_names))
        if n_flip_random > 0:
            idx_random = rng.choice(np.arange(len(unique_names)), size=n_flip_random, replace=False)
            is_flipped_random[idx_random] = 1

        rows = []
        assigned_set = set(int(x) % args.n_clusters for x in assigned)
        for i, name in enumerate(unique_names):
            flip_item = flip_info.get(i)
            is_flipped = int(y_clean[i] != y_noisy[i])
            rows.append(
                {
                    "client_id": client_id,
                    "name": name,
                    "cluster_id": int(cluster_ids[i]),
                    "y_clean": int(y_clean[i]),
                    "y_noisy": int(y_noisy[i]),
                    "is_flipped": is_flipped,
                    "is_flipped_random": int(is_flipped_random[i]),
                    "is_assigned_cluster": int(int(cluster_ids[i]) in assigned_set),
                    "flip_source": (flip_item["flip_source"] if flip_item is not None else "none"),
                    "m_target": int(m_target),
                }
            )

        df_client = pd.DataFrame(rows)
        df_client.to_csv(os.path.join(out_dir, f"client_{client_id}_noise_dump.csv"), index=False)
        all_rows.append(df_client)

    if not all_rows:
        raise RuntimeError("No client dump generated. Please check data paths and arguments.")

    df_all = pd.concat(all_rows, ignore_index=True)
    df_unique = df_all.drop_duplicates(subset=["name"]).copy()

    names_for_proj = df_unique["name"].tolist()
    feats = read_pretrain_feature(names_for_proj, pre_feature_dir)
    feats = StandardScaler().fit_transform(feats)
    xy = project_2d(feats, args.method, int(args.seed), args.perplexity)

    df_unique["x"] = xy[:, 0]
    df_unique["y"] = xy[:, 1]

    # Figure 1: sys vs random on same coordinates
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=160)
    cmap = plt.get_cmap("tab20")

    for ax, flip_col, title in [
        (axes[0], "is_flipped", "(a) systemetic noise"),
        (axes[1], "is_flipped_random", "(b) random noise (control)"),
    ]:
        colors = [cmap(int(c) % 20) if c >= 0 else (0.7, 0.7, 0.7, 1.0) for c in df_unique["cluster_id"]]
        ax.scatter(df_unique["x"], df_unique["y"], c=colors, s=12, alpha=0.5, linewidths=0)

        flipped = df_unique[df_unique[flip_col] == 1]
        ax.scatter(
            flipped["x"],
            flipped["y"],
            s=60,
            facecolors="none",
            edgecolors="red",
            linewidths=1.2,
            label="flipped",
        )
        ax.set_title(title)
        ax.set_xlabel("Dim-1")
        ax.set_ylabel("Dim-2")
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure1_sys_vs_random.png"))
    plt.close(fig)

    # Figure 2: client-cluster heatmap + assigned markers
    heat = np.zeros((args.client_num, args.n_clusters), dtype=float)
    for client_id in range(args.client_num):
        for c in range(args.n_clusters):
            heat[client_id, c] = client_cluster_counts.get(client_id, {}).get(c, 0)

    fig2, ax2 = plt.subplots(figsize=(max(10, args.n_clusters * 0.5), 1.8 + args.client_num * 0.6), dpi=160)
    if sns is not None:
        sns.heatmap(heat, cmap="YlOrRd", ax=ax2, cbar=True)
    else:
        im = ax2.imshow(heat, aspect="auto", cmap="YlOrRd")
        fig2.colorbar(im, ax=ax2)
    ax2.set_xlabel("cluster_id")
    ax2.set_ylabel("client_id")
    ax2.set_title("Client-Cluster sample counts with assigned clusters")

    for client_id, clusters in assigned_clusters_dict.items():
        for c in clusters:
            ax2.scatter(c + 0.5, client_id + 0.5, marker="*", s=100, c="deepskyblue", edgecolors="black", linewidths=0.4)

    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "figure2_client_cluster_heatmap.png"))
    plt.close(fig2)

    # Table 1: concentration metrics
    m_sys = concentration_metrics(df_unique, "is_flipped")
    m_rand = concentration_metrics(df_unique, "is_flipped_random")
    table = pd.DataFrame(
        [
            {"noise": "sys_noise", **m_sys},
            {"noise": "random_noise", **m_rand},
        ]
    )
    table.to_csv(os.path.join(out_dir, "table1_concentration.csv"), index=False)

    print(f"[Done] outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
