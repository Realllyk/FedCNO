import argparse
import hashlib
import os
import re
import torch
import pandas as pd


KEYWORDS = [
    "function",
    "contract",
    "modifier",
    "mapping",
    "event",
    "emit",
    "require",
    "assert",
    "call",
    "delegatecall",
    "transfer",
    "send",
]


def parse_args():
    parser = argparse.ArgumentParser("Build lightweight MANDO graph cache from Solidity")
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./data/mando_graph/tod")
    parser.add_argument("--max_lines", type=int, default=256)
    parser.add_argument("--hash_dim", type=int, default=64)
    parser.add_argument("--feature_dim", type=int, default=80)
    parser.add_argument("--failed_csv", type=str, default="./data/mando_graph/tod_failed.csv")
    return parser.parse_args()


def _hash_bucket(token, mod):
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % mod


def _line_feature(line, hash_dim):
    text = line.strip().lower()
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text)
    vec = torch.zeros(hash_dim + len(KEYWORDS) + 4, dtype=torch.float32)
    for t in tokens:
        vec[_hash_bucket(t, hash_dim)] += 1.0
    for i, kw in enumerate(KEYWORDS):
        if kw in text:
            vec[hash_dim + i] = 1.0
    vec[hash_dim + len(KEYWORDS) + 0] = float(len(tokens))
    vec[hash_dim + len(KEYWORDS) + 1] = float(len(text))
    vec[hash_dim + len(KEYWORDS) + 2] = 1.0 if "{" in text else 0.0
    vec[hash_dim + len(KEYWORDS) + 3] = 1.0 if "}" in text else 0.0
    return vec


def _build_graph(sol_path, max_lines, hash_dim):
    with open(sol_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    lines = lines[:max_lines]
    if len(lines) == 0:
        lines = [""]
    node_features = torch.stack([_line_feature(line, hash_dim) for line in lines], dim=0)
    return {"node_features": node_features}


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.manifest_csv)
    failed = []

    for _, row in df.iterrows():
        cid = str(row["contract_id"])
        sol_path = str(row["sol_path"])
        out_path = os.path.join(args.out_dir, f"{cid}.pt")
        try:
            graph = _build_graph(sol_path, args.max_lines, args.hash_dim)
            torch.save(graph, out_path)
        except Exception as exc:
            failed.append({"contract_id": cid, "sol_path": sol_path, "error": str(exc)})

    if failed:
        fdf = pd.DataFrame(failed)
        os.makedirs(os.path.dirname(os.path.abspath(args.failed_csv)), exist_ok=True)
        fdf.to_csv(args.failed_csv, index=False)
        print(f"Graph cache done with failures: {len(failed)}")
    else:
        print("Graph cache done without failures.")


if __name__ == "__main__":
    main()

