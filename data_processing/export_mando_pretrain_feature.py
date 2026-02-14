import argparse
import os
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser("Export pretrain_feature vectors from MANDO graph cache")
    parser.add_argument("--graph_dir", type=str, default="./data/mando_graph/tod")
    parser.add_argument("--out_dir", type=str, default="./data/pretrain_feature/tod")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    files = [x for x in os.listdir(args.graph_dir) if x.endswith(".pt")]
    for fn in files:
        cid = os.path.splitext(fn)[0]
        path = os.path.join(args.graph_dir, fn)
        item = torch.load(path, map_location="cpu")
        node_features = item["node_features"].float()
        emb = node_features.mean(dim=0).numpy()
        out_path = os.path.join(args.out_dir, f"{cid}.txt")
        np.savetxt(out_path, emb)
    print(f"Exported {len(files)} feature files to {args.out_dir}")


if __name__ == "__main__":
    main()

