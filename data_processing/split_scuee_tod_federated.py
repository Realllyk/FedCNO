import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser("Split SC_UEE TOD for federated training")
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--out_root", type=str, default="./data/graduate_client_split/mando/tod")
    parser.add_argument("--client_num", type=int, default=4)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _write_split(df, root, name):
    names_path = os.path.join(root, f"contract_name_{name}.txt")
    labels_path = os.path.join(root, f"label_{name}.csv")
    with open(names_path, "w", encoding="utf-8") as f:
        for cid in df["contract_id"].tolist():
            f.write(f"{cid}\n")
    df["label"].astype(int).to_csv(labels_path, index=False, header=False)


def _write_client(df, root, client_id):
    cdir = os.path.join(root, f"client_{client_id}")
    os.makedirs(cdir, exist_ok=True)
    names_path = os.path.join(cdir, "contract_name_train.txt")
    labels_path = os.path.join(cdir, "label_train.csv")
    with open(names_path, "w", encoding="utf-8") as f:
        for cid in df["contract_id"].tolist():
            f.write(f"{cid}\n")
    df["label"].astype(int).to_csv(labels_path, index=False, header=False)


def main():
    args = parse_args()
    os.makedirs(args.out_root, exist_ok=True)
    df = pd.read_csv(args.manifest_csv)

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.0")

    train_df, temp_df = train_test_split(
        df,
        train_size=args.train_ratio,
        random_state=args.seed,
        stratify=df["label"],
    )
    val_share = args.val_ratio / (args.val_ratio + args.test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_share,
        random_state=args.seed,
        stratify=temp_df["label"],
    )

    _write_split(val_df, args.out_root, "valid")
    _write_split(test_df, args.out_root, "test")

    # Stratified split train into N clients: iterative split
    remain = train_df.copy()
    for i in range(args.client_num):
        if i == args.client_num - 1:
            part = remain
        else:
            part_ratio = 1.0 / (args.client_num - i)
            part, remain = train_test_split(
                remain,
                train_size=part_ratio,
                random_state=args.seed + i,
                stratify=remain["label"],
            )
        _write_client(part, args.out_root, i)

    print(f"Split done. train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")


if __name__ == "__main__":
    main()

