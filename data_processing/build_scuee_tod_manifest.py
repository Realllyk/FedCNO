import argparse
import glob
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser("Build SC_UEE TOD manifest")
    parser.add_argument("--sol_dir", type=str, required=True, help="Directory containing .sol files")
    parser.add_argument(
        "--label_file",
        type=str,
        default=None,
        help="Optional CSV/TSV with contract_id and label. If omitted, labels are inferred from subfolder names.",
    )
    parser.add_argument("--id_col", type=str, default="contract_id")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument(
        "--out_csv",
        type=str,
        default="./data/graduate_client_split/mando/scuee_tod_manifest.csv",
        help="Output manifest CSV path",
    )
    return parser.parse_args()


def _read_label_table(path):
    # csv 用逗号，其它格式交给 pandas 自动推断分隔符
    sep = "," if path.lower().endswith(".csv") else None
    return pd.read_csv(path, sep=sep, engine="python")


def _label_from_parent_dir(sol_path):
    # SC_UEE TOD_sourcecode: exploitable=1, unexploitable=0
    parent = os.path.basename(os.path.dirname(sol_path)).lower()
    if parent == "exploitable":
        return 1
    if parent == "unexploitable":
        return 0
    raise ValueError(f"Unsupported parent folder for label inference: {parent} ({sol_path})")


def main():
    args = parse_args()
    use_label_file = bool(args.label_file)

    id_to_label = None
    if use_label_file:
        df = _read_label_table(args.label_file)
        if args.id_col not in df.columns or args.label_col not in df.columns:
            raise ValueError(f"Missing columns. need: {args.id_col}, {args.label_col}")
        # 形成 contract_id -> 0/1 标签映射
        id_to_label = dict(zip(df[args.id_col].astype(str), df[args.label_col].astype(int)))

    records = []

    # 递归读取 .sol，兼容 SC_UEE 的 exploitable/unexploitable 子目录结构
    for path in glob.glob(os.path.join(args.sol_dir, "**", "*.sol"), recursive=True):
        cid = os.path.splitext(os.path.basename(path))[0]

        if use_label_file:
            if cid not in id_to_label:
                # 无标签样本直接跳过
                continue
            label = int(id_to_label[cid])
        else:
            label = _label_from_parent_dir(path)

        records.append(
            {
                "contract_id": cid,
                "sol_path": os.path.abspath(path),
                "label": label,
                # 仅语义标识，训练仍使用二分类 label
                "vul_name": "front_running",
            }
        )

    if not records:
        raise ValueError("No .sol samples were collected. Please check --sol_dir/--label_file.")

    out_df = pd.DataFrame(records).sort_values("contract_id").reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Saved manifest: {args.out_csv}, samples={len(out_df)}")


if __name__ == "__main__":
    main()
