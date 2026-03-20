import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


DEFAULT_NOISE_RATES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
METRIC_KEYS = {
    "acc": "Accuracy",
    "precision": "Precision",
    "recall": "Recall(TPR)",
    "f1": "F1 score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For one FedAvg config (dataset/model_type/noise_type), select the latest N "
            "results at each noise_rate, average metrics, and draw noise-rate curves."
        )
    )
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=Path("graduate_final_result"),
        help="Base directory for both input and output paths, e.g., graduate_final_result.",
    )
    parser.add_argument("--model_type", type=str, required=True, help="Model type, e.g., CBGRU/CGE/MANDO.")
    parser.add_argument("--noise_type", type=str, required=True, help="Noise type, e.g., non_noise/sys_noise.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset/vul name, e.g., reentrancy/timestamp/tod.")
    parser.add_argument("--last_n", type=int, default=3, help="Number of latest runs to average at each noise rate.")
    parser.add_argument(
        "--noise_rates",
        type=str,
        default=",".join(str(x) for x in DEFAULT_NOISE_RATES),
        help="Comma-separated noise rates, e.g., 0.0,0.05,0.1,0.15,0.2,0.25,0.3.",
    )
    parser.add_argument(
        "--from_csv",
        action="store_true",
        help="If set, read existing CSV files and draw figure directly without recomputing from JSON.",
    )
    return parser.parse_args()


def safe_float(value) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num):
        return None
    return num


def parse_time_key(record: Dict, fallback_index: int) -> Tuple[str, int]:
    time_str = str(record.get("time", ""))
    return time_str, fallback_index


def load_json_records(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def pick_result_file(rate_dir: Path, dataset: str) -> Optional[Path]:
    dataset = dataset.lower()
    candidates = sorted(
        p for p in rate_dir.glob("*_result.json") if not p.name.endswith("_valid.json")
    )
    if not candidates:
        return None

    name_matches = [p for p in candidates if dataset in p.stem.lower()]
    if len(name_matches) == 1:
        return name_matches[0]
    if len(name_matches) > 1:
        return sorted(name_matches)[0]

    content_matches: List[Path] = []
    for path in candidates:
        try:
            records = load_json_records(path)
        except Exception:
            continue
        for rec in records[:5]:
            hparams = rec.get("hparams", {})
            vul = str(hparams.get("vul", "")).lower()
            if vul == dataset:
                content_matches.append(path)
                break

    if len(content_matches) == 1:
        return content_matches[0]
    if len(content_matches) > 1:
        return sorted(content_matches)[0]
    return None


def average_latest_metrics(result_file: Path, last_n: int) -> Tuple[Dict[str, Optional[float]], int]:
    records = load_json_records(result_file)
    test_records = [r for r in records if str(r.get("tag", "")).lower() == "test"]
    if not test_records:
        test_records = records

    sorted_records = sorted(
        enumerate(test_records),
        key=lambda x: parse_time_key(x[1], x[0]),
    )
    latest = [x[1] for x in sorted_records[-last_n:]]

    metric_values: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}
    for rec in latest:
        for metric_name, json_key in METRIC_KEYS.items():
            val = safe_float(rec.get(json_key))
            if val is not None:
                metric_values[metric_name].append(val)

    averaged: Dict[str, Optional[float]] = {}
    for metric_name, values in metric_values.items():
        averaged[metric_name] = sum(values) / len(values) if values else None

    return averaged, len(latest)


def resolve_output_dirs(args: argparse.Namespace) -> Tuple[Path, Path]:
    default_csv = args.base_dir / "analysis_csv" / "Fed_Avg" / args.model_type / args.noise_type
    default_fig = args.base_dir / "figure" / "Fed_Avg" / args.model_type / args.noise_type
    return default_csv, default_fig


def to_rate_tag(rate: float) -> str:
    return f"{rate:.2f}".rstrip("0").rstrip(".")


def resolve_rate_dir(root: Path, rate: float) -> Optional[Path]:
    candidates = []
    for tag in [to_rate_tag(rate), f"{rate:.1f}", f"{rate:.2f}", str(rate)]:
        if tag not in candidates:
            candidates.append(tag)

    for tag in candidates:
        rate_dir = root / tag
        if rate_dir.exists():
            return rate_dir
    return None


def write_metric_csv(rows: List[Dict], metric: str, csv_out: Path) -> None:
    with csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["noise_rate", "used_runs", metric, "result_file"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "noise_rate": row["noise_rate"],
                    "used_runs": row["used_runs"],
                    metric: row[metric],
                    "result_file": row["result_file"],
                }
            )


def load_rows_from_csv(args: argparse.Namespace, noise_rates: List[float]) -> List[Dict]:
    csv_dir, _ = resolve_output_dirs(args)
    base_rows: Dict[str, Dict] = {
        to_rate_tag(rate): {
            "noise_rate": rate,
            "used_runs": 0,
            "acc": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "result_file": "",
        }
        for rate in noise_rates
    }

    for metric in ["acc", "precision", "recall", "f1"]:
        csv_file = csv_dir / f"{args.dataset}_{metric}_last{args.last_n}.csv"
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        with csv_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rate = safe_float(row.get("noise_rate"))
                if rate is None:
                    continue
                rate_key = to_rate_tag(rate)
                if rate_key not in base_rows:
                    continue

                metric_value = safe_float(row.get(metric))
                if metric_value is not None:
                    base_rows[rate_key][metric] = metric_value

                used_runs = safe_float(row.get("used_runs"))
                if used_runs is not None:
                    base_rows[rate_key]["used_runs"] = max(base_rows[rate_key]["used_runs"], int(used_runs))

                result_file = str(row.get("result_file", "") or "")
                if result_file:
                    base_rows[rate_key]["result_file"] = result_file

    return [base_rows[to_rate_tag(rate)] for rate in noise_rates]


def collect_rows_from_json(args: argparse.Namespace, noise_rates: List[float]) -> List[Dict]:
    root = args.base_dir / "Fed_Avg" / args.model_type / args.noise_type
    if not root.exists():
        raise FileNotFoundError(f"Config path not found: {root}")

    rows = []
    for rate in noise_rates:
        rate_dir = resolve_rate_dir(root, rate)
        if rate_dir is None:
            rows.append(
                {
                    "noise_rate": rate,
                    "used_runs": 0,
                    "acc": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "result_file": "",
                }
            )
            continue

        result_file = pick_result_file(rate_dir, args.dataset)
        if result_file is None:
            rows.append(
                {
                    "noise_rate": rate,
                    "used_runs": 0,
                    "acc": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "result_file": "",
                }
            )
            continue

        averaged, used_runs = average_latest_metrics(result_file, args.last_n)
        rows.append(
            {
                "noise_rate": rate,
                "used_runs": used_runs,
                "acc": averaged["acc"],
                "precision": averaged["precision"],
                "recall": averaged["recall"],
                "f1": averaged["f1"],
                "result_file": str(result_file),
            }
        )

    return rows


def plot_metrics_figure(rows: List[Dict], noise_rates: List[float], fig_out: Path, args: argparse.Namespace) -> None:
    metrics = ["acc", "precision", "f1", "recall"]
    metric_titles = {
        "acc": "Accuracy(%)",
        "precision": "Precision(%)",
        "f1": "F1 Score(%)",
        "recall": "Recall(%)",
    }
    x = [r["noise_rate"] * 100 for r in rows]
    x_ticks = [r * 100 for r in noise_rates]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for ax, metric in zip(axes, metrics):
        y = [safe_float(r[metric]) * 100 if safe_float(r[metric]) is not None else None for r in rows]
        ax.plot(x, y, marker="o")
        ax.set_title(metric_titles[metric])
        ax.set_xlabel("Noise Level(%)")
        ax.set_ylabel("Metric Value(%)")
        ax.set_xticks(x_ticks)
        ax.set_ylim(0.0, 100.0)
        # Keep each subplot coordinate box in 1:1 shape.
        ax.set_box_aspect(1)

    fig.tight_layout()
    fig.savefig(fig_out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    noise_rates = [float(x.strip()) for x in args.noise_rates.split(",") if x.strip()]

    if args.from_csv:
        rows = load_rows_from_csv(args, noise_rates)
    else:
        rows = collect_rows_from_json(args, noise_rates)

    csv_dir, fig_dir = resolve_output_dirs(args)
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not args.from_csv:
        for metric in ["acc", "precision", "recall", "f1"]:
            csv_out = csv_dir / f"{args.dataset}_{metric}_last{args.last_n}.csv"
            write_metric_csv(rows, metric, csv_out)
            print(f"[OK] CSV saved: {csv_out}")

    fig_out = fig_dir / f"{args.dataset}_metrics_last{args.last_n}.png"
    plot_metrics_figure(rows, noise_rates, fig_out, args)
    print(f"[OK] Figure saved: {fig_out}")


if __name__ == "__main__":
    main()
