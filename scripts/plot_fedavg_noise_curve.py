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
        "--fedavg_root",
        type=Path,
        default=Path("graduate_final_result") / "Fed_Avg",
        help="FedAvg result root directory.",
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
        "--csv_out",
        type=Path,
        default=None,
        help="Output CSV path. Default: analysis_csv/Fed_Avg/<model>/<noise_type>/<dataset>_lastN.csv",
    )
    parser.add_argument(
        "--fig_out",
        type=Path,
        default=None,
        help="Output figure path. Default: figure/Fed_Avg/<model>/<noise_type>/<dataset>_lastN.png",
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

    averaged["score"] = averaged["f1"]
    return averaged, len(latest)


def resolve_output_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    default_csv = Path("analysis_csv") / "Fed_Avg" / args.model_type / args.noise_type
    default_fig = Path("figure") / "Fed_Avg" / args.model_type / args.noise_type
    csv_out = args.csv_out or (default_csv / f"{args.dataset}_last{args.last_n}.csv")
    fig_out = args.fig_out or (default_fig / f"{args.dataset}_last{args.last_n}.png")
    return csv_out, fig_out


def to_rate_tag(rate: float) -> str:
    return f"{rate:.2f}".rstrip("0").rstrip(".")


def main() -> None:
    args = parse_args()
    noise_rates = [float(x.strip()) for x in args.noise_rates.split(",") if x.strip()]

    root = args.fedavg_root / args.model_type / args.noise_type
    if not root.exists():
        raise FileNotFoundError(f"Config path not found: {root}")

    rows = []
    for rate in noise_rates:
        rate_tag = to_rate_tag(rate)
        rate_dir = root / rate_tag
        if not rate_dir.exists():
            rows.append(
                {
                    "noise_rate": rate,
                    "used_runs": 0,
                    "acc": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "score": None,
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
                    "score": None,
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
                "score": averaged["score"],
                "result_file": str(result_file),
            }
        )

    csv_out, fig_out = resolve_output_paths(args)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    fig_out.parent.mkdir(parents=True, exist_ok=True)

    with csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["noise_rate", "used_runs", "acc", "precision", "recall", "f1", "score", "result_file"],
        )
        writer.writeheader()
        writer.writerows(rows)

    x = [r["noise_rate"] for r in rows]
    plt.figure(figsize=(8, 5))
    for metric in ["acc", "precision", "recall", "f1", "score"]:
        y = [safe_float(r[metric]) for r in rows]
        plt.plot(x, y, marker="o", label=metric)

    plt.title(
        f"Fed_Avg | {args.model_type} | {args.noise_type} | {args.dataset} | latest {args.last_n} mean"
    )
    plt.xlabel("noise_rate")
    plt.ylabel("metric")
    plt.xticks(noise_rates)
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_out, dpi=200)
    plt.close()

    print(f"[OK] CSV saved: {csv_out}")
    print(f"[OK] Figure saved: {fig_out}")


if __name__ == "__main__":
    main()
