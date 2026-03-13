import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


METRIC_KEYS = {
    "acc": "Accuracy",
    "precision": "Precision",
    "recall": "Recall(TPR)",
    "f1": "F1 score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize latest N results for all labs under base_dir by one fixed config "
            "(model_type, noise_type, noise_rate, vul)."
        )
    )
    parser.add_argument("--base_dir", type=Path, required=True, help="Base result directory.")
    parser.add_argument("--model_type", type=str, required=True, help="Model type, e.g., CBGRU/CGE/MANDO.")
    parser.add_argument("--noise_type", type=str, required=True, help="Noise type, e.g., non_noise/sys_noise.")
    parser.add_argument("--noise_rate", type=float, required=True, help="Noise rate, e.g., 0.3.")
    parser.add_argument("--vul", type=str, required=True, help="Vul/dataset name, e.g., reentrancy/timestamp/tod.")
    parser.add_argument("--last_n", type=int, default=3, help="Use latest N records per lab.")
    return parser.parse_args()


def safe_float(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def parse_time_key(record: Dict, fallback_index: int) -> Tuple[str, int]:
    return str(record.get("time", "")), fallback_index


def load_json_records(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def rate_tags(noise_rate: float) -> List[str]:
    tags = [
        f"{noise_rate:.2f}".rstrip("0").rstrip("."),
        f"{noise_rate:.1f}",
        f"{noise_rate:.2f}",
        str(noise_rate),
    ]
    deduped = []
    for t in tags:
        if t not in deduped:
            deduped.append(t)
    return deduped


def resolve_result_file(
    lab_dir: Path, model_type: str, noise_type: str, noise_rate: float, vul: str
) -> Optional[Path]:
    if noise_type == "sys_noise":
        file_name = f"sys_{vul}_result.json"
    else:
        file_name = f"{vul}_result.json"

    for tag in rate_tags(noise_rate):
        candidate = lab_dir / model_type / noise_type / tag / file_name
        if candidate.exists():
            return candidate
    return None


def average_latest_metrics(result_file: Path, last_n: int) -> Tuple[Dict[str, Optional[float]], int, str]:
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
    used_times = []
    for rec in latest:
        used_times.append(str(rec.get("time", "")))
        for metric_name, json_key in METRIC_KEYS.items():
            val = safe_float(rec.get(json_key))
            if val is not None:
                metric_values[metric_name].append(val)

    averaged: Dict[str, Optional[float]] = {}
    for metric_name, values in metric_values.items():
        averaged[metric_name] = sum(values) / len(values) if values else None

    return averaged, len(latest), "|".join(used_times)


def normalize_lab_name(name: str) -> str:
    return name.replace("_", "").replace("-", "").lower()


def lab_sort_key(name: str) -> Tuple[int, str]:
    norm = normalize_lab_name(name)
    if norm in {"fedavg"}:
        return (0, norm)
    if norm in {"fedcno"}:
        return (2, norm)
    if norm in {"fedcrd"}:
        return (3, norm)
    return (1, norm)


def main() -> None:
    args = parse_args()

    base_dir = args.base_dir
    if not base_dir.exists():
        raise FileNotFoundError(f"base_dir not found: {base_dir}")

    rows: List[Dict] = []
    for item in sorted(base_dir.iterdir()):
        if not item.is_dir():
            continue

        result_file = resolve_result_file(item, args.model_type, args.noise_type, args.noise_rate, args.vul)
        if result_file is None:
            continue

        averaged, used_runs, _ = average_latest_metrics(result_file, args.last_n)
        rows.append(
            {
                "lab_name": item.name,
                "model_type": args.model_type,
                "noise_type": args.noise_type,
                "noise_rate": args.noise_rate,
                "vul": args.vul,
                "used_runs": used_runs,
                "acc": averaged["acc"],
                "precision": averaged["precision"],
                "recall": averaged["recall"],
                "f1": averaged["f1"],
            }
        )

    rows.sort(key=lambda x: lab_sort_key(x["lab_name"]))

    output_dir = Path("graduate_final_result") / "analysis_csv" / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    rate_tag = f"{args.noise_rate:.2f}".rstrip("0").rstrip(".")
    output_file = (
        output_dir
        / f"summary_{args.model_type}_{args.noise_type}_{rate_tag}_{args.vul}_last{args.last_n}.csv"
    )

    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lab_name",
                "model_type",
                "noise_type",
                "noise_rate",
                "vul",
                "used_runs",
                "acc",
                "precision",
                "recall",
                "f1",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Labs found: {len(rows)}")
    print(f"[OK] CSV saved: {output_file}")


if __name__ == "__main__":
    main()
