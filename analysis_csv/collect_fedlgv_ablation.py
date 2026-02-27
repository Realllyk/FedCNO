import json
import math
import statistics
from pathlib import Path

import pandas as pd

METRICS = [
    "Accuracy",
    "Precision",
    "Recall(TPR)",
    "F1 score",
    "False positive rate(FPR)",
    "False negative rate(FNR)",
]

BASE_DIR = Path("graduate_result")
OUTPUT = Path("analysis_csv/Fed_LGV_ablation_compare.csv")

LAB_MAP = {
    "Full": "Fed_LGV",
    "No Local": "lgv_abl_no_local",
    "No Global": "lgv_abl_no_global",
    "No Unc Alpha": "lgv_abl_no_unc_alpha",
    "No Cons Loss": "lgv_abl_no_cons_loss",
    "No Local + No Global": "lgv_abl_no_local_no_global",
    "No Global + No Unc Alpha": "lgv_abl_no_global_no_unc_alpha",
}


def normalize_vul(filename: str) -> str:
    name = filename
    for prefix in ("fn_", "diff_", "sys_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    if name.endswith("_result.json"):
        name = name[: -len("_result.json")]
    if name.endswith("_test"):
        name = name[: -len("_test")]
    return name


def load_records(json_file: Path):
    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    return data


def collect_for_lab(label: str, lab_dir_name: str):
    lab_dir = BASE_DIR / lab_dir_name
    rows = []
    if not lab_dir.exists():
        return rows

    for model_dir in sorted(lab_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_type = model_dir.name

        for noise_dir in sorted(model_dir.iterdir()):
            if not noise_dir.is_dir():
                continue
            noise_type = noise_dir.name

            for rate_dir in sorted(noise_dir.iterdir()):
                if not rate_dir.is_dir():
                    continue
                noise_rate = rate_dir.name

                for json_file in sorted(rate_dir.glob("*.json")):
                    if json_file.name.endswith("_valid.json"):
                        continue

                    vulnerability = normalize_vul(json_file.name)
                    records = load_records(json_file)

                    for metric in METRICS:
                        values = [r.get(metric) for r in records if metric in r]
                        values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
                        if not values:
                            continue

                        rows.append(
                            {
                                "vulnerability": vulnerability,
                                "model_type": model_type,
                                "noise_type": noise_type,
                                "noise_rate": noise_rate,
                                "method": label,
                                "metric": metric,
                                "mean": statistics.mean(values),
                                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                                "n": len(values),
                            }
                        )

    return rows


def main():
    all_rows = []
    for label, lab_dir in LAB_MAP.items():
        all_rows.extend(collect_for_lab(label, lab_dir))

    if not all_rows:
        print("No result rows found. Check graduate_result paths.")
        return

    df = pd.DataFrame(all_rows)
    df = df.sort_values(
        by=["vulnerability", "model_type", "noise_type", "noise_rate", "metric", "method"]
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Saved: {OUTPUT}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
