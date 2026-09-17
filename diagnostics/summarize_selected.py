import argparse
import json
from pathlib import Path

import pandas as pd


def _summarize_csv(path):
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} has no rows")
    row = df.iloc[-1]
    return {
        "csv": str(path),
        "run": path.stem,
        "epoch": int(row["epoch"]),
        "itr": int(row["itr"]),
        "loss": float(row["loss"]),
        "loss_jepa": float(row["loss-jepa"]),
        "reg_loss": float(row["reg-loss"]),
        "target_var": float(row["target-var"]),
        "pred_var": float(row["pred-var"]),
        "enc_grad_norm": float(row["enc-grad-norm"]),
        "pred_grad_norm": float(row["pred-grad-norm"]),
        "mem_mb": float(row["mem-mb"]),
        "clips_per_sec": float(row["clips-per-sec"]),
        "total_device_hours": float(df["device-hours"].sum()),
        "wall_time_ms": float(row["wall-time(ms)"]),
        "gpu_time_ms": float(row["gpu-time(ms)"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize an explicit list of diagnostic CSV files.")
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = [_summarize_csv(Path(path)) for path in args.csv]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
