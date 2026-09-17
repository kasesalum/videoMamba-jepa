import argparse
import csv
import json
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize enriched V-JEPA diagnostic CSV logs.')
    parser.add_argument(
        '--root',
        default='diagnostic_runs',
        help='Root folder containing run CSV files.')
    parser.add_argument(
        '--output',
        default='diagnostic_runs/summary.json',
        help='Summary JSON path.')
    return parser.parse_args()


def _read_rows(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row and row.get('epoch') != 'epoch':
                rows.append(row)
    return rows


def _as_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def main():
    args = _parse_args()
    root = Path(args.root)
    summaries = []
    for csv_path in sorted(root.rglob('*.csv')):
        rows = _read_rows(csv_path)
        if not rows:
            continue
        row = rows[-1]
        total_device_hours = sum(_as_float(r, 'device-hours') for r in rows)
        summaries.append({
            'csv': str(csv_path),
            'run': csv_path.stem,
            'epoch': int(float(row.get('epoch', 0))),
            'itr': int(float(row.get('itr', 0))),
            'loss': _as_float(row, 'loss'),
            'loss_jepa': _as_float(row, 'loss-jepa'),
            'reg_loss': _as_float(row, 'reg-loss'),
            'target_var': _as_float(row, 'target-var'),
            'pred_var': _as_float(row, 'pred-var'),
            'target_abs_mean': _as_float(row, 'target-abs-mean'),
            'pred_abs_mean': _as_float(row, 'pred-abs-mean'),
            'clips_per_sec': _as_float(row, 'clips-per-sec'),
            'total_device_hours': total_device_hours,
            'wall_time_ms': _as_float(row, 'wall-time(ms)'),
            'gpu_time_ms': _as_float(row, 'gpu-time(ms)'),
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
