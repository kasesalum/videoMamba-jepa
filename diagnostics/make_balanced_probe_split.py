import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


def _read_manifest(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=" ")
        for row in reader:
            if not row:
                continue
            rows.append((row[0], str(row[1])))
    return rows


def _write_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for video_path, label in rows:
            handle.write(f"{video_path} {label}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create class-balanced train/eval manifests for frozen probes.")
    parser.add_argument("--input", required=True, help="Input manifest.")
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--train-per-class", type=int, default=3)
    parser.add_argument("--eval-per-class", type=int, default=1)
    parser.add_argument("--max-classes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=234)
    args = parser.parse_args()

    rows = _read_manifest(args.input)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[1]].append(row)

    required = args.train_per_class + args.eval_per_class
    counts = Counter({label: len(items) for label, items in grouped.items()})
    labels = [
        label for label, count in counts.most_common()
        if count >= required
    ][:args.max_classes]
    if not labels:
        raise SystemExit(
            f"No labels in {args.input} have at least {required} clips.")

    rng = random.Random(args.seed)
    train_rows = []
    eval_rows = []
    selected_counts = {}
    for label in sorted(labels):
        items = list(grouped[label])
        rng.shuffle(items)
        train = items[:args.train_per_class]
        eval_ = items[args.train_per_class:required]
        train_rows.extend(train)
        eval_rows.extend(eval_)
        selected_counts[label] = len(items)

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)
    _write_manifest(args.train_output, train_rows)
    _write_manifest(args.eval_output, eval_rows)

    print({
        "input": args.input,
        "train_output": args.train_output,
        "eval_output": args.eval_output,
        "classes": len(labels),
        "train_clips": len(train_rows),
        "eval_clips": len(eval_rows),
        "train_per_class": args.train_per_class,
        "eval_per_class": args.eval_per_class,
        "seed": args.seed,
        "selected_label_counts": selected_counts,
    })


if __name__ == "__main__":
    main()
