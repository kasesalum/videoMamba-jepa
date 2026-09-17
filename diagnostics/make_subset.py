import argparse
import random
from pathlib import Path


def _read_manifest(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"{path}:{line_no} must contain at least '<video_path> <label>'")
            rows.append((parts[0], parts[1]))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Create a fixed-seed subset manifest for small VideoMamba-JEPA sanity runs."
    )
    parser.add_argument("--input", required=True, help="Source manifest in '<video_path> <label>' format.")
    parser.add_argument("--output", required=True, help="Output subset manifest path.")
    parser.add_argument("--num-clips", type=int, default=128, help="Number of rows to sample.")
    parser.add_argument("--seed", type=int, default=234, help="Sampling seed.")
    parser.add_argument(
        "--require-existing",
        action="store_true",
        help="Drop rows whose video path does not exist before sampling.",
    )
    args = parser.parse_args()

    rows = _read_manifest(args.input)
    if args.require_existing:
        rows = [row for row in rows if Path(row[0]).exists()]
    if not rows:
        raise SystemExit("No usable rows found in source manifest.")

    rng = random.Random(args.seed)
    if args.num_clips >= len(rows):
        subset = list(rows)
        rng.shuffle(subset)
    else:
        subset = rng.sample(rows, args.num_clips)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="\n") as handle:
        for video_path, label in subset:
            handle.write(f"{video_path} {label}\n")

    print(
        {
            "input": str(args.input),
            "output": str(output),
            "source_rows": len(rows),
            "subset_rows": len(subset),
            "seed": args.seed,
        }
    )


if __name__ == "__main__":
    main()
