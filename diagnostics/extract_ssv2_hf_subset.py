import argparse
import json
import tarfile
from pathlib import Path


def _load_train_labels(train_json, labels_json):
    with open(train_json, "r", encoding="utf-8") as handle:
        train_rows = json.load(handle)
    with open(labels_json, "r", encoding="utf-8") as handle:
        label_map = json.load(handle)

    id_to_label = {}
    for row in train_rows:
        template = row["template"].replace("[", "").replace("]", "")
        if template not in label_map:
            raise KeyError(f"Template not found in labels.json: {row['template']}")
        id_to_label[str(row["id"])] = str(label_map[template])
    return id_to_label


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract a small train-labelled SSv2 subset from a Hugging Face "
            "Something-Something-v2 split archive part."
        )
    )
    parser.add_argument(
        "--archive-part",
        default="diagnostic_runs/data/ssv2_hf_raw/videos/20bn-something-something-v2-00",
        help="Path to one split gzip-tar archive part from morpheushoc/something-something-v2.",
    )
    parser.add_argument(
        "--train-json",
        default="diagnostic_runs/data/ssv2_hf_raw/train.json",
        help="Path to SSv2 train.json.",
    )
    parser.add_argument(
        "--labels-json",
        default="diagnostic_runs/data/ssv2_hf_raw/labels.json",
        help="Path to SSv2 labels.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="diagnostic_runs/data/ssv2_subset/videos",
        help="Directory where extracted webm files will be written.",
    )
    parser.add_argument(
        "--manifest",
        default="diagnostic_runs/data/ssv2_subset/ssv2_train_128.csv",
        help="Output manifest in '<video_path> <label>' format.",
    )
    parser.add_argument("--num-clips", type=int, default=128)
    args = parser.parse_args()

    archive_part = Path(args.archive_part)
    output_dir = Path(args.output_dir)
    manifest = Path(args.manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    id_to_label = _load_train_labels(args.train_json, args.labels_json)
    rows = []

    # The SSv2 video archive is a split gzip-tar stream. Opening in stream mode
    # lets us extract early members from part 00 without reading all 20 parts.
    with tarfile.open(archive_part, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".webm"):
                continue

            video_id = Path(member.name).stem
            if video_id not in id_to_label:
                continue

            target = output_dir / f"{video_id}.webm"
            source = tar.extractfile(member)
            if source is None:
                continue
            with source, open(target, "wb") as handle:
                handle.write(source.read())

            rows.append((target.as_posix(), id_to_label[video_id]))
            if len(rows) >= args.num_clips:
                break

    if len(rows) < args.num_clips:
        raise SystemExit(
            f"Only extracted {len(rows)} train-labelled clips from {archive_part}; "
            "download a later archive part or lower --num-clips."
        )

    with open(manifest, "w", encoding="utf-8", newline="\n") as handle:
        for video_path, label in rows:
            handle.write(f"{video_path} {label}\n")

    print(
        {
            "archive_part": str(archive_part),
            "output_dir": str(output_dir),
            "manifest": str(manifest),
            "num_clips": len(rows),
        }
    )


if __name__ == "__main__":
    main()
