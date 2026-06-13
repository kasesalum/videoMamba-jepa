#!/usr/bin/env python3
"""
Generate space-delimited CSV file lists for Something-Something V2.

Output format (compatible with src/datasets/video_dataset.py):
    /absolute/path/to/video.webm 42

Example:
    python -m src.datasets.generate_ssv2_filelist \\
        --video-dir /data/SSv2/videos/20bn-something-something-v2 \\
        --labels-json /data/SSv2/labels/train.json \\
        --label-map /data/SSv2/labels/labels.json \\
        --output /data/SSv2/labels/SSv2_train_probe_filelist.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Iterable


def load_label_map(label_map_path: str) -> dict[str, int]:
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    return {str(k): int(v) for k, v in label_map.items()}


def template_to_label_key(template: str) -> str:
    return template.replace("[", "").replace("]", "")


def iter_entries(
    labels_json_path: str,
    video_dir: str,
    label_map: dict[str, int],
    video_ext: str,
) -> Iterable[tuple[str, int]]:
    with open(labels_json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    missing_labels = set()
    missing_videos = 0

    for record in records:
        video_id = record["id"]
        label_key = template_to_label_key(record["template"])
        if label_key not in label_map:
            missing_labels.add(label_key)
            continue

        video_path = os.path.join(video_dir, f"{video_id}{video_ext}")
        if not os.path.isfile(video_path):
            missing_videos += 1
            continue

        yield os.path.abspath(video_path), label_map[label_key]

    if missing_labels:
        print(
            f"Warning: {len(missing_labels)} template(s) missing from label map.",
            file=sys.stderr,
        )
    if missing_videos:
        print(
            f"Warning: skipped {missing_videos} record(s) with missing video files.",
            file=sys.stderr,
        )


def write_filelist(
    labels_json_path: str,
    video_dir: str,
    label_map_path: str,
    output_csv: str,
    video_ext: str = ".webm",
) -> int:
    label_map = load_label_map(label_map_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    count = 0
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=" ")
        for video_path, class_label in iter_entries(
            labels_json_path, video_dir, label_map, video_ext
        ):
            writer.writerow([video_path, class_label])
            count += 1

    print(f"Wrote {count} entries to {output_csv}")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SSv2 CSV file lists for V-JEPA training/evaluation."
    )
    parser.add_argument(
        "--video-dir",
        required=True,
        help="Directory containing SSv2 videos (e.g. .../20bn-something-something-v2).",
    )
    parser.add_argument(
        "--labels-json",
        required=True,
        help="Official SSv2 split JSON (train.json, validation.json, or test.json).",
    )
    parser.add_argument(
        "--label-map",
        required=True,
        help="Path to labels.json mapping template strings to integer class ids.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--video-ext",
        default=".webm",
        help="Video file extension including dot (default: .webm).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_filelist(
        labels_json_path=args.labels_json,
        video_dir=args.video_dir,
        label_map_path=args.label_map,
        output_csv=args.output,
        video_ext=args.video_ext,
    )


if __name__ == "__main__":
    main()
