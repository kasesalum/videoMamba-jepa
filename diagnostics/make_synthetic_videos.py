import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def _frame(index, frame_index, size):
    y, x = np.mgrid[0:size, 0:size]
    shift = (frame_index * 5 + index * 11) % size
    red = (x + shift) % 256
    green = (y * 2 + index * 17) % 256
    blue = ((x // 8 + y // 8 + frame_index + index) % 2) * 180 + 40
    return np.stack([red, green, blue], axis=-1).astype(np.uint8)


def _write_video(path, index, frames, size, fps):
    video = [_frame(index, frame_index, size) for frame_index in range(frames)]
    imageio.mimsave(path, video, fps=fps, macro_block_size=16)


def main():
    parser = argparse.ArgumentParser(
        description="Generate tiny MP4 videos and a manifest for local diagnostic trainer sanity checks."
    )
    parser.add_argument("--output-dir", default="diagnostic_runs/data/generated_videos")
    parser.add_argument("--manifest", default="diagnostic_runs/data/generated_videos.csv")
    parser.add_argument("--num-videos", type=int, default=8)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute paths. Leave off when the working directory path contains spaces.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for index in range(args.num_videos):
        path = output_dir / f"synthetic_{index:04d}.mp4"
        _write_video(path, index=index, frames=args.frames, size=args.size, fps=args.fps)
        manifest_path = path.resolve() if args.absolute_paths else path
        rows.append((manifest_path, index % 4))

    with open(manifest, "w", encoding="utf-8", newline="\n") as handle:
        for path, label in rows:
            handle.write(f"{path} {label}\n")

    print(
        {
            "manifest": str(manifest),
            "videos": len(rows),
            "frames": args.frames,
            "size": args.size,
            "fps": args.fps,
        }
    )


if __name__ == "__main__":
    main()
