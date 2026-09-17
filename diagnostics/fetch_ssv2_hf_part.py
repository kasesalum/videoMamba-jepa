import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_FILES = [
    "labels.json",
    "train.json",
    "validation.json",
    "20bn-something-something_download_instructions_-_091622.pdf",
    "jester_something_something_exercise_research_license_revised_final_qti_28jul2022.pdf",
]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download SSv2 annotations and selected video archive parts from "
            "the morpheushoc/something-something-v2 Hugging Face mirror."
        )
    )
    parser.add_argument(
        "--repo-id",
        default="morpheushoc/something-something-v2",
        help="Hugging Face dataset repository.",
    )
    parser.add_argument(
        "--output-dir",
        default="diagnostic_runs/data/ssv2_hf_raw",
        help="Local destination for downloaded files.",
    )
    parser.add_argument(
        "--parts",
        nargs="*",
        default=["00"],
        help="Archive part suffixes to fetch, e.g. 00 01 02. Part 00 is 1 GB.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for filename in DEFAULT_FILES:
        downloaded.append(
            hf_hub_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                filename=filename,
                local_dir=str(output_dir),
            )
        )

    for part in args.parts:
        filename = f"videos/20bn-something-something-v2-{part}"
        downloaded.append(
            hf_hub_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                filename=filename,
                local_dir=str(output_dir),
            )
        )

    print({"repo_id": args.repo_id, "output_dir": str(output_dir), "files": downloaded})


if __name__ == "__main__":
    main()
