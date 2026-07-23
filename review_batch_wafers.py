"""Run defect detection/review separately for every wafer in batch_wafers.txt.

Each subprocess sees only one wafer's 76 cell images and builds its own normal
template, detector JSON, reviewed GDS JSON, review state, and reviewed wafer
stitch.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from batch_wafers_parser import parse_batch_file
from wafer_run_layout import (
    count_analysis_images,
    detector_paths,
    select_wafer_ids,
)


def _quote_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def build_detector_command(
    *,
    python_executable: str,
    detector_script: Path,
    paths: dict[str, Path],
    wafer_id: str,
    review_ui: bool,
    quick_review: bool,
    extra_args: list[str],
) -> list[str]:
    command = [
        python_executable,
        str(detector_script),
        "--input",
        str(paths["input"]),
        "--output",
        str(paths["detector_json"]),
        "--preview-dir",
        str(paths["previews"]),
        "--metadata-dir",
        str(paths["metadata"]),
        "--seam-mask-dir",
        str(paths["seam_masks"]),
        "--gds-output-json",
        str(paths["gds_json"]),
        "--review-preview-dir",
        str(paths["review_previews"]),
        "--review-state",
        str(paths["review_state"]),
        "--wafer-id",
        wafer_id,
    ]
    if review_ui:
        command.append("--review-ui")
    if quick_review:
        command.append("--quick-review")
    command.extend(extra_args)
    return command


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Detect and review each batch wafer independently."
    )
    parser.add_argument("--batch", default="batch_wafers.txt")
    parser.add_argument("--base", default="extracted_cells")
    parser.add_argument("--detector", default="defect_detector.py")
    parser.add_argument("--wafer", action="append", default=[])
    parser.add_argument("--no-review", action="store_true")
    parser.add_argument("--quick-review", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--expect-cells",
        type=int,
        default=76,
        help="Warn when a wafer does not contain this many analysis images.",
    )
    return parser.parse_known_args()


def main() -> int:
    args, extra_args = parse_args()
    records = parse_batch_file(args.batch)
    wafer_ids = select_wafer_ids(records, args.wafer)

    detector_script = Path(args.detector).resolve()
    if not detector_script.exists():
        raise FileNotFoundError(detector_script)

    failures = []
    for index, wafer_id in enumerate(wafer_ids, start=1):
        paths = detector_paths(args.base, wafer_id)
        image_count = count_analysis_images(paths["root"])
        print("\n" + "=" * 72)
        print(f" DEFECT RUN [{index}/{len(wafer_ids)}]: {wafer_id}")
        print("=" * 72)
        print(f"[{wafer_id}] Analysis images: {image_count}")
        print(f"[{wafer_id}] Run directory: {paths['root']}")

        if image_count == 0:
            print(f"[{wafer_id}] ERROR: no images found; skipping.")
            failures.append(wafer_id)
            continue
        if args.expect_cells > 0 and image_count != args.expect_cells:
            print(
                f"[{wafer_id}] WARNING: expected {args.expect_cells} cells, "
                f"found {image_count}."
            )

        command = build_detector_command(
            python_executable=sys.executable,
            detector_script=detector_script,
            paths=paths,
            wafer_id=wafer_id,
            review_ui=not bool(args.no_review),
            quick_review=bool(args.quick_review),
            extra_args=extra_args,
        )
        print(f"[{wafer_id}] Command: {_quote_command(command)}")
        if args.dry_run:
            continue

        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            failures.append(wafer_id)
            print(
                f"[{wafer_id}] ERROR: defect detector exited with "
                f"code {completed.returncode}."
            )

    print("\n" + "=" * 72)
    if failures:
        print(" BATCH DEFECT REVIEW COMPLETE WITH FAILURES")
        print(" Failed: " + ", ".join(failures))
        return 1
    print(" BATCH DEFECT REVIEW COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
