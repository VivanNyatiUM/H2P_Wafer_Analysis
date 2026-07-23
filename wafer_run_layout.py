"""Per-wafer output layout helpers for H2P batch runs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


VERSION = "per-wafer-batch-output-v20-2026-07-23"

_CELL_NAME = re.compile(
    r"^(?P<wafer>.+)_cell_(?P<row>\d+)-(?P<col>\d+)"
    r"(?:[_.-].*)?$",
    re.IGNORECASE,
)


def normalize_wafer_id(value: str) -> str:
    name = str(value).strip()
    if not name:
        raise ValueError("Wafer id cannot be empty.")
    return name if name.lower().startswith("wafer_") else f"Wafer_{name}"


def wafer_run_root(base_output_dir: str | Path, wafer_id: str) -> Path:
    return Path(base_output_dir) / normalize_wafer_id(wafer_id)


def infer_wafer_id_from_cell_name(filename: str | Path) -> str | None:
    stem = Path(filename).stem
    match = _CELL_NAME.match(stem)
    if not match:
        return None
    return normalize_wafer_id(match.group("wafer"))


def detector_paths(
    base_output_dir: str | Path,
    wafer_id: str,
) -> dict[str, Path]:
    wafer_id = normalize_wafer_id(wafer_id)
    root = wafer_run_root(base_output_dir, wafer_id)
    return {
        "root": root,
        "input": root / "analysis_png",
        "metadata": root / "metadata",
        "seam_masks": root / "seam_masks",
        "previews": root / "algo_previews",
        "review_previews": root / "previews",
        "detector_json": root / "algo_defects.json",
        "gds_json": root / f"{wafer_id}_device_defects.json",
        "review_state": root / f"{wafer_id}_device_defects.json.review_state.json",
    }


def existing_wafer_runs(base_output_dir: str | Path) -> list[Path]:
    base = Path(base_output_dir)
    if not base.exists():
        return []
    result = []
    for child in base.iterdir():
        if not child.is_dir():
            continue
        if (child / "analysis_png").is_dir() and (child / "metadata").is_dir():
            result.append(child)
    return sorted(result, key=lambda path: path.name.casefold())


def count_analysis_images(run_root: str | Path) -> int:
    analysis_dir = Path(run_root) / "analysis_png"
    if not analysis_dir.is_dir():
        return 0
    return sum(
        1
        for path in analysis_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )


def select_wafer_ids(
    records: Iterable[dict],
    requested: Iterable[str] | None = None,
) -> list[str]:
    all_ids = [normalize_wafer_id(str(record["id"])) for record in records]
    if not requested:
        return all_ids

    requested_ids = {normalize_wafer_id(value).casefold() for value in requested}
    selected = [wafer_id for wafer_id in all_ids if wafer_id.casefold() in requested_ids]
    missing = sorted(requested_ids - {value.casefold() for value in selected})
    if missing:
        raise ValueError(
            "Requested wafer(s) are not present in the batch file: "
            + ", ".join(missing)
        )
    return selected
