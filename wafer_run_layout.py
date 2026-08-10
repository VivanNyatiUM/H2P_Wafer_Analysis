"""Per-wafer output layout helpers for H2P batch runs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


VERSION = "per-wafer-batch-output-v20-2026-07-23"

GROUPED_WAFER_IDS_EXACT_V1 = True
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
    wafer_id = str(wafer_id).strip()
    if not wafer_id:
        raise ValueError("Wafer id cannot be empty.")
    return Path(base_output_dir) / wafer_id

def infer_wafer_id_from_cell_name(filename: str | Path) -> str | None:
    stem = Path(filename).stem
    match = _CELL_NAME.match(stem)
    if not match:
        return None
    wafer_id = match.group("wafer").strip()
    return wafer_id or None

def detector_paths(
    base_output_dir: str | Path,
    wafer_id: str,
) -> dict[str, Path]:
    wafer_id = str(wafer_id).strip()
    if not wafer_id:
        raise ValueError("Wafer id cannot be empty.")
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
    all_ids = []
    for record in records:
        wafer_id = str(record["id"]).strip()
        if not wafer_id:
            raise ValueError("Wafer id cannot be empty.")
        all_ids.append(wafer_id)

    requested_values = [str(value).strip() for value in (requested or []) if str(value).strip()]
    if not requested_values:
        return all_ids

    selected: list[str] = []
    missing: list[str] = []
    for request in requested_values:
        aliases = {request.casefold(), normalize_wafer_id(request).casefold()}
        match = next(
            (wafer_id for wafer_id in all_ids if wafer_id.casefold() in aliases),
            None,
        )
        if match is None:
            missing.append(request)
        elif match not in selected:
            selected.append(match)

    if missing:
        raise ValueError(
            "Requested wafer(s) are not present in the batch file: "
            + ", ".join(missing)
        )
    return selected

