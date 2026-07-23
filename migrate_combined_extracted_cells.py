"""Move a previously combined extracted_cells tree into per-wafer folders.

Example:
    python migrate_combined_extracted_cells.py --base extracted_cells

Before:
    extracted_cells/analysis_png/Wafer_topcontact_cell_1-1.png
    extracted_cells/analysis_png/Wafer_mesaetch_cell_1-1.png

After:
    extracted_cells/Wafer_topcontact/analysis_png/Wafer_topcontact_cell_1-1.png
    extracted_cells/Wafer_mesaetch/analysis_png/Wafer_mesaetch_cell_1-1.png

Combined detector caches/results are not reusable because they were built from
multiple process-stage wafers. They are moved to the cleanup archive.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from wafer_run_layout import infer_wafer_id_from_cell_name


KNOWN_GENERATED_SUBDIRS = {
    "analysis_png",
    "metadata",
    "previews",
    "seam_masks",
    "algo_previews",
    "boundary_debug",
}

INVALID_COMBINED_FILES = {
    "algo_defects.json",
    "normal_template_auto.npz",
}


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    index = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def _archive(path: Path, archive_dir: Path) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    destination = _unique_destination(archive_dir / path.name)
    shutil.move(str(path), str(destination))
    return destination


def _wafer_from_metadata(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    wafer_id = str(payload.get("wafer_id", "") or "").strip()
    if wafer_id:
        return wafer_id
    stem = str(payload.get("cell_stem", "") or "").strip()
    return infer_wafer_id_from_cell_name(stem)


def migrate(
    base: Path,
    *,
    archive_dir: Path,
    dry_run: bool = False,
) -> dict:
    base = base.expanduser().resolve()
    archive_dir = archive_dir.expanduser().resolve()

    if not base.exists():
        raise FileNotFoundError(base)
    if not base.is_dir():
        raise ValueError(f"Not a directory: {base}")

    moved = []
    archived = []
    skipped = []
    counts = Counter()

    for name in sorted(INVALID_COMBINED_FILES):
        path = base / name
        if not path.exists():
            continue
        if dry_run:
            archived.append({"source": str(path), "destination": "<archive>"})
        else:
            destination = _archive(path, archive_dir)
            archived.append({"source": str(path), "destination": str(destination)})

    candidates = []

    # Legacy JPG cell crops live directly under extracted_cells.
    for path in base.iterdir():
        if path.is_file():
            candidates.append((path, Path(".")))

    # Generated subfolders hold PNGs, metadata, masks, and previews.
    for subdir_name in sorted(KNOWN_GENERATED_SUBDIRS):
        subdir = base / subdir_name
        if not subdir.is_dir():
            continue
        for path in subdir.rglob("*"):
            if path.is_file():
                candidates.append((path, path.parent.relative_to(base)))

    for source, relative_parent in candidates:
        # Every existing algo_previews image came from the combined multi-wafer
        # detector/template run. Archive it even when its filename identifies
        # a wafer; the per-wafer detector will regenerate a valid preview.
        if relative_parent.parts and relative_parent.parts[0] == "algo_previews":
            if dry_run:
                archived.append({"source": str(source), "destination": "<archive>"})
            else:
                destination = _archive(source, archive_dir / "algo_previews")
                archived.append(
                    {"source": str(source), "destination": str(destination)}
                )
            continue

        wafer_id = infer_wafer_id_from_cell_name(source.name)
        if wafer_id is None and source.suffix.lower() == ".json":
            wafer_id = _wafer_from_metadata(source)

        if wafer_id is None:
            skipped.append(str(source))
            continue

        destination = base / wafer_id / relative_parent / source.name
        counts[wafer_id] += 1
        if dry_run:
            moved.append({"source": str(source), "destination": str(destination)})
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing migrated file: {destination}"
            )
        shutil.move(str(source), str(destination))
        moved.append({"source": str(source), "destination": str(destination)})

    if not dry_run:
        for subdir_name in sorted(KNOWN_GENERATED_SUBDIRS, reverse=True):
            subdir = base / subdir_name
            if not subdir.exists():
                continue
            for directory in sorted(
                [path for path in subdir.rglob("*") if path.is_dir()],
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                try:
                    directory.rmdir()
                except OSError:
                    pass
            try:
                subdir.rmdir()
            except OSError:
                pass

    report = {
        "base": str(base),
        "archive_dir": str(archive_dir),
        "dry_run": bool(dry_run),
        "moved_file_count": len(moved),
        "archived_file_count": len(archived),
        "skipped_file_count": len(skipped),
        "files_by_wafer": dict(sorted(counts.items())),
        "moved": moved,
        "archived": archived,
        "skipped": skipped,
    }

    report_path = base / "per_wafer_migration_report.json"
    if not dry_run:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report_path"] = str(report_path)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="extracted_cells")
    parser.add_argument("--archive-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.base)
    archive = (
        Path(args.archive_dir)
        if args.archive_dir
        else base.resolve().parent.parent
        / f"{base.resolve().parent.name}_cleanup_archive"
        / "pre_per_wafer_layout"
    )
    report = migrate(base, archive_dir=archive, dry_run=bool(args.dry_run))
    print(
        "[Per-Wafer Migration] "
        f"moved={report['moved_file_count']}, "
        f"archived={report['archived_file_count']}, "
        f"skipped={report['skipped_file_count']}"
    )
    for wafer_id, count in report["files_by_wafer"].items():
        print(f"  {wafer_id}: {count} files")
    if report.get("report_path"):
        print(f"[Per-Wafer Migration] Report: {report['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
