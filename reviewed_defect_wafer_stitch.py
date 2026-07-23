"""
Build a GDS-positioned wafer overview from extracted cell images and the final
reviewed defect JSON.

The reviewed JSON is the same file edited by defect_mapper_gui.AutoLabelReviewTool,
so it already contains:
  * automatic detections that the reviewer kept;
  * manual detections added in the review UI;
  * automatic detections removed by the reviewer no longer appear.

Default outputs are written beside the reviewed JSON in:
    <wafer_id>_reviewed_wafer/

Files:
    <wafer_id>_reviewed_wafer_clean.png
    <wafer_id>_reviewed_wafer_defects.png
    <wafer_id>_reviewed_wafer_outline.png
    <wafer_id>_reviewed_wafer_defect_mask.png
    <wafer_id>_reviewed_wafer_auto_mask.png
    <wafer_id>_reviewed_wafer_manual_mask.png
    <wafer_id>_reviewed_wafer_report.json

Automatic polygons are red. Manual-review regions are cyan.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
VERSION = "reviewed-wafer-stitch-v1-2026-07-23"

AUTO_COLOR = (0, 0, 255)
MANUAL_COLOR = (255, 255, 0)
UNKNOWN_COLOR = (0, 215, 255)
CELL_BORDER_COLOR = (80, 80, 80)
EXCLUDED_COLOR = (0, 0, 180)


@dataclass
class CellRecord:
    annotation_name: str
    stem: str
    image_path: Path
    metadata_path: Path
    metadata: dict[str, Any]
    image_width: int
    image_height: int
    image_corners_gds: np.ndarray
    row: int
    col: int


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def _read_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix or ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def _fit_canvas_to_gds_affine(
    canvas_points: np.ndarray,
    gds_points: np.ndarray,
) -> np.ndarray:
    canvas_points = np.asarray(canvas_points, dtype=np.float64)
    gds_points = np.asarray(gds_points, dtype=np.float64)
    if canvas_points.shape[0] < 3 or gds_points.shape[0] < 3:
        raise ValueError("Need at least three canvas/GDS corner pairs.")

    matrix_rows: list[list[float]] = []
    values: list[float] = []
    for (x, y), (gx, gy) in zip(canvas_points, gds_points):
        matrix_rows.append([x, y, 1.0, 0.0, 0.0, 0.0])
        values.append(float(gx))
        matrix_rows.append([0.0, 0.0, 0.0, x, y, 1.0])
        values.append(float(gy))

    coefficients, *_ = np.linalg.lstsq(
        np.asarray(matrix_rows, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
        rcond=None,
    )
    return coefficients.reshape(2, 3)


def _apply_affine_2x3(
    matrix: np.ndarray,
    x: float,
    y: float,
) -> tuple[float, float]:
    result = np.asarray(matrix, dtype=np.float64) @ np.asarray(
        [float(x), float(y), 1.0],
        dtype=np.float64,
    )
    return float(result[0]), float(result[1])


def _metadata_canvas_to_gds_affine(metadata: dict[str, Any]) -> np.ndarray:
    gds = np.asarray(
        metadata.get("gds_corners_um")
        or metadata.get("gds_corners")
        or [],
        dtype=np.float64,
    )
    if gds.shape[0] < 3:
        raise ValueError(
            f"Metadata for {metadata.get('cell_stem', '')} lacks gds_corners_um."
        )

    if "canvas_corners_px_downscaled" in metadata:
        canvas = np.asarray(
            metadata["canvas_corners_px_downscaled"],
            dtype=np.float64,
        )
    elif "canvas_corners_px" in metadata:
        canvas = np.asarray(metadata["canvas_corners_px"], dtype=np.float64)
    elif (
        "canvas_corners_px_fullres" in metadata
        and "canvas_to_downscaled_scale_used" in metadata
    ):
        canvas = (
            np.asarray(
                metadata["canvas_corners_px_fullres"],
                dtype=np.float64,
            )
            * float(metadata["canvas_to_downscaled_scale_used"])
        )
    elif "canvas_corners_px_fullres" in metadata:
        canvas = np.asarray(
            metadata["canvas_corners_px_fullres"],
            dtype=np.float64,
        )
    else:
        raise ValueError(
            f"Metadata for {metadata.get('cell_stem', '')} lacks canvas corners."
        )

    return _fit_canvas_to_gds_affine(canvas, gds)


class _DirectGDSPoint(Exception):
    def __init__(self, gx: float, gy: float):
        super().__init__()
        self.gx = float(gx)
        self.gy = float(gy)


def _crop_px_to_canvas_px(
    px: float,
    py: float,
    metadata: dict[str, Any],
) -> tuple[float, float]:
    if "crop_bounds_local_px_downscaled_before_resize" in metadata:
        crop_x1, crop_y1, _crop_x2, _crop_y2 = [
            float(value)
            for value in metadata[
                "crop_bounds_local_px_downscaled_before_resize"
            ]
        ]
        resize_scale = float(
            metadata.get("output_resize_scale", 1.0) or 1.0
        )
        if resize_scale <= 0:
            resize_scale = 1.0

        x_rotated = crop_x1 + float(px) / resize_scale
        y_rotated = crop_y1 + float(py) / resize_scale
        rotation_matrix = np.asarray(
            metadata["rotation_matrix_2x3_downscaled"],
            dtype=np.float64,
        )
        origin = np.asarray(
            metadata["local_origin_px_downscaled"],
            dtype=np.float64,
        )
    elif "crop_bounds_local_px" in metadata:
        crop_x1, crop_y1, _crop_x2, _crop_y2 = [
            float(value) for value in metadata["crop_bounds_local_px"]
        ]
        x_rotated = crop_x1 + float(px)
        y_rotated = crop_y1 + float(py)
        rotation_matrix = np.asarray(
            metadata["rotation_matrix_2x3"],
            dtype=np.float64,
        )
        origin = np.asarray(
            metadata["local_origin_px"],
            dtype=np.float64,
        )
    else:
        crop_size = metadata.get("crop_size_px") or [1, 1]
        crop_width = max(float(crop_size[0]) - 1.0, 1.0)
        crop_height = max(float(crop_size[1]) - 1.0, 1.0)
        bbox = metadata.get("gds_bbox_um") or metadata.get("gds_bbox")
        if bbox and len(bbox) >= 4:
            min_x, min_y, max_x, max_y = [float(value) for value in bbox[:4]]
            gx = min_x + float(px) / crop_width * (max_x - min_x)
            gy = max_y - float(py) / crop_height * (max_y - min_y)
            raise _DirectGDSPoint(gx, gy)
        raise ValueError(
            f"Metadata for {metadata.get('cell_stem', '')} lacks crop transform."
        )

    inverse = cv2.invertAffineTransform(rotation_matrix)
    local = inverse @ np.asarray(
        [x_rotated, y_rotated, 1.0],
        dtype=np.float64,
    )
    canvas = local + origin
    return float(canvas[0]), float(canvas[1])


def crop_px_to_gds(
    px: float,
    py: float,
    metadata: dict[str, Any],
    canvas_to_gds: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    if canvas_to_gds is None:
        try:
            canvas_to_gds = _metadata_canvas_to_gds_affine(metadata)
        except Exception:
            canvas_to_gds = None

    try:
        canvas_x, canvas_y = _crop_px_to_canvas_px(px, py, metadata)
    except _DirectGDSPoint as direct:
        return direct.gx, direct.gy

    if canvas_to_gds is None:
        raise ValueError(
            f"Cannot map crop pixels for {metadata.get('cell_stem', '')}."
        )
    return _apply_affine_2x3(canvas_to_gds, canvas_x, canvas_y)


def _normalize_annotations(raw: Any) -> dict[str, list[dict[str, Any]]]:
    if isinstance(raw, dict):
        if isinstance(raw.get("annotations"), dict):
            raw = raw["annotations"]

        if all(isinstance(value, list) for value in raw.values()):
            return {
                str(key): [
                    dict(record)
                    for record in value
                    if isinstance(record, dict)
                ]
                for key, value in raw.items()
            }

        images = raw.get("images")
        if isinstance(images, list):
            result: dict[str, list[dict[str, Any]]] = {}
            for entry in images:
                if not isinstance(entry, dict):
                    continue
                name = str(
                    entry.get("image")
                    or Path(str(entry.get("image_path", ""))).name
                )
                if not name:
                    continue
                result[name] = [
                    dict(record)
                    for record in entry.get("defects", [])
                    if isinstance(record, dict)
                ]
            return result

    raise ValueError(
        "Unsupported annotations JSON. Expected the reviewed filename->list "
        "schema or detector output with an images list."
    )


def _infer_wafer_id(
    annotations_path: Path,
    metadata: Iterable[dict[str, Any]],
    explicit: str,
) -> str:
    if explicit:
        return explicit

    for record in metadata:
        wafer_id = str(record.get("wafer_id", "") or "").strip()
        if wafer_id:
            return wafer_id

    stem = annotations_path.stem
    for suffix in ("_device_defects", "_reviewed_defects", "_defects"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or "wafer"


def _image_dimensions(path: Path) -> tuple[int, int]:
    image = _read_bgr(path)
    return int(image.shape[1]), int(image.shape[0])


def _index_images(image_dir: Path) -> dict[str, Path]:
    if image_dir.is_file():
        candidates = [image_dir]
    else:
        candidates = [
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

    result: dict[str, Path] = {}
    for path in sorted(candidates):
        lower_name = path.name.lower()
        if any(token in lower_name for token in ("preview", "overlay", "mask")):
            continue
        result[path.name.lower()] = path
        result[path.stem.lower()] = path
    return result


def _resolve_image(
    metadata: dict[str, Any],
    metadata_path: Path,
    image_index: dict[str, Path],
) -> Optional[Path]:
    candidates: list[str] = []
    for key in ("analysis_png", "legacy_jpg", "image_path", "filepath"):
        value = str(metadata.get(key, "") or "").strip()
        if value:
            candidates.append(value)

    stem = str(metadata.get("cell_stem", "") or "").strip()
    if stem:
        candidates.extend([f"{stem}.png", f"{stem}.jpg", stem])

    for raw in candidates:
        path = Path(raw)
        for option in (
            path,
            metadata_path.parent / path,
            metadata_path.parent.parent / path,
            Path.cwd() / path,
        ):
            if option.exists() and option.is_file():
                return option

        by_name = image_index.get(path.name.lower())
        if by_name is not None:
            return by_name
        by_stem = image_index.get(path.stem.lower())
        if by_stem is not None:
            return by_stem

    return None


def _annotation_name_for_metadata(metadata: dict[str, Any]) -> str:
    legacy = str(metadata.get("legacy_jpg", "") or "").strip()
    if legacy:
        return Path(legacy).name

    stem = str(metadata.get("cell_stem", "") or "").strip()
    return f"{stem}.jpg" if stem else ""


def _discover_cells(
    image_dir: Path,
    metadata_dir: Path,
) -> tuple[list[CellRecord], list[str]]:
    image_index = _index_images(image_dir)
    cells: list[CellRecord] = []
    warnings: list[str] = []

    for metadata_path in sorted(metadata_dir.glob("*.json")):
        if metadata_path.name.endswith("_cell_index.json"):
            continue
        try:
            metadata = _read_json(metadata_path)
        except Exception as exc:
            warnings.append(f"{metadata_path.name}: unreadable metadata: {exc}")
            continue
        if not isinstance(metadata, dict):
            continue

        image_path = _resolve_image(metadata, metadata_path, image_index)
        if image_path is None:
            warnings.append(
                f"{metadata_path.name}: matching cell image not found."
            )
            continue

        try:
            width, height = _image_dimensions(image_path)
            try:
                canvas_to_gds = _metadata_canvas_to_gds_affine(metadata)
            except Exception:
                canvas_to_gds = None

            corners_gds = np.asarray(
                [
                    crop_px_to_gds(0.0, 0.0, metadata, canvas_to_gds),
                    crop_px_to_gds(max(width - 1, 0), 0.0, metadata, canvas_to_gds),
                    crop_px_to_gds(max(width - 1, 0), max(height - 1, 0), metadata, canvas_to_gds),
                    crop_px_to_gds(0.0, max(height - 1, 0), metadata, canvas_to_gds),
                ],
                dtype=np.float64,
            )
        except Exception as exc:
            warnings.append(
                f"{metadata_path.name}: cannot map crop into GDS: {exc}"
            )
            continue

        stem = str(metadata.get("cell_stem") or image_path.stem)
        row = int(metadata.get("cell_row", 0) or 0)
        col = int(metadata.get("cell_col", 0) or 0)
        if row <= 0 or col <= 0:
            match = re.search(r"_cell_(\d+)-(\d+)$", stem)
            if match:
                row = int(match.group(1))
                col = int(match.group(2))

        annotation_name = _annotation_name_for_metadata(metadata)
        if not annotation_name:
            annotation_name = f"{stem}.jpg"

        cells.append(
            CellRecord(
                annotation_name=annotation_name,
                stem=stem,
                image_path=image_path,
                metadata_path=metadata_path,
                metadata=metadata,
                image_width=width,
                image_height=height,
                image_corners_gds=corners_gds,
                row=row,
                col=col,
            )
        )

    cells.sort(key=lambda cell: (cell.row, cell.col, cell.stem))
    if not cells:
        raise RuntimeError(
            f"No usable cell image/metadata pairs found under "
            f"{image_dir} and {metadata_dir}."
        )
    return cells, warnings


def _gds_to_canvas_points(
    points_gds: np.ndarray,
    *,
    min_x: float,
    max_y: float,
    scale: float,
    padding: int,
) -> np.ndarray:
    points = np.asarray(points_gds, dtype=np.float64)
    output = np.empty_like(points)
    output[:, 0] = padding + (points[:, 0] - min_x) * scale
    output[:, 1] = padding + (max_y - points[:, 1]) * scale
    return output


def _warp_cell_into_canvas(
    canvas: np.ndarray,
    cell: CellRecord,
    destination_points: np.ndarray,
    *,
    cell_border: int,
) -> None:
    image = _read_bgr(cell.image_path)
    source_points = np.asarray(
        [
            [0.0, 0.0],
            [max(image.shape[1] - 1, 0), 0.0],
            [max(image.shape[1] - 1, 0), max(image.shape[0] - 1, 0)],
            [0.0, max(image.shape[0] - 1, 0)],
        ],
        dtype=np.float32,
    )

    destination = np.asarray(destination_points, dtype=np.float32)
    x0 = max(0, int(math.floor(float(destination[:, 0].min()))) - 2)
    y0 = max(0, int(math.floor(float(destination[:, 1].min()))) - 2)
    x1 = min(canvas.shape[1], int(math.ceil(float(destination[:, 0].max()))) + 3)
    y1 = min(canvas.shape[0], int(math.ceil(float(destination[:, 1].max()))) + 3)
    if x1 <= x0 or y1 <= y0:
        return

    local_destination = destination - np.asarray([float(x0), float(y0)], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source_points, local_destination)
    roi_width = x1 - x0
    roi_height = y1 - y0

    warped = cv2.warpPerspective(
        image,
        transform,
        (roi_width, roi_height),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    source_alpha = np.full(image.shape[:2], 255, dtype=np.uint8)
    alpha = cv2.warpPerspective(
        source_alpha,
        transform,
        (roi_width, roi_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    roi = canvas[y0:y1, x0:x1]
    mask = alpha > 0
    roi[mask] = warped[mask]

    if cell_border > 0:
        outline = np.round(destination).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            canvas,
            [outline],
            True,
            CELL_BORDER_COLOR,
            int(cell_border),
            cv2.LINE_AA,
        )


def _source_kind(record: dict[str, Any]) -> str:
    source = str(record.get("source", "") or "").lower()
    record_type = str(record.get("type", "") or "").lower()

    if "manual" in source or "review" in source:
        return "manual"
    if "algorithm" in source or "auto" in source:
        return "automatic"
    if record_type == "auto_defect":
        return "automatic"
    return "manual"


def _record_points_gds(
    record: dict[str, Any],
    cell: CellRecord,
) -> Optional[np.ndarray]:
    for key in ("polygon_gds", "corners_gds"):
        points = record.get(key)
        if isinstance(points, list) and len(points) >= 3:
            try:
                array = np.asarray(points, dtype=np.float64)
                if array.ndim == 2 and array.shape[1] >= 2:
                    return array[:, :2]
            except Exception:
                pass

    pixel_polygon = record.get("polygon_px")
    if isinstance(pixel_polygon, list) and len(pixel_polygon) >= 3:
        try:
            try:
                canvas_to_gds = _metadata_canvas_to_gds_affine(cell.metadata)
            except Exception:
                canvas_to_gds = None
            return np.asarray(
                [
                    crop_px_to_gds(
                        float(point[0]),
                        float(point[1]),
                        cell.metadata,
                        canvas_to_gds,
                    )
                    for point in pixel_polygon
                ],
                dtype=np.float64,
            )
        except Exception:
            pass

    box = record.get("box_px") or record.get("bbox_px")
    if isinstance(box, list) and len(box) >= 4:
        try:
            x, y, width, height = [float(value) for value in box[:4]]
            corners_px = [
                (x, y),
                (x + width, y),
                (x + width, y + height),
                (x, y + height),
            ]
            try:
                canvas_to_gds = _metadata_canvas_to_gds_affine(cell.metadata)
            except Exception:
                canvas_to_gds = None
            return np.asarray(
                [
                    crop_px_to_gds(px, py, cell.metadata, canvas_to_gds)
                    for px, py in corners_px
                ],
                dtype=np.float64,
            )
        except Exception:
            pass

    return None


def _annotation_lookup(
    annotations: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for name, records in annotations.items():
        path = Path(name)
        result[name.lower()] = records
        result[path.name.lower()] = records
        result[path.stem.lower()] = records
    return result


def _annotations_for_cell(
    cell: CellRecord,
    lookup: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    for key in (
        cell.annotation_name.lower(),
        Path(cell.annotation_name).stem.lower(),
        cell.image_path.name.lower(),
        cell.image_path.stem.lower(),
        cell.stem.lower(),
    ):
        if key in lookup:
            return lookup[key]
    return []


def _draw_legend(
    image: np.ndarray,
    *,
    wafer_id: str,
    cell_count: int,
    automatic_count: int,
    manual_count: int,
    excluded_count: int,
) -> None:
    panel_width = min(image.shape[1] - 20, 720)
    panel_height = 135 if excluded_count else 112
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (12, 12),
        (12 + panel_width, 12 + panel_height),
        (18, 18, 18),
        -1,
    )
    cv2.addWeighted(overlay, 0.78, image, 0.22, 0.0, image)

    cv2.putText(
        image,
        f"{wafer_id} - reviewed defect wafer",
        (30, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.86,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(image, (30, 67), (52, 89), AUTO_COLOR, -1)
    cv2.putText(
        image,
        f"automatic kept: {automatic_count}",
        (64, 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(image, (300, 67), (322, 89), MANUAL_COLOR, -1)
    cv2.putText(
        image,
        f"manual added: {manual_count}",
        (334, 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"cells: {cell_count}",
        (560, 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    if excluded_count:
        cv2.putText(
            image,
            f"excluded cells marked with X: {excluded_count}",
            (30, 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (150, 150, 255),
            1,
            cv2.LINE_AA,
        )


def _load_exclusions(
    annotations_json: Path,
    explicit_path: Optional[Path],
) -> set[str]:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    candidates.extend(
        [
            annotations_json.parent / "manual_exclusions.json",
            Path.cwd() / "manual_exclusions.json",
        ]
    )
    for path in candidates:
        if not path.exists():
            continue
        try:
            raw = _read_json(path)
            if isinstance(raw, list):
                return {str(value) for value in raw}
        except Exception:
            continue
    return set()


def build_reviewed_wafer_stitch(
    *,
    image_dir: Path | str,
    annotations_json: Path | str,
    metadata_dir: Path | str,
    output_dir: Path | str | None = None,
    wafer_id: str = "",
    max_size: int = 9000,
    padding: int = 170,
    fill_alpha: float = 0.30,
    outline_thickness: int = 3,
    cell_border: int = 1,
    exclusions_json: Path | str | None = None,
) -> dict[str, Any]:
    image_dir = Path(image_dir)
    annotations_json = Path(annotations_json)
    metadata_dir = Path(metadata_dir)

    if not annotations_json.exists():
        raise FileNotFoundError(
            f"Reviewed annotations JSON not found: {annotations_json}"
        )
    if not metadata_dir.exists():
        raise FileNotFoundError(
            f"Metadata directory not found: {metadata_dir}"
        )

    raw_annotations = _read_json(annotations_json)
    annotations = _normalize_annotations(raw_annotations)
    cells, warnings = _discover_cells(image_dir, metadata_dir)
    annotation_lookup = _annotation_lookup(annotations)

    wafer_id = _infer_wafer_id(
        annotations_json,
        [cell.metadata for cell in cells],
        wafer_id,
    )
    if output_dir is None:
        output_dir = annotations_json.parent / f"{wafer_id}_reviewed_wafer"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_corners = np.vstack([cell.image_corners_gds for cell in cells])
    min_x = float(np.min(all_corners[:, 0]))
    max_x = float(np.max(all_corners[:, 0]))
    min_y = float(np.min(all_corners[:, 1]))
    max_y = float(np.max(all_corners[:, 1]))

    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    max_size = max(1200, int(max_size))
    padding = max(20, int(padding))
    available = max(max_size - 2 * padding, 100)
    scale = min(available / span_x, available / span_y)

    canvas_width = max(1, int(math.ceil(span_x * scale)) + 2 * padding)
    canvas_height = max(1, int(math.ceil(span_y * scale)) + 2 * padding)

    clean = np.full((canvas_height, canvas_width, 3), 22, dtype=np.uint8)

    for cell in cells:
        destination = _gds_to_canvas_points(
            cell.image_corners_gds,
            min_x=min_x,
            max_y=max_y,
            scale=scale,
            padding=padding,
        )
        _warp_cell_into_canvas(
            clean,
            cell,
            destination,
            cell_border=cell_border,
        )

    composite = clean.copy()
    outline = clean.copy()
    auto_mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    manual_mask = np.zeros_like(auto_mask)
    combined_mask = np.zeros_like(auto_mask)
    transparent_layer = np.zeros_like(clean)

    automatic_count = 0
    manual_count = 0
    skipped_annotations = 0
    records_rendered: list[dict[str, Any]] = []

    for cell in cells:
        records = _annotations_for_cell(cell, annotation_lookup)
        for record in records:
            points_gds = _record_points_gds(record, cell)
            if points_gds is None or len(points_gds) < 3:
                skipped_annotations += 1
                continue

            points_canvas = _gds_to_canvas_points(
                points_gds,
                min_x=min_x,
                max_y=max_y,
                scale=scale,
                padding=padding,
            )
            points_int = np.round(points_canvas).astype(np.int32).reshape(-1, 1, 2)

            source_kind = _source_kind(record)
            if source_kind == "automatic":
                color = AUTO_COLOR
                target_mask = auto_mask
                automatic_count += 1
            else:
                color = MANUAL_COLOR
                target_mask = manual_mask
                manual_count += 1

            cv2.fillPoly(transparent_layer, [points_int], color, cv2.LINE_AA)
            cv2.fillPoly(target_mask, [points_int], 255, cv2.LINE_AA)
            cv2.fillPoly(combined_mask, [points_int], 255, cv2.LINE_AA)
            cv2.polylines(
                composite,
                [points_int],
                True,
                color,
                int(max(1, outline_thickness)),
                cv2.LINE_AA,
            )
            cv2.polylines(
                outline,
                [points_int],
                True,
                color,
                int(max(1, outline_thickness)),
                cv2.LINE_AA,
            )

            records_rendered.append(
                {
                    "cell": cell.annotation_name,
                    "type": str(record.get("type", "defect")),
                    "source": source_kind,
                    "vertex_count": int(len(points_gds)),
                }
            )

    alpha = float(np.clip(fill_alpha, 0.0, 1.0))
    active = combined_mask > 0
    if np.any(active) and alpha > 0:
        blended = cv2.addWeighted(
            composite,
            1.0 - alpha,
            transparent_layer,
            alpha,
            0.0,
        )
        composite[active] = blended[active]

    exclusions = _load_exclusions(
        annotations_json,
        Path(exclusions_json) if exclusions_json else None,
    )
    excluded_count = 0
    for cell in cells:
        if (
            cell.annotation_name not in exclusions
            and cell.image_path.name not in exclusions
            and cell.stem not in exclusions
        ):
            continue
        destination = _gds_to_canvas_points(
            cell.image_corners_gds,
            min_x=min_x,
            max_y=max_y,
            scale=scale,
            padding=padding,
        )
        polygon = np.round(destination).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            composite,
            [polygon],
            True,
            EXCLUDED_COLOR,
            4,
            cv2.LINE_AA,
        )
        points = polygon.reshape(-1, 2)
        cv2.line(composite, tuple(points[0]), tuple(points[2]), EXCLUDED_COLOR, 4, cv2.LINE_AA)
        cv2.line(composite, tuple(points[1]), tuple(points[3]), EXCLUDED_COLOR, 4, cv2.LINE_AA)
        excluded_count += 1

    _draw_legend(
        composite,
        wafer_id=wafer_id,
        cell_count=len(cells),
        automatic_count=automatic_count,
        manual_count=manual_count,
        excluded_count=excluded_count,
    )
    _draw_legend(
        outline,
        wafer_id=wafer_id,
        cell_count=len(cells),
        automatic_count=automatic_count,
        manual_count=manual_count,
        excluded_count=excluded_count,
    )

    clean_path = output_dir / f"{wafer_id}_reviewed_wafer_clean.png"
    composite_path = output_dir / f"{wafer_id}_reviewed_wafer_defects.png"
    outline_path = output_dir / f"{wafer_id}_reviewed_wafer_outline.png"
    mask_path = output_dir / f"{wafer_id}_reviewed_wafer_defect_mask.png"
    auto_mask_path = output_dir / f"{wafer_id}_reviewed_wafer_auto_mask.png"
    manual_mask_path = output_dir / f"{wafer_id}_reviewed_wafer_manual_mask.png"

    _write_image(clean_path, clean)
    _write_image(composite_path, composite)
    _write_image(outline_path, outline)
    _write_image(mask_path, combined_mask)
    _write_image(auto_mask_path, auto_mask)
    _write_image(manual_mask_path, manual_mask)

    report = {
        "version": VERSION,
        "wafer_id": wafer_id,
        "annotations_json": str(annotations_json),
        "image_dir": str(image_dir),
        "metadata_dir": str(metadata_dir),
        "output_dir": str(output_dir),
        "canvas_size_px": [canvas_width, canvas_height],
        "gds_bounds_um": [min_x, min_y, max_x, max_y],
        "pixels_per_um": scale,
        "cell_count": len(cells),
        "automatic_kept_count": automatic_count,
        "manual_added_count": manual_count,
        "rendered_annotation_count": automatic_count + manual_count,
        "skipped_annotation_count": skipped_annotations,
        "excluded_cell_count": excluded_count,
        "warnings": warnings,
        "outputs": {
            "clean": str(clean_path),
            "composite": str(composite_path),
            "outline": str(outline_path),
            "combined_mask": str(mask_path),
            "automatic_mask": str(auto_mask_path),
            "manual_mask": str(manual_mask_path),
        },
        "rendered_records": records_rendered,
    }
    report_path = output_dir / f"{wafer_id}_reviewed_wafer_report.json"
    _write_json(report_path, report)
    report["outputs"]["report"] = str(report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a GDS-positioned stitched wafer overview using the final "
            "reviewed automatic and manual defect polygons."
        )
    )
    parser.add_argument("--images", default="extracted_cells/analysis_png")
    parser.add_argument("--annotations", default="Wafer_A_device_defects.json")
    parser.add_argument("--metadata", default="extracted_cells/metadata")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--wafer-id", default="")
    parser.add_argument("--max-size", type=int, default=9000)
    parser.add_argument("--padding", type=int, default=170)
    parser.add_argument("--fill-alpha", type=float, default=0.30)
    parser.add_argument("--outline-thickness", type=int, default=3)
    parser.add_argument("--cell-border", type=int, default=1)
    parser.add_argument("--exclusions-json", default="")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    report = build_reviewed_wafer_stitch(
        image_dir=Path(arguments.images),
        annotations_json=Path(arguments.annotations),
        metadata_dir=Path(arguments.metadata),
        output_dir=Path(arguments.output_dir) if arguments.output_dir else None,
        wafer_id=str(arguments.wafer_id or ""),
        max_size=int(arguments.max_size),
        padding=int(arguments.padding),
        fill_alpha=float(arguments.fill_alpha),
        outline_thickness=int(arguments.outline_thickness),
        cell_border=int(arguments.cell_border),
        exclusions_json=Path(arguments.exclusions_json) if arguments.exclusions_json else None,
    )
    print(
        "[Reviewed Wafer Stitch] "
        f"cells={report['cell_count']}, "
        f"automatic={report['automatic_kept_count']}, "
        f"manual={report['manual_added_count']}"
    )
    print(
        "[Reviewed Wafer Stitch] Composite: "
        f"{report['outputs']['composite']}"
    )
    print(
        "[Reviewed Wafer Stitch] Report: "
        f"{report['outputs']['report']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
