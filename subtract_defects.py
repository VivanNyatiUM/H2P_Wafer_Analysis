import os
import re
import json
import math
import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import gdstk
from h2p_progress import ProgressBar
from batch_wafers_parser import parse_batch_file
from wafer_run_layout import normalize_wafer_id, select_wafer_ids

# >>> H2P SUBTRACTION PROGRESS V2 >>>

DEFAULT_ALIGNMENT_ERROR_ANGLE_DEG = 0.001
DEFAULT_ALIGNMENT_ERROR_X_PX = 2.0
DEFAULT_ALIGNMENT_ERROR_Y_PX = 2.0
DEFAULT_EXTRA_MARGIN_UM = 0.0
DEFAULT_WAFER_CENTER_X_UM = 0.0
DEFAULT_WAFER_CENTER_Y_UM = 0.0
SUBTRACT_DEFECTS_VERSION = "tight-mask-removal-report-v2-2026-07-13"


def _polygon_signed_area(points: np.ndarray) -> float:
    """Return signed polygon area. Positive means CCW in GDS x/y."""
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _edge_length(points: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(points[j] - points[i]))


def _safe_float(value, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def estimate_um_per_pixel(defect: dict, scale_corners_gds: np.ndarray | None) -> tuple[float, float]:
    """Estimate crop micron-per-pixel scale from the bbox corners."""
    px_w = px_h = None
    box_px = defect.get("box_px")
    if isinstance(box_px, (list, tuple)) and len(box_px) >= 4:
        px_w = abs(_safe_float(box_px[2], 0.0) or 0.0)
        px_h = abs(_safe_float(box_px[3], 0.0) or 0.0)

    um_per_px_x = None
    um_per_px_y = None
    if scale_corners_gds is not None and len(scale_corners_gds) >= 4:
        top = _edge_length(scale_corners_gds, 0, 1)
        right = _edge_length(scale_corners_gds, 1, 2)
        bottom = _edge_length(scale_corners_gds, 2, 3)
        left = _edge_length(scale_corners_gds, 3, 0)
        if px_w and px_w > 1e-9:
            um_per_px_x = 0.5 * (top + bottom) / px_w
        if px_h and px_h > 1e-9:
            um_per_px_y = 0.5 * (right + left) / px_h

    if um_per_px_x is None and px_w and px_w > 1e-9:
        width_um = abs(_safe_float(defect.get("width_um"), 0.0) or 0.0)
        if width_um > 1e-9:
            um_per_px_x = width_um / px_w
    if um_per_px_y is None and px_h and px_h > 1e-9:
        height_um = abs(_safe_float(defect.get("height_um"), 0.0) or 0.0)
        if height_um > 1e-9:
            um_per_px_y = height_um / px_h

    if um_per_px_x is None or not np.isfinite(um_per_px_x) or um_per_px_x <= 0:
        um_per_px_x = 1.0
    if um_per_px_y is None or not np.isfinite(um_per_px_y) or um_per_px_y <= 0:
        um_per_px_y = um_per_px_x
    return float(um_per_px_x), float(um_per_px_y)


def compute_alignment_error_margin_um(
    defect_points_gds: np.ndarray,
    defect: dict,
    scale_corners_gds: np.ndarray | None = None,
    error_angle_deg: float = DEFAULT_ALIGNMENT_ERROR_ANGLE_DEG,
    error_x_px: float = DEFAULT_ALIGNMENT_ERROR_X_PX,
    error_y_px: float = DEFAULT_ALIGNMENT_ERROR_Y_PX,
    extra_margin_um: float = DEFAULT_EXTRA_MARGIN_UM,
    wafer_center: tuple[float, float] = (DEFAULT_WAFER_CENTER_X_UM, DEFAULT_WAFER_CENTER_Y_UM),
) -> float:
    """Convert residual alignment uncertainty into one conservative GDS margin."""
    if scale_corners_gds is None:
        scale_corners_gds = defect_points_gds
    um_per_px_x, um_per_px_y = estimate_um_per_pixel(defect, scale_corners_gds)

    translation_margin_um = math.hypot(
        abs(float(error_x_px)) * um_per_px_x,
        abs(float(error_y_px)) * um_per_px_y,
    )

    theta = math.radians(abs(float(error_angle_deg)))
    cx, cy = wafer_center
    radii = np.linalg.norm(
        defect_points_gds - np.array([[float(cx), float(cy)]], dtype=np.float64),
        axis=1,
    )
    max_radius_um = float(np.max(radii)) if len(radii) else 0.0
    rotation_margin_um = max_radius_um * math.sin(theta)
    margin = translation_margin_um + rotation_margin_um + max(0.0, float(extra_margin_um))
    return float(max(0.0, margin))


def _scale_polygon_about_centroid(points: np.ndarray, margin_um: float) -> np.ndarray:
    centroid = np.mean(points, axis=0)
    vectors = points - centroid
    max_radius = float(np.max(np.linalg.norm(vectors, axis=1))) if len(vectors) else 0.0
    if max_radius <= 1e-9:
        return points.copy()
    scale = (max_radius + margin_um) / max_radius
    return centroid + vectors * scale


def expand_defect_polygon_for_alignment_error(
    points: np.ndarray,
    margin_um: float,
    precision: float = 1e-3,
) -> list[gdstk.Polygon]:
    """Expand one defect polygon outward by a fixed GDS-space margin."""
    if margin_um <= 0:
        return [gdstk.Polygon(points)]
    base_poly = gdstk.Polygon(points)
    try:
        expanded = gdstk.offset(
            [base_poly],
            margin_um,
            join="miter",
            tolerance=2,
            precision=precision,
            use_union=True,
        )
        if expanded:
            return expanded
    except Exception as exc:
        print(f"[WARN] gdstk.offset failed on a defect polygon; using centroid scaling fallback: {exc}")
    return [gdstk.Polygon(_scale_polygon_about_centroid(points, margin_um))]


def _legacy_axis_aligned_points(defect: dict) -> np.ndarray:
    cx = float(defect["center_x_um"])
    cy = float(defect["center_y_um"])
    w = float(defect["width_um"])
    h = float(defect["height_um"])
    x1, y1 = cx - w / 2.0, cy - h / 2.0
    x2, y2 = cx + w / 2.0, cy + h / 2.0
    return np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], dtype=np.float64)


def _points_from_field(defect: dict, field: str) -> np.ndarray | None:
    pts = defect.get(field)
    if not pts:
        return None
    try:
        arr = np.array([(float(x), float(y)) for x, y in pts], dtype=np.float64)
    except Exception:
        return None
    if len(arr) > 3 and np.linalg.norm(arr[0] - arr[-1]) < 1e-9:
        arr = arr[:-1]
    return arr if len(arr) >= 3 else None


def _normalize_device_key(value: str | Path) -> str:
    """Normalize image/metadata names so JPG, PNG, and JSON keys match."""
    return Path(str(value)).stem.strip().lower()


def _wafer_prefix_from_device_name(value: str) -> str:
    stem = _normalize_device_key(value)
    return re.sub(r"_cell_\d+-\d+$", "", stem, flags=re.IGNORECASE)


def _load_defect_polygons_grouped(
    defects_data: dict,
    compensate_alignment_error: bool = True,
    error_angle_deg: float = DEFAULT_ALIGNMENT_ERROR_ANGLE_DEG,
    error_x_px: float = DEFAULT_ALIGNMENT_ERROR_X_PX,
    error_y_px: float = DEFAULT_ALIGNMENT_ERROR_Y_PX,
    extra_margin_um: float = DEFAULT_EXTRA_MARGIN_UM,
    wafer_center: tuple[float, float] = (DEFAULT_WAFER_CENTER_X_UM, DEFAULT_WAFER_CENTER_Y_UM),
    strict_corners: bool = False,
    precision: float = 1e-3,
) -> tuple[list[gdstk.Polygon], dict[str, list[gdstk.Polygon]], int, int, int, list[float]]:
    all_polygons: list[gdstk.Polygon] = []
    by_device: dict[str, list[gdstk.Polygon]] = {}
    legacy_count = 0
    bbox_corner_count = 0
    polygon_count = 0
    margins: list[float] = []

    for filename, defects in defects_data.items():
        if not isinstance(defects, list):
            continue
        device_key = _normalize_device_key(filename)
        device_polygons = by_device.setdefault(device_key, [])

        for defect in defects:
            if not isinstance(defect, dict):
                continue

            scale_corners = _points_from_field(defect, "corners_gds")
            polygon_points = _points_from_field(defect, "polygon_gds")
            if polygon_points is not None:
                points = polygon_points
                polygon_count += 1
            elif scale_corners is not None:
                points = scale_corners
                bbox_corner_count += 1
            else:
                if strict_corners:
                    raise RuntimeError(f"Defect in {filename} is missing corners_gds/polygon_gds")
                points = _legacy_axis_aligned_points(defect)
                scale_corners = points
                legacy_count += 1

            if len(points) < 3:
                print(f"[WARN] Skipping malformed defect in {filename}: fewer than 3 polygon points.")
                continue
            if abs(_polygon_signed_area(points)) < 1e-9:
                print(f"[WARN] Skipping degenerate zero-area defect polygon in {filename}.")
                continue

            if compensate_alignment_error:
                margin_um = compute_alignment_error_margin_um(
                    points,
                    defect,
                    scale_corners_gds=scale_corners,
                    error_angle_deg=error_angle_deg,
                    error_x_px=error_x_px,
                    error_y_px=error_y_px,
                    extra_margin_um=extra_margin_um,
                    wafer_center=wafer_center,
                )
                margins.append(margin_um)
                expanded = expand_defect_polygon_for_alignment_error(points, margin_um, precision=precision)
            else:
                expanded = [gdstk.Polygon(points)]

            all_polygons.extend(expanded)
            device_polygons.extend(expanded)

    return all_polygons, by_device, legacy_count, bbox_corner_count, polygon_count, margins


def _load_defect_polygons(
    defects_data: dict,
    compensate_alignment_error: bool = True,
    error_angle_deg: float = DEFAULT_ALIGNMENT_ERROR_ANGLE_DEG,
    error_x_px: float = DEFAULT_ALIGNMENT_ERROR_X_PX,
    error_y_px: float = DEFAULT_ALIGNMENT_ERROR_Y_PX,
    extra_margin_um: float = DEFAULT_EXTRA_MARGIN_UM,
    wafer_center: tuple[float, float] = (DEFAULT_WAFER_CENTER_X_UM, DEFAULT_WAFER_CENTER_Y_UM),
    strict_corners: bool = False,
    precision: float = 1e-3,
) -> tuple[list[gdstk.Polygon], int, int, int, list[float]]:
    """Backward-compatible ungrouped loader."""
    polygons, _, legacy, bbox, exact, margins = _load_defect_polygons_grouped(
        defects_data,
        compensate_alignment_error=compensate_alignment_error,
        error_angle_deg=error_angle_deg,
        error_x_px=error_x_px,
        error_y_px=error_y_px,
        extra_margin_um=extra_margin_um,
        wafer_center=wafer_center,
        strict_corners=strict_corners,
        precision=precision,
    )
    return polygons, legacy, bbox, exact, margins


def _polygon_area_sum(polygons: Iterable[gdstk.Polygon]) -> float:
    return float(sum(abs(float(poly.area())) for poly in polygons))


def _metadata_aliases(meta_path: Path, meta: dict) -> set[str]:
    aliases = {_normalize_device_key(meta_path.stem)}
    for field in ("cell_stem", "analysis_png", "legacy_jpg"):
        value = meta.get(field)
        if value:
            aliases.add(_normalize_device_key(Path(str(value)).name))
    wafer = str(meta.get("wafer_id", "")).strip()
    row = meta.get("cell_row")
    col = meta.get("cell_col")
    if wafer and row is not None and col is not None:
        aliases.add(_normalize_device_key(f"{wafer}_cell_{row}-{col}"))
    return aliases


def _metadata_device_polygon(meta: dict) -> gdstk.Polygon | None:
    corners = meta.get("gds_corners_um")
    if isinstance(corners, list) and len(corners) >= 3:
        try:
            pts = np.asarray([(float(x), float(y)) for x, y in corners], dtype=np.float64)
            if abs(_polygon_signed_area(pts)) > 1e-9:
                return gdstk.Polygon(pts)
        except Exception:
            pass

    bbox = meta.get("gds_bbox_um")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            min_x, min_y, max_x, max_y = [float(v) for v in bbox[:4]]
            if max_x > min_x and max_y > min_y:
                return gdstk.rectangle((min_x, min_y), (max_x, max_y))
        except Exception:
            pass
    return None


def load_device_metadata(
    metadata_dir: str | Path,
    defects_data: dict,
) -> tuple[dict[str, dict], list[str]]:
    """Load per-device GDS footprints produced during wafer extraction.

    Returned dictionary is keyed by canonical cell stem.  Aliases are kept in each
    record so JPG/PNG filename differences do not break device matching.
    """
    metadata_dir = Path(metadata_dir)
    warnings: list[str] = []
    if not metadata_dir.exists():
        warnings.append(f"Metadata directory not found: {metadata_dir}")
        return {}, warnings

    json_device_keys = [str(k) for k, v in defects_data.items() if isinstance(v, list)]
    wafer_prefixes = {_wafer_prefix_from_device_name(k) for k in json_device_keys}
    wafer_prefixes.discard("")

    devices: dict[str, dict] = {}
    for meta_path in sorted(metadata_dir.glob("*.json")):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as exc:
            warnings.append(f"Could not read metadata {meta_path.name}: {exc}")
            continue
        if not isinstance(meta, dict):
            continue

        aliases = _metadata_aliases(meta_path, meta)
        if wafer_prefixes and not any(
            any(alias.startswith(prefix + "_cell_") for prefix in wafer_prefixes)
            for alias in aliases
        ):
            continue

        polygon = _metadata_device_polygon(meta)
        if polygon is None:
            warnings.append(f"Metadata {meta_path.name} has no usable gds_bbox_um/gds_corners_um")
            continue

        canonical = _normalize_device_key(meta.get("cell_stem") or meta_path.stem)
        display_name = Path(str(meta.get("legacy_jpg") or f"{canonical}.jpg")).name
        devices[canonical] = {
            "canonical": canonical,
            "display_name": display_name,
            "aliases": aliases,
            "polygon": polygon,
            "metadata_path": str(meta_path),
            "row": meta.get("cell_row"),
            "col": meta.get("cell_col"),
        }

    if not devices:
        warnings.append(f"No usable per-device metadata found in {metadata_dir}")
    return devices, warnings


def _device_defects_for_record(
    record: dict,
    defects_by_device: dict[str, list[gdstk.Polygon]],
) -> list[gdstk.Polygon]:
    out: list[gdstk.Polygon] = []
    seen: set[int] = set()
    for alias in record.get("aliases", set()):
        for poly in defects_by_device.get(alias, []):
            ident = id(poly)
            if ident not in seen:
                seen.add(ident)
                out.append(poly)
    return out


def create_removal_report(
    *,
    report_path: str | Path,
    metadata_dir: str | Path,
    defects_data: dict,
    defects_by_device: dict[str, list[gdstk.Polygon]],
    polys_by_layer_type: dict[tuple[int, int], list[gdstk.Polygon]],
    target_layers: set[int],
    precision: float,
    gds_path: str,
    json_path: str,
    output_path: str,
    compensate_alignment_error: bool,
) -> Path:
    """Write per-device and aggregate removed-area percentages.

    Denominator: original geometry area on the selected target layers inside the
    exact GDS footprint of each extracted device cell.

    Numerator: the portion of that geometry intersected by the final expanded
    defect polygons used for subtraction.  Overlapping defect polygons are counted
    once because gdstk boolean operations union the set before area measurement.
    """
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    devices, warnings = load_device_metadata(metadata_dir, defects_data)
    selected_groups = {
        key: polys for key, polys in polys_by_layer_type.items() if key[0] in target_layers
    }

    rows: list[dict] = []
    # H2P_PROGRESS_REPORT_DEVICES_V2
    _h2p_sorted_devices = sorted(devices.items(), key=lambda kv: (kv[1].get('row') if kv[1].get('row') is not None else 10 ** 9, kv[1].get('col') if kv[1].get('col') is not None else 10 ** 9, kv[0]))
    _h2p_report_bar = ProgressBar(
        "Removal report: devices",
        len(_h2p_sorted_devices),
    )
    for _h2p_device_index, (canonical, record) in enumerate(
        _h2p_sorted_devices,
        start=1,
    ):
        device_window = record["polygon"]
        device_defects = _device_defects_for_record(record, defects_by_device)
        per_layer: dict[int, dict[str, float]] = {}
        total_original = 0.0
        total_removed = 0.0

        for (layer, datatype), layer_polys in selected_groups.items():
            inside = gdstk.boolean(
                layer_polys,
                [device_window],
                "and",
                precision=precision,
                layer=layer,
                datatype=datatype,
            )
            original_area = _polygon_area_sum(inside)
            removed_area = 0.0
            if inside and device_defects:
                removed = gdstk.boolean(
                    inside,
                    device_defects,
                    "and",
                    precision=precision,
                    layer=layer,
                    datatype=datatype,
                )
                removed_area = min(original_area, _polygon_area_sum(removed))

            entry = per_layer.setdefault(layer, {"original": 0.0, "removed": 0.0})
            entry["original"] += original_area
            entry["removed"] += removed_area
            total_original += original_area
            total_removed += removed_area

        percent = 100.0 * total_removed / total_original if total_original > 1e-12 else float("nan")
        rows.append({
            "name": record["display_name"],
            "canonical": canonical,
            "defect_polygon_count": len(device_defects),
            "original": total_original,
            "removed": total_removed,
            "percent": percent,
            "layers": per_layer,
        })
        _h2p_report_bar.update(
            _h2p_device_index,
            extra=str(record.get("display_name", canonical)),
        )
    _h2p_report_bar.done(
        extra=f"calculated {len(rows)} devices",
    )

    valid_rows = [r for r in rows if np.isfinite(r["percent"])]
    mean_percent = float(np.mean([r["percent"] for r in valid_rows])) if valid_rows else float("nan")
    total_original = float(sum(r["original"] for r in valid_rows))
    total_removed = float(sum(r["removed"] for r in valid_rows))
    weighted_percent = 100.0 * total_removed / total_original if total_original > 1e-12 else float("nan")

    # Mention JSON devices that had no matching metadata.  Their masks are still
    # subtracted globally, but a trustworthy device percentage needs a denominator.
    metadata_aliases = set()
    for record in devices.values():
        metadata_aliases.update(record.get("aliases", set()))
    unmatched = sorted(
        _normalize_device_key(k)
        for k, v in defects_data.items()
        if isinstance(v, list) and _normalize_device_key(k) not in metadata_aliases
    )
    if unmatched:
        warnings.append(
            f"{len(unmatched)} JSON device(s) had no matching extraction metadata; "
            "their per-device percentage could not be calculated."
        )

    lines: list[str] = []
    lines.append("GDS DEFECT MASK REMOVAL REPORT")
    lines.append("=" * 78)
    lines.append(f"Input GDS: {gds_path}")
    lines.append(f"Defect JSON: {json_path}")
    lines.append(f"Output GDS: {output_path}")
    lines.append(f"Target layers: {', '.join(str(v) for v in sorted(target_layers))}")
    lines.append(f"Alignment-error expansion included: {'yes' if compensate_alignment_error else 'no'}")
    lines.append(f"Metadata directory: {metadata_dir}")
    lines.append("")
    lines.append(
        "Percent removed = area of selected-layer device geometry intersected by "
        "the final subtraction mask / original selected-layer geometry area."
    )
    lines.append("Overlapping defect masks are counted once.")
    lines.append("")

    if warnings:
        lines.append("WARNINGS")
        lines.append("-" * 78)
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")

    lines.append("PER-DEVICE RESULTS")
    lines.append("-" * 78)
    if not rows:
        lines.append("No device metadata was available, so per-device percentages were not calculated.")
    else:
        header = f"{'Device':<38} {'Mask polys':>10} {'Original um^2':>15} {'Removed um^2':>14} {'Removed %':>11}"
        lines.append(header)
        lines.append("-" * len(header))
        for row in rows:
            pct = f"{row['percent']:.6f}" if np.isfinite(row["percent"]) else "N/A"
            lines.append(
                f"{row['name'][:38]:<38} {row['defect_polygon_count']:>10d} "
                f"{row['original']:>15.3f} {row['removed']:>14.3f} {pct:>11}"
            )
            for layer in sorted(row["layers"]):
                layer_original = row["layers"][layer]["original"]
                layer_removed = row["layers"][layer]["removed"]
                layer_pct = 100.0 * layer_removed / layer_original if layer_original > 1e-12 else float("nan")
                layer_pct_text = f"{layer_pct:.6f}%" if np.isfinite(layer_pct) else "N/A"
                lines.append(
                    f"    layer {layer:<4d}: original={layer_original:.3f} um^2, "
                    f"removed={layer_removed:.3f} um^2, removed={layer_pct_text}"
                )

    lines.append("")
    lines.append("SUMMARY")
    lines.append("-" * 78)
    lines.append(f"Devices with valid denominators: {len(valid_rows)}")
    if valid_rows:
        lines.append(f"Average device removal percentage (unweighted mean): {mean_percent:.6f}%")
        lines.append(f"Overall removal percentage (area weighted): {weighted_percent:.6f}%")
        lines.append(f"Total original selected-layer area: {total_original:.3f} um^2")
        lines.append(f"Total removed selected-layer area: {total_removed:.3f} um^2")
    else:
        lines.append("Average device removal percentage (unweighted mean): N/A")
        lines.append("Overall removal percentage (area weighted): N/A")

    # H2P_PROGRESS_REPORT_FILE_V2
    _h2p_report_file_bar = ProgressBar("Removal report: file", 1)
    _h2p_report_file_bar.status(f"writing {report_path.name}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _h2p_report_file_bar.done(extra=f"saved {report_path.name}")
    print(f"[REPORT] Per-device removal report saved to: {report_path}")
    return report_path


def subtract_defects_from_gds(
    gds_path,
    json_path,
    output_path,
    target_layers=(4,),
    compensate_alignment_error: bool = True,
    error_angle_deg: float = DEFAULT_ALIGNMENT_ERROR_ANGLE_DEG,
    error_x_px: float = DEFAULT_ALIGNMENT_ERROR_X_PX,
    error_y_px: float = DEFAULT_ALIGNMENT_ERROR_Y_PX,
    extra_margin_um: float = DEFAULT_EXTRA_MARGIN_UM,
    wafer_center: tuple[float, float] = (DEFAULT_WAFER_CENTER_X_UM, DEFAULT_WAFER_CENTER_Y_UM),
    strict_corners: bool = False,
    precision: float = 1e-3,
    metadata_dir: str | Path = "extracted_cells/metadata",
    report_path: str | Path | None = None,
    write_report: bool = True,
):
    if not os.path.exists(gds_path):
        raise FileNotFoundError(f"GDS file not found: {gds_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Defect JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        defects_data = json.load(f)

    (
        defect_polygons,
        defects_by_device,
        legacy_count,
        bbox_count,
        polygon_count,
        margins,
    ) = _load_defect_polygons_grouped(
        defects_data,
        compensate_alignment_error=compensate_alignment_error,
        error_angle_deg=error_angle_deg,
        error_x_px=error_x_px,
        error_y_px=error_y_px,
        extra_margin_um=extra_margin_um,
        wafer_center=wafer_center,
        strict_corners=strict_corners,
        precision=precision,
    )

    if legacy_count > 0:
        msg = (
            f"{legacy_count} defect(s) had no corners_gds/polygon_gds and were reconstructed as "
            "axis-aligned boxes from center/width/height. These can be misaligned when the wafer is rotated."
        )
        if strict_corners:
            raise RuntimeError(msg)
        print(f"[WARNING] {msg}")

    print(f"[INFO] {polygon_count} defect(s) loaded using exact polygon_gds.")
    print(f"[INFO] {bbox_count} defect(s) loaded using bbox corners_gds.")
    if compensate_alignment_error:
        print(
            f"[INFO] Alignment-error compensation enabled: angle={error_angle_deg:.6f}°, "
            f"x={error_x_px:.3f}px, y={error_y_px:.3f}px, extra={extra_margin_um:.3f} um"
        )
        if margins:
            arr = np.array(margins, dtype=np.float64)
            print(
                f"[INFO] Defect expansion margin: min={arr.min():.3f} um, "
                f"mean={arr.mean():.3f} um, max={arr.max():.3f} um"
            )
    else:
        print("[INFO] Alignment-error compensation disabled.")

    print(f"[INFO] Reading full GDS: {gds_path}...")
    lib = gdstk.read_gds(gds_path)

    top_cells = lib.top_level()
    if not top_cells:
        raise ValueError("No top-level cells found in GDS.")
    original_top = top_cells[0]

    print("[INFO] Recursively traversing design hierarchy and converting paths...")
    all_polygons = original_top.get_polygons(apply_repetitions=True)
    print(f"[INFO] Discovered {len(all_polygons)} total polygons across all layers.")

    polys_by_layer_type: dict[tuple[int, int], list[gdstk.Polygon]] = {}
    # H2P_PROGRESS_INDEX_POLYGONS_V2
    _h2p_grouping_bar = ProgressBar(
        "Mask: index GDS polygons",
        len(all_polygons),
    )
    for _h2p_polygon_index, poly in enumerate(all_polygons, start=1):
        key = (poly.layer, poly.datatype)
        polys_by_layer_type.setdefault(key, []).append(poly)
        _h2p_grouping_bar.update(
            _h2p_polygon_index,
            extra=f"layer {poly.layer}/{poly.datatype}",
        )
    _h2p_grouping_bar.done(
        extra=f"indexed {len(all_polygons)} polygons",
    )

    final_polygons: list[gdstk.Polygon] = []
    target_layers_set = set(int(layer) for layer in target_layers)
    # H2P_PROGRESS_MASK_LAYERS_V2
    _h2p_layer_groups = list(polys_by_layer_type.items())
    _h2p_mask_bar = ProgressBar(
        "Mask: subtract layers",
        len(_h2p_layer_groups),
    )
    if not defect_polygons:
        print("[INFO] No defects found in JSON. Writing original file unchanged.")
        for _h2p_group_index, ((layer, datatype), layer_polys) in enumerate(
            _h2p_layer_groups,
            start=1,
        ):
            final_polygons.extend(layer_polys)
            _h2p_mask_bar.update(
                _h2p_group_index,
                extra=f"copied layer {layer}/{datatype}",
            )
    else:
        for _h2p_group_index, ((layer, datatype), layer_polys) in enumerate(
            _h2p_layer_groups,
            start=1,
        ):
            if layer in target_layers_set:
                print(
                    f" -> Cutting defect regions from Layer {layer}, Datatype {datatype} "
                    f"({len(layer_polys)} polys)..."
                )
                subtracted = gdstk.boolean(
                    layer_polys,
                    defect_polygons,
                    "not",
                    precision=precision,
                    layer=layer,
                    datatype=datatype,
                )
                final_polygons.extend(subtracted)
            else:
                final_polygons.extend(layer_polys)
            _h2p_mask_bar.update(
                _h2p_group_index,
                extra=f"processed layer {layer}/{datatype}",
                force=True,
            )
    _h2p_mask_bar.done(
        extra=f"prepared {len(final_polygons)} output polygons",
    )

    new_top_cell = gdstk.Cell(original_top.name)
    # H2P_PROGRESS_GDS_ASSEMBLY_V2
    _h2p_assembly_bar = ProgressBar(
        "Output GDS: assemble",
        len(final_polygons),
    )
    for _h2p_output_index, polygon in enumerate(final_polygons, start=1):
        new_top_cell.add(polygon)
        _h2p_assembly_bar.update(
            _h2p_output_index,
            extra=f"polygon {_h2p_output_index}",
        )
    _h2p_assembly_bar.done(
        extra=f"assembled {len(final_polygons)} polygons",
    )
    output_lib = gdstk.Library(name=lib.name, unit=lib.unit, precision=lib.precision)
    output_lib.add(new_top_cell)
    # H2P_PROGRESS_GDS_FILE_V2
    _h2p_gds_file_bar = ProgressBar("Output GDS: file", 1)
    _h2p_gds_file_bar.status(
        f"writing {Path(output_path).name}",
    )
    output_lib.write_gds(output_path)
    _h2p_gds_file_bar.done(
        extra=f"saved {Path(output_path).name}",
    )
    print(f"[SUCCESS] Subtraction complete. Saved flat GDS file to: {output_path}")

    if write_report:
        if report_path is None:
            output = Path(output_path)
            report_path = output.with_name(f"{output.stem}_removal_report.txt")
        create_removal_report(
            report_path=report_path,
            metadata_dir=metadata_dir,
            defects_data=defects_data,
            defects_by_device=defects_by_device,
            polys_by_layer_type=polys_by_layer_type,
            target_layers=target_layers_set,
            precision=precision,
            gds_path=str(gds_path),
            json_path=str(json_path),
            output_path=str(output_path),
            compensate_alignment_error=compensate_alignment_error,
        )


# >>> H2P WAFER-AWARE OUTPUT NAME >>>
def _default_subtracted_output_path(json_path: str | Path) -> Path:
    """Derive a stable output name from the input wafer defect JSON."""
    json_file = Path(json_path)
    wafer_stem = json_file.stem.strip()
    for suffix in ("_device_defects", "_defects"):
        if wafer_stem.lower().endswith(suffix):
            wafer_stem = wafer_stem[: -len(suffix)].rstrip("_- ")
            break
    if not wafer_stem or wafer_stem.lower() in {"device", "devices", "defect", "defects"}:
        wafer_stem = json_file.parent.name.strip() or "wafer"
    return Path(f"{wafer_stem}_subtracted_defects.gds")
# <<< H2P WAFER-AWARE OUTPUT NAME <<<


# >>> H2P BATCH MASK CLI V1 >>>
def _is_none_value(value: object) -> bool:
    return str(value or "").strip().lower() in {"", "none", "null"}


def _batch_defect_json_path(
    record: dict,
    *,
    base_dir: str | Path,
) -> Path:
    """Resolve one wafer's reviewed defect JSON from a batch record."""
    configured = record.get("defect_json", "none")
    if not _is_none_value(configured):
        return Path(str(configured))

    wafer_id = normalize_wafer_id(str(record["id"]))
    return Path(base_dir) / wafer_id / f"{wafer_id}_device_defects.json"


def _default_metadata_dir_for_json(json_path: str | Path) -> Path:
    """Use the per-wafer metadata folder beside the reviewed JSON."""
    return Path(json_path).parent / "metadata"


def _clean_existing_output(path: Path, *, label: str, no_clean: bool) -> None:
    if not path.exists() or no_clean:
        return
    if path.is_dir():
        raise RuntimeError(f"Refusing to remove directory {label.lower()} path: {path}")
    print(f"[Cleanup] Removing stale {label}: {path}")
    path.unlink()


def _run_subtraction_job(
    *,
    json_path: str | Path,
    gds_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None,
    metadata_dir: str | Path | None,
    layers: Iterable[int],
    error_angle_deg: float,
    error_x_px: float,
    error_y_px: float,
    extra_margin_um: float,
    wafer_center_x_um: float,
    wafer_center_y_um: float,
    no_error_compensation: bool,
    strict_corners: bool,
    precision: float,
    no_report: bool,
    no_clean: bool,
) -> None:
    json_path = Path(json_path)
    output_path = Path(output_path)
    report_path = (
        Path(report_path)
        if report_path is not None
        else output_path.with_name(f"{output_path.stem}_removal_report.txt")
    )
    metadata_dir = (
        Path(metadata_dir)
        if metadata_dir is not None
        else _default_metadata_dir_for_json(json_path)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not no_report:
        report_path.parent.mkdir(parents=True, exist_ok=True)

    _clean_existing_output(output_path, label="output GDS", no_clean=no_clean)
    if not no_report:
        _clean_existing_output(report_path, label="report", no_clean=no_clean)

    subtract_defects_from_gds(
        str(gds_path),
        str(json_path),
        str(output_path),
        tuple(int(layer) for layer in layers),
        compensate_alignment_error=not no_error_compensation,
        error_angle_deg=error_angle_deg,
        error_x_px=error_x_px,
        error_y_px=error_y_px,
        extra_margin_um=extra_margin_um,
        wafer_center=(wafer_center_x_um, wafer_center_y_um),
        strict_corners=strict_corners,
        precision=precision,
        metadata_dir=metadata_dir,
        report_path=report_path,
        write_report=not no_report,
    )


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Subtract reviewed defect regions from one wafer or every wafer in "
            "batch_wafers.txt, then write per-device removed-area reports. "
            "Batch mode is selected automatically when no defect JSON positional "
            "argument is supplied."
        )
    )
    parser.add_argument(
        "json_path",
        nargs="?",
        help=(
            "Path to one subtraction-ready defect JSON. Omit this argument to "
            "process all wafers listed in --batch."
        ),
    )
    parser.add_argument(
        "--json",
        dest="json_opt",
        type=str,
        default=None,
        help="Path to one subtraction-ready defect JSON (legacy form)",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default="batch_wafers.txt",
        help="Batch wafer definition file used when no JSON path is supplied",
    )
    parser.add_argument(
        "--base",
        type=str,
        default="extracted_cells",
        help="Base directory containing per-wafer extraction folders",
    )
    parser.add_argument(
        "--wafer",
        action="append",
        default=[],
        help=(
            "In batch mode, process only this wafer. May be supplied multiple "
            "times; names may include or omit the Wafer_ prefix."
        ),
    )
    parser.add_argument(
        "--gds",
        type=str,
        default="future_design.gds",
        help="Path to original GDS. Default: future_design.gds",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Single-wafer output GDS path. In batch mode this is allowed only "
            "when exactly one wafer is selected."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help=(
            "Batch output directory. Each wafer is written as "
            "<Wafer_id>_subtracted_defects.gds. Default: current directory"
        ),
    )
    parser.add_argument(
        "-l",
        "--layers",
        type=int,
        nargs="+",
        default=[1, 4],
        help="Target GDS layers for boolean subtraction, e.g. -l 1 4",
    )
    parser.add_argument(
        "--error-angle-deg",
        type=float,
        default=DEFAULT_ALIGNMENT_ERROR_ANGLE_DEG,
        help="Worst-case residual rotation error in degrees; default: 0.001",
    )
    parser.add_argument(
        "--error-x-px",
        type=float,
        default=DEFAULT_ALIGNMENT_ERROR_X_PX,
        help="Worst-case residual x registration error in crop pixels; default: 2",
    )
    parser.add_argument(
        "--error-y-px",
        type=float,
        default=DEFAULT_ALIGNMENT_ERROR_Y_PX,
        help="Worst-case residual y registration error in crop pixels; default: 2",
    )
    parser.add_argument(
        "--extra-margin-um",
        type=float,
        default=DEFAULT_EXTRA_MARGIN_UM,
        help="Additional fixed safety margin in GDS microns; default: 0",
    )
    parser.add_argument(
        "--wafer-center-x-um",
        type=float,
        default=DEFAULT_WAFER_CENTER_X_UM,
        help="Wafer rotation center X in GDS microns; default: 0",
    )
    parser.add_argument(
        "--wafer-center-y-um",
        type=float,
        default=DEFAULT_WAFER_CENTER_Y_UM,
        help="Wafer rotation center Y in GDS microns; default: 0",
    )
    parser.add_argument(
        "--no-error-compensation",
        action="store_true",
        help="Disable conservative polygon expansion for alignment uncertainty",
    )
    parser.add_argument(
        "--strict-corners",
        action="store_true",
        default=True,
        help=(
            "Fail instead of falling back when any defect is missing GDS "
            "geometry. Default: on"
        ),
    )
    parser.add_argument(
        "--allow-legacy-geometry",
        action="store_false",
        dest="strict_corners",
        help="Allow legacy center/width/height defects without GDS polygons",
    )
    parser.add_argument(
        "--precision",
        type=float,
        default=1e-3,
        help="GDS boolean/offset precision in microns; default: 1e-3",
    )
    parser.add_argument(
        "--metadata-dir",
        type=str,
        default=None,
        help=(
            "Per-cell metadata directory. Default: the metadata folder beside "
            "each wafer's reviewed JSON."
        ),
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help=(
            "Single-wafer report path. In batch mode this is allowed only when "
            "exactly one wafer is selected."
        ),
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write per-device removal reports",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete existing output files before writing",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="In batch mode, stop immediately when one wafer fails",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the batch jobs without creating masks",
    )
    return parser.parse_args()


def _run_single_mode(args: argparse.Namespace, json_path: str | Path) -> int:
    if args.wafer:
        raise ValueError("--wafer is only valid when using batch mode.")
    output_path = Path(args.out) if args.out else _default_subtracted_output_path(json_path)
    print(f"[Runtime] version={SUBTRACT_DEFECTS_VERSION}")
    print("[Mode] Single-wafer mask creation")
    print(f"[Input] Defect JSON: {json_path}")
    print(f"[Input] GDS: {args.gds}")
    print(f"[Output] GDS: {output_path}")
    if args.dry_run:
        return 0

    _run_subtraction_job(
        json_path=json_path,
        gds_path=args.gds,
        output_path=output_path,
        report_path=args.report,
        metadata_dir=args.metadata_dir,
        layers=args.layers,
        error_angle_deg=args.error_angle_deg,
        error_x_px=args.error_x_px,
        error_y_px=args.error_y_px,
        extra_margin_um=args.extra_margin_um,
        wafer_center_x_um=args.wafer_center_x_um,
        wafer_center_y_um=args.wafer_center_y_um,
        no_error_compensation=args.no_error_compensation,
        strict_corners=args.strict_corners,
        precision=args.precision,
        no_report=args.no_report,
        no_clean=args.no_clean,
    )
    return 0


def _run_batch_mode(args: argparse.Namespace) -> int:
    records = parse_batch_file(args.batch)
    wafer_ids = select_wafer_ids(records, args.wafer)
    selected_keys = {wafer_id.casefold() for wafer_id in wafer_ids}
    selected_records = [
        record
        for record in records
        if normalize_wafer_id(str(record["id"])).casefold() in selected_keys
    ]

    if not selected_records:
        raise ValueError("No wafers were selected from the batch file.")
    if len(selected_records) > 1 and args.out:
        raise ValueError("--out cannot represent multiple batch outputs; use --output-dir.")
    if len(selected_records) > 1 and args.report:
        raise ValueError("--report cannot represent multiple batch reports; use default names.")
    if len(selected_records) > 1 and args.metadata_dir:
        raise ValueError(
            "--metadata-dir cannot represent multiple wafer metadata folders. "
            "Place metadata beside each reviewed JSON or select one --wafer."
        )

    output_dir = Path(args.output_dir)
    total = len(selected_records)
    failures: list[tuple[str, str]] = []
    attempted = 0
    succeeded = 0
    print(f"[Runtime] version={SUBTRACT_DEFECTS_VERSION}")
    print(f"[Mode] Batch mask creation from {args.batch}")
    print(f"[Batch] Selected wafers: {', '.join(wafer_ids)}")
    print(f"[Batch] Input GDS: {args.gds}")
    print(f"[Batch] Target layers: {', '.join(str(v) for v in args.layers)}")

    batch_bar = ProgressBar("Batch masks", total)
    for index, record in enumerate(selected_records, start=1):
        attempted += 1
        wafer_id = normalize_wafer_id(str(record["id"]))
        json_path = _batch_defect_json_path(record, base_dir=args.base)
        output_path = (
            Path(args.out)
            if args.out
            else output_dir / f"{wafer_id}_subtracted_defects.gds"
        )
        report_path = (
            Path(args.report)
            if args.report
            else output_path.with_name(f"{output_path.stem}_removal_report.txt")
        )
        metadata_dir = (
            Path(args.metadata_dir)
            if args.metadata_dir
            else _default_metadata_dir_for_json(json_path)
        )

        print("\n" + "=" * 72)
        print(f" MASK RUN [{index}/{total}]: {wafer_id}")
        print("=" * 72)
        print(f"[{wafer_id}] Defect JSON: {json_path}")
        print(f"[{wafer_id}] Metadata: {metadata_dir}")
        print(f"[{wafer_id}] Output GDS: {output_path}")
        if not args.no_report:
            print(f"[{wafer_id}] Removal report: {report_path}")

        if args.dry_run:
            succeeded += 1
            batch_bar.update(index, extra=f"planned {wafer_id}", force=True)
            continue

        try:
            _run_subtraction_job(
                json_path=json_path,
                gds_path=args.gds,
                output_path=output_path,
                report_path=report_path,
                metadata_dir=metadata_dir,
                layers=args.layers,
                error_angle_deg=args.error_angle_deg,
                error_x_px=args.error_x_px,
                error_y_px=args.error_y_px,
                extra_margin_um=args.extra_margin_um,
                wafer_center_x_um=args.wafer_center_x_um,
                wafer_center_y_um=args.wafer_center_y_um,
                no_error_compensation=args.no_error_compensation,
                strict_corners=args.strict_corners,
                precision=args.precision,
                no_report=args.no_report,
                no_clean=args.no_clean,
            )
        except KeyboardInterrupt:
            print(f"\n[{wafer_id}] Interrupted by user.", flush=True)
            raise
        except Exception as exc:
            failures.append((wafer_id, str(exc)))
            print(f"[{wafer_id}] ERROR: {exc}", flush=True)
            batch_bar.update(index, extra=f"FAILED {wafer_id}", force=True)
            if args.stop_on_error:
                break
            continue

        succeeded += 1
        batch_bar.update(index, extra=f"completed {wafer_id}", force=True)

    unprocessed = total - attempted
    batch_bar.done(
        extra=(
            f"succeeded {succeeded}/{total}; failed {len(failures)}; "
            f"unprocessed {unprocessed}"
        )
    )
    print("\n" + "=" * 72)
    if args.dry_run:
        print(" BATCH MASK PLAN COMPLETE")
        return 0
    if failures:
        print(" BATCH MASK CREATION COMPLETE WITH FAILURES")
        for wafer_id, error in failures:
            print(f" - {wafer_id}: {error}")
        return 1
    print(" BATCH MASK CREATION COMPLETE")
    return 0


def main() -> int:
    args = _parse_cli_args()
    if args.json_opt and args.json_path and Path(args.json_opt) != Path(args.json_path):
        raise ValueError("Provide either the positional JSON path or --json, not both.")
    json_path = args.json_opt or args.json_path
    if json_path:
        return _run_single_mode(args, json_path)
    return _run_batch_mode(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[Interrupted] Mask creation stopped by user.", flush=True)
        raise SystemExit(130)
