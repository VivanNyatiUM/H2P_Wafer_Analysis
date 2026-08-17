import design_geometry
import os
import re
import json
import math
import argparse
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import gdstk
from h2p_progress import ProgressBar
from batch_wafers_parser import parse_batch_file
try:
    from remove_hanging_gridlines import (
        find_hanging_boundaries,
        infer_cells,
        parse_flat_gds,
        write_filtered_gds,
    )
except (ImportError, SystemExit) as exc:  # Delayed error unless cleanup is enabled.
    find_hanging_boundaries = None
    infer_cells = None
    parse_flat_gds = None
    write_filtered_gds = None
    _GRIDLINE_CLEANUP_IMPORT_ERROR = exc
else:
    _GRIDLINE_CLEANUP_IMPORT_ERROR = None

# >>> H2P SUBTRACTION PROGRESS V2 >>>

DEFAULT_ALIGNMENT_ERROR_ANGLE_DEG = 0.001
DEFAULT_ALIGNMENT_ERROR_X_PX = 2.0
DEFAULT_ALIGNMENT_ERROR_Y_PX = 2.0
DEFAULT_EXTRA_MARGIN_UM = 0.0
DEFAULT_WAFER_CENTER_X_UM = 0.0
DEFAULT_WAFER_CENTER_Y_UM = 0.0
SUBTRACT_DEFECTS_VERSION = "post-gridline-cleanup-report-v3-2026-07-31"


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


def apply_smart_shunts(polygons, defect_polygons, layer, datatype, precision=1e-3):
    if not polygons:
        return []

    # 1. Extract segment data
    segments = []
    for p in polygons:
        bbox = p.bounding_box()
        # bbox is ((min_x, min_y), (max_x, max_y))
        w = bbox[1][0] - bbox[0][0]
        h = bbox[1][1] - bbox[0][1]
        
        # Only process vertical-ish segments
        if h > w:
            segments.append({
                'x_mid': (bbox[0][0] + bbox[1][0]) / 2,
                'x_min': bbox[0][0],
                'x_max': bbox[1][0],
                'y_min': bbox[0][1],
                'y_max': bbox[1][1],
                'poly': p
            })

    if not segments:
        return polygons

    # 2. Estimate grid properties
    # Use a set to get unique X positions to calculate pitch accurately
    unique_xs = sorted(list(set([round(s['x_mid'], 2) for s in segments])))
    if len(unique_xs) > 1:
        avg_pitch = np.median(np.diff(unique_xs))
    else:
        avg_pitch = 100.0 # Fallback if only one line exists
        
    avg_width = np.median([s['x_max'] - s['x_min'] for s in segments])
    
    # Range to look for a neighbor (up to 2.5 times the pitch)
    search_range = avg_pitch * 2.5 
    # Vertical "forgiveness" - looks for a neighbor within this Y-distance of the cut end
    y_tolerance = 5.0 

    new_shunts = []

    for seg in segments:
        # Try to shunt from both the top and the bottom of every cut segment
        for y_level in [seg['y_min'], seg['y_max']]:
            
            best_neighbor_x = None
            min_dist = float('inf')
            direction = 0 # 1 for right, -1 for left

            for other in segments:
                # Skip self
                if abs(other['x_mid'] - seg['x_mid']) < 0.1:
                    continue
                
                # Check if this neighbor is within horizontal search range
                dist_x = other['x_mid'] - seg['x_mid']
                if abs(dist_x) > search_range:
                    continue

                # Robust Y-Overlap Check:
                # Does the neighbor exist at this Y level (with a small tolerance)?
                if other['y_min'] - y_tolerance <= y_level <= other['y_max'] + y_tolerance:
                    if abs(dist_x) < min_dist:
                        min_dist = abs(dist_x)
                        best_neighbor_x = other['x_mid']
                        direction = 1 if dist_x > 0 else -1

            # 3. Create the shunt if a neighbor was found
            if direction != 0:
                # OLD LOGIC: x_end = best_neighbor_x (stops at the center)
                # IMPROVED LOGIC: 
                # We start from the center of the current segment 
                # and end at the center of the neighbor segment.
                # This ensures a massive overlap and a guaranteed connection.
                
                x_start = seg['x_mid'] 
                x_end = best_neighbor_x
                
                # Center the shunt vertically on the cut point
                shunt_y_min = y_level - (avg_width / 2)
                
                shunt = gdstk.rectangle(
                    (x_start, shunt_y_min),
                    (x_end, shunt_y_min + avg_width),
                    layer=layer, datatype=datatype
                )
                new_shunts.append(shunt)

    if not new_shunts:
        return polygons

    # NEW: Clip the shunts so they do not cross into defect areas
    # This removes any part of a shunt that overlaps with a defect blob
    valid_shunts = gdstk.boolean(
        new_shunts, 
        defect_polygons, 
        "not", 
        precision=precision, 
        layer=layer, 
        datatype=datatype
    )

    # Union the clipped shunts with the original gridlines
    return gdstk.boolean(
        polygons, 
        valid_shunts, 
        "or", 
        precision=precision*10, 
        layer=layer, 
        datatype=datatype
    )


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



def _group_polygons_by_layer_type(
    polygons: Iterable[gdstk.Polygon],
) -> dict[tuple[int, int], list[gdstk.Polygon]]:
    grouped: dict[tuple[int, int], list[gdstk.Polygon]] = {}
    for polygon in polygons:
        grouped.setdefault((polygon.layer, polygon.datatype), []).append(polygon)
    return grouped


def _read_flat_polygons_by_layer_type(
    gds_path: str | Path,
) -> dict[tuple[int, int], list[gdstk.Polygon]]:
    """Read the final flat GDS and group polygons by layer/datatype."""
    lib = gdstk.read_gds(str(gds_path))
    top_cells = lib.top_level()
    if not top_cells:
        raise ValueError(f"No top-level cells found in final GDS: {gds_path}")
    polygons = top_cells[0].get_polygons(apply_repetitions=True)
    return _group_polygons_by_layer_type(polygons)


def run_hanging_gridline_cleanup(
    *,
    input_gds: str | Path,
    output_gds: str | Path,
    target_layer: int = 3,
    reference_layer: int = 7,
    span_fraction: float = 0.60,
    report_path: str | Path | None = None,
    logical_input_path: str | Path | None = None,
) -> dict:
    """Apply remove_hanging_gridlines.py to an already-flat subtraction result."""
    if _GRIDLINE_CLEANUP_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Hanging-gridline cleanup is enabled, but remove_hanging_gridlines.py "
            "could not be imported. Keep it beside subtract_defects.py and ensure "
            "Shapely is installed."
        ) from _GRIDLINE_CLEANUP_IMPORT_ERROR

    input_gds = Path(input_gds)
    output_gds = Path(output_gds)
    if input_gds.resolve() == output_gds.resolve():
        raise ValueError("Gridline-cleanup input and output paths must be different")

    print(
        f"[CLEANUP] Removing hanging layer-{target_layer} gridlines "
        f"using reference layer {reference_layer}..."
    )
    cleanup_lib = gdstk.read_gds(str(input_gds))
    if cleanup_lib.unit <= 0 or cleanup_lib.precision <= 0:
        raise ValueError("Invalid GDS unit/precision for gridline cleanup")
    dbu_to_user_unit = cleanup_lib.precision / cleanup_lib.unit

    parsed = parse_flat_gds(input_gds)
    cells, x_pitch, y_pitch = infer_cells(parsed.boundaries, reference_layer)
    removed, cell_reports, outside_count = find_hanging_boundaries(
        parsed.boundaries,
        cells,
        target_layer=target_layer,
        x_pitch=x_pitch,
        y_pitch=y_pitch,
        span_fraction=span_fraction,
    )
    write_filtered_gds(parsed, output_gds, removed)

    for entry in cell_reports:
        center_dbu = entry.get("center_dbu", [0.0, 0.0])
        entry["center_user"] = [
            float(center_dbu[0]) * dbu_to_user_unit,
            float(center_dbu[1]) * dbu_to_user_unit,
        ]

    report = {
        "input_gds": str(logical_input_path or input_gds),
        "output_gds": str(output_gds),
        "target_layer": int(target_layer),
        "reference_layer": int(reference_layer),
        "span_fraction": float(span_fraction),
        "cell_count": len(cells),
        "cell_pitch_dbu": [x_pitch, y_pitch],
        "dbu_to_user_unit": dbu_to_user_unit,
        "cell_pitch_user": [x_pitch * dbu_to_user_unit, y_pitch * dbu_to_user_unit],
        "removed_boundary_count": len(removed),
        "unchanged_target_boundaries_outside_cells": outside_count,
        "cells_with_removals": sum(
            1 for entry in cell_reports if entry["removed_boundary_count"] > 0
        ),
        "fallback_cell_count": sum(
            1 for entry in cell_reports if entry["fallback_used"]
        ),
        "cells": cell_reports,
    }

    if report_path is not None:
        cleanup_report_path = Path(report_path)
        cleanup_report_path.parent.mkdir(parents=True, exist_ok=True)
        cleanup_report_path.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[CLEANUP] Cleanup report saved to: {cleanup_report_path}")

    print(
        f"[CLEANUP] Detected {len(cells)} cells and removed "
        f"{len(removed)} hanging layer-{target_layer} BOUNDARY elements."
    )
    return report


def _point_in_polygon(point: tuple[float, float], polygon: gdstk.Polygon) -> bool:
    """Return True when a point is inside or on the edge of a device footprint."""
    x, y = point
    points = np.asarray(polygon.points, dtype=np.float64)
    if len(points) < 3:
        return False

    inside = False
    j = len(points) - 1
    for i in range(len(points)):
        xi, yi = points[i]
        xj, yj = points[j]

        edge = np.array([xj - xi, yj - yi], dtype=np.float64)
        rel = np.array([x - xi, y - yi], dtype=np.float64)
        cross = abs(float(edge[0] * rel[1] - edge[1] * rel[0]))
        if cross <= 1e-9:
            dot = float(np.dot(rel, edge))
            edge_len_sq = float(np.dot(edge, edge))
            if -1e-9 <= dot <= edge_len_sq + 1e-9:
                return True

        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-300) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _cleanup_removed_for_device(record: dict, cleanup_summary: dict | None) -> int:
    if not cleanup_summary:
        return 0

    device_polygon = record.get("polygon")
    if device_polygon is not None:
        spatial_matches = []
        for entry in cleanup_summary.get("cells", []):
            center = entry.get("center_user")
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                point = (float(center[0]), float(center[1]))
                if _point_in_polygon(point, device_polygon):
                    spatial_matches.append(entry)
        if len(spatial_matches) == 1:
            return int(spatial_matches[0].get("removed_boundary_count", 0))

    # Compatibility fallback for older cleanup summaries without center_user.
    try:
        row = int(record.get("row"))
        col = int(record.get("col"))
    except (TypeError, ValueError):
        return 0
    for entry in cleanup_summary.get("cells", []):
        try:
            if int(entry.get("row")) == row and int(entry.get("column")) == col:
                return int(entry.get("removed_boundary_count", 0))
        except (TypeError, ValueError):
            continue
    return 0


def create_removal_report(
    *,
    report_path: str | Path,
    metadata_dir: str | Path,
    defects_data: dict,
    defects_by_device: dict[str, list[gdstk.Polygon]],
    original_polys_by_layer_type: dict[tuple[int, int], list[gdstk.Polygon]],
    final_polys_by_layer_type: dict[tuple[int, int], list[gdstk.Polygon]],
    report_layers: set[int],
    precision: float,
    gds_path: str,
    json_path: str,
    output_path: str,
    compensate_alignment_error: bool,
    cleanup_summary: dict | None = None,
) -> Path:
    """Write per-device removal percentages for the actual final output GDS.

    Denominator: original geometry on all reported layers inside each exact device
    footprint. Numerator: original geometry absent from the final post-cleanup GDS.
    This includes both defect subtraction and hanging-gridline cleanup while
    ignoring newly added shunt area.
    """
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    devices, warnings = load_device_metadata(metadata_dir, defects_data)
    selected_original = {
        key: polys
        for key, polys in original_polys_by_layer_type.items()
        if key[0] in report_layers
    }

    rows: list[dict] = []
    sorted_devices = sorted(
        devices.items(),
        key=lambda kv: (
            kv[1].get("row") if kv[1].get("row") is not None else 10**9,
            kv[1].get("col") if kv[1].get("col") is not None else 10**9,
            kv[0],
        ),
    )
    report_bar = ProgressBar("Removal report: devices", len(sorted_devices))
    for device_index, (canonical, record) in enumerate(sorted_devices, start=1):
        device_window = record["polygon"]
        device_defects = _device_defects_for_record(record, defects_by_device)
        cleanup_removed_boundaries = _cleanup_removed_for_device(record, cleanup_summary)
        per_layer: dict[int, dict[str, float]] = {}
        total_original = 0.0
        total_remaining_original = 0.0
        total_removed = 0.0

        for (layer, datatype), original_layer_polys in selected_original.items():
            original_inside = gdstk.boolean(
                original_layer_polys,
                [device_window],
                "and",
                precision=precision,
                layer=layer,
                datatype=datatype,
            )
            original_area = _polygon_area_sum(original_inside)
            removed_area = 0.0
            remaining_original_area = original_area

            final_layer_polys = final_polys_by_layer_type.get((layer, datatype), [])
            if original_inside:
                if final_layer_polys:
                    removed_original = gdstk.boolean(
                        original_inside,
                        final_layer_polys,
                        "not",
                        precision=precision,
                        layer=layer,
                        datatype=datatype,
                    )
                    removed_area = min(original_area, _polygon_area_sum(removed_original))
                else:
                    removed_area = original_area
                remaining_original_area = max(0.0, original_area - removed_area)

            entry = per_layer.setdefault(
                layer,
                {"original": 0.0, "remaining": 0.0, "removed": 0.0},
            )
            entry["original"] += original_area
            entry["remaining"] += remaining_original_area
            entry["removed"] += removed_area
            total_original += original_area
            total_remaining_original += remaining_original_area
            total_removed += removed_area

        percent = (
            100.0 * total_removed / total_original
            if total_original > 1e-12
            else float("nan")
        )
        rows.append(
            {
                "name": record["display_name"],
                "canonical": canonical,
                "defect_polygon_count": len(device_defects),
                "cleanup_removed_boundaries": cleanup_removed_boundaries,
                "original": total_original,
                "remaining": total_remaining_original,
                "removed": total_removed,
                "percent": percent,
                "layers": per_layer,
            }
        )
        report_bar.update(
            device_index,
            extra=str(record.get("display_name", canonical)),
        )
    report_bar.done(extra=f"calculated {len(rows)} devices")

    valid_rows = [row for row in rows if np.isfinite(row["percent"])]
    mean_percent = (
        float(np.mean([row["percent"] for row in valid_rows]))
        if valid_rows
        else float("nan")
    )
    total_original = float(sum(row["original"] for row in valid_rows))
    total_remaining = float(sum(row["remaining"] for row in valid_rows))
    total_removed = float(sum(row["removed"] for row in valid_rows))
    weighted_percent = (
        100.0 * total_removed / total_original
        if total_original > 1e-12
        else float("nan")
    )

    metadata_aliases = set()
    for record in devices.values():
        metadata_aliases.update(record.get("aliases", set()))
    unmatched = sorted(
        _normalize_device_key(key)
        for key, value in defects_data.items()
        if isinstance(value, list) and _normalize_device_key(key) not in metadata_aliases
    )
    if unmatched:
        warnings.append(
            f"{len(unmatched)} JSON device(s) had no matching extraction metadata; "
            "their per-device percentage could not be calculated."
        )

    cleanup_enabled = cleanup_summary is not None
    lines: list[str] = []
    lines.append("FINAL GDS DEVICE REMOVAL REPORT")
    lines.append("=" * 92)
    lines.append(f"Input GDS: {gds_path}")
    lines.append(f"Defect JSON: {json_path}")
    lines.append(f"Final output GDS: {output_path}")
    lines.append(f"Reported layers: {', '.join(str(v) for v in sorted(report_layers))}")
    lines.append(
        f"Alignment-error expansion included: "
        f"{'yes' if compensate_alignment_error else 'no'}"
    )
    lines.append(f"Hanging-gridline cleanup included: {'yes' if cleanup_enabled else 'no'}")
    if cleanup_enabled:
        lines.append(
            "Cleanup settings: "
            f"target layer={cleanup_summary['target_layer']}, "
            f"reference layer={cleanup_summary['reference_layer']}, "
            f"span fraction={cleanup_summary['span_fraction']:.3f}"
        )
        lines.append(
            "Cleanup result: "
            f"removed boundaries={cleanup_summary['removed_boundary_count']}, "
            f"cells with removals={cleanup_summary['cells_with_removals']}, "
            f"fallback cells={cleanup_summary['fallback_cell_count']}"
        )
    lines.append(f"Metadata directory: {metadata_dir}")
    lines.append("")
    lines.append(
        "Percent removed = original reported-layer geometry that is absent from "
        "the final post-cleanup GDS / original reported-layer geometry."
    )
    lines.append(
        "This measures the actual final GDS, including defect subtraction and "
        "hanging-gridline cleanup; newly added shunt area does not hide removals."
    )
    lines.append("")

    if warnings:
        lines.append("WARNINGS")
        lines.append("-" * 92)
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")

    lines.append("PER-DEVICE RESULTS")
    lines.append("-" * 92)
    if not rows:
        lines.append(
            "No device metadata was available, so per-device percentages were not calculated."
        )
    else:
        header = (
            f"{'Device':<34} {'Defect polys':>12} {'Grid rm':>8} "
            f"{'Original um^2':>15} {'Final orig um^2':>16} "
            f"{'Removed um^2':>14} {'Removed %':>11}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for row in rows:
            pct = f"{row['percent']:.6f}" if np.isfinite(row["percent"]) else "N/A"
            lines.append(
                f"{row['name'][:34]:<34} {row['defect_polygon_count']:>12d} "
                f"{row['cleanup_removed_boundaries']:>8d} "
                f"{row['original']:>15.3f} {row['remaining']:>16.3f} "
                f"{row['removed']:>14.3f} {pct:>11}"
            )
            for layer in sorted(row["layers"]):
                layer_original = row["layers"][layer]["original"]
                layer_remaining = row["layers"][layer]["remaining"]
                layer_removed = row["layers"][layer]["removed"]
                layer_pct = (
                    100.0 * layer_removed / layer_original
                    if layer_original > 1e-12
                    else float("nan")
                )
                layer_pct_text = (
                    f"{layer_pct:.6f}%" if np.isfinite(layer_pct) else "N/A"
                )
                lines.append(
                    f"    layer {layer:<4d}: original={layer_original:.3f} um^2, "
                    f"final-original={layer_remaining:.3f} um^2, "
                    f"removed={layer_removed:.3f} um^2 ({layer_pct_text})"
                )

    lines.append("")
    lines.append("SUMMARY")
    lines.append("-" * 92)
    lines.append(f"Devices with valid denominators: {len(valid_rows)}")
    if valid_rows:
        lines.append(
            f"Average device removal percentage (unweighted mean): {mean_percent:.6f}%"
        )
        lines.append(
            f"Overall removal percentage (area weighted): {weighted_percent:.6f}%"
        )
        lines.append(f"Total original reported-layer area: {total_original:.3f} um^2")
        lines.append(
            f"Total original area still present in final GDS: {total_remaining:.3f} um^2"
        )
        lines.append(f"Total original area removed in final GDS: {total_removed:.3f} um^2")
    else:
        lines.append("Average device removal percentage (unweighted mean): N/A")
        lines.append("Overall removal percentage (area weighted): N/A")

    report_file_bar = ProgressBar("Removal report: file", 1)
    report_file_bar.status(f"writing {report_path.name}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report_file_bar.done(extra=f"saved {report_path.name}")
    print(f"[REPORT] Final-device removal report saved to: {report_path}")
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
    cleanup_hanging_gridlines: bool = True,
    cleanup_target_layer: int = 3,
    cleanup_reference_layer: int = 7,
    cleanup_span_fraction: float = 0.60,
    cleanup_report_path: str | Path | None = None,
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
                # --- ROBUST LOGIC FOR LAYER 3 ---
                if layer == 3:
                    print(f" -> Separating Borders and Fingers for Layer {layer}...")
                    
                    # 1. Determine typical finger width
                    # We take the median width of all polygons to find the "standard" finger size
                    all_widths = sorted([p.bounding_box()[1][0] - p.bounding_box()[0][0] for p in layer_polys])
                    typical_finger_width = np.median(all_widths)
                    print(f"    - Detected typical finger width: {typical_finger_width:.3f}um")

                    fingers = []
                    borders = []
                    
                    # 2. Refined Classification
                    for p in layer_polys:
                        bbox = p.bounding_box()
                        w = bbox[1][0] - bbox[0][0]
                        h = bbox[1][1] - bbox[0][1]
                        
                        # A FINGER is narrow (close to typical width) AND vertically oriented (height > width)
                        # A BORDER is either:
                        #  a) Horizontal (Width > Height)
                        #  b) Thick (Width is significantly larger than a gridline)
                        is_narrow = w < (typical_finger_width * 1.5)
                        is_vertical = h > w
                        
                        if is_narrow and is_vertical:
                            fingers.append(p)
                        else:
                            # This catches fractured busbar segments and vertical frame bars
                            borders.append(p)
                    
                    print(f"    - Classified {len(borders)} polygons as Protected Border.")
                    print(f"    - Classified {len(fingers)} polygons as Gridline Fingers.")

                    # 3. Process Fingers (Cut defects + Apply shunts)
                    subtracted_fingers = gdstk.boolean(fingers, defect_polygons, "not", precision=precision, layer=layer, datatype=datatype)
                    fixed_fingers = apply_smart_shunts(subtracted_fingers, defect_polygons, layer, datatype, precision)

                    # 4. RECOMBINE (Border is preserved 100%)
                    subtracted = gdstk.boolean(fixed_fingers, borders, "or", precision=precision*10, layer=layer, datatype=datatype)

                else:
                    # Standard logic for other layers
                    print(f" -> Cutting defect regions from Layer {layer}...")
                    subtracted = gdstk.boolean(layer_polys, defect_polygons, "not", precision=precision, layer=layer, datatype=datatype)
                
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
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    cleanup_summary: dict | None = None

    if cleanup_hanging_gridlines:
        with tempfile.NamedTemporaryFile(
            dir=output_path_obj.parent,
            prefix=f".{output_path_obj.stem}_",
            suffix="_before_gridline_cleanup.gds",
            delete=False,
        ) as temp_stream:
            intermediate_path = Path(temp_stream.name)
        try:
            gds_file_bar = ProgressBar("Output GDS: subtraction file", 1)
            gds_file_bar.status(f"writing {intermediate_path.name}")
            output_lib.write_gds(str(intermediate_path))
            gds_file_bar.done(extra="saved temporary subtraction result")

            cleanup_summary = run_hanging_gridline_cleanup(
                input_gds=intermediate_path,
                output_gds=output_path_obj,
                target_layer=cleanup_target_layer,
                reference_layer=cleanup_reference_layer,
                span_fraction=cleanup_span_fraction,
                report_path=cleanup_report_path,
                logical_input_path=f"{output_path_obj} (before hanging-gridline cleanup)",
            )
        finally:
            try:
                intermediate_path.unlink()
            except FileNotFoundError:
                pass
    else:
        gds_file_bar = ProgressBar("Output GDS: file", 1)
        gds_file_bar.status(f"writing {output_path_obj.name}")
        output_lib.write_gds(str(output_path_obj))
        gds_file_bar.done(extra=f"saved {output_path_obj.name}")

    print(f"[SUCCESS] Final flat GDS saved to: {output_path_obj}")

    if write_report:
        if report_path is None:
            report_path = output_path_obj.with_name(
                f"{output_path_obj.stem}_removal_report.txt"
            )
        final_polys_by_layer_type = _read_flat_polygons_by_layer_type(output_path_obj)
        report_layers = set(target_layers_set)
        if cleanup_hanging_gridlines:
            report_layers.add(int(cleanup_target_layer))
        create_removal_report(
            report_path=report_path,
            metadata_dir=metadata_dir,
            defects_data=defects_data,
            defects_by_device=defects_by_device,
            original_polys_by_layer_type=polys_by_layer_type,
            final_polys_by_layer_type=final_polys_by_layer_type,
            report_layers=report_layers,
            precision=precision,
            gds_path=str(gds_path),
            json_path=str(json_path),
            output_path=str(output_path_obj),
            compensate_alignment_error=compensate_alignment_error,
            cleanup_summary=cleanup_summary,
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

    wafer_id = str(record["id"]).strip()
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
    no_gridline_cleanup: bool,
    cleanup_target_layer: int,
    cleanup_reference_layer: int,
    cleanup_span_fraction: float,
    cleanup_report_path: str | Path | None,
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
    cleanup_report_path = (
        Path(cleanup_report_path)
        if cleanup_report_path is not None
        else output_path.with_name(f"{output_path.stem}_cleanup_report.json")
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not no_report:
        report_path.parent.mkdir(parents=True, exist_ok=True)

    _clean_existing_output(output_path, label="output GDS", no_clean=no_clean)
    if not no_report:
        _clean_existing_output(report_path, label="report", no_clean=no_clean)
    if not no_gridline_cleanup:
        _clean_existing_output(
            cleanup_report_path,
            label="cleanup report",
            no_clean=no_clean,
        )

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
        cleanup_hanging_gridlines=not no_gridline_cleanup,
        cleanup_target_layer=cleanup_target_layer,
        cleanup_reference_layer=cleanup_reference_layer,
        cleanup_span_fraction=cleanup_span_fraction,
        cleanup_report_path=cleanup_report_path,
    )


def _select_batch_records(
    records: list[dict[str, str | None]],
    wafer_selectors: list[str] | None = None,
    folder_selectors: list[str] | None = None,
) -> list[dict[str, str | None]]:
    """Select batch records by wafer ID/name and/or folder_name group."""
    requested_wafers = {
        str(value).strip().casefold()
        for value in (wafer_selectors or [])
        if str(value).strip()
    }
    requested_folders = {
        str(value).strip().casefold()
        for value in (folder_selectors or [])
        if str(value).strip()
    }

    if not requested_wafers and not requested_folders:
        return list(records)

    selected_records: list[dict[str, str | None]] = []
    for record in records:
        wafer_id = str(record["id"]).strip()
        aliases = {wafer_id.casefold()}
        if wafer_id.casefold().startswith("wafer_"):
            aliases.add(wafer_id[6:].casefold())

        source_group = str(record.get("source_group") or "").strip()
        wafer_match = bool(requested_wafers & aliases)
        folder_match = (
            bool(source_group)
            and source_group.casefold() in requested_folders
        )
        if wafer_match or folder_match:
            selected_records.append(record)

    if not selected_records:
        available_wafers = ", ".join(str(record["id"]) for record in records)
        available_folders = ", ".join(
            sorted(
                {
                    str(record.get("source_group")).strip()
                    for record in records
                    if record.get("source_group")
                },
                key=str.casefold,
            )
        )
        raise ValueError(
            "No wafers matched the requested filter. "
            f"Available wafers: {available_wafers or '(none)'}. "
            f"Available folder_name groups: {available_folders or '(none)'}"
        )

    return selected_records


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Subtract reviewed defect regions from one wafer or every wafer in "
            "batch_wafers.txt, remove hanging gridlines, then write final per-device removed-area reports. "
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
        "--folder",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "In batch mode, process every wafer expanded from this folder_name "
            "group. Repeatable and may be combined with --wafer."
        ),
    )
    parser.add_argument(
        "--gds",
        type=str,
        default=None,
        help="Path to original GDS. Default: config.json gds_path",
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
        "--no-gridline-cleanup",
        action="store_true",
        help="Skip the integrated remove_hanging_gridlines.py pass",
    )
    parser.add_argument(
        "--cleanup-target-layer",
        type=int,
        default=3,
        help="Layer cleaned for disconnected grid pieces; default: 3",
    )
    parser.add_argument(
        "--cleanup-reference-layer",
        type=int,
        default=7,
        help="Regular finger-array layer used to infer cells; default: 7",
    )
    parser.add_argument(
        "--cleanup-span-fraction",
        type=float,
        default=0.60,
        help="Required connected-component cell span; default: 0.60",
    )
    parser.add_argument(
        "--cleanup-report",
        type=str,
        default=None,
        help=(
            "Single-wafer cleanup JSON path. Default: "
            "<output>_cleanup_report.json"
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
    if args.wafer or args.folder:
        raise ValueError("--wafer and --folder are only valid when using batch mode.")
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
        no_gridline_cleanup=args.no_gridline_cleanup,
        cleanup_target_layer=args.cleanup_target_layer,
        cleanup_reference_layer=args.cleanup_reference_layer,
        cleanup_span_fraction=args.cleanup_span_fraction,
        cleanup_report_path=args.cleanup_report,
    )
    return 0


def _run_batch_mode(args: argparse.Namespace) -> int:
    records = parse_batch_file(args.batch)

    selected_records = _select_batch_records(
        records,
        wafer_selectors=args.wafer,
        folder_selectors=args.folder,
    )

    wafer_ids = [str(record["id"]) for record in selected_records]

    if len(selected_records) > 1 and args.out:
        raise ValueError("--out cannot represent multiple batch outputs; use --output-dir.")
    if len(selected_records) > 1 and args.report:
        raise ValueError("--report cannot represent multiple batch reports; use default names.")
    if len(selected_records) > 1 and args.cleanup_report:
        raise ValueError(
            "--cleanup-report cannot represent multiple batch reports; use default names."
        )
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
        wafer_id = str(record["id"]).strip()
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
        cleanup_report_path = (
            Path(args.cleanup_report)
            if args.cleanup_report
            else output_path.with_name(f"{output_path.stem}_cleanup_report.json")
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
                no_gridline_cleanup=args.no_gridline_cleanup,
                cleanup_target_layer=args.cleanup_target_layer,
                cleanup_reference_layer=args.cleanup_reference_layer,
                cleanup_span_fraction=args.cleanup_span_fraction,
                cleanup_report_path=cleanup_report_path,
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
    args.gds = str(design_geometry.resolve_design_path(args.gds))
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
