"""Single-design GDS geometry for the H2P device viewer.

The active production GDS is selected by ``config.json`` via ``gds_path``.
Layer selection, device indexing, marker extraction, and wafer geometry live
here so the rest of the pipeline does not contain design-mode branches.
"""
from __future__ import annotations

import json

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import gdstk
import numpy as np

WAFER_LAYER = (2, 0)
MARKER_LAYER = (4, 0)
DEVICE_LAYER = (8, 0)
EXPECTED_ROW_COUNTS = (4, 6, 8, 10, 10, 10, 10, 8, 6, 4)
EXPECTED_DEVICE_COUNT = sum(EXPECTED_ROW_COUNTS)
EXPECTED_MARKER_SQUARES_PER_SIDE = 12


@dataclass(frozen=True)
class DesignGeometry:
    path: Path
    design_bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    radius: float
    boundary: np.ndarray
    marker_polygons: tuple[np.ndarray, ...]
    marker_centers: dict[str, tuple[float, float]]
    cells: tuple[dict[str, Any], ...]


_CACHE: dict[str, DesignGeometry] = {}


def resolve_design_path(
    path: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
) -> Path:
    # The default design path comes from config.json:gds_path.
    # An explicit path remains supported as a one-off override.
    module_dir = Path(__file__).resolve().parent

    if path is None:
        cfg = (
            Path(config_path).expanduser()
            if config_path is not None
            else module_dir / "config.json"
        )
        if not cfg.is_absolute():
            cfg = module_dir / cfg
        cfg = cfg.resolve()

        if not cfg.exists():
            raise FileNotFoundError(f"Configuration file not found: {cfg}")

        with cfg.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        configured = str(config.get("gds_path", "")).strip()
        if not configured:
            raise ValueError(
                f"{cfg.name} must define a non-empty 'gds_path'."
            )

        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = cfg.parent / candidate
    else:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = module_dir / candidate

    candidate = candidate.resolve()

    if candidate.suffix.lower() != ".gds":
        raise ValueError(
            f"Configured design must be a .gds file; received: {candidate}"
        )
    if not candidate.exists():
        raise FileNotFoundError(f"Configured GDS file not found: {candidate}")

    return candidate

def _flatten_gds(path: Path):
    library = gdstk.read_gds(str(path))
    top = library.top_level()
    if not top:
        raise RuntimeError(f"No top-level cell found in {path}")
    cell = top[0].copy(f"{top[0].name}__h2p_flat")
    cell.flatten()
    return cell


def _cluster_axis(values: list[float], tolerance: float) -> list[float]:
    groups: list[list[float]] = []
    for value in sorted(float(v) for v in values):
        if groups and abs(value - float(np.mean(groups[-1]))) <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [float(np.mean(group)) for group in groups]


def load_design_geometry(path: str | Path | None = None) -> DesignGeometry:
    gds_path = resolve_design_path(path)
    cache_key = str(gds_path)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    cell = _flatten_gds(gds_path)
    bbox = cell.bounding_box()
    if bbox is None:
        raise RuntimeError(f"GDS has no geometry: {gds_path}")
    design_bbox = (
        float(bbox[0][0]),
        float(bbox[0][1]),
        float(bbox[1][0]),
        float(bbox[1][1]),
    )

    by_layer: dict[tuple[int, int], list[np.ndarray]] = {}
    for polygon in cell.polygons:
        by_layer.setdefault((int(polygon.layer), int(polygon.datatype)), []).append(
            np.asarray(polygon.points, dtype=np.float64)
        )

    boundary_candidates = list(by_layer.get(WAFER_LAYER, ()))
    if not boundary_candidates:
        width = design_bbox[2] - design_bbox[0]
        height = design_bbox[3] - design_bbox[1]
        boundary_candidates = [
            points
            for polygons in by_layer.values()
            for points in polygons
            if np.ptp(points[:, 0]) > 0.75 * width
            and np.ptp(points[:, 1]) > 0.75 * height
        ]
    if not boundary_candidates:
        raise RuntimeError(f"Could not identify wafer boundary on layer {WAFER_LAYER}")

    boundary = max(
        boundary_candidates,
        key=lambda points: float(np.ptp(points[:, 0]) * np.ptp(points[:, 1])),
    )
    bx0, by0 = np.min(boundary, axis=0)
    bx1, by1 = np.max(boundary, axis=0)
    center = (float((bx0 + bx1) * 0.5), float((by0 + by1) * 0.5))
    radius = float((bx1 - bx0) * 0.5)

    cell_candidates: list[dict[str, Any]] = []
    for points in by_layer.get(DEVICE_LAYER, ()):  # layer 8/0 = device boundaries
        x0, y0 = np.min(points, axis=0)
        x1, y1 = np.max(points, axis=0)
        width = float(x1 - x0)
        height = float(y1 - y0)
        if not (3000.0 <= width <= 15000.0 and 3000.0 <= height <= 15000.0):
            continue
        cx = float((x0 + x1) * 0.5)
        cy = float((y0 + y1) * 0.5)
        if math.hypot(cx - center[0], cy - center[1]) > radius * 1.02:
            continue
        cell_candidates.append(
            {
                "bbox": (float(x0), float(y0), float(x1), float(y1)),
                "center": (cx, cy),
                "polygon": points.copy(),
            }
        )

    unique_cells: dict[tuple[float, float, float, float], dict[str, Any]] = {}
    for candidate in cell_candidates:
        key = tuple(round(value, 3) for value in candidate["bbox"])
        unique_cells[key] = candidate
    cells = list(unique_cells.values())

    y_centers = _cluster_axis(
        [cell_info["center"][1] for cell_info in cells],
        tolerance=max(50.0, radius * 0.015),
    )
    y_centers.sort(reverse=True)
    if len(y_centers) != len(EXPECTED_ROW_COUNTS):
        raise RuntimeError(
            f"Expected {len(EXPECTED_ROW_COUNTS)} device rows on layer {DEVICE_LAYER}, "
            f"found {len(y_centers)}"
        )

    for cell_info in cells:
        cell_info["row"] = (
            int(np.argmin([abs(cell_info["center"][1] - y) for y in y_centers])) + 1
        )
    for row in range(1, len(EXPECTED_ROW_COUNTS) + 1):
        row_cells = sorted(
            (cell_info for cell_info in cells if cell_info["row"] == row),
            key=lambda cell_info: cell_info["center"][0],
        )
        for col, cell_info in enumerate(row_cells, start=1):
            cell_info["col"] = col
            cell_info["col_global"] = col

    cells.sort(key=lambda cell_info: (cell_info["row"], cell_info["col"]))
    row_counts = tuple(
        sum(cell_info["row"] == row for cell_info in cells)
        for row in range(1, len(EXPECTED_ROW_COUNTS) + 1)
    )
    if len(cells) != EXPECTED_DEVICE_COUNT or row_counts != EXPECTED_ROW_COUNTS:
        raise RuntimeError(
            f"Layer {DEVICE_LAYER} device parse mismatch: "
            f"count={len(cells)}, rows={list(row_counts)}"
        )

    marker_polygons: list[np.ndarray] = []
    square_centers: dict[str, list[tuple[float, float]]] = {"left": [], "right": []}
    seen: set[tuple[float, ...]] = set()
    for points in by_layer.get(MARKER_LAYER, ()):  # layer 4/0 = compact fiducials
        x0, y0 = np.min(points, axis=0)
        x1, y1 = np.max(points, axis=0)
        width = float(x1 - x0)
        height = float(y1 - y0)
        area = float(
            0.5
            * abs(
                np.dot(points[:, 0], np.roll(points[:, 1], -1))
                - np.dot(points[:, 1], np.roll(points[:, 0], -1))
            )
        )
        key = (
            round(float(x0), 3),
            round(float(y0), 3),
            round(float(x1), 3),
            round(float(y1), 3),
            round(area, 2),
            float(len(points)),
        )
        if key in seen:
            continue
        seen.add(key)

        cx = float((x0 + x1) * 0.5)
        cy = float((y0 + y1) * 0.5)
        is_square = (
            150.0 <= width <= 260.0
            and 150.0 <= height <= 260.0
            and 43000.0 <= abs(cx) <= 47000.0
            and abs(cy) <= 1500.0
        )
        is_rail = (
            5000.0 <= width <= 10000.0
            and 100.0 <= height <= 300.0
            and 43000.0 <= abs(cx) <= 50000.0
            and abs(cy) <= 300.0
        )
        if is_square or is_rail:
            marker_polygons.append(points.copy())
        if is_square:
            side = "left" if cx < center[0] else "right"
            square_centers[side].append((cx, cy))

    marker_centers: dict[str, tuple[float, float]] = {}
    for side in ("left", "right"):
        points = square_centers[side]
        if len(points) != EXPECTED_MARKER_SQUARES_PER_SIDE:
            raise RuntimeError(
                f"Expected {EXPECTED_MARKER_SQUARES_PER_SIDE} {side} marker squares on "
                f"layer {MARKER_LAYER}, found {len(points)}"
            )
        marker_centers[side] = (
            float(np.mean([point[0] for point in points])),
            float(np.mean([point[1] for point in points])),
        )

    geometry = DesignGeometry(
        path=gds_path,
        design_bbox=design_bbox,
        center=center,
        radius=radius,
        boundary=boundary.copy(),
        marker_polygons=tuple(marker_polygons),
        marker_centers=marker_centers,
        cells=tuple(cells),
    )
    _CACHE[cache_key] = geometry
    return geometry


def clean_boundary_polygon(geometry: DesignGeometry, point_count: int = 360) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, max(64, int(point_count)), endpoint=False)
    return np.column_stack(
        (
            geometry.center[0] + geometry.radius * np.cos(angles),
            geometry.center[1] + geometry.radius * np.sin(angles),
        )
    ).astype(np.float64)


def marker_records(geometry: DesignGeometry) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    for points in geometry.marker_polygons:
        x0, y0 = np.min(points, axis=0)
        x1, y1 = np.max(points, axis=0)
        width = float(x1 - x0)
        height = float(y1 - y0)
        cx = float((x0 + x1) * 0.5)
        cy = float((y0 + y1) * 0.5)
        result["left" if cx < geometry.center[0] else "right"].append(
            {
                "type": "square" if width < 500.0 and height < 500.0 else "bar",
                "bbox": (float(x0), float(y0), float(x1), float(y1)),
                "center": (cx, cy),
                "polygon": points.tolist(),
            }
        )
    return result


def overlay_polygons(geometry: DesignGeometry) -> list[np.ndarray]:
    output: list[np.ndarray] = [clean_boundary_polygon(geometry)]
    for cell_info in geometry.cells:
        polygon = np.asarray(cell_info["polygon"], dtype=np.float64)
        output.append(polygon.copy())
    output.extend(np.asarray(points, dtype=np.float64).copy() for points in geometry.marker_polygons)
    return output


def cell_records(geometry: DesignGeometry) -> list[dict[str, Any]]:
    return [
        {
            "bbox": tuple(cell_info["bbox"]),
            "center": tuple(cell_info["center"]),
            "row": int(cell_info["row"]),
            "col": int(cell_info["col"]),
            "col_global": int(cell_info["col_global"]),
        }
        for cell_info in geometry.cells
    ]


# Small compatibility-shaped API used by the extraction pipeline. These functions
# are future-design-only; there is no alternate GDS mode behind them.
def parse_gds_wafer_boundary(path: str | Path | None = None, layer: int = 2, datatype: int = 0):
    if (int(layer), int(datatype)) != WAFER_LAYER:
        raise ValueError(f"Wafer boundary is fixed to layer/datatype {WAFER_LAYER}")
    geometry = load_design_geometry(path)
    return geometry.center[0], geometry.center[1], geometry.radius


def get_gds_overlay_polygons(path: str | Path | None = None, config: dict[str, Any] | None = None):
    del config
    return overlay_polygons(load_design_geometry(path))


def parse_alignment_markers(path: str | Path | None = None):
    return marker_records(load_design_geometry(path))


def get_gds_cells_list(polygons=None, gds_radius: float | None = None):
    del polygons, gds_radius
    return cell_records(load_design_geometry())
