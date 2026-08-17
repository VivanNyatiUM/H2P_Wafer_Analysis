#!/usr/bin/env python3
"""Production detector for the compact 3 x 4 future-design alignment fiducial.

It detects only the small inner fiducial: twelve equal squares arranged as
three columns and four rows around a horizontal rail.

The same fiducial center is used as the correspondence point in both spaces:

* GDS anchor: centroid of the 12 square centers, in micrometers.
* Image anchor: centroid of the fitted 3 x 4 lattice, in pixels.

The image detector allows missing squares and small rotations.  It uses square
connected components to hypothesize a similarity transform of the known grid,
then validates the horizontal rail to determine whether the marker is the left
or right wafer marker.

Dependencies:
    python -m pip install numpy opencv-python matplotlib gdstk pillow

PowerShell example:
    python .\test_future_inner_alignment_markers.py `
        --gds ./your_design.gds `
        --wafer-image .\future_marker_candidate.png `
        --output-dir .\future_inner_marker_test_output
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import gdstk
try:
    import matplotlib.pyplot as plt
except ImportError:  # Plot helpers are optional in production.
    plt = None
import numpy as np
from PIL import Image, ImageOps
try:
    from matplotlib.patches import Polygon as MplPolygon
except ImportError:  # Plot helpers are optional in production.
    MplPolygon = None


# The active design row centers, normalized by the 400 um pitch.
# The central gap is deliberately about twice the ordinary row pitch because
# the horizontal rail passes between the two middle rows.
TEMPLATE_POINTS = np.asarray(
    [(x, y) for y in (-1.97708, -0.97708, 0.97708, 1.97708) for x in (-1.0, 0.0, 1.0)],
    dtype=np.float64,
)


@dataclass
class PolyInfo:
    index: int
    layer: int
    datatype: int
    points: np.ndarray
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    width: float
    height: float
    area: float


@dataclass
class GDSInnerMarker:
    side: str
    layer: int
    datatype: int
    anchor_um: tuple[float, float]
    square_size_um: float
    x_pitch_um: float
    y_pitch_um: float
    rail_bbox_um: tuple[float, float, float, float]
    marker_bbox_um: tuple[float, float, float, float]
    square_count: int
    square_indices: list[int]
    rail_index: int
    score: float


@dataclass
class SquareCandidate:
    label: int
    center: tuple[float, float]
    bbox: tuple[int, int, int, int]
    size: float
    area: int
    fill_ratio: float


@dataclass
class ImageLoadInfo:
    original_size_px: tuple[int, int]
    working_size_px: tuple[int, int]
    scale_x: float
    scale_y: float
    decoder: str


@dataclass
class ImageInnerMarker:
    side: str
    anchor_px: tuple[float, float]
    angle_deg: float
    pitch_px: float
    square_size_px: float
    matched_square_count: int
    mean_error_px: float
    rail_left_pixels: int
    rail_right_pixels: int
    score: float
    matched_candidate_labels: list[int]
    predicted_square_centers_px: list[tuple[float, float]]
    matched_square_centers_px: list[tuple[float, float]]
    anchor_original_px: tuple[float, float] | None = None
    pitch_original_px: float | None = None
    error_pitch_ratio: float = 0.0
    square_pitch_ratio: float = 0.0
    symmetry_error_px: float | None = None
    detection_stage: str = "initial"
    template_score: float | None = None


def _shoelace_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _bbox_union(boxes: Iterable[Sequence[float]]) -> tuple[float, float, float, float]:
    boxes = list(boxes)
    return (
        float(min(b[0] for b in boxes)),
        float(min(b[1] for b in boxes)),
        float(max(b[2] for b in boxes)),
        float(max(b[3] for b in boxes)),
    )


def _cluster_1d(values: Sequence[float], tolerance: float) -> list[list[float]]:
    values = sorted(float(v) for v in values)
    if not values:
        return []
    groups: list[list[float]] = [[values[0]]]
    for value in values[1:]:
        if abs(value - float(np.mean(groups[-1]))) <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return groups


def load_flat_gds_polygons(gds_path: Path) -> tuple[list[PolyInfo], tuple[float, float, float, float]]:
    library = gdstk.read_gds(str(gds_path))
    top_cells = library.top_level()
    if not top_cells:
        raise RuntimeError(f"No top-level cell found in {gds_path}")

    flat = top_cells[0].copy(f"{top_cells[0].name}__future_inner_marker_test")
    flat.flatten()
    bbox = flat.bounding_box()
    if bbox is None:
        raise RuntimeError(f"Top-level cell in {gds_path} has no geometry")
    design_bbox = (float(bbox[0][0]), float(bbox[0][1]), float(bbox[1][0]), float(bbox[1][1]))

    polygons: list[PolyInfo] = []
    seen: set[tuple] = set()
    for index, polygon in enumerate(flat.polygons):
        points = np.asarray(polygon.points, dtype=np.float64)
        if len(points) < 3:
            continue
        x0, y0 = np.min(points, axis=0)
        x1, y1 = np.max(points, axis=0)
        width = float(x1 - x0)
        height = float(y1 - y0)
        area = _shoelace_area(points)
        key = (
            int(polygon.layer),
            int(polygon.datatype),
            round(float(x0), 3),
            round(float(y0), 3),
            round(float(x1), 3),
            round(float(y1), 3),
            round(area, 2),
            len(points),
        )
        if key in seen:
            continue
        seen.add(key)
        polygons.append(
            PolyInfo(
                index=index,
                layer=int(polygon.layer),
                datatype=int(polygon.datatype),
                points=points,
                bbox=(float(x0), float(y0), float(x1), float(y1)),
                center=(float((x0 + x1) * 0.5), float((y0 + y1) * 0.5)),
                width=width,
                height=height,
                area=area,
            )
        )
    return polygons, design_bbox


def _find_square_lattice_groups(polygons: list[PolyInfo]) -> list[tuple[list[PolyInfo], float, float, float]]:
    """Return candidate 12-square groups and their size/x-pitch/y-pitch."""
    square_like = [
        p
        for p in polygons
        if p.width > 0
        and p.height > 0
        and 0.85 <= p.width / p.height <= 1.18
        and len(p.points) <= 8
    ]
    groups: list[tuple[list[PolyInfo], float, float, float]] = []
    seen_sets: set[tuple[int, ...]] = set()

    for seed in square_like:
        size = 0.5 * (seed.width + seed.height)
        peers = [
            p
            for p in square_like
            if p.layer == seed.layer
            and p.datatype == seed.datatype
            and 0.85 * size <= 0.5 * (p.width + p.height) <= 1.18 * size
            and abs(p.center[0] - seed.center[0]) <= 6.0 * size
            and abs(p.center[1] - seed.center[1]) <= 9.0 * size
        ]
        if len(peers) < 12:
            continue

        # The compact marker is isolated from the larger outer-dot columns,
        # so the local same-size peer set should be exactly the 12-square grid.
        local = peers
        if len(local) != 12:
            continue
        x_groups = _cluster_1d([p.center[0] for p in local], tolerance=0.30 * size)
        y_groups = _cluster_1d([p.center[1] for p in local], tolerance=0.30 * size)
        if len(x_groups) != 3 or len(y_groups) != 4:
            continue
        if sorted(len(g) for g in x_groups) != [4, 4, 4]:
            continue
        if sorted(len(g) for g in y_groups) != [3, 3, 3, 3]:
            continue

        xs = np.asarray([np.mean(g) for g in x_groups], dtype=np.float64)
        ys = np.asarray([np.mean(g) for g in y_groups], dtype=np.float64)
        dx = np.diff(xs)
        dy = np.diff(ys)
        x_pitch = float(np.mean(dx))
        ordinary_y_pitch = float(0.5 * (dy[0] + dy[2]))
        if not (1.5 * size <= x_pitch <= 2.5 * size):
            continue
        if not (1.5 * size <= ordinary_y_pitch <= 2.5 * size):
            continue
        if not (1.65 <= dy[1] / max(ordinary_y_pitch, 1e-9) <= 2.25):
            continue
        if np.std(dx) > 0.15 * x_pitch:
            continue
        if abs(dy[0] - dy[2]) > 0.20 * ordinary_y_pitch:
            continue

        key = tuple(sorted(p.index for p in local))
        if key in seen_sets:
            continue
        seen_sets.add(key)
        groups.append((local, size, x_pitch, ordinary_y_pitch))
    return groups


def detect_gds_inner_markers(
    gds_path: Path,
) -> tuple[list[GDSInnerMarker], list[PolyInfo], tuple[float, float, float, float]]:
    polygons, design_bbox = load_flat_gds_polygons(gds_path)
    lattice_groups = _find_square_lattice_groups(polygons)
    detections: list[GDSInnerMarker] = []

    for squares, size, x_pitch, y_pitch in lattice_groups:
        layer = squares[0].layer
        datatype = squares[0].datatype
        anchor = np.mean(np.asarray([p.center for p in squares], dtype=np.float64), axis=0)
        square_bbox = _bbox_union(p.bbox for p in squares)

        rail_candidates = [
            p
            for p in polygons
            if p.layer == layer
            and p.datatype == datatype
            and p.width >= 10.0 * size
            and 0.55 * size <= p.height <= 1.55 * size
            and abs(p.center[1] - anchor[1]) <= 0.75 * size
            and p.bbox[0] <= square_bbox[2] + 1.5 * size
            and p.bbox[2] >= square_bbox[0] - 1.5 * size
        ]
        if not rail_candidates:
            continue
        rail = max(rail_candidates, key=lambda p: p.width)

        left_extent = anchor[0] - rail.bbox[0]
        right_extent = rail.bbox[2] - anchor[0]
        if max(left_extent, right_extent) < 8.0 * size:
            continue
        side = "left" if left_extent > right_extent else "right"

        # The rail should terminate near the square grid on the accessory side.
        near_grid_endpoint = (
            abs(rail.bbox[2] - square_bbox[2]) <= 2.0 * size
            if side == "left"
            else abs(rail.bbox[0] - square_bbox[0]) <= 2.0 * size
        )
        if not near_grid_endpoint:
            continue

        score = 12.0
        score += min(rail.width / max(size, 1e-9), 40.0) / 20.0
        score += 1.0 if near_grid_endpoint else 0.0
        detections.append(
            GDSInnerMarker(
                side=side,
                layer=layer,
                datatype=datatype,
                anchor_um=(float(anchor[0]), float(anchor[1])),
                square_size_um=float(size),
                x_pitch_um=float(x_pitch),
                y_pitch_um=float(y_pitch),
                rail_bbox_um=rail.bbox,
                marker_bbox_um=_bbox_union([square_bbox, rail.bbox]),
                square_count=len(squares),
                square_indices=sorted(p.index for p in squares),
                rail_index=rail.index,
                score=float(score),
            )
        )

    result: list[GDSInnerMarker] = []
    for side in ("left", "right"):
        candidates = [d for d in detections if d.side == side]
        if candidates:
            result.append(max(candidates, key=lambda d: d.score))
    return result, polygons, design_bbox


def plot_gds_inner_markers(
    polygons: list[PolyInfo], markers: list[GDSInnerMarker], output_path: Path
) -> None:
    fig, axes = plt.subplots(1, max(len(markers), 1), figsize=(12, 5), squeeze=False)
    axes_list = list(axes[0])
    if not markers:
        axes_list[0].text(0.5, 0.5, "No inner markers found", ha="center", va="center")
        axes_list[0].set_axis_off()
    for ax, marker in zip(axes_list, markers):
        selected = set(marker.square_indices + [marker.rail_index])
        for p in polygons:
            if p.index not in selected:
                continue
            ax.add_patch(MplPolygon(p.points, closed=True, fill=False, linewidth=1.2))
        ax.scatter([marker.anchor_um[0]], [marker.anchor_um[1]], marker="x", s=90)
        x0, y0, x1, y1 = marker.marker_bbox_um
        pad_x = max(600.0, 0.08 * (x1 - x0))
        pad_y = max(600.0, 0.35 * (y1 - y0))
        ax.set_xlim(x0 - pad_x, x1 + pad_x)
        ax.set_ylim(y0 - pad_y, y1 + pad_y)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"{marker.side.upper()} inner fiducial  L{marker.layer}/{marker.datatype}\n"
            f"anchor=({marker.anchor_um[0]:.2f}, {marker.anchor_um[1]:.2f}) µm"
        )
        ax.set_xlabel("GDS x (µm)")
        ax.set_ylabel("GDS y (µm)")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Compact 3 × 4 alignment fiducials detected from GDS geometry")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _robust_unit_scale(response: np.ndarray) -> np.ndarray:
    """Map a feature response to 0..255 without letting a few hot pixels dominate."""
    response = np.asarray(response, dtype=np.float32)
    finite = response[np.isfinite(response)]
    if finite.size == 0:
        return np.zeros(response.shape, dtype=np.uint8)
    low = float(np.percentile(finite, 70.0))
    high = float(np.percentile(finite, 99.7))
    if high <= low + 1e-6:
        low = float(np.min(finite))
        high = float(np.max(finite))
    scaled = np.clip((response - low) / max(high - low, 1e-6), 0.0, 1.0)
    return np.asarray(np.rint(255.0 * scaled), dtype=np.uint8)


def _bright_square_response(image: np.ndarray, expected_size_px: float) -> np.ndarray:
    """Return a local response that emphasizes small white squares on metal.

    The right marker in real stitched wafer images is often a set of white
    squares on a bright gold pad. A global threshold sees the entire pad as one
    component, so this uses both color whiteness and a morphological top-hat to
    remove the slowly varying pad background.
    """
    expected_size_px = max(float(expected_size_px), 3.0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)

    # White features are bright and less saturated than the surrounding gold.
    whiteness = 0.62 * gray + 0.28 * value + 0.10 * (255.0 - saturation)
    whiteness_u8 = np.asarray(np.clip(whiteness, 0, 255), dtype=np.uint8)

    kernel_size = max(7, int(round(2.6 * expected_size_px)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    top_hat = cv2.morphologyEx(whiteness_u8, cv2.MORPH_TOPHAT, kernel).astype(np.float32)

    sigma = max(2.0, 1.35 * expected_size_px)
    background = cv2.GaussianBlur(whiteness, (0, 0), sigmaX=sigma, sigmaY=sigma)
    local = np.maximum(whiteness - background, 0.0)

    top_hat_scaled = _robust_unit_scale(top_hat).astype(np.float32) / 255.0
    local_scaled = _robust_unit_scale(local).astype(np.float32) / 255.0
    response = np.maximum(top_hat_scaled, local_scaled)
    return cv2.GaussianBlur(response, (3, 3), 0)


def _build_foreground_masks(
    image: np.ndarray,
    expected_size_px: float | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Build plausible masks, including local white-on-metal feature masks."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    masks: list[tuple[str, np.ndarray]] = []

    # Global dark and bright masks.
    _t_dark, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _t_bright, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(("otsu-dark", dark))
    masks.append(("otsu-bright", bright))

    # Local contrast handles non-uniform microscope illumination.
    block = max(15, (min(gray.shape) // 12) | 1)
    local_dark = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 4
    )
    local_bright = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 4
    )
    masks.append(("adaptive-dark", local_dark))
    masks.append(("adaptive-bright", local_bright))

    if expected_size_px is not None:
        fine_block = max(15, int(round(5.0 * expected_size_px)) | 1)
        fine_bright = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            fine_block,
            2,
        )
        masks.append(("adaptive-bright-fine", fine_bright))

        response = _bright_square_response(image, expected_size_px)
        response_u8 = np.asarray(np.rint(255.0 * response), dtype=np.uint8)
        otsu_value, response_mask = cv2.threshold(
            response_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # Otsu can be too permissive on a mostly smooth metal pad. Enforce a
        # modest floor while retaining partially blurred squares.
        floor_value = max(int(otsu_value), 52)
        response_mask = np.where(response_u8 >= floor_value, 255, 0).astype(np.uint8)
        masks.append(("white-square-tophat", response_mask))

    # Color separation remains useful for rendered/debug images.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    if float(np.percentile(sat, 99.0)) >= 45.0:
        _t, sat_mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(("saturation", sat_mask))
        masks.append(("low-saturation", cv2.bitwise_not(sat_mask)))

    cleaned: list[tuple[str, np.ndarray]] = []
    for name, mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        # Do not close the top-hat mask aggressively: the square rows are close
        # to the horizontal rail and can otherwise merge into one component.
        if name != "white-square-tophat":
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        cleaned.append((name, mask))
    return cleaned

def _extract_square_candidates(
    mask: np.ndarray,
    expected_size_px: float | None = None,
    max_candidates: int = 280,
) -> list[SquareCandidate]:
    h_img, w_img = mask.shape
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: list[SquareCandidate] = []
    for label in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[label]]
        if w < 3 or h < 3:
            continue
        if w > 0.10 * w_img or h > 0.16 * h_img:
            continue
        aspect = w / max(h, 1)
        if not (0.62 <= aspect <= 1.60):
            continue
        fill = area / max(w * h, 1)
        # Filled squares and hollow square rings are both accepted.
        if not (0.14 <= fill <= 1.0):
            continue
        size = math.sqrt(float(w * h))
        if expected_size_px is not None:
            if not (0.35 * expected_size_px <= size <= 3.0 * expected_size_px):
                continue
        candidates.append(
            SquareCandidate(
                label=label,
                center=(float(centroids[label][0]), float(centroids[label][1])),
                bbox=(x, y, x + w, y + h),
                size=size,
                area=area,
                fill_ratio=float(fill),
            )
        )
    if expected_size_px is not None and len(candidates) > max_candidates:
        # Keep components whose size is most consistent with the GDS-derived
        # expectation.  This prevents the pairwise lattice fit from exploding
        # on a full wafer containing thousands of unrelated square features.
        candidates.sort(
            key=lambda c: abs(math.log(max(c.size, 1e-9) / max(expected_size_px, 1e-9)))
        )
        candidates = candidates[:max_candidates]
    return candidates


def _greedy_template_matches(
    predicted: np.ndarray,
    candidate_points: np.ndarray,
    tolerance: float,
) -> tuple[list[tuple[int, int, float]], float]:
    distances = np.linalg.norm(predicted[:, None, :] - candidate_points[None, :, :], axis=2)
    pairs: list[tuple[float, int, int]] = []
    for ti in range(distances.shape[0]):
        for ci in range(distances.shape[1]):
            if distances[ti, ci] <= tolerance:
                pairs.append((float(distances[ti, ci]), ti, ci))
    pairs.sort()
    used_t: set[int] = set()
    used_c: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for distance, ti, ci in pairs:
        if ti in used_t or ci in used_c:
            continue
        used_t.add(ti)
        used_c.add(ci)
        matches.append((ti, ci, distance))
    mean_error = float(np.mean([m[2] for m in matches])) if matches else float("inf")
    return matches, mean_error


def _fit_lattice_hypotheses(candidates: list[SquareCandidate]) -> list[dict]:
    if len(candidates) < 8:
        return []
    points = np.asarray([c.center for c in candidates], dtype=np.float64)
    sizes = np.asarray([c.size for c in candidates], dtype=np.float64)
    hypotheses: list[dict] = []

    # Ordered candidate pairs are treated as adjacent columns.  That one vector
    # fixes scale and rotation; every possible template source position fixes
    # translation.  Both perpendicular signs cover image-coordinate handedness.
    for i in range(len(candidates)):
        for j in range(len(candidates)):
            if i == j:
                continue
            vector = points[j] - points[i]
            pitch = float(np.linalg.norm(vector))
            local_size = 0.5 * (sizes[i] + sizes[j])
            if not (1.15 * local_size <= pitch <= 4.0 * local_size):
                continue
            ex = vector / pitch
            for perp_sign in (-1.0, 1.0):
                ey = perp_sign * np.asarray([-ex[1], ex[0]], dtype=np.float64)
                transform = np.column_stack([ex, ey]) * pitch
                for row_index in range(4):
                    for col_index in (0, 1):
                        template_index = row_index * 3 + col_index
                        template_source = TEMPLATE_POINTS[template_index]
                        translation = points[i] - transform @ template_source
                        predicted = TEMPLATE_POINTS @ transform.T + translation
                        matches, mean_error = _greedy_template_matches(
                            predicted, points, tolerance=max(2.0, 0.42 * pitch)
                        )
                        if len(matches) < 8:
                            continue
                        matched_candidate_indices = [m[1] for m in matches]
                        matched_sizes = sizes[matched_candidate_indices]
                        size_cv = float(np.std(matched_sizes) / max(np.mean(matched_sizes), 1e-9))
                        if size_cv > 0.38:
                            continue
                        # Reward occupancy, tight fit, and consistent square size.
                        score = 10.0 * len(matches) - 5.0 * mean_error / max(pitch, 1e-9) - 8.0 * size_cv
                        hypotheses.append(
                            {
                                "score": float(score),
                                "pitch": pitch,
                                "transform": transform,
                                "translation": translation,
                                "predicted": predicted,
                                "matches": matches,
                                "mean_error": mean_error,
                                "size_cv": size_cv,
                            }
                        )
    hypotheses.sort(key=lambda h: h["score"], reverse=True)
    return hypotheses


def _classify_side_from_rail(
    mask: np.ndarray,
    anchor: np.ndarray,
    transform: np.ndarray,
    pitch: float,
) -> tuple[str, int, int]:
    # Examine only a local strip around the hypothesized marker.  Calling
    # np.nonzero on an entire stitched wafer for every hypothesis is both slow
    # and extremely memory-hungry.
    radius_x = int(math.ceil(15.5 * pitch))
    radius_y = int(math.ceil(3.0 * pitch))
    x0 = max(0, int(math.floor(anchor[0])) - radius_x)
    x1 = min(mask.shape[1], int(math.ceil(anchor[0])) + radius_x + 1)
    y0 = max(0, int(math.floor(anchor[1])) - radius_y)
    y1 = min(mask.shape[0], int(math.ceil(anchor[1])) + radius_y + 1)
    if x1 <= x0 or y1 <= y0:
        return "unknown", 0, 0

    ys, xs = np.nonzero(mask[y0:y1, x0:x1])
    if len(xs) == 0:
        return "unknown", 0, 0
    pixels = np.column_stack([xs + x0, ys + y0]).astype(np.float64)
    # The transform columns are local x/y basis vectors times pitch.
    basis = transform / max(pitch, 1e-9)
    local = (pixels - anchor) @ basis
    central = local[np.abs(local[:, 1]) <= 0.34 * pitch]
    left = int(np.count_nonzero((central[:, 0] <= -1.25 * pitch) & (central[:, 0] >= -14.0 * pitch)))
    right = int(np.count_nonzero((central[:, 0] >= 1.25 * pitch) & (central[:, 0] <= 14.0 * pitch)))
    if max(left, right) < 8:
        return "unknown", left, right
    # Left wafer marker has the square lattice at the right end of its rail.
    return ("left" if left > right else "right"), left, right


def _load_image_for_detection(
    image_path: Path,
    max_megapixels: float,
    max_dimension: int,
) -> tuple[np.ndarray, ImageLoadInfo]:
    """Load a huge microscope JPEG as a bounded-resolution working image.

    Pillow's JPEG draft mode asks libjpeg to decode at 1/2, 1/4, or 1/8
    resolution before allocating the RGB array.  A final thumbnail step enforces
    the exact working-size cap.  This avoids OpenCV's 1-gigapixel imread guard
    and, more importantly, avoids allocating several full-wafer masks.
    """
    max_pixels = max(1, int(max_megapixels * 1_000_000))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(image_path) as opened:
            source_format = (opened.format or "").upper()
            original_w, original_h = opened.size
            if original_w <= 0 or original_h <= 0:
                raise RuntimeError(f"Invalid image dimensions for {image_path}: {opened.size}")

            scale_pixels = math.sqrt(max_pixels / float(original_w * original_h))
            scale_dimension = max_dimension / float(max(original_w, original_h))
            target_scale = min(1.0, scale_pixels, scale_dimension)
            target_size = (
                max(1, int(round(original_w * target_scale))),
                max(1, int(round(original_h * target_scale))),
            )

            decoder = "pillow-full"
            if target_scale < 1.0 and source_format in {"JPEG", "JPG", "MPO"}:
                # draft() must happen before conversion/exif_transpose so libjpeg
                # can perform reduced-resolution decoding rather than allocating
                # the original gigantic raster first.
                opened.draft("RGB", target_size)
                decoder = "pillow-jpeg-draft"
            rgb_image = ImageOps.exif_transpose(opened).convert("RGB")
            if (
                rgb_image.width * rgb_image.height > max_pixels
                or max(rgb_image.size) > max_dimension
            ):
                rgb_image.thumbnail(target_size, Image.Resampling.LANCZOS)
                decoder += "+thumbnail"
            rgb = np.asarray(rgb_image, dtype=np.uint8)

    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    working_h, working_w = image.shape[:2]
    return image, ImageLoadInfo(
        original_size_px=(int(original_w), int(original_h)),
        working_size_px=(int(working_w), int(working_h)),
        scale_x=float(working_w / original_w),
        scale_y=float(working_h / original_h),
        decoder=decoder,
    )


def _angle_difference_mod_180(angle_a: float, angle_b: float) -> float:
    """Smallest angular difference when a lattice axis is directionless."""
    return float(abs((angle_a - angle_b + 90.0) % 180.0 - 90.0))


def _estimate_wafer_center(image: np.ndarray) -> tuple[np.ndarray, float, str]:
    """Estimate the wafer center from the large bright/dark circular region.

    The estimate is intentionally conservative.  If segmentation does not find
    a plausible large contour, the image center is returned.  This is still a
    useful symmetry prior because stitched wafer exports are normally centered.
    """
    h_img, w_img = image.shape[:2]
    fallback = np.asarray([(w_img - 1) * 0.5, (h_img - 1) * 0.5], dtype=np.float64)
    fallback_radius = 0.48 * min(w_img, h_img)

    scale = min(1.0, 1400.0 / max(w_img, h_img))
    if scale < 1.0:
        small = cv2.resize(
            image,
            (max(1, int(round(w_img * scale))), max(1, int(round(h_img * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = image
    h_small, w_small = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)

    _threshold, bright = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    masks = [("bright", bright), ("dark", cv2.bitwise_not(bright))]
    kernel_size = max(5, int(round(0.012 * min(w_small, h_small))) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    best: tuple[float, np.ndarray, float, str] | None = None
    image_area = float(w_small * h_small)
    image_center = np.asarray([(w_small - 1) * 0.5, (h_small - 1) * 0.5])
    for polarity, mask in masks:
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _hierarchy = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            area = float(cv2.contourArea(contour))
            area_fraction = area / max(image_area, 1.0)
            if not (0.18 <= area_fraction <= 0.96):
                continue
            x, y, width, height = cv2.boundingRect(contour)
            if width < 0.45 * w_small or height < 0.45 * h_small:
                continue
            moments = cv2.moments(contour)
            if abs(moments["m00"]) < 1e-9:
                continue
            center = np.asarray(
                [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
                dtype=np.float64,
            )
            center_offset = np.linalg.norm(center - image_center) / max(min(w_small, h_small), 1)
            circularity = 0.0
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter > 1e-9:
                circularity = min(1.0, 4.0 * math.pi * area / (perimeter * perimeter))
            score = 2.5 * area_fraction + 0.8 * circularity - 1.8 * center_offset
            radius = math.sqrt(area / math.pi)
            if best is None or score > best[0]:
                best = (score, center, radius, polarity)

    if best is None:
        return fallback, fallback_radius, "image-center-fallback"
    _score, center_small, radius_small, polarity = best
    center = center_small / max(scale, 1e-12)
    radius = radius_small / max(scale, 1e-12)
    return center, float(radius), f"wafer-contour-{polarity}"


def _candidate_metrics(result: dict) -> dict[str, float]:
    pitch = float(result["pitch"])
    matched_candidates: list[SquareCandidate] = result["matched_candidates"]
    if matched_candidates:
        square_size = float(np.median([candidate.size for candidate in matched_candidates]))
    else:
        square_size = float(result.get("square_size", 0.0))
    ex = result["transform"][:, 0] / max(pitch, 1e-9)
    angle = float(math.degrees(math.atan2(ex[1], ex[0])))
    left_pixels = int(result["left_pixels"])
    right_pixels = int(result["right_pixels"])
    return {
        "pitch": pitch,
        "square_size": square_size,
        "error_pitch_ratio": float(result["mean_error"] / max(pitch, 1e-9)),
        "square_pitch_ratio": float(square_size / max(pitch, 1e-9)),
        "rail_asymmetry": float(
            abs(left_pixels - right_pixels) / max(left_pixels + right_pixels, 1)
        ),
        "angle_deg": angle,
    }


def _validate_candidate(
    result: dict,
    target_side: str,
    expected_pitch_px: float,
    reference_marker: dict | None = None,
) -> tuple[bool, list[str], dict[str, float]]:
    """Apply hard geometric checks before a hypothesis may become a detection."""
    metrics = _candidate_metrics(result)
    reasons: list[str] = []
    matched_count = len(result["matches"])
    is_template_fallback = str(result.get("detection_stage", "")).startswith(
        "symmetry-template"
    )
    if matched_count < 10:
        reasons.append(f"matched {matched_count}/12 < 10/12")
    max_error_ratio = 0.14 if is_template_fallback else 0.10
    if metrics["error_pitch_ratio"] > max_error_ratio:
        reasons.append(
            f"fit error/pitch {metrics['error_pitch_ratio']:.3f} > {max_error_ratio:.2f}"
        )
    if not (0.35 <= metrics["square_pitch_ratio"] <= 0.75):
        reasons.append(
            f"square/pitch {metrics['square_pitch_ratio']:.3f} outside 0.35..0.75"
        )
    if float(result["size_cv"]) > 0.25:
        reasons.append(f"square-size CV {result['size_cv']:.3f} > 0.25")

    # Connected-component searches can validate the rail directly. The guided
    # fallback already includes the correctly directed rail in its correlation
    # template, so repeating a brittle binary rail count would reject the same
    # white-on-gold marker that the fallback exists to recover.
    if not is_template_fallback:
        if result["rail_side"] != target_side:
            reasons.append(
                f"rail classified as {result['rail_side']!r}, expected {target_side!r}"
            )
        if metrics["rail_asymmetry"] < 0.04:
            reasons.append(f"rail asymmetry {metrics['rail_asymmetry']:.3f} < 0.04")
    else:
        template_score = float(result.get("template_score", 0.0))
        if template_score < 0.20:
            reasons.append(f"template score {template_score:.3f} < 0.20")

    if not (0.65 * expected_pitch_px <= metrics["pitch"] <= 1.45 * expected_pitch_px):
        reasons.append(
            f"pitch {metrics['pitch']:.2f}px inconsistent with image/GDS prior "
            f"{expected_pitch_px:.2f}px"
        )

    if reference_marker is not None:
        reference_metrics = _candidate_metrics(reference_marker)
        angle_difference = _angle_difference_mod_180(
            metrics["angle_deg"], reference_metrics["angle_deg"]
        )
        pitch_difference = abs(metrics["pitch"] / reference_metrics["pitch"] - 1.0)
        size_difference = abs(
            metrics["square_size"] / max(reference_metrics["square_size"], 1e-9) - 1.0
        )
        if angle_difference > 10.0:
            reasons.append(f"angle differs from first marker by {angle_difference:.1f}°")
        if pitch_difference > 0.20:
            reasons.append(f"pitch differs from first marker by {100*pitch_difference:.1f}%")
        if size_difference > 0.30:
            reasons.append(
                f"square size differs from first marker by {100*size_difference:.1f}%"
            )
    return not reasons, reasons, metrics

def _search_marker_window(
    image: np.ndarray,
    target_side: str,
    center_px: np.ndarray,
    half_width: int,
    half_height: int,
    expected_square_px: float,
    expected_pitch_px: float,
    stage: str,
    reference_marker: dict | None = None,
    max_results: int = 80,
) -> tuple[list[dict], np.ndarray, list[SquareCandidate], list[str]]:
    """Search one bounded window and return validated and rejected hypotheses."""
    h_img, w_img = image.shape[:2]
    cx, cy = [int(round(value)) for value in center_px]
    x0 = max(0, cx - half_width)
    x1 = min(w_img, cx + half_width + 1)
    y0 = max(0, cy - half_height)
    y1 = min(h_img, cy + half_height + 1)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return [], np.zeros((0, 0), dtype=np.uint8), [], []

    global_offset = np.asarray([x0, y0], dtype=np.float64)
    results: list[dict] = []
    best_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    best_candidates: list[SquareCandidate] = []
    selected_modes: list[str] = []
    best_raw_score = -float("inf")

    for mask_name, mask in _build_foreground_masks(crop, expected_size_px=expected_square_px):
        local_candidates = _extract_square_candidates(
            mask,
            expected_size_px=expected_square_px,
            max_candidates=220 if reference_marker is not None else 280,
        )
        hypotheses = _fit_lattice_hypotheses(local_candidates)
        for hypothesis in hypotheses[:max_results]:
            pitch = float(hypothesis["pitch"])
            if reference_marker is not None:
                reference_metrics = _candidate_metrics(reference_marker)
                if not (0.78 * reference_metrics["pitch"] <= pitch <= 1.22 * reference_metrics["pitch"]):
                    continue
                ex = hypothesis["transform"][:, 0] / max(pitch, 1e-9)
                angle = math.degrees(math.atan2(ex[1], ex[0]))
                if _angle_difference_mod_180(angle, reference_metrics["angle_deg"]) > 12.0:
                    continue

            matched_indices = [match[1] for match in hypothesis["matches"]]
            global_candidates = [
                SquareCandidate(
                    label=candidate.label,
                    center=(candidate.center[0] + x0, candidate.center[1] + y0),
                    bbox=(
                        candidate.bbox[0] + x0,
                        candidate.bbox[1] + y0,
                        candidate.bbox[2] + x0,
                        candidate.bbox[3] + y0,
                    ),
                    size=candidate.size,
                    area=candidate.area,
                    fill_ratio=candidate.fill_ratio,
                )
                for candidate in local_candidates
            ]
            local_anchor = np.mean(hypothesis["predicted"], axis=0)
            rail_side, left_pixels, right_pixels = _classify_side_from_rail(
                mask, local_anchor, hypothesis["transform"], pitch
            )
            rail_asymmetry = abs(left_pixels - right_pixels) / max(
                left_pixels + right_pixels, 1
            )
            raw_score = float(hypothesis["score"] + 8.0 * rail_asymmetry)
            if rail_side == target_side:
                raw_score += 10.0
            elif rail_side == "unknown":
                raw_score -= 8.0
            else:
                raw_score -= 18.0

            result = {
                **hypothesis,
                "predicted": hypothesis["predicted"] + global_offset,
                "anchor": local_anchor + global_offset,
                "target_side": target_side,
                "rail_side": rail_side,
                "left_pixels": left_pixels,
                "right_pixels": right_pixels,
                "matched_candidates": [global_candidates[index] for index in matched_indices],
                "crop_bbox": (x0, y0, x1, y1),
                "mask_name": mask_name,
                "mask": mask,
                "global_candidates": global_candidates,
                "raw_score": raw_score,
                "detection_stage": stage,
            }
            valid, rejection_reasons, metrics = _validate_candidate(
                result,
                target_side=target_side,
                expected_pitch_px=expected_pitch_px,
                reference_marker=reference_marker,
            )
            result["valid"] = valid
            result["rejection_reasons"] = rejection_reasons
            result["metrics"] = metrics
            # Strongly prefer candidates that pass every hard validation check.
            result["selection_score"] = raw_score + (50.0 if valid else -25.0 * len(rejection_reasons))
            results.append(result)

            if raw_score > best_raw_score:
                best_raw_score = raw_score
                best_mask = mask
                best_candidates = global_candidates
                selected_modes = [f"{target_side}:{mask_name}:{stage}"]

    results.sort(key=lambda item: item["selection_score"], reverse=True)
    return results, best_mask, best_candidates, selected_modes



def _draw_rotated_square(
    canvas: np.ndarray,
    center: np.ndarray,
    size: float,
    angle_deg: float,
    value: float,
) -> None:
    rectangle = ((float(center[0]), float(center[1])), (float(size), float(size)), float(angle_deg))
    points = cv2.boxPoints(rectangle)
    cv2.fillConvexPoly(canvas, np.rint(points).astype(np.int32), float(value))


def _make_guided_marker_template(
    target_side: str,
    pitch: float,
    square_size: float,
    angle_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a signed square+rail template and return its anchor offset."""
    theta = math.radians(angle_deg)
    ex = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float64)
    ey = np.asarray([-math.sin(theta), math.cos(theta)], dtype=np.float64)
    transform = np.column_stack([ex, ey]) * pitch
    relative_points = TEMPLATE_POINTS @ transform.T

    rail_start = -9.0 * pitch if target_side == "left" else -1.25 * pitch
    rail_end = 1.25 * pitch if target_side == "left" else 9.0 * pitch
    rail_points = np.asarray([rail_start * ex, rail_end * ex])
    all_points = np.vstack([relative_points, rail_points])
    margin = max(1.5 * square_size, 0.75 * pitch)
    minimum = np.floor(np.min(all_points, axis=0) - margin)
    maximum = np.ceil(np.max(all_points, axis=0) + margin)
    shape = np.maximum(np.rint(maximum - minimum + 1).astype(int), 3)
    canvas = np.zeros((int(shape[1]), int(shape[0])), dtype=np.float32)
    anchor_offset = -minimum

    # Negative halos penalize broad bright pads while the positive centers match
    # the individual white squares.
    for point in relative_points:
        center = point + anchor_offset
        _draw_rotated_square(canvas, center, 1.55 * square_size, angle_deg, -0.30)
        _draw_rotated_square(canvas, center, 0.78 * square_size, angle_deg, 1.00)

    rail_width = max(1, int(round(0.18 * square_size)))
    p0 = np.rint(rail_points[0] + anchor_offset).astype(int)
    p1 = np.rint(rail_points[1] + anchor_offset).astype(int)
    cv2.line(canvas, tuple(p0), tuple(p1), 0.28, rail_width, cv2.LINE_AA)
    canvas -= float(np.mean(canvas))
    norm = float(np.linalg.norm(canvas))
    if norm > 1e-9:
        canvas /= norm
    return canvas, anchor_offset, transform


def _refine_template_points(
    response: np.ndarray,
    predicted_local: np.ndarray,
    pitch: float,
    square_size: float,
) -> tuple[list[SquareCandidate], list[tuple[int, int, float]], float]:
    """Refine predicted square centers to nearby response peaks."""
    candidates: list[SquareCandidate] = []
    matches: list[tuple[int, int, float]] = []
    errors: list[float] = []
    search_radius = max(3, int(round(0.42 * pitch)))
    patch_radius = max(2, int(round(0.42 * square_size)))
    h, w = response.shape

    for index, point in enumerate(predicted_local):
        cx, cy = float(point[0]), float(point[1])
        x0 = max(0, int(math.floor(cx)) - search_radius)
        x1 = min(w, int(math.floor(cx)) + search_radius + 1)
        y0 = max(0, int(math.floor(cy)) - search_radius)
        y1 = min(h, int(math.floor(cy)) + search_radius + 1)
        patch = response[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        _minimum, maximum, _min_loc, max_loc = cv2.minMaxLoc(patch)
        peak = np.asarray([x0 + max_loc[0], y0 + max_loc[1]], dtype=np.float64)
        error = float(np.linalg.norm(peak - point))

        inner_x0 = max(0, int(round(peak[0])) - patch_radius)
        inner_x1 = min(w, int(round(peak[0])) + patch_radius + 1)
        inner_y0 = max(0, int(round(peak[1])) - patch_radius)
        inner_y1 = min(h, int(round(peak[1])) + patch_radius + 1)
        inner = response[inner_y0:inner_y1, inner_x0:inner_x1]
        inner_mean = float(np.mean(inner)) if inner.size else 0.0

        annulus_radius = max(search_radius, int(round(0.75 * pitch)))
        ax0 = max(0, int(round(peak[0])) - annulus_radius)
        ax1 = min(w, int(round(peak[0])) + annulus_radius + 1)
        ay0 = max(0, int(round(peak[1])) - annulus_radius)
        ay1 = min(h, int(round(peak[1])) + annulus_radius + 1)
        neighborhood = response[ay0:ay1, ax0:ax1]
        background = float(np.median(neighborhood)) if neighborhood.size else 0.0
        contrast = max(inner_mean - background, float(maximum) - background)

        if maximum < 0.22 or contrast < 0.055 or error > 0.38 * pitch:
            continue
        bbox_half = max(2, int(round(0.5 * square_size)))
        candidate = SquareCandidate(
            label=-(index + 1),
            center=(float(peak[0]), float(peak[1])),
            bbox=(
                int(round(peak[0])) - bbox_half,
                int(round(peak[1])) - bbox_half,
                int(round(peak[0])) + bbox_half + 1,
                int(round(peak[1])) + bbox_half + 1,
            ),
            size=float(square_size),
            area=int(round(square_size * square_size)),
            fill_ratio=1.0,
        )
        candidate_index = len(candidates)
        candidates.append(candidate)
        matches.append((index, candidate_index, error))
        errors.append(error)

    mean_error = float(np.mean(errors)) if errors else float("inf")
    return candidates, matches, mean_error


def _guided_template_search(
    image: np.ndarray,
    target_side: str,
    center_px: np.ndarray,
    half_width: int,
    half_height: int,
    reference_marker: dict,
) -> tuple[list[dict], np.ndarray, list[SquareCandidate], list[str]]:
    """Recover a white-on-metal marker using the first marker as a template prior."""
    h_img, w_img = image.shape[:2]
    cx, cy = [int(round(value)) for value in center_px]
    x0 = max(0, cx - half_width)
    x1 = min(w_img, cx + half_width + 1)
    y0 = max(0, cy - half_height)
    y1 = min(h_img, cy + half_height + 1)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return [], np.zeros((0, 0), dtype=np.uint8), [], []

    reference_metrics = _candidate_metrics(reference_marker)
    response = _bright_square_response(crop, reference_metrics["square_size"])
    base_angle = (reference_metrics["angle_deg"] + 90.0) % 180.0 - 90.0
    best: dict | None = None

    for scale in np.linspace(0.86, 1.14, 8):
        pitch = float(reference_metrics["pitch"] * scale)
        square_size = float(reference_metrics["square_size"] * scale)
        for angle_delta in (-5.0, -3.0, -1.5, 0.0, 1.5, 3.0, 5.0):
            angle = float(base_angle + angle_delta)
            template, anchor_offset, transform = _make_guided_marker_template(
                target_side, pitch, square_size, angle
            )
            if template.shape[0] >= response.shape[0] or template.shape[1] >= response.shape[1]:
                continue
            correlation = cv2.matchTemplate(
                response.astype(np.float32), template.astype(np.float32), cv2.TM_CCOEFF_NORMED
            )
            _minimum, score, _min_loc, location = cv2.minMaxLoc(correlation)
            if not np.isfinite(score):
                continue
            anchor_local = np.asarray(location, dtype=np.float64) + anchor_offset
            predicted_local = TEMPLATE_POINTS @ transform.T + anchor_local
            refined, matches, mean_error = _refine_template_points(
                response, predicted_local, pitch, square_size
            )
            support = len(matches)
            support_score = support - 2.0 * mean_error / max(pitch, 1e-9)
            total_score = 80.0 * float(score) + 7.0 * support_score
            if best is None or total_score > best["total_score"]:
                best = {
                    "score": float(score),
                    "total_score": float(total_score),
                    "pitch": pitch,
                    "square_size": square_size,
                    "angle": angle,
                    "transform": transform,
                    "anchor_local": anchor_local,
                    "predicted_local": predicted_local,
                    "candidates_local": refined,
                    "matches": matches,
                    "mean_error": mean_error,
                }

    response_mask = np.where(response >= 0.20, 255, 0).astype(np.uint8)
    if best is None:
        return [], response_mask, [], [f"{target_side}:guided-white-template:none"]

    global_offset = np.asarray([x0, y0], dtype=np.float64)
    global_candidates = [
        SquareCandidate(
            label=candidate.label,
            center=(candidate.center[0] + x0, candidate.center[1] + y0),
            bbox=(
                candidate.bbox[0] + x0,
                candidate.bbox[1] + y0,
                candidate.bbox[2] + x0,
                candidate.bbox[3] + y0,
            ),
            size=candidate.size,
            area=candidate.area,
            fill_ratio=candidate.fill_ratio,
        )
        for candidate in best["candidates_local"]
    ]
    predicted_global = best["predicted_local"] + global_offset
    anchor_global = best["anchor_local"] + global_offset
    matched_indices = [match[1] for match in best["matches"]]
    result = {
        "score": best["total_score"],
        "pitch": best["pitch"],
        "square_size": best["square_size"],
        "transform": best["transform"],
        "translation": anchor_global,
        "predicted": predicted_global,
        "matches": best["matches"],
        "mean_error": best["mean_error"],
        "size_cv": 0.0,
        "anchor": anchor_global,
        "target_side": target_side,
        "rail_side": target_side,
        "left_pixels": 1 if target_side == "left" else 0,
        "right_pixels": 1 if target_side == "right" else 0,
        "matched_candidates": [global_candidates[index] for index in matched_indices],
        "crop_bbox": (x0, y0, x1, y1),
        "mask_name": "guided-white-template",
        "mask": response_mask,
        "global_candidates": global_candidates,
        "raw_score": best["total_score"],
        "detection_stage": "symmetry-template",
        "template_score": best["score"],
    }
    valid, rejection_reasons, metrics = _validate_candidate(
        result,
        target_side=target_side,
        expected_pitch_px=reference_metrics["pitch"],
        reference_marker=reference_marker,
    )
    result["valid"] = valid
    result["rejection_reasons"] = rejection_reasons
    result["metrics"] = metrics
    result["selection_score"] = best["total_score"] + (60.0 if valid else -25.0 * len(rejection_reasons))
    return [result], response_mask, global_candidates, [f"{target_side}:guided-white-template:symmetry-template"]


def _pair_consistency(
    first: dict,
    second: dict,
    wafer_center: np.ndarray,
    wafer_radius: float,
    gds_markers: list[GDSInnerMarker],
) -> tuple[bool, list[str], float]:
    reasons: list[str] = []
    first_metrics = _candidate_metrics(first)
    second_metrics = _candidate_metrics(second)
    angle_difference = _angle_difference_mod_180(
        first_metrics["angle_deg"], second_metrics["angle_deg"]
    )
    pitch_difference = abs(second_metrics["pitch"] / first_metrics["pitch"] - 1.0)
    size_difference = abs(
        second_metrics["square_size"] / max(first_metrics["square_size"], 1e-9) - 1.0
    )
    if angle_difference > 10.0:
        reasons.append(f"pair angle mismatch {angle_difference:.1f}°")
    if pitch_difference > 0.20:
        reasons.append(f"pair pitch mismatch {100*pitch_difference:.1f}%")
    if size_difference > 0.30:
        reasons.append(f"pair square-size mismatch {100*size_difference:.1f}%")

    side_to_result = {first["target_side"]: first, second["target_side"]: second}
    if set(side_to_result) != {"left", "right"}:
        reasons.append("pair does not contain one left and one right marker")
        return False, reasons, float("inf")
    left_anchor = np.asarray(side_to_result["left"]["anchor"], dtype=np.float64)
    right_anchor = np.asarray(side_to_result["right"]["anchor"], dtype=np.float64)
    if right_anchor[0] <= left_anchor[0]:
        reasons.append("right marker is not to the right of left marker")

    reflected_left = 2.0 * wafer_center - left_anchor
    reflected_right = 2.0 * wafer_center - right_anchor
    symmetry_error = 0.5 * (
        np.linalg.norm(right_anchor - reflected_left)
        + np.linalg.norm(left_anchor - reflected_right)
    )
    mean_pitch = 0.5 * (first_metrics["pitch"] + second_metrics["pitch"])
    symmetry_limit = max(15.0 * mean_pitch, 0.055 * wafer_radius)
    if symmetry_error > symmetry_limit:
        reasons.append(
            f"wafer-symmetry error {symmetry_error:.1f}px > {symmetry_limit:.1f}px"
        )

    gds_by_side = {marker.side: marker for marker in gds_markers}
    if set(gds_by_side) == {"left", "right"}:
        gds_separation = abs(
            gds_by_side["right"].anchor_um[0] - gds_by_side["left"].anchor_um[0]
        )
        gds_pitch = 0.5 * (
            gds_by_side["left"].x_pitch_um + gds_by_side["right"].x_pitch_um
        )
        expected_separation_px = mean_pitch * gds_separation / max(gds_pitch, 1e-9)
        observed_separation_px = float(np.linalg.norm(right_anchor - left_anchor))
        distance_error = abs(observed_separation_px / max(expected_separation_px, 1e-9) - 1.0)
        if distance_error > 0.18:
            reasons.append(
                f"marker separation differs from GDS/scale prediction by {100*distance_error:.1f}%"
            )

    pair_score = (
        float(first["selection_score"])
        + float(second["selection_score"])
        - 0.10 * symmetry_error
        - 2.0 * angle_difference
        - 30.0 * pitch_difference
    )
    return not reasons, reasons, float(pair_score)


def _result_to_image_marker(
    result: dict,
    load_info: ImageLoadInfo,
    symmetry_error_px: float | None,
) -> ImageInnerMarker:
    anchor = np.asarray(result["anchor"], dtype=np.float64)
    metrics = _candidate_metrics(result)
    pitch = metrics["pitch"]
    matched_candidates: list[SquareCandidate] = result["matched_candidates"]
    original_anchor = (
        float(anchor[0] / max(load_info.scale_x, 1e-12)),
        float(anchor[1] / max(load_info.scale_y, 1e-12)),
    )
    original_pitch = float(
        pitch / max(0.5 * (load_info.scale_x + load_info.scale_y), 1e-12)
    )
    return ImageInnerMarker(
        side=result["target_side"],
        anchor_px=(float(anchor[0]), float(anchor[1])),
        angle_deg=metrics["angle_deg"],
        pitch_px=pitch,
        square_size_px=metrics["square_size"],
        matched_square_count=len(result["matches"]),
        mean_error_px=float(result["mean_error"]),
        rail_left_pixels=int(result["left_pixels"]),
        rail_right_pixels=int(result["right_pixels"]),
        score=float(result["selection_score"]),
        matched_candidate_labels=[candidate.label for candidate in matched_candidates],
        predicted_square_centers_px=[
            (float(point[0]), float(point[1])) for point in result["predicted"]
        ],
        matched_square_centers_px=[candidate.center for candidate in matched_candidates],
        anchor_original_px=original_anchor,
        pitch_original_px=original_pitch,
        error_pitch_ratio=metrics["error_pitch_ratio"],
        square_pitch_ratio=metrics["square_pitch_ratio"],
        symmetry_error_px=symmetry_error_px,
        detection_stage=str(result["detection_stage"]),
        template_score=(float(result["template_score"]) if result.get("template_score") is not None else None),
    )


def detect_image_inner_markers(
    image_path: Path,
    gds_markers: list[GDSInnerMarker],
    design_bbox: tuple[float, float, float, float],
    max_megapixels: float = 30.0,
    max_dimension: int = 7000,
    search_half_width_fraction: float = 0.10,
    search_half_height_fraction: float = 0.16,
) -> tuple[
    list[ImageInnerMarker],
    np.ndarray,
    np.ndarray,
    str,
    list[SquareCandidate],
    ImageLoadInfo,
]:
    image, load_info = _load_image_for_detection(
        image_path, max_megapixels=max_megapixels, max_dimension=max_dimension
    )
    h_img, w_img = image.shape[:2]
    x_min_gds, y_min_gds, x_max_gds, y_max_gds = design_bbox
    gds_w = max(x_max_gds - x_min_gds, 1e-9)
    gds_h = max(y_max_gds - y_min_gds, 1e-9)

    expected_square_px = 0.5 * (
        w_img * 200.0 / gds_w + h_img * 200.0 / gds_h
    )
    expected_pitch_px = 2.0 * expected_square_px
    wafer_center, wafer_radius, center_mode = _estimate_wafer_center(image)

    combined_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    combined_candidates: list[SquareCandidate] = []
    selected_modes: list[str] = [center_mode]
    all_results_by_side: dict[str, list[dict]] = {"left": [], "right": []}

    for gds_marker in gds_markers:
        nx = (gds_marker.anchor_um[0] - x_min_gds) / gds_w
        ny = (y_max_gds - gds_marker.anchor_um[1]) / gds_h
        predicted_center = np.asarray(
            [nx * (w_img - 1), ny * (h_img - 1)], dtype=np.float64
        )
        half_w = max(
            int(round(search_half_width_fraction * w_img)),
            int(round(35.0 * expected_square_px)),
        )
        half_h = max(
            int(round(search_half_height_fraction * h_img)),
            int(round(20.0 * expected_square_px)),
        )
        results, local_mask, local_candidates, modes = _search_marker_window(
            image=image,
            target_side=gds_marker.side,
            center_px=predicted_center,
            half_width=half_w,
            half_height=half_h,
            expected_square_px=expected_square_px,
            expected_pitch_px=expected_pitch_px,
            stage="initial",
        )
        all_results_by_side[gds_marker.side].extend(results)
        if results:
            x0, y0, x1, y1 = results[0]["crop_bbox"]
            if local_mask.size:
                combined_mask[y0:y1, x0:x1] = np.maximum(
                    combined_mask[y0:y1, x0:x1], local_mask
                )
            combined_candidates.extend(local_candidates)
            selected_modes.extend(modes)

    valid_by_side = {
        side: [result for result in results if result["valid"]]
        for side, results in all_results_by_side.items()
    }
    for results in valid_by_side.values():
        results.sort(key=lambda item: item["selection_score"], reverse=True)

    best_pair: tuple[dict, dict, float, float] | None = None
    for left in valid_by_side["left"][:30]:
        for right in valid_by_side["right"][:30]:
            pair_valid, _reasons, pair_score = _pair_consistency(
                left, right, wafer_center, wafer_radius, gds_markers
            )
            if not pair_valid:
                continue
            symmetry_error = float(
                np.linalg.norm(np.asarray(right["anchor"]) - (2.0 * wafer_center - np.asarray(left["anchor"])))
            )
            if best_pair is None or pair_score > best_pair[2]:
                best_pair = (left, right, pair_score, symmetry_error)

    # If the broad independent searches do not yield a consistent pair, use the
    # strongest valid marker as a seed and search tightly at its reflected wafer
    # position.  This prevents wafer-edge clutter from being promoted merely to
    # satisfy the expectation of two markers.
    if best_pair is None:
        seed_candidates = valid_by_side["left"][:1] + valid_by_side["right"][:1]
        seed_candidates.sort(key=lambda item: item["selection_score"], reverse=True)
        if seed_candidates:
            seed = seed_candidates[0]
            opposite_side = "right" if seed["target_side"] == "left" else "left"
            reflected_center = 2.0 * wafer_center - np.asarray(seed["anchor"], dtype=np.float64)
            seed_metrics = _candidate_metrics(seed)
            target_half_w = max(int(round(0.055 * w_img)), int(round(24.0 * seed_metrics["pitch"])))
            target_half_h = max(int(round(0.070 * h_img)), int(round(18.0 * seed_metrics["pitch"])))
            targeted_results, targeted_mask, targeted_candidates, targeted_modes = _search_marker_window(
                image=image,
                target_side=opposite_side,
                center_px=reflected_center,
                half_width=target_half_w,
                half_height=target_half_h,
                expected_square_px=seed_metrics["square_size"],
                expected_pitch_px=seed_metrics["pitch"],
                stage="symmetry-targeted",
                reference_marker=seed,
                max_results=120,
            )
            # A real right-side marker can be white squares on a bright gold pad.
            # If connected components cannot separate those squares, run the
            # guided square+rail correlation fallback at the same symmetric ROI.
            if not any(result.get("valid", False) for result in targeted_results):
                fallback_results, fallback_mask, fallback_candidates, fallback_modes = _guided_template_search(
                    image=image,
                    target_side=opposite_side,
                    center_px=reflected_center,
                    half_width=target_half_w,
                    half_height=target_half_h,
                    reference_marker=seed,
                )
                targeted_results.extend(fallback_results)
                targeted_modes.extend(fallback_modes)
                targeted_candidates.extend(fallback_candidates)
                if fallback_results and fallback_mask.size:
                    fx0, fy0, fx1, fy1 = fallback_results[0]["crop_bbox"]
                    combined_mask[fy0:fy1, fx0:fx1] = np.maximum(
                        combined_mask[fy0:fy1, fx0:fx1], fallback_mask
                    )

            selected_modes.extend(targeted_modes)
            combined_candidates.extend(targeted_candidates)
            if targeted_results:
                x0, y0, x1, y1 = targeted_results[0]["crop_bbox"]
                if targeted_mask.size:
                    combined_mask[y0:y1, x0:x1] = np.maximum(
                        combined_mask[y0:y1, x0:x1], targeted_mask
                    )
            for opposite in targeted_results:
                if not opposite["valid"]:
                    continue
                first, second = (
                    (seed, opposite)
                    if seed["target_side"] == "left"
                    else (opposite, seed)
                )
                pair_valid, _reasons, pair_score = _pair_consistency(
                    first, second, wafer_center, wafer_radius, gds_markers
                )
                if not pair_valid:
                    continue
                symmetry_error = float(
                    np.linalg.norm(
                        np.asarray(second["anchor"])
                        - (2.0 * wafer_center - np.asarray(first["anchor"]))
                    )
                )
                if best_pair is None or pair_score > best_pair[2]:
                    best_pair = (first, second, pair_score, symmetry_error)

    detections: list[ImageInnerMarker] = []
    if best_pair is not None:
        left, right, _pair_score, symmetry_error = best_pair
        detections = [
            _result_to_image_marker(left, load_info, symmetry_error),
            _result_to_image_marker(right, load_info, symmetry_error),
        ]
    else:
        # Preserve the one genuinely validated marker, but do not fabricate its
        # partner from a weak edge candidate.  The nonzero exit status in main()
        # makes this incomplete result visible to automated tests.
        valid_all = valid_by_side["left"] + valid_by_side["right"]
        if valid_all:
            seed = max(valid_all, key=lambda item: item["selection_score"])
            detections = [_result_to_image_marker(seed, load_info, None)]

    detections.sort(key=lambda detection: detection.anchor_px[0])
    return (
        detections,
        image,
        combined_mask,
        ", ".join(dict.fromkeys(selected_modes)) if selected_modes else "none",
        combined_candidates,
        load_info,
    )


def detect_image_inner_markers_array(
    image: np.ndarray,
    gds_markers: list[GDSInnerMarker],
    design_bbox: tuple[float, float, float, float],
    max_megapixels: float = 30.0,
    max_dimension: int = 7000,
    search_half_width_fraction: float = 0.10,
    search_half_height_fraction: float = 0.16,
) -> tuple[
    list[ImageInnerMarker],
    np.ndarray,
    np.ndarray,
    str,
    list[SquareCandidate],
    ImageLoadInfo,
]:
    if image is None or image.size == 0:
        raise RuntimeError("Input wafer canvas is empty")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim != 3 or image.shape[2] not in (3, 4):
        raise RuntimeError(f"Unsupported wafer canvas shape: {image.shape}")
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    original_h, original_w = image.shape[:2]
    max_pixels = max(1, int(float(max_megapixels) * 1_000_000))
    scale_pixels = math.sqrt(max_pixels / float(max(original_w * original_h, 1)))
    scale_dimension = float(max_dimension) / float(max(original_w, original_h))
    working_scale = min(1.0, scale_pixels, scale_dimension)
    if working_scale < 1.0:
        working_w = max(1, int(round(original_w * working_scale)))
        working_h = max(1, int(round(original_h * working_scale)))
        image = cv2.resize(image, (working_w, working_h), interpolation=cv2.INTER_AREA)
        decoder = "opencv-array-resize"
    else:
        working_h, working_w = original_h, original_w
        image = image.copy()
        decoder = "opencv-array-copy"
    load_info = ImageLoadInfo(
        original_size_px=(int(original_w), int(original_h)),
        working_size_px=(int(working_w), int(working_h)),
        scale_x=float(working_w / original_w),
        scale_y=float(working_h / original_h),
        decoder=decoder,
    )
    h_img, w_img = image.shape[:2]
    x_min_gds, y_min_gds, x_max_gds, y_max_gds = design_bbox
    gds_w = max(x_max_gds - x_min_gds, 1e-9)
    gds_h = max(y_max_gds - y_min_gds, 1e-9)

    expected_square_px = 0.5 * (
        w_img * 200.0 / gds_w + h_img * 200.0 / gds_h
    )
    expected_pitch_px = 2.0 * expected_square_px
    wafer_center, wafer_radius, center_mode = _estimate_wafer_center(image)

    combined_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    combined_candidates: list[SquareCandidate] = []
    selected_modes: list[str] = [center_mode]
    all_results_by_side: dict[str, list[dict]] = {"left": [], "right": []}

    for gds_marker in gds_markers:
        nx = (gds_marker.anchor_um[0] - x_min_gds) / gds_w
        ny = (y_max_gds - gds_marker.anchor_um[1]) / gds_h
        predicted_center = np.asarray(
            [nx * (w_img - 1), ny * (h_img - 1)], dtype=np.float64
        )
        half_w = max(
            int(round(search_half_width_fraction * w_img)),
            int(round(35.0 * expected_square_px)),
        )
        half_h = max(
            int(round(search_half_height_fraction * h_img)),
            int(round(20.0 * expected_square_px)),
        )
        results, local_mask, local_candidates, modes = _search_marker_window(
            image=image,
            target_side=gds_marker.side,
            center_px=predicted_center,
            half_width=half_w,
            half_height=half_h,
            expected_square_px=expected_square_px,
            expected_pitch_px=expected_pitch_px,
            stage="initial",
        )
        all_results_by_side[gds_marker.side].extend(results)
        if results:
            x0, y0, x1, y1 = results[0]["crop_bbox"]
            if local_mask.size:
                combined_mask[y0:y1, x0:x1] = np.maximum(
                    combined_mask[y0:y1, x0:x1], local_mask
                )
            combined_candidates.extend(local_candidates)
            selected_modes.extend(modes)

    valid_by_side = {
        side: [result for result in results if result["valid"]]
        for side, results in all_results_by_side.items()
    }
    for results in valid_by_side.values():
        results.sort(key=lambda item: item["selection_score"], reverse=True)

    best_pair: tuple[dict, dict, float, float] | None = None
    for left in valid_by_side["left"][:30]:
        for right in valid_by_side["right"][:30]:
            pair_valid, _reasons, pair_score = _pair_consistency(
                left, right, wafer_center, wafer_radius, gds_markers
            )
            if not pair_valid:
                continue
            symmetry_error = float(
                np.linalg.norm(np.asarray(right["anchor"]) - (2.0 * wafer_center - np.asarray(left["anchor"])))
            )
            if best_pair is None or pair_score > best_pair[2]:
                best_pair = (left, right, pair_score, symmetry_error)

    # If the broad independent searches do not yield a consistent pair, use the
    # strongest valid marker as a seed and search tightly at its reflected wafer
    # position.  This prevents wafer-edge clutter from being promoted merely to
    # satisfy the expectation of two markers.
    if best_pair is None:
        seed_candidates = valid_by_side["left"][:1] + valid_by_side["right"][:1]
        seed_candidates.sort(key=lambda item: item["selection_score"], reverse=True)
        if seed_candidates:
            seed = seed_candidates[0]
            opposite_side = "right" if seed["target_side"] == "left" else "left"
            reflected_center = 2.0 * wafer_center - np.asarray(seed["anchor"], dtype=np.float64)
            seed_metrics = _candidate_metrics(seed)
            target_half_w = max(int(round(0.055 * w_img)), int(round(24.0 * seed_metrics["pitch"])))
            target_half_h = max(int(round(0.070 * h_img)), int(round(18.0 * seed_metrics["pitch"])))
            targeted_results, targeted_mask, targeted_candidates, targeted_modes = _search_marker_window(
                image=image,
                target_side=opposite_side,
                center_px=reflected_center,
                half_width=target_half_w,
                half_height=target_half_h,
                expected_square_px=seed_metrics["square_size"],
                expected_pitch_px=seed_metrics["pitch"],
                stage="symmetry-targeted",
                reference_marker=seed,
                max_results=120,
            )
            # A real right-side marker can be white squares on a bright gold pad.
            # If connected components cannot separate those squares, run the
            # guided square+rail correlation fallback at the same symmetric ROI.
            if not any(result.get("valid", False) for result in targeted_results):
                fallback_results, fallback_mask, fallback_candidates, fallback_modes = _guided_template_search(
                    image=image,
                    target_side=opposite_side,
                    center_px=reflected_center,
                    half_width=target_half_w,
                    half_height=target_half_h,
                    reference_marker=seed,
                )
                targeted_results.extend(fallback_results)
                targeted_modes.extend(fallback_modes)
                targeted_candidates.extend(fallback_candidates)
                if fallback_results and fallback_mask.size:
                    fx0, fy0, fx1, fy1 = fallback_results[0]["crop_bbox"]
                    combined_mask[fy0:fy1, fx0:fx1] = np.maximum(
                        combined_mask[fy0:fy1, fx0:fx1], fallback_mask
                    )

            selected_modes.extend(targeted_modes)
            combined_candidates.extend(targeted_candidates)
            if targeted_results:
                x0, y0, x1, y1 = targeted_results[0]["crop_bbox"]
                if targeted_mask.size:
                    combined_mask[y0:y1, x0:x1] = np.maximum(
                        combined_mask[y0:y1, x0:x1], targeted_mask
                    )
            for opposite in targeted_results:
                if not opposite["valid"]:
                    continue
                first, second = (
                    (seed, opposite)
                    if seed["target_side"] == "left"
                    else (opposite, seed)
                )
                pair_valid, _reasons, pair_score = _pair_consistency(
                    first, second, wafer_center, wafer_radius, gds_markers
                )
                if not pair_valid:
                    continue
                symmetry_error = float(
                    np.linalg.norm(
                        np.asarray(second["anchor"])
                        - (2.0 * wafer_center - np.asarray(first["anchor"]))
                    )
                )
                if best_pair is None or pair_score > best_pair[2]:
                    best_pair = (first, second, pair_score, symmetry_error)

    detections: list[ImageInnerMarker] = []
    if best_pair is not None:
        left, right, _pair_score, symmetry_error = best_pair
        detections = [
            _result_to_image_marker(left, load_info, symmetry_error),
            _result_to_image_marker(right, load_info, symmetry_error),
        ]
    else:
        # Preserve the one genuinely validated marker, but do not fabricate its
        # partner from a weak edge candidate.  The nonzero exit status in main()
        # makes this incomplete result visible to automated tests.
        valid_all = valid_by_side["left"] + valid_by_side["right"]
        if valid_all:
            seed = max(valid_all, key=lambda item: item["selection_score"])
            detections = [_result_to_image_marker(seed, load_info, None)]

    detections.sort(key=lambda detection: detection.anchor_px[0])
    return (
        detections,
        image,
        combined_mask,
        ", ".join(dict.fromkeys(selected_modes)) if selected_modes else "none",
        combined_candidates,
        load_info,
    )


FUTURE_ALIGNMENT_VERSION = "inner-fiducial-v3-production-2026-07-20"

def detect_future_alignment_on_canvas(
    image: np.ndarray,
    gds_path: str | Path,
    *,
    max_megapixels: float = 30.0,
    max_dimension: int = 7000,
) -> tuple[list[GDSInnerMarker], list[ImageInnerMarker], np.ndarray, np.ndarray, str, ImageLoadInfo]:
    """Detect both future-design inner fiducials on an in-memory wafer canvas."""
    gds_path = Path(gds_path)
    gds_markers, _polygons, design_bbox = detect_gds_inner_markers(gds_path)
    if len(gds_markers) != 2:
        raise RuntimeError(f"Expected 2 GDS inner fiducials, found {len(gds_markers)}")
    detections, working_image, mask, mode, _candidates, load_info = detect_image_inner_markers_array(
        image,
        gds_markers,
        design_bbox,
        max_megapixels=max_megapixels,
        max_dimension=max_dimension,
    )
    return gds_markers, detections, working_image, mask, mode, load_info

def plot_image_inner_markers(
    image: np.ndarray,
    mask: np.ndarray,
    markers: list[ImageInnerMarker],
    mode: str,
    candidates: list[SquareCandidate],
    output_path: Path,
    load_info: ImageLoadInfo | None = None,
) -> None:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(rgb)
    if load_info is None:
        axes[0].set_title("Inner-fiducial detections")
    else:
        axes[0].set_title(
            f"Inner-fiducial detections — working {load_info.working_size_px[0]}×{load_info.working_size_px[1]} "
            f"from {load_info.original_size_px[0]}×{load_info.original_size_px[1]}"
        )
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"Selected foreground mask ({mode})")

    for ax in axes:
        for candidate in candidates:
            x0, y0, x1, y1 = candidate.bbox
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], linewidth=0.35, alpha=0.25)
        for marker in markers:
            predicted = np.asarray(marker.predicted_square_centers_px)
            matched = np.asarray(marker.matched_square_centers_px)
            ax.scatter(predicted[:, 0], predicted[:, 1], marker="+", s=45)
            ax.scatter(matched[:, 0], matched[:, 1], facecolors="none", s=90)
            ax.scatter([marker.anchor_px[0]], [marker.anchor_px[1]], marker="x", s=100)
            ax.text(
                marker.anchor_px[0],
                marker.anchor_px[1] - 3.0 * marker.pitch_px,
                f"{marker.side.upper()}  {marker.matched_square_count}/12\n"
                f"work=({marker.anchor_px[0]:.1f}, {marker.anchor_px[1]:.1f}) px"
                + (
                    f"\norig=({marker.anchor_original_px[0]:.1f}, {marker.anchor_original_px[1]:.1f}) px"
                    if marker.anchor_original_px is not None
                    else ""
                ),
                ha="center",
                va="bottom",
                bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
            )
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _jsonable_dataclass(obj) -> dict:
    return asdict(obj)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gds", type=Path, required=True, help="Path to GDS file")
    parser.add_argument(
        "--wafer-image",
        type=Path,
        action="append",
        default=[],
        help="Image containing one or both compact markers. May be supplied more than once.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("future_inner_marker_test_output")
    )
    parser.add_argument(
        "--max-image-megapixels",
        type=float,
        default=30.0,
        help="Maximum decoded working-image area. Large JPEGs are reduced during decode.",
    )
    parser.add_argument(
        "--max-image-dimension",
        type=int,
        default=7000,
        help="Maximum width or height of the working image.",
    )
    parser.add_argument(
        "--search-half-width-fraction",
        type=float,
        default=0.10,
        help="Half-width of each GDS-predicted marker search window as a fraction of image width.",
    )
    parser.add_argument(
        "--search-half-height-fraction",
        type=float,
        default=0.16,
        help="Half-height of each marker search window as a fraction of image height.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gds_markers, polygons, design_bbox = detect_gds_inner_markers(args.gds)
    gds_plot = args.output_dir / "gds_inner_marker_detection.png"
    plot_gds_inner_markers(polygons, gds_markers, gds_plot)

    report: dict = {
        "gds_path": str(args.gds),
        "design_bbox_um": list(design_bbox),
        "gds_inner_markers": [_jsonable_dataclass(marker) for marker in gds_markers],
        "gds_plot": str(gds_plot),
        "image_tests": [],
    }

    image_failure = False
    for image_index, image_path in enumerate(args.wafer_image, start=1):
        markers, image, mask, mode, candidates, load_info = detect_image_inner_markers(
            image_path,
            gds_markers=gds_markers,
            design_bbox=design_bbox,
            max_megapixels=args.max_image_megapixels,
            max_dimension=args.max_image_dimension,
            search_half_width_fraction=args.search_half_width_fraction,
            search_half_height_fraction=args.search_half_height_fraction,
        )
        image_plot = args.output_dir / f"image_{image_index}_inner_marker_detection.png"
        plot_image_inner_markers(
            image, mask, markers, mode, candidates, image_plot, load_info=load_info
        )
        report["image_tests"].append(
            {
                "image_path": str(image_path),
                "threshold_mode": mode,
                "image_load": _jsonable_dataclass(load_info),
                "markers": [_jsonable_dataclass(marker) for marker in markers],
                "plot": str(image_plot),
            }
        )
        print(f"Image: {image_path}")
        print(
            f"  loaded {load_info.original_size_px[0]}x{load_info.original_size_px[1]} -> "
            f"{load_info.working_size_px[0]}x{load_info.working_size_px[1]} "
            f"using {load_info.decoder}"
        )
        print(f"  inner markers found: {len(markers)} using {mode}")
        for marker in markers:
            print(
                f"  {marker.side:>7}: work_anchor={marker.anchor_px}, "
                f"original_anchor={marker.anchor_original_px}, "
                f"matched={marker.matched_square_count}/12, angle={marker.angle_deg:.3f} deg, "
                f"error={marker.mean_error_px:.2f}px "
                f"({marker.error_pitch_ratio:.3f} pitch), "
                f"square/pitch={marker.square_pitch_ratio:.3f}, stage={marker.detection_stage}"
                + (f", template={marker.template_score:.3f}" if marker.template_score is not None else "")
            )
        if {marker.side for marker in markers} != {"left", "right"}:
            image_failure = True
            print("  WARNING: a validated left/right pair was not found; weak candidates were rejected.")

    report_path = args.output_dir / "future_inner_marker_detection.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"GDS inner markers found: {len(gds_markers)}")
    for marker in gds_markers:
        print(
            f"  {marker.side:>5}: anchor={marker.anchor_um}, layer={marker.layer}/{marker.datatype}, "
            f"squares={marker.square_count}, pitch=({marker.x_pitch_um:.1f}, {marker.y_pitch_um:.1f})um"
        )
    print(f"Report: {report_path}")

    if {marker.side for marker in gds_markers} != {"left", "right"}:
        return 2
    if image_failure:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
