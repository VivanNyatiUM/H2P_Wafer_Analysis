"""Cell-border refinement for H2P device image extraction.

The GDS-mapped cell rectangle is treated as a seed, not as the final crop.
The image is first rectified using that approximate quadrilateral, then the
nearest long physical border is searched for on each side.  The four fitted
border lines are intersected and the original full-resolution local stitch is
warped once into the final cell image.

Only NumPy and OpenCV are required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Callable
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import threading
import math
import sys
import time

import cv2
import numpy as np

BOUND_HELPER_VERSION = "bound-crop-v6-parallel-2026-07-15"


@dataclass
class BoundaryAlignmentResult:
    success: bool
    image: np.ndarray | None = None
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    detected_corners: np.ndarray | None = None  # TL, TR, BR, BL in source-local pixels
    homography: np.ndarray | None = None  # source-local -> final crop
    confidence: float = 0.0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def order_quad_points(points: np.ndarray) -> np.ndarray:
    """Return four 2-D points in TL, TR, BR, BL order."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    cyc = pts[np.argsort(angles)]
    # Sorted angles normally start around the top-left, but normalize explicitly.
    start = int(np.argmin(cyc[:, 0] + cyc[:, 1]))
    cyc = np.roll(cyc, -start, axis=0)
    # Ensure clockwise TL,TR,BR,BL rather than TL,BL,BR,TR.
    cross = np.cross(cyc[1] - cyc[0], cyc[2] - cyc[1])
    if cross < 0:
        cyc = cyc[[0, 3, 2, 1]]
    return cyc.astype(np.float32)


def _quad_side_lengths(quad: np.ndarray) -> tuple[float, float]:
    q = order_quad_points(quad)
    width = 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3]))
    height = 0.5 * (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1]))
    return float(width), float(height)


def _robust_z(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    return (v - med) / max(1.4826 * mad, 1e-6)


def _smooth_1d(values: np.ndarray, sigma: float) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32).reshape(1, -1)
    if v.shape[1] < 3:
        return v.ravel()
    k = max(3, int(round(sigma * 6.0)) | 1)
    k = min(k, (v.shape[1] - 1) | 1)
    return cv2.GaussianBlur(v, (k, 1), sigmaX=max(float(sigma), 0.5)).ravel()


def _moving_mean(values: np.ndarray, radius: int) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32).ravel()
    r = max(1, int(radius))
    kernel = np.ones(2 * r + 1, dtype=np.float32) / float(2 * r + 1)
    return np.convolve(v, kernel, mode="same")


def _directional_transition(profile: np.ndarray, radius: int, inside_positive: bool) -> np.ndarray:
    """Difference across each candidate position using broad side bands."""
    p = np.asarray(profile, dtype=np.float32).ravel()
    r = max(2, int(radius))
    cs = np.concatenate(([0.0], np.cumsum(p, dtype=np.float64)))
    n = len(p)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        la, lb = max(0, i - r), i
        ra, rb = i + 1, min(n, i + r + 1)
        left = (cs[lb] - cs[la]) / max(lb - la, 1)
        right = (cs[rb] - cs[ra]) / max(rb - ra, 1)
        d = right - left
        out[i] = d if inside_positive else -d
    return np.maximum(out, 0.0)


def _build_rectified_search_image(
    image: np.ndarray,
    approximate_quad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rectify the approximate quad and retain all source pixels around it."""
    src_quad = order_quad_points(approximate_quad)
    approx_w, approx_h = _quad_side_lengths(src_quad)
    if approx_w < 8 or approx_h < 8:
        raise ValueError("Approximate GDS quadrilateral is too small")

    dst_quad = np.array(
        [[0.0, 0.0], [approx_w, 0.0], [approx_w, approx_h], [0.0, approx_h]],
        dtype=np.float32,
    )
    h0 = cv2.getPerspectiveTransform(src_quad, dst_quad)
    ih, iw = image.shape[:2]
    src_corners = np.array([[[0.0, 0.0], [iw - 1.0, 0.0], [iw - 1.0, ih - 1.0], [0.0, ih - 1.0]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(src_corners, h0)[0]
    min_xy = np.floor(mapped.min(axis=0) - 3.0)
    max_xy = np.ceil(mapped.max(axis=0) + 3.0)
    shift = np.array(
        [[1.0, 0.0, -float(min_xy[0])], [0.0, 1.0, -float(min_xy[1])], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    h_rect = shift @ h0
    out_w = int(max(16, math.ceil(max_xy[0] - min_xy[0] + 1.0)))
    out_h = int(max(16, math.ceil(max_xy[1] - min_xy[1] + 1.0)))
    # Prevent a pathological transform from allocating a ridiculous canvas.
    if out_w > max(iw * 8, 30000) or out_h > max(ih * 8, 30000):
        raise ValueError("Approximate quadrilateral produced an unstable rectification")
    rectified = cv2.warpPerspective(
        image,
        h_rect,
        (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    seed_rect = cv2.perspectiveTransform(src_quad.reshape(1, 4, 2), h_rect.astype(np.float32))[0]
    return rectified, h_rect.astype(np.float64), order_quad_points(seed_rect)


def _feature_maps(image: np.ndarray) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    gray = np.asarray(gray, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (0, 0), 1.0)
    gx = np.abs(cv2.Scharr(blur, cv2.CV_32F, 1, 0))
    gy = np.abs(cv2.Scharr(blur, cv2.CV_32F, 0, 1))

    min_side = max(16, min(gray.shape[:2]))
    k = max(9, int(round(min_side * 0.018)) | 1)
    f = gray.astype(np.float32)
    mean = cv2.boxFilter(f, cv2.CV_32F, (k, k), normalize=True)
    mean2 = cv2.boxFilter(f * f, cv2.CV_32F, (k, k), normalize=True)
    texture = np.sqrt(np.maximum(mean2 - mean * mean, 0.0))
    energy = cv2.boxFilter(gx + 0.35 * gy, cv2.CV_32F, (k, k), normalize=True)

    if image.ndim == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    else:
        lab = np.repeat(gray[:, :, None].astype(np.float32), 3, axis=2)
    return {"gray": gray, "gx": gx, "gy": gy, "texture": texture, "energy": energy, "lab": lab}


def _side_profile(
    features: Mapping[str, np.ndarray],
    axis: int,
    orth_lo: int,
    orth_hi: int,
    inside_positive: bool,
    transition_radius: int,
) -> np.ndarray:
    """Build a 1-D border score; axis=0 means x candidates, axis=1 means y."""
    gx = features["gx"]
    gy = features["gy"]
    texture = features["texture"]
    energy = features["energy"]
    lab = features["lab"]

    h, w = gx.shape[:2]
    if axis == 0:
        lo, hi = max(0, orth_lo), min(h, orth_hi)
        normal_edge = gx[lo:hi, :]
        tex_p = texture[lo:hi, :].mean(axis=0)
        energy_p = energy[lo:hi, :].mean(axis=0)
        lab_p = lab[lo:hi, :, :].mean(axis=0)
        # A robust high quantile rewards a line that is present for much of its length.
        edge_p = np.percentile(normal_edge, 70, axis=0)
    else:
        lo, hi = max(0, orth_lo), min(w, orth_hi)
        normal_edge = gy[:, lo:hi]
        tex_p = texture[:, lo:hi].mean(axis=1)
        energy_p = energy[:, lo:hi].mean(axis=1)
        lab_p = lab[:, lo:hi, :].mean(axis=1)
        edge_p = np.percentile(normal_edge, 70, axis=1)

    tex_t = _directional_transition(tex_p, transition_radius, inside_positive)
    energy_t = _directional_transition(energy_p, transition_radius, inside_positive)
    color_t = np.zeros_like(edge_p, dtype=np.float32)
    for channel in range(3):
        color_t += _directional_transition(lab_p[:, channel], transition_radius, inside_positive) ** 2
    color_t = np.sqrt(color_t)

    score = (
        0.65 * _robust_z(edge_p)
        + 1.65 * _robust_z(energy_t)
        + 1.05 * _robust_z(tex_t)
        + 0.80 * _robust_z(color_t)
    )
    return _smooth_1d(score, sigma=max(1.2, transition_radius * 0.16))


def _choose_side_position(
    score: np.ndarray,
    search_lo: int,
    search_hi: int,
    seed_edge: float,
    outward_sign: int,
) -> tuple[float, float, dict[str, float]]:
    n = len(score)
    lo = max(1, min(int(search_lo), n - 2))
    hi = max(lo + 1, min(int(search_hi), n - 1))
    segment = score[lo:hi]
    if segment.size < 3:
        raise ValueError("Boundary search band collapsed")

    # Prefer a strong transition, with only a mild distance penalty. This keeps
    # the search moving outward instead of snapping to the next device stripe.
    coords = np.arange(lo, hi, dtype=np.float32)
    span = max(float(hi - lo), 1.0)
    distance = np.abs(coords - float(seed_edge)) / span
    adjusted = segment - 0.12 * distance
    local_max = np.ones_like(adjusted, dtype=bool)
    local_max[1:-1] = (adjusted[1:-1] >= adjusted[:-2]) & (adjusted[1:-1] >= adjusted[2:])
    candidates = np.where(local_max)[0]
    seg_med = float(np.median(segment))
    seg_mad = float(np.median(np.abs(segment - seg_med)))
    seg_scale = max(1.4826 * seg_mad, 1e-6)
    z_values = (segment - seg_med) / seg_scale
    if candidates.size == 0:
        idx_local = int(np.argmax(adjusted))
    else:
        # Ignore a thin band at the outer ROI edge. Perspective warps use
        # replicated pixels there, and that artificial image edge can otherwise
        # beat the real physical border by a small margin.
        guard = max(3, int(round(0.04 * span)))
        absolute = lo + candidates
        if outward_sign > 0:
            valid = absolute <= (hi - guard)
        else:
            valid = absolute >= (lo + guard)
        guarded = candidates[valid]
        pool = guarded if guarded.size else candidates
        idx_local = int(pool[np.argmax(adjusted[pool])])
    pos = float(lo + idx_local)

    peak_z = float(z_values[idx_local])
    # Convert a useful z range into 0..1 without pretending it is a probability.
    confidence = float(np.clip((peak_z - 0.5) / 5.0, 0.0, 1.0))
    diagnostics = {
        "peak_score": float(segment[idx_local]),
        "peak_z": float(peak_z),
        "search_lo": float(lo),
        "search_hi": float(hi),
        "distance_from_seed": float((pos - seed_edge) * outward_sign),
    }
    return pos, confidence, diagnostics


def _robust_line_fit(points: np.ndarray, mode: str) -> tuple[float, float, float]:
    """Fit y=a*x+b (horizontal) or x=a*y+b (vertical)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 8:
        raise ValueError("Not enough edge points to fit a border line")
    keep = np.ones(len(pts), dtype=bool)
    a = 0.0
    b = 0.0
    for _ in range(5):
        p = pts[keep]
        if len(p) < 8:
            break
        if mode == "horizontal":
            independent, dependent = p[:, 0], p[:, 1]
        else:
            independent, dependent = p[:, 1], p[:, 0]
        a, b = np.polyfit(independent, dependent, 1)
        residual = dependent - (a * independent + b)
        mad = np.median(np.abs(residual - np.median(residual)))
        threshold = max(1.5, 2.8 * 1.4826 * mad)
        new_keep_local = np.abs(residual) <= threshold
        full_indices = np.flatnonzero(keep)
        new_keep = np.zeros_like(keep)
        new_keep[full_indices[new_keep_local]] = True
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    p = pts[keep]
    if len(p) < 8:
        p = pts
    if mode == "horizontal":
        residual = p[:, 1] - (a * p[:, 0] + b)
    else:
        residual = p[:, 0] - (a * p[:, 1] + b)
    rms = float(np.sqrt(np.mean(residual * residual))) if len(residual) else float("inf")
    return float(a), float(b), rms


def _collect_line_points(
    normal_map: np.ndarray,
    coarse_position: float,
    mode: str,
    span_lo: int,
    span_hi: int,
    refine_radius: int,
) -> np.ndarray:
    h, w = normal_map.shape[:2]
    points: list[tuple[float, float]] = []
    if mode == "horizontal":
        x0, x1 = max(0, span_lo), min(w, span_hi)
        step = max(1, int(round((x1 - x0) / 500.0)))
        y0 = max(0, int(round(coarse_position)) - refine_radius)
        y1 = min(h, int(round(coarse_position)) + refine_radius + 1)
        if y1 <= y0:
            return np.empty((0, 2), dtype=np.float32)
        for x in range(x0, x1, step):
            column = normal_map[y0:y1, x]
            yi = int(np.argmax(column))
            points.append((float(x), float(y0 + yi)))
    else:
        y0, y1 = max(0, span_lo), min(h, span_hi)
        step = max(1, int(round((y1 - y0) / 500.0)))
        x0 = max(0, int(round(coarse_position)) - refine_radius)
        x1 = min(w, int(round(coarse_position)) + refine_radius + 1)
        if x1 <= x0:
            return np.empty((0, 2), dtype=np.float32)
        for y in range(y0, y1, step):
            row = normal_map[y, x0:x1]
            xi = int(np.argmax(row))
            points.append((float(x0 + xi), float(y)))
    return np.asarray(points, dtype=np.float32)


def _intersect(horizontal: tuple[float, float], vertical: tuple[float, float]) -> np.ndarray:
    # horizontal: y = ah*x + bh; vertical: x = av*y + bv
    ah, bh = horizontal
    av, bv = vertical
    denom = 1.0 - ah * av
    if abs(denom) < 1e-6:
        raise ValueError("Detected border lines are nearly singular")
    y = (ah * bv + bh) / denom
    x = av * y + bv
    return np.array([x, y], dtype=np.float32)


def _is_convex_enclosing_quad(quad: np.ndarray, seed_quad: np.ndarray) -> bool:
    q = order_quad_points(quad)
    contour = q.reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour.astype(np.float32)):
        return False
    area = abs(float(cv2.contourArea(contour.astype(np.float32))))
    seed_area = abs(float(cv2.contourArea(order_quad_points(seed_quad).reshape(-1, 1, 2))))
    if area < seed_area * 0.75:
        return False
    # The physical cell border should enclose the GDS seed. Allow a few pixels
    # because a seed edge can legitimately touch a boundary.
    for p in order_quad_points(seed_quad):
        if cv2.pointPolygonTest(contour.astype(np.float32), (float(p[0]), float(p[1])), True) < -5.0:
            return False
    return True


def _save_debug_image(
    path: str | Path,
    source_image: np.ndarray,
    approx_quad: np.ndarray,
    detected_quad: np.ndarray | None,
    confidence: float,
    reason: str,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    canvas = source_image.copy()
    cv2.polylines(canvas, [order_quad_points(approx_quad).astype(np.int32)], True, (0, 165, 255), 3, cv2.LINE_AA)
    if detected_quad is not None:
        cv2.polylines(canvas, [order_quad_points(detected_quad).astype(np.int32)], True, (0, 255, 0), 4, cv2.LINE_AA)
    label = f"bound confidence={confidence:.3f} {reason}"[:180]
    cv2.putText(canvas, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(p), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])


def _refine_and_rectify_cell_core(
    image: np.ndarray,
    approximate_quad: np.ndarray,
    *,
    masks: Mapping[str, np.ndarray] | None = None,
    search_inward_fraction: float = 0.0,
    min_confidence: float = 0.22,
    output_scale: float = 1.0,
    shave_px: int = 0,
    debug_path: str | Path | None = None,
) -> BoundaryAlignmentResult:
    """Find the physical cell border around a GDS-derived seed and rectify it.

    The returned image is generated directly from ``image`` with one perspective
    warp.  On failure, ``success`` is false and callers should use their existing
    GDS crop as a safe fallback.
    """
    if image is None or image.size == 0:
        return BoundaryAlignmentResult(False, reason="empty source image")
    try:
        source_quad = order_quad_points(np.asarray(approximate_quad, dtype=np.float32))
        rectified, h_rect, seed = _build_rectified_search_image(image, source_quad)
        features = _feature_maps(rectified)
        rh, rw = rectified.shape[:2]
        sx0, sy0 = seed.min(axis=0)
        sx1, sy1 = seed.max(axis=0)
        seed_w = max(float(sx1 - sx0), 8.0)
        seed_h = max(float(sy1 - sy0), 8.0)

        inset_x = int(round(seed_w * float(search_inward_fraction)))
        inset_y = int(round(seed_h * float(search_inward_fraction)))
        outer_margin = max(3, int(round(min(seed_w, seed_h) * 0.015)))
        min_gap_x = max(2, int(round(seed_w * 0.015)))
        min_gap_y = max(2, int(round(seed_h * 0.015)))
        trans_radius = max(5, int(round(min(seed_w, seed_h) * 0.035)))

        # Use a broad central span at first. Top/bottom are less affected by the
        # wafer's dense vertical line texture, so solve them before left/right.
        x_span_lo = max(0, int(round(sx0 - 0.20 * seed_w)))
        x_span_hi = min(rw, int(round(sx1 + 0.20 * seed_w)))
        top_profile = _side_profile(features, 1, x_span_lo, x_span_hi, True, trans_radius)
        bottom_profile = _side_profile(features, 1, x_span_lo, x_span_hi, False, trans_radius)
        top_pos, top_conf, top_diag = _choose_side_position(
            top_profile,
            outer_margin,
            int(round(sy0)) + inset_y - min_gap_y,
            sy0,
            -1,
        )
        bottom_pos, bottom_conf, bottom_diag = _choose_side_position(
            bottom_profile,
            int(round(sy1)) - inset_y + min_gap_y,
            rh - outer_margin,
            sy1,
            +1,
        )
        if bottom_pos <= top_pos + max(10.0, seed_h * 0.5):
            raise ValueError("top/bottom border candidates do not enclose the seed")

        y_span_lo = max(0, int(round(top_pos + 0.04 * (bottom_pos - top_pos))))
        y_span_hi = min(rh, int(round(bottom_pos - 0.04 * (bottom_pos - top_pos))))
        left_profile = _side_profile(features, 0, y_span_lo, y_span_hi, True, trans_radius)
        right_profile = _side_profile(features, 0, y_span_lo, y_span_hi, False, trans_radius)
        left_pos, left_conf, left_diag = _choose_side_position(
            left_profile,
            outer_margin,
            int(round(sx0)) + inset_x - min_gap_x,
            sx0,
            -1,
        )
        right_pos, right_conf, right_diag = _choose_side_position(
            right_profile,
            int(round(sx1)) - inset_x + min_gap_x,
            rw - outer_margin,
            sx1,
            +1,
        )
        if right_pos <= left_pos + max(10.0, seed_w * 0.5):
            raise ValueError("left/right border candidates do not enclose the seed")

        # Refine each side into a line, rather than assuming perfect axis alignment.
        normal_x = features["gx"] + 1.3 * np.abs(cv2.Sobel(features["energy"], cv2.CV_32F, 1, 0, ksize=3))
        normal_y = features["gy"] + 1.3 * np.abs(cv2.Sobel(features["energy"], cv2.CV_32F, 0, 1, ksize=3))
        refine_r = max(5, int(round(min(seed_w, seed_h) * 0.035)))
        horizontal_span_lo = max(0, int(round(left_pos + 0.02 * (right_pos - left_pos))))
        horizontal_span_hi = min(rw, int(round(right_pos - 0.02 * (right_pos - left_pos))))
        vertical_span_lo = max(0, int(round(top_pos + 0.02 * (bottom_pos - top_pos))))
        vertical_span_hi = min(rh, int(round(bottom_pos - 0.02 * (bottom_pos - top_pos))))

        top_pts = _collect_line_points(normal_y, top_pos, "horizontal", horizontal_span_lo, horizontal_span_hi, refine_r)
        bottom_pts = _collect_line_points(normal_y, bottom_pos, "horizontal", horizontal_span_lo, horizontal_span_hi, refine_r)
        left_pts = _collect_line_points(normal_x, left_pos, "vertical", vertical_span_lo, vertical_span_hi, refine_r)
        right_pts = _collect_line_points(normal_x, right_pos, "vertical", vertical_span_lo, vertical_span_hi, refine_r)
        top_a, top_b, top_rms = _robust_line_fit(top_pts, "horizontal")
        bottom_a, bottom_b, bottom_rms = _robust_line_fit(bottom_pts, "horizontal")
        left_a, left_b, left_rms = _robust_line_fit(left_pts, "vertical")
        right_a, right_b, right_rms = _robust_line_fit(right_pts, "vertical")

        corners_rect = np.vstack(
            [
                _intersect((top_a, top_b), (left_a, left_b)),
                _intersect((top_a, top_b), (right_a, right_b)),
                _intersect((bottom_a, bottom_b), (right_a, right_b)),
                _intersect((bottom_a, bottom_b), (left_a, left_b)),
            ]
        ).astype(np.float32)
        corners_rect = order_quad_points(corners_rect)
        if not _is_convex_enclosing_quad(corners_rect, seed):
            raise ValueError("detected border quadrilateral does not enclose the GDS seed")

        # Map detected borders back into original local-stitch coordinates.
        h_inv = np.linalg.inv(h_rect)
        detected_source = cv2.perspectiveTransform(corners_rect.reshape(1, 4, 2), h_inv.astype(np.float32))[0]
        detected_source = order_quad_points(detected_source)
        out_w0, out_h0 = _quad_side_lengths(detected_source)
        scale = max(0.25, min(float(output_scale), 4.0))
        out_w = max(16, int(round(out_w0 * scale)))
        out_h = max(16, int(round(out_h0 * scale)))
        max_side = 20000
        if out_w > max_side or out_h > max_side:
            safe = min(max_side / float(out_w), max_side / float(out_h))
            out_w = max(16, int(round(out_w * safe)))
            out_h = max(16, int(round(out_h * safe)))

        shave = max(0, int(shave_px))
        if 2 * shave >= out_w - 4 or 2 * shave >= out_h - 4:
            shave = 0
        final_w = out_w - 2 * shave
        final_h = out_h - 2 * shave
        dst = np.array(
            [
                [-float(shave), -float(shave)],
                [out_w - 1.0 - shave, -float(shave)],
                [out_w - 1.0 - shave, out_h - 1.0 - shave],
                [-float(shave), out_h - 1.0 - shave],
            ],
            dtype=np.float32,
        )
        h_final = cv2.getPerspectiveTransform(detected_source, dst)
        cropped = cv2.warpPerspective(
            image,
            h_final,
            (final_w, final_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REPLICATE,
        )
        warped_masks: dict[str, np.ndarray] = {}
        for name, mask in (masks or {}).items():
            if mask is None or mask.size == 0:
                continue
            warped_masks[str(name)] = cv2.warpPerspective(
                mask,
                h_final,
                (final_w, final_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        side_conf = np.array([top_conf, right_conf, bottom_conf, left_conf], dtype=np.float32)
        rms = np.array([top_rms, right_rms, bottom_rms, left_rms], dtype=np.float32)
        rms_penalty = float(np.exp(-np.mean(rms) / max(2.0, 0.015 * min(seed_w, seed_h))))
        confidence = float(np.clip(0.82 * np.mean(side_conf) + 0.18 * rms_penalty, 0.0, 1.0))
        reason = "ok" if confidence >= float(min_confidence) else "confidence below threshold"

        metadata = {
            "success": bool(confidence >= float(min_confidence)),
            "confidence": confidence,
            "reason": reason,
            "approximate_corners_local_px": source_quad.tolist(),
            "detected_corners_local_px": detected_source.tolist(),
            "rectified_seed_corners_px": seed.tolist(),
            "rectified_detected_corners_px": corners_rect.tolist(),
            "homography_local_to_crop_3x3": h_final.tolist(),
            "output_size_px": [int(final_w), int(final_h)],
            "unshaved_output_size_px": [int(out_w), int(out_h)],
            "shave_px": int(shave),
            "output_scale": float(scale),
            "side_confidence": {
                "top": float(top_conf),
                "right": float(right_conf),
                "bottom": float(bottom_conf),
                "left": float(left_conf),
            },
            "line_fit_rms_px": {
                "top": float(top_rms),
                "right": float(right_rms),
                "bottom": float(bottom_rms),
                "left": float(left_rms),
            },
            "side_diagnostics": {
                "top": top_diag,
                "right": right_diag,
                "bottom": bottom_diag,
                "left": left_diag,
            },
        }
        if debug_path is not None:
            _save_debug_image(debug_path, image, source_quad, detected_source, confidence, reason)
        if confidence < float(min_confidence):
            return BoundaryAlignmentResult(
                False,
                detected_corners=detected_source,
                homography=h_final,
                confidence=confidence,
                reason=reason,
                metadata=metadata,
            )
        return BoundaryAlignmentResult(
            True,
            image=cropped,
            masks=warped_masks,
            detected_corners=detected_source,
            homography=h_final,
            confidence=confidence,
            reason=reason,
            metadata=metadata,
        )
    except Exception as exc:
        reason = str(exc)
        try:
            if debug_path is not None:
                _save_debug_image(debug_path, image, approximate_quad, None, 0.0, reason)
        except Exception:
            pass
        return BoundaryAlignmentResult(False, confidence=0.0, reason=reason, metadata={"success": False, "reason": reason})


def _warp_quad_from_source(
    image: np.ndarray,
    source_quad: np.ndarray,
    *,
    masks: Mapping[str, np.ndarray] | None = None,
    output_scale: float = 1.0,
    shave_px: int = 0,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, tuple[int, int]]:
    """Perspective-warp a source quadrilateral once, including matching masks."""
    quad = order_quad_points(np.asarray(source_quad, dtype=np.float32))
    width0, height0 = _quad_side_lengths(quad)
    scale = float(np.clip(float(output_scale), 0.25, 4.0))
    out_w = max(16, int(round(width0 * scale)))
    out_h = max(16, int(round(height0 * scale)))

    max_side = 20000
    if out_w > max_side or out_h > max_side:
        safe = min(max_side / float(out_w), max_side / float(out_h))
        out_w = max(16, int(round(out_w * safe)))
        out_h = max(16, int(round(out_h * safe)))

    shave = max(0, int(shave_px))
    if 2 * shave >= out_w - 4 or 2 * shave >= out_h - 4:
        shave = 0
    final_w = out_w - 2 * shave
    final_h = out_h - 2 * shave
    dst = np.array(
        [
            [-float(shave), -float(shave)],
            [out_w - 1.0 - shave, -float(shave)],
            [out_w - 1.0 - shave, out_h - 1.0 - shave],
            [-float(shave), out_h - 1.0 - shave],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(
        image,
        homography,
        (final_w, final_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )
    warped_masks: dict[str, np.ndarray] = {}
    for name, mask in (masks or {}).items():
        if mask is None or mask.size == 0:
            continue
        warped_masks[str(name)] = cv2.warpPerspective(
            mask,
            homography,
            (final_w, final_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return warped, warped_masks, homography, (final_w, final_h)


def refine_and_rectify_cell(
    image: np.ndarray,
    approximate_quad: np.ndarray,
    *,
    masks: Mapping[str, np.ndarray] | None = None,
    search_inward_fraction: float = 0.0,
    min_confidence: float = 0.22,
    output_scale: float = 1.0,
    shave_px: int = 0,
    debug_path: str | Path | None = None,
    detection_max_side: int = 2200,
) -> BoundaryAlignmentResult:
    """Detect on a bounded preview, then warp the higher-resolution source once.

    The v2 implementation ran every feature map at full native resolution.  That
    produced several hundred megabytes of temporary arrays per cell.  This
    wrapper keeps the final image sharp but limits the expensive detection pass.
    """
    if image is None or image.size == 0:
        return BoundaryAlignmentResult(False, reason="empty source image")

    max_side = max(0, int(detection_max_side or 0))
    ih, iw = image.shape[:2]
    detect_scale = 1.0
    if max_side > 0 and max(ih, iw) > max_side:
        detect_scale = max_side / float(max(ih, iw))

    if detect_scale >= 0.999:
        return _refine_and_rectify_cell_core(
            image,
            approximate_quad,
            masks=masks,
            search_inward_fraction=search_inward_fraction,
            min_confidence=min_confidence,
            output_scale=output_scale,
            shave_px=shave_px,
            debug_path=debug_path,
        )

    small_w = max(16, int(round(iw * detect_scale)))
    small_h = max(16, int(round(ih * detect_scale)))
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_quad = np.asarray(approximate_quad, dtype=np.float32) * float(detect_scale)
    core = _refine_and_rectify_cell_core(
        small,
        small_quad,
        masks=None,
        search_inward_fraction=search_inward_fraction,
        min_confidence=min_confidence,
        output_scale=1.0,
        shave_px=0,
        debug_path=None,
    )

    detected_full = None
    if core.detected_corners is not None:
        detected_full = np.asarray(core.detected_corners, dtype=np.float32) / float(detect_scale)

    metadata = dict(core.metadata or {})
    metadata["detection_preview_scale"] = float(detect_scale)
    metadata["detection_preview_size_px"] = [int(small_w), int(small_h)]
    metadata["approximate_corners_local_px"] = order_quad_points(approximate_quad).tolist()
    if detected_full is not None:
        metadata["detected_corners_local_px"] = order_quad_points(detected_full).tolist()

    if not core.success or detected_full is None:
        reason = core.reason or "physical border detection failed"
        if debug_path is not None:
            try:
                _save_debug_image(
                    debug_path,
                    image,
                    approximate_quad,
                    detected_full,
                    core.confidence,
                    reason,
                )
            except Exception:
                pass
        return BoundaryAlignmentResult(
            False,
            detected_corners=detected_full,
            confidence=float(core.confidence),
            reason=reason,
            metadata=metadata,
        )

    try:
        cropped, warped_masks, homography, final_size = _warp_quad_from_source(
            image,
            detected_full,
            masks=masks,
            output_scale=output_scale,
            shave_px=shave_px,
        )
    except Exception as exc:
        reason = f"final high-resolution warp failed: {exc}"
        return BoundaryAlignmentResult(
            False,
            detected_corners=detected_full,
            confidence=float(core.confidence),
            reason=reason,
            metadata=metadata,
        )

    metadata["success"] = True
    metadata["reason"] = "ok"
    metadata["homography_local_to_crop_3x3"] = homography.tolist()
    metadata["output_size_px"] = [int(final_size[0]), int(final_size[1])]
    metadata["output_scale"] = float(output_scale)
    metadata["shave_px"] = int(shave_px)
    if debug_path is not None:
        try:
            _save_debug_image(
                debug_path,
                image,
                approximate_quad,
                detected_full,
                core.confidence,
                "ok",
            )
        except Exception:
            pass
    return BoundaryAlignmentResult(
        True,
        image=cropped,
        masks=warped_masks,
        detected_corners=order_quad_points(detected_full),
        homography=homography,
        confidence=float(core.confidence),
        reason="ok",
        metadata=metadata,
    )


class _ScaledNormalizedTileCache:
    """Thread-safe LRU cache of downscaled, normalized source tiles.

    At the default 0.20 local scale, JPEGs are decoded at the camera codec's
    half-resolution DCT path and then resized to the same target dimensions.
    This avoids decoding millions of pixels that are immediately discarded.
    Use ``fast_jpeg_decode=False`` for byte-for-byte compatibility with v4.
    """

    def __init__(
        self,
        *,
        scale: float,
        max_items: int,
        target_luma: float | None,
        config: dict,
        shared_flatfield_model: Any,
        fast_jpeg_decode: bool = True,
    ) -> None:
        self.scale = float(scale)
        self.max_items = max(0, int(max_items))
        self.target_luma = target_luma
        self.config = config
        self.shared_flatfield_model = shared_flatfield_model
        self.fast_jpeg_decode = bool(fast_jpeg_decode)
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._inflight: dict[str, Future] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _read_scaled_jpeg(path: Path, target_w: int, target_h: int) -> np.ndarray | None:
        if path.suffix.lower() not in {".jpg", ".jpeg", ".jpe"}:
            return None
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_REDUCED_COLOR_2)
        except Exception:
            return None
        if img is None:
            return None
        return cv2.resize(img, (int(target_w), int(target_h)), interpolation=cv2.INTER_AREA)

    def _load(self, path: Path, tile_width: int, tile_height: int) -> np.ndarray:
        import illumination_stitching

        target_w = max(1, int(round(tile_width * self.scale)))
        target_h = max(1, int(round(tile_height * self.scale)))
        img = None
        if self.fast_jpeg_decode and self.scale <= 0.45:
            img = self._read_scaled_jpeg(path, target_w, target_h)
        if img is None:
            img = illumination_stitching.read_bgr(path)
            img = illumination_stitching.resize_if_needed(img, target_w, target_h)
        img = illumination_stitching.normalize_tile_bgr(
            img,
            target_luma=self.target_luma,
            illumination_enabled=bool(self.config.get("_illumination_enabled", True)),
            brightness_match_enabled=bool(self.config.get("_brightness_match_enabled", True)),
            illumination_strength=float(self.config.get("_illumination_strength", 1.0)),
            blur_sigma_frac=float(self.config.get("_illumination_blur_sigma_frac", 0.18)),
            brightness_match_strength=float(self.config.get("_brightness_match_strength", 0.65)),
            shared_flatfield_model=self.shared_flatfield_model,
        )
        return img

    def get(self, path: Path, tile_width: int, tile_height: int) -> np.ndarray:
        key = str(path)
        loader = False
        with self._lock:
            if self.max_items > 0 and key in self.cache:
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            future = self._inflight.get(key)
            if future is None:
                future = Future()
                self._inflight[key] = future
                loader = True

        if not loader:
            return future.result()

        try:
            value = self._load(path, tile_width, tile_height)
        except BaseException as exc:
            with self._lock:
                self._inflight.pop(key, None)
                future.set_exception(exc)
            raise

        with self._lock:
            if self.max_items > 0:
                self.cache[key] = value
                while len(self.cache) > self.max_items:
                    self.cache.popitem(last=False)
            self._inflight.pop(key, None)
            future.set_result(value)
        return value


class _FeatherWeightCache:
    """Cache the small set of feather masks reused by every cell."""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, ...], np.ndarray] = {}
        self._lock = threading.Lock()

    def get(
        self,
        illumination_stitching: Any,
        height: int,
        width: int,
        overlap_x: int,
        overlap_y: int,
        *,
        has_left: bool,
        has_right: bool,
        has_top: bool,
        has_bottom: bool,
    ) -> np.ndarray:
        key = (
            int(height), int(width), int(overlap_x), int(overlap_y),
            int(has_left), int(has_right), int(has_top), int(has_bottom),
        )
        with self._lock:
            value = self._cache.get(key)
        if value is not None:
            return value
        value = illumination_stitching.make_feather_weight(
            height,
            width,
            overlap_x,
            overlap_y,
            has_left=has_left,
            has_right=has_right,
            has_top=has_top,
            has_bottom=has_bottom,
        )
        with self._lock:
            existing = self._cache.setdefault(key, value)
        return existing

def _normalize_scaled_accumulator(acc: np.ndarray, weights: np.ndarray) -> np.ndarray:
    h, w = weights.shape
    out = np.empty((h, w, 3), dtype=np.uint8)
    # Small chunks prevent the exact 120 MiB temporary that crashed the v2 run.
    chunk_rows = max(16, min(128, int(8_000_000 / max(w, 1))))
    for y0 in range(0, h, chunk_rows):
        y1 = min(h, y0 + chunk_rows)
        ws = weights[y0:y1]
        safe = np.maximum(ws, 1e-6)
        chunk = acc[y0:y1] / safe[:, :, None]
        chunk = np.clip(chunk, 0, 255).astype(np.uint8)
        chunk[ws <= 1e-6] = 0
        out[y0:y1] = chunk
    return out


def stitch_scaled_local_canvas(
    *,
    folder: str | Path,
    tile_ext: str,
    overlapping_tiles: list[tuple[int, int, int, int, int, int]],
    local_origin: tuple[int, int],
    local_size: tuple[int, int],
    config: dict,
    scale: float,
    tile_cache: _ScaledNormalizedTileCache,
    weight_cache: _FeatherWeightCache,
    excluded_tile_names: set[str] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Stitch one local region directly at a bounded fraction of native size."""
    import illumination_stitching

    folder = Path(folder)
    scale = float(np.clip(scale, 0.05, 1.0))
    local_x1, local_y1 = map(int, local_origin)
    local_w_native, local_h_native = map(int, local_size)
    out_w = max(1, int(round(local_w_native * scale)))
    out_h = max(1, int(round(local_h_native * scale)))
    acc = np.zeros((out_h, out_w, 3), dtype=np.float32)
    wacc = np.zeros((out_h, out_w), dtype=np.float32)
    coverage = np.zeros((out_h, out_w), dtype=np.uint8)

    tw = int(config["tile_width"])
    th = int(config["tile_height"])
    step_x = tw * (1.0 - float(config["overlap_x_percent"]) / 100.0)
    step_y = th * (1.0 - float(config["overlap_y_percent"]) / 100.0)
    overlap_x = max(1, int(round((tw - step_x) * scale)))
    overlap_y = max(1, int(round((th - step_y) * scale)))
    tile_w_scaled = max(1, int(round(tw * scale)))
    tile_h_scaled = max(1, int(round(th * scale)))
    keys = {(int(t[0]), int(t[1])) for t in overlapping_tiles}
    exclusions = set(excluded_tile_names or set())

    for col, row, tile_x1, tile_y1, tile_x2, tile_y2 in overlapping_tiles:
        tile_name = f"tile_x{int(col):03d}_y{int(row):03d}{tile_ext}"
        if tile_name in exclusions:
            continue
        path = folder / tile_name
        if not path.exists():
            continue

        ix1 = max(local_x1, int(tile_x1))
        iy1 = max(local_y1, int(tile_y1))
        ix2 = min(local_x1 + local_w_native, int(tile_x2))
        iy2 = min(local_y1 + local_h_native, int(tile_y2))
        if ix2 <= ix1 or iy2 <= iy1:
            continue

        tile_img = tile_cache.get(path, tw, th)
        full_weight = weight_cache.get(
            illumination_stitching,
            tile_h_scaled,
            tile_w_scaled,
            overlap_x,
            overlap_y,
            has_left=(int(col) - 1, int(row)) in keys,
            has_right=(int(col) + 1, int(row)) in keys,
            has_top=(int(col), int(row) - 1) in keys,
            has_bottom=(int(col), int(row) + 1) in keys,
        )

        dx1 = int(round((ix1 - local_x1) * scale))
        dy1 = int(round((iy1 - local_y1) * scale))
        dx2 = int(round((ix2 - local_x1) * scale))
        dy2 = int(round((iy2 - local_y1) * scale))
        dx1, dy1 = max(0, dx1), max(0, dy1)
        dx2, dy2 = min(out_w, dx2), min(out_h, dy2)
        if dx2 <= dx1 or dy2 <= dy1:
            continue

        sx1 = int(round((ix1 - int(tile_x1)) * scale))
        sy1 = int(round((iy1 - int(tile_y1)) * scale))
        sx2 = int(round((ix2 - int(tile_x1)) * scale))
        sy2 = int(round((iy2 - int(tile_y1)) * scale))
        sx1, sy1 = max(0, sx1), max(0, sy1)
        sx2, sy2 = min(tile_img.shape[1], sx2), min(tile_img.shape[0], sy2)
        if sx2 <= sx1 or sy2 <= sy1:
            continue

        target_size = (dx2 - dx1, dy2 - dy1)
        tile_crop = tile_img[sy1:sy2, sx1:sx2]
        weight_crop = full_weight[sy1:sy2, sx1:sx2]
        if tile_crop.shape[1] != target_size[0] or tile_crop.shape[0] != target_size[1]:
            tile_crop = cv2.resize(tile_crop, target_size, interpolation=cv2.INTER_AREA)
            weight_crop = cv2.resize(weight_crop, target_size, interpolation=cv2.INTER_LINEAR)

        acc[dy1:dy2, dx1:dx2] += tile_crop.astype(np.float32) * weight_crop[:, :, None]
        wacc[dy1:dy2, dx1:dx2] += weight_crop
        # A pixel can be covered by at most four grid neighbors, so uint8
        # addition is exact here and avoids two full temporary arrays per tile.
        coverage[dy1:dy2, dx1:dx2] += (weight_crop > 1e-4).astype(np.uint8)

    canvas = _normalize_scaled_accumulator(acc, wacc)
    seam = np.where(coverage > 1, 255, 0).astype(np.uint8)
    if seam.size:
        seam = cv2.dilate(seam, np.ones((3, 3), dtype=np.uint8), iterations=1)
    masks = {
        "seam_mask": seam,
        "coverage_count": coverage,
    }
    return canvas, masks


def _fallback_gds_rectification(
    image: np.ndarray,
    approximate_quad: np.ndarray,
    masks: Mapping[str, np.ndarray],
    *,
    output_scale: float,
    shave_px: int,
) -> BoundaryAlignmentResult:
    try:
        crop, warped_masks, h, size = _warp_quad_from_source(
            image,
            approximate_quad,
            masks=masks,
            output_scale=output_scale,
            shave_px=shave_px,
        )
        return BoundaryAlignmentResult(
            True,
            image=crop,
            masks=warped_masks,
            detected_corners=order_quad_points(approximate_quad),
            homography=h,
            confidence=0.0,
            reason="GDS fallback",
            metadata={"output_size_px": [int(size[0]), int(size[1])]},
        )
    except Exception as exc:
        return BoundaryAlignmentResult(False, reason=f"GDS fallback failed: {exc}")


def extract_bound_crops(
    *,
    folder: str | Path,
    tile_ext: str,
    cells: list[dict],
    transformer: Any,
    config_run: dict,
    args: Any,
    out_stem: str,
    out_dir: Path,
    preview_dir: Path,
    analysis_dir: Path,
    mask_dir: Path,
    meta_dir: Path,
    save_crop_artifacts: Callable[..., None],
) -> tuple[int, list[dict]]:
    """Parallel bounded extraction with the same per-cell geometry as v4.

    Cell results are independent, so two cells are processed concurrently while
    sharing a single-flight tile cache. The output records are sorted back into
    their original GDS order, making the cell index deterministic.
    """
    import illumination_stitching

    folder = Path(folder)
    tw = int(config_run["tile_width"])
    th = int(config_run["tile_height"])
    tile_cols = int(config_run["tile_cols"])
    tile_rows = int(config_run["tile_rows"])
    step_x = tw * (1.0 - float(config_run["overlap_x_percent"]) / 100.0)
    step_y = th * (1.0 - float(config_run["overlap_y_percent"]) / 100.0)

    target_luma = config_run.get("_illumination_target_luma")
    if target_luma is None:
        target_luma = illumination_stitching.estimate_global_luma_for_folder(
            folder,
            max_samples=int(getattr(args, "illumination_brightness_samples", 250)),
        )
        config_run["_illumination_target_luma"] = float(target_luma)
    shared_model = illumination_stitching.get_or_build_shared_flatfield_model(
        folder, config_run, verbose=True
    )

    requested_scale = float(np.clip(getattr(args, "bound_native_scale", 0.20), 0.05, 1.0))
    max_megapixels = max(1.0, float(getattr(args, "bound_max_local_megapixels", 8.0)))
    detect_max_side = max(400, int(getattr(args, "bound_detect_max_side", 1600)))
    cache_items = max(0, int(getattr(args, "bound_tile_cache_size", 64)))
    workers = max(1, int(getattr(args, "bound_workers", 2)))
    fast_jpeg = not bool(getattr(args, "bound_exact_jpeg_decode", False))
    total = max(1, len(cells))
    start_time = time.time()

    tile_caches: dict[float, _ScaledNormalizedTileCache] = {}
    weight_caches: dict[float, _FeatherWeightCache] = {}
    jobs: list[dict[str, Any]] = []

    def candidate_range(start: int, end: int, tile_size: int, step: float, count: int) -> range:
        lo = max(1, int(math.floor((float(start) - tile_size) / step)) + 1)
        hi = min(count, int(math.ceil(float(end) / step)) + 1)
        return range(lo, hi + 1)

    # Geometry and tile-overlap discovery are done once, before worker threads.
    for idx, cell in enumerate(cells, start=1):
        row = int(cell["row"])
        col = int(cell["col"])
        min_x, min_y, max_x, max_y = cell["bbox"]
        gds_corners = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
        pts_canvas = np.array(
            [transformer.gds_to_canvas(gx, gy) for gx, gy in gds_corners],
            dtype=np.float64,
        )
        side_lengths = [
            float(np.linalg.norm(pts_canvas[(i + 1) % 4] - pts_canvas[i]))
            for i in range(4)
        ]
        long_side = max(side_lengths + [1.0])
        pad = max(
            int(getattr(args, "bound_pad", 250)),
            int(round(long_side * float(getattr(args, "bound_expand", 0.18)))),
        )
        cx_min, cy_min = np.min(pts_canvas, axis=0)
        cx_max, cy_max = np.max(pts_canvas, axis=0)
        x1 = int(np.floor(cx_min)) - pad
        y1 = int(np.floor(cy_min)) - pad
        x2 = int(np.ceil(cx_max)) + pad
        y2 = int(np.ceil(cy_max)) + pad
        native_w, native_h = x2 - x1, y2 - y1
        if native_w <= 0 or native_h <= 0:
            continue

        effective_scale = requested_scale
        projected_pixels = native_w * native_h * effective_scale * effective_scale
        max_pixels = max_megapixels * 1_000_000.0
        if projected_pixels > max_pixels:
            effective_scale = math.sqrt(max_pixels / float(native_w * native_h))
            effective_scale = float(np.clip(effective_scale, 0.05, requested_scale))

        overlapping_tiles: list[tuple[int, int, int, int, int, int]] = []
        for c_col in candidate_range(x1, x2, tw, step_x, tile_cols):
            tx1 = int(round((c_col - 1) * step_x))
            tx2 = tx1 + tw
            for r_row in candidate_range(y1, y2, th, step_y, tile_rows):
                tile_key = f"tile_x{c_col:03d}_y{r_row:03d}{tile_ext}"
                if tile_key in transformer.exclusions:
                    continue
                ty1 = int(round((r_row - 1) * step_y))
                ty2 = ty1 + th
                if max(x1, tx1) < min(x2, tx2) and max(y1, ty1) < min(y2, ty2):
                    overlapping_tiles.append((c_col, r_row, tx1, ty1, tx2, ty2))
        if not overlapping_tiles:
            continue

        cache_key = round(float(effective_scale), 4)
        if cache_key not in tile_caches:
            tile_caches[cache_key] = _ScaledNormalizedTileCache(
                scale=effective_scale,
                max_items=cache_items,
                target_luma=target_luma,
                config=config_run,
                shared_flatfield_model=shared_model,
                fast_jpeg_decode=fast_jpeg,
            )
            weight_caches[cache_key] = _FeatherWeightCache()
        jobs.append({
            "idx": idx, "row": row, "col": col, "cell": cell,
            "gds_corners": gds_corners, "pts_canvas": pts_canvas,
            "x1": x1, "y1": y1, "native_w": native_w, "native_h": native_h,
            "pad": pad, "effective_scale": effective_scale,
            "cache_key": cache_key, "overlapping_tiles": overlapping_tiles,
        })

    workers = min(workers, max(1, len(jobs)))
    print(
        f"[{out_stem} Bound] parallel fast path: workers={workers}, "
        f"OpenCV threads={max(1, int(getattr(args, 'bound_opencv_threads', 1)))}, "
        f"JPEG decode={'reduced-2' if fast_jpeg else 'exact'}, "
        f"pad=max({int(getattr(args, 'bound_pad', 250))} px, "
        f"{float(getattr(args, 'bound_expand', 0.18)):.3f} x cell side), "
        f"scale={requested_scale:.3f}, cap={max_megapixels:.1f} MP, cache={cache_items}"
    )

    def process_job(job: dict[str, Any]) -> tuple[int, dict | None, str]:
        idx, row, col = int(job["idx"]), int(job["row"]), int(job["col"])
        effective_scale = float(job["effective_scale"])
        local_canvas, local_masks = stitch_scaled_local_canvas(
            folder=folder,
            tile_ext=tile_ext,
            overlapping_tiles=job["overlapping_tiles"],
            local_origin=(job["x1"], job["y1"]),
            local_size=(job["native_w"], job["native_h"]),
            config=config_run,
            scale=effective_scale,
            tile_cache=tile_caches[job["cache_key"]],
            weight_cache=weight_caches[job["cache_key"]],
            excluded_tile_names=transformer.exclusions,
        )
        approximate_local_quad = (
            job["pts_canvas"].astype(np.float32)
            - np.array([job["x1"], job["y1"]], dtype=np.float32)
        ) * effective_scale
        cell_stem = f"{out_stem}_cell_{row}-{col}"
        debug_path = None
        if bool(getattr(args, "bound_debug", False)):
            debug_path = out_dir / "boundary_debug" / f"{cell_stem}_boundary.jpg"
        output_scale = float(getattr(args, "bound_output_scale", 1.0))
        shave_local = int(round(int(getattr(args, "shave", 0)) * effective_scale * output_scale))
        result = refine_and_rectify_cell(
            local_canvas,
            approximate_local_quad,
            masks=local_masks,
            search_inward_fraction=float(getattr(args, "bound_search_inward_fraction", 0.08)),
            min_confidence=float(getattr(args, "bound_min_confidence", 0.22)),
            output_scale=output_scale,
            shave_px=shave_local,
            debug_path=debug_path,
            detection_max_side=detect_max_side,
        )
        detected = bool(result.success)
        boundary_reason = result.reason
        boundary_confidence = float(result.confidence)
        if not result.success:
            fallback = _fallback_gds_rectification(
                local_canvas,
                approximate_local_quad,
                local_masks,
                output_scale=output_scale,
                shave_px=shave_local,
            )
            if not fallback.success:
                return idx, None, f"skip {row}-{col}: {result.reason}; {fallback.reason}"
            result = fallback

        cell_crop = result.image
        if cell_crop is None or cell_crop.size == 0:
            return idx, None, f"skip {row}-{col}: empty crop"
        final_masks = result.masks
        final_bounds = (0, 0, int(cell_crop.shape[1]), int(cell_crop.shape[0]))
        min_x, min_y, max_x, max_y = job["cell"]["bbox"]
        metadata = {
            "wafer_id": out_stem,
            "cell_row": row,
            "cell_col": col,
            "cell_stem": cell_stem,
            "crop_source": "scaled_native_physical_boundary" if detected else "scaled_native_gds_fallback",
            "gds_bbox_um": [float(min_x), float(min_y), float(max_x), float(max_y)],
            "gds_corners_um": [[float(a), float(b)] for a, b in job["gds_corners"]],
            "canvas_corners_px_fullres": job["pts_canvas"].tolist(),
            "local_origin_px_fullres": [int(job["x1"]), int(job["y1"])],
            "local_size_px_fullres": [int(job["native_w"]), int(job["native_h"])],
            "local_stitch_scale": effective_scale,
            "requested_local_stitch_scale": requested_scale,
            "local_stitch_size_px": [int(local_canvas.shape[1]), int(local_canvas.shape[0])],
            "local_stitch_max_megapixels": max_megapixels,
            "search_padding_px_fullres": int(job["pad"]),
            "overlapping_tiles": [list(map(int, t)) for t in job["overlapping_tiles"]],
            "crop_size_px": [int(cell_crop.shape[1]), int(cell_crop.shape[0])],
            "boundary_alignment": {
                **dict(result.metadata or {}),
                "enabled": True,
                "applied": detected,
                "confidence": boundary_confidence,
                "fallback_reason": "" if detected else boundary_reason,
            },
            "performance": {
                "workers": workers,
                "fast_jpeg_decode": fast_jpeg,
                "helper_version": BOUND_HELPER_VERSION,
            },
            "analysis_png": str((analysis_dir / f"{cell_stem}.png").as_posix()),
            "legacy_jpg": str((out_dir / f"{cell_stem}.jpg").as_posix()),
            "seam_mask": str((mask_dir / f"{cell_stem}_seam_mask.png").as_posix()),
        }
        save_crop_artifacts(
            out_dir=out_dir,
            preview_dir=preview_dir,
            analysis_dir=analysis_dir,
            mask_dir=mask_dir,
            meta_dir=meta_dir,
            out_stem=out_stem,
            row=row,
            col=col,
            cell_crop=cell_crop,
            rotated_local_masks=final_masks,
            crop_bounds=final_bounds,
            metadata=metadata,
            preview_width=int(getattr(args, "preview_width", 2000)),
        )
        return idx, metadata, ("detected" if detected else "fallback") + f" {row}-{col}"

    completed = 0
    results_by_idx: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="h2p-bound") as pool:
        futures = [pool.submit(process_job, job) for job in jobs]
        for future in as_completed(futures):
            completed += 1
            try:
                idx, metadata, status = future.result()
            except Exception as exc:
                status = f"error: {exc}"
                metadata = None
                idx = -1
            if metadata is not None:
                results_by_idx[int(idx)] = metadata
            elapsed = time.time() - start_time
            eta = elapsed * (len(jobs) - completed) / max(completed, 1)
            avg_s = elapsed / max(completed, 1)
            sys.stdout.write(
                f"\r[{out_stem} Bound Crops] {completed}/{len(jobs)} | saved {len(results_by_idx)} | "
                f"{status} | wall/cell={avg_s:4.1f}s | ETA {eta/60:4.1f}m\033[K"
            )
            sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    records = [results_by_idx[i] for i in sorted(results_by_idx)]
    return len(records), records

