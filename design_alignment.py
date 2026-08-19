"""Alignment detection/review implementation for the single supported H2P design. This module is called directly by the extraction pipeline; it does not patch other modules."""
from __future__ import annotations
import copy
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import cv2
import numpy as np
ALIGNMENT_VERSION = 'design-direct-v23-production-score-auto-refinement-2026-08-19'

@dataclass(frozen=True)
class DesignGeometry:
    path: Path
    design_bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    radius: float
    boundary: np.ndarray
    marker_polygons: list[np.ndarray]
    marker_centers: dict[str, tuple[float, float]]
    cells: list[dict[str, Any]]

class _ArrayReviewImage:
    """Minimal PIL-like view over the production downscaled BGR stitch.

    LargeWaferTester only needs ``size``, ``reduce`` and ``crop``.  Keeping the
    shared NumPy canvas in memory avoids rebuilding a second stitch and, more
    importantly, guarantees that review clicks use the exact coordinate frame
    consumed by the downstream alignment GUI.
    """

    def __init__(self, image_bgr: np.ndarray) -> None:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError('Downscaled review canvas is empty')
        self.image_bgr = image_bgr
        self.height, self.width = image_bgr.shape[:2]
        self.size = (self.width, self.height)

    def reduce(self, factor: int):
        from PIL import Image
        factor = max(1, int(factor))
        out_w = max(1, self.width // factor)
        out_h = max(1, self.height // factor)
        resized = cv2.resize(self.image_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

    def crop(self, box: tuple[int, int, int, int]):
        from PIL import Image
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        out_w = max(1, x2 - x1)
        out_h = max(1, y2 - y1)
        output = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        src_x1, src_y1 = (max(0, x1), max(0, y1))
        src_x2, src_y2 = (min(self.width, x2), min(self.height, y2))
        if src_x2 > src_x1 and src_y2 > src_y1:
            dst_x1, dst_y1 = (src_x1 - x1, src_y1 - y1)
            dst_x2 = dst_x1 + (src_x2 - src_x1)
            dst_y2 = dst_y1 + (src_y2 - src_y1)
            output[dst_y1:dst_y2, dst_x1:dst_x2] = self.image_bgr[src_y1:src_y2, src_x1:src_x2]
        return Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

def _normalized_wafer_view(image: np.ndarray, center_px: tuple[float, float], radius_px: float, design_bbox: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Warp the stitched wafer into the GDS bounding-box coordinate frame.

    Returns ``(normalized_image, normalized_to_source_2x3)``.
    """
    x0, y0, x1, y1 = design_bbox
    gds_half_x = 0.5 * (x1 - x0)
    gds_half_y = 0.5 * (y1 - y0)
    gds_radius = 50000.0
    source_half_x = float(radius_px) * gds_half_x / gds_radius
    source_half_y = float(radius_px) * gds_half_y / gds_radius
    source_side_x = max(2.0, 2.0 * source_half_x)
    source_side_y = max(2.0, 2.0 * source_half_y)
    output_w = max(1200, min(7000, int(round(source_side_x))))
    output_h = max(1200, min(7000, int(round(source_side_y))))
    scale_x = source_side_x / output_w
    scale_y = source_side_y / output_h
    source_x0 = float(center_px[0]) - source_half_x
    source_y0 = float(center_px[1]) - source_half_y
    normalized_to_source = np.asarray([[scale_x, 0.0, source_x0], [0.0, scale_y, source_y0]], dtype=np.float64)
    source_to_normalized = cv2.invertAffineTransform(normalized_to_source)
    normalized = cv2.warpAffine(image, source_to_normalized, (output_w, output_h), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return (normalized, normalized_to_source)

def _extract_detection_anchor(detection: Any) -> tuple[float, float]:
    anchor = getattr(detection, 'anchor_original_px', None)
    if anchor is None:
        anchor = getattr(detection, 'anchor_px', None)
    if anchor is None:
        raise RuntimeError('Compact-marker detection has no anchor coordinates')
    return (float(anchor[0]), float(anchor[1]))

def _run_compact_detector_details(normalized: np.ndarray, geometry: DesignGeometry) -> dict[str, dict[str, Any]]:
    """Return validated compact detections plus coordinate-scale metadata."""
    import future_alignment
    detections = None
    load_info = None
    if hasattr(future_alignment, 'detect_future_alignment_on_canvas'):
        result = future_alignment.detect_future_alignment_on_canvas(normalized, geometry.path)
        detections = result[1]
        if len(result) >= 6:
            load_info = result[5]
    elif hasattr(future_alignment, 'detect_image_inner_markers_array'):
        if hasattr(future_alignment, 'detect_gds_inner_markers'):
            gds_result = future_alignment.detect_gds_inner_markers(geometry.path)
            if isinstance(gds_result, tuple) and len(gds_result) >= 3:
                gds_markers, _polygons, design_bbox = gds_result[:3]
            else:
                raise RuntimeError('Unsupported detect_gds_inner_markers return value')
        else:
            raise RuntimeError('future_alignment lacks GDS inner-marker parsing')
        result = future_alignment.detect_image_inner_markers_array(normalized, gds_markers, design_bbox)
        detections = result[0]
        if len(result) >= 6:
            load_info = result[5]
    else:
        raise RuntimeError('future_alignment.py does not export detect_future_alignment_on_canvas or detect_image_inner_markers_array')
    scale_x = float(getattr(load_info, 'scale_x', 1.0) or 1.0)
    scale_y = float(getattr(load_info, 'scale_y', 1.0) or 1.0)
    by_side: dict[str, dict[str, Any]] = {}
    for detection in detections or []:
        side = str(getattr(detection, 'side', '')).lower()
        if side not in ('left', 'right'):
            continue
        by_side[side] = {'detection': detection, 'anchor': _extract_detection_anchor(detection), 'work_to_input_x': 1.0 / max(scale_x, 1e-12), 'work_to_input_y': 1.0 / max(scale_y, 1e-12)}
    if set(by_side) != {'left', 'right'}:
        raise RuntimeError(f"compact detector did not return a validated left/right pair (found: {sorted(by_side) or 'none'})")
    return by_side

def _fitted_detection_geometry(detail: dict[str, Any], normalized_to_review: np.ndarray, factor: float) -> tuple[tuple[float, float], dict[str, Any], dict[str, Any]]:
    """Return the fitted lattice anchor and review boxes.

    Raw connected-component centroids are useful for validating that squares
    exist, but bubbles, residue, and incomplete outlines bias those centroids.
    Registration and displayed boxes must instead use the detector's fitted
    3 x 4 lattice (``predicted_square_centers_px``).
    """
    detection = detail['detection']
    sx = float(detail.get('work_to_input_x', 1.0))
    sy = float(detail.get('work_to_input_y', 1.0))
    predicted = getattr(detection, 'predicted_square_centers_px', None) or []
    if len(predicted) != 12:
        raise RuntimeError(f'Detector returned {len(predicted)} fitted square centers, expected 12')
    centers_input = np.asarray([[float(point[0]) * sx, float(point[1]) * sy] for point in predicted], dtype=np.float64)
    pitch_input = float(getattr(detection, 'pitch_original_px', 0.0) or 0.0)
    if pitch_input <= 0.0:
        pitch_input = float(getattr(detection, 'pitch_px', 0.0) or 0.0) * 0.5 * (sx + sy)
    angle_deg = float(getattr(detection, 'angle_deg', 0.0) or 0.0)
    angle = math.radians(angle_deg)
    ex = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
    ey = np.asarray([-math.sin(angle), math.cos(angle)], dtype=np.float64)
    half_square = 0.25 * pitch_input
    order = np.argsort(centers_input[:, 1])
    ordered = centers_input[order]
    rows: list[np.ndarray] = []
    for start in range(0, 12, 3):
        row = ordered[start:start + 3]
        rows.append(row[np.argsort(row[:, 0])])
    ordered = np.vstack(rows)
    boxes: dict[Any, list[tuple[float, float]]] = {}
    squares: dict[Any, tuple[float, float]] = {}
    for index, center in enumerate(ordered):
        corners_input = [center - half_square * ex - half_square * ey, center + half_square * ex - half_square * ey, center + half_square * ex + half_square * ey, center - half_square * ex + half_square * ey]
        corners_global: list[tuple[float, float]] = []
        for corner in corners_input:
            review = normalized_to_review @ np.asarray([corner[0], corner[1], 1.0], dtype=np.float64)
            corners_global.append((float(review[0] * factor), float(review[1] * factor)))
        review_center = normalized_to_review @ np.asarray([center[0], center[1], 1.0], dtype=np.float64)
        key = (index // 3, index % 3)
        boxes[key] = corners_global
        squares[key] = (float(review_center[0] * factor), float(review_center[1] * factor))
    all_corners = [point for corners in boxes.values() for point in corners]
    if all_corners:
        xs = [point[0] for point in all_corners]
        ys = [point[1] for point in all_corners]
        padding = max(8.0, 0.55 * pitch_input * factor)
        boxes['area', 0] = [(min(xs) - padding, min(ys) - padding), (max(xs) + padding, min(ys) - padding), (max(xs) + padding, max(ys) + padding), (min(xs) - padding, max(ys) + padding)]
    anchor_input = np.mean(centers_input, axis=0)
    anchor_review = normalized_to_review @ np.asarray([anchor_input[0], anchor_input[1], 1.0], dtype=np.float64)
    diagnostics = {'pitch_input_px': float(pitch_input), 'angle_deg': float(angle_deg), 'raw_match_count': int(len(getattr(detection, 'matched_square_centers_px', None) or [])), 'fit_error_px': float(getattr(detection, 'mean_error_px', 0.0) or 0.0)}
    return ((float(anchor_review[0]), float(anchor_review[1])), {'boxes': boxes, 'squares': squares}, diagnostics)
_TILE_NAME_RE = re.compile('tile_x(?P<x>\\d+)_y(?P<y>\\d+)(?:\\(\\d+\\))?\\.(?:jpg|jpeg|png|tif|tiff)$', re.IGNORECASE)

def _index_raw_tiles(folder: str | Path) -> dict[tuple[int, int], Path]:
    root = Path(folder)
    if not root.exists() or not root.is_dir():
        return {}
    indexed: dict[tuple[int, int], Path] = {}
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        match = _TILE_NAME_RE.match(path.name)
        if not match:
            continue
        key = (int(match.group('x')), int(match.group('y')))
        current = indexed.get(key)
        if current is None or ('(' in current.stem and '(' not in path.stem):
            indexed[key] = path
    return indexed

def _tile_edge_map(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    p90 = float(np.percentile(magnitude, 90.0))
    if p90 > 1e-09:
        magnitude = np.clip(magnitude / (2.0 * p90), 0.0, 1.0)
    return magnitude.astype(np.float32)

def _match_tile_origin_on_stitch(ds_canvas: np.ndarray, raw_tile: np.ndarray, expected_size_ds: tuple[int, int], approximate_origin: tuple[float, float]) -> tuple[tuple[float, float], float]:
    """Locate one raw tile on the production stitch using edge correlation."""
    tile_w, tile_h = [max(8, int(value)) for value in expected_size_ds]
    preview = cv2.resize(raw_tile, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    margin = max(2, int(round(0.05 * min(tile_w, tile_h))))
    template = _tile_edge_map(preview)[margin:tile_h - margin, margin:tile_w - margin]
    if template.size == 0:
        return (approximate_origin, 0.0)
    search_x = max(36, int(round(0.85 * tile_w)))
    search_y = max(28, int(round(0.85 * tile_h)))
    approx_x, approx_y = approximate_origin
    x0 = max(0, int(math.floor(approx_x - search_x)))
    y0 = max(0, int(math.floor(approx_y - search_y)))
    x1 = min(ds_canvas.shape[1], int(math.ceil(approx_x + tile_w + search_x)))
    y1 = min(ds_canvas.shape[0], int(math.ceil(approx_y + tile_h + search_y)))
    roi = ds_canvas[y0:y1, x0:x1]
    if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
        return (approximate_origin, 0.0)
    response = cv2.matchTemplate(_tile_edge_map(roi), template, cv2.TM_CCOEFF_NORMED)
    _minimum, maximum, _min_location, max_location = cv2.minMaxLoc(response)
    origin = (float(x0 + max_location[0] - margin), float(y0 + max_location[1] - margin))
    return (origin, float(maximum))

def _raw_square_candidates_v13(image: np.ndarray, expected_square_px: float) -> list[dict[str, Any]]:
    """Find filled or outlined square candidates on an original-resolution tile."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    value_threshold = max(190, int(np.percentile(value, 84.0)))
    saturation_threshold = max(70, int(np.percentile(saturation, 68.0)))
    bright = ((value >= value_threshold) & (saturation <= saturation_threshold)).astype(np.uint8) * 255
    kernel_size = max(15, int(round(1.5 * expected_square_px)))
    kernel_size = min(kernel_size | 1, 501)
    background = cv2.morphologyEx(lab[:, :, 0], cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)))
    top_hat = cv2.subtract(lab[:, :, 0], background)
    _threshold, top_hat_mask = cv2.threshold(top_hat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates: list[dict[str, Any]] = []
    for mask_name, mask in (('bright', bright), ('top-hat', top_hat_mask)):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if not (0.42 * expected_square_px <= width <= 1.75 * expected_square_px and 0.42 * expected_square_px <= height <= 1.75 * expected_square_px):
                continue
            aspect = width / max(float(height), 1.0)
            if not 0.52 <= aspect <= 1.92:
                continue
            contour_area = float(abs(cv2.contourArea(contour)))
            fill = contour_area / max(float(width * height), 1.0)
            if fill < 0.1:
                continue
            (cx, cy), (rw, rh), _angle = cv2.minAreaRect(contour)
            if rw <= 0.0 or rh <= 0.0:
                continue
            size = math.sqrt(float(rw * rh))
            size_error = abs(math.log(max(size, 1e-09) / max(expected_square_px, 1e-09)))
            rectangularity = contour_area / max(float(rw * rh), 1.0)
            candidates.append({'center': np.asarray([cx, cy], dtype=np.float64), 'size': float(size), 'score': float(3.0 - size_error + rectangularity + fill), 'source': mask_name})
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 25, 85)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter < 1.6 * expected_square_px:
            continue
        (cx, cy), (rw, rh), _angle = cv2.minAreaRect(contour)
        if rw <= 0.0 or rh <= 0.0:
            continue
        if not (0.42 * expected_square_px <= rw <= 1.75 * expected_square_px and 0.42 * expected_square_px <= rh <= 1.75 * expected_square_px):
            continue
        ratio = rw / max(rh, 1e-09)
        if not 0.55 <= ratio <= 1.82:
            continue
        size = math.sqrt(float(rw * rh))
        score = 1.5 - abs(math.log(max(size, 1e-09) / max(expected_square_px, 1e-09)))
        candidates.append({'center': np.asarray([cx, cy], dtype=np.float64), 'size': float(size), 'score': float(score), 'source': 'edge'})
    candidates.sort(key=lambda candidate: float(candidate['score']), reverse=True)
    deduplicated: list[dict[str, Any]] = []
    for candidate in candidates:
        if all((float(np.linalg.norm(candidate['center'] - kept['center'])) > 0.28 * expected_square_px for kept in deduplicated)):
            deduplicated.append(candidate)
    return deduplicated

def _greedy_lattice_matches(predicted: np.ndarray, candidates: np.ndarray, tolerance: float) -> list[tuple[int, int, float]]:
    distances = np.linalg.norm(predicted[:, None, :] - candidates[None, :, :], axis=2)
    pairs: list[tuple[float, int, int]] = []
    for template_index in range(len(predicted)):
        for candidate_index in range(len(candidates)):
            distance = float(distances[template_index, candidate_index])
            if distance <= tolerance:
                pairs.append((distance, template_index, candidate_index))
    pairs.sort()
    used_template: set[int] = set()
    used_candidate: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for distance, template_index, candidate_index in pairs:
        if template_index in used_template or candidate_index in used_candidate:
            continue
        used_template.add(template_index)
        used_candidate.add(candidate_index)
        matches.append((template_index, candidate_index, distance))
    return matches

def _raw_similarity_fit(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    source_zero = source - source_mean
    target_zero = target - target_mean
    covariance = target_zero.T @ source_zero / float(len(source))
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(2, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    variance = float(np.mean(np.sum(source_zero * source_zero, axis=1)))
    if variance <= 1e-12:
        raise ValueError('Degenerate raw-tile lattice')
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return (scale, rotation, translation)

def _raw_template_for_side(geometry: DesignGeometry, side: str) -> np.ndarray:
    nominal = _nominal_marker_square_centers(geometry, side)
    anchor = np.mean(nominal, axis=0)
    return np.column_stack([(nominal[:, 0] - anchor[0]) / 400.0, -(nominal[:, 1] - anchor[1]) / 400.0])

def _fit_raw_lattice_v13(candidates: list[dict[str, Any]], expected_pitch_px: float, predicted_anchor: tuple[float, float], template: np.ndarray) -> dict[str, Any] | None:
    if len(candidates) < 7:
        return None
    points = np.asarray([candidate['center'] for candidate in candidates], dtype=np.float64)
    predicted_anchor_array = np.asarray(predicted_anchor, dtype=np.float64)
    best: dict[str, Any] | None = None
    for first in range(len(points)):
        for second in range(len(points)):
            if first == second:
                continue
            vector = points[second] - points[first]
            pitch = float(np.linalg.norm(vector))
            if not 0.65 * expected_pitch_px <= pitch <= 1.35 * expected_pitch_px:
                continue
            ex = vector / pitch
            for sign in (-1.0, 1.0):
                ey = sign * np.asarray([-ex[1], ex[0]], dtype=np.float64)
                transform = np.column_stack([ex, ey]) * pitch
                for row in range(4):
                    for col in (0, 1):
                        template_index = row * 3 + col
                        translation = points[first] - transform @ template[template_index]
                        predicted = template @ transform.T + translation
                        anchor = np.mean(predicted, axis=0)
                        anchor_distance = float(np.linalg.norm(anchor - predicted_anchor_array))
                        if anchor_distance > 4.0 * expected_pitch_px:
                            continue
                        matches = _greedy_lattice_matches(predicted, points, tolerance=0.26 * pitch)
                        if len(matches) < 7:
                            continue
                        mean_error = float(np.mean([match[2] for match in matches]))
                        score = 100.0 * len(matches) - 4.0 * mean_error - 1.5 * abs(pitch - expected_pitch_px) - 0.15 * anchor_distance
                        if best is None or score > float(best['score']):
                            best = {'score': score, 'predicted': predicted, 'matches': matches, 'pitch': pitch, 'transform': transform, 'translation': translation, 'anchor': anchor, 'mean_error': mean_error}
    if best is None:
        return None
    source = np.asarray([template[match[0]] for match in best['matches']], dtype=np.float64)
    target = np.asarray([points[match[1]] for match in best['matches']], dtype=np.float64)
    scale, rotation, translation = _raw_similarity_fit(source, target)
    predicted = scale * (rotation @ template.T).T + translation
    matches = _greedy_lattice_matches(predicted, points, tolerance=0.24 * max(scale, 1e-09))
    if len(matches) >= len(best['matches']):
        best.update({'predicted': predicted, 'matches': matches, 'pitch': float(scale), 'transform': scale * rotation, 'translation': translation, 'anchor': np.mean(predicted, axis=0), 'mean_error': float(np.mean([match[2] for match in matches]))})
    return best

def _raw_square_candidates_unified(image: np.ndarray, expected_square_px: float) -> list[dict[str, Any]]:
    """Find filled and outline square candidates with one shared representation.

    The production wafers contain two optical appearances of the same GDS
    fiducial: bright filled squares and low-contrast outline squares.  This
    function deliberately keeps candidate extraction separate from lattice
    fitting.  Every returned item is only a possible square; the ordered 3 x 4
    fit below decides which 12 belong to the marker.
    """
    if image is None or image.size == 0:
        return []
    expected = max(float(expected_square_px), 8.0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    candidates: list[dict[str, Any]] = []

    def add_component_candidates(mask: np.ndarray, source: str, base_score: float) -> None:
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for label in range(1, count):
            x, y, width, height, area = [int(v) for v in stats[label]]
            if not (0.62 * expected <= width <= 1.38 * expected and 0.62 * expected <= height <= 1.38 * expected):
                continue
            aspect = width / max(float(height), 1.0)
            if not 0.72 <= aspect <= 1.38:
                continue
            fill = area / max(float(width * height), 1.0)
            if fill < 0.32:
                continue
            size = 0.5 * (width + height)
            size_error = abs(math.log(max(size, 1e-09) / expected))
            candidates.append({'center': np.asarray(centroids[label], dtype=np.float64), 'size': float(size), 'score': float(base_score - size_error + fill), 'source': source, 'bbox': (int(x), int(y), int(width), int(height))})
    value_threshold = max(205, int(np.percentile(value, 88.0)))
    saturation_threshold = min(105, max(55, int(np.percentile(saturation, 72.0))))
    bright = ((value >= value_threshold) & (saturation <= saturation_threshold)).astype(np.uint8) * 255
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    add_component_candidates(bright, 'bright-filled', 4.4)
    kernel_size = max(15, int(round(1.45 * expected)))
    kernel_size = min(kernel_size | 1, 501)
    background = cv2.morphologyEx(lab[:, :, 0], cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)))
    top_hat = cv2.subtract(lab[:, :, 0], background)
    _threshold, top_hat_mask = cv2.threshold(top_hat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    top_hat_mask = cv2.morphologyEx(top_hat_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    top_hat_mask = cv2.morphologyEx(top_hat_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    add_component_candidates(top_hat_mask, 'top-hat-filled', 4.1)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(clahe, 18, 55)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if not (0.62 * expected <= width <= 1.38 * expected and 0.62 * expected <= height <= 1.38 * expected):
            continue
        aspect = width / max(float(height), 1.0)
        if not 0.72 <= aspect <= 1.38:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter < 1.55 * expected:
            continue
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if not 4 <= len(approx) <= 18:
            continue
        size = 0.5 * (width + height)
        size_error = abs(math.log(max(size, 1e-09) / expected))
        candidates.append({'center': np.asarray([x + 0.5 * width, y + 0.5 * height], dtype=np.float64), 'size': float(size), 'score': float(2.2 - size_error), 'source': 'outline-edge', 'bbox': (int(x), int(y), int(width), int(height))})
    candidates.sort(key=lambda candidate: float(candidate['score']), reverse=True)
    deduplicated: list[dict[str, Any]] = []
    for candidate in candidates:
        if all((float(np.linalg.norm(candidate['center'] - kept['center'])) > 0.24 * expected for kept in deduplicated)):
            deduplicated.append(candidate)
    return deduplicated

def _fit_raw_lattice_unified(candidates: list[dict[str, Any]], expected_pitch_px: float, predicted_anchor: tuple[float, float], template: np.ndarray) -> dict[str, Any] | None:
    """Select four physical rows of three and fit the exact GDS lattice.

    This is the production form of the locally validated algorithm.  It does
    not infer the 3 x 4 structure from arbitrary point ordering.  Instead it
    explicitly finds horizontal triples, selects four disjoint rows with the
    design vertical spacing pattern, orders all 12 observations by row,
    and only then performs the similarity fit.
    """
    if len(candidates) < 12:
        return None
    expected_pitch = max(float(expected_pitch_px), 1e-09)
    predicted_anchor_array = np.asarray(predicted_anchor, dtype=np.float64)
    nearby = [candidate for candidate in candidates if float(np.linalg.norm(candidate['center'] - predicted_anchor_array)) <= 4.8 * expected_pitch and 0.28 * expected_pitch <= float(candidate['size']) <= 0.82 * expected_pitch]
    nearby.sort(key=lambda item: float(item['score']), reverse=True)
    nearby = nearby[:36]
    if len(nearby) < 12:
        return None
    points = np.asarray([item['center'] for item in nearby], dtype=np.float64)
    row_hypotheses: list[dict[str, Any]] = []
    n = len(nearby)
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                indices = np.asarray([i, j, k], dtype=np.int32)
                row_points = points[indices]
                order = np.argsort(row_points[:, 0])
                indices = indices[order]
                row_points = row_points[order]
                dx1 = float(row_points[1, 0] - row_points[0, 0])
                dx2 = float(row_points[2, 0] - row_points[1, 0])
                if not (0.7 * expected_pitch <= dx1 <= 1.3 * expected_pitch and 0.7 * expected_pitch <= dx2 <= 1.3 * expected_pitch):
                    continue
                pitch = 0.5 * (dx1 + dx2)
                y_spread = float(np.ptp(row_points[:, 1]))
                if y_spread > 0.24 * pitch:
                    continue
                spacing_error = abs(dx1 - pitch) + abs(dx2 - pitch)
                center = np.mean(row_points, axis=0)
                angle = float(math.atan2(row_points[2, 1] - row_points[0, 1], row_points[2, 0] - row_points[0, 0]))
                quality = sum((float(nearby[index]['score']) for index in indices))
                score = quality - 0.08 * spacing_error - 0.1 * y_spread
                row_hypotheses.append({'indices': tuple((int(index) for index in indices)), 'points': row_points, 'center': center, 'pitch': pitch, 'angle': angle, 'score': score})
    if len(row_hypotheses) < 3:
        return None
    row_hypotheses.sort(key=lambda row: float(row['score']), reverse=True)
    row_hypotheses = row_hypotheses[:48]
    template_rows = np.asarray([float(np.mean(template[row * 3:row * 3 + 3, 1])) for row in range(4)], dtype=np.float64)
    template_gaps = np.diff(template_rows)
    best: dict[str, Any] | None = None
    import itertools
    for row_combo in itertools.combinations(row_hypotheses, 4):
        used = [index for row in row_combo for index in row['indices']]
        if len(set(used)) != 12:
            continue
        rows = sorted(row_combo, key=lambda row: float(row['center'][1]))
        pitches = np.asarray([float(row['pitch']) for row in rows])
        pitch = float(np.median(pitches))
        if float(np.max(np.abs(pitches / max(pitch, 1e-09) - 1.0))) > 0.16:
            continue
        angles = np.unwrap(np.asarray([float(row['angle']) for row in rows]))
        if float(np.ptp(angles)) > math.radians(3.0):
            continue
        x_centers = np.asarray([float(row['center'][0]) for row in rows])
        if float(np.ptp(x_centers)) > 0.3 * pitch:
            continue
        y_centers = np.asarray([float(row['center'][1]) for row in rows])
        observed_gaps = np.diff(y_centers)
        expected_gaps = template_gaps * pitch
        gap_error = np.abs(observed_gaps - expected_gaps)
        if np.any(gap_error > 0.34 * pitch):
            continue
        observed_ordered = np.vstack([row['points'] for row in rows])
        try:
            scale, rotation, translation = _raw_similarity_fit(template, observed_ordered)
        except ValueError:
            continue
        predicted = scale * (rotation @ template.T).T + translation
        residuals = np.linalg.norm(predicted - observed_ordered, axis=1)
        rms_error = float(np.sqrt(np.mean(residuals * residuals)))
        mean_error = float(np.mean(residuals))
        anchor = np.mean(predicted, axis=0)
        anchor_distance = float(np.linalg.norm(anchor - predicted_anchor_array))
        if not 0.7 * expected_pitch <= scale <= 1.3 * expected_pitch:
            continue
        if rms_error > 0.1 * scale:
            continue
        if anchor_distance > 2.2 * expected_pitch:
            continue
        score = sum((float(row['score']) for row in rows)) - 0.3 * float(np.sum(gap_error)) - 3.0 * rms_error - 0.08 * anchor_distance
        if best is None or score > float(best['score']):
            transform = scale * rotation
            best = {'score': score, 'predicted': predicted, 'observed_ordered': observed_ordered, 'matches': [(index, index, float(residuals[index])) for index in range(12)], 'pitch': float(scale), 'transform': transform, 'translation': translation, 'anchor': anchor, 'mean_error': mean_error, 'rms_error': rms_error, 'fit_mode': 'four ordered row triples -> exact GDS similarity', 'selected_candidate_indices': [int(index) for row in rows for index in row['indices']], 'matched_sizes': [float(nearby[index]['size']) for row in rows for index in row['indices']]}
    if best is not None:
        return best
    partial_best: dict[str, Any] | None = None
    for row_combo in itertools.combinations(row_hypotheses, 3):
        used = [index for row in row_combo for index in row['indices']]
        if len(set(used)) != 9:
            continue
        rows = sorted(row_combo, key=lambda row: float(row['center'][1]))
        pitches = np.asarray([float(row['pitch']) for row in rows])
        pitch = float(np.median(pitches))
        if float(np.max(np.abs(pitches / max(pitch, 1e-09) - 1.0))) > 0.16:
            continue
        x_centers = np.asarray([float(row['center'][0]) for row in rows])
        if float(np.ptp(x_centers)) > 0.3 * pitch:
            continue
        observed_nine = np.vstack([row['points'] for row in rows])
        for template_row_indices in itertools.combinations(range(4), 3):
            source_nine = np.vstack([template[row * 3:row * 3 + 3] for row in template_row_indices])
            try:
                scale, rotation, translation = _raw_similarity_fit(source_nine, observed_nine)
            except ValueError:
                continue
            if not 0.7 * expected_pitch <= scale <= 1.3 * expected_pitch:
                continue
            predicted = scale * (rotation @ template.T).T + translation
            matches = _greedy_lattice_matches(predicted, points, tolerance=0.27 * max(scale, 1e-09))
            if len(matches) < 10:
                continue
            source = np.asarray([template[match[0]] for match in matches], dtype=np.float64)
            target = np.asarray([points[match[1]] for match in matches], dtype=np.float64)
            try:
                refined_scale, refined_rotation, refined_translation = _raw_similarity_fit(source, target)
            except ValueError:
                continue
            refined = refined_scale * (refined_rotation @ template.T).T + refined_translation
            refined_matches = _greedy_lattice_matches(refined, points, tolerance=0.22 * max(refined_scale, 1e-09))
            if len(refined_matches) < 10:
                continue
            residuals = np.asarray([match[2] for match in refined_matches], dtype=np.float64)
            mean_error = float(np.mean(residuals))
            rms_error = float(np.sqrt(np.mean(residuals * residuals)))
            anchor = np.mean(refined, axis=0)
            anchor_distance = float(np.linalg.norm(anchor - predicted_anchor_array))
            if anchor_distance > 2.2 * expected_pitch:
                continue
            score = 120.0 * len(refined_matches) - 5.0 * mean_error - 1.5 * abs(refined_scale - expected_pitch) - 0.08 * anchor_distance
            if partial_best is None or score > float(partial_best['score']):
                partial_best = {'score': score, 'predicted': refined, 'observed_ordered': target, 'matches': refined_matches, 'pitch': float(refined_scale), 'transform': refined_scale * refined_rotation, 'translation': refined_translation, 'anchor': anchor, 'mean_error': mean_error, 'rms_error': rms_error, 'fit_mode': f'three ordered row triples + {len(refined_matches)}/12 exact-template matches', 'matched_candidate_indices': [int(match[1]) for match in refined_matches], 'matched_sizes': [float(nearby[int(match[1])]['size']) for match in refined_matches]}
    if partial_best is not None:
        return partial_best
    hypothesis_best: dict[str, Any] | None = None
    for first in range(len(points)):
        for second in range(len(points)):
            if first == second:
                continue
            vector = points[second] - points[first]
            pitch = float(np.linalg.norm(vector))
            if not 0.68 * expected_pitch <= pitch <= 1.32 * expected_pitch:
                continue
            ex = vector / max(pitch, 1e-09)
            for sign in (-1.0, 1.0):
                ey = sign * np.asarray([-ex[1], ex[0]], dtype=np.float64)
                transform = np.column_stack([ex, ey]) * pitch
                for template_index in range(len(template)):
                    translation = points[first] - transform @ template[template_index]
                    predicted = template @ transform.T + translation
                    anchor = np.mean(predicted, axis=0)
                    anchor_distance = float(np.linalg.norm(anchor - predicted_anchor_array))
                    if anchor_distance > 3.0 * expected_pitch:
                        continue
                    matches = _greedy_lattice_matches(predicted, points, tolerance=0.3 * pitch)
                    if len(matches) < 10:
                        continue
                    source = np.asarray([template[match[0]] for match in matches], dtype=np.float64)
                    target = np.asarray([points[match[1]] for match in matches], dtype=np.float64)
                    try:
                        scale, rotation, refined_translation = _raw_similarity_fit(source, target)
                    except ValueError:
                        continue
                    refined = scale * (rotation @ template.T).T + refined_translation
                    refined_matches = _greedy_lattice_matches(refined, points, tolerance=0.24 * max(scale, 1e-09))
                    if len(refined_matches) < 10:
                        continue
                    residuals = np.asarray([match[2] for match in refined_matches], dtype=np.float64)
                    mean_error = float(np.mean(residuals))
                    rms_error = float(np.sqrt(np.mean(residuals * residuals)))
                    refined_anchor = np.mean(refined, axis=0)
                    refined_anchor_distance = float(np.linalg.norm(refined_anchor - predicted_anchor_array))
                    score = 120.0 * len(refined_matches) - 5.0 * mean_error - 1.5 * abs(scale - expected_pitch) - 0.08 * refined_anchor_distance
                    if hypothesis_best is None or score > float(hypothesis_best['score']):
                        hypothesis_best = {'score': score, 'predicted': refined, 'observed_ordered': target, 'matches': refined_matches, 'pitch': float(scale), 'transform': scale * rotation, 'translation': refined_translation, 'anchor': refined_anchor, 'mean_error': mean_error, 'rms_error': rms_error, 'fit_mode': f'exact GDS similarity with {len(refined_matches)}/12 measured squares', 'matched_candidate_indices': [int(match[1]) for match in refined_matches], 'matched_sizes': [float(nearby[int(match[1])]['size']) for match in refined_matches]}
    return hypothesis_best

def _raw_fit_template(fit: dict[str, Any], candidate_sizes: list[float]) -> dict[str, Any]:
    predicted = np.asarray(fit['predicted'], dtype=np.float64)
    transform = np.asarray(fit['transform'], dtype=np.float64)
    pitch = float(fit['pitch'])
    ex = transform[:, 0] / max(float(np.linalg.norm(transform[:, 0])), 1e-09)
    ey = transform[:, 1] / max(float(np.linalg.norm(transform[:, 1])), 1e-09)
    observed_square = float(np.median(candidate_sizes)) if candidate_sizes else 0.5 * pitch
    square_size = float(np.clip(observed_square, 0.4 * pitch, 0.62 * pitch))
    half_square = 0.5 * square_size
    boxes: dict[Any, list[tuple[float, float]]] = {}
    squares: dict[Any, tuple[float, float]] = {}
    for index, center in enumerate(predicted):
        corners = [center - half_square * ex - half_square * ey, center + half_square * ex - half_square * ey, center + half_square * ex + half_square * ey, center - half_square * ex + half_square * ey]
        key = (index // 3, index % 3)
        boxes[key] = [(float(point[0]), float(point[1])) for point in corners]
        squares[key] = (float(center[0]), float(center[1]))
    points = [point for corners in boxes.values() for point in corners]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    padding = max(8.0, 0.55 * pitch)
    boxes['area', 0] = [(min(xs) - padding, min(ys) - padding), (max(xs) + padding, min(ys) - padding), (max(xs) + padding, max(ys) + padding), (min(xs) - padding, max(ys) + padding)]
    return {'boxes': boxes, 'squares': squares}

def _raw_lattice_edge_map(image: np.ndarray) -> np.ndarray:
    """Normalize weak and strong marker edges into one correlation image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(12, 12)).apply(gray)
    gx = cv2.Scharr(clahe, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(clahe, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gx, gy)
    scale = float(np.percentile(magnitude, 97.5))
    if scale > 1e-06:
        magnitude = np.clip(magnitude / scale, 0.0, 1.0)
    return cv2.GaussianBlur(magnitude.astype(np.float32), (0, 0), 1.2)

def _raw_lattice_edge_template(template: np.ndarray, pitch: float, square_size: float, angle_deg: float, side: str, include_rail: bool=True) -> tuple[np.ndarray, np.ndarray]:
    """Render the 12-square lattice plus its directional GDS rail."""
    centers = np.asarray(template, dtype=np.float32) * float(pitch)
    half = 0.5 * float(square_size)
    angle = math.radians(float(angle_deg))
    rotation = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)
    corner_offsets = np.asarray([[-half, -half], [half, -half], [half, half], [-half, half]], dtype=np.float32)
    rotated_boxes = [((center + corner_offsets) @ rotation.T) for center in centers]
    rail_x0, rail_x1 = (-4.2, 1.3) if side == 'left' else (-1.3, 4.2)
    rail = (np.asarray([[rail_x0, -0.25], [rail_x1, -0.25], [rail_x1, 0.25], [rail_x0, 0.25]], dtype=np.float32) * float(pitch)) @ rotation.T
    all_points = np.vstack(rotated_boxes + ([rail] if include_rail else []))
    margin = max(8, int(round(0.15 * pitch)))
    minimum = np.floor(np.min(all_points, axis=0)).astype(np.int32) - margin
    maximum = np.ceil(np.max(all_points, axis=0)).astype(np.int32) + margin
    shape = (int(maximum[1] - minimum[1] + 1), int(maximum[0] - minimum[0] + 1))
    rendered = np.zeros(shape, dtype=np.float32)
    thickness = max(2, int(round(0.035 * pitch)))
    for box in rotated_boxes:
        cv2.polylines(rendered, [np.rint(box - minimum).astype(np.int32)], True, 1.0, thickness, cv2.LINE_AA)
    if include_rail:
        cv2.polylines(rendered, [np.rint(rail - minimum).astype(np.int32)], True, 0.65, thickness, cv2.LINE_AA)
    rendered = cv2.GaussianBlur(rendered, (0, 0), max(1.0, 0.015 * pitch))
    return rendered, -minimum.astype(np.float32)

def _stitch_raw_marker_neighborhood(indexed: dict[tuple[int, int], Path], approximate_anchor_raw: tuple[float, float], expected_pitch_raw: float, raw_size: tuple[int, int], config: dict[str, Any], work_factor: float=0.25) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a small downsampled raw-tile mosaic around one predicted marker."""
    from illumination_stitching import make_feather_weight

    raw_w, raw_h = raw_size
    configured_w = int(config.get('tile_width', raw_w) or raw_w)
    configured_h = int(config.get('tile_height', raw_h) or raw_h)
    step_x = configured_w * (1.0 - float(config.get('overlap_x_percent', 0.0) or 0.0) / 100.0)
    step_y = configured_h * (1.0 - float(config.get('overlap_y_percent', 0.0) or 0.0) / 100.0)
    # Marker scale can differ a few percent from radius metrology.  At the wafer
    # edge that becomes a several-pitch position error, so retain enough context
    # to contain the physical marker rather than clipping it at the prior.
    half_size = max(1600, int(math.ceil(12.0 * expected_pitch_raw)))
    x0 = int(round(float(approximate_anchor_raw[0]) - half_size))
    y0 = int(round(float(approximate_anchor_raw[1]) - half_size))
    x1 = x0 + 2 * half_size
    y1 = y0 + 2 * half_size
    local_w = max(8, int(round((x1 - x0) * work_factor)))
    local_h = max(8, int(round((y1 - y0) * work_factor)))
    tile_w_work = max(1, int(round(configured_w * work_factor)))
    tile_h_work = max(1, int(round(configured_h * work_factor)))
    overlap_x_work = max(1, int(round(configured_w * float(config.get('overlap_x_percent', 0.0) or 0.0) * 0.01 * work_factor)))
    overlap_y_work = max(1, int(round(configured_h * float(config.get('overlap_y_percent', 0.0) or 0.0) * 0.01 * work_factor)))
    acc = np.zeros((local_h, local_w, 3), dtype=np.float32)
    weight_sum = np.zeros((local_h, local_w), dtype=np.float32)

    for key, path in indexed.items():
        col, row = key
        tile_x0 = max(0, int((col - 1) * step_x))
        tile_y0 = max(0, int((row - 1) * step_y))
        tile_x1 = tile_x0 + configured_w
        tile_y1 = tile_y0 + configured_h
        if tile_x1 <= x0 or tile_x0 >= x1 or tile_y1 <= y0 or tile_y0 >= y1:
            continue
        raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if raw is None:
            continue
        if raw.shape[1] != configured_w or raw.shape[0] != configured_h:
            raw = cv2.resize(raw, (configured_w, configured_h), interpolation=cv2.INTER_AREA)
        tile_work = cv2.resize(raw, (tile_w_work, tile_h_work), interpolation=cv2.INTER_AREA)
        weight = make_feather_weight(tile_h_work, tile_w_work, overlap_x_work, overlap_y_work, has_left=(col - 1, row) in indexed, has_right=(col + 1, row) in indexed, has_top=(col, row - 1) in indexed, has_bottom=(col, row + 1) in indexed)
        destination_x0 = int(round((tile_x0 - x0) * work_factor))
        destination_y0 = int(round((tile_y0 - y0) * work_factor))
        destination_x1 = destination_x0 + tile_w_work
        destination_y1 = destination_y0 + tile_h_work
        ox0 = max(0, destination_x0)
        oy0 = max(0, destination_y0)
        ox1 = min(local_w, destination_x1)
        oy1 = min(local_h, destination_y1)
        if ox1 <= ox0 or oy1 <= oy0:
            continue
        sx0 = ox0 - destination_x0
        sy0 = oy0 - destination_y0
        sx1 = sx0 + (ox1 - ox0)
        sy1 = sy0 + (oy1 - oy0)
        tile_crop = tile_work[sy0:sy1, sx0:sx1]
        weight_crop = weight[sy0:sy1, sx0:sx1]
        acc[oy0:oy1, ox0:ox1] += tile_crop.astype(np.float32) * weight_crop[:, :, None]
        weight_sum[oy0:oy1, ox0:ox1] += weight_crop

    if not np.any(weight_sum > 1e-06):
        raise RuntimeError('No raw tiles overlap the predicted marker neighborhood')
    mosaic = np.zeros_like(acc, dtype=np.uint8)
    valid = weight_sum > 1e-06
    mosaic[valid] = np.clip(acc[valid] / weight_sum[valid, None], 0.0, 255.0).astype(np.uint8)
    return mosaic, (x0, y0)

def _match_raw_lattice_edge_template(image: np.ndarray, template: np.ndarray, expected_pitch_work: float, expected_anchor_work: tuple[float, float], expected_angle_deg: float, side: str, include_rail: bool=True) -> dict[str, Any]:
    """Find a whole marker by correlating all 12 expected square borders."""
    edges = _raw_lattice_edge_map(image)
    best: dict[str, Any] | None = None
    hypotheses: list[dict[str, Any]] = []
    coarse_scales = np.linspace(0.92, 1.08, 5)
    coarse_angles = float(expected_angle_deg) + np.linspace(-4.0, 4.0, 5)

    def evaluate(scale_factor: float, angle_deg: float) -> None:
        nonlocal best
        pitch = float(expected_pitch_work) * float(scale_factor)
        square_size = 0.5 * pitch
        rendered, anchor_in_template = _raw_lattice_edge_template(template, pitch, square_size, angle_deg, side, include_rail=include_rail)
        if rendered.shape[0] >= edges.shape[0] or rendered.shape[1] >= edges.shape[1]:
            return
        response = cv2.matchTemplate(edges, rendered, cv2.TM_CCORR_NORMED)
        yy, xx = np.indices(response.shape, dtype=np.float32)
        distance_squared = (xx + float(anchor_in_template[0]) - float(expected_anchor_work[0])) ** 2 + (yy + float(anchor_in_template[1]) - float(expected_anchor_work[1])) ** 2
        objective = response - 0.0003 * distance_squared / max(float(expected_pitch_work) ** 2, 1e-09)
        objective_peaks = objective.copy()
        suppression_radius = max(3, int(round(0.8 * expected_pitch_work)))
        for _peak_index in range(3):
            _minimum, objective_score, _min_location, location = cv2.minMaxLoc(objective_peaks)
            score = float(response[location[1], location[0]])
            candidate = {'score': score, 'objective_score': float(objective_score), 'location': location, 'anchor_work': (float(location[0] + anchor_in_template[0]), float(location[1] + anchor_in_template[1])), 'pitch_work': pitch, 'square_work': square_size, 'angle_deg': float(angle_deg)}
            hypotheses.append(candidate)
            if best is None or float(objective_score) > float(best['objective_score']):
                best = candidate
            cv2.circle(objective_peaks, location, suppression_radius, -1.0e6, -1)

    for scale_factor in coarse_scales:
        for angle_deg in coarse_angles:
            evaluate(float(scale_factor), float(angle_deg))
    if best is None:
        raise RuntimeError('No valid full-lattice edge-correlation template')
    base_scale = float(best['pitch_work']) / max(float(expected_pitch_work), 1e-09)
    base_angle = float(best['angle_deg'])
    for scale_factor in np.linspace(base_scale - 0.02, base_scale + 0.02, 3):
        for angle_deg in np.linspace(base_angle - 1.0, base_angle + 1.0, 3):
            evaluate(float(scale_factor), float(angle_deg))
    assert best is not None
    distinct: list[dict[str, Any]] = []
    for candidate in sorted(hypotheses, key=lambda item: float(item['objective_score']), reverse=True):
        if all(float(np.linalg.norm(np.asarray(candidate['anchor_work']) - np.asarray(kept['anchor_work']))) >= 0.45 * expected_pitch_work or abs(float(candidate['pitch_work']) - float(kept['pitch_work'])) >= 0.025 * expected_pitch_work for kept in distinct):
            distinct.append(candidate)
        if len(distinct) >= 24:
            break
    best = dict(best)
    best['alternatives'] = [dict(candidate) for candidate in distinct if abs(float(candidate['objective_score']) - float(best['objective_score'])) > 1e-12 or float(np.linalg.norm(np.asarray(candidate['anchor_work']) - np.asarray(best['anchor_work']))) > 1e-06][:23]
    return best

def _run_raw_tile_edge_template_detector(indexed: dict[tuple[int, int], Path], predicted: dict[str, tuple[float, float]], expected_pitch_ds: float, exact_ds: float, raw_size: tuple[int, int], geometry: DesignGeometry, config: dict[str, Any], *, include_rail: bool=True, fixed_matches: dict[str, dict[str, Any]] | None=None) -> dict[str, dict[str, Any]]:
    """Fallback for faint, damaged, or partially merged optical markers."""
    work_factor = 0.20
    stitch_factor = 0.20
    expected_pitch_raw = float(expected_pitch_ds) / max(float(exact_ds), 1e-09)
    expected_angle_deg = math.degrees(float(config.get('_wafer_flat_angle_rad', 0.0) or 0.0))
    template = _raw_template_for_side(geometry, 'left')
    side_options: dict[str, list[dict[str, Any]]] = {}
    for side in ('left', 'right'):
        if fixed_matches and side in fixed_matches:
            fixed = copy.deepcopy(fixed_matches[side])
            diagnostics = fixed['diagnostics']
            diagnostics.setdefault('template_correlation', float(np.clip(0.45 + 0.03 * float(diagnostics.get('match_count', 0)) - float(diagnostics.get('error_pitch_ratio', 0.0) or 0.0), 0.0, 0.85)))
            diagnostics.setdefault('template_objective', diagnostics['template_correlation'])
            side_options[side] = [fixed]
            continue
        predicted_raw = (float(predicted[side][0]) / exact_ds, float(predicted[side][1]) / exact_ds)
        mosaic, origin_raw = _stitch_raw_marker_neighborhood(indexed, predicted_raw, expected_pitch_raw, raw_size, config, work_factor=stitch_factor)
        if stitch_factor != work_factor:
            resize_factor = work_factor / stitch_factor
            mosaic = cv2.resize(mosaic, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
        expected_anchor_work = ((predicted_raw[0] - float(origin_raw[0])) * work_factor, (predicted_raw[1] - float(origin_raw[1])) * work_factor)
        match = _match_raw_lattice_edge_template(mosaic, template, expected_pitch_raw * work_factor, expected_anchor_work, expected_angle_deg, side, include_rail=include_rail)
        template_mode = '12-square + directional-rail edge correlation' if include_rail else '12-square edge correlation'
        options: list[dict[str, Any]] = []
        for hypothesis in [match] + list(match.get('alternatives', [])):
            anchor_raw = (float(origin_raw[0]) + float(hypothesis['anchor_work'][0]) / work_factor, float(origin_raw[1]) + float(hypothesis['anchor_work'][1]) / work_factor)
            anchor_ds = np.asarray([anchor_raw[0] * exact_ds, anchor_raw[1] * exact_ds], dtype=np.float64)
            pitch_ds = float(hypothesis['pitch_work']) / work_factor * exact_ds
            square_ds = float(hypothesis['square_work']) / work_factor * exact_ds
            angle = math.radians(float(hypothesis['angle_deg']))
            rotation = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float64)
            predicted_squares = pitch_ds * (rotation @ template.T).T + anchor_ds
            fit = {'predicted': predicted_squares, 'pitch': pitch_ds, 'transform': pitch_ds * rotation, 'anchor': anchor_ds}
            options.append({'anchor': (float(anchor_ds[0]), float(anchor_ds[1])), 'template': _raw_fit_template(fit, [square_ds] * len(template)), 'diagnostics': {'source': 'raw-tile full-marker edge template', 'optical_mode': 'edge-template', 'fit_mode': template_mode, 'algorithm_detail': f'CLAHE + Scharr edge map correlated against exact GDS {template_mode}', 'match_count': 12, 'pitch_ds_px': pitch_ds, 'mean_error_ds_px': None, 'rms_error_ds_px': None, 'error_pitch_ratio': None, 'template_correlation': float(hypothesis['score']), 'template_objective': float(hypothesis['objective_score']), 'angle_deg': float(hypothesis['angle_deg']), 'candidate_count': 0, 'legacy_candidate_count': 0, 'unified_candidate_count': 0, 'canvas_translation_refinement': {'applied': False, 'shift_ds_px': [0.0, 0.0], 'mode': 'full-marker template fitted in nominal raw-tile coordinates'}, 'tiles': []}})
        side_options[side] = options

    predicted_left = np.asarray(predicted['left'], dtype=np.float64)
    predicted_right = np.asarray(predicted['right'], dtype=np.float64)
    expected_separation = float(np.linalg.norm(predicted_right - predicted_left))
    pair_candidates: list[tuple[float, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for left_option in side_options['left']:
        for right_option in side_options['right']:
            left = np.asarray(left_option['anchor'], dtype=np.float64)
            right = np.asarray(right_option['anchor'], dtype=np.float64)
            separation_ratio = float(np.linalg.norm(right - left)) / max(expected_separation, 1e-09)
            midpoint_error = float(np.linalg.norm(0.5 * (left + right) - 0.5 * (predicted_left + predicted_right)))
            scores = [float(left_option['diagnostics']['template_correlation']), float(right_option['diagnostics']['template_correlation'])]
            angles = [float(left_option['diagnostics']['angle_deg']), float(right_option['diagnostics']['angle_deg'])]
            pitches = [float(left_option['diagnostics']['pitch_ds_px']), float(right_option['diagnostics']['pitch_ds_px'])]
            baseline_angle = math.degrees(math.atan2(float(right[1] - left[1]), float(right[0] - left[0])))
            doubled_x = math.cos(math.radians(2.0 * angles[0])) + math.cos(math.radians(2.0 * angles[1]))
            doubled_y = math.sin(math.radians(2.0 * angles[0])) + math.sin(math.radians(2.0 * angles[1]))
            mean_marker_angle = 0.5 * math.degrees(math.atan2(doubled_y, doubled_x))
            baseline_angle_error = abs(((baseline_angle - mean_marker_angle + 90.0) % 180.0) - 90.0)
            angle_difference = abs(((angles[0] - angles[1] + 90.0) % 180.0) - 90.0)
            pitch_ratio_error = abs(pitches[0] / max(pitches[1], 1e-09) - 1.0)
            pair_ok = 0.985 <= separation_ratio <= 1.026 and midpoint_error <= 3.5 * expected_pitch_ds and min(scores) >= 0.30 and max(scores) >= 0.35 and angle_difference <= 1.25 and pitch_ratio_error <= 0.12 and baseline_angle_error <= 0.75
            pair_diagnostics = {'separation_ratio': separation_ratio, 'midpoint_error_ds_px': midpoint_error, 'minimum_correlation': min(scores), 'maximum_correlation': max(scores), 'angles_deg': angles, 'angle_difference_deg': angle_difference, 'baseline_angle_deg': baseline_angle, 'baseline_angle_error_deg': baseline_angle_error, 'pitches_ds_px': pitches, 'anchors_ds_px': {'left': [float(left[0]), float(left[1])], 'right': [float(right[0]), float(right[1])]}, 'accepted': bool(pair_ok)}
            quality = sum(scores) - 0.08 * baseline_angle_error - 0.45 * pitch_ratio_error - 0.004 * midpoint_error / max(expected_pitch_ds, 1e-09) - 0.25 * abs(separation_ratio - 1.0)
            if pair_ok:
                pair_candidates.append((quality, left_option, right_option, pair_diagnostics))
    if not pair_candidates:
        primary_left = side_options['left'][0]
        primary_right = side_options['right'][0]
        raise RuntimeError(f'Full-lattice edge-template pair failed joint validation; primary anchors={primary_left["anchor"]}, {primary_right["anchor"]}')
    _quality, left_match, right_match, pair_diagnostics = max(pair_candidates, key=lambda item: item[0])
    matches = {'left': left_match, 'right': right_match}
    for side in ('left', 'right'):
        matches[side]['diagnostics']['pair_validation'] = pair_diagnostics
        matches[side]['diagnostics']['joint_hypotheses_evaluated'] = len(side_options['left']) * len(side_options['right'])
    return matches

def _run_raw_tile_lattice_detector(tile_folder: str | Path, ds_canvas: np.ndarray, canvas_center_ds: tuple[float, float], canvas_radius_ds: float, geometry: DesignGeometry, ds_factor: float, stitch_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Detect both optical marker styles without changing v13 coordinates.

    The bright/filled wafer uses the exact v13 detector and coordinate mapping,
    which was empirically aligned in the production GUI.  The low-contrast
    yellow/outline wafer uses the newer ordered-row lattice fit only when the
    marker is classified as outline-dominant or the v13 fit fails.

    No rendered-canvas translation correction is applied.  That v15 correction
    fitted the already drawn GUI overlay rather than the underlying marker and
    could move a correct v13 result.
    """
    indexed = _index_raw_tiles(tile_folder)
    if len(indexed) < 4:
        raise RuntimeError('Raw tile lattice detector found too few indexed tiles')
    sample = cv2.imread(str(next(iter(indexed.values()))), cv2.IMREAD_COLOR)
    if sample is None:
        raise RuntimeError('Could not read a raw acquisition tile')
    raw_h, raw_w = sample.shape[:2]
    config = dict(stitch_config or {})
    configured_w = int(config.get('tile_width', raw_w) or raw_w)
    configured_h = int(config.get('tile_height', raw_h) or raw_h)
    cols = int(config.get('tile_cols', max((key[0] for key in indexed))) or 0)
    rows = int(config.get('tile_rows', max((key[1] for key in indexed))) or 0)
    overlap_x = float(config.get('overlap_x_percent', 0.0) or 0.0)
    overlap_y = float(config.get('overlap_y_percent', 0.0) or 0.0)
    if config.get('_stitch_downscale_divisor') and float(config['_stitch_downscale_divisor']) > 0:
        exact_ds = 1.0 / float(config['_stitch_downscale_divisor'])
    elif config.get('downscale') and float(config['downscale']) > 0:
        exact_ds = 1.0 / float(config['downscale'])
    else:
        exact_ds = float(config.get('downscale_factor', ds_factor) or ds_factor)
    if exact_ds <= 0.0:
        exact_ds = float(ds_factor)
    step_x_raw = configured_w * (1.0 - overlap_x / 100.0)
    step_y_raw = configured_h * (1.0 - overlap_y / 100.0)
    tile_w_ds = max(1, int(configured_w * exact_ds))
    tile_h_ds = max(1, int(configured_h * exact_ds))
    expected_canvas_w = int(((max(cols, 1) - 1) * step_x_raw + configured_w) * exact_ds)
    expected_canvas_h = int(((max(rows, 1) - 1) * step_y_raw + configured_h) * exact_ds)
    if abs(expected_canvas_w - ds_canvas.shape[1]) > 2 or abs(expected_canvas_h - ds_canvas.shape[0]) > 2:
        raise RuntimeError(f'Stitch geometry/config mismatch: config predicts {expected_canvas_w}x{expected_canvas_h}, runtime canvas is {ds_canvas.shape[1]}x{ds_canvas.shape[0]}')
    scale_x = tile_w_ds / float(raw_w)
    scale_y = tile_h_ds / float(raw_h)
    nominal_origins = {key: (float(max(0, int((key[0] - 1) * step_x_raw * exact_ds))), float(max(0, int((key[1] - 1) * step_y_raw * exact_ds)))) for key in indexed if 1 <= key[0] <= cols and 1 <= key[1] <= rows}
    if len(nominal_origins) < 4:
        raise RuntimeError('Too few valid 1-based tiles for exact production mapping')
    predicted = _predict_marker_pixels(canvas_center_ds, canvas_radius_ds, geometry, angle_rad=float(config.get('_wafer_flat_angle_rad', 0.0) or 0.0))
    expected_pitch_ds = 400.0 * float(canvas_radius_ds) / max(geometry.radius, 1e-09)
    expected_square_ds = 0.5 * expected_pitch_ds
    expected_square_raw = expected_square_ds / max(0.5 * (scale_x + scale_y), 1e-09)

    def deduplicate(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        candidates.sort(key=lambda candidate: float(candidate['score']), reverse=True)
        kept: list[dict[str, Any]] = []
        for candidate in candidates:
            if all((float(np.linalg.norm(candidate['center'] - other['center'])) > 0.32 * expected_square_ds for other in kept)):
                kept.append(candidate)
        return kept

    def validate_fit(fit: dict[str, Any] | None, minimum_matches: int, maximum_ratio: float) -> bool:
        if fit is None:
            return False
        match_count = len(fit.get('matches', []))
        ratio = float(fit.get('mean_error', float('inf'))) / max(float(fit.get('pitch', 0.0)), 1e-09)
        return match_count >= minimum_matches and ratio <= maximum_ratio
    output: dict[str, dict[str, Any]] = {}
    fit_failures: dict[str, str] = {}
    for side in ('left', 'right'):
        marker_prediction = np.asarray(predicted[side], dtype=np.float64)
        ranked_tiles: list[tuple[float, tuple[int, int], Path]] = []
        for key, origin in nominal_origins.items():
            path = indexed[key]
            nearest_x = float(np.clip(marker_prediction[0], origin[0], origin[0] + tile_w_ds))
            nearest_y = float(np.clip(marker_prediction[1], origin[1], origin[1] + tile_h_ds))
            distance = math.hypot(marker_prediction[0] - nearest_x, marker_prediction[1] - nearest_y)
            ranked_tiles.append((distance, key, path))
        ranked_tiles.sort(key=lambda item: item[0])
        selected_tiles = ranked_tiles[:16]
        legacy_global: list[dict[str, Any]] = []
        unified_global: list[dict[str, Any]] = []
        tile_diagnostics: list[dict[str, Any]] = []
        for nominal_distance, key, path in selected_tiles:
            raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if raw is None:
                continue
            origin = nominal_origins[key]
            correlated_origin, match_score = _match_tile_origin_on_stitch(ds_canvas, raw, (tile_w_ds, tile_h_ds), origin)
            correlation_delta = (float(correlated_origin[0] - origin[0]), float(correlated_origin[1] - origin[1]))
            local_legacy = _raw_square_candidates_v13(raw, expected_square_raw)
            local_unified = _raw_square_candidates_unified(raw, expected_square_raw)
            for mode, local_candidates, destination in (('v13', local_legacy, legacy_global), ('unified', local_unified, unified_global)):
                for candidate in local_candidates:
                    center = np.asarray([origin[0] + float(candidate['center'][0]) * scale_x, origin[1] + float(candidate['center'][1]) * scale_y], dtype=np.float64)
                    if float(np.linalg.norm(center - marker_prediction)) > 6.2 * expected_pitch_ds:
                        continue
                    destination.append({'center': center, 'size': float(candidate['size']) * 0.5 * (scale_x + scale_y), 'score': float(candidate['score']) + 2.0 * max(match_score, 0.0), 'source': str(candidate.get('source', mode)), 'tile': path.name})
            tile_diagnostics.append({'tile': path.name, 'index': list(key), 'nominal_distance_px': float(nominal_distance), 'origin_ds_px': [float(origin[0]), float(origin[1])], 'origin_mode': 'v13 exact production stitch geometry', 'correlation_delta_px_not_applied': [float(correlation_delta[0]), float(correlation_delta[1])], 'match_score': float(match_score), 'legacy_candidate_count': len(local_legacy), 'unified_candidate_count': len(local_unified)})
        legacy_candidates = deduplicate(legacy_global)
        unified_candidates = deduplicate(unified_global)
        template = _raw_template_for_side(geometry, side)
        legacy_fit = _fit_raw_lattice_v13(legacy_candidates, expected_pitch_ds, predicted[side], template)
        unified_fit = _fit_raw_lattice_unified(unified_candidates, expected_pitch_ds, predicted[side], template)
        legacy_valid = validate_fit(legacy_fit, minimum_matches=9, maximum_ratio=0.12)
        unified_valid = validate_fit(unified_fit, minimum_matches=10, maximum_ratio=0.08)
        filled_count = sum((1 for candidate in unified_candidates if 'filled' in str(candidate.get('source', '')) and float(np.linalg.norm(candidate['center'] - marker_prediction)) <= 4.5 * expected_pitch_ds))
        optical_mode = 'filled' if filled_count >= 6 else 'outline'
        if optical_mode == 'filled' and legacy_valid:
            fit = legacy_fit
            chosen_candidates = legacy_candidates
            fit_mode = 'v13 exact filled-marker fit'
        elif unified_valid:
            fit = unified_fit
            chosen_candidates = unified_candidates
            fit_mode = 'unified ordered-row outline/filled fit'
        elif legacy_valid:
            fit = legacy_fit
            chosen_candidates = legacy_candidates
            fit_mode = 'v13 exact fallback'
        else:
            legacy_summary = None if legacy_fit is None else {'matches': len(legacy_fit.get('matches', [])), 'mean_error': float(legacy_fit.get('mean_error', float('nan'))), 'pitch': float(legacy_fit.get('pitch', float('nan')))}
            unified_summary = None if unified_fit is None else {'matches': len(unified_fit.get('matches', [])), 'mean_error': float(unified_fit.get('mean_error', float('nan'))), 'pitch': float(unified_fit.get('pitch', float('nan')))}
            fit_failures[side] = f'optical_mode={optical_mode}, filled_candidates={filled_count}, legacy={legacy_summary}, unified={unified_summary}'
            continue
        assert fit is not None
        match_count = len(fit['matches'])
        error_ratio = float(fit['mean_error']) / max(float(fit['pitch']), 1e-09)
        matched_sizes = [float(value) for value in fit.get('matched_sizes', [])]
        if not matched_sizes:
            matched_sizes = []
            for match in fit['matches']:
                candidate_index = int(match[1])
                if 0 <= candidate_index < len(chosen_candidates):
                    matched_sizes.append(float(chosen_candidates[candidate_index]['size']))
        if not matched_sizes:
            matched_sizes = [0.5 * float(fit['pitch'])]
        fit_angle_deg = ((math.degrees(math.atan2(float(np.asarray(fit['transform'])[1, 0]), float(np.asarray(fit['transform'])[0, 0]))) + 90.0) % 180.0) - 90.0
        output[side] = {'anchor': (float(fit['anchor'][0]), float(fit['anchor'][1])), 'template': _raw_fit_template(fit, matched_sizes), 'diagnostics': {'source': 'original-resolution tile lattice', 'optical_mode': optical_mode, 'fit_mode': fit_mode, 'algorithm_detail': str(fit.get('fit_mode', fit_mode)), 'filled_candidate_count': int(filled_count), 'match_count': int(match_count), 'pitch_ds_px': float(fit['pitch']), 'mean_error_ds_px': float(fit['mean_error']), 'rms_error_ds_px': float(fit.get('rms_error', fit['mean_error'])), 'error_pitch_ratio': error_ratio, 'angle_deg': fit_angle_deg, 'candidate_count': len(chosen_candidates), 'legacy_candidate_count': len(legacy_candidates), 'unified_candidate_count': len(unified_candidates), 'canvas_translation_refinement': {'applied': False, 'shift_ds_px': [0.0, 0.0], 'mode': 'disabled; preserve empirically correct v13 coordinate map'}, 'tiles': tile_diagnostics}}
    if fit_failures:
        fixed_matches = dict(output)
        try:
            # The repeated process rails near the LOR left marker can resemble the
            # optional directional rail more strongly than the marker itself.  On
            # LOR3 that moved the otherwise correct 12-square solution two columns
            # right.  Make the complete square lattice authoritative and consult
            # the rail only if the square-only left/right pair cannot validate.
            try:
                output = _run_raw_tile_edge_template_detector(indexed, predicted, expected_pitch_ds, exact_ds, (raw_w, raw_h), geometry, config, include_rail=False, fixed_matches=fixed_matches)
            except Exception as square_exc:
                try:
                    output = _run_raw_tile_edge_template_detector(indexed, predicted, expected_pitch_ds, exact_ds, (raw_w, raw_h), geometry, config, include_rail=True, fixed_matches=fixed_matches)
                    for side in ('left', 'right'):
                        output[side]['diagnostics']['square_template_failure'] = str(square_exc)
                except Exception as rail_exc:
                    raise RuntimeError(f'square-only template failed ({square_exc}); rail-aware template failed ({rail_exc})') from rail_exc
            for side in ('left', 'right'):
                output[side]['diagnostics']['candidate_lattice_failures'] = dict(fit_failures)
        except Exception as exc:
            summaries = '; '.join((f'{side}: {reason}' for side, reason in fit_failures.items()))
            raise RuntimeError(f'Raw tile candidate lattice failed ({summaries}); edge-template fallback failed ({exc})') from exc
    debug_dir = Path('future_alignment_debug')
    debug_dir.mkdir(exist_ok=True)
    debug_report = {'adapter_version': ALIGNMENT_VERSION, 'tile_folder': str(tile_folder), 'ds_canvas_size': [int(ds_canvas.shape[1]), int(ds_canvas.shape[0])], 'ds_factor': float(exact_ds), 'mapping_mode': 'v13 exact production index/overlap map; no canvas nudge', 'configured_tile_size': [int(configured_w), int(configured_h)], 'downscaled_tile_size': [int(tile_w_ds), int(tile_h_ds)], 'raw_step_px': [float(step_x_raw), float(step_y_raw)], 'canvas_center_ds': [float(canvas_center_ds[0]), float(canvas_center_ds[1])], 'canvas_radius_ds': float(canvas_radius_ds), 'markers': {side: {'anchor_ds_px': [float(output[side]['anchor'][0]), float(output[side]['anchor'][1])], **output[side]['diagnostics']} for side in ('left', 'right')}}
    (debug_dir / 'raw_tile_lattice_latest.json').write_text(json.dumps(debug_report, indent=2), encoding='utf-8')
    for side in ('left', 'right'):
        anchor = output[side]['anchor']
        template_geometry = output[side]['template']
        area = template_geometry['boxes'].get(('area', 0), [])
        if area:
            xs = [float(point[0]) for point in area]
            ys = [float(point[1]) for point in area]
            padding = max(25, int(round(2.0 * output[side]['diagnostics']['pitch_ds_px'])))
            x0 = max(0, int(math.floor(min(xs) - padding)))
            x1 = min(ds_canvas.shape[1], int(math.ceil(max(xs) + padding)))
            y0 = max(0, int(math.floor(min(ys) - padding)))
            y1 = min(ds_canvas.shape[0], int(math.ceil(max(ys) + padding)))
            if x1 > x0 and y1 > y0:
                crop = ds_canvas[y0:y1, x0:x1].copy()
                for key, corners in template_geometry['boxes'].items():
                    points = np.asarray([[float(x) - x0, float(y) - y0] for x, y in corners], dtype=np.int32)
                    cv2.polylines(crop, [points], True, (0, 255, 0), 2 if key == ('area', 0) else 1, cv2.LINE_AA)
                cv2.drawMarker(crop, (int(round(anchor[0] - x0)), int(round(anchor[1] - y0))), (0, 0, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
                cv2.imwrite(str(debug_dir / f'raw_tile_{side}_latest.png'), crop)
    return output

def _refine_raw_details_with_production_snap(ds_canvas: np.ndarray, raw_details: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Use the production UI score to correct a strong full-marker AUTO mismatch.

    Raw edge correlation is useful for generating hypotheses, but process rails can
    give a partial grid a plausible score.  The production snap scorer has a much
    cleaner separation on these wafers (roughly 0.6 for the decoy versus 0.9+ for
    the complete marker), so it is the final AUTO acceptance gate as well.
    """
    from alignment_review import find_local_template_snap

    def transform(points: list[tuple[float, float]], center: tuple[float, float], result: dict[str, Any]) -> list[tuple[float, float]]:
        theta = math.radians(float(result['angle_deg']))
        cosine, sine = math.cos(theta), math.sin(theta)
        scale = float(result['scale'])
        dx, dy = float(result['dx']), float(result['dy'])
        cx, cy = float(center[0]), float(center[1])
        transformed: list[tuple[float, float]] = []
        for x, y in points:
            rx, ry = (float(x) - cx) * scale, (float(y) - cy) * scale
            transformed.append((cx + cosine * rx - sine * ry + dx, cy + sine * rx + cosine * ry + dy))
        return transformed

    for side in ('left', 'right'):
        detail = raw_details.get(side)
        if not detail:
            continue
        template = detail.get('template', {}) or {}
        boxes = template.get('boxes', {}) or {}
        squares = template.get('squares', {}) or {}
        anchor = tuple(detail.get('anchor', ()))
        if len(anchor) != 2:
            continue
        result = find_local_template_snap(ds_canvas, boxes, squares, anchor)
        score = float(result.get('score', 0.0) or 0.0)
        baseline = float(result.get('baseline_score', 0.0) or 0.0)
        improvement = float(result.get('improvement', score - baseline) or 0.0)
        # Correct LOR poses score 0.9+ even when mildly damaged. Requiring both
        # high absolute confidence and a large gain prevents a perfect right-side
        # result from being changed for a negligible numerical improvement.
        applied = bool(result.get('ok')) and score >= 0.88 and improvement >= 0.10
        refinement = {
            'applied': applied,
            'score': score,
            'baseline_score': baseline,
            'improvement': improvement,
            'move_ds_px': [float(result.get('dx', 0.0) or 0.0), float(result.get('dy', 0.0) or 0.0)],
            'scale': float(result.get('scale', 1.0) or 1.0),
            'angle_deg': float(result.get('angle_deg', 0.0) or 0.0),
            'acceptance': 'score >= 0.88 and improvement >= 0.10',
        }
        detail.setdefault('diagnostics', {})['production_snap_refinement'] = refinement
        if not applied:
            continue
        detail['template'] = {
            'boxes': {key: transform(corners, anchor, result) for key, corners in boxes.items()},
            'squares': {key: transform([point], anchor, result)[0] for key, point in squares.items()},
        }
        detail['anchor'] = (float(anchor[0]) + float(result['dx']), float(anchor[1]) + float(result['dy']))
        detail['diagnostics']['source'] = f"{detail['diagnostics'].get('source', 'raw detector')} + production-score AUTO refinement"
        detail['diagnostics']['fit_mode'] = f"{detail['diagnostics'].get('fit_mode', 'raw fit')} + high-confidence full-template snap"
    return raw_details

def _similarity_from_two_points(physical_gds_frame: dict[str, tuple[float, float]], nominal: dict[str, tuple[float, float]]) -> tuple[float, float, float, float]:
    p_left = np.asarray(physical_gds_frame['left'], dtype=np.float64)
    p_right = np.asarray(physical_gds_frame['right'], dtype=np.float64)
    q_left = np.asarray(nominal['left'], dtype=np.float64)
    q_right = np.asarray(nominal['right'], dtype=np.float64)
    vp = p_right - p_left
    vq = q_right - q_left
    lp = float(np.linalg.norm(vp))
    lq = float(np.linalg.norm(vq))
    if lp <= 1e-09 or lq <= 1e-09:
        raise RuntimeError('Degenerate compact-marker pair')
    scale = lq / lp
    angle = math.atan2(vq[1], vq[0]) - math.atan2(vp[1], vp[0])
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    translation = 0.5 * (q_left + q_right) - scale * rotation @ (0.5 * (p_left + p_right))
    return (float(angle), float(translation[0]), float(translation[1]), float(scale))

def _estimate_review_wafer(image_bgr: np.ndarray) -> tuple[tuple[float, float], float]:
    """Estimate the circular wafer frame without using contour centroids.

    The old helper used the filled-contour centroid, which is biased upward by
    the wafer flat.  We use the horizontal diameter and top arc instead.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)
    _threshold, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates: list[tuple[float, tuple[float, float], float]] = []
    for mask in (bright, cv2.bitwise_not(bright)):
        kernel_side = max(5, int(round(0.01 * min(h, w))) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_side, kernel_side))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _hierarchy = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 0.15 * h * w:
                continue
            points = contour.reshape(-1, 2).astype(np.float64)
            x_min, y_min = np.min(points, axis=0)
            x_max, _y_max = np.max(points, axis=0)
            radius = 0.5 * float(x_max - x_min)
            if radius < 0.25 * min(h, w):
                continue
            center_x = 0.5 * float(x_min + x_max)
            center_y = float(y_min + radius)
            center_offset = math.hypot(center_x - w / 2.0, center_y - h / 2.0)
            score = area - 0.2 * center_offset * min(h, w)
            candidates.append((score, (center_x, center_y), radius))
    if candidates:
        _score, center, radius = max(candidates, key=lambda item: item[0])
        return (center, radius)
    return (((w - 1) * 0.5, (h - 1) * 0.5), 0.48 * min(w, h))

def _project_gds_point_to_pixels(point_gds: tuple[float, float] | np.ndarray, center_px: tuple[float, float], radius_px: float, geometry: DesignGeometry, angle_rad: float=0.0) -> tuple[float, float]:
    """Project one GDS point into the shared stitched-image coordinate frame."""
    scale = float(radius_px) / max(geometry.radius, 1e-09)
    cosine = math.cos(float(angle_rad))
    sine = math.sin(float(angle_rad))
    gx, gy = float(point_gds[0]), float(point_gds[1])
    dx = (gx - geometry.center[0]) * scale
    dy = -(gy - geometry.center[1]) * scale
    return (float(center_px[0] + cosine * dx - sine * dy), float(center_px[1] + sine * dx + cosine * dy))

def _predict_marker_pixels(center_px: tuple[float, float], radius_px: float, geometry: DesignGeometry, angle_rad: float=0.0) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for side, (gx, gy) in geometry.marker_centers.items():
        result[side] = _project_gds_point_to_pixels((gx, gy), center_px, radius_px, geometry, angle_rad)
    return result

def _make_marker_tester(original_tester_class: type, geometry: DesignGeometry, runtime_state: dict[str, Any]):
    """Create a dual-panel marker review UI with persistent left/right boxes."""

    class FutureMarkerReviewTester(original_tester_class):

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._future_review_ready = False
            ds_canvas = runtime_state.get('ds_canvas')
            if not (isinstance(ds_canvas, np.ndarray) and ds_canvas.size):
                raise RuntimeError('Marker review requires the production downscaled stitch')
            image_path = kwargs.get('image_path', args[0] if args else '')
            display_height = int(kwargs.get('display_height', 800))
            debug = bool(kwargs.get('debug', False))
            self.image_path = str(image_path)
            self.wafer_id = str(runtime_state.get('wafer_id') or Path(self.image_path).name or 'Unknown wafer')
            self.target_height = display_height
            self.debug = debug
            self.im = _ArrayReviewImage(ds_canvas)
            self.orig_w, self.orig_h = self.im.size
            self.sidebar_w, self.top_bar_h, self.bottom_bar_h = (320, 60, 80)
            master_reduce = max(1, self.orig_h // (self.target_height * 2))
            self.master_gray = np.array(self.im.reduce(master_reduce).convert('L'))
            self.recompute_display_sizes()
            self.STATE_IDLE, self.STATE_WAIT_LEFT, self.STATE_WAIT_RIGHT, self.STATE_FINISHED = (0, 1, 2, 3)
            self.current_state = self.STATE_IDLE
            self.status_text = "STATUS: IDLE. CLICK 'START ALIGNMENT' TO BEGIN"
            self.status_bg_color = (15, 15, 15)
            self.left_click_global = self.right_click_global = None
            self.left_marker_global = self.right_marker_global = None
            self.left_resolved_success = self.right_resolved_success = False
            self.left_squares_global = self.right_squares_global = None
            self.left_boxes_global = self.right_boxes_global = None
            self.calibrated_theta = None
            self.calibrated_scale_x = None
            self.calibrated_scale_y = None
            self.expected_col_spacing = None
            self._future_review_coordinate_frame = 'production downscaled stitch'
            self._future_review_ds_factor = float(runtime_state.get('ds_factor', 1.0) or 1.0)
            self.redraw_gui()
            self.sidebar_w = 370
            self.recompute_display_sizes()
            self.left_boxes_global = {}
            self.right_boxes_global = {}
            self.left_squares_global = {}
            self.right_squares_global = {}
            self.left_was_manual = False
            self.right_was_manual = False
            self.auto_detection_error: str | None = None
            self._auto_left_boxes_global: dict[Any, Any] = {}
            self._auto_right_boxes_global: dict[Any, Any] = {}
            self._auto_left_squares_global: dict[Any, Any] = {}
            self._auto_right_squares_global: dict[Any, Any] = {}
            self._manual_edit_enabled = False
            self._panel_view_centers: dict[str, tuple[float, float]] = {}
            self._panel_maps: dict[str, dict[str, Any]] = {}
            self._panel_view_sizes: dict[str, tuple[float, float]] = {}
            self._panel_drag_side: str | None = None
            self._panel_drag_mode: str | None = None
            self._panel_drag_last_global: tuple[float, float] | None = None
            self._panel_drag_start_global: tuple[float, float] | None = None
            self._panel_drag_start_radius: float | None = None
            self._panel_drag_start_angle: float | None = None
            self._panel_drag_baseline_boxes: dict[Any, Any] | None = None
            self._panel_drag_baseline_squares: dict[Any, Any] | None = None
            self._manual_similarity_edited = {'left': False, 'right': False}
            self._manual_template_scale = {'left': 1.0, 'right': 1.0}
            self._manual_template_rotation_deg = {'left': 0.0, 'right': 0.0}
            self._future_review_ready = True
            self._initialize_future_defaults()

        def _review_image(self) -> tuple[np.ndarray, float]:
            max_dimension = 6000.0
            reduce_factor = max(1, int(math.ceil(max(self.orig_w, self.orig_h) / max_dimension)))
            pil_image = self.im.reduce(reduce_factor).convert('RGB')
            rgb = np.asarray(pil_image, dtype=np.uint8)
            return (cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), float(reduce_factor))

        @staticmethod
        def _axis_aligned_area_box(boxes: dict[Any, list[tuple[float, float]]], padding: float) -> list[tuple[float, float]]:
            points = [point for key, corners in boxes.items() if key != ('area', 0) for point in corners]
            if not points:
                return []
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            x0, x1 = (min(xs) - padding, max(xs) + padding)
            y0, y1 = (min(ys) - padding, max(ys) + padding)
            return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

        def _gds_template_global(self, center_review: tuple[float, float], radius_review: float, factor: float, angle_rad: float=0.0) -> dict[str, dict[str, Any]]:
            """Project the exact GDS square polygons into review coordinates."""
            scale = float(radius_review) / max(float(geometry.radius), 1e-09)
            result: dict[str, dict[str, Any]] = {}
            for side in ('left', 'right'):
                square_polygons: list[np.ndarray] = []
                for polygon in geometry.marker_polygons:
                    x0, y0 = np.min(polygon, axis=0)
                    x1, y1 = np.max(polygon, axis=0)
                    width = float(x1 - x0)
                    height = float(y1 - y0)
                    cx = float((x0 + x1) * 0.5)
                    if not (width < 500.0 and height < 500.0):
                        continue
                    if side == 'left' and cx >= geometry.center[0] or (side == 'right' and cx < geometry.center[0]):
                        continue
                    square_polygons.append(polygon)
                square_polygons.sort(key=lambda poly: (-float(np.mean(poly[:, 1])), float(np.mean(poly[:, 0]))))
                boxes: dict[Any, list[tuple[float, float]]] = {}
                squares: dict[Any, tuple[float, float]] = {}
                for index, polygon in enumerate(square_polygons):
                    global_corners: list[tuple[float, float]] = []
                    for gx, gy in polygon:
                        rx, ry = _project_gds_point_to_pixels((gx, gy), center_review, radius_review, geometry, angle_rad)
                        global_corners.append((rx * factor, ry * factor))
                    key = (index // 3, index % 3)
                    boxes[key] = global_corners
                    squares[key] = (float(np.mean([point[0] for point in global_corners])), float(np.mean([point[1] for point in global_corners])))
                pitch_global = 400.0 * scale * factor
                area = self._axis_aligned_area_box(boxes, padding=max(20.0, 0.55 * pitch_global))
                if area:
                    boxes['area', 0] = area
                result[side] = {'boxes': boxes, 'squares': squares}
            return result

        def _detected_template_global(self, detail: dict[str, Any], normalized_to_review: np.ndarray, factor: float) -> dict[str, Any]:
            detection = detail['detection']
            sx = float(detail.get('work_to_input_x', 1.0))
            sy = float(detail.get('work_to_input_y', 1.0))
            matched = getattr(detection, 'matched_square_centers_px', None) or []
            predicted = getattr(detection, 'predicted_square_centers_px', None) or []
            centers_to_draw = matched if len(matched) >= 10 else predicted
            square_size = float(getattr(detection, 'square_size_px', 0.0) or 0.0)
            angle_deg = float(getattr(detection, 'angle_deg', 0.0) or 0.0)
            angle = math.radians(angle_deg)
            ex = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
            ey = np.asarray([-math.sin(angle), math.cos(angle)], dtype=np.float64)
            boxes: dict[Any, list[tuple[float, float]]] = {}
            squares: dict[Any, tuple[float, float]] = {}
            for index, raw_center in enumerate(centers_to_draw):
                center_input = np.asarray([float(raw_center[0]) * sx, float(raw_center[1]) * sy], dtype=np.float64)
                half_x = max(1.0, 0.5 * square_size * sx)
                half_y = max(1.0, 0.5 * square_size * sy)
                normalized_corners = [center_input - half_x * ex - half_y * ey, center_input + half_x * ex - half_y * ey, center_input + half_x * ex + half_y * ey, center_input - half_x * ex + half_y * ey]
                global_corners: list[tuple[float, float]] = []
                for corner in normalized_corners:
                    review = normalized_to_review @ np.asarray([corner[0], corner[1], 1.0], dtype=np.float64)
                    global_corners.append((float(review[0] * factor), float(review[1] * factor)))
                review_center = normalized_to_review @ np.asarray([center_input[0], center_input[1], 1.0], dtype=np.float64)
                key = (index // 3, index % 3)
                boxes[key] = global_corners
                squares[key] = (float(review_center[0] * factor), float(review_center[1] * factor))
            if boxes:
                pitch = float(getattr(detection, 'pitch_original_px', 0.0) or 0.0)
                if pitch <= 0.0:
                    pitch = float(getattr(detection, 'pitch_px', 0.0) or 0.0) * 0.5 * (sx + sy)
                area = self._axis_aligned_area_box(boxes, padding=max(20.0, 0.55 * pitch * factor))
                if area:
                    boxes['area', 0] = area
            return {'boxes': boxes, 'squares': squares}

        def _set_side_geometry(self, side: str, marker: tuple[float, float], template: dict[str, Any]) -> None:
            setattr(self, f'{side}_marker_global', tuple(marker))
            setattr(self, f'{side}_click_global', tuple(marker))
            setattr(self, f'{side}_boxes_global', copy.deepcopy(template.get('boxes', {})))
            setattr(self, f'{side}_squares_global', copy.deepcopy(template.get('squares', {})))
            if hasattr(self, '_panel_view_centers'):
                self._panel_view_centers.setdefault(side, tuple(marker))

        def _set_panel_view_center(self, side: str, center: tuple[float, float]) -> None:
            self._panel_view_centers[side] = (float(center[0]), float(center[1]))

        def _move_side_geometry(self, side: str, new_marker: tuple[float, float], *, update_panel_view: bool=False) -> None:
            old_marker = getattr(self, f'{side}_marker_global', None)
            if old_marker is None:
                old_marker = new_marker
            dx = float(new_marker[0]) - float(old_marker[0])
            dy = float(new_marker[1]) - float(old_marker[1])
            boxes = getattr(self, f'{side}_boxes_global', {}) or {}
            squares = getattr(self, f'{side}_squares_global', {}) or {}
            moved_boxes = {key: [(float(x) + dx, float(y) + dy) for x, y in corners] for key, corners in boxes.items()}
            moved_squares = {key: (float(x) + dx, float(y) + dy) for key, (x, y) in squares.items()}
            setattr(self, f'{side}_boxes_global', moved_boxes)
            setattr(self, f'{side}_squares_global', moved_squares)
            setattr(self, f'{side}_marker_global', tuple(new_marker))
            setattr(self, f'{side}_click_global', tuple(new_marker))
            if update_panel_view:
                self._set_panel_view_center(side, tuple(new_marker))

        @staticmethod
        def _transform_points_about_center(points: list[tuple[float, float]], center: tuple[float, float], scale_factor: float, angle_delta: float) -> list[tuple[float, float]]:
            """Apply a uniform scale and rotation about ``center``."""
            cosine = math.cos(float(angle_delta))
            sine = math.sin(float(angle_delta))
            cx, cy = (float(center[0]), float(center[1]))
            transformed: list[tuple[float, float]] = []
            for x, y in points:
                dx = (float(x) - cx) * float(scale_factor)
                dy = (float(y) - cy) * float(scale_factor)
                transformed.append((cx + cosine * dx - sine * dy, cy + sine * dx + cosine * dy))
            return transformed

        def _apply_side_similarity_from_drag(self, side: str, *, scale_factor: float, angle_delta: float) -> None:
            """Transform the complete marker template from the drag baseline."""
            marker = getattr(self, f'{side}_marker_global', None)
            baseline_boxes = self._panel_drag_baseline_boxes
            baseline_squares = self._panel_drag_baseline_squares
            if marker is None or baseline_boxes is None or baseline_squares is None:
                return
            transformed_boxes = {key: self._transform_points_about_center(corners, marker, scale_factor, angle_delta) for key, corners in baseline_boxes.items()}
            transformed_squares: dict[Any, tuple[float, float]] = {}
            for key, point in baseline_squares.items():
                transformed = self._transform_points_about_center([point], marker, scale_factor, angle_delta)
                transformed_squares[key] = transformed[0]
            setattr(self, f'{side}_boxes_global', transformed_boxes)
            setattr(self, f'{side}_squares_global', transformed_squares)
            setattr(self, f'{side}_marker_global', tuple(marker))
            setattr(self, f'{side}_click_global', tuple(marker))

        def _panel_edit_hit(self, side: str, x: int, y: int) -> str:
            """Choose direct-manipulation mode from the visible panel handles."""
            mapping = self._panel_maps.get(side) or {}
            pointer = np.asarray([float(x), float(y)], dtype=np.float64)
            rotation_handle = mapping.get('rotation_handle_screen')
            if rotation_handle is not None:
                distance = float(np.linalg.norm(pointer - np.asarray(rotation_handle, dtype=np.float64)))
                if distance <= 16.0:
                    return 'rotate'
            for handle in mapping.get('scale_handles_screen', []):
                distance = float(np.linalg.norm(pointer - np.asarray(handle, dtype=np.float64)))
                if distance <= 15.0:
                    return 'scale'
            return 'translate'

        def _panel_hit_side(self, x: int, y: int) -> str | None:
            for side in ('left', 'right'):
                mapping = self._panel_maps.get(side)
                if not mapping:
                    continue
                x0, y0, x1, y1 = mapping['rect']
                if x0 <= x <= x1 and y0 <= y <= y1:
                    return side
            return None

        def _panel_point_to_global(self, side: str, x: int, y: int) -> tuple[float, float] | None:
            mapping = self._panel_maps.get(side)
            if not mapping:
                return None
            panel_x0, panel_y0, _panel_x1, _panel_y1 = mapping['rect']
            crop_x1, crop_y1, _crop_x2, _crop_y2 = mapping['crop']
            scale_x = float(mapping['scale_x'])
            scale_y = float(mapping['scale_y'])
            if scale_x <= 0.0 or scale_y <= 0.0:
                return None
            return (float(crop_x1) + (float(x) - float(panel_x0)) / scale_x, float(crop_y1) + (float(y) - float(panel_y0)) / scale_y)

        def _begin_panel_drag(self, side: str, x: int, y: int) -> bool:
            point = self._panel_point_to_global(side, x, y)
            if point is None:
                return False
            marker = getattr(self, f'{side}_marker_global', None)
            if marker is None:
                return False
            mode = self._panel_edit_hit(side, x, y)
            self._panel_drag_side = side
            self._panel_drag_mode = mode
            self._panel_drag_last_global = point
            self._panel_drag_start_global = point
            self._panel_drag_baseline_boxes = copy.deepcopy(getattr(self, f'{side}_boxes_global', {}) or {})
            self._panel_drag_baseline_squares = copy.deepcopy(getattr(self, f'{side}_squares_global', {}) or {})
            vector_x = float(point[0]) - float(marker[0])
            vector_y = float(point[1]) - float(marker[1])
            self._panel_drag_start_radius = max(1e-06, math.hypot(vector_x, vector_y))
            self._panel_drag_start_angle = math.atan2(vector_y, vector_x)
            setattr(self, f'{side}_was_manual', True)
            setattr(self, f'{side}_resolved_success', True)
            self.status_bg_color = (80, 45, 0)
            action = {'translate': 'MOVING', 'scale': 'SCALING', 'rotate': 'ROTATING'}[mode]
            self.status_text = f'{action} {side.upper()} TEMPLATE. RELEASE TO KEEP. BODY=MOVE, CORNERS=SCALE, CIRCLE=ROTATE'
            self.redraw_gui()
            return True

        def _continue_panel_drag(self, x: int, y: int) -> None:
            side = self._panel_drag_side
            mode = self._panel_drag_mode
            previous = self._panel_drag_last_global
            if side is None or mode is None or previous is None:
                return
            current = self._panel_point_to_global(side, x, y)
            if current is None:
                return
            marker = getattr(self, f'{side}_marker_global', None)
            if marker is None:
                return
            if mode == 'translate':
                dx = float(current[0]) - float(previous[0])
                dy = float(current[1]) - float(previous[1])
                if abs(dx) > 1e-09 or abs(dy) > 1e-09:
                    self._move_side_geometry(side, (float(marker[0]) + dx, float(marker[1]) + dy), update_panel_view=False)
                    self._panel_drag_last_global = current
                    self.redraw_gui()
                return
            start_radius = float(self._panel_drag_start_radius or 1.0)
            start_angle = float(self._panel_drag_start_angle or 0.0)
            vector_x = float(current[0]) - float(marker[0])
            vector_y = float(current[1]) - float(marker[1])
            if mode == 'scale':
                current_radius = max(1e-06, math.hypot(vector_x, vector_y))
                scale_factor = float(np.clip(current_radius / start_radius, 0.25, 4.0))
                angle_delta = 0.0
                self._manual_similarity_edited[side] = True
                self._manual_template_scale[side] = scale_factor
            else:
                current_angle = math.atan2(vector_y, vector_x)
                angle_delta = math.atan2(math.sin(current_angle - start_angle), math.cos(current_angle - start_angle))
                scale_factor = 1.0
                self._manual_similarity_edited[side] = True
                self._manual_template_rotation_deg[side] = math.degrees(angle_delta)
            self._apply_side_similarity_from_drag(side, scale_factor=scale_factor, angle_delta=angle_delta)
            self._panel_drag_last_global = current
            self.redraw_gui()

        def _finish_panel_drag(self) -> None:
            side = self._panel_drag_side
            mode = self._panel_drag_mode
            if side is None:
                return
            self._panel_drag_side = None
            self._panel_drag_mode = None
            self._panel_drag_last_global = None
            self._panel_drag_start_global = None
            self._panel_drag_start_radius = None
            self._panel_drag_start_angle = None
            self._panel_drag_baseline_boxes = None
            self._panel_drag_baseline_squares = None
            self.status_bg_color = (0, 85, 0)
            action = {'translate': 'MOVED', 'scale': 'SCALED', 'rotate': 'ROTATED', None: 'EDITED'}.get(mode, 'EDITED')
            if self.current_state == self.STATE_WAIT_LEFT:
                self.status_text = f'{side.upper()} TEMPLATE {action}. CLICK LEFT ON WAFER FOR A NEW ABSOLUTE START, OR EDIT EITHER PANEL AGAIN'
            elif self.current_state == self.STATE_WAIT_RIGHT:
                self.status_text = f'{side.upper()} TEMPLATE {action}. CLICK RIGHT ON WAFER FOR A NEW ABSOLUTE START, OR EDIT EITHER PANEL AGAIN'
            else:
                self.status_text = 'MANUAL TEMPLATES READY. BODY=MOVE, CORNERS=SCALE, CIRCLE=ROTATE; EXIT TO ACCEPT'
            self.redraw_gui()

        def _initialize_future_defaults(self) -> None:
            review_bgr, factor = self._review_image()
            if self._future_review_coordinate_frame == 'production downscaled stitch' and runtime_state.get('canvas_center_ds') is not None and (runtime_state.get('canvas_radius_ds') is not None):
                center_ds = runtime_state['canvas_center_ds']
                center = (float(center_ds[0]) / factor, float(center_ds[1]) / factor)
                radius = float(runtime_state['canvas_radius_ds']) / factor
            else:
                center, radius = _estimate_review_wafer(review_bgr)
            self._review_reduce_factor = float(factor)
            self._review_wafer_center_global = (float(center[0] * factor), float(center[1] * factor))
            self._review_wafer_radius_global = float(radius * factor)
            predicted = _predict_marker_pixels(center, radius, geometry, angle_rad=float(runtime_state.get('wafer_flat_angle_rad', 0.0) or 0.0))
            selected = dict(predicted)
            source = 'GDS/metrology predictions'
            templates = self._gds_template_global(center, radius, factor, angle_rad=float(runtime_state.get('wafer_flat_angle_rad', 0.0) or 0.0))
            raw_error: str | None = None
            tile_folder = runtime_state.get('tile_folder')
            ds_canvas = runtime_state.get('ds_canvas')
            center_ds = runtime_state.get('canvas_center_ds')
            radius_ds = runtime_state.get('canvas_radius_ds')
            ds_factor = float(runtime_state.get('ds_factor', 1.0) or 1.0)
            if tile_folder is not None and isinstance(ds_canvas, np.ndarray) and ds_canvas.size and (center_ds is not None) and (radius_ds is not None):
                try:
                    raw_details = _run_raw_tile_lattice_detector(tile_folder, ds_canvas, (float(center_ds[0]), float(center_ds[1])), float(radius_ds), geometry, ds_factor, runtime_state.get('stitch_config_run', {}))
                    raw_details = _refine_raw_details_with_production_snap(ds_canvas, raw_details)
                    if set(raw_details) == {'left', 'right'}:
                        selected = {side: (float(raw_details[side]['anchor'][0]) / factor, float(raw_details[side]['anchor'][1]) / factor) for side in ('left', 'right')}
                        templates = {side: raw_details[side]['template'] for side in ('left', 'right')}
                        runtime_state['marker_lattice_fit'] = {side: raw_details[side]['diagnostics'] for side in ('left', 'right')}
                        source = 'original-resolution raw-tile lattice detections with exact production mapping'
                        for side in ('left', 'right'):
                            diagnostics = raw_details[side]['diagnostics']
                            error = diagnostics.get('mean_error_ds_px')
                            quality = f"error={float(error):.3f} ds px" if error is not None else f"correlation={float(diagnostics.get('template_correlation', 0.0)):.3f}"
                            refinement = diagnostics.get('production_snap_refinement', {}) or {}
                            snap_quality = f", production_snap={float(refinement.get('score', 0.0)):.3f}, improvement={float(refinement.get('improvement', 0.0)):+.3f}, applied={bool(refinement.get('applied', False))}"
                            print(f"[Future Raw-Tile Align v23] {side}: mode={diagnostics.get('optical_mode', 'unknown')}, fit={diagnostics.get('fit_mode', 'unknown')}, {diagnostics['match_count']}/12 squares, pitch={diagnostics['pitch_ds_px']:.3f} ds px, {quality}{snap_quality}, canvas_shift=(+0.000,+0.000) px")
                except Exception as exc:
                    raw_error = str(exc)
                    print(f'[Future Raw-Tile Align v23] FAILED: {raw_error}')
            if source == 'GDS/metrology predictions':
                try:
                    normalized, normalized_to_review = _normalized_wafer_view(review_bgr, center, radius, geometry.design_bbox)
                    details = _run_compact_detector_details(normalized, geometry)
                    detected_review: dict[str, tuple[float, float]] = {}
                    fit_diagnostics: dict[str, Any] = {}
                    for side, detail in details.items():
                        anchor_review, fitted_template, diagnostics = _fitted_detection_geometry(detail, normalized_to_review, factor)
                        detected_review[side] = anchor_review
                        templates[side] = fitted_template
                        fit_diagnostics[side] = diagnostics
                    runtime_state['marker_lattice_fit'] = fit_diagnostics
                    if set(detected_review) == {'left', 'right'}:
                        selected = detected_review
                        source = 'downscaled compact-marker detections'
                except Exception as exc:
                    errors = [error for error in (raw_error, str(exc)) if error]
                    self.auto_detection_error = ' | '.join(errors)
            self._auto_left_global = (selected['left'][0] * factor, selected['left'][1] * factor)
            self._auto_right_global = (selected['right'][0] * factor, selected['right'][1] * factor)
            self._set_side_geometry('left', self._auto_left_global, templates['left'])
            self._set_side_geometry('right', self._auto_right_global, templates['right'])
            self._set_panel_view_center('left', self._auto_left_global)
            self._set_panel_view_center('right', self._auto_right_global)
            self._auto_left_boxes_global = copy.deepcopy(self.left_boxes_global)
            self._auto_right_boxes_global = copy.deepcopy(self.right_boxes_global)
            self._auto_left_squares_global = copy.deepcopy(self.left_squares_global)
            self._auto_right_squares_global = copy.deepcopy(self.right_squares_global)
            self.left_was_manual = False
            self.right_was_manual = False
            self.left_resolved_success = True
            self.right_resolved_success = True
            self.current_state = self.STATE_FINISHED
            self.status_bg_color = (0, 85, 0)
            self.status_text = 'BOTH AUTO MARKERS SHOWN. EXIT TO ACCEPT OR START ALIGNMENT TO OVERRIDE'
            print(f'[Future Marker Review] Loaded {source}; displaying both marker boxes.')
            if self.auto_detection_error:
                print(f'[Future Marker Review] Automatic detector unavailable; showing predicted defaults ({self.auto_detection_error}).')
            print('[Future Marker Review] Both LEFT and RIGHT zooms remain visible. START ALIGNMENT keeps the coarse full-wafer LEFT/RIGHT clicks. In either zoom panel: drag the body to move, a corner to scale, or the circular handle to rotate. RESET SYSTEM restores automatic boxes.')
            self.redraw_gui()

        def _restore_auto_defaults(self) -> None:
            self._manual_edit_enabled = False
            self._panel_drag_side = None
            self._panel_drag_mode = None
            self._panel_drag_last_global = None
            self._panel_drag_start_global = None
            self._panel_drag_start_radius = None
            self._panel_drag_start_angle = None
            self._panel_drag_baseline_boxes = None
            self._panel_drag_baseline_squares = None
            self._panel_maps = {}
            self._panel_view_sizes = {}
            self._manual_similarity_edited = {'left': False, 'right': False}
            self._manual_template_scale = {'left': 1.0, 'right': 1.0}
            self._manual_template_rotation_deg = {'left': 0.0, 'right': 0.0}
            self.left_marker_global = self._auto_left_global
            self.right_marker_global = self._auto_right_global
            self.left_click_global = self.left_marker_global
            self.right_click_global = self.right_marker_global
            self.left_boxes_global = copy.deepcopy(self._auto_left_boxes_global)
            self.right_boxes_global = copy.deepcopy(self._auto_right_boxes_global)
            self.left_squares_global = copy.deepcopy(self._auto_left_squares_global)
            self.right_squares_global = copy.deepcopy(self._auto_right_squares_global)
            self._set_panel_view_center('left', self._auto_left_global)
            self._set_panel_view_center('right', self._auto_right_global)
            self.left_was_manual = False
            self.right_was_manual = False
            self.left_resolved_success = True
            self.right_resolved_success = True
            self.current_state = self.STATE_FINISHED
            self.status_bg_color = (0, 85, 0)
            self.status_text = 'AUTOMATIC BOXES RESTORED. EXIT TESTER TO ACCEPT'
            self.redraw_gui()

        def process_wafer_click(self, x: int, y: int) -> None:
            if self.current_state not in (self.STATE_WAIT_LEFT, self.STATE_WAIT_RIGHT):
                return
            orig_x = float(x / self.scale)
            orig_y = float(y / self.scale)
            if self.current_state == self.STATE_WAIT_LEFT:
                self._move_side_geometry('left', (orig_x, orig_y), update_panel_view=True)
                self.left_was_manual = True
                self.left_resolved_success = True
                self.current_state = self.STATE_WAIT_RIGHT
                self.status_bg_color = (80, 45, 0)
                self.status_text = 'LEFT ABSOLUTE START SET. CLICK RIGHT ON WAFER; EDIT EITHER PANEL: BODY=MOVE, CORNERS=SCALE, CIRCLE=ROTATE'
            else:
                self._move_side_geometry('right', (orig_x, orig_y), update_panel_view=True)
                self.right_was_manual = True
                self.right_resolved_success = True
                self.current_state = self.STATE_FINISHED
                self.status_bg_color = (0, 85, 0)
                self.status_text = 'BOTH ABSOLUTE STARTS SET. BODY=MOVE, CORNERS=SCALE, CIRCLE=ROTATE; EXIT TO CONTINUE'
            self.redraw_gui()

        @staticmethod
        def _draw_polyline(image: np.ndarray, corners: list[tuple[int, int]], color: tuple[int, int, int], thickness: int, dashed: bool=False) -> None:
            if len(corners) < 2:
                return
            if not dashed:
                cv2.polylines(image, [np.asarray(corners, dtype=np.int32).reshape((-1, 1, 2))], True, color, thickness, cv2.LINE_AA)
                return
            for index in range(len(corners)):
                p0 = np.asarray(corners[index], dtype=np.float64)
                p1 = np.asarray(corners[(index + 1) % len(corners)], dtype=np.float64)
                length = float(np.linalg.norm(p1 - p0))
                if length <= 1e-09:
                    continue
                segments = max(1, int(length // 10))
                for segment in range(0, segments, 2):
                    a = p0 + (p1 - p0) * (segment / segments)
                    b = p0 + (p1 - p0) * (min(segment + 1, segments) / segments)
                    cv2.line(image, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), color, thickness, cv2.LINE_AA)

        def _draw_side_on_main(self, side: str) -> None:
            boxes = getattr(self, f'{side}_boxes_global', {}) or {}
            squares = getattr(self, f'{side}_squares_global', {}) or {}
            marker = getattr(self, f'{side}_marker_global', None)
            manual = bool(getattr(self, f'{side}_was_manual', False))
            color = (255, 0, 255) if manual else (0, 220, 0)
            for key, corners in boxes.items():
                display_points = [(int(round(float(x) * self.scale)), int(round(float(y) * self.scale)) + self.top_bar_h) for x, y in corners]
                self._draw_polyline(self.canvas, display_points, color, 2 if key == ('area', 0) else 1, dashed=key == ('area', 0))
            if marker is not None:
                point = (int(round(float(marker[0]) * self.scale)), int(round(float(marker[1]) * self.scale)) + self.top_bar_h)
                cv2.drawMarker(self.canvas, point, color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)

        def _draw_zoom_panel(self, side: str, top_y: int, panel_height: int) -> None:
            sidebar_x = self.display_width
            margin = 10
            panel_width = self.sidebar_w - 2 * margin
            label_height = 28
            image_height = max(80, panel_height - label_height)
            marker = getattr(self, f'{side}_marker_global', None)
            boxes = getattr(self, f'{side}_boxes_global', {}) or {}
            squares = getattr(self, f'{side}_squares_global', {}) or {}
            manual = bool(getattr(self, f'{side}_was_manual', False))
            color = (255, 0, 255) if manual else (0, 220, 0)
            source = 'MANUAL' if manual else 'AUTO'
            active = side == 'left' and self.current_state == self.STATE_WAIT_LEFT or (side == 'right' and self.current_state == self.STATE_WAIT_RIGHT)
            square_box_count = sum((1 for key in boxes if key != ('area', 0)))
            label = f'{side.upper()} MARKER - {source} - {square_box_count} BOXES' + (' - SELECTED' if active else '')
            cv2.putText(self.canvas, label, (sidebar_x + margin, top_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
            image_y = top_y + label_height
            if marker is None:
                cv2.rectangle(self.canvas, (sidebar_x + margin, image_y), (self.canvas_w - margin, image_y + image_height), (20, 20, 20), -1)
                return
            geometry_points = [point for key, corners in boxes.items() if key != ('area', 0) for point in corners]
            gx, gy = (float(marker[0]), float(marker[1]))
            view_center = self._panel_view_centers.get(side, marker)
            view_x = float(view_center[0])
            view_y = float(view_center[1])
            if geometry_points:
                xs = [float(point[0]) for point in geometry_points]
                ys = [float(point[1]) for point in geometry_points]
                marker_width = max(1.0, max(xs) - min(xs))
                marker_height = max(1.0, max(ys) - min(ys))
            else:
                marker_width = marker_height = 100.0
            if side in self._panel_view_sizes:
                crop_width, crop_height = self._panel_view_sizes[side]
            else:
                crop_width = max(180.0, 3.2 * marker_width)
                crop_height = max(180.0, 1.8 * marker_height)
                panel_aspect = panel_width / max(float(image_height), 1.0)
                crop_aspect = crop_width / max(crop_height, 1.0)
                if crop_aspect < panel_aspect:
                    crop_width = crop_height * panel_aspect
                else:
                    crop_height = crop_width / panel_aspect
                self._panel_view_sizes[side] = (float(crop_width), float(crop_height))
            x1 = max(0, int(math.floor(view_x - 0.5 * crop_width)))
            y1 = max(0, int(math.floor(view_y - 0.5 * crop_height)))
            x2 = min(self.orig_w, int(math.ceil(view_x + 0.5 * crop_width)))
            y2 = min(self.orig_h, int(math.ceil(view_y + 0.5 * crop_height)))
            if x2 <= x1 or y2 <= y1:
                return
            crop = np.asarray(self.im.crop((x1, y1, x2, y2)).convert('RGB'))
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            interpolation = cv2.INTER_CUBIC if panel_width > x2 - x1 or image_height > y2 - y1 else cv2.INTER_AREA
            resized = cv2.resize(crop_bgr, (panel_width, image_height), interpolation=interpolation)
            scale_x = panel_width / float(x2 - x1)
            scale_y = image_height / float(y2 - y1)
            panel_rect = (sidebar_x + margin, image_y, sidebar_x + margin + panel_width, image_y + image_height)
            self._panel_maps[side] = {'rect': panel_rect, 'crop': (x1, y1, x2, y2), 'scale_x': scale_x, 'scale_y': scale_y}
            for key, corners in boxes.items():
                panel_points = [(int(round((float(x) - x1) * scale_x)), int(round((float(y) - y1) * scale_y))) for x, y in corners]
                self._draw_polyline(resized, panel_points, color, 2 if key == ('area', 0) else 1, dashed=key == ('area', 0))
            center_point = (int(round((gx - x1) * scale_x)), int(round((gy - y1) * scale_y)))
            cv2.drawMarker(resized, center_point, color, cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
            scale_handles_screen: list[tuple[float, float]] = []
            rotation_handle_screen: tuple[float, float] | None = None
            area_corners = boxes.get(('area', 0), [])
            if self._manual_edit_enabled and len(area_corners) == 4:
                area_panel = [np.asarray([(float(px) - x1) * scale_x, (float(py) - y1) * scale_y], dtype=np.float64) for px, py in area_corners]
                for point in area_panel:
                    handle = tuple(np.rint(point).astype(int))
                    cv2.rectangle(resized, (handle[0] - 6, handle[1] - 6), (handle[0] + 6, handle[1] + 6), (0, 255, 255), -1, cv2.LINE_AA)
                    cv2.rectangle(resized, (handle[0] - 6, handle[1] - 6), (handle[0] + 6, handle[1] + 6), (20, 20, 20), 1, cv2.LINE_AA)
                    scale_handles_screen.append((float(sidebar_x + margin + point[0]), float(image_y + point[1])))
                top_midpoint = 0.5 * (area_panel[0] + area_panel[1])
                center_array = np.asarray(center_point, dtype=np.float64)
                outward = top_midpoint - center_array
                norm = float(np.linalg.norm(outward))
                if norm <= 1e-09:
                    outward = np.asarray([0.0, -1.0], dtype=np.float64)
                else:
                    outward /= norm
                rotation_point = top_midpoint + 30.0 * outward
                cv2.line(resized, tuple(np.rint(top_midpoint).astype(int)), tuple(np.rint(rotation_point).astype(int)), (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(resized, tuple(np.rint(rotation_point).astype(int)), 8, (0, 255, 255), 2, cv2.LINE_AA)
                rotation_handle_screen = (float(sidebar_x + margin + rotation_point[0]), float(image_y + rotation_point[1]))
                instruction = 'BODY MOVE  |  CORNERS SCALE  |  CIRCLE ROTATE'
                cv2.putText(resized, instruction, (8, image_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1, cv2.LINE_AA)
            self._panel_maps[side]['scale_handles_screen'] = scale_handles_screen
            self._panel_maps[side]['rotation_handle_screen'] = rotation_handle_screen
            self.canvas[image_y:image_y + image_height, sidebar_x + margin:sidebar_x + margin + panel_width] = resized
            drag_active = self._panel_drag_side == side
            manual_panel = bool(self._manual_edit_enabled)
            border_color = (0, 255, 255) if drag_active else color if active else (150, 150, 150) if manual_panel else (90, 90, 90)
            border_thickness = 3 if drag_active else 2 if active or manual_panel else 1
            cv2.rectangle(self.canvas, (sidebar_x + margin, image_y), (sidebar_x + margin + panel_width, image_y + image_height), border_color, border_thickness)

        def redraw_gui(self) -> None:
            if not getattr(self, '_future_review_ready', False):
                return super().redraw_gui()
            self.canvas[:] = 0
            cv2.rectangle(self.canvas, (0, 0), (self.canvas_w, self.top_bar_h), self.status_bg_color, -1)
            status = str(self.status_text).replace('\n', ' ')
            text_width = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
            text_x = max(8, (self.canvas_w - text_width) // 2)
            cv2.putText(self.canvas, status, (text_x, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0) if self.current_state == self.STATE_FINISHED else (0, 255, 255), 1, cv2.LINE_AA)
            self.canvas[self.top_bar_h:self.top_bar_h + self.target_height, 0:self.display_width] = self.preview_color.copy()
            self._draw_side_on_main('left')
            self._draw_side_on_main('right')
            sidebar_x = self.display_width
            cv2.rectangle(self.canvas, (sidebar_x, self.top_bar_h), (self.canvas_w, self.top_bar_h + self.target_height), (35, 35, 35), -1)
            self._panel_maps = {}
            gap = 10
            panel_height = (self.target_height - 3 * gap) // 2
            self._draw_zoom_panel('left', self.top_bar_h + gap, panel_height)
            self._draw_zoom_panel('right', self.top_bar_h + 2 * gap + panel_height, panel_height)
            panel_y = self.target_height + self.top_bar_h
            cv2.rectangle(self.canvas, (0, panel_y), (self.canvas_w, self.canvas_h), (25, 25, 25), -1)
            for button in (self.btn_start, self.btn_reset, self.btn_exit):
                x1, y1, x2, y2 = button['box']
                if button['label'] == 'START ALIGNMENT':
                    color = (0, 140, 0)
                elif button['label'] == 'RESET SYSTEM':
                    color = (0, 120, 120)
                else:
                    color = (0, 0, 140)
                cv2.rectangle(self.canvas, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(self.canvas, (x1, y1), (x2, y2), (90, 90, 90), 1)
                label = button['label']
                width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
                cv2.putText(self.canvas, label, (x1 + (x2 - x1 - width) // 2, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('Large Wafer Tester', self.canvas)

        def handle_click(self, event, x, y, flags, param) -> None:
            del param
            if event == cv2.EVENT_LBUTTONDOWN:
                if y >= self.top_bar_h + self.target_height:
                    if self.btn_start['box'][0] <= x <= self.btn_start['box'][2] and self.btn_start['box'][1] <= y <= self.btn_start['box'][3]:
                        self._manual_edit_enabled = True
                        self._panel_drag_side = None
                        self._panel_drag_mode = None
                        self._panel_drag_last_global = None
                        self._panel_drag_start_global = None
                        self._panel_drag_start_radius = None
                        self._panel_drag_start_angle = None
                        self._panel_drag_baseline_boxes = None
                        self._panel_drag_baseline_squares = None
                        self.current_state = self.STATE_WAIT_LEFT
                        self.status_bg_color = (80, 45, 0)
                        self.status_text = 'MANUAL MODE: CLICK LEFT ON WAFER FOR ABSOLUTE START; PANEL BODY=MOVE, CORNERS=SCALE, CIRCLE=ROTATE'
                        self.redraw_gui()
                    elif self.btn_reset['box'][0] <= x <= self.btn_reset['box'][2] and self.btn_reset['box'][1] <= y <= self.btn_reset['box'][3]:
                        self._restore_auto_defaults()
                    elif self.btn_exit['box'][0] <= x <= self.btn_exit['box'][2] and self.btn_exit['box'][1] <= y <= self.btn_exit['box'][3]:
                        self.running = False
                    return
                if self._manual_edit_enabled:
                    side = self._panel_hit_side(x, y)
                    if side is not None and self._begin_panel_drag(side, x, y):
                        return
                if self.top_bar_h <= y < self.top_bar_h + self.target_height and x < self.display_width:
                    self.process_wafer_click(x, y - self.top_bar_h)
                return
            if event == cv2.EVENT_MOUSEMOVE:
                if self._panel_drag_side is not None and flags & cv2.EVENT_FLAG_LBUTTON:
                    self._continue_panel_drag(x, y)
                return
            if event == cv2.EVENT_LBUTTONUP:
                if self._panel_drag_side is not None:
                    self._continue_panel_drag(x, y)
                    self._finish_panel_drag()
                return

        def run(self) -> None:
            window = 'Large Wafer Tester'
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, self.canvas_w, self.canvas_h)
            cv2.setMouseCallback(window, self.handle_click)
            self.running = True
            self.redraw_gui()
            while self.running:
                key = cv2.waitKeyEx(20)
                if key in (13, 32, ord('a'), ord('A')):
                    if self.left_marker_global is not None and self.right_marker_global is not None:
                        self.running = False
                elif key in (ord('l'), ord('L')):
                    self._manual_edit_enabled = True
                    self.current_state = self.STATE_WAIT_LEFT
                    self.status_bg_color = (80, 45, 0)
                    self.status_text = 'OVERRIDE LEFT: CLICK WAFER; PANEL BODY=MOVE, CORNERS=SCALE, CIRCLE=ROTATE'
                    self.redraw_gui()
                elif key in (ord('r'), ord('R')):
                    self._manual_edit_enabled = True
                    self.current_state = self.STATE_WAIT_RIGHT
                    self.status_bg_color = (80, 45, 0)
                    self.status_text = 'OVERRIDE RIGHT: CLICK WAFER; PANEL BODY=MOVE, CORNERS=SCALE, CIRCLE=ROTATE'
                    self.redraw_gui()
                elif key in (ord('p'), ord('P')):
                    self._restore_auto_defaults()
                elif key in (27, ord('q'), ord('Q')):
                    self.running = False
            cv2.destroyWindow(window)
    return FutureMarkerReviewTester

def _ordered_grid_centers(points: list[tuple[float, float]]) -> np.ndarray:
    """Order a compact 3 x 4 grid top-to-bottom, then left-to-right."""
    array = np.asarray(points, dtype=np.float64)
    if array.shape != (12, 2):
        raise ValueError(f'Expected a 12-point grid, got shape {array.shape}')
    order_y = np.argsort(-array[:, 1])
    ordered = array[order_y]
    rows: list[np.ndarray] = []
    for start in range(0, 12, 3):
        row = ordered[start:start + 3]
        row = row[np.argsort(row[:, 0])]
        rows.append(row)
    return np.vstack(rows)

def _nominal_marker_square_centers(geometry: DesignGeometry, side: str) -> np.ndarray:
    centers: list[tuple[float, float]] = []
    for polygon in geometry.marker_polygons:
        x0, y0 = np.min(polygon, axis=0)
        x1, y1 = np.max(polygon, axis=0)
        width = float(x1 - x0)
        height = float(y1 - y0)
        cx = float((x0 + x1) * 0.5)
        cy = float((y0 + y1) * 0.5)
        if width >= 500.0 or height >= 500.0:
            continue
        if side == 'left' and cx >= geometry.center[0]:
            continue
        if side == 'right' and cx < geometry.center[0]:
            continue
        centers.append((cx, cy))
    return _ordered_grid_centers(centers)

def _fit_similarity_multipoint(source: np.ndarray, target: np.ndarray) -> tuple[float, float, float, float, float]:
    """Least-squares similarity mapping ``source`` onto ``target``."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError('Source and target point arrays must both be Nx2')
    if len(source) < 2:
        raise ValueError('At least two point pairs are required')
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    source_zero = source - source_mean
    target_zero = target - target_mean
    covariance = target_zero.T @ source_zero / float(len(source))
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(2, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    source_variance = float(np.mean(np.sum(source_zero * source_zero, axis=1)))
    if source_variance <= 1e-12:
        raise ValueError('Degenerate source point geometry')
    scale = float(np.sum(singular_values * np.diag(correction)) / source_variance)
    translation = target_mean - scale * (rotation @ source_mean)
    fitted = (scale * (rotation @ source.T)).T + translation
    rms = float(np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1))))
    angle = float(math.atan2(rotation[1, 0], rotation[0, 0]))
    return (angle, float(translation[0]), float(translation[1]), scale, rms)

def _resolve_marker_pair(tester: Any, markers: dict[str, list[dict[str, Any]]], gds_R: float, canvas_xc: float, canvas_yc: float, canvas_R: float, gds_xc: float, gds_yc: float, out_stem: str, *, geometry: DesignGeometry, runtime_state: dict[str, Any]):
    del markers
    left = getattr(tester, 'left_marker_global', None)
    right = getattr(tester, 'right_marker_global', None)
    if left is None or right is None:
        print(f'[{out_stem} Auto-Align] Marker review did not provide both points.')
        return None
    center_ds = runtime_state.get('canvas_center_ds')
    radius_ds = runtime_state.get('canvas_radius_ds')
    if center_ds is None or radius_ds is None:
        raise RuntimeError('Alignment runtime is missing downscaled wafer geometry')
    source_center_x = float(center_ds[0])
    source_center_y = float(center_ds[1])
    source_radius = float(radius_ds)
    coordinate_frame = 'production downscaled stitch'
    base_scale = float(gds_R) / max(source_radius, 1e-09)
    physical = {'left': ((float(left[0]) - source_center_x) * base_scale, (source_center_y - float(left[1])) * base_scale), 'right': ((float(right[0]) - source_center_x) * base_scale, (source_center_y - float(right[1])) * base_scale)}
    nominal_centered = {side: (float(point[0]) - float(gds_xc), float(point[1]) - float(gds_yc)) for side, point in geometry.marker_centers.items()}
    angle, tx, ty, scale = _similarity_from_two_points(physical, nominal_centered)
    fit_mode = 'two fitted-lattice marker centers'
    fit_rms_um: float | None = None
    # The two reviewed marker centers define the global wafer similarity exactly.
    # Per-side square edits and snaps refine those centers locally; allowing the
    # 24 square corners to re-solve the global transform can compromise the two
    # authoritative centers and make the following alignment UI show a different
    # registration.  Keep the grids as a diagnostic validation only.
    try:
        source_points: list[np.ndarray] = []
        target_points: list[np.ndarray] = []
        cosine = math.cos(angle)
        sine = math.sin(angle)
        rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
        for side in ('left', 'right'):
            square_dict = getattr(tester, f'{side}_squares_global', {}) or {}
            square_pixels = [(float(point[0]), float(point[1])) for point in square_dict.values()]
            if len(square_pixels) != 12:
                raise ValueError(f'{side} review contains {len(square_pixels)} square centers')
            square_physical = np.asarray([((px - source_center_x) * base_scale, (source_center_y - py) * base_scale) for px, py in square_pixels], dtype=np.float64)
            source_points.append(_ordered_grid_centers(square_physical))
            target_points.append(_nominal_marker_square_centers(geometry, side) - np.asarray([gds_xc, gds_yc], dtype=np.float64))
        source_array = np.vstack(source_points)
        target_array = np.vstack(target_points)
        fitted = (scale * (rotation @ source_array.T)).T + np.asarray([tx, ty])
        fit_rms_um = float(np.sqrt(np.mean(np.sum((fitted - target_array) ** 2, axis=1))))
        fit_mode += '; 24-square validation only'
    except Exception as fit_exc:
        print(f'[{out_stem} Auto-Align] Square-grid validation unavailable: {fit_exc}')
    left_mode = 'manual' if getattr(tester, 'left_was_manual', False) else 'automatic'
    right_mode = 'manual' if getattr(tester, 'right_was_manual', False) else 'automatic'
    debug_dir = Path('alignment_debug')
    debug_dir.mkdir(exist_ok=True)
    review_report = {'alignment_version': ALIGNMENT_VERSION, 'coordinate_frame': coordinate_frame, 'review_points_global_px': {'left': [float(left[0]), float(left[1])], 'right': [float(right[0]), float(right[1])]}, 'review_wafer_global_px': {'center': [source_center_x, source_center_y], 'radius': source_radius}, 'downscaled_stitch_wafer_px': {'center': [float(canvas_xc), float(canvas_yc)], 'radius': float(canvas_R)}, 'physical_detected_um': physical, 'nominal_gds_um': geometry.marker_centers, 'fit_mode': fit_mode, 'fit_rms_um': fit_rms_um, 'marker_lattice_fit': runtime_state.get('marker_lattice_fit', {}), 'manual_panel_similarity': {'edited': dict(getattr(tester, '_manual_similarity_edited', {}) or {}), 'drag_scale_factor': dict(getattr(tester, '_manual_template_scale', {}) or {}), 'drag_rotation_deg': dict(getattr(tester, '_manual_template_rotation_deg', {}) or {})}, 'solved': {'flat_angle_rad': angle, 'flat_angle_deg': angle * 180.0 / math.pi, 'x_offset_um': tx, 'y_offset_um': ty, 'scale_mult': scale}}
    (debug_dir / 'review_latest.json').write_text(json.dumps(review_report, indent=2), encoding='utf-8')
    print(f'[{out_stem} Auto-Align] Reviewed marker registration:')
    print(f'  Coordinate frame: {coordinate_frame}')
    print(f'  Left point:  {left_mode} at ({left[0]:.1f}, {left[1]:.1f}) px')
    print(f'  Right point: {right_mode} at ({right[0]:.1f}, {right[1]:.1f}) px')
    print(f'  Review wafer: center=({source_center_x:.1f}, {source_center_y:.1f}) px, radius={source_radius:.1f} px')
    print(f'  Fit: {fit_mode}' + (f', RMS={fit_rms_um:.3f} um' if fit_rms_um is not None else ''))
    print(f'  Rotation: {angle * 180.0 / math.pi:+.4f} deg')
    print(f'  Translation: X={tx:+.2f} um, Y={ty:+.2f} um')
    print(f'  Scale: {scale:.8f}')
    return (angle, tx, ty, scale)

def create_marker_reviewer(base_tester_class, geometry, runtime_state):
    from alignment_review import build_staged_marker_reviewer
    reviewer = _make_marker_tester(base_tester_class, geometry, runtime_state)
    return build_staged_marker_reviewer(reviewer)

def resolve_alignment(tester, markers, gds_R, canvas_xc, canvas_yc, canvas_R, gds_xc, gds_yc, out_stem, *, geometry, runtime_state):
    return _resolve_marker_pair(tester, markers, gds_R, canvas_xc, canvas_yc, canvas_R, gds_xc, gds_yc, out_stem, geometry=geometry, runtime_state=runtime_state)
