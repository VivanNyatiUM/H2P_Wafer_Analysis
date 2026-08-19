"""Staged marker-review UI for the production alignment workflow. The reviewer is constructed directly; no pipeline monkey-patching is performed."""
from __future__ import annotations
import math
import os
import time
from typing import Any, Iterable
import cv2
import numpy as np
UPGRADE_VERSION = 'alignment-marker-review-v5-production-score-auto-refine-2026-08-19'

def _real_box_items(boxes: dict[Any, Any]) -> list[tuple[Any, list[tuple[float, float]]]]:
    result: list[tuple[Any, list[tuple[float, float]]]] = []
    for key, corners in (boxes or {}).items():
        if key == ('area', 0) or not isinstance(corners, (list, tuple)) or len(corners) < 3:
            continue
        clean: list[tuple[float, float]] = []
        for point in corners:
            if not isinstance(point, (list, tuple, np.ndarray)) or len(point) < 2:
                continue
            x, y = (float(point[0]), float(point[1]))
            if math.isfinite(x) and math.isfinite(y):
                clean.append((x, y))
        if len(clean) >= 3:
            result.append((key, clean))
    return result

def _nearest_neighbor_pitch(points: Iterable[tuple[float, float]]) -> float:
    array = np.asarray(list(points), dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] != 2:
        return 80.0
    deltas = array[:, None, :] - array[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    distances[distances < 1e-09] = np.inf
    nearest = np.min(distances, axis=1)
    nearest = nearest[np.isfinite(nearest)]
    return float(np.median(nearest)) if nearest.size else 80.0

def _transform_points(points: Iterable[tuple[float, float]], center: tuple[float, float], scale: float, angle_deg: float, translation: tuple[float, float]=(0.0, 0.0)) -> list[tuple[float, float]]:
    angle = math.radians(float(angle_deg))
    cosine, sine = (math.cos(angle), math.sin(angle))
    cx, cy = (float(center[0]), float(center[1]))
    tx, ty = (float(translation[0]), float(translation[1]))
    output: list[tuple[float, float]] = []
    for x, y in points:
        dx = (float(x) - cx) * float(scale)
        dy = (float(y) - cy) * float(scale)
        output.append((cx + cosine * dx - sine * dy + tx, cy + sine * dx + cosine * dy + ty))
    return output

def _edge_proximity_image(image_bgr: np.ndarray) -> np.ndarray:
    """Return a float image where values near optical edges approach one."""
    if image_bgr.ndim == 2:
        gray = image_bgr.astype(np.uint8, copy=False)
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if gray.size == 0:
        return np.zeros(gray.shape, dtype=np.float32)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0.0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    positive = magnitude[magnitude > 0]
    normalizer = float(np.percentile(positive, 92.0)) if positive.size else 1.0
    gradient = np.clip(magnitude / max(normalizer, 1e-06), 0.0, 1.0)
    median = float(np.median(blurred))
    lower = int(max(8.0, 0.55 * median))
    upper = int(min(255.0, max(lower + 12.0, 1.45 * median)))
    canny = cv2.Canny(blurred, lower, upper, L2gradient=True)
    gradient_cut = float(np.percentile(gradient, 78.0))
    binary = ((canny > 0) | (gradient >= max(0.16, gradient_cut))).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    inverse = (1 - binary) * 255
    distance = cv2.distanceTransform(inverse.astype(np.uint8), cv2.DIST_L2, 3)
    sigma = 2.4
    proximity = np.exp(-distance / sigma).astype(np.float32)
    proximity *= 0.55 + 0.45 * gradient.astype(np.float32)
    proximity[binary > 0] = np.maximum(proximity[binary > 0], 0.82)
    return proximity

def _score_template_pose(proximity: np.ndarray, items: list[tuple[Any, list[tuple[float, float]]]], marker: tuple[float, float], scale: float, angle_deg: float, translation: tuple[float, float], common_origin: tuple[int, int], line_thickness: int) -> float:
    """Score one fully specified similarity pose against the edge image."""
    transformed_sets = [_transform_points(corners, marker, scale, angle_deg, translation) for _key, corners in items]
    transformed_points = [point for corners in transformed_sets for point in corners]
    if not transformed_points:
        return 0.0
    min_x = math.floor(min((point[0] for point in transformed_points))) - 4
    min_y = math.floor(min((point[1] for point in transformed_points))) - 4
    max_x = math.ceil(max((point[0] for point in transformed_points))) + 4
    max_y = math.ceil(max((point[1] for point in transformed_points))) + 4
    template_w = int(max_x - min_x + 1)
    template_h = int(max_y - min_y + 1)
    if template_w < 5 or template_h < 5:
        return 0.0
    template = np.zeros((template_h, template_w), dtype=np.float32)
    for corners in transformed_sets:
        local = np.asarray([[int(round(x - min_x)), int(round(y - min_y))] for x, y in corners], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(template, [local], True, 1.0, line_thickness, cv2.LINE_AA)
    template = np.clip(template, 0.0, 1.0)
    template_mass = float(np.sum(template))
    if template_mass < 8.0:
        return 0.0
    common_x0, common_y0 = common_origin
    px0 = int(min_x - common_x0)
    py0 = int(min_y - common_y0)
    px1 = px0 + template_w
    py1 = py0 + template_h
    if px0 < 0 or py0 < 0 or px1 > proximity.shape[1] or (py1 > proximity.shape[0]):
        return 0.0
    patch = proximity[py0:py1, px0:px1]
    if patch.shape != template.shape:
        return 0.0
    return float(np.sum(patch * template) / template_mass)

def _svd_similarity_fit(source: np.ndarray, target: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray] | None:
    """Least-squares 2-D similarity fit using an SVD/Procrustes solve."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.ndim != 2 or target.shape != source.shape or source.shape[0] < 3 or (source.shape[1] != 2):
        return None
    src_mean = np.mean(source, axis=0)
    dst_mean = np.mean(target, axis=0)
    src_centered = source - src_mean
    dst_centered = target - dst_mean
    denominator = float(np.sum(src_centered * src_centered))
    if denominator < 1e-09:
        return None
    covariance = src_centered.T @ dst_centered
    u, _singular, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    rotated = src_centered @ rotation.T
    scale = float(np.sum(rotated * dst_centered) / max(np.sum(rotated * rotated), 1e-09))
    if not math.isfinite(scale) or scale <= 0.0:
        return None
    translation = dst_mean - scale * (rotation @ src_mean)
    angle_deg = math.degrees(math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))
    fitted = scale * (source @ rotation.T) + translation
    residuals = np.linalg.norm(fitted - target, axis=1)
    return (scale, angle_deg, translation, residuals)

def _refine_snap_with_vertices(image_bgr: np.ndarray, items: list[tuple[Any, list[tuple[float, float]]]], marker: tuple[float, float], best: dict[str, Any], proximity: np.ndarray, common_bounds: tuple[int, int, int, int], square_side: float, pitch: float, search_radius: int, line_thickness: int) -> dict[str, Any] | None:
    """Use nearby optical vertices to refine rotation/scale with an SVD fit."""
    common_x0, common_y0, common_x1, common_y1 = common_bounds
    crop = image_bgr[common_y0:common_y1, common_x0:common_x1]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.astype(np.uint8, copy=False)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    all_source = np.asarray([point for _key, corners in items for point in corners], dtype=np.float64)
    if all_source.shape[0] < 6:
        return None
    max_corners = int(np.clip(all_source.shape[0] * 10, 120, 1200))
    min_distance = max(3.0, 0.08 * max(square_side, 1.0))
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=0.008, minDistance=min_distance, blockSize=5, useHarrisDetector=False)
    if corners is None or len(corners) < 6:
        return None
    optical = corners.reshape((-1, 2)).astype(np.float64)
    optical[:, 0] += common_x0
    optical[:, 1] += common_y0
    predicted = np.asarray(_transform_points(all_source, marker, float(best['scale']), float(best['angle_deg']), (float(best['dx']), float(best['dy']))), dtype=np.float64)
    vertex_radius = float(np.clip(round(0.32 * max(square_side, 1.0)), 7, max(18, round(0.24 * pitch))))
    used: set[int] = set()
    src_matches: list[np.ndarray] = []
    dst_matches: list[np.ndarray] = []
    order = np.argsort(-np.linalg.norm(predicted - np.asarray(marker, dtype=np.float64), axis=1))
    for index in order:
        distances = np.linalg.norm(optical - predicted[index], axis=1)
        for candidate_index in np.argsort(distances)[:8]:
            ci = int(candidate_index)
            if ci in used or float(distances[ci]) > vertex_radius:
                continue
            used.add(ci)
            src_matches.append(all_source[index])
            dst_matches.append(optical[ci])
            break
    if len(src_matches) < 6:
        return None
    src = np.asarray(src_matches, dtype=np.float64)
    dst = np.asarray(dst_matches, dtype=np.float64)
    fit = _svd_similarity_fit(src, dst)
    if fit is None:
        return None
    scale, angle_deg, world_translation, residuals = fit
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    residual_limit = min(vertex_radius, max(3.0, median + 2.8 * max(mad, 1.0)))
    inliers = residuals <= residual_limit
    if int(np.count_nonzero(inliers)) >= 6 and int(np.count_nonzero(inliers)) < len(src):
        refit = _svd_similarity_fit(src[inliers], dst[inliers])
        if refit is not None:
            scale, angle_deg, world_translation, residuals = refit
            src = src[inliers]
            dst = dst[inliers]
    if not 0.86 <= scale <= 1.14 or abs(angle_deg) > 12.0:
        return None
    theta = math.radians(angle_deg)
    rotation = np.asarray([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=np.float64)
    marker_vec = np.asarray(marker, dtype=np.float64)
    centered_translation = world_translation - marker_vec + scale * (rotation @ marker_vec)
    dx, dy = (float(centered_translation[0]), float(centered_translation[1]))
    if math.hypot(dx, dy) > max(1.35 * float(search_radius), 1.1 * pitch):
        return None
    coarse_scale = float(best['scale'])
    scale_trials = {float(scale), coarse_scale}
    for offset in np.arange(-0.03, 0.0301, 0.01):
        trial = coarse_scale + float(offset)
        if 0.86 <= trial <= 1.14:
            scale_trials.add(round(trial, 5))
    edge_score = -1.0
    chosen_scale = float(scale)
    for trial_scale in sorted(scale_trials):
        trial_score = _score_template_pose(proximity, items, marker, float(trial_scale), angle_deg, (dx, dy), (common_x0, common_y0), line_thickness)
        if trial_score > edge_score:
            edge_score = trial_score
            chosen_scale = float(trial_scale)
    scale = chosen_scale
    if edge_score <= 0.0:
        return None
    fitted = scale * (src @ rotation.T) + (np.asarray(marker, dtype=np.float64) + np.asarray((dx, dy), dtype=np.float64) - scale * (rotation @ np.asarray(marker, dtype=np.float64)))
    rms = float(np.sqrt(np.mean(np.sum((fitted - dst) ** 2, axis=1)))) if len(src) else float('inf')
    displacement = math.hypot(dx, dy)
    spatial_penalty = 0.13 * (displacement / max(float(search_radius), 1.0)) ** 2
    transform_penalty = 0.008 * (abs(angle_deg) / 8.0) ** 2 + 0.008 * (abs(scale - 1.0) / 0.08) ** 2
    vertex_quality = max(0.0, 1.0 - rms / max(vertex_radius, 1.0))
    vertex_bonus = 0.045 * min(1.0, len(src) / 12.0) * vertex_quality
    selection_score = edge_score - spatial_penalty - transform_penalty + vertex_bonus
    return {'score': edge_score, 'selection_score': selection_score, 'dx': dx, 'dy': dy, 'scale': scale, 'angle_deg': angle_deg, 'pitch': pitch, 'search_radius': search_radius, 'svd_refined': True, 'svd_vertices': int(len(src)), 'svd_rms': rms}

def find_local_template_snap(image_bgr: np.ndarray, boxes: dict[Any, Any], squares: dict[Any, Any], marker: tuple[float, float], *, angle_candidates: Iterable[float] | None=None, scale_candidates: Iterable[float] | None=None) -> dict[str, Any]:
    """Fit a GDS-derived square template to nearby optical edges and vertices.

    The coarse pass searches several lattice pitches so that a tempting partial
    row/column match cannot hide a stronger full-marker match. A second pass refines
    the best angle/scale neighborhood, then a bounded SVD/Procrustes solve uses
    nearby optical corners to improve the rotation and scale when the vertices
    support it.
    """
    items = _real_box_items(boxes)
    if len(items) < 4:
        return {'ok': False, 'reason': 'not enough GDS marker boxes'}
    marker = (float(marker[0]), float(marker[1]))
    center_points = [(float(point[0]), float(point[1])) for key, point in (squares or {}).items() if key != ('area', 0) and isinstance(point, (list, tuple, np.ndarray)) and (len(point) >= 2)]
    if len(center_points) < 4:
        center_points = [(float(np.mean([p[0] for p in corners])), float(np.mean([p[1] for p in corners]))) for _key, corners in items]
    pitch = max(8.0, _nearest_neighbor_pitch(center_points))
    # A manual absolute click can easily land one to three repeated-square pitches
    # from the actual marker.  The old 0.9-pitch window made the incorrect nearby
    # partial lattice the only candidate.  Keep this bounded, but include the whole
    # ambiguity range observed on the LOR left markers.
    search_radius = int(np.clip(round(4.25 * pitch), 72, 720))
    supplied_angles = angle_candidates is not None
    supplied_scales = scale_candidates is not None
    if angle_candidates is None:
        angle_candidates = (-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0)
    if scale_candidates is None:
        scale_candidates = (0.94, 1.0, 1.06)
    angles = [float(value) for value in angle_candidates]
    scales = [float(value) for value in scale_candidates]
    all_points = [point for _key, corners in items for point in corners]
    relative = np.asarray([[x - marker[0], y - marker[1]] for x, y in all_points], dtype=np.float64)
    max_scale_for_crop = max([1.14] + [abs(value) for value in scales])
    max_radius_x = max(20.0, float(np.max(np.abs(relative[:, 0]))) * max_scale_for_crop + 12.0)
    max_radius_y = max(20.0, float(np.max(np.abs(relative[:, 1]))) * max_scale_for_crop + 12.0)
    common_x0 = max(0, int(math.floor(marker[0] - max_radius_x - search_radius - 12)))
    common_y0 = max(0, int(math.floor(marker[1] - max_radius_y - search_radius - 12)))
    common_x1 = min(image_bgr.shape[1], int(math.ceil(marker[0] + max_radius_x + search_radius + 12)))
    common_y1 = min(image_bgr.shape[0], int(math.ceil(marker[1] + max_radius_y + search_radius + 12)))
    if common_x1 - common_x0 < 20 or common_y1 - common_y0 < 20:
        return {'ok': False, 'reason': 'snap search region falls outside image'}
    proximity = _edge_proximity_image(image_bgr[common_y0:common_y1, common_x0:common_x1])
    if proximity.size == 0:
        return {'ok': False, 'reason': 'empty snap search image'}
    side_lengths: list[float] = []
    for _key, corners in items:
        for p0, p1 in zip(corners, corners[1:] + corners[:1]):
            side_lengths.append(math.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    square_side = float(np.median(side_lengths)) if side_lengths else 0.5 * pitch
    line_thickness = int(np.clip(round(0.035 * square_side), 1, 4))
    baseline_score = _score_template_pose(proximity, items, marker, 1.0, 0.0, (0.0, 0.0), (common_x0, common_y0), line_thickness)
    best: dict[str, Any] | None = None
    raw_best: dict[str, Any] | None = None

    def evaluate(candidate_angles: Iterable[float], candidate_scales: Iterable[float]) -> None:
        nonlocal best, raw_best
        seen: set[tuple[float, float]] = set()
        for scale in candidate_scales:
            scale = float(scale)
            if not 0.84 <= scale <= 1.16:
                continue
            for angle_deg in candidate_angles:
                angle_deg = float(angle_deg)
                signature = (round(scale, 5), round(angle_deg, 5))
                if signature in seen:
                    continue
                seen.add(signature)
                transformed_sets = [_transform_points(corners, marker, scale, angle_deg) for _key, corners in items]
                transformed_points = [point for corners in transformed_sets for point in corners]
                min_x = math.floor(min((point[0] for point in transformed_points))) - 4
                min_y = math.floor(min((point[1] for point in transformed_points))) - 4
                max_x = math.ceil(max((point[0] for point in transformed_points))) + 4
                max_y = math.ceil(max((point[1] for point in transformed_points))) + 4
                template_w = int(max_x - min_x + 1)
                template_h = int(max_y - min_y + 1)
                if template_w < 5 or template_h < 5:
                    continue
                template = np.zeros((template_h, template_w), dtype=np.float32)
                for corners in transformed_sets:
                    local = np.asarray([[int(round(x - min_x)), int(round(y - min_y))] for x, y in corners], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(template, [local], True, 1.0, line_thickness, cv2.LINE_AA)
                template = np.clip(template, 0.0, 1.0)
                template_mass = float(np.sum(template))
                if template_mass < 8.0:
                    continue
                expected_x = int(round(min_x - common_x0))
                expected_y = int(round(min_y - common_y0))
                search_x0 = max(0, expected_x - search_radius)
                search_y0 = max(0, expected_y - search_radius)
                search_x1 = min(proximity.shape[1], expected_x + search_radius + template_w)
                search_y1 = min(proximity.shape[0], expected_y + search_radius + template_h)
                search = proximity[search_y0:search_y1, search_x0:search_x1]
                if search.shape[0] < template_h or search.shape[1] < template_w:
                    continue
                response = cv2.matchTemplate(search, template, cv2.TM_CCORR)
                _min_value, max_value, _min_location, max_location = cv2.minMaxLoc(response)
                score = float(max_value / template_mass)
                placed_x = search_x0 + int(max_location[0])
                placed_y = search_y0 + int(max_location[1])
                dx = float(placed_x - expected_x)
                dy = float(placed_y - expected_y)
                displacement = math.hypot(dx, dy)
                spatial_penalty = 0.13 * (displacement / max(float(search_radius), 1.0)) ** 2
                transform_penalty = 0.008 * (abs(angle_deg) / 8.0) ** 2 + 0.008 * (abs(scale - 1.0) / 0.08) ** 2
                selection_score = score - spatial_penalty - transform_penalty
                candidate = {'score': score, 'selection_score': selection_score, 'dx': dx, 'dy': dy, 'scale': scale, 'angle_deg': angle_deg, 'pitch': pitch, 'search_radius': search_radius, 'svd_refined': False}
                if best is None or selection_score > float(best['selection_score']):
                    best = candidate
                if raw_best is None or score > float(raw_best['score']):
                    raw_best = candidate
    evaluate(angles, scales)
    if best is None:
        return {'ok': False, 'reason': 'no valid snap candidates'}
    if raw_best is not None and float(raw_best['score']) >= float(best['score']) + 0.035:
        best = raw_best
    if not supplied_angles:
        center_angle = float(best['angle_deg'])
        fine_angles = np.arange(center_angle - 2.0, center_angle + 2.001, 0.5)
    else:
        fine_angles = angles
    if not supplied_scales:
        center_scale = float(best['scale'])
        fine_scales = np.arange(center_scale - 0.04, center_scale + 0.0401, 0.02)
    else:
        fine_scales = scales
    evaluate(fine_angles, fine_scales)
    if raw_best is not None and float(raw_best['score']) >= float(best['score']) + 0.035:
        best = raw_best
    refined = _refine_snap_with_vertices(image_bgr, items, marker, best, proximity, (common_x0, common_y0, common_x1, common_y1), square_side, pitch, search_radius, line_thickness)
    if refined is not None:
        if float(refined['score']) >= float(best['score']) - 0.025 and float(refined['selection_score']) >= float(best['selection_score']) - 0.004:
            best = refined
    best['baseline_score'] = baseline_score
    improvement = float(best['score'] - baseline_score)
    best['improvement'] = improvement
    displacement = math.hypot(float(best['dx']), float(best['dy']))
    transform_size = abs(math.log(max(float(best['scale']), 1e-09))) + abs(math.radians(float(best['angle_deg'])))
    reliable = float(best['score']) >= 0.3 and (improvement >= 0.015 or float(best['score']) >= 0.47 or (displacement <= max(2.0, 0.04 * pitch) and transform_size <= 0.018) or (bool(best.get('svd_refined')) and float(best.get('svd_rms', 999.0)) <= max(3.0, 0.12 * square_side)))
    best['ok'] = bool(reliable)
    if not reliable:
        best['reason'] = f"weak edge/vertex match (score {best['score']:.2f}, improvement {improvement:+.2f})"
    return best

def _decode_arrow_key(key: int) -> tuple[int, int] | None:
    """Return a unit nudge vector for common waitKeyEx arrow codes."""
    mapping = {2424832: (-1, 0), 2490368: (0, -1), 2555904: (1, 0), 2621440: (0, 1), 65361: (-1, 0), 65362: (0, -1), 65363: (1, 0), 65364: (0, 1), 63234: (-1, 0), 63232: (0, -1), 63235: (1, 0), 63233: (0, 1)}
    return mapping.get(int(key))

def _shift_key_down() -> bool:
    """Check Shift while an arrow is handled; Win32 waitKeyEx omits modifiers."""
    if os.name != 'nt':
        return False
    try:
        import ctypes
        return bool(ctypes.windll.user32.GetAsyncKeyState(16) & 32768)
    except Exception:
        return False

def _inside(box: tuple[int, int, int, int], x: int, y: int) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def build_staged_marker_reviewer(base_class) -> None:
    """Install the staged marker-review workflow on an imported pipeline."""
    if getattr(base_class, '_h2p_alignment_marker_ui_upgrade', False):
        return base_class

    class StagedMarkerReviewTester(base_class):
        _h2p_alignment_marker_ui_upgrade = True
        _h2p_alignment_marker_ui_upgrade_version = UPGRADE_VERSION

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._alignment_marker_upgrade_ready = False
            super().__init__(*args, **kwargs)
            self.sidebar_w = max(420, int(getattr(self, 'sidebar_w', 370)))
            self.top_bar_h = 72
            self.bottom_bar_h = 112
            self.recompute_display_sizes()
            self._workflow_stage = 'auto'
            self._manual_active_side = 'left'
            self._manual_button_flash_until = 0.0
            self._main_drag_side: str | None = None
            self._snap_button_bounds: dict[str, tuple[int, int, int, int]] = {}
            self._snap_status = {'left': 'Available after absolute placement', 'right': 'Available after absolute placement'}
            self._snap_busy = False
            self._wafer_badge_hovered = False
            self._wafer_badge_text = 'hover for name'
            badge_text_size = cv2.getTextSize(self._wafer_badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.29, 1)[0]
            badge_x1, badge_y1 = (14, 23)
            self._wafer_badge_bounds = (badge_x1, badge_y1, badge_x1 + badge_text_size[0] + 12, badge_y1 + badge_text_size[1] + 12)
            self._alignment_marker_upgrade_ready = True
            self._layout_upgrade_buttons()
            self.redraw_gui()

        def _layout_upgrade_buttons(self) -> None:
            labels = ('MANUAL ALIGN', 'LEFT', 'RIGHT', 'DONE', 'RESET', 'ACCEPT / EXIT')
            left_margin, right_margin, gap = (12, 12, 7)
            total_gap = gap * (len(labels) - 1)
            width = max(78, (self.canvas_w - left_margin - right_margin - total_gap) // len(labels))
            y1 = self.top_bar_h + self.target_height + 34
            y2 = min(self.canvas_h - 23, y1 + 48)
            self._upgrade_buttons: dict[str, dict[str, Any]] = {}
            x = left_margin
            keys = ('manual', 'left', 'right', 'done', 'reset', 'accept')
            for key, label in zip(keys, labels):
                x2 = self.canvas_w - right_margin if key == 'accept' else x + width
                self._upgrade_buttons[key] = {'label': label, 'box': (int(x), int(y1), int(x2), int(y2))}
                x = x2 + gap

        def _active_absolute_side(self) -> str | None:
            if self._workflow_stage == 'absolute_left':
                return 'left'
            if self._workflow_stage == 'absolute_right':
                return 'right'
            return None

        def _stage_instruction(self) -> str:
            if self._workflow_stage == 'absolute_left':
                return 'LEFT selected: click/drag wafer or use arrows (Shift+arrows = 10 px); switch RIGHT anytime; DONE when ready'
            if self._workflow_stage == 'absolute_right':
                return 'RIGHT selected: click/drag wafer or use arrows (Shift+arrows = 10 px); switch LEFT anytime; DONE when ready'
            if self._workflow_stage == 'relative':
                return 'DONE: relative move/scale/rotate and ATTEMPT SNAP are enabled; press DONE again to resume LEFT/RIGHT absolute editing'
            return 'Automatic marker boxes are active. MANUAL ALIGN starts the override workflow'

        def _set_stage_status(self, action: str | None=None) -> None:
            instruction = self._stage_instruction()
            if action:
                self.status_text = f'{action}. {instruction}'
            else:
                self.status_text = instruction
            if self._workflow_stage == 'auto':
                self.status_bg_color = (0, 78, 0)
            elif self._workflow_stage == 'relative':
                self.status_bg_color = (48, 78, 0)
            else:
                self.status_bg_color = (78, 48, 0)

        def _set_manual_side(self, side: str, *, announce: bool=True) -> None:
            if side not in ('left', 'right'):
                return
            self._manual_active_side = side
            self._workflow_stage = f'absolute_{side}'
            self.current_state = self.STATE_WAIT_LEFT if side == 'left' else self.STATE_WAIT_RIGHT
            self._manual_edit_enabled = True
            self._main_drag_side = None
            self._panel_drag_side = None
            self._panel_drag_mode = None
            self._snap_status = {'left': 'Press DONE to enable snap', 'right': 'Press DONE to enable snap'}
            marker = getattr(self, f'{side}_marker_global', None)
            if marker is not None:
                self._set_panel_view_center(side, tuple(marker))
            if announce:
                self._set_stage_status(f'{side.upper()} SELECTED')
            self.redraw_gui()

        def _start_manual_workflow(self) -> None:
            self._manual_button_flash_until = time.monotonic() + 0.38
            side = self._manual_active_side if self._manual_active_side in ('left', 'right') else 'left'
            self._set_manual_side(side, announce=False)
            self._set_stage_status('MANUAL ALIGNMENT STARTED')
            self.redraw_gui()

        def _select_manual_side(self, side: str) -> None:
            if self._workflow_stage == 'auto':
                self._manual_button_flash_until = time.monotonic() + 0.24
            if self._workflow_stage == 'relative':
                self._set_stage_status('PRESS DONE AGAIN BEFORE SWITCHING ABSOLUTE SIDE')
                self.redraw_gui()
                return
            self._set_manual_side(side)

        def _toggle_done(self) -> None:
            if self._workflow_stage == 'auto':
                self._set_stage_status('START MANUAL ALIGNMENT FIRST')
                self.redraw_gui()
                return
            if self._workflow_stage == 'relative':
                self._set_manual_side(self._manual_active_side, announce=False)
                self._set_stage_status('DONE RELEASED - ABSOLUTE EDITING ENABLED')
                self.redraw_gui()
                return
            if self.left_marker_global is None or self.right_marker_global is None:
                self._set_stage_status('LEFT AND RIGHT MARKERS MUST BOTH EXIST BEFORE DONE')
                self.redraw_gui()
                return
            active = self._active_absolute_side()
            if active is not None:
                self._manual_active_side = active
            self._main_drag_side = None
            self._panel_drag_side = None
            self._panel_drag_mode = None
            self._workflow_stage = 'relative'
            self.current_state = self.STATE_FINISHED
            self._manual_edit_enabled = True
            self._snap_status = {'left': 'Ready', 'right': 'Ready'}
            self._set_stage_status('MANUAL ABSOLUTE PLACEMENT MARKED DONE')
            self.redraw_gui()

        def _nudge_active_side(self, dx: int, dy: int, *, coarse: bool=False) -> None:
            side = self._active_absolute_side()
            if side is None:
                if self._workflow_stage == 'relative':
                    self._set_stage_status('PRESS DONE AGAIN BEFORE NUDGING LEFT/RIGHT')
                    self.redraw_gui()
                return
            marker = getattr(self, f'{side}_marker_global', None)
            if marker is None:
                return
            step = 10.0 if coarse else 1.0
            new_marker = (float(marker[0]) + step * dx, float(marker[1]) + step * dy)
            self._move_side_geometry(side, new_marker, update_panel_view=True)
            setattr(self, f'{side}_was_manual', True)
            setattr(self, f'{side}_resolved_success', True)
            label = '10 px' if coarse else '1 px'
            self._set_stage_status(f'{side.upper()} NUDGED {label}')
            self.redraw_gui()

        def _restore_auto_defaults(self) -> None:
            self._workflow_stage = 'auto'
            self._manual_active_side = 'left'
            self._main_drag_side = None
            self._snap_status = {'left': 'Available after DONE', 'right': 'Available after DONE'}
            super()._restore_auto_defaults()
            self._set_stage_status('AUTOMATIC BOXES RESTORED')
            self.redraw_gui()

        def process_wafer_click(self, x: int, y: int) -> None:
            side = self._active_absolute_side()
            if side is None:
                return
            orig_x = float(x / self.scale)
            orig_y = float(y / self.scale)
            self._move_side_geometry(side, (orig_x, orig_y), update_panel_view=True)
            setattr(self, f'{side}_was_manual', True)
            setattr(self, f'{side}_resolved_success', True)
            self._set_stage_status(f'{side.upper()} ABSOLUTE LOCATION MOVED')
            self.redraw_gui()

        def _finish_panel_drag(self) -> None:
            side = self._panel_drag_side
            mode = self._panel_drag_mode
            super()._finish_panel_drag()
            if side is not None:
                label = {'translate': 'MOVED', 'scale': 'SCALED', 'rotate': 'ROTATED'}.get(mode, 'EDITED')
                self._set_stage_status(f'{side.upper()} TEMPLATE {label}')
                self.redraw_gui()

        def _apply_snap_result(self, side: str, result: dict[str, Any]) -> None:
            marker = getattr(self, f'{side}_marker_global', None)
            boxes = getattr(self, f'{side}_boxes_global', {}) or {}
            squares = getattr(self, f'{side}_squares_global', {}) or {}
            if marker is None:
                return
            translation = (float(result['dx']), float(result['dy']))
            scale = float(result['scale'])
            angle_deg = float(result['angle_deg'])
            transformed_boxes = {key: _transform_points(corners, marker, scale, angle_deg, translation) for key, corners in boxes.items()}
            transformed_squares = {key: _transform_points([point], marker, scale, angle_deg, translation)[0] for key, point in squares.items()}
            new_marker = (float(marker[0]) + translation[0], float(marker[1]) + translation[1])
            setattr(self, f'{side}_boxes_global', transformed_boxes)
            setattr(self, f'{side}_squares_global', transformed_squares)
            setattr(self, f'{side}_marker_global', new_marker)
            setattr(self, f'{side}_click_global', new_marker)
            setattr(self, f'{side}_was_manual', True)
            setattr(self, f'{side}_resolved_success', True)
            self._manual_similarity_edited[side] = True
            self._manual_template_scale[side] = float(self._manual_template_scale.get(side, 1.0)) * scale
            self._manual_template_rotation_deg[side] = float(self._manual_template_rotation_deg.get(side, 0.0)) + angle_deg

        def _attempt_snap(self, side: str) -> None:
            if self._workflow_stage != 'relative' or self._snap_busy:
                return
            marker = getattr(self, f'{side}_marker_global', None)
            if marker is None:
                self._snap_status[side] = 'No marker geometry'
                self.redraw_gui()
                return
            self._snap_busy = True
            self._snap_status[side] = 'Searching nearby edges + vertices...'
            self.redraw_gui()
            try:
                boxes = getattr(self, f'{side}_boxes_global', {}) or {}
                squares = getattr(self, f'{side}_squares_global', {}) or {}
                items = _real_box_items(boxes)
                points = [point for _key, corners in items for point in corners]
                centers = [point for key, point in squares.items() if key != ('area', 0)]
                pitch = max(8.0, _nearest_neighbor_pitch(centers))
                # The snapper searches 4.25 pitches.  Retain a little more than that
                # around the complete template so the high-score pose is not clipped
                # by this outer UI crop before the snapper can evaluate it.
                margin = int(np.clip(round(5.0 * pitch), 180, 1600))
                x0 = max(0, int(math.floor(min([p[0] for p in points] + [marker[0]]) - margin)))
                y0 = max(0, int(math.floor(min([p[1] for p in points] + [marker[1]]) - margin)))
                x1 = min(self.orig_w, int(math.ceil(max([p[0] for p in points] + [marker[0]]) + margin)))
                y1 = min(self.orig_h, int(math.ceil(max([p[1] for p in points] + [marker[1]]) + margin)))
                crop_rgb = np.asarray(self.im.crop((x0, y0, x1, y1)).convert('RGB'))
                crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                local_boxes = {key: [(float(x) - x0, float(y) - y0) for x, y in corners] for key, corners in boxes.items()}
                local_squares = {key: (float(point[0]) - x0, float(point[1]) - y0) for key, point in squares.items()}
                local_marker = (float(marker[0]) - x0, float(marker[1]) - y0)
                result = find_local_template_snap(crop_bgr, local_boxes, local_squares, local_marker)
                if not result.get('ok'):
                    self._snap_status[side] = str(result.get('reason', 'No reliable local match'))
                    self._set_stage_status(f'{side.upper()} SNAP NOT APPLIED')
                else:
                    self._apply_snap_result(side, result)
                    svd_note = ''
                    if result.get('svd_refined'):
                        svd_note = f"  SVD {int(result.get('svd_vertices', 0))}v rms {float(result.get('svd_rms', 0.0)):.1f}"
                    self._snap_status[side] = f"score {result['score']:.2f}  move ({result['dx']:+.1f}, {result['dy']:+.1f}) px  rot {result['angle_deg']:+.1f} deg  scale {result['scale']:.3f}{svd_note}"
                    self._set_stage_status(f'{side.upper()} TEMPLATE SNAPPED TO LOCAL BOX EDGES')
            except Exception as exc:
                self._snap_status[side] = f'Snap failed: {exc}'
                self._set_stage_status(f'{side.upper()} SNAP FAILED')
            finally:
                self._snap_busy = False
                self.redraw_gui()

        def _draw_panel_with_snap(self, side: str, top_y: int, panel_height: int) -> None:
            footer_h = 48
            base_height = max(120, panel_height - footer_h)
            super()._draw_zoom_panel(side, top_y, base_height)
            sidebar_x = self.display_width
            margin = 10
            footer_y1 = top_y + base_height + 4
            footer_y2 = top_y + panel_height - 3
            cv2.rectangle(self.canvas, (sidebar_x + margin, footer_y1), (self.canvas_w - margin, footer_y2), (29, 29, 33), -1)
            enabled = self._workflow_stage == 'relative' and (not self._snap_busy)
            button_w = 148
            bx1 = sidebar_x + margin
            bx2 = bx1 + button_w
            by1 = footer_y1 + 4
            by2 = footer_y2 - 4
            self._snap_button_bounds[side] = (bx1, by1, bx2, by2)
            fill = (0, 128, 165) if enabled else (62, 62, 62)
            if self._snap_busy:
                fill = (0, 95, 135)
            cv2.rectangle(self.canvas, (bx1, by1), (bx2, by2), fill, -1)
            cv2.rectangle(self.canvas, (bx1, by1), (bx2, by2), (120, 180, 190) if enabled else (90, 90, 90), 1)
            label = 'ATTEMPT SNAP'
            tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
            cv2.putText(self.canvas, label, (bx1 + max(4, (bx2 - bx1 - tw) // 2), by1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (245, 245, 245), 1, cv2.LINE_AA)
            status = str(self._snap_status.get(side, ''))
            max_chars = 39
            if len(status) > max_chars:
                status = status[:max_chars - 1] + '...'
            cv2.putText(self.canvas, status, (bx2 + 10, by1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (205, 205, 205), 1, cv2.LINE_AA)

        def _draw_bottom_button(self, key: str) -> None:
            button = self._upgrade_buttons[key]
            x1, y1, x2, y2 = button['box']
            stage = self._workflow_stage
            active_side = self._manual_active_side
            enabled = True
            pressed = False
            if key in ('left', 'right'):
                enabled = stage not in ('auto', 'relative')
                pressed = stage != 'auto' and active_side == key
                fill = (0, 103, 150) if pressed else (0, 135, 188) if enabled else (58, 58, 58)
            elif key == 'done':
                enabled = stage != 'auto'
                pressed = stage == 'relative'
                fill = (0, 100, 72) if pressed else (0, 142, 98) if enabled else (58, 58, 58)
            elif key == 'manual':
                flashing = time.monotonic() < self._manual_button_flash_until
                pressed = stage != 'auto'
                fill = (55, 220, 75) if flashing else (0, 128, 32) if pressed else (0, 112, 28)
            elif key == 'reset':
                fill = (0, 112, 130)
            else:
                enabled = stage in ('auto', 'relative')
                fill = (42, 62, 150) if enabled else (58, 58, 58)
            border = (150, 255, 205) if enabled else (95, 95, 95)
            if pressed:
                cv2.rectangle(self.canvas, (x1, y1), (x2, y2), (38, 38, 42), -1)
                draw_box = (x1 + 2, y1 + 2, x2, y2)
            else:
                draw_box = (x1, y1, x2, y2)
            dx1, dy1, dx2, dy2 = draw_box
            cv2.rectangle(self.canvas, (dx1, dy1), (dx2, dy2), fill, -1)
            cv2.rectangle(self.canvas, (dx1, dy1), (dx2, dy2), border, 1)
            label = button['label']
            font_scale = 0.42 if key != 'accept' else 0.38
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            text_y = dy1 + (dy2 - dy1 + text_size[1]) // 2 + (1 if pressed else 0)
            cv2.putText(self.canvas, label, (dx1 + max(5, (dx2 - dx1 - text_size[0]) // 2), text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (245, 245, 245), 1, cv2.LINE_AA)

        def redraw_gui(self) -> None:
            if not getattr(self, '_alignment_marker_upgrade_ready', False):
                return super().redraw_gui()
            if self.canvas.shape[:2] != (self.canvas_h, self.canvas_w):
                self.canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
            self.canvas[:] = 0
            cv2.rectangle(self.canvas, (0, 0), (self.canvas_w, self.top_bar_h), self.status_bg_color, -1)
            badge_x1, badge_y1, badge_x2, badge_y2 = self._wafer_badge_bounds
            badge_fill = (38, 73, 67) if self._wafer_badge_hovered else (28, 58, 53)
            badge_border = (118, 218, 186) if self._wafer_badge_hovered else (74, 157, 137)
            cv2.rectangle(self.canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), badge_fill, -1)
            cv2.rectangle(self.canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), badge_border, 1)
            badge_text_size = cv2.getTextSize(self._wafer_badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.29, 1)[0]
            badge_text_y = badge_y1 + (badge_y2 - badge_y1 + badge_text_size[1]) // 2
            cv2.putText(self.canvas, self._wafer_badge_text, (badge_x1 + 6, badge_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (205, 234, 224), 1, cv2.LINE_AA)
            title = str(self.status_text).replace('\n', ' ')
            if len(title) > 112:
                title = title[:109] + '...'
            title_w = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
            status_left = badge_x2 + 18
            cv2.putText(self.canvas, title, (max(status_left, status_left + (self.canvas_w - status_left - title_w) // 2), 27), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 255, 235), 1, cv2.LINE_AA)
            instruction = self._stage_instruction()
            if len(instruction) > 118:
                instruction = instruction[:115] + '...'
            inst_w = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
            cv2.putText(self.canvas, instruction, (max(status_left, status_left + (self.canvas_w - status_left - inst_w) // 2), 54), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
            self.canvas[self.top_bar_h:self.top_bar_h + self.target_height, 0:self.display_width] = self.preview_color.copy()
            self._draw_side_on_main('left')
            self._draw_side_on_main('right')
            sidebar_x = self.display_width
            cv2.rectangle(self.canvas, (sidebar_x, self.top_bar_h), (self.canvas_w, self.top_bar_h + self.target_height), (35, 35, 38), -1)
            self._panel_maps = {}
            self._snap_button_bounds = {}
            gap = 10
            panel_height = (self.target_height - 3 * gap) // 2
            self._draw_panel_with_snap('left', self.top_bar_h + gap, panel_height)
            self._draw_panel_with_snap('right', self.top_bar_h + 2 * gap + panel_height, panel_height)
            bottom_y = self.top_bar_h + self.target_height
            cv2.rectangle(self.canvas, (0, bottom_y), (self.canvas_w, self.canvas_h), (24, 24, 27), -1)
            cv2.putText(self.canvas, 'Full wafer: click/drag absolute location   |   Panels: body move, corners scale, circle rotate', (14, bottom_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (195, 195, 195), 1, cv2.LINE_AA)
            self._layout_upgrade_buttons()
            for key in ('manual', 'left', 'right', 'done', 'reset', 'accept'):
                self._draw_bottom_button(key)
            cv2.putText(self.canvas, 'Keys: M manual | L/R select | D done toggle | arrows 1 px | Shift+arrows 10 px | P reset | Enter/A accept | Q/Esc exit', (14, self.canvas_h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)
            if self._wafer_badge_hovered:
                wafer_name = str(getattr(self, 'wafer_id', '') or 'Unknown wafer')
                tooltip = f'Current wafer:  {wafer_name}'
                tooltip_w = cv2.getTextSize(tooltip, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
                tx1, ty1 = (badge_x1, self.top_bar_h + 8)
                tx2, ty2 = (min(self.canvas_w - 12, tx1 + tooltip_w + 28), ty1 + 38)
                cv2.rectangle(self.canvas, (tx1 + 3, ty1 + 4), (tx2 + 3, ty2 + 4), (10, 10, 12), -1)
                cv2.rectangle(self.canvas, (tx1, ty1), (tx2, ty2), (31, 37, 39), -1)
                cv2.rectangle(self.canvas, (tx1, ty1), (tx2, ty2), (104, 215, 180), 1)
                cv2.putText(self.canvas, tooltip, (tx1 + 14, ty1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (232, 247, 242), 1, cv2.LINE_AA)
            cv2.imshow('Large Wafer Tester', self.canvas)

        def handle_click(self, event, x, y, flags, param) -> None:
            del param
            image_top = self.top_bar_h
            image_bottom = self.top_bar_h + self.target_height
            if event == cv2.EVENT_LBUTTONDOWN:
                if y >= image_bottom:
                    if _inside(self._upgrade_buttons['manual']['box'], x, y):
                        self._start_manual_workflow()
                    elif _inside(self._upgrade_buttons['left']['box'], x, y):
                        self._select_manual_side('left')
                    elif _inside(self._upgrade_buttons['right']['box'], x, y):
                        self._select_manual_side('right')
                    elif _inside(self._upgrade_buttons['done']['box'], x, y):
                        self._toggle_done()
                    elif _inside(self._upgrade_buttons['reset']['box'], x, y):
                        self._restore_auto_defaults()
                    elif _inside(self._upgrade_buttons['accept']['box'], x, y):
                        if self._workflow_stage in ('auto', 'relative'):
                            self.running = False
                        else:
                            self._set_stage_status('PRESS DONE BEFORE ACCEPT / EXIT')
                            self.redraw_gui()
                    return
                for side, bounds in self._snap_button_bounds.items():
                    if _inside(bounds, x, y):
                        self._attempt_snap(side)
                        return
                panel_side = self._panel_hit_side(x, y)
                if panel_side is not None and self._manual_edit_enabled:
                    active = self._active_absolute_side()
                    allowed = self._workflow_stage == 'relative' or panel_side == active
                    if allowed and self._begin_panel_drag(panel_side, x, y):
                        return
                active = self._active_absolute_side()
                if active is not None and image_top <= y < image_bottom and (x < self.display_width):
                    self.process_wafer_click(x, y - image_top)
                    self._main_drag_side = active
                return
            if event == cv2.EVENT_MOUSEMOVE:
                hovered = _inside(self._wafer_badge_bounds, x, y)
                if hovered != self._wafer_badge_hovered:
                    self._wafer_badge_hovered = hovered
                    self.redraw_gui()
                if self._panel_drag_side is not None and flags & cv2.EVENT_FLAG_LBUTTON:
                    self._continue_panel_drag(x, y)
                    return
                if self._main_drag_side is not None and flags & cv2.EVENT_FLAG_LBUTTON:
                    if image_top <= y < image_bottom and x < self.display_width:
                        active = self._active_absolute_side()
                        if active == self._main_drag_side:
                            self.process_wafer_click(x, y - image_top)
                    return
            if event == cv2.EVENT_LBUTTONUP:
                if self._panel_drag_side is not None:
                    self._continue_panel_drag(x, y)
                    self._finish_panel_drag()
                elif self._main_drag_side is not None:
                    if image_top <= y < image_bottom and x < self.display_width:
                        self.process_wafer_click(x, y - image_top)
                    self._main_drag_side = None
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
                if self._manual_button_flash_until and time.monotonic() >= self._manual_button_flash_until:
                    self._manual_button_flash_until = 0.0
                    self.redraw_gui()
                arrow = _decode_arrow_key(key) if key != -1 else None
                if arrow is not None:
                    self._nudge_active_side(arrow[0], arrow[1], coarse=_shift_key_down())
                elif key in (13, ord('a'), ord('A')):
                    if self._workflow_stage in ('auto', 'relative') and self.left_marker_global is not None and (self.right_marker_global is not None):
                        self.running = False
                    elif key != -1:
                        self._set_stage_status('PRESS DONE BEFORE ACCEPT / EXIT')
                        self.redraw_gui()
                elif key in (ord('m'), ord('M')):
                    self._start_manual_workflow()
                elif key in (ord('l'), ord('L')):
                    self._select_manual_side('left')
                elif key in (ord('r'), ord('R')):
                    self._select_manual_side('right')
                elif key in (ord('d'), ord('D'), 32):
                    self._toggle_done()
                elif key in (ord('p'), ord('P')):
                    self._restore_auto_defaults()
                elif key in (27, ord('q'), ord('Q')):
                    self.running = False
            cv2.destroyWindow(window)
    return StagedMarkerReviewTester
