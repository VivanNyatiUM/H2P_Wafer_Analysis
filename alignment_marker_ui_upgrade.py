"""Workflow and local-snap upgrade for the future-design marker review UI.

This module intentionally modifies only ``large_wafer_tester.LargeWaferTester``
after ``future_design_adapter`` has installed its marker-review subclass.  It
does not touch the downstream Tk wafer alignment UI in ``wafer_align_gui.py``.
"""

from __future__ import annotations

from h2p_ui_branding import install_global_branding as _install_h2p_ui_branding
_install_h2p_ui_branding()


import copy
import math
import time
from typing import Any, Iterable

import cv2
import numpy as np

UPGRADE_VERSION = "alignment-marker-review-v1-staged-drag-snap-2026-07-24"


def _real_box_items(boxes: dict[Any, Any]) -> list[tuple[Any, list[tuple[float, float]]]]:
    result: list[tuple[Any, list[tuple[float, float]]]] = []
    for key, corners in (boxes or {}).items():
        if key == ("area", 0) or not isinstance(corners, (list, tuple)) or len(corners) < 3:
            continue
        clean: list[tuple[float, float]] = []
        for point in corners:
            if not isinstance(point, (list, tuple, np.ndarray)) or len(point) < 2:
                continue
            x, y = float(point[0]), float(point[1])
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
    distances[distances < 1e-9] = np.inf
    nearest = np.min(distances, axis=1)
    nearest = nearest[np.isfinite(nearest)]
    return float(np.median(nearest)) if nearest.size else 80.0


def _transform_points(
    points: Iterable[tuple[float, float]],
    center: tuple[float, float],
    scale: float,
    angle_deg: float,
    translation: tuple[float, float] = (0.0, 0.0),
) -> list[tuple[float, float]]:
    angle = math.radians(float(angle_deg))
    cosine, sine = math.cos(angle), math.sin(angle)
    cx, cy = float(center[0]), float(center[1])
    tx, ty = float(translation[0]), float(translation[1])
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
    gradient = np.clip(magnitude / max(normalizer, 1e-6), 0.0, 1.0)

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
    # Preserve stronger gradients at the edge while still rewarding faint,
    # nearby outlines. This matters on the low-contrast yellow marker images.
    proximity *= 0.55 + 0.45 * gradient.astype(np.float32)
    proximity[binary > 0] = np.maximum(proximity[binary > 0], 0.82)
    return proximity


def find_local_template_snap(
    image_bgr: np.ndarray,
    boxes: dict[Any, Any],
    squares: dict[Any, Any],
    marker: tuple[float, float],
    *,
    angle_candidates: Iterable[float] | None = None,
    scale_candidates: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Fit a GDS-derived square template to nearby optical edges.

    Coordinates in ``boxes``, ``squares`` and ``marker`` are in the coordinate
    system of ``image_bgr``. The search is deliberately local; the absolute
    placement must already be approximately correct.
    """
    items = _real_box_items(boxes)
    if len(items) < 4:
        return {"ok": False, "reason": "not enough GDS marker boxes"}

    marker = (float(marker[0]), float(marker[1]))
    center_points = [
        (float(point[0]), float(point[1]))
        for key, point in (squares or {}).items()
        if key != ("area", 0) and isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2
    ]
    if len(center_points) < 4:
        center_points = [
            (float(np.mean([p[0] for p in corners])), float(np.mean([p[1] for p in corners])))
            for _key, corners in items
        ]
    pitch = max(8.0, _nearest_neighbor_pitch(center_points))
    search_radius = int(np.clip(round(0.55 * pitch), 24, 420))

    if angle_candidates is None:
        angle_candidates = (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)
    if scale_candidates is None:
        scale_candidates = (0.94, 0.97, 1.0, 1.03, 1.06)
    angles = [float(value) for value in angle_candidates]
    scales = [float(value) for value in scale_candidates]

    all_points = [point for _key, corners in items for point in corners]
    relative = np.asarray([[x - marker[0], y - marker[1]] for x, y in all_points], dtype=np.float64)
    max_radius_x = max(20.0, float(np.max(np.abs(relative[:, 0]))) * max(scales) + 10.0)
    max_radius_y = max(20.0, float(np.max(np.abs(relative[:, 1]))) * max(scales) + 10.0)
    common_x0 = max(0, int(math.floor(marker[0] - max_radius_x - search_radius - 8)))
    common_y0 = max(0, int(math.floor(marker[1] - max_radius_y - search_radius - 8)))
    common_x1 = min(image_bgr.shape[1], int(math.ceil(marker[0] + max_radius_x + search_radius + 8)))
    common_y1 = min(image_bgr.shape[0], int(math.ceil(marker[1] + max_radius_y + search_radius + 8)))
    if common_x1 - common_x0 < 20 or common_y1 - common_y0 < 20:
        return {"ok": False, "reason": "snap search region falls outside image"}

    proximity = _edge_proximity_image(image_bgr[common_y0:common_y1, common_x0:common_x1])
    if proximity.size == 0:
        return {"ok": False, "reason": "empty snap search image"}

    # Approximate physical square size controls line thickness without tying the
    # fitter to a particular acquisition resolution.
    side_lengths: list[float] = []
    for _key, corners in items:
        for p0, p1 in zip(corners, corners[1:] + corners[:1]):
            side_lengths.append(math.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    square_side = float(np.median(side_lengths)) if side_lengths else 0.5 * pitch
    line_thickness = int(np.clip(round(0.035 * square_side), 1, 4))

    best: dict[str, Any] | None = None
    baseline_score = 0.0

    for scale in scales:
        for angle_deg in angles:
            transformed_sets = [
                _transform_points(corners, marker, scale, angle_deg)
                for _key, corners in items
            ]
            transformed_points = [point for corners in transformed_sets for point in corners]
            min_x = math.floor(min(point[0] for point in transformed_points)) - 4
            min_y = math.floor(min(point[1] for point in transformed_points)) - 4
            max_x = math.ceil(max(point[0] for point in transformed_points)) + 4
            max_y = math.ceil(max(point[1] for point in transformed_points)) + 4
            template_w = int(max_x - min_x + 1)
            template_h = int(max_y - min_y + 1)
            if template_w < 5 or template_h < 5:
                continue

            template = np.zeros((template_h, template_w), dtype=np.float32)
            for corners in transformed_sets:
                local = np.asarray(
                    [[int(round(x - min_x)), int(round(y - min_y))] for x, y in corners],
                    dtype=np.int32,
                ).reshape((-1, 1, 2))
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

            if abs(scale - 1.0) < 1e-9 and abs(angle_deg) < 1e-9:
                bx = int(np.clip(expected_x, search_x0, search_x1 - template_w))
                by = int(np.clip(expected_y, search_y0, search_y1 - template_h))
                patch = proximity[by : by + template_h, bx : bx + template_w]
                if patch.shape == template.shape:
                    baseline_score = float(np.sum(patch * template) / template_mass)

            displacement = math.hypot(dx, dy)
            spatial_penalty = 0.22 * (displacement / max(float(search_radius), 1.0)) ** 2
            transform_penalty = 0.012 * (abs(angle_deg) / 3.0) ** 2 + 0.010 * (abs(scale - 1.0) / 0.06) ** 2
            selection_score = score - spatial_penalty - transform_penalty
            candidate = {
                "score": score,
                "selection_score": selection_score,
                "dx": dx,
                "dy": dy,
                "scale": scale,
                "angle_deg": angle_deg,
                "pitch": pitch,
                "search_radius": search_radius,
                "baseline_score": baseline_score,
            }
            if best is None or selection_score > float(best["selection_score"]):
                best = candidate

    if best is None:
        return {"ok": False, "reason": "no valid snap candidates"}

    best["baseline_score"] = baseline_score
    improvement = float(best["score"] - baseline_score)
    best["improvement"] = improvement
    displacement = math.hypot(float(best["dx"]), float(best["dy"]))
    transform_size = abs(math.log(max(float(best["scale"]), 1e-9))) + abs(math.radians(float(best["angle_deg"])))

    # Edge coverage around 0.30 is already useful on faint bright-field scans.
    # A nearly motionless high-confidence result is reported as success so the
    # button can confirm an already-good manual placement rather than pretending
    # nothing happened.
    reliable = float(best["score"]) >= 0.30 and (
        improvement >= 0.018
        or float(best["score"]) >= 0.48
        or (displacement <= max(2.0, 0.04 * pitch) and transform_size <= 0.015)
    )
    best["ok"] = bool(reliable)
    if not reliable:
        best["reason"] = (
            f"weak edge match (score {best['score']:.2f}, improvement {improvement:+.2f})"
        )
    return best


def _inside(box: tuple[int, int, int, int], x: int, y: int) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def install_alignment_marker_ui_upgrade(pipeline: Any) -> None:
    """Install the staged marker-review workflow on an imported pipeline."""
    base_class = pipeline.large_wafer_tester.LargeWaferTester
    if getattr(base_class, "_h2p_alignment_marker_ui_upgrade", False):
        return

    class StagedMarkerReviewTester(base_class):
        _h2p_alignment_marker_ui_upgrade = True
        _h2p_alignment_marker_ui_upgrade_version = UPGRADE_VERSION

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._alignment_marker_upgrade_ready = False
            super().__init__(*args, **kwargs)

            # Re-layout only the marker-finding UI. The downstream Tk alignment
            # application remains the unmodified automatic-branch version.
            self.sidebar_w = max(420, int(getattr(self, "sidebar_w", 370)))
            self.top_bar_h = 72
            self.bottom_bar_h = 112
            self.recompute_display_sizes()

            self._workflow_stage = "auto"
            self._manual_button_flash_until = 0.0
            self._main_drag_side: str | None = None
            self._snap_button_bounds: dict[str, tuple[int, int, int, int]] = {}
            self._snap_status = {"left": "Available after absolute placement", "right": "Available after absolute placement"}
            self._snap_busy = False
            self._alignment_marker_upgrade_ready = True
            self._layout_upgrade_buttons()
            self.redraw_gui()

        def _layout_upgrade_buttons(self) -> None:
            labels = ("MANUAL ALIGN", "DONE LEFT", "DONE RIGHT", "RESET", "ACCEPT / EXIT")
            left_margin, right_margin, gap = 12, 12, 8
            total_gap = gap * (len(labels) - 1)
            width = max(90, (self.canvas_w - left_margin - right_margin - total_gap) // len(labels))
            y1 = self.top_bar_h + self.target_height + 34
            y2 = min(self.canvas_h - 23, y1 + 48)
            self._upgrade_buttons: dict[str, dict[str, Any]] = {}
            x = left_margin
            keys = ("manual", "done_left", "done_right", "reset", "accept")
            for key, label in zip(keys, labels):
                x2 = self.canvas_w - right_margin if key == "accept" else x + width
                self._upgrade_buttons[key] = {"label": label, "box": (int(x), int(y1), int(x2), int(y2))}
                x = x2 + gap

        def _active_absolute_side(self) -> str | None:
            if self._workflow_stage == "absolute_left":
                return "left"
            if self._workflow_stage == "absolute_right":
                return "right"
            return None

        def _stage_instruction(self) -> str:
            if self._workflow_stage == "absolute_left":
                return "LEFT: click or drag on the wafer, then fine-tune the left panel and press DONE LEFT"
            if self._workflow_stage == "absolute_right":
                return "RIGHT: click or drag on the wafer, then fine-tune the right panel and press DONE RIGHT"
            if self._workflow_stage == "relative":
                return "RELATIVE: move/scale/rotate either panel or use ATTEMPT SNAP, then ACCEPT / EXIT"
            return "Automatic marker boxes are active. MANUAL ALIGN starts the staged override workflow"

        def _set_stage_status(self, action: str | None = None) -> None:
            instruction = self._stage_instruction()
            if action:
                self.status_text = f"{action}. {instruction}"
            else:
                self.status_text = instruction
            if self._workflow_stage == "auto":
                self.status_bg_color = (0, 78, 0)
            elif self._workflow_stage == "relative":
                self.status_bg_color = (48, 78, 0)
            else:
                self.status_bg_color = (78, 48, 0)

        def _start_manual_workflow(self) -> None:
            self._manual_button_flash_until = time.monotonic() + 0.38
            self._workflow_stage = "absolute_left"
            self._manual_edit_enabled = True
            self.current_state = self.STATE_WAIT_LEFT
            self._main_drag_side = None
            self._panel_drag_side = None
            self._panel_drag_mode = None
            self._snap_status = {"left": "Locked until both absolute sides are done", "right": "Locked until both absolute sides are done"}
            left = getattr(self, "left_marker_global", None)
            if left is not None:
                self._set_panel_view_center("left", tuple(left))
            self._set_stage_status("MANUAL ALIGNMENT STARTED")
            self.redraw_gui()

        def _complete_left(self) -> None:
            if self._workflow_stage != "absolute_left":
                return
            self._main_drag_side = None
            self._workflow_stage = "absolute_right"
            self.current_state = self.STATE_WAIT_RIGHT
            right = getattr(self, "right_marker_global", None)
            if right is not None:
                self._set_panel_view_center("right", tuple(right))
            self._set_stage_status("LEFT ABSOLUTE LOCATION LOCKED")
            self.redraw_gui()

        def _complete_right(self) -> None:
            if self._workflow_stage != "absolute_right":
                return
            self._main_drag_side = None
            self._workflow_stage = "relative"
            self.current_state = self.STATE_FINISHED
            self._manual_edit_enabled = True
            self._snap_status = {"left": "Ready", "right": "Ready"}
            self._set_stage_status("RIGHT ABSOLUTE LOCATION LOCKED")
            self.redraw_gui()

        def _restore_auto_defaults(self) -> None:
            self._workflow_stage = "auto"
            self._main_drag_side = None
            self._snap_status = {"left": "Available after absolute placement", "right": "Available after absolute placement"}
            super()._restore_auto_defaults()
            self._set_stage_status("AUTOMATIC BOXES RESTORED")
            self.redraw_gui()

        def process_wafer_click(self, x: int, y: int) -> None:
            side = self._active_absolute_side()
            if side is None:
                return
            orig_x = float(x / self.scale)
            orig_y = float(y / self.scale)
            self._move_side_geometry(side, (orig_x, orig_y), update_panel_view=True)
            setattr(self, f"{side}_was_manual", True)
            setattr(self, f"{side}_resolved_success", True)
            self._set_stage_status(f"{side.upper()} ABSOLUTE LOCATION MOVED")
            self.redraw_gui()

        def _finish_panel_drag(self) -> None:
            side = self._panel_drag_side
            mode = self._panel_drag_mode
            super()._finish_panel_drag()
            if side is not None:
                label = {"translate": "MOVED", "scale": "SCALED", "rotate": "ROTATED"}.get(mode, "EDITED")
                self._set_stage_status(f"{side.upper()} TEMPLATE {label}")
                self.redraw_gui()

        def _apply_snap_result(self, side: str, result: dict[str, Any]) -> None:
            marker = getattr(self, f"{side}_marker_global", None)
            boxes = getattr(self, f"{side}_boxes_global", {}) or {}
            squares = getattr(self, f"{side}_squares_global", {}) or {}
            if marker is None:
                return
            translation = (float(result["dx"]), float(result["dy"]))
            scale = float(result["scale"])
            angle_deg = float(result["angle_deg"])
            transformed_boxes = {
                key: _transform_points(corners, marker, scale, angle_deg, translation)
                for key, corners in boxes.items()
            }
            transformed_squares = {
                key: _transform_points([point], marker, scale, angle_deg, translation)[0]
                for key, point in squares.items()
            }
            new_marker = (float(marker[0]) + translation[0], float(marker[1]) + translation[1])
            setattr(self, f"{side}_boxes_global", transformed_boxes)
            setattr(self, f"{side}_squares_global", transformed_squares)
            setattr(self, f"{side}_marker_global", new_marker)
            setattr(self, f"{side}_click_global", new_marker)
            setattr(self, f"{side}_was_manual", True)
            setattr(self, f"{side}_resolved_success", True)
            self._manual_similarity_edited[side] = True
            self._manual_template_scale[side] = float(self._manual_template_scale.get(side, 1.0)) * scale
            self._manual_template_rotation_deg[side] = float(self._manual_template_rotation_deg.get(side, 0.0)) + angle_deg

        def _attempt_snap(self, side: str) -> None:
            if self._workflow_stage != "relative" or self._snap_busy:
                return
            marker = getattr(self, f"{side}_marker_global", None)
            if marker is None:
                self._snap_status[side] = "No marker geometry"
                self.redraw_gui()
                return
            self._snap_busy = True
            self._snap_status[side] = "Searching local edges..."
            self.redraw_gui()
            try:
                # The marker-review object may be backed by a virtual stitched
                # image; crop only the local search region indirectly by making a
                # bounded review image around the current template.
                boxes = getattr(self, f"{side}_boxes_global", {}) or {}
                squares = getattr(self, f"{side}_squares_global", {}) or {}
                items = _real_box_items(boxes)
                points = [point for _key, corners in items for point in corners]
                centers = [point for key, point in squares.items() if key != ("area", 0)]
                pitch = max(8.0, _nearest_neighbor_pitch(centers))
                margin = int(np.clip(round(1.8 * pitch), 80, 900))
                x0 = max(0, int(math.floor(min([p[0] for p in points] + [marker[0]]) - margin)))
                y0 = max(0, int(math.floor(min([p[1] for p in points] + [marker[1]]) - margin)))
                x1 = min(self.orig_w, int(math.ceil(max([p[0] for p in points] + [marker[0]]) + margin)))
                y1 = min(self.orig_h, int(math.ceil(max([p[1] for p in points] + [marker[1]]) + margin)))
                crop_rgb = np.asarray(self.im.crop((x0, y0, x1, y1)).convert("RGB"))
                crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                local_boxes = {
                    key: [(float(x) - x0, float(y) - y0) for x, y in corners]
                    for key, corners in boxes.items()
                }
                local_squares = {
                    key: (float(point[0]) - x0, float(point[1]) - y0)
                    for key, point in squares.items()
                }
                local_marker = (float(marker[0]) - x0, float(marker[1]) - y0)
                result = find_local_template_snap(crop_bgr, local_boxes, local_squares, local_marker)
                if not result.get("ok"):
                    self._snap_status[side] = str(result.get("reason", "No reliable local match"))
                    self._set_stage_status(f"{side.upper()} SNAP NOT APPLIED")
                else:
                    self._apply_snap_result(side, result)
                    self._snap_status[side] = (
                        f"score {result['score']:.2f}  move ({result['dx']:+.1f}, {result['dy']:+.1f}) px  "
                        f"rot {result['angle_deg']:+.1f} deg  scale {result['scale']:.3f}"
                    )
                    self._set_stage_status(f"{side.upper()} TEMPLATE SNAPPED TO LOCAL BOX EDGES")
            except Exception as exc:
                self._snap_status[side] = f"Snap failed: {exc}"
                self._set_stage_status(f"{side.upper()} SNAP FAILED")
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
            enabled = self._workflow_stage == "relative" and not self._snap_busy
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
            label = "ATTEMPT SNAP"
            tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0][0]
            cv2.putText(self.canvas, label, (bx1 + max(4, (bx2 - bx1 - tw) // 2), by1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (245, 245, 245), 1, cv2.LINE_AA)
            status = str(self._snap_status.get(side, ""))
            max_chars = 39
            if len(status) > max_chars:
                status = status[: max_chars - 1] + "..."
            cv2.putText(self.canvas, status, (bx2 + 10, by1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (205, 205, 205), 1, cv2.LINE_AA)

        def _draw_bottom_button(self, key: str) -> None:
            button = self._upgrade_buttons[key]
            x1, y1, x2, y2 = button["box"]
            stage = self._workflow_stage
            enabled = True
            completed = False
            if key == "done_left":
                enabled = stage == "absolute_left"
                completed = stage in ("absolute_right", "relative")
            elif key == "done_right":
                enabled = stage == "absolute_right"
                completed = stage == "relative"

            if key == "manual":
                flashing = time.monotonic() < self._manual_button_flash_until
                fill = (55, 220, 75) if flashing else ((0, 145, 35) if stage != "auto" else (0, 112, 28))
            elif key in ("done_left", "done_right"):
                fill = (0, 132, 95) if enabled else ((0, 92, 58) if completed else (58, 58, 58))
            elif key == "reset":
                fill = (0, 112, 130)
            else:
                enabled = stage in ("auto", "relative")
                fill = (42, 62, 150) if enabled else (58, 58, 58)
            border = (150, 255, 205) if enabled else (95, 95, 95)
            cv2.rectangle(self.canvas, (x1, y1), (x2, y2), fill, -1)
            cv2.rectangle(self.canvas, (x1, y1), (x2, y2), border, 1)
            label = button["label"]
            font_scale = 0.43 if key != "accept" else 0.40
            text = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            cv2.putText(self.canvas, label, (x1 + max(5, (x2 - x1 - text[0]) // 2), y1 + (y2 - y1 + text[1]) // 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (245, 245, 245), 1, cv2.LINE_AA)

        def redraw_gui(self) -> None:
            if not getattr(self, "_alignment_marker_upgrade_ready", False):
                return super().redraw_gui()

            if self.canvas.shape[:2] != (self.canvas_h, self.canvas_w):
                self.canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
            self.canvas[:] = 0

            cv2.rectangle(self.canvas, (0, 0), (self.canvas_w, self.top_bar_h), self.status_bg_color, -1)
            title = str(self.status_text).replace("\n", " ")
            if len(title) > 112:
                title = title[:109] + "..."
            title_w = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
            cv2.putText(self.canvas, title, (max(8, (self.canvas_w - title_w) // 2), 27), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 255, 235), 1, cv2.LINE_AA)
            instruction = self._stage_instruction()
            if len(instruction) > 118:
                instruction = instruction[:115] + "..."
            inst_w = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
            cv2.putText(self.canvas, instruction, (max(8, (self.canvas_w - inst_w) // 2), 54), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

            self.canvas[self.top_bar_h : self.top_bar_h + self.target_height, 0 : self.display_width] = self.preview_color.copy()
            self._draw_side_on_main("left")
            self._draw_side_on_main("right")

            sidebar_x = self.display_width
            cv2.rectangle(self.canvas, (sidebar_x, self.top_bar_h), (self.canvas_w, self.top_bar_h + self.target_height), (35, 35, 38), -1)
            self._panel_maps = {}
            self._snap_button_bounds = {}
            gap = 10
            panel_height = (self.target_height - 3 * gap) // 2
            self._draw_panel_with_snap("left", self.top_bar_h + gap, panel_height)
            self._draw_panel_with_snap("right", self.top_bar_h + 2 * gap + panel_height, panel_height)

            bottom_y = self.top_bar_h + self.target_height
            cv2.rectangle(self.canvas, (0, bottom_y), (self.canvas_w, self.canvas_h), (24, 24, 27), -1)
            cv2.putText(self.canvas, "Full wafer: click/drag absolute location   |   Panels: body move, corners scale, circle rotate", (14, bottom_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (195, 195, 195), 1, cv2.LINE_AA)
            self._layout_upgrade_buttons()
            for key in ("manual", "done_left", "done_right", "reset", "accept"):
                self._draw_bottom_button(key)
            cv2.putText(self.canvas, "Keys: M/L manual  |  P reset  |  Enter/A accept when ready  |  Q/Esc exit", (14, self.canvas_h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1, cv2.LINE_AA)
            cv2.imshow("Large Wafer Tester", self.canvas)

        def handle_click(self, event, x, y, flags, param) -> None:
            del param
            image_top = self.top_bar_h
            image_bottom = self.top_bar_h + self.target_height

            if event == cv2.EVENT_LBUTTONDOWN:
                if y >= image_bottom:
                    if _inside(self._upgrade_buttons["manual"]["box"], x, y):
                        self._start_manual_workflow()
                    elif _inside(self._upgrade_buttons["done_left"]["box"], x, y):
                        self._complete_left()
                    elif _inside(self._upgrade_buttons["done_right"]["box"], x, y):
                        self._complete_right()
                    elif _inside(self._upgrade_buttons["reset"]["box"], x, y):
                        self._restore_auto_defaults()
                    elif _inside(self._upgrade_buttons["accept"]["box"], x, y):
                        if self._workflow_stage in ("auto", "relative"):
                            self.running = False
                        else:
                            self._set_stage_status("FINISH LEFT AND RIGHT ABSOLUTE PLACEMENT FIRST")
                            self.redraw_gui()
                    return

                for side, bounds in self._snap_button_bounds.items():
                    if _inside(bounds, x, y):
                        self._attempt_snap(side)
                        return

                panel_side = self._panel_hit_side(x, y)
                if panel_side is not None and self._manual_edit_enabled:
                    active = self._active_absolute_side()
                    allowed = self._workflow_stage == "relative" or panel_side == active
                    if allowed and self._begin_panel_drag(panel_side, x, y):
                        return

                active = self._active_absolute_side()
                if active is not None and image_top <= y < image_bottom and x < self.display_width:
                    self.process_wafer_click(x, y - image_top)
                    self._main_drag_side = active
                return

            if event == cv2.EVENT_MOUSEMOVE:
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
            window = "Large Wafer Tester"
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
                if key in (13, 32, ord("a"), ord("A")):
                    if self._workflow_stage in ("auto", "relative") and self.left_marker_global is not None and self.right_marker_global is not None:
                        self.running = False
                    elif key != -1:
                        self._set_stage_status("FINISH LEFT AND RIGHT ABSOLUTE PLACEMENT FIRST")
                        self.redraw_gui()
                elif key in (ord("m"), ord("M"), ord("l"), ord("L")):
                    self._start_manual_workflow()
                elif key in (ord("r"), ord("R")):
                    if self._workflow_stage == "absolute_left":
                        self._complete_left()
                    elif self._workflow_stage == "absolute_right":
                        self._complete_right()
                elif key in (ord("p"), ord("P")):
                    self._restore_auto_defaults()
                elif key in (27, ord("q"), ord("Q")):
                    self.running = False
            cv2.destroyWindow(window)

    pipeline.large_wafer_tester.LargeWaferTester = StagedMarkerReviewTester
    current = str(getattr(pipeline, "WAFER_EXTRACTION_VERSION", "unknown"))
    if UPGRADE_VERSION not in current:
        pipeline.WAFER_EXTRACTION_VERSION = f"{current}+{UPGRADE_VERSION}"
    print(f"[Alignment Marker UI] Installed {UPGRADE_VERSION}")
