"""
OpenCV defect review/labeling UI for h2p_device_viewer.

This file provides two compatible tools:

1. DeviceDefectMapperTool
   Used by wafer_alignment_and_extraction.py during the original manual
   labeling stage.

2. AutoLabelReviewTool
   Used by defect_detector.py after automatic detection. It opens the same
   style of labeling UI with the auto-detected boxes already loaded from the
   subtraction-ready JSON. Edits are saved back into that JSON, so the reviewed
   output can be passed directly to subtract_defects.py.

Controls:
    Left drag             draw new defect box
    Right click box       delete the clicked box
    1..5                  choose defect type after drawing (standard mode)
    --quick-review         new boxes use generic "defect" with no type prompt
    N / Right / Space     next cell
    P / Left              previous cell
    Up / Down             jump to cell above / below
    X                     toggle excluded/damaged cell
    C                     clear all boxes in current cell
    Ctrl+Z                undo
    Ctrl+Shift+Z          redo
    Ctrl+Y                redo fallback
    Q / Esc               save and quit
"""

from __future__ import annotations

import copy
import io
import json
import math
import os
import re
import sys
import time
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

CLASS_COLORS = {
    "auto_defect": (0, 165, 255),
    "defect": (0, 165, 255),
    "blister": (0, 255, 0),
    "tear": (255, 0, 0),
    "delamination": (255, 0, 255),
    "particulate": (0, 0, 255),
    "hole": (0, 255, 255),
}

KEY_MAPPING = {
    ord("1"): "blister",
    ord("2"): "tear",
    ord("3"): "delamination",
    ord("4"): "particulate",
    ord("5"): "hole",
}

CLASS_NUMBERS = {v: str(k - ord("0")) for k, v in KEY_MAPPING.items()}

LEFT_ARROW_CODES = [2424832, 65361, 81, 0x250000, 63234]
RIGHT_ARROW_CODES = [2555904, 65363, 83, 0x270000, 63235]
UP_ARROW_CODES = [2490368, 65362, 82, 0x260000, 63232]
DOWN_ARROW_CODES = [2621440, 65364, 84, 0x280000, 63233]

GUI_VERSION = "fast-untyped-review-v8-2026-07-14"
WINDOW_NAME = "Device Defect Register"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _read_bgr(path: Path | str) -> Optional[np.ndarray]:
    path = Path(path)
    if not path.exists():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def _imwrite(path: Path | str, img: np.ndarray, params: Optional[list[int]] = None) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img, params or [])
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def _copy_bgr_to_windows_clipboard(img_bgr: np.ndarray) -> bool:
    """Copy an image to the Windows clipboard as CF_DIB. Best-effort."""
    if os.name != "nt":
        return False
    try:
        import ctypes
        import time
        from ctypes import wintypes

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        with io.BytesIO() as bio:
            pil_img.save(bio, format="BMP")
            dib = bio.getvalue()[14:]  # strip BITMAPFILEHEADER; clipboard wants DIB

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        CF_DIB = 8
        GMEM_MOVEABLE = 0x0002

        kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
        kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
        kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalLock.restype = ctypes.c_void_p
        kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalUnlock.restype = wintypes.BOOL
        user32.OpenClipboard.argtypes = [wintypes.HWND]
        user32.OpenClipboard.restype = wintypes.BOOL
        user32.EmptyClipboard.argtypes = []
        user32.EmptyClipboard.restype = wintypes.BOOL
        user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
        user32.SetClipboardData.restype = wintypes.HANDLE
        user32.CloseClipboard.argtypes = []
        user32.CloseClipboard.restype = wintypes.BOOL

        hglobal = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(dib))
        if not hglobal:
            return False
        ptr = kernel32.GlobalLock(hglobal)
        if not ptr:
            return False
        ctypes.memmove(ptr, dib, len(dib))
        kernel32.GlobalUnlock(hglobal)

        opened = False
        for _ in range(12):
            if user32.OpenClipboard(None):
                opened = True
                break
            time.sleep(0.035)
        if not opened:
            return False
        try:
            if not user32.EmptyClipboard():
                return False
            if not user32.SetClipboardData(CF_DIB, hglobal):
                return False
            # Ownership of hglobal transfers to clipboard on success.
            return True
        finally:
            user32.CloseClipboard()
    except Exception:
        return False


def _draw_card(
    canvas: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    fill: tuple[int, int, int] = (32, 34, 42),
    border: tuple[int, int, int] = (0, 215, 255),
    alpha: float = 0.96,
    thickness: int = 1,
) -> None:
    """Draw a clean semi-opaque side-panel card in-place."""
    h, w = canvas.shape[:2]
    x0 = max(0, min(int(x0), w - 1))
    x1 = max(0, min(int(x1), w - 1))
    y0 = max(0, min(int(y0), h - 1))
    y1 = max(0, min(int(y1), h - 1))
    if x1 <= x0 or y1 <= y0:
        return
    roi = canvas[y0:y1, x0:x1].copy()
    fill_img = np.full_like(roi, fill, dtype=np.uint8)
    cv2.addWeighted(fill_img, float(alpha), roi, 1.0 - float(alpha), 0, dst=roi)
    canvas[y0:y1, x0:x1] = roi
    cv2.rectangle(canvas, (x0, y0), (x1, y1), border, thickness, cv2.LINE_AA)


def _load_json(path: Path | str, default):
    path = Path(path)
    if not path.exists():
        return copy.deepcopy(default)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return copy.deepcopy(default)


def _save_json(path: Path | str, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def _round_pair(p: tuple[float, float] | list[float], ndigits: int = 3) -> list[float]:
    return [round(float(p[0]), ndigits), round(float(p[1]), ndigits)]


# ---------------------------------------------------------------------------
# Metadata-based crop pixel -> GDS mapping. Duplicated here intentionally so
# the review UI does not need to import defect_detector.py and create a circular
# dependency when defect_detector.py launches this UI.
# ---------------------------------------------------------------------------


def _fit_canvas_to_gds_affine(canvas_pts: np.ndarray, gds_pts: np.ndarray) -> np.ndarray:
    canvas_pts = np.asarray(canvas_pts, dtype=np.float64)
    gds_pts = np.asarray(gds_pts, dtype=np.float64)
    if canvas_pts.shape[0] < 3 or gds_pts.shape[0] < 3:
        raise ValueError("Need at least 3 canvas/GDS corner pairs to fit affine transform")
    A = []
    b = []
    for (x, y), (gx, gy) in zip(canvas_pts, gds_pts):
        A.append([x, y, 1.0, 0.0, 0.0, 0.0])
        b.append(gx)
        A.append([0.0, 0.0, 0.0, x, y, 1.0])
        b.append(gy)
    coeff, *_ = np.linalg.lstsq(np.asarray(A, dtype=np.float64), np.asarray(b, dtype=np.float64), rcond=None)
    return coeff.reshape(2, 3)


def _apply_affine_2x3(m: np.ndarray, x: float, y: float) -> tuple[float, float]:
    v = np.asarray([float(x), float(y), 1.0], dtype=np.float64)
    out = np.asarray(m, dtype=np.float64) @ v
    return float(out[0]), float(out[1])


def _metadata_canvas_to_gds_affine(meta: dict) -> np.ndarray:
    gds = np.asarray(meta.get("gds_corners_um") or meta.get("gds_corners") or [], dtype=np.float64)
    if gds.shape[0] < 3:
        raise ValueError(f"Metadata for {meta.get('cell_stem', '<unknown>')} lacks gds_corners_um")

    if "canvas_corners_px_downscaled" in meta:
        canvas = np.asarray(meta["canvas_corners_px_downscaled"], dtype=np.float64)
    elif "canvas_corners_px" in meta:
        canvas = np.asarray(meta["canvas_corners_px"], dtype=np.float64)
    elif "canvas_corners_px_fullres" in meta and "canvas_to_downscaled_scale_used" in meta:
        canvas = np.asarray(meta["canvas_corners_px_fullres"], dtype=np.float64) * float(meta["canvas_to_downscaled_scale_used"])
    elif "canvas_corners_px_fullres" in meta:
        canvas = np.asarray(meta["canvas_corners_px_fullres"], dtype=np.float64)
    else:
        raise ValueError(f"Metadata for {meta.get('cell_stem', '<unknown>')} lacks canvas corners")
    return _fit_canvas_to_gds_affine(canvas, gds)


def _crop_px_to_canvas_px(px: float, py: float, meta: dict) -> tuple[float, float]:
    if "crop_bounds_local_px_downscaled_before_resize" in meta:
        crop_x1, crop_y1, _crop_x2, _crop_y2 = [float(v) for v in meta["crop_bounds_local_px_downscaled_before_resize"]]
        resize_scale = float(meta.get("output_resize_scale", 1.0) or 1.0)
        if resize_scale <= 0:
            resize_scale = 1.0
        x_rot = crop_x1 + float(px) / resize_scale
        y_rot = crop_y1 + float(py) / resize_scale
        m = np.asarray(meta["rotation_matrix_2x3_downscaled"], dtype=np.float64)
        origin = np.asarray(meta["local_origin_px_downscaled"], dtype=np.float64)
    else:
        crop_x1, crop_y1, _crop_x2, _crop_y2 = [float(v) for v in meta["crop_bounds_local_px"]]
        x_rot = crop_x1 + float(px)
        y_rot = crop_y1 + float(py)
        m = np.asarray(meta["rotation_matrix_2x3"], dtype=np.float64)
        origin = np.asarray(meta["local_origin_px"], dtype=np.float64)

    inv = cv2.invertAffineTransform(m)
    local = inv @ np.asarray([x_rot, y_rot, 1.0], dtype=np.float64)
    canvas = local + origin
    return float(canvas[0]), float(canvas[1])


def crop_px_to_gds(px: float, py: float, meta: dict, canvas_to_gds: Optional[np.ndarray] = None) -> tuple[float, float]:
    if canvas_to_gds is None:
        canvas_to_gds = _metadata_canvas_to_gds_affine(meta)
    cx, cy = _crop_px_to_canvas_px(px, py, meta)
    return _apply_affine_2x3(canvas_to_gds, cx, cy)


# ---------------------------------------------------------------------------
# Keyboard helpers, including modifier detection on Windows so Ctrl+Z and
# Ctrl+Shift+Z work inside an OpenCV window.
# ---------------------------------------------------------------------------


def _win_key_down(vk_code: int) -> bool:
    if os.name != "nt":
        return False
    try:
        import ctypes
        return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)
    except Exception:
        return False


def _modifiers_down() -> tuple[bool, bool]:
    if os.name == "nt":
        ctrl = _win_key_down(0x11) or _win_key_down(0xA2) or _win_key_down(0xA3)
        shift = _win_key_down(0x10) or _win_key_down(0xA0) or _win_key_down(0xA1)
        return ctrl, shift
    return False, False


def _is_z_key(key: int) -> bool:
    key8 = key & 0xFF
    return key8 in (ord("z"), ord("Z"), 26)


def _is_y_key(key: int) -> bool:
    key8 = key & 0xFF
    return key8 in (ord("y"), ord("Y"), 25)


# ---------------------------------------------------------------------------
# Shared review UI base
# ---------------------------------------------------------------------------


class _BaseDefectReviewUI:
    def __init__(
        self,
        quick_label: bool = False,
        default_defect_type: str = "defect",
        image_cache_size: int = 12,
        autosave_seconds: float = 4.0,
        show_annotation_labels: Optional[bool] = None,
    ):
        self.max_disp_w = 950
        self.max_disp_h = 750
        self.panel_width = 360
        self.current_idx = 0
        self.cell_files: list[dict] = []
        self.annotations: dict[str, list[dict]] = {}
        self.exclusions: set[str] = set()
        self.drawing = False
        self.start_pt = (0, 0)
        self.current_pt = (0, 0)
        self.is_waiting_for_key = False
        self.undo_stack: list[dict] = []
        self.redo_stack: list[dict] = []
        self.max_history = 100

        # Fast review mode: new boxes are immediately recorded with one generic
        # type.  This avoids the blocking 1-5 prompt when defect class is not used
        # by downstream GDS subtraction.
        self.quick_label = bool(quick_label)
        self.default_defect_type = str(default_defect_type or "defect")
        self.show_annotation_labels = (not self.quick_label) if show_annotation_labels is None else bool(show_annotation_labels)

        # Persistence is dirty/debounced.  The previous UI rewrote the complete
        # multi-thousand-defect JSON on every arrow-key press, even when nothing
        # changed.  That was the main one-second navigation stall.
        self._annotations_dirty = False
        self._exclusions_dirty = False
        self._review_state_dirty = False
        self._last_edit_time = 0.0
        self.autosave_seconds = max(1.0, float(autosave_seconds))

        # Decode/resize cache and background neighbor prefetch.  Cached entries
        # contain only display-sized images, so a dozen cells costs modest RAM.
        self.image_cache_size = max(2, int(image_cache_size))
        self._image_cache: OrderedDict[int, dict] = OrderedDict()
        self._image_cache_lock = threading.Lock()
        self._prefetch_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="defect-review-prefetch")
        self._prefetch_pending: set[int] = set()

        # Display geometry caches are independent of subtraction geometry.
        # Polygons are simplified only for drawing, never in the output JSON.
        self._annotation_revision: dict[str, int] = {}
        self._display_annotation_cache: OrderedDict[tuple, list] = OrderedDict()
        self._display_annotation_cache_max = 32
        self._last_drag_present = 0.0
        self.verbose_navigation = False

        # While using Up/Down, keep a fixed physical x anchor so repeated
        # row jumps move vertically instead of drifting diagonally on edge rows.
        # Left/Right/N/P/minimap jumps reset this anchor.
        self.vertical_nav_anchor_x: Optional[float] = None
        self.copy_button_bounds: tuple[int, int, int, int] | None = None
        self.copy_feedback_until = 0.0
        self.copy_feedback_ok = False
        self.copy_feedback_redraw_pending = False

    # ---- persistence ------------------------------------------------------

    def load_existing_annotations(self) -> dict:
        return _load_json(self.output_json_path, {})

    def save_annotations_to_file(self, force: bool = False) -> None:
        if not force and not self._annotations_dirty:
            return
        _save_json(self.output_json_path, self.annotations)
        self._annotations_dirty = False

    def _mark_annotations_dirty(self, filename: Optional[str] = None) -> None:
        self._annotations_dirty = True
        self._last_edit_time = time.time()
        if filename is None and self.cell_files:
            filename = str(self.cell_files[self.current_idx].get("filename", ""))
        if filename:
            self._annotation_revision[filename] = int(self._annotation_revision.get(filename, 0)) + 1
            self._invalidate_display_annotation_cache(filename)

    def _mark_exclusions_dirty(self) -> None:
        self._exclusions_dirty = True
        self._last_edit_time = time.time()

    def _invalidate_display_annotation_cache(self, filename: str) -> None:
        stale = [k for k in self._display_annotation_cache if k and k[0] == filename]
        for key in stale:
            self._display_annotation_cache.pop(key, None)

    def _autosave_if_due(self) -> None:
        if not (self._annotations_dirty or self._exclusions_dirty or self._review_state_dirty):
            return
        if self.drawing or self.is_waiting_for_key:
            return
        if time.time() - float(self._last_edit_time) < self.autosave_seconds:
            return
        self.save_annotations_to_file(force=True)
        self.save_exclusions_file(force=True)
        self.save_review_state(force=True)

    def _review_state_path(self) -> Optional[Path]:
        value = getattr(self, "resume_state_path", None)
        return Path(value) if value else None

    def save_review_state(self, force: bool = False) -> None:
        path = self._review_state_path()
        if path is None or not self.cell_files:
            return
        if not force and not self._review_state_dirty:
            return
        idx = max(0, min(int(self.current_idx), len(self.cell_files) - 1))
        entry = self.cell_files[idx]
        _save_json(path, {
            "current_idx": idx,
            "filename": str(entry.get("filename", "")),
            "filepath": str(entry.get("filepath", "")),
            "annotation_json": str(getattr(self, "output_json_path", "")),
            "saved_at_unix": time.time(),
        })
        self._review_state_dirty = False

    def restore_review_state(self) -> None:
        path = self._review_state_path()
        if path is None or not path.exists() or not self.cell_files:
            return
        state = _load_json(path, {})
        wanted_name = str(state.get("filename", ""))
        wanted_path = str(state.get("filepath", ""))
        for idx, entry in enumerate(self.cell_files):
            if wanted_name and str(entry.get("filename", "")) == wanted_name:
                self.current_idx = idx
                return
            if wanted_path and str(entry.get("filepath", "")) == wanted_path:
                self.current_idx = idx
                return
        try:
            self.current_idx = max(0, min(int(state.get("current_idx", 0)), len(self.cell_files) - 1))
        except Exception:
            self.current_idx = 0

    def load_exclusions_file(self) -> set[str]:
        data = _load_json(self.exclusions_path, [])
        try:
            return set(data)
        except Exception:
            return set()

    def save_exclusions_file(self, force: bool = False) -> None:
        if not force and not self._exclusions_dirty:
            return
        _save_json(self.exclusions_path, sorted(list(self.exclusions)))
        self._exclusions_dirty = False

    # ---- undo/redo --------------------------------------------------------

    def _snapshot(self) -> dict:
        # Undo history stores only the active cell, not the complete wafer JSON.
        # Deep-copying thousands of polygons on every click was a major source of
        # lag while adding/deleting boxes.
        filename = ""
        cell_annotations: list[dict] = []
        if self.cell_files:
            filename = str(self.cell_files[self.current_idx].get("filename", ""))
            cell_annotations = copy.deepcopy(self.annotations.get(filename, []))
        return {
            "filename": filename,
            "cell_annotations": cell_annotations,
            "exclusions": sorted(list(self.exclusions)),
            "current_idx": int(self.current_idx),
        }

    def _restore_snapshot(self, snap: dict) -> None:
        filename = str(snap.get("filename", ""))
        if filename:
            self.annotations[filename] = copy.deepcopy(snap.get("cell_annotations", []))
            self._mark_annotations_dirty(filename)
        self.exclusions = set(snap.get("exclusions", []))
        self._mark_exclusions_dirty()
        self.current_idx = max(0, min(int(snap.get("current_idx", self.current_idx)), len(self.cell_files) - 1))
        self.load_cell_at_index(self.current_idx, verify=False)
        self.save_annotations_to_file(force=True)
        self.save_exclusions_file(force=True)

    def push_undo(self) -> None:
        self.undo_stack.append(self._snapshot())
        if len(self.undo_stack) > self.max_history:
            self.undo_stack = self.undo_stack[-self.max_history:]
        self.redo_stack.clear()

    def undo(self) -> None:
        if not self.undo_stack:
            self._status("Nothing to undo")
            return
        self.redo_stack.append(self._snapshot())
        snap = self.undo_stack.pop()
        self._restore_snapshot(snap)
        self._status("Undo")

    def redo(self) -> None:
        if not self.redo_stack:
            self._status("Nothing to redo")
            return
        self.undo_stack.append(self._snapshot())
        snap = self.redo_stack.pop()
        self._restore_snapshot(snap)
        self._status("Redo")

    def _status(self, msg: str) -> None:
        print(f"[Review UI] {msg}")

    # ---- cell/image loading ----------------------------------------------

    def _read_image_dims(self, path: Path) -> tuple[int, int]:
        try:
            with Image.open(path) as img_hdr:
                return img_hdr.size
        except Exception:
            img = _read_bgr(path)
            if img is not None:
                return int(img.shape[1]), int(img.shape[0])
        return self.max_disp_w, self.max_disp_h

    def _prepare_image_bundle(self, idx: int) -> dict:
        idx = max(0, min(int(idx), len(self.cell_files) - 1))
        with self._image_cache_lock:
            cached = self._image_cache.get(idx)
            if cached is not None:
                self._image_cache.move_to_end(idx)
                return cached

        entry = self.cell_files[idx]
        full_path = Path(entry["filepath"])
        native_w, native_h = self._read_image_dims(full_path)
        preview_path = Path(entry["preview_path"]) if entry.get("preview_path") else None
        use_preview = preview_path is not None and preview_path.exists()
        disp_path = preview_path if use_preview else full_path
        img = _read_bgr(disp_path)
        if img is None:
            img = np.zeros((self.max_disp_h, self.max_disp_w, 3), dtype=np.uint8)

        src_h, src_w = img.shape[:2]
        scale = min(self.max_disp_w / float(max(src_w, 1)), self.max_disp_h / float(max(src_h, 1)), 1.0)
        display_width = max(1, int(round(src_w * scale)))
        display_height = max(1, int(round(src_h * scale)))
        if (display_width, display_height) != (src_w, src_h):
            img_disp = cv2.resize(img, (display_width, display_height), interpolation=cv2.INTER_AREA)
        else:
            img_disp = img

        # Create a persistent display-sized preview after the first full-image
        # decode.  Future sessions can navigate without repeatedly decoding and
        # shrinking 3000px crops.  This contains no annotations.
        if not use_preview and preview_path is not None:
            try:
                _imwrite(preview_path, img_disp, [cv2.IMWRITE_JPEG_QUALITY, 91])
            except Exception:
                pass

        bundle = {
            "native_w": int(native_w),
            "native_h": int(native_h),
            "img_disp": img_disp,
            "display_width": int(display_width),
            "display_height": int(display_height),
            "source_w": int(src_w),
            "source_h": int(src_h),
        }
        with self._image_cache_lock:
            self._image_cache[idx] = bundle
            self._image_cache.move_to_end(idx)
            while len(self._image_cache) > self.image_cache_size:
                self._image_cache.popitem(last=False)
            self._prefetch_pending.discard(idx)
        return bundle

    def _prefetch_one(self, idx: int) -> None:
        try:
            self._prepare_image_bundle(idx)
        except Exception:
            with self._image_cache_lock:
                self._prefetch_pending.discard(idx)

    def _schedule_neighbor_prefetch(self, idx: int) -> None:
        # Favor the likely next cell, but warm both directions and row-neighbors.
        order = [idx + 1, idx - 1, idx + 2, idx - 2]
        for candidate in order:
            if candidate < 0 or candidate >= len(self.cell_files):
                continue
            with self._image_cache_lock:
                if candidate in self._image_cache or candidate in self._prefetch_pending:
                    continue
                self._prefetch_pending.add(candidate)
            try:
                self._prefetch_executor.submit(self._prefetch_one, candidate)
            except RuntimeError:
                with self._image_cache_lock:
                    self._prefetch_pending.discard(candidate)

    def _shutdown_prefetch(self) -> None:
        try:
            self._prefetch_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    def load_cell_at_index(self, idx: int, verify: bool = True) -> None:
        self.current_idx = max(0, min(int(idx), len(self.cell_files) - 1))
        entry = self.cell_files[self.current_idx]
        bundle = self._prepare_image_bundle(self.current_idx)
        self.native_w = int(bundle["native_w"])
        self.native_h = int(bundle["native_h"])
        self.display_width = int(bundle["display_width"])
        self.display_height = int(bundle["display_height"])
        self.img_disp = bundle["img_disp"]
        self.img_orig = self.img_disp
        self.orig_h, self.orig_w = int(bundle["source_h"]), int(bundle["source_w"])
        self.scale = min(self.display_width / float(max(self.orig_w, 1)), self.display_height / float(max(self.orig_h, 1)))
        self.annotations.setdefault(entry["filename"], [])
        if verify and hasattr(self, "transformer") and self.transformer is not None and entry.get("cell_data") is not None:
            try:
                if self.verbose_navigation:
                    print(
                        f"\n[DefectMapper] Loading cell {entry['cell_data']['row']}-{entry['cell_data']['col']} "
                        f"(native {self.native_w}x{self.native_h}, display source {self.orig_w}x{self.orig_h})"
                    )
                if hasattr(self.transformer, "verify_crop_corners"):
                    self.transformer.verify_crop_corners(entry["cell_data"], shave=self.shave, pad=self.pad)
            except Exception as exc:
                print(f"[DefectMapper] corner verification skipped: {exc}")
        elif self.verbose_navigation:
            print(f"\n[Review UI] Loading {entry['filename']} ({self.current_idx + 1}/{len(self.cell_files)})")
        self.redraw_canvas()
        self._review_state_dirty = True
        self._last_edit_time = time.time()
        self._schedule_neighbor_prefetch(self.current_idx)

    # ---- geometry ---------------------------------------------------------

    def _display_to_native_rect(self) -> tuple[int, int, int, int]:
        scale_up_x = self.native_w / float(self.display_width)
        scale_up_y = self.native_h / float(self.display_height)
        orig_x1 = max(0, min(int(round(min(self.start_pt[0], self.current_pt[0]) * scale_up_x)), self.native_w))
        orig_y1 = max(0, min(int(round(min(self.start_pt[1], self.current_pt[1]) * scale_up_y)), self.native_h))
        orig_x2 = max(0, min(int(round(max(self.start_pt[0], self.current_pt[0]) * scale_up_x)), self.native_w))
        orig_y2 = max(0, min(int(round(max(self.start_pt[1], self.current_pt[1]) * scale_up_y)), self.native_h))
        return orig_x1, orig_y1, orig_x2, orig_y2

    def _record_from_native_bbox(self, entry: dict, bbox_px: list[int], assigned_class: str) -> dict:
        raise NotImplementedError

    def _delete_box_at_display_point(self, x: int, y: int) -> bool:
        if x >= self.display_width or y >= self.display_height:
            return False
        entry = self.cell_files[self.current_idx]
        filename = entry["filename"]
        anns = self.annotations.get(filename, [])
        if not anns:
            return False
        native_x = x * self.native_w / float(self.display_width)
        native_y = y * self.native_h / float(self.display_height)
        for idx in range(len(anns) - 1, -1, -1):
            box = anns[idx].get("box_px") or anns[idx].get("bbox_px")
            if not box or len(box) < 4:
                continue
            bx, by, bw, bh = [float(v) for v in box[:4]]
            pad = 3.0
            if bx - pad <= native_x <= bx + bw + pad and by - pad <= native_y <= by + bh + pad:
                self.push_undo()
                removed = anns.pop(idx)
                self._mark_annotations_dirty(filename)
                self.redraw_canvas()
                self._status(f"Deleted {removed.get('type', 'defect')} from {filename}")
                return True
        return False

    # ---- drawing ----------------------------------------------------------

    def redraw_canvas(self) -> None:
        shape = (self.display_height, self.display_width + self.panel_width, 3)
        if not hasattr(self, "canvas") or self.canvas.shape != shape:
            self.canvas = np.empty(shape, dtype=np.uint8)
        self.canvas.fill(0)
        self.canvas[:, :self.display_width] = self.img_disp
        self.canvas[:, self.display_width:] = 40

        entry = self.cell_files[self.current_idx]
        filename = entry["filename"]

        cv2.putText(self.canvas, "DEVICE DEFECT REVIEW", (self.display_width + 15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(self.canvas, f"Index: {self.current_idx + 1} / {len(self.cell_files)}", (self.display_width + 15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(self.canvas, f"File: {filename}", (self.display_width + 15, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(self.canvas, f"Boxes: {len(self.annotations.get(filename, []))}", (self.display_width + 15, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

        controls = [
            "Drag: add generic defect" if self.quick_label else "Drag: add box",
            "Right-click box: delete",
            "No type prompt (quick mode)" if self.quick_label else "1-5: type after drawing",
            "N/Space/Right: next",
            "P/Left: previous",
            "Up/Down: same-column row jump",
            "C: clear current cell",
            "X: exclude/current damaged",
            "L: toggle text labels",
            "Ctrl+Z: undo",
            "Ctrl+Shift+Z or Ctrl+Y: redo",
            "K or COPY: copy current image",
            "Q/Esc: quit/save & quit",
        ]
        y0 = 136
        cv2.putText(self.canvas, "CONTROLS:", (self.display_width + 15, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        for i, line in enumerate(controls):
            cv2.putText(self.canvas, line, (self.display_width + 15, y0 + 22 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)

        self._draw_cell_map()
        self._draw_copy_button()

        if filename in self.exclusions:
            cv2.rectangle(self.canvas, (0, 0), (self.display_width, self.display_height), (0, 0, 255), 4)
            cv2.putText(self.canvas, "EXCLUDED", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        self._draw_annotations(filename)
        cv2.imshow(WINDOW_NAME, self.canvas)

    def _draw_copy_button(self, target: Optional[np.ndarray] = None) -> None:
        """Draw a slim vertical copy button beside the wafer map."""
        if target is None:
            target = self.canvas
        panel_left = self.display_width
        map_bounds = getattr(self, "cell_map_outer_bounds", None)

        if map_bounds is not None:
            _mx1, map_y1, map_x2, map_y2 = map_bounds
            x1 = min(panel_left + self.panel_width - 82, map_x2 + 16)
            x2 = panel_left + self.panel_width - 18
            y1 = map_y1
            y2 = map_y2
        else:
            x1 = panel_left + self.panel_width - 82
            x2 = panel_left + self.panel_width - 18
            y1 = max(500, self.display_height - 230)
            y2 = self.display_height - 10

        # Keep the button as a separate vertical control beside the minimap, not
        # above or on top of it.
        if x2 - x1 < 48:
            x1 = panel_left + self.panel_width - 70
        if y2 - y1 < 120:
            y1 = max(330, self.display_height - 230)
            y2 = self.display_height - 10

        self.copy_button_bounds = (int(x1), int(y1), int(x2), int(y2))

        active = time.time() < float(getattr(self, "copy_feedback_until", 0.0))
        ok = bool(getattr(self, "copy_feedback_ok", False))
        copied = active and ok
        border = (80, 255, 130) if copied else (0, 210, 255)
        fill = (26, 48, 32) if copied else (35, 38, 46)
        label_color = (80, 255, 130) if copied else (0, 255, 255)

        _draw_card(target, x1 + 3, y1 + 4, x2 + 3, y2 + 4, fill=(18, 18, 22), border=(18, 18, 22), alpha=0.55, thickness=0)
        _draw_card(target, x1, y1, x2, y2, fill=fill, border=border, alpha=0.98, thickness=1)
        cv2.rectangle(target, (x1 + 5, y1 + 5), (x2 - 5, y2 - 5), (70, 70, 78), 1, cv2.LINE_AA)

        letters = list("COPIED" if copied else "COPY")
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        line_h = 22 if len(letters) >= 6 else 24
        text_heights = []
        for ch in letters:
            (_tw, th), base = cv2.getTextSize(ch, font, font_scale, thickness)
            text_heights.append(th + base)
        block_h = line_h * (len(letters) - 1) + max(text_heights)
        center_y = (y1 + y2) // 2
        start_baseline = int(center_y - block_h / 2 + max(text_heights))

        for i, ch in enumerate(letters):
            (tw, th), base = cv2.getTextSize(ch, font, font_scale, thickness)
            tx = int((x1 + x2 - tw) / 2)
            ty = int(start_baseline + i * line_h)
            cv2.putText(target, ch, (tx, ty), font, font_scale, label_color, thickness, cv2.LINE_AA)
    def copy_current_view(self) -> None:
        """Copy the current annotated left-pane image to clipboard and save a file."""
        try:
            entry = self.cell_files[self.current_idx]
            stem = Path(str(entry.get("filename", f"cell_{self.current_idx+1}"))).stem
            # self.canvas already includes the current annotation overlays.  Only copy
            # the image pane, not the right-side controls panel.
            view = self.canvas[:, :self.display_width].copy()
            out_dir = Path(self.output_json_path).parent / "review_copies"
            out_path = out_dir / f"{stem}_review_copy.png"
            _imwrite(out_path, view)
            copied = _copy_bgr_to_windows_clipboard(view)
            self.copy_feedback_until = time.time() + 1.0
            self.copy_feedback_ok = True
            self.copy_feedback_redraw_pending = True
            if copied:
                self._status(f"Copied current view to clipboard and saved {out_path}")
            else:
                self._status(f"Saved current view to {out_path} (clipboard unavailable)")
        except Exception as exc:
            self.copy_feedback_until = time.time() + 1.0
            self.copy_feedback_ok = True
            self.copy_feedback_redraw_pending = True
            self._status(f"Copy failed: {exc}")


    def _draw_cell_map(self) -> None:
        if not self.cell_files:
            return
        entries = [e for e in self.cell_files if e.get("gds_bbox") is not None]
        if not entries:
            return
        xs = []
        ys = []
        for e in entries:
            min_x, min_y, max_x, max_y = e["gds_bbox"]
            xs.extend([min_x, max_x])
            ys.extend([min_y, max_y])
        all_min_x, all_max_x = min(xs), max(xs)
        all_min_y, all_max_y = min(ys), max(ys)
        gds_w = max(all_max_x - all_min_x, 1e-9)
        gds_h = max(all_max_y - all_min_y, 1e-9)

        # Keep the map low enough that the temporary defect-type chooser can
        # live in the blank space above it without covering the map.
        map_size, map_padding = 220, 13
        map_draw_size = map_size - 2 * map_padding
        map_x_start = self.display_width + 40
        preferred_map_y = max(500, self.display_height - map_size - 10)
        map_y_start = min(preferred_map_y, max(330, self.display_height - map_size - 10))
        cv2.rectangle(self.canvas, (map_x_start - 10, map_y_start - 10), (map_x_start + map_size - 10, map_y_start + map_size - 10), (30, 30, 30), -1)
        cv2.rectangle(self.canvas, (map_x_start - 10, map_y_start - 10), (map_x_start + map_size - 10, map_y_start + map_size - 10), (100, 100, 100), 1)
        self.cell_map_outer_bounds = (map_x_start - 10, map_y_start - 10, map_x_start + map_size - 10, map_y_start + map_size - 10)

        self.cell_map_bounds = []
        for idx, entry in enumerate(self.cell_files):
            bbox = entry.get("gds_bbox")
            if bbox is None:
                self.cell_map_bounds.append((0, 0, -1, -1))
                continue
            min_x, min_y, max_x, max_y = bbox
            norm_x1 = (min_x - all_min_x) / gds_w
            norm_y1 = (min_y - all_min_y) / gds_h
            norm_x2 = (max_x - all_min_x) / gds_w
            norm_y2 = (max_y - all_min_y) / gds_h
            mx1 = int(map_x_start + min(norm_x1, norm_x2) * map_draw_size)
            my1 = int(map_y_start + (1.0 - max(norm_y1, norm_y2)) * map_draw_size)
            mx2 = int(map_x_start + max(norm_x1, norm_x2) * map_draw_size)
            my2 = int(map_y_start + (1.0 - min(norm_y1, norm_y2)) * map_draw_size)
            self.cell_map_bounds.append((mx1, my1, mx2, my2))
            is_current = idx == self.current_idx
            is_excluded = entry["filename"] in self.exclusions
            has_ann = len(self.annotations.get(entry["filename"], [])) > 0
            color = (0, 165, 255) if is_current else ((0, 0, 150) if is_excluded else ((0, 120, 0) if has_ann else (80, 80, 80)))
            thickness = -1 if (is_current or is_excluded or has_ann) else 1
            cv2.rectangle(self.canvas, (mx1, my1), (mx2, my2), color, thickness)
            if thickness == -1:
                cv2.rectangle(self.canvas, (mx1, my1), (mx2, my2), (200, 200, 200) if is_current else (40, 40, 40), 1)

    def _prepared_display_annotations(self, filename: str) -> list[dict]:
        revision = int(self._annotation_revision.get(filename, 0))
        key = (filename, revision, int(self.display_width), int(self.display_height), int(self.native_w), int(self.native_h))
        cached = self._display_annotation_cache.get(key)
        if cached is not None:
            self._display_annotation_cache.move_to_end(key)
            return cached

        scale_down_x = self.display_width / float(max(self.native_w, 1))
        scale_down_y = self.display_height / float(max(self.native_h, 1))
        prepared: list[dict] = []
        for box in self.annotations.get(filename, []):
            raw = box.get("box_px") or box.get("bbox_px")
            if not raw or len(raw) < 4:
                continue
            x_tl, y_tl, w, h = [float(v) for v in raw[:4]]
            item = {
                "rect": (
                    int(round(x_tl * scale_down_x)),
                    int(round(y_tl * scale_down_y)),
                    int(round((x_tl + w) * scale_down_x)),
                    int(round((y_tl + h) * scale_down_y)),
                ),
                "type": str(box.get("type", "auto_defect")),
                "score": box.get("score"),
                "polygon": None,
            }
            polygon = box.get("polygon_px") or []
            if len(polygon) >= 3:
                pts = np.asarray(
                    [[int(round(float(px) * scale_down_x)), int(round(float(py) * scale_down_y))] for px, py in polygon],
                    dtype=np.int32,
                ).reshape(-1, 1, 2)
                # Display-only simplification prevents huge contour point lists
                # from stalling OpenCV.  Exact polygon_px/polygon_gds is untouched.
                if len(pts) > 48:
                    peri = cv2.arcLength(pts, True)
                    pts = cv2.approxPolyDP(pts, max(0.65, 0.0015 * peri), True)
                item["polygon"] = pts
            prepared.append(item)

        self._display_annotation_cache[key] = prepared
        self._display_annotation_cache.move_to_end(key)
        while len(self._display_annotation_cache) > self._display_annotation_cache_max:
            self._display_annotation_cache.popitem(last=False)
        return prepared

    def _draw_annotations(self, filename: str) -> None:
        for item in self._prepared_display_annotations(filename):
            sc_x1, sc_y1, sc_x2, sc_y2 = item["rect"]
            defect_type = item["type"]
            color = CLASS_COLORS.get(defect_type, (255, 255, 255))
            pts = item.get("polygon")
            has_polygon = pts is not None and len(pts) >= 3
            rect_color = (0, 105, 190) if has_polygon else color
            rect_thickness = 1 if has_polygon else 2
            cv2.rectangle(self.canvas, (sc_x1, sc_y1), (sc_x2, sc_y2), rect_color, rect_thickness, cv2.LINE_8)
            if has_polygon:
                cv2.polylines(self.canvas, [pts], True, (0, 255, 255), 2, cv2.LINE_8)

            if self.show_annotation_labels:
                class_num = CLASS_NUMBERS.get(defect_type)
                label = f"[{class_num}] {defect_type}" if class_num else defect_type
                score = item.get("score")
                if score is not None:
                    try:
                        label += f" {float(score):.1f}"
                    except Exception:
                        pass
                cv2.putText(self.canvas, label, (sc_x1, max(12, sc_y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    def _type_prompt_rect(self) -> tuple[int, int, int, int]:
        """Return the right-panel classification-card rect.

        The card uses the full available panel width above the minimap.  It no
        longer reserves a right strip for COPY because the vertical COPY button
        lives beside the minimap below this card.  This prevents the two-column
        class legend from clipping labels like "particulate".
        """
        x0 = self.display_width + 14
        x1 = self.display_width + self.panel_width - 14
        card_h = 132

        # Controls block starts at y=136. Keep the chooser below it.
        controls_bottom = 136 + 22 + 18 * 11 + 18

        map_size = 220
        preferred_map_y = max(500, self.display_height - map_size - 10)
        map_y_start = min(preferred_map_y, max(330, self.display_height - map_size - 10))
        max_y0_without_map_overlap = map_y_start - card_h - 12

        if max_y0_without_map_overlap >= controls_bottom:
            y0 = controls_bottom
        else:
            y0 = min(max(controls_bottom, 330), max(0, self.display_height - card_h - 8))
        y1 = min(self.display_height - 8, y0 + card_h)
        return int(x0), int(y0), int(x1), int(y1)

    def _draw_type_prompt_card(self, target: np.ndarray, copied: bool = False) -> None:
        """Draw the 1-5 classifier prompt without text clipping."""
        x0, y0, x1, y1 = self._type_prompt_rect()
        w = max(1, x1 - x0)

        _draw_card(target, x0, y0, x1, y1, fill=(26, 28, 34), border=(0, 220, 255), alpha=0.98, thickness=1)
        cv2.rectangle(target, (x0, y0), (x1, y0 + 28), (43, 46, 55), -1)
        cv2.rectangle(target, (x0, y0), (x1, y0 + 28), (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(target, "CLASSIFY NEW BOX", (x0 + 12, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1, cv2.LINE_AA)
        hint = "Copied. Press 1-5, or Esc" if copied else "Press 1-5, or Esc to cancel"
        hint_color = (80, 255, 130) if copied else (205, 205, 205)
        cv2.putText(target, hint, (x0 + 12, y0 + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.32, hint_color, 1, cv2.LINE_AA)

        classes = [("1", "blister"), ("2", "tear"), ("3", "delamination"), ("4", "particulate"), ("5", "hole")]
        left_x = x0 + 18
        # The second column starts around halfway across the card, but is clamped
        # so labels never run outside the card even in a narrow window.
        right_x = min(x0 + max(150, w // 2 + 18), x1 - 135)
        row_y = [y0 + 70, y0 + 93, y0 + 116]
        positions = [
            (left_x, row_y[0]),
            (right_x, row_y[0]),
            (left_x, row_y[1]),
            (right_x, row_y[1]),
            (left_x, row_y[2]),
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        for (num, c_name), (px, py) in zip(classes, positions):
            color = CLASS_COLORS[c_name]
            cv2.circle(target, (px, py - 4), 5, color, -1, cv2.LINE_AA)
            cv2.circle(target, (px, py - 4), 5, (245, 245, 245), 1, cv2.LINE_AA)
            cv2.putText(target, f"[{num}]", (px + 16, py), font, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

            label_x = px + 48
            max_label_right = x1 - 10
            # Slightly shrink only if the user's window is unusually narrow.
            scale = 0.35
            (tw, _th), _base = cv2.getTextSize(c_name, font, scale, 1)
            if label_x + tw > max_label_right:
                scale = max(0.28, scale * (max_label_right - label_x) / max(tw, 1))
            cv2.putText(target, c_name, (label_x, py), font, scale, (245, 245, 245), 1, cv2.LINE_AA)

    def _prompt_for_defect_type(self) -> Optional[str]:
        temp = self.canvas.copy()
        self._draw_type_prompt_card(temp, copied=False)
        cv2.imshow(WINDOW_NAME, temp)
        assigned_class = None
        self.is_waiting_for_key = True
        while True:
            key_press = cv2.waitKeyEx(0) & 0xFF
            if key_press in KEY_MAPPING:
                assigned_class = KEY_MAPPING[key_press]
                break
            if key_press in (ord("k"), ord("K")):
                # COPY remains usable even while the 1-5 classifier prompt is open.
                self.copy_current_view()
                temp = self.canvas.copy()
                self._draw_type_prompt_card(temp, copied=True)
                self._draw_copy_button(temp)
                cv2.imshow(WINDOW_NAME, temp)
                continue
            if key_press == 27:
                break
        self.is_waiting_for_key = False
        return assigned_class

    # ---- mouse/key handling ----------------------------------------------

    def handle_mouse(self, event, x, y, flags, param) -> None:
        if self.is_waiting_for_key:
            # Keep COPY usable while the 1-5 classifier prompt is open.
            if event == cv2.EVENT_LBUTTONDOWN and x >= self.display_width:
                cb = getattr(self, "copy_button_bounds", None)
                if cb is not None:
                    bx1, by1, bx2, by2 = cb
                    if bx1 <= x <= bx2 and by1 <= y <= by2:
                        self.copy_current_view()
            return
        if event == cv2.EVENT_RBUTTONDOWN:
            self._delete_box_at_display_point(x, y)
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if x >= self.display_width:
                cb = getattr(self, "copy_button_bounds", None)
                if cb is not None:
                    bx1, by1, bx2, by2 = cb
                    if bx1 <= x <= bx2 and by1 <= y <= by2:
                        self.copy_current_view()
                        self.redraw_canvas()
                        return
                for idx, bounds in enumerate(getattr(self, "cell_map_bounds", [])):
                    bx1, by1, bx2, by2 = bounds
                    if bx1 <= x <= bx2 and by1 <= y <= by2:
                        self.vertical_nav_anchor_x = None
                        self.load_cell_at_index(idx)
                        return
            else:
                self.drawing = True
                self.start_pt = (max(0, min(x, self.display_width - 1)), max(0, min(y, self.display_height - 1)))
                self.current_pt = self.start_pt
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            clamped_x = max(0, min(x, self.display_width - 1))
            clamped_y = max(0, min(y, self.display_height - 1))
            self.current_pt = (clamped_x, clamped_y)
            now = time.perf_counter()
            if now - self._last_drag_present < (1.0 / 60.0):
                return
            self._last_drag_present = now
            temp = self.canvas.copy()
            cv2.rectangle(temp, self.start_pt, self.current_pt, (0, 255, 255), 2, cv2.LINE_8)
            cv2.imshow(WINDOW_NAME, temp)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            clamped_x = max(0, min(x, self.display_width - 1))
            clamped_y = max(0, min(y, self.display_height - 1))
            self.current_pt = (clamped_x, clamped_y)
            w_disp = abs(self.current_pt[0] - self.start_pt[0])
            h_disp = abs(self.current_pt[1] - self.start_pt[1])
            if w_disp < 4 or h_disp < 4:
                self.redraw_canvas()
                return
            temp = self.canvas.copy()
            cv2.rectangle(temp, self.start_pt, self.current_pt, (0, 165, 255), 3)
            cv2.imshow(WINDOW_NAME, temp)
            assigned_class = self.default_defect_type if self.quick_label else self._prompt_for_defect_type()
            if assigned_class is not None:
                orig_x1, orig_y1, orig_x2, orig_y2 = self._display_to_native_rect()
                if orig_x2 > orig_x1 and orig_y2 > orig_y1:
                    entry = self.cell_files[self.current_idx]
                    self.push_undo()
                    record = self._record_from_native_bbox(
                        entry,
                        [orig_x1, orig_y1, orig_x2 - orig_x1, orig_y2 - orig_y1],
                        assigned_class,
                    )
                    self.annotations.setdefault(entry["filename"], []).append(record)
                    self._mark_annotations_dirty(entry["filename"])
            self.redraw_canvas()


    def _entry_row_col(self, entry: dict) -> tuple[Optional[int], Optional[int]]:
        try:
            row = int(entry.get("row", entry.get("cell_data", {}).get("row", 0)))
            col = int(entry.get("col", entry.get("cell_data", {}).get("col", 0)))
            return row, col
        except Exception:
            return None, None

    def _entry_gds_center(self, entry: dict) -> Optional[tuple[float, float]]:
        bbox = entry.get("gds_bbox")
        if bbox is None or len(bbox) < 4:
            return None
        try:
            min_x, min_y, max_x, max_y = [float(v) for v in bbox[:4]]
            return (0.5 * (min_x + max_x), 0.5 * (min_y + max_y))
        except Exception:
            return None

    def _median_cell_pitch_x(self) -> float:
        centers = []
        for entry in self.cell_files:
            c = self._entry_gds_center(entry)
            if c is not None:
                centers.append(c[0])
        centers = sorted(set(round(x, 3) for x in centers))
        diffs = [abs(b - a) for a, b in zip(centers, centers[1:]) if abs(b - a) > 1e-6]
        if not diffs:
            return float("inf")
        return float(np.median(diffs))

    def _jump_by_grid_row(self, direction: int) -> None:
        """Jump to the nearest cell in the adjacent row using a fixed x anchor.

        Rows near the wafer edge often have fewer cells. If the exact same
        row/column slot does not exist, still move one row up/down; choose the
        cell whose physical GDS x-center is closest to the original Up/Down
        anchor. Holding that anchor across repeated Up/Down presses prevents
        the navigation from stair-stepping diagonally.
        """
        if not self.cell_files:
            return
        cur = self.cell_files[self.current_idx]
        cur_row, _cur_col = self._entry_row_col(cur)
        cur_center = self._entry_gds_center(cur)

        if cur_row is None or cur_center is None:
            fallback = self.current_idx + (1 if direction > 0 else -1)
            if 0 <= fallback < len(self.cell_files):
                self.load_cell_at_index(fallback)
            return

        if self.vertical_nav_anchor_x is None:
            self.vertical_nav_anchor_x = float(cur_center[0])
        anchor_x = float(self.vertical_nav_anchor_x)

        rows = sorted(set(r for r, _c in (self._entry_row_col(e) for e in self.cell_files) if r is not None))
        if direction < 0:
            target_rows = [r for r in rows if r < cur_row]
            target_row = max(target_rows) if target_rows else None
        else:
            target_rows = [r for r in rows if r > cur_row]
            target_row = min(target_rows) if target_rows else None

        if target_row is None:
            self._status("No row in that direction")
            return

        target_candidates = []
        for idx, entry in enumerate(self.cell_files):
            row, _col = self._entry_row_col(entry)
            if row != target_row:
                continue
            center = self._entry_gds_center(entry)
            if center is None:
                continue
            dx = abs(center[0] - anchor_x)
            target_candidates.append((dx, idx))

        if not target_candidates:
            self._status("No cell in adjacent row")
            return

        _dx, best_idx = min(target_candidates)
        self.load_cell_at_index(best_idx)

    def _handle_key(self, key: int) -> bool:
        if key < 0:
            return False
        ctrl, shift = _modifiers_down()
        key8 = key & 0xFF

        if (ctrl and _is_z_key(key)) or key8 == 26:
            if shift:
                self.redo()
            else:
                self.undo()
            return False
        if (ctrl and _is_y_key(key)) or key8 == 25:
            self.redo()
            return False

        if key in RIGHT_ARROW_CODES or key8 == 32 or key8 in (ord("n"), ord("N")):
            self.vertical_nav_anchor_x = None
            if self.current_idx < len(self.cell_files) - 1:
                self.load_cell_at_index(self.current_idx + 1)
            return False
        if key in LEFT_ARROW_CODES or key8 in (ord("p"), ord("P")):
            self.vertical_nav_anchor_x = None
            if self.current_idx > 0:
                self.load_cell_at_index(self.current_idx - 1)
            return False
        if key in UP_ARROW_CODES:
            self._jump_by_grid_row(-1)
            return False
        if key in DOWN_ARROW_CODES:
            self._jump_by_grid_row(1)
            return False
        if key8 in (ord("x"), ord("X")):
            filename = self.cell_files[self.current_idx]["filename"]
            self.push_undo()
            if filename in self.exclusions:
                self.exclusions.remove(filename)
            else:
                self.exclusions.add(filename)
            self._mark_exclusions_dirty()
            self.redraw_canvas()
            return False
        if key8 in (ord("c"), ord("C")):
            filename = self.cell_files[self.current_idx]["filename"]
            self.push_undo()
            self.annotations[filename] = []
            self._mark_annotations_dirty(filename)
            self.redraw_canvas()
            return False
        if key8 in (ord("l"), ord("L")):
            self.show_annotation_labels = not self.show_annotation_labels
            self.redraw_canvas()
            self._status(f"Annotation labels {'on' if self.show_annotation_labels else 'off'}")
            return False
        if key8 in (ord("k"), ord("K")):
            self.copy_current_view()
            self.redraw_canvas()
            return False
        if key8 == 27 or key8 in (ord("q"), ord("Q")):
            self.save_annotations_to_file(force=True)
            self.save_exclusions_file(force=True)
            self.save_review_state(force=True)
            return True
        return False

    def run(self) -> None:
        if not self.cell_files:
            raise FileNotFoundError("No cell images available for defect review UI")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.handle_mouse)
        try:
            self.load_cell_at_index(self.current_idx)
            while True:
                if (not self.drawing and not self.is_waiting_for_key
                        and bool(getattr(self, "copy_feedback_redraw_pending", False))
                        and time.time() >= float(getattr(self, "copy_feedback_until", 0.0))):
                    self.copy_feedback_redraw_pending = False
                    self.redraw_canvas()
                self._autosave_if_due()
                key = cv2.waitKeyEx(16)
                if self._handle_key(key):
                    break
        finally:
            self.save_annotations_to_file(force=True)
            self.save_exclusions_file(force=True)
            self.save_review_state(force=True)
            self._shutdown_prefetch()
            cv2.destroyAllWindows()


class DeviceDefectMapperTool(_BaseDefectReviewUI):
    """Manual labeling UI used by wafer_alignment_and_extraction.py."""

    def __init__(self, wafer_id, cells, out_dir, transformer, gds_R, config, shave: int = 10, pad: int = 200):
        super().__init__()
        self.verbose_navigation = True
        self.wafer_id = wafer_id
        self.cells = cells
        self.out_dir = Path(out_dir)
        self.transformer = transformer
        self.gds_R = gds_R
        self.config = config
        self.shave = shave
        self.pad = pad
        self.output_json_path = Path(f"{wafer_id}_device_defects.json")
        self.output_stitch_path = Path(f"{wafer_id}_stitched_devices.jpg")
        self.exclusions_path = Path("manual_exclusions.json")

        self.cell_files = []
        for cell in self.cells:
            row = int(cell["row"])
            col = int(cell["col"])
            stem = f"{wafer_id}_cell_{row}-{col}"
            analysis_path = self.out_dir / "analysis_png" / f"{stem}.png"
            jpg_path = self.out_dir / f"{stem}.jpg"
            filepath = analysis_path if analysis_path.exists() else jpg_path
            if not filepath.exists():
                continue
            preview_path = self.out_dir / "previews" / f"{stem}_preview.jpg"
            meta_path = self.out_dir / "metadata" / f"{stem}.json"
            meta = _load_json(meta_path, {}) if meta_path.exists() else {}
            gds_bbox = meta.get("gds_bbox_um") or cell.get("bbox")
            self.cell_files.append(
                {
                    "filename": Path(meta.get("legacy_jpg", f"{stem}.jpg")).name,
                    "filepath": filepath,
                    "preview_path": preview_path,
                    "cell_data": cell,
                    "metadata": meta,
                    "metadata_path": meta_path if meta_path.exists() else None,
                    "gds_bbox": gds_bbox,
                }
            )
        if not self.cell_files:
            raise FileNotFoundError(f"No cell crops discovered in '{self.out_dir}'. Ensure extraction succeeded first.")
        self.cell_files.sort(key=lambda x: (int(x["cell_data"]["row"]), int(x["cell_data"]["col"])))
        self.annotations = self.load_existing_annotations()
        self.exclusions = self.load_exclusions_file()

    def _record_from_native_bbox(self, entry: dict, bbox_px: list[int], assigned_class: str) -> dict:
        x, y, w, h = [float(v) for v in bbox_px]
        cell_data = entry.get("cell_data")
        meta = entry.get("metadata") or {}
        if meta:
            canvas_to_gds = _metadata_canvas_to_gds_affine(meta)
            corners_px = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            corners_gds = [_round_pair(crop_px_to_gds(px, py, meta, canvas_to_gds)) for px, py in corners_px]
            center_x_um, center_y_um = crop_px_to_gds(x + w / 2.0, y + h / 2.0, meta, canvas_to_gds)
        else:
            px_cx = x + w / 2.0
            px_cy = y + h / 2.0
            center_x_um, center_y_um = self.transformer.crop_pixel_to_gds(px_cx, px_cy, cell_data, shave=self.shave, pad=self.pad)
            corners_px = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            corners_gds = [self.transformer.crop_pixel_to_gds(px, py, cell_data, shave=self.shave, pad=self.pad) for px, py in corners_px]
            corners_gds = [_round_pair(c) for c in corners_gds]
        xs = [c[0] for c in corners_gds]
        ys = [c[1] for c in corners_gds]
        return {
            "type": assigned_class,
            "box_px": [int(round(x)), int(round(y)), int(round(w)), int(round(h))],
            "center_x_um": round(float(center_x_um), 3),
            "center_y_um": round(float(center_y_um), 3),
            "width_um": round(float(max(xs) - min(xs)), 3),
            "height_um": round(float(max(ys) - min(ys)), 3),
            "corners_gds": corners_gds,
            "source": "manual_review",
        }

    def stitch_and_save_wafer_layout(self) -> None:
        # Best-effort legacy overview for manual stage.
        try:
            print("\nCompiling GDS-aligned physical wafer overview composite...")
            out_size = int(self.config.get("output_image_size", 4000))
            composite_canvas = np.zeros((out_size, out_size, 3), dtype=np.uint8)
            half = out_size / 2.0
            if self.gds_R:
                scale = (0.925 * half) / self.gds_R
                cv2.circle(composite_canvas, (int(half), int(half)), int(self.gds_R * scale), (60, 60, 60), 2, lineType=cv2.LINE_AA)
            for entry in self.cell_files:
                cell_img = _read_bgr(entry["filepath"])
                if cell_img is None:
                    continue
                for box in self.annotations.get(entry["filename"], []):
                    x, y, w, h = [int(v) for v in box.get("box_px", [0, 0, 0, 0])]
                    color = CLASS_COLORS.get(box.get("type", "auto_defect"), (255, 255, 255))
                    p_x1, p_y1 = max(0, x - 40), max(0, y - 40)
                    p_x2, p_y2 = min(cell_img.shape[1], x + w + 40), min(cell_img.shape[0], y + h + 40)
                    cv2.rectangle(cell_img, (p_x1, p_y1), (p_x2, p_y2), color, 16)
                    cv2.putText(cell_img, str(box.get("type", "defect")).upper(), (p_x1, max(30, p_y1 - 15)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 5, cv2.LINE_AA)
                cell = entry.get("cell_data") or {}
                min_x, min_y, max_x, max_y = cell.get("bbox", entry.get("gds_bbox", [0, 0, 1, 1]))
                pts_img = [self.transformer.transform_gds_to_target_img(gx, gy, out_size) for gx, gy in [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]]
                pts_img = np.asarray(pts_img)
                tx_min, ty_min = np.min(pts_img, axis=0)
                tx_max, ty_max = np.max(pts_img, axis=0)
                pt_x1, pt_y1 = int(round(tx_min)), int(round(ty_min))
                pt_x2, pt_y2 = int(round(tx_max)), int(round(ty_max))
                cell_w, cell_h = pt_x2 - pt_x1, pt_y2 - pt_y1
                if cell_w > 0 and cell_h > 0 and 0 <= pt_x1 < out_size and 0 <= pt_y1 < out_size:
                    pt_x2 = min(out_size, pt_x2)
                    pt_y2 = min(out_size, pt_y2)
                    composite_canvas[pt_y1:pt_y2, pt_x1:pt_x2] = cv2.resize(cell_img, (pt_x2 - pt_x1, pt_y2 - pt_y1), interpolation=cv2.INTER_AREA)
            _imwrite(self.output_stitch_path, composite_canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"Overview composite stitch saved: {self.output_stitch_path}")
        except Exception as exc:
            print(f"[DefectMapper] overview stitch skipped: {exc}")

    def run(self) -> None:
        super().run()
        self.stitch_and_save_wafer_layout()


class AutoLabelReviewTool(_BaseDefectReviewUI):
    """Review UI for boxes generated by defect_detector.py."""

    def __init__(
        self,
        image_dir: str | Path,
        annotations_json: str | Path,
        metadata_dir: str | Path,
        preview_dir: str | Path | None = None,
        wafer_id: str = "",
        exclusions_path: str | Path = "manual_exclusions.json",
        resume_state_path: str | Path | None = None,
        quick_label: bool = False,
        default_defect_type: str = "defect",
        image_cache_size: int = 12,
        autosave_seconds: float = 4.0,
        show_annotation_labels: Optional[bool] = None,
    ):
        super().__init__(
            quick_label=quick_label,
            default_defect_type=default_defect_type,
            image_cache_size=image_cache_size,
            autosave_seconds=autosave_seconds,
            show_annotation_labels=show_annotation_labels,
        )
        self.image_dir = Path(image_dir)
        self.metadata_dir = Path(metadata_dir)
        self.preview_dir = Path(preview_dir) if preview_dir else self.image_dir.parent / "previews"
        self.output_json_path = Path(annotations_json)
        self.exclusions_path = Path(exclusions_path)
        self.resume_state_path = (
            Path(resume_state_path)
            if resume_state_path
            else Path(str(self.output_json_path) + ".review_state.json")
        )
        self.wafer_id = wafer_id or self._infer_wafer_id()
        self.transformer = None
        self.shave = 0
        self.pad = 0

        self.annotations = self.load_existing_annotations()
        self.exclusions = self.load_exclusions_file()
        self.cell_files = self._discover_cells()
        if not self.cell_files:
            raise FileNotFoundError(f"No reviewable cell images discovered under {self.image_dir}")

        # Make sure every discovered cell has a JSON key. This also makes cells
        # with zero detected defects visible in the side map as pending/unannotated.
        for entry in self.cell_files:
            self.annotations.setdefault(entry["filename"], [])
        self.restore_review_state()
        self._annotations_dirty = True
        self.save_annotations_to_file(force=True)
        self._review_state_dirty = True
        self.save_review_state(force=True)

    def _infer_wafer_id(self) -> str:
        m = re.match(r"(.+)_cell_\d+-\d+$", self.image_dir.name)
        if m:
            return m.group(1)
        return ""

    def _discover_cells(self) -> list[dict]:
        files = []
        if self.image_dir.is_file():
            candidates = [self.image_dir]
        else:
            candidates = sorted(p for p in self.image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        for img_path in candidates:
            low = img_path.name.lower()
            if "preview" in low or "overlay" in low or "mask" in low:
                continue
            stem = img_path.stem
            meta_path = self.metadata_dir / f"{stem}.json"
            meta = _load_json(meta_path, {}) if meta_path.exists() else {}
            legacy_name = Path(meta.get("legacy_jpg", f"{stem}.jpg")).name
            m = re.search(r"_cell_(\d+)-(\d+)$", stem)
            row = int(m.group(1)) if m else 0
            col = int(m.group(2)) if m else 0
            preview_path = self.preview_dir / f"{stem}_preview.jpg"
            gds_bbox = meta.get("gds_bbox_um")
            files.append(
                {
                    "filename": legacy_name,
                    "filepath": img_path,
                    "preview_path": preview_path,
                    "metadata": meta,
                    "metadata_path": meta_path if meta_path.exists() else None,
                    "gds_bbox": gds_bbox,
                    "row": row,
                    "col": col,
                }
            )
        files.sort(key=lambda e: (int(e.get("row", 0)), int(e.get("col", 0)), str(e["filepath"])))
        return files

    def _record_from_native_bbox(self, entry: dict, bbox_px: list[int], assigned_class: str) -> dict:
        meta = entry.get("metadata") or {}
        if not meta:
            raise RuntimeError(f"Cannot add box for {entry['filename']}: missing metadata needed for GDS mapping")
        x, y, w, h = [float(v) for v in bbox_px]
        canvas_to_gds = _metadata_canvas_to_gds_affine(meta)
        corners_px = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        corners_gds = [_round_pair(crop_px_to_gds(px, py, meta, canvas_to_gds)) for px, py in corners_px]
        cx_gds, cy_gds = crop_px_to_gds(x + w / 2.0, y + h / 2.0, meta, canvas_to_gds)
        xs = [c[0] for c in corners_gds]
        ys = [c[1] for c in corners_gds]
        return {
            "type": assigned_class,
            "box_px": [int(round(x)), int(round(y)), int(round(w)), int(round(h))],
            "center_x_um": round(float(cx_gds), 3),
            "center_y_um": round(float(cy_gds), 3),
            "width_um": round(float(max(xs) - min(xs)), 3),
            "height_um": round(float(max(ys) - min(ys)), 3),
            "corners_gds": corners_gds,
            "source": "manual_review",
        }
