"""
Algorithmic cell defect detector for h2p_device_viewer crops.

This is not a neural net. It treats the device as a regular vertical-line cell
canvas, models the normal repeating structure, subtracts that model, then finds
localized residual clusters.

The detector is intentionally conservative about the normal vertical cell lines:
normal lines are captured by a column-wise template or a multi-cell median
template before thresholding, so ordinary device stripes should mostly vanish in
residual space.

Typical h2p detection + labeling use:
    python defect_detector.py --review-ui

This defaults to:
    --input extracted_cells/analysis_png
    --output extracted_cells/algo_defects.json
    --preview-dir extracted_cells/algo_previews
    --metadata-dir extracted_cells/metadata
    --gds-output-json Wafer_A_device_defects.json

By default it cleans the output JSONs and preview folder before each run.
Use --no-clean to keep existing outputs.

Optional template workflow:
    python defect_detector.py --input extracted_cells --save-template normal_template.npz
    python defect_detector.py --input extracted_cells --template normal_template.npz --output algo_defects.json

Inputs:
    .png/.jpg/.jpeg/.tif/.tiff cell crops

Optional seam mask naming:
    If --seam-mask-dir is provided, the detector looks for:
        <cell_stem>_seam_mask.png
    and uses it to suppress seam-shaped low-frequency artifacts without blindly
    discarding real localized defects that happen to lie on a seam.

Output JSON:
    {
      "images": [
        {
          "image": "Wafer_A_cell_4-10.png",
          "defects": [
            {
              "bbox_px": [x, y, w, h],
              "polygon_px": [[x0,y0], ...],
              "area_px": 1234,
              "score": 8.7,
              "reason": "kept"
            }
          ]
        }
      ]
    }
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
DETECTOR_VERSION = "periodic-stripe-guard-v18-fast-review-2026-07-14"

# v17 focuses on the annotation failures from v16:
# - keep thin connected protrusions as separate tight polygons instead of trimming them;
# - selectively fill defect interiors when image evidence supports the enclosed area;
# - recover compact side-border defects without enabling broad frame growth;
# - keep GDS-derived suppression opt-in and disabled by default.



# ---------------------------------------------------------------------------
# Runtime / progress helpers
# ---------------------------------------------------------------------------

class ProgressBar:
    """Dependency-free progress bar that plays nicely with PowerShell."""

    def __init__(self, label: str, total: int, width: int = 28):
        self.label = str(label)
        self.total = max(1, int(total))
        self.width = max(8, int(width))
        self.start = time.time()
        self.last_len = 0

    def update(self, current: int, extra: str = "") -> None:
        current = max(0, min(int(current), self.total))
        frac = current / self.total
        filled = int(round(frac * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self.start
        eta = elapsed * (self.total - current) / max(current, 1) if current else 0.0
        msg = f"\r[{self.label}] |{bar}| {current}/{self.total} {frac*100:5.1f}% elapsed {elapsed/60:4.1f}m"
        if current:
            msg += f" ETA {eta/60:4.1f}m"
        if extra:
            msg += f" | {extra}"
        pad = max(0, self.last_len - len(msg))
        sys.stdout.write(msg + " " * pad)
        sys.stdout.flush()
        self.last_len = len(msg)

    def done(self, extra: str = "") -> None:
        self.update(self.total, extra=extra)
        sys.stdout.write("\n")
        sys.stdout.flush()


def configure_runtime(cv_threads: int = 1, low_priority: bool = True) -> None:
    """Keep the detector responsive on a normal desktop/laptop.

    OpenCV can otherwise use many CPU threads and make the machine feel frozen.
    The default is intentionally polite: one OpenCV thread and below-normal
    process priority where the OS supports it.
    """
    threads = max(1, int(cv_threads or 1))
    try:
        cv2.setNumThreads(threads)
    except Exception:
        pass
    try:
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(threads))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(threads))
    except Exception:
        pass

    if not low_priority:
        return
    try:
        if os.name == "nt":
            BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(),
                BELOW_NORMAL_PRIORITY_CLASS,
            )
        else:
            os.nice(10)
    except Exception:
        # Priority changes are best-effort only.
        pass




def restore_normal_priority() -> None:
    """Best-effort restore to normal process priority before opening the review UI."""
    try:
        if os.name == "nt":
            NORMAL_PRIORITY_CLASS = 0x00000020
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(),
                NORMAL_PRIORITY_CLASS,
            )
        else:
            # Do not try to undo os.nice() without privileges on POSIX.
            pass
    except Exception:
        pass


def resize_bgr_max_width(img: np.ndarray, max_width: int = 0) -> np.ndarray:
    max_width = int(max_width or 0)
    if max_width <= 0 or img.shape[1] <= max_width:
        return img
    scale = max_width / float(img.shape[1])
    new_w = int(round(img.shape[1] * scale))
    new_h = max(1, int(round(img.shape[0] * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def read_bgr_resized(path: Path | str, max_width: int = 0) -> np.ndarray:
    return resize_bgr_max_width(read_bgr(path), max_width=max_width)

# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def read_bgr(path: Path | str) -> np.ndarray:
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    return img


def read_gray(path: Path | str) -> np.ndarray:
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    return img


def imwrite(path: Path | str, img: np.ndarray, params: Optional[list[int]] = None) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def list_images(path: Path | str) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    files = []
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            # Skip previews and masks by default; they can accidentally live under input.
            low = p.name.lower()
            if "preview" in low or "mask" in low or "overlay" in low:
                continue
            files.append(p)
    return files


# ---------------------------------------------------------------------------
# Robust statistics
# ---------------------------------------------------------------------------

def robust_mad(x: np.ndarray, mask: Optional[np.ndarray] = None, floor: float = 1.0) -> float:
    if mask is not None:
        vals = x[mask]
    else:
        vals = x.reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float(floor)
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) * 1.4826
    if not np.isfinite(mad) or mad < floor:
        mad = floor
    return float(mad)


def robust_center(x: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    vals = x[mask] if mask is not None else x.reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    return float(np.median(vals))


def robust_z(x: np.ndarray, mask: Optional[np.ndarray] = None, floor: float = 1.0) -> np.ndarray:
    c = robust_center(x, mask)
    s = robust_mad(x, mask, floor=floor)
    return (x.astype(np.float32) - c) / s


def odd_kernel(k: int, minimum: int = 3) -> int:
    k = int(max(minimum, k))
    if k % 2 == 0:
        k += 1
    return k


# ---------------------------------------------------------------------------
# Masking and normal models
# ---------------------------------------------------------------------------

def make_interior_mask(
    img_bgr: np.ndarray,
    border_px: int = 14,
    border_frac: float = 0.015,
    auto_yellow_border: bool = True,
) -> np.ndarray:
    """Mask out the crop border/frame so it doesn't become a defect."""
    h, w = img_bgr.shape[:2]
    m = np.ones((h, w), dtype=bool)
    pad = max(int(border_px), int(round(min(h, w) * float(border_frac))))
    if pad > 0:
        m[:pad, :] = False
        m[-pad:, :] = False
        m[:, :pad] = False
        m[:, -pad:] = False

    if auto_yellow_border:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        # Yellow/gold frame is high saturation, high value, hue around 18..42 in OpenCV's 0..179 scale.
        yellowish = (H >= 14) & (H <= 45) & (S > 45) & (V > 110)
        # Only suppress connected yellowish regions that touch the image border.
        yy = yellowish.astype(np.uint8) * 255
        n, labels, stats, _ = cv2.connectedComponentsWithStats(yy, connectivity=8)
        border_touch = np.zeros_like(m, dtype=bool)
        for i in range(1, n):
            x, y, ww, hh, area = stats[i]
            touches = x <= pad or y <= pad or (x + ww) >= (w - pad) or (y + hh) >= (h - pad)
            touches = x <= pad or y <= pad or (x + ww) >= (w - pad) or (y + hh) >= (h - pad)
            if touches and area > max(64, 0.0005 * h * w):
                border_touch |= labels == i
        if np.any(border_touch):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            border_touch = cv2.dilate(border_touch.astype(np.uint8), kernel, iterations=2).astype(bool)
            m &= ~border_touch

    return m


def smooth_vector(v: np.ndarray, sigma: float) -> np.ndarray:
    v2 = v.astype(np.float32)[None, :]
    out = cv2.GaussianBlur(v2, (0, 0), sigmaX=float(max(0.01, sigma)), sigmaY=0.01)
    return out.reshape(-1).astype(np.float32)


def smooth_vector_y(v: np.ndarray, sigma: float) -> np.ndarray:
    v2 = v.astype(np.float32)[:, None]
    out = cv2.GaussianBlur(v2, (0, 0), sigmaX=0.01, sigmaY=float(max(0.01, sigma)))
    return out.reshape(-1).astype(np.float32)


def column_row_expected(channel: np.ndarray, interior_mask: np.ndarray, col_smooth: float = 0.0, row_smooth: float = 25.0) -> np.ndarray:
    """Expected channel from column stripe pattern plus slow row background.

    Normal vertical device lines are long in y, so the per-column robust median
    captures them. Local defects do not dominate the y-median and survive in the
    residual.
    """
    ch = channel.astype(np.float32)
    h, w = ch.shape

    col_med = np.zeros(w, dtype=np.float32)
    global_med = robust_center(ch, interior_mask)
    for x in range(w):
        mask_x = interior_mask[:, x]
        if np.count_nonzero(mask_x) > max(8, h * 0.1):
            col_med[x] = robust_center(ch[:, x], mask_x)
        else:
            col_med[x] = global_med

    # Preserve real line periodicity. Do not over-smooth x unless requested.
    if col_smooth > 0:
        col_med = smooth_vector(col_med, sigma=col_smooth)

    no_col = ch - col_med[None, :] + global_med
    row_med = np.zeros(h, dtype=np.float32)
    for y in range(h):
        mask_y = interior_mask[y, :]
        if np.count_nonzero(mask_y) > max(8, w * 0.1):
            row_med[y] = robust_center(no_col[y, :], mask_y)
        else:
            row_med[y] = global_med
    if row_smooth > 0:
        row_med = smooth_vector_y(row_med, sigma=row_smooth)

    expected = col_med[None, :] + row_med[:, None] - global_med
    return expected.astype(np.float32)


def per_cell_normal_residuals(img_bgr: np.ndarray, interior_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return residual LAB image and expected LAB image using per-cell model."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    expected = np.empty_like(lab, dtype=np.float32)

    # L preserves the vertical stripe pattern strongly. A/B are smoother but catch brown/blue defect colors.
    expected[:, :, 0] = column_row_expected(lab[:, :, 0], interior_mask, col_smooth=0.0, row_smooth=max(12, h * 0.025))
    expected[:, :, 1] = column_row_expected(lab[:, :, 1], interior_mask, col_smooth=2.0, row_smooth=max(18, h * 0.035))
    expected[:, :, 2] = column_row_expected(lab[:, :, 2], interior_mask, col_smooth=2.0, row_smooth=max(18, h * 0.035))

    residual = lab - expected
    return residual.astype(np.float32), expected.astype(np.float32)


@dataclass
class MedianTemplate:
    template_lab: np.ndarray
    width: int
    height: int
    source_count: int
    mad_lab: Optional[np.ndarray] = None
    alignment_radius_px: int = 6


def _highpass_l_for_alignment(lab_or_bgr: np.ndarray) -> np.ndarray:
    """Return a normalized high-pass L image for small translation alignment."""
    if lab_or_bgr.ndim == 3 and lab_or_bgr.shape[2] == 3:
        if lab_or_bgr.dtype == np.float32 or lab_or_bgr.dtype == np.float64:
            L = lab_or_bgr[:, :, 0].astype(np.float32)
        else:
            L = cv2.cvtColor(lab_or_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    else:
        L = lab_or_bgr.astype(np.float32)
    slow = cv2.GaussianBlur(L, (0, 0), sigmaX=9.0, sigmaY=9.0)
    hp = L - slow
    hp -= float(np.mean(hp))
    sd = float(np.std(hp))
    if sd > 1e-6:
        hp /= sd
    return hp.astype(np.float32)


def _estimate_small_translation(reference_lab: np.ndarray, image_lab: np.ndarray, max_shift: int = 6) -> tuple[float, float]:
    """Estimate a small translation while avoiding pitch-period ambiguity.

    Phase correlation is fast, but the repeating fingers create equivalent peaks
    one pitch apart.  We therefore clamp to a small window and verify/refine the
    result with a robust central-crop absolute-error search.
    """
    ref = _highpass_l_for_alignment(reference_lab)
    mov = _highpass_l_for_alignment(image_lab)
    h, w = ref.shape
    try:
        window = cv2.createHanningWindow((w, h), cv2.CV_32F)
        (dx0, dy0), _ = cv2.phaseCorrelate(ref, mov, window)
    except Exception:
        dx0 = dy0 = 0.0
    if not np.isfinite(dx0) or abs(dx0) > max_shift:
        dx0 = 0.0
    if not np.isfinite(dy0) or abs(dy0) > max_shift:
        dy0 = 0.0

    cx = int(round(dx0))
    cy = int(round(dy0))
    best = (float("inf"), 0, 0)
    pad = max(max_shift + 3, int(round(0.04 * min(h, w))))
    ys = slice(pad, max(pad + 1, h - pad))
    xs = slice(pad, max(pad + 1, w - pad))
    for dy in range(max(-max_shift, cy - 2), min(max_shift, cy + 2) + 1):
        for dx in range(max(-max_shift, cx - 2), min(max_shift, cx + 2) + 1):
            M = np.float32([[1, 0, -dx], [0, 1, -dy]])
            warped = cv2.warpAffine(mov, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            d = np.abs(ref[ys, xs] - warped[ys, xs])
            err = float(np.median(d))
            if err < best[0]:
                best = (err, dx, dy)
    return float(best[1]), float(best[2])


def _warp_lab_translation(lab: np.ndarray, dx: float, dy: float, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    h, w = lab.shape[:2]
    M = np.float32([[1, 0, -float(dx)], [0, 1, -float(dy)]])
    return cv2.warpAffine(
        lab.astype(np.float32), M, (w, h), flags=interpolation,
        borderMode=cv2.BORDER_REFLECT,
    ).astype(np.float32)


def build_median_template(
    image_paths: Iterable[Path],
    max_images: int = 80,
    border_px: int = 14,
    border_frac: float = 0.015,
    max_width: int = 1400,
    show_progress: bool = True,
) -> MedianTemplate:
    """Build an aligned robust normal template and per-pixel variation map.

    The median rejects defects that occur in only a minority of cells.  The MAD
    map records normal location-specific variation, so a faint speck can be
    significant at a stable device location without lowering a global threshold.
    """
    paths = list(image_paths)
    if not paths:
        raise ValueError("No images provided for template build")
    max_images = max(3, int(max_images))
    if len(paths) > max_images:
        idx = np.linspace(0, len(paths) - 1, max_images).round().astype(int)
        paths = [paths[int(i)] for i in idx]

    first = read_bgr_resized(paths[0], max_width=max_width)
    h0, w0 = first.shape[:2]
    reference_lab = cv2.cvtColor(first, cv2.COLOR_BGR2LAB).astype(np.float32)
    labs_u8: list[np.ndarray] = []
    bar = ProgressBar("Template", len(paths)) if show_progress else None
    for idx, pth in enumerate(paths, start=1):
        try:
            img = read_bgr_resized(pth, max_width=max_width)
            if img.shape[:2] != (h0, w0):
                img = cv2.resize(img, (w0, h0), interpolation=cv2.INTER_AREA if img.shape[1] > w0 else cv2.INTER_LINEAR)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
            if labs_u8:
                dx, dy = _estimate_small_translation(reference_lab, lab, max_shift=6)
                lab = _warp_lab_translation(lab, dx, dy)
            labs_u8.append(np.clip(lab, 0, 255).astype(np.uint8))
            # Update a gentle running reference so the first cell is not special.
            if len(labs_u8) in (5, 12, 24):
                reference_lab = np.median(np.stack(labs_u8, axis=0), axis=0).astype(np.float32)
        except Exception:
            pass
        if bar:
            bar.update(idx, extra=pth.name)
    if bar:
        bar.done(extra=f"used {len(labs_u8)}")
    if not labs_u8:
        raise RuntimeError("Could not read any images for template build")

    stack = np.stack(labs_u8, axis=0)
    templ = np.median(stack, axis=0).astype(np.float32)
    abs_dev = np.abs(stack.astype(np.float32) - templ[None, ...])
    mad = 1.4826 * np.median(abs_dev, axis=0).astype(np.float32)
    # Floors prevent JPEG quantization and perfectly stable pixels from exploding.
    mad[:, :, 0] = np.maximum(mad[:, :, 0], 1.8)
    mad[:, :, 1] = np.maximum(mad[:, :, 1], 0.9)
    mad[:, :, 2] = np.maximum(mad[:, :, 2], 0.9)
    return MedianTemplate(
        template_lab=templ,
        mad_lab=mad,
        width=w0,
        height=h0,
        source_count=len(labs_u8),
        alignment_radius_px=6,
    )


def save_template(t: MedianTemplate, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "template_lab": t.template_lab.astype(np.float32),
        "width": np.array([t.width], dtype=np.int32),
        "height": np.array([t.height], dtype=np.int32),
        "source_count": np.array([t.source_count], dtype=np.int32),
        "alignment_radius_px": np.array([int(t.alignment_radius_px)], dtype=np.int32),
    }
    if t.mad_lab is not None:
        payload["mad_lab"] = t.mad_lab.astype(np.float32)
    np.savez_compressed(str(path), **payload)


def load_template(path: Path | str) -> MedianTemplate:
    data = np.load(str(path))
    mad = data["mad_lab"].astype(np.float32) if "mad_lab" in data.files else None
    radius = int(data["alignment_radius_px"][0]) if "alignment_radius_px" in data.files else 6
    return MedianTemplate(
        template_lab=data["template_lab"].astype(np.float32),
        mad_lab=mad,
        width=int(data["width"][0]),
        height=int(data["height"][0]),
        source_count=int(data["source_count"][0]),
        alignment_radius_px=radius,
    )


def template_residuals(
    img_bgr: np.ndarray,
    template: MedianTemplate,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Return aligned raw residual, expected LAB, and optional standardized residual."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    templ = template.template_lab
    mad = template.mad_lab
    if templ.shape[:2] != (h, w):
        templ = cv2.resize(templ, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        if mad is not None:
            mad = cv2.resize(mad, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    dx, dy = _estimate_small_translation(templ, lab, max_shift=max(1, int(template.alignment_radius_px)))
    templ = _warp_lab_translation(templ, -dx, -dy)
    if mad is not None:
        mad = _warp_lab_translation(mad, -dx, -dy)
        mad[:, :, 0] = np.maximum(mad[:, :, 0], 1.8)
        mad[:, :, 1] = np.maximum(mad[:, :, 1], 0.9)
        mad[:, :, 2] = np.maximum(mad[:, :, 2], 0.9)
    residual = (lab - templ).astype(np.float32)
    z = (residual / mad).astype(np.float32) if mad is not None else None
    return residual, templ.astype(np.float32), z


# ---------------------------------------------------------------------------
# Scoring and connected components
# ---------------------------------------------------------------------------

@dataclass
class DetectorParams:
    threshold: float = 8.8
    min_area: int = 18
    max_area_frac: float = 0.22
    min_score: float = 8.8
    border_px: int = 42
    border_frac: float = 0.015
    open_radius: int = 1
    close_radius: int = 5
    seam_suppress: bool = True
    seam_overlap_reject_frac: float = 0.55
    seam_like_aspect: float = 7.0
    seam_low_score_margin: float = 1.25
    tiny_component_area: int = 0
    cluster_dilate_px: int = 11


@dataclass
class DefectComponent:
    bbox_px: list[int]
    polygon_px: list[list[int]]
    area_px: int
    score: float
    mean_score: float
    seam_overlap_frac: float
    reason: str = "kept"


def estimate_vertical_pitch_px(img_bgr: np.ndarray, interior_mask: np.ndarray) -> float:
    """Estimate the normal vertical-finger pitch in pixels.

    The detector uses this only to suppress stripe-frequency residuals.  If the
    estimate fails, a conservative width-based fallback is used.
    """
    h, w = img_bgr.shape[:2]
    fallback = float(np.clip(w / 115.0, 5.0, 18.0))
    try:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]
        col = np.zeros(w, dtype=np.float32)
        valid = np.zeros(w, dtype=bool)
        for x in range(w):
            m = interior_mask[:, x]
            if np.count_nonzero(m) > max(12, h * 0.20):
                col[x] = robust_center(L[:, x], m)
                valid[x] = True
        if np.count_nonzero(valid) < max(40, w * 0.25):
            return fallback
        # Fill invalid columns and remove slow illumination trend.
        xs = np.arange(w)
        col[~valid] = np.interp(xs[~valid], xs[valid], col[valid])
        slow = cv2.GaussianBlur(col[None, :], (0, 0), sigmaX=max(10.0, w / 40.0)).reshape(-1)
        prof = col - slow
        prof = prof.astype(np.float32)
        prof -= np.mean(prof)
        # Autocorrelation; the first plausible positive peak is the finger pitch.
        ac = np.correlate(prof, prof, mode="full")[w - 1:]
        if ac[0] <= 1e-6:
            return fallback
        ac = ac / ac[0]
        lo = max(4, int(round(w / 220.0)))
        hi = min(int(round(w / 55.0)), 28)
        if hi <= lo + 1:
            return fallback
        segment = ac[lo:hi]
        if segment.size == 0:
            return fallback
        lag = int(np.argmax(segment) + lo)
        if lag < lo or lag > hi:
            return fallback
        return float(np.clip(lag, 4.5, 22.0))
    except Exception:
        return fallback


def remove_stripe_frequency(arr: np.ndarray, pitch: float) -> np.ndarray:
    """Suppress high-frequency vertical finger residuals while preserving defects.

    False positives after higher-resolution stitching are mostly tiny vertical
    fragments caused by normal line phase/width residuals.  A horizontal blur over
    roughly one to two line pitches cancels those residuals without erasing real
    stains, blobs, holes, or diagonal tears.
    """
    sx = max(2.0, float(pitch) * 1.35)
    sy = max(1.2, float(pitch) * 0.35)
    return cv2.GaussianBlur(arr.astype(np.float32), (0, 0), sigmaX=sx, sigmaY=sy)


def local_background_residual(channel: np.ndarray, pitch: float) -> np.ndarray:
    """Local residual for compact particles/holes after removing slow shading."""
    k = odd_kernel(int(round(max(13.0, pitch * 5.0))))
    # Median background is robust to small dust and holes.
    if channel.dtype != np.uint8:
        ch8 = np.clip(channel, 0, 255).astype(np.uint8)
    else:
        ch8 = channel
    bg = cv2.medianBlur(ch8, k).astype(np.float32)
    return channel.astype(np.float32) - bg


def compute_transverse_texture_score(
    img_bgr: np.ndarray,
    interior_mask: np.ndarray,
    pitch: float,
) -> np.ndarray:
    """Score local disruptions that vary along the normally uniform line direction.

    Normal device fingers are nearly constant along y.  Faint scratches, smears,
    particles, and delamination edges introduce transverse (dy) structure even
    when their raw intensity/chroma contrast is weak.  The score is normalized
    robustly and row-wide responses are suppressed so stitch bands do not turn
    into hundreds of candidates.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    # Remove slow illumination and the dominant per-column line level first.
    expected = column_row_expected(
        L, interior_mask, col_smooth=0.0, row_smooth=max(16.0, 0.030 * L.shape[0])
    )
    resid = L - expected
    gy = cv2.Sobel(resid, cv2.CV_32F, 0, 1, ksize=3)
    gy = np.abs(gy)
    # Aggregate across roughly one finger pitch, but only lightly along y.
    transverse = cv2.GaussianBlur(
        gy, (0, 0), sigmaX=max(1.5, 0.55 * float(pitch)), sigmaY=1.1
    )
    z = np.maximum(0.0, robust_z(transverse, interior_mask, floor=0.65))
    # Eliminate broad horizontal/stitch bands unless they have a sharp local peak.
    row_bg = cv2.GaussianBlur(z, (0, 0), sigmaX=max(16.0, 0.065 * L.shape[1]), sigmaY=0.6)
    local = np.maximum(0.0, z - 0.72 * row_bg)
    local[~interior_mask] = 0.0
    return local.astype(np.float32)


def large_yellow_stain_override(img_bgr: np.ndarray, fixed_border_px: int = 18) -> np.ndarray:
    """Detect huge yellow/gold stains that touch the cell frame.

    The normal interior mask deliberately removes yellow components touching the
    border so the frame does not become a defect.  That is good for clean cells,
    but a catastrophic gold/yellow stain can touch the top frame and get masked
    out.  This override only keeps yellow components that extend deeply into the
    cell and are far too large/thick to be ordinary border/finger geometry.
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    yellow = (H >= 12) & (H <= 48) & (S > 42) & (V > 115)
    fixed = np.ones((h, w), dtype=bool)
    pad = max(int(fixed_border_px), int(round(0.012 * min(h, w))))
    fixed[:pad, :] = False
    fixed[-pad:, :] = False
    fixed[:, :pad] = False
    fixed[:, -pad:] = False
    m = (yellow & fixed).astype(np.uint8) * 255
    # Remove one-pixel noise, but do not close horizontally or normal fingers
    # would all merge into one monster component.
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros((h, w), dtype=bool)
    for i in range(1, n):
        x, y, ww, hh, area = stats[i].tolist()
        if area < max(1200, int(0.006 * h * w)):
            continue
        deep = hh > 0.18 * h and ww > 0.14 * w
        thick = area / max(ww * hh, 1) > 0.10
        not_line = ww > max(30, 0.04 * w)
        if deep and thick and not_line:
            out |= labels == i
    if np.any(out):
        out = cv2.morphologyEx(out.astype(np.uint8) * 255, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)
    return out




def medium_color_blob_override(img_bgr: np.ndarray, fixed_border_px: int = 18) -> np.ndarray:
    """Detect compact/medium gold, brown, black, or residue-colored defects.

    The normal device contains thousands of yellow vertical fingers, so this
    function is deliberately component-based: hue/brightness selects suspicious
    material, then skinny periodic vertical components and frame slivers are
    rejected by shape.  This catches holes/particles/delamination patches that
    are visually obvious but can be under-scored by the stripe residual model.
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    # Gold/brown/black defects.  Normal blue/purple device background has hue
    # near 110..130.  Gold/brown damage is usually hue 8..50 or very dark.
    yellow_brown = (H >= 8) & (H <= 52) & (S > 30) & (V > 45)
    very_dark = (V < 90) & (S > 25)
    yellow_lab = (B > 145) & (L > 95) & (A < 142)
    raw = yellow_brown | very_dark | yellow_lab

    # Fixed border pad only; larger edge defects are allowed back if they extend
    # materially into the cell.  This avoids turning the ordinary frame into a
    # defect while still catching edge delamination.
    pad = max(int(fixed_border_px), int(round(0.018 * min(h, w))))
    valid = np.ones((h, w), dtype=bool)
    valid[:pad, :] = False
    valid[-pad:, :] = False
    valid[:, :pad] = False
    valid[:, -pad:] = False
    raw &= valid

    raw_u8 = raw.astype(np.uint8) * 255
    raw_u8 = cv2.morphologyEx(raw_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw_u8, connectivity=8)
    out = np.zeros((h, w), dtype=bool)
    image_area = h * w
    for i in range(1, n):
        x, y, ww, hh, area = stats[i].tolist()
        if area < max(20, int(0.000035 * image_area)):
            continue
        aspect = max(ww / max(hh, 1), hh / max(ww, 1))
        fill = area / max(ww * hh, 1)

        # Periodic fingers: tall, skinny, and almost vertical.  They can have the
        # same yellow hue as real damage, so shape rejection is mandatory.
        if hh > 0.22 * h and ww <= max(8, int(0.012 * w)):
            continue
        if aspect > 14.0 and fill < 0.45 and area < 0.006 * image_area:
            continue
        if aspect > 25.0 and ww < 0.025 * w:
            continue

        # Ordinary frame remnants are long horizontal/vertical slivers near the
        # edge; real edge defects tend to be compact blobs or broad patches.
        near_left_right = x <= pad + 4 or (x + ww) >= w - pad - 4
        near_top_bottom = y <= pad + 4 or (y + hh) >= h - pad - 4
        if near_left_right and hh > 0.35 * h and ww < 0.07 * w:
            continue
        if near_top_bottom and ww > 0.35 * w and hh < 0.07 * h:
            continue
        # Broad components touching the top/bottom frame are usually the ordinary
        # cell frame connected to a stain. The huge-stain detector handles the
        # real damage separately without dragging the whole frame into the bbox.
        if near_top_bottom and ww > 0.50 * w and area > 0.01 * image_area:
            continue

        # Keep compact blobs, ragged patches, and broad stains; discard very thin
        # color ticks unless their area is substantial.
        compact = ww >= 5 and hh >= 5 and aspect <= 10.0
        broad_patch = area >= max(350, int(0.00065 * image_area)) and ww >= 10 and hh >= 10
        if compact or broad_patch:
            out |= labels == i

    if np.any(out):
        out_u8 = out.astype(np.uint8) * 255
        out_u8 = cv2.morphologyEx(out_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        out = out_u8 > 0
    return out


def append_large_material_density_components(
    img_bgr: np.ndarray,
    comps: list[DefectComponent],
    kept_mask: np.ndarray,
) -> tuple[list[DefectComponent], np.ndarray]:
    """Add broad low-contrast material/stain regions that component thresholds fragment.

    This pass is aimed at catastrophic pale yellow/brown delamination sheets. It
    uses a local density of suspicious material pixels, not raw color alone, so
    ordinary thin yellow fingers and the normal top frame do not become full-cell
    boxes.  The top frame itself is ignored; only dense material extending well
    into the active cell is kept.
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    suspicious = (
        ((H >= 7) & (H <= 62) & (S > 12) & (V > 80))
        | ((B > 136) & (L > 118) & (S > 8))
        | ((V < 105) & (S > 3))
    )
    # Absolute crop edge is never evidence for this broad-density pass.
    edge_pad = max(8, int(round(0.012 * min(h, w))))
    suspicious[:edge_pad, :] = False
    suspicious[-edge_pad:, :] = False
    suspicious[:, :edge_pad] = False
    suspicious[:, -edge_pad:] = False

    k = odd_kernel(int(round(max(25, 0.034 * min(h, w)))))
    density = cv2.blur(suspicious.astype(np.float32), (k, k))
    dense = (density > 0.34).astype(np.uint8) * 255

    # Do not let the ordinary top metal frame connect the entire device.  A real
    # sheet defect that reaches the frame will still be recovered from rows below
    # this cut and then expanded upward within its own x-span.
    top_cut = max(46, int(round(0.065 * h)))
    dense[:top_cut, :] = 0
    dense = cv2.morphologyEx(dense, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    dense = cv2.morphologyEx(dense, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(dense, connectivity=8)
    image_area = h * w
    new = list(comps)
    out_mask = kept_mask.copy()

    def _box_area(b):
        return max(0, int(b[2])) * max(0, int(b[3]))

    def _inter_area(a, b):
        ax0, ay0, aw, ah = [int(v) for v in a]
        bx0, by0, bw, bh = [int(v) for v in b]
        ax1, ay1 = ax0 + aw, ay0 + ah
        bx1, by1 = bx0 + bw, by0 + bh
        return max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))

    for i in range(1, n):
        x, y, ww, hh, area = stats[i].tolist()
        bbox_area = ww * hh
        if area < max(1500, int(0.0028 * image_area)):
            continue
        if ww < max(28, int(0.040 * w)) or hh < max(28, int(0.040 * h)):
            continue
        # Ordinary frame strips are long and shallow.  Real sheets extend into the
        # cell and/or are compact-ish components below the frame cut.
        if hh < 0.070 * h and ww > 0.45 * w:
            continue
        if y <= top_cut + 2 and hh < 0.13 * h:
            continue
        fill = area / max(bbox_area, 1)
        if fill < 0.16 and area < 0.010 * image_area:
            continue

        # If an existing large component already covers this, skip the duplicate.
        candidate_box = [x, y, ww, hh]
        duplicate = False
        for d in new:
            b = d.bbox_px
            ia = _inter_area(candidate_box, b)
            if ia / max(1, min(_box_area(candidate_box), _box_area(b))) >= 0.60:
                # Do not let small specks inside a sheet suppress the sheet.
                if _box_area(candidate_box) > 4 * _box_area(b):
                    continue
                duplicate = True
                break
        if duplicate:
            continue

        # Expand a bit so the visible halo/dirty boundary is included, but do not
        # expand across the whole wafer row.
        grow = max(8, int(round(0.018 * min(h, w))))
        x0 = max(0, x - grow)
        y0 = max(0, y - grow)
        x1 = min(w, x + ww + grow)
        y1 = min(h, y + hh + grow)
        if y <= top_cut + 4 and hh > 0.18 * h:
            y0 = 0

        local = (labels[y:y + hh, x:x + ww] == i).astype(np.uint8) * 255
        full = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        ly0, lx0 = y - y0, x - x0
        full[ly0:ly0 + hh, lx0:lx0 + ww] = local
        full = cv2.dilate(full, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
        full = cv2.morphologyEx(full, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
        # For very large sheets, the dense mask is the evidence, but using the
        # expanded bbox as the review rectangle makes it obvious what to inspect.
        poly = _component_polygon(full, x=x0, y=y0)
        score_val = 36.0 if bbox_area > 0.04 * image_area else 24.0
        new.append(DefectComponent(
            bbox_px=[int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
            polygon_px=poly,
            area_px=int(np.count_nonzero(full)),
            score=round(float(score_val), 3),
            mean_score=round(float(score_val), 3),
            seam_overlap_frac=0.0,
            reason="kept_large_material_density",
        ))
        out_mask[y0:y1, x0:x1] = np.maximum(out_mask[y0:y1, x0:x1], full)

    new.sort(key=lambda d: d.score, reverse=True)
    return new, out_mask



def compute_defect_score(
    residual_lab: np.ndarray,
    img_bgr: np.ndarray,
    interior_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build a multi-scale defect score.

    The older detector scored raw residuals directly.  That worked on very soft
    crops, but once the images got sharper it fired on normal vertical stripe
    residuals.  This version separates the problem into two detectors:

    1. A stripe-suppressed, smoothed residual detector for large stains, holes,
       delamination-like smears, and broad damage.
    2. A compact-particle detector for small dark/colored objects, with later
       component filters that reject vertical-finger fragments.
    """
    lab_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L_img = lab_img[:, :, 0]
    A_img = lab_img[:, :, 1]
    B_img = lab_img[:, :, 2]

    rL = residual_lab[:, :, 0].astype(np.float32)
    rA = residual_lab[:, :, 1].astype(np.float32)
    rB = residual_lab[:, :, 2].astype(np.float32)

    pitch = estimate_vertical_pitch_px(img_bgr, interior_mask)

    # Large/smear channel: suppress stripe-frequency residuals before z-scoring.
    rL_large = remove_stripe_frequency(rL, pitch)
    rA_large = remove_stripe_frequency(rA, pitch)
    rB_large = remove_stripe_frequency(rB, pitch)
    zL_large = robust_z(rL_large, interior_mask, floor=1.4)
    zA_large = robust_z(rA_large, interior_mask, floor=0.9)
    zB_large = robust_z(rB_large, interior_mask, floor=0.9)
    chroma_large = np.sqrt(zA_large * zA_large + zB_large * zB_large)
    large_score = np.maximum.reduce([
        0.95 * np.abs(zL_large),
        1.10 * chroma_large,
        np.maximum(0.0, -zL_large),
        0.90 * np.maximum(0.0, zL_large),
    ]).astype(np.float32)

    # Compact channel: detect genuinely local dark/color particles.  Use local
    # background so slow illumination and stitch troughs do not matter much.
    locL = local_background_residual(L_img, pitch)
    locA = local_background_residual(A_img, pitch)
    locB = local_background_residual(B_img, pitch)
    zLocL = robust_z(locL, interior_mask, floor=1.1)
    zLocA = robust_z(locA, interior_mask, floor=0.8)
    zLocB = robust_z(locB, interior_mask, floor=0.8)
    dark_local = np.maximum(0.0, -zLocL)
    bright_local = np.maximum(0.0, zLocL)
    chroma_local = np.sqrt(zLocA * zLocA + zLocB * zLocB)

    # Texture/edge channels.  The Laplacian catches compact roughness while the
    # transverse score targets faint scratches/smears that interrupt the normally
    # y-uniform device fingers.
    lap = cv2.Laplacian(rL_large.astype(np.float32), cv2.CV_32F, ksize=3)
    tex_z = np.abs(robust_z(lap, interior_mask, floor=1.0))
    transverse_score = compute_transverse_texture_score(img_bgr, interior_mask, pitch)

    # Raw residual evidence is kept for component decisions, not directly as a
    # high-weight score.
    zL_raw = robust_z(rL, interior_mask, floor=1.4)
    zA_raw = robust_z(rA, interior_mask, floor=0.9)
    zB_raw = robust_z(rB, interior_mask, floor=0.9)
    chroma_raw = np.sqrt(zA_raw * zA_raw + zB_raw * zB_raw)

    compact_score = np.maximum.reduce([
        1.05 * dark_local,
        0.80 * bright_local,
        0.95 * chroma_local,
        0.45 * tex_z,
        0.58 * transverse_score,
    ]).astype(np.float32)

    # A transverse-only response is too permissive on its own.  Promote it into
    # the main score only when another low-frequency or compact cue corroborates
    # the same location.
    transverse_supported = np.where(
        (large_score >= 1.45) | (compact_score >= 2.8) | (chroma_local >= 1.8),
        0.78 * transverse_score,
        0.0,
    ).astype(np.float32)
    score = np.maximum.reduce([large_score, compact_score, transverse_supported]).astype(np.float32)

    stain_override = large_yellow_stain_override(img_bgr)
    color_blob_override = medium_color_blob_override(img_bgr)
    force_mask = stain_override | color_blob_override

    score[~interior_mask] = 0.0
    if np.any(color_blob_override):
        score[color_blob_override] = np.maximum(score[color_blob_override], 11.5)
    if np.any(stain_override):
        # Use a moderate-high score: enough to force a candidate, not so huge that
        # it dominates score-based sorting forever. The component shape/area carries
        # the actual evidence.
        score[stain_override] = np.maximum(score[stain_override], 13.0)

    parts = {
        "stain_override": stain_override.astype(np.float32),
        "color_blob_override": color_blob_override.astype(np.float32),
        "force_mask": force_mask.astype(np.float32),
        "pitch_px": np.full(score.shape, float(pitch), dtype=np.float32),
        "large_score": large_score.astype(np.float32),
        "compact_score": compact_score.astype(np.float32),
        "zL": zL_raw.astype(np.float32),
        "zL_large": zL_large.astype(np.float32),
        "chroma_z": chroma_raw.astype(np.float32),
        "chroma_large": chroma_large.astype(np.float32),
        "chroma_local": chroma_local.astype(np.float32),
        "tex_z": tex_z.astype(np.float32),
        "transverse_score": transverse_score.astype(np.float32),
        "dark_z": np.maximum(0.0, -zL_raw).astype(np.float32),
        "dark_local": dark_local.astype(np.float32),
        "bright_z": np.maximum(0.0, zL_raw).astype(np.float32),
    }
    return score, parts


def morphology_score_mask(score: np.ndarray, params: DetectorParams) -> np.ndarray:
    mask = (score >= params.threshold).astype(np.uint8) * 255
    if params.open_radius > 0:
        k = odd_kernel(params.open_radius * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if params.close_radius > 0:
        k = odd_kernel(params.close_radius * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _component_polygon(component_mask: np.ndarray, x: int, y: int) -> list[list[int]]:
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [[x, y]]
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    eps = max(1.0, 0.006 * peri)
    approx = cv2.approxPolyDP(cnt, eps, True)
    pts = approx.reshape(-1, 2)
    return [[int(px + x), int(py + y)] for px, py in pts]


def _open_mask_holes_with_notches(mask: np.ndarray, min_hole_area: int = 10) -> np.ndarray:
    """Open internal clean islands to the exterior with the shortest narrow notch.

    The output schema stores one simple polygon per defect and cannot directly
    represent polygon holes.  RETR_EXTERNAL would silently fill those islands.
    A one-to-two pixel notch preserves the clean island in the exported simple
    polygon while sacrificing only a negligible amount of defect mask.
    """
    out = (mask > 0).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(out.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None or not contours:
        return out
    hierarchy = hierarchy[0]
    holes = [i for i, hrec in enumerate(hierarchy) if int(hrec[3]) >= 0 and cv2.contourArea(contours[i]) >= min_hole_area]
    for hi in holes:
        parent = int(hierarchy[hi][3])
        if parent < 0 or parent >= len(contours):
            continue
        hc = contours[hi].reshape(-1, 2)
        oc = contours[parent].reshape(-1, 2)
        if hc.size == 0 or oc.size == 0:
            continue
        hs = hc[::max(1, len(hc) // 160)]
        os = oc[::max(1, len(oc) // 240)]
        # Small sampled pairwise search; masks here are component-local.
        d2 = np.sum((hs[:, None, :].astype(np.float32) - os[None, :, :].astype(np.float32)) ** 2, axis=2)
        iy, ix = np.unravel_index(int(np.argmin(d2)), d2.shape)
        p0 = tuple(int(v) for v in hs[iy])
        p1 = tuple(int(v) for v in os[ix])
        cv2.line(out, p0, p1, 0, 2, cv2.LINE_8)
    return out


def _complete_supported_internal_regions(
    mask: np.ndarray,
    score: np.ndarray,
    parts: dict[str, np.ndarray],
    strong: np.ndarray,
    material_seed: np.ndarray,
    valid: np.ndarray,
    reasons: set[str],
) -> np.ndarray:
    """Fill defect interiors that are enclosed by a supported defect shell.

    The detector often sees the high-contrast rim of a blister, particle crater,
    delamination bubble, or circular edge defect while the interior retains some
    normal-looking vertical fingers.  Treating that interior as a clean polygon
    hole under-subtracts the physical defect.  Conversely, blindly filling every
    contour would recreate the destructive giant masks from older versions.

    A hole is filled only when its enclosing boundary is well supported and the
    hole is compact/small or contains independent low-threshold physical evidence.
    Unsupported large clean islands remain holes.
    """
    out = (mask > 0).astype(np.uint8) * 255
    ys0, xs0 = np.nonzero(out)
    if xs0.size == 0:
        return out

    h, w = out.shape[:2]
    image_area = h * w
    x0, x1 = max(0, int(xs0.min()) - 3), min(w, int(xs0.max()) + 4)
    y0, y1 = max(0, int(ys0.min()) - 3), min(h, int(ys0.max()) + 4)
    local = out[y0:y1, x0:x1].copy()

    # Close only one- or two-pixel notches before topology analysis.  This joins
    # a broken shell but cannot bridge the wide clean gaps that caused giant boxes.
    topo = cv2.morphologyEx(
        local,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    contours, hierarchy = cv2.findContours(topo.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None or not contours:
        return out
    hierarchy = hierarchy[0]

    large_score = parts.get("large_score", score)
    compact_score = parts.get("compact_score", score)
    dark_local = parts.get("dark_local", np.zeros_like(score))
    chroma_local = parts.get("chroma_local", np.zeros_like(score))
    transverse = parts.get("transverse_score", np.zeros_like(score))
    template_local = parts.get("template_local", np.zeros_like(score))

    reason_text = "+".join(sorted(reasons)).lower()
    broad_reason = any(k in reason_text for k in (
        "large_material", "edge_material", "edge_grown", "clustered_diffuse",
        "delamination", "border_transverse",
    ))

    changed = False
    for hi, hrec in enumerate(hierarchy):
        parent = int(hrec[3])
        if parent < 0:
            continue
        hole_area = float(cv2.contourArea(contours[hi]))
        if hole_area < 6.0:
            continue
        outer_area = float(cv2.contourArea(contours[parent]))
        if outer_area <= 0:
            continue
        if hole_area > 0.035 * image_area:
            continue
        if hole_area > 0.82 * outer_area and not broad_reason:
            continue

        hx, hy, hw, hh = cv2.boundingRect(contours[hi])
        if hw <= 1 or hh <= 1:
            continue
        hole_local = np.zeros_like(local)
        cv2.drawContours(hole_local, contours, hi, 255, -1)
        hole = hole_local > 0
        if not np.any(hole):
            continue

        # Boundary support around the hole.  A real shell should surround most of
        # the hole perimeter; accidental gaps between unrelated proposals do not.
        ring = cv2.dilate(
            hole_local,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        ) > 0
        ring &= ~hole
        ring_count = int(np.count_nonzero(ring))
        if ring_count == 0:
            continue
        current_local = local > 0
        ring_coverage = float(np.count_nonzero(ring & current_local) / ring_count)

        full_hole = np.zeros((h, w), dtype=bool)
        full_hole[y0:y1, x0:x1] = hole
        full_ring = np.zeros((h, w), dtype=bool)
        full_ring[y0:y1, x0:x1] = ring
        full_hole &= valid
        if np.count_nonzero(full_hole) < 4:
            continue

        vals = np.maximum.reduce([
            score[full_hole] / 8.8,
            large_score[full_hole] / 3.2,
            compact_score[full_hole] / 4.8,
            dark_local[full_hole] / 2.1,
            chroma_local[full_hole] / 2.3,
            transverse[full_hole] / 3.2,
            template_local[full_hole] / 2.8,
        ]).astype(np.float32)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            ev_mean = float(np.mean(vals))
            ev_p70 = float(np.percentile(vals, 70))
            ev_p90 = float(np.percentile(vals, 90))
        else:
            ev_mean = ev_p70 = ev_p90 = 0.0

        ring_material = float(np.count_nonzero(full_ring & (material_seed | strong)) / max(np.count_nonzero(full_ring), 1))
        inside_material = float(np.count_nonzero(full_hole & (material_seed | strong)) / max(np.count_nonzero(full_hole), 1))
        compactness = hole_area / max(hw * hh, 1)
        parent_pixels = int(np.count_nonzero(current_local))
        hole_to_shell = hole_area / max(parent_pixels, 1)

        small_closed = (
            hole_area <= max(220.0, 0.00048 * image_area)
            and ring_coverage >= 0.52
            and compactness >= 0.18
        )
        evidence_closed = (
            ring_coverage >= 0.58
            and compactness >= 0.12
            and (
                ev_p70 >= 0.62
                or ev_p90 >= 1.18
                or inside_material >= 0.045
            )
        )
        shell_enclosed = (
            ring_coverage >= 0.68
            and ring_material >= 0.10
            and hole_to_shell <= (2.8 if broad_reason else 1.65)
            and compactness >= 0.22
            and (ev_p90 >= 0.42 or broad_reason)
        )
        broad_interior = (
            broad_reason
            and ring_coverage >= 0.62
            and hole_area <= 0.70 * outer_area
            and (ev_mean >= 0.24 or ring_material >= 0.18)
        )

        if small_closed or evidence_closed or shell_enclosed or broad_interior:
            local[hole] = 255
            changed = True

    if changed:
        local = cv2.morphologyEx(
            local,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        out[y0:y1, x0:x1] = local
    return out


def _tighten_periodic_texture_only_region(
    mask: np.ndarray,
    img_bgr: np.ndarray,
    score: np.ndarray,
    parts: dict[str, np.ndarray],
    strong: np.ndarray,
    material_seed: np.ndarray,
    valid: np.ndarray,
    reasons: set[str],
) -> tuple[np.ndarray, bool]:
    """Remove or tighten large polygons caused only by periodic finger residuals.

    A recurring failure mode is a tall/medium polygon whose boundary is built from
    several vertical-line residuals while its interior is visually normal.  The
    transverse and compact channels can be very high on those residuals even when
    there is no independent dark, chromatic, or low-frequency material evidence.

    This guard is deliberately late and conservative:
      * obvious material defects are returned unchanged;
      * only comparatively large regions dominated by transverse texture qualify;
      * suspicious regions are reduced to corroborated physical-evidence islands;
      * when no corroborated island exists, the proposal is removed instead of
        turning a clean stripe patch into subtraction geometry.

    Small particles, diagonal scratches, colored/dark defects, broad stains, and
    force-mask detections do not enter this branch.
    """
    out = (mask > 0).astype(np.uint8) * 255
    ys, xs = np.nonzero(out)
    if xs.size < 12:
        return out, False

    h, w = out.shape[:2]
    image_area = h * w
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    bw, bh = x1 - x0, y1 - y0
    bbox_area = max(1, bw * bh)
    area = int(xs.size)
    aspect_y = bh / max(bw, 1)
    near_edge = x0 <= 8 or y0 <= 8 or x1 >= w - 8 or y1 >= h - 8

    # Do not touch ordinary compact detections.  The problematic regions are
    # visibly large review/subtraction shapes, usually tall and finger-aligned.
    geometry_candidate = (
        (bh >= max(48, int(round(0.065 * h))) and aspect_y >= 1.30 and bw <= 0.20 * w)
        or (bbox_area >= max(900, int(0.0017 * image_area)) and area >= 180)
    )
    if not geometry_candidate:
        return out, False

    m = out > 0
    force = parts.get("force_mask", np.zeros_like(score)) > 0
    large = parts.get("large_score", np.zeros_like(score)).astype(np.float32)
    compact = parts.get("compact_score", np.zeros_like(score)).astype(np.float32)
    dark = parts.get("dark_local", np.zeros_like(score)).astype(np.float32)
    chroma = parts.get("chroma_local", np.zeros_like(score)).astype(np.float32)
    transverse = parts.get("transverse_score", np.zeros_like(score)).astype(np.float32)
    template_local = parts.get("template_local", np.zeros_like(score)).astype(np.float32)

    hsv = parts.get("_periodic_guard_hsv")
    if hsv is None:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        parts["_periodic_guard_hsv"] = hsv
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    gx = parts.get("_periodic_guard_gx")
    gy = parts.get("_periodic_guard_gy")
    if gx is None or gy is None:
        L_guard = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        gx = np.abs(cv2.Sobel(L_guard, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(L_guard, cv2.CV_32F, 0, 1, ksize=3))
        parts["_periodic_guard_gx"] = gx
        parts["_periodic_guard_gy"] = gy

    def pct(arr: np.ndarray, q: float, default: float = 0.0) -> float:
        vals = arr[m]
        vals = vals[np.isfinite(vals)]
        return float(np.percentile(vals, q)) if vals.size else float(default)

    force_frac = float(np.count_nonzero(m & force) / max(area, 1))
    qualified_material = material_seed & (
        (dark >= 1.25) | (chroma >= 1.75) | (large >= 2.70) | (V < 118)
    )
    material_frac = float(np.count_nonzero(m & qualified_material) / max(area, 1))
    large_p90 = pct(large, 90)
    large_p98 = pct(large, 98)
    compact_p90 = pct(compact, 90)
    dark_p95 = pct(dark, 95)
    chroma_p90 = pct(chroma, 90)
    transverse_p90 = pct(transverse, 90)
    template_p90 = pct(template_local, 90)
    v_p10 = pct(V.astype(np.float32), 10, 255.0)
    gx_vals = np.minimum(gx[m], 80.0)
    gy_vals = np.minimum(gy[m], 80.0)
    stripe_gradient_ratio = float((np.mean(gx_vals) + 1.0) / (np.mean(gy_vals) + 1.0)) if gx_vals.size else 1.0
    cross_gradient_frac = float(np.mean(gy[m] > 0.75 * gx[m])) if gx_vals.size else 1.0

    # Independent physical evidence.  Requiring two moderate cues avoids treating
    # one ringing/phase channel as material, while the strong single-cue clauses
    # preserve genuinely dark or strongly chromatic defects.
    votes = (
        (large >= 2.80).astype(np.uint8)
        + (dark >= 1.50).astype(np.uint8)
        + (chroma >= 1.90).astype(np.uint8)
        + ((V < 145) & (S > 2)).astype(np.uint8)
        + (template_local >= 3.8).astype(np.uint8)
    )
    physical_core = (
        (votes >= 2)
        | (large >= 4.50)
        | (dark >= 3.20)
        | (chroma >= 3.50)
        | force
        | qualified_material
    ) & m & valid
    core_frac = float(np.count_nonzero(physical_core) / max(area, 1))

    reason_text = "+".join(sorted(reasons)).lower()
    explicit_material_reason = any(k in reason_text for k in (
        "large_material", "edge_material", "salient_spot", "tiny_dark",
        "sparse_particle", "color", "stain",
    ))

    transverse_dominated = (
        transverse_p90 >= 5.5
        and transverse_p90 >= 1.75 * max(large_p90, 1.0)
        and compact_p90 >= 6.0
    )
    edge_texture_only = (
        near_edge
        and transverse_p90 >= 5.0
        and compact_p90 >= 6.0
        and chroma_p90 < 2.25
        and v_p10 > 142.0
    )
    stripe_phase_dominated = (
        ("fused_micro" in reason_text or "cluster" in reason_text)
        and stripe_gradient_ratio >= 11.5
        and cross_gradient_frac <= 0.14
        and transverse_p90 < 2.8
        and dark_p95 < 2.35
        and v_p10 > 145.0
    )
    low_independent_evidence = (
        force_frac < 0.002
        and material_frac < 0.035
        and dark_p95 < 2.35
        and chroma_p90 < 2.25
        and core_frac < 0.065
        and v_p10 > 138.0
    )

    # Strong, non-periodic evidence protects real defects completely.  At image
    # edges, large-score alone is not protective because the frame can generate it.
    if stripe_phase_dominated:
        # Normal fingers can make material/core/large/chroma fractions look huge.
        # In this special case protection must come from evidence that actually
        # crosses or disrupts the fingers, or from genuinely dark material.
        protected = (
            force_frac >= 0.002
            or v_p10 < 135.0
            or dark_p95 >= 3.20
            or (chroma_p90 >= 3.10 and cross_gradient_frac >= 0.18)
            or (explicit_material_reason and cross_gradient_frac >= 0.18 and material_frac >= 0.055)
        )
    elif near_edge:
        protected = (
            force_frac >= 0.002
            or chroma_p90 >= 3.10
            or (material_frac >= 0.080 and v_p10 < 135.0)
            or (dark_p95 >= 4.80 and v_p10 < 125.0)
            or (explicit_material_reason and material_frac >= 0.055 and v_p10 < 140.0)
        )
    else:
        protected = (
            force_frac >= 0.002
            or material_frac >= 0.055
            or dark_p95 >= 3.20
            or chroma_p90 >= 3.10
            or core_frac >= 0.14
            or large_p90 >= 4.20
            or (explicit_material_reason and core_frac >= 0.045)
        )
    suspicious_texture = transverse_dominated or edge_texture_only or stripe_phase_dominated
    suspicious_low_evidence = low_independent_evidence or edge_texture_only or stripe_phase_dominated
    if protected or not suspicious_texture or not suspicious_low_evidence:
        return out, False

    if edge_texture_only:
        # The crop/frame can create strong large/dark residuals without any real
        # material.  Do not allow those boundary pixels to seed a large polygon.
        edge_inner = np.ones_like(m, dtype=bool)
        edge_guard = max(8, int(round(0.012 * min(h, w))))
        edge_inner[:edge_guard, :] = False
        edge_inner[-edge_guard:, :] = False
        edge_inner[:, :edge_guard] = False
        edge_inner[:, -edge_guard:] = False
        physical_core &= edge_inner & (
            (chroma >= 2.05) | (V < 130) | (dark >= 2.8) | force
        )

    if stripe_phase_dominated:
        # In this mode large/chroma scores are produced by normal gold fingers
        # being out of phase with the template.  They are not valid seeds by
        # themselves.  Require a cross-line, dark, or multi-cue disturbance.
        stripe_core = (
            force
            | ((V < 128) & (S > 3))
            | ((dark >= 2.20) & (V < 155))
            | ((chroma >= 3.00) & ((dark >= 0.80) | (transverse >= 2.4)))
            | ((transverse >= 3.0) & ((dark >= 0.55) | (chroma >= 1.20) | (large >= 2.6)))
        ) & m & valid
        physical_core &= stripe_core

    # Keep only compact islands with corroborated physical support.  A little
    # geodesic growth retains the visible halo, but cannot reconnect distant
    # parallel fingers into a tall clean polygon.
    core_u8 = physical_core.astype(np.uint8) * 255
    if not np.any(core_u8):
        return np.zeros_like(out), True

    allowed = (
        physical_core
        | (
            (transverse >= 3.8)
            & ((dark >= 0.70) | (chroma >= 1.00) | (large >= 2.05) | (template_local >= 2.8))
        )
    ) & m & valid
    grown = core_u8.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    allowed_u8 = allowed.astype(np.uint8) * 255
    for _ in range(4):
        nxt = cv2.bitwise_and(cv2.dilate(grown, kernel, iterations=1), allowed_u8)
        nxt = cv2.bitwise_or(nxt, core_u8)
        if np.array_equal(nxt, grown):
            break
        grown = nxt

    # Long vertical strands that do not remain close to a physical core are the
    # signature of finger-phase residuals.  Remove them after growth.
    vlen = odd_kernel(max(15, int(round(0.34 * bh))))
    long_vertical = cv2.morphologyEx(
        grown,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, vlen)),
    ) > 0
    core_near = cv2.dilate(
        core_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    ) > 0
    grown[(long_vertical & ~core_near)] = 0

    # Emit only connected pieces that actually overlap a corroborated core.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grown, connectivity=8)
    tightened = np.zeros_like(out)
    min_piece = max(3, int(round(0.000008 * image_area)))
    for i in range(1, n):
        x, y, ww, hh, piece_area = stats[i].tolist()
        if piece_area < min_piece:
            continue
        piece = labels == i
        if not np.any(piece & physical_core):
            continue
        piece_core = int(np.count_nonzero(piece & physical_core))
        piece_aspect = max(ww / max(hh, 1), hh / max(ww, 1))
        if piece_aspect > 8.0 and piece_core < max(3, int(0.12 * piece_area)):
            continue
        tightened[piece] = 255

    if not np.any(tightened):
        return np.zeros_like(out), True

    # A suspicious region should get substantially smaller.  If numerical noise
    # leaves most of it intact, fall back to a compact dilation of the core.
    if np.count_nonzero(tightened) > 0.62 * area:
        tightened = cv2.dilate(
            core_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        tightened = cv2.bitwise_and(tightened, out)

    return tightened, True

def _is_line_like(w: int, h: int, area: int) -> bool:
    if w <= 0 or h <= 0:
        return False
    aspect = max(w / max(h, 1), h / max(w, 1))
    fill = area / max(w * h, 1)
    return aspect > 8.0 and fill < 0.65


def _component_orientation_deg(component_mask: np.ndarray) -> float | None:
    """Return dominant component angle in degrees, where 90 is vertical.

    Used only to reject normal vertical-finger fragments while keeping diagonal
    scratches/tears.
    """
    ys, xs = np.nonzero(component_mask)
    if xs.size < 6:
        return None
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    pts -= np.mean(pts, axis=0, keepdims=True)
    cov = np.cov(pts, rowvar=False)
    try:
        vals, vecs = np.linalg.eigh(cov)
        v = vecs[:, int(np.argmax(vals))]
        ang = math.degrees(math.atan2(float(v[1]), float(v[0])))
        # Fold to [0, 180).
        if ang < 0:
            ang += 180.0
        return float(ang)
    except Exception:
        return None


def _is_near_vertical_angle(angle_deg: float | None, tolerance_deg: float = 18.0) -> bool:
    if angle_deg is None:
        return False
    return abs(float(angle_deg) - 90.0) <= float(tolerance_deg)


def find_components(
    score: np.ndarray,
    raw_mask: np.ndarray,
    interior_mask: np.ndarray,
    params: DetectorParams,
    seam_mask: Optional[np.ndarray] = None,
    parts: Optional[dict[str, np.ndarray]] = None,
) -> tuple[list[DefectComponent], np.ndarray]:
    h, w = score.shape
    max_area = int(params.max_area_frac * h * w)

    # First pass for medium/large components.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    kept: list[DefectComponent] = []
    kept_mask = np.zeros_like(raw_mask)

    seam_bool = None
    if seam_mask is not None:
        if seam_mask.shape != score.shape:
            seam_mask = cv2.resize(seam_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        seam_bool = seam_mask > 0

    for i in range(1, n):
        x, y, ww, hh, area = stats[i].tolist()
        if area < params.min_area:
            continue

        comp = labels[y:y + hh, x:x + ww] == i
        force_frac = 0.0
        if parts is not None and parts.get("force_mask") is not None:
            force_local = parts["force_mask"][y:y + hh, x:x + ww] > 0
            force_frac = float(np.count_nonzero(force_local & comp) / max(area, 1))
        forced_component = force_frac >= 0.04
        if area > max_area and not forced_component:
            continue

        comp_scores = score[y:y + hh, x:x + ww][comp]
        if comp_scores.size == 0:
            continue
        peak = float(np.max(comp_scores))
        mean = float(np.mean(comp_scores))
        if peak < params.min_score:
            continue

        # Reject remnants of the normal vertical finger pattern.  After the
        # downscale fix, the most common false positives are skinny, near-vertical
        # pieces of periodic line residual.  Keep diagonal scratches, compact dust,
        # and strong chroma/dark particles.
        chroma_peak = 0.0
        chroma_large_peak = 0.0
        dark_peak = 0.0
        dark_local_peak = 0.0
        large_peak = 0.0
        compact_peak = 0.0
        tex_peak = 0.0
        force_peak = force_frac
        if parts is not None:
            for key, var in [
                ("chroma_z", "chroma_peak"),
                ("chroma_large", "chroma_large_peak"),
                ("dark_z", "dark_peak"),
                ("dark_local", "dark_local_peak"),
                ("large_score", "large_peak"),
                ("compact_score", "compact_peak"),
                ("tex_z", "tex_peak"),
            ]:
                arr = parts.get(key)
                if arr is None:
                    continue
                vals = arr[y:y + hh, x:x + ww][comp]
                if vals.size:
                    val = float(np.max(vals))
                    if var == "chroma_peak":
                        chroma_peak = val
                    elif var == "chroma_large_peak":
                        chroma_large_peak = val
                    elif var == "dark_peak":
                        dark_peak = val
                    elif var == "dark_local_peak":
                        dark_local_peak = val
                    elif var == "large_peak":
                        large_peak = val
                    elif var == "compact_peak":
                        compact_peak = val
                    elif var == "tex_peak":
                        tex_peak = val

        angle = _component_orientation_deg(comp)
        aspect = max(ww / max(hh, 1), hh / max(ww, 1))
        fill = area / max(ww * hh, 1)
        narrow_vertical = (ww <= max(5, int(0.0065 * w))) and (hh >= max(14, 2.2 * ww))
        near_vertical = _is_near_vertical_angle(angle) or narrow_vertical
        line_like = _is_line_like(ww, hh, area) or narrow_vertical

        strong_nonstripe_evidence = (
            forced_component
            or chroma_large_peak >= params.min_score + 1.2
            or dark_local_peak >= params.min_score + 2.0
            or dark_peak >= params.min_score + 4.0
            or large_peak >= params.min_score + 2.5
        )

        # Very skinny vertical fragments are almost always normal finger residuals.
        # This is the main "orange confetti" guard.
        if near_vertical and line_like and not strong_nonstripe_evidence:
            continue

        # Short skinny ticks also appear along many fingers.  Reject them unless
        # they are genuinely dark/colorful enough to be particles.
        if narrow_vertical and area < max(180, int(0.00010 * h * w)) and chroma_peak < params.min_score + 2.0 and dark_local_peak < params.min_score + 2.5:
            continue

        # Low-fill, low-evidence fragments are usually line edge/ringing artifacts.
        if fill < 0.22 and peak < params.min_score + 1.0 and compact_peak < params.min_score + 1.0 and not forced_component:
            continue

        # Suppress tiny/slender artifacts riding the crop frame.  Large/forced
        # edge defects survive, but clean-cell frame residuals disappear.
        edge_margin = max(60, int(round(0.080 * min(h, w))))
        near_edge = x <= edge_margin or y <= edge_margin or (x + ww) >= w - edge_margin or (y + hh) >= h - edge_margin
        edge_sliver = (near_edge and (line_like or area < max(260, int(0.00045 * h * w)) or (aspect > 5.0 and fill < 0.55)))
        if edge_sliver and not forced_component and peak < params.min_score + 14.0:
            continue
        top_bottom_small = (y <= edge_margin or (y + hh) >= h - edge_margin) and area < max(1500, int(0.0030 * h * w))
        if top_bottom_small:
            continue
        corner_small = (
            ((x <= edge_margin and y <= int(1.6 * edge_margin))
             or ((x + ww) >= w - edge_margin and y <= int(1.6 * edge_margin))
             or (x <= edge_margin and (y + hh) >= h - int(1.6 * edge_margin))
             or ((x + ww) >= w - edge_margin and (y + hh) >= h - int(1.6 * edge_margin)))
            and area < max(650, int(0.0012 * h * w))
        )
        if corner_small:
            continue

        seam_frac = 0.0
        if seam_bool is not None and params.seam_suppress:
            seam_local = seam_bool[y:y + hh, x:x + ww]
            seam_frac = float(np.count_nonzero(seam_local & comp) / max(area, 1))
            aspect = max(ww / max(hh, 1), hh / max(ww, 1))
            seam_like = aspect >= params.seam_like_aspect or (hh > 0.45 * h and ww < 0.08 * w)
            lowish = peak < params.min_score + params.seam_low_score_margin
            if seam_frac >= params.seam_overlap_reject_frac and seam_like and lowish:
                continue

        comp_u8 = comp.astype(np.uint8) * 255
        poly = _component_polygon(comp_u8, x=x, y=y)
        kept.append(
            DefectComponent(
                bbox_px=[int(x), int(y), int(ww), int(hh)],
                polygon_px=poly,
                area_px=int(area),
                score=round(peak, 3),
                mean_score=round(mean, 3),
                seam_overlap_frac=round(seam_frac, 3),
            )
        )
        kept_mask[y:y + hh, x:x + ww][comp] = 255

    # Tiny speckles often matter only when clustered around a real defect.
    # Keep them if they are near already-kept components.
    if kept and params.tiny_component_area > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (odd_kernel(params.cluster_dilate_px), odd_kernel(params.cluster_dilate_px)),
        )
        vicinity = cv2.dilate(kept_mask, kernel, iterations=1) > 0
        tiny_candidates = ((score >= params.threshold + 0.65) & interior_mask & vicinity).astype(np.uint8) * 255
        tiny_candidates[kept_mask > 0] = 0
        nt, lt, st, _ = cv2.connectedComponentsWithStats(tiny_candidates, connectivity=8)
        for i in range(1, nt):
            x, y, ww, hh, area = st[i].tolist()
            if area < params.tiny_component_area or area >= params.min_area:
                continue
            edge_margin = max(60, int(round(0.080 * min(h, w))))
            if x <= edge_margin or y <= edge_margin or (x + ww) >= w - edge_margin or (y + hh) >= h - edge_margin:
                continue
            comp = lt[y:y + hh, x:x + ww] == i
            comp_scores = score[y:y + hh, x:x + ww][comp]
            peak = float(np.max(comp_scores)) if comp_scores.size else 0.0
            if peak < params.min_score + 0.65:
                continue
            poly = _component_polygon(comp.astype(np.uint8) * 255, x=x, y=y)
            kept.append(
                DefectComponent(
                    bbox_px=[int(x), int(y), int(ww), int(hh)],
                    polygon_px=poly,
                    area_px=int(area),
                    score=round(peak, 3),
                    mean_score=round(float(np.mean(comp_scores)), 3),
                    seam_overlap_frac=0.0,
                    reason="kept_clustered_tiny",
                )
            )
            kept_mask[y:y + hh, x:x + ww][comp] = 255

    kept.sort(key=lambda d: d.score, reverse=True)
    return kept, kept_mask



def _edge_local_features(img_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> dict[str, np.ndarray]:
    """Vertical-local residual maps that preserve the finger x-phase.

    Used only by edge expansion and tiny recall.  It cancels normal vertical
    fingers by comparing each pixel to a vertical-only local background in the
    same column; local chips/specks/scratches remain.
    """
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    if mask is None:
        mask = np.ones((h, w), dtype=bool)
        pad = max(8, int(round(0.012 * min(h, w))))
        mask[:pad, :] = False
        mask[-pad:, :] = False
        mask[:, :pad] = False
        mask[:, -pad:] = False
    k = odd_kernel(int(round(max(51, 0.09 * min(h, w)))))
    bgL = cv2.blur(lab[:, :, 0], (1, k))
    bgA = cv2.blur(lab[:, :, 1], (1, k))
    bgB = cv2.blur(lab[:, :, 2], (1, k))
    dL = lab[:, :, 0] - bgL
    dA = lab[:, :, 1] - bgA
    dB = lab[:, :, 2] - bgB
    zL = robust_z(dL, mask, floor=1.6)
    zA = robust_z(dA, mask, floor=0.85)
    zB = robust_z(dB, mask, floor=0.85)
    dark = np.maximum(0.0, -zL).astype(np.float32)
    chroma = np.sqrt(zA * zA + zB * zB).astype(np.float32)
    dx = cv2.Sobel(dL.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(dL.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    edge = robust_z(np.sqrt(dx * dx + dy * dy), mask, floor=1.0).astype(np.float32)
    return {"H": hsv[:, :, 0], "S": hsv[:, :, 1], "V": hsv[:, :, 2], "dark": dark, "chroma": chroma, "edge": edge, "dL": dL.astype(np.float32)}


def _mask_to_polygon(mask_u8: np.ndarray, x: int, y: int) -> list[list[int]]:
    return _component_polygon(mask_u8, x=x, y=y)




def append_compact_border_specks(
    img_bgr: np.ndarray,
    comps: list[DefectComponent],
    kept_mask: np.ndarray,
    parts: dict[str, np.ndarray],
) -> tuple[list[DefectComponent], np.ndarray]:
    """Recover only compact, high-confidence particles clipped by a crop edge.

    This is intentionally not a general border detector.  It addresses isolated
    edge specks without reopening the false-positive path from frame strips or
    long periodic fingers.
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    dark_local = parts.get("dark_local", np.zeros((h, w), dtype=np.float32))
    chroma_local = parts.get("chroma_local", np.zeros((h, w), dtype=np.float32))
    transverse = parts.get("transverse_score", np.zeros((h, w), dtype=np.float32))
    force = parts.get("force_mask", np.zeros((h, w), dtype=np.float32)) > 0

    depth = max(8, int(round(0.014 * min(h, w))))
    zone = np.zeros((h, w), dtype=bool)
    # Only the left/right crop edge.  Top/bottom material is handled by connected
    # edge-component growth; enabling a general top/bottom speck pass produced
    # many frame false positives.
    zone[:, :depth] = True
    zone[:, -depth:] = True
    strong = (
        ((V < 72) & (S > 2))
        | ((dark_local >= 5.6) & (V < 155))
        | ((chroma_local >= 6.0) & (S > 12) & (V < 175))
        | ((transverse >= 7.0) & (dark_local >= 2.4) & (V < 145))
        | force
    ) & zone
    strong[0, :] = strong[-1, :] = False
    strong[:, 0] = strong[:, -1] = False
    u8 = strong.astype(np.uint8) * 255
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    new = list(comps)
    out = kept_mask.copy()
    image_area = h * w
    for i in range(1, n):
        x, y, ww, hh, area = stats[i].tolist()
        if area < 2 or area > max(450, int(0.0012 * image_area)):
            continue
        aspect = max(ww / max(hh, 1), hh / max(ww, 1))
        fill = area / max(ww * hh, 1)
        if aspect > 5.5 and area < 35:
            continue
        if aspect > 9.0 or fill < 0.12:
            continue
        cm = labels[y:y + hh, x:x + ww] == i
        if not np.any(cm):
            continue
        vals_v = V[y:y + hh, x:x + ww][cm]
        vals_d = dark_local[y:y + hh, x:x + ww][cm]
        vals_c = chroma_local[y:y + hh, x:x + ww][cm]
        if float(np.percentile(vals_d, 90)) < 5.0 and float(np.percentile(vals_c, 90)) < 5.5 and float(np.min(vals_v)) > 68:
            continue
        full = np.zeros((h, w), dtype=np.uint8)
        full[y:y + hh, x:x + ww][cm] = 255
        full = cv2.dilate(full, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        overlap = np.count_nonzero((full > 0) & (out > 0)) / max(np.count_nonzero(full), 1)
        if overlap >= 0.80:
            continue
        ys, xs = np.nonzero(full)
        bx0, bx1 = int(xs.min()), int(xs.max()) + 1
        by0, by1 = int(ys.min()), int(ys.max()) + 1
        local = full[by0:by1, bx0:bx1]
        p90d = float(np.percentile(vals_d, 90))
        p90c = float(np.percentile(vals_c, 90))
        new.append(DefectComponent(
            bbox_px=[bx0, by0, bx1 - bx0, by1 - by0],
            polygon_px=_component_polygon(local, bx0, by0),
            area_px=int(np.count_nonzero(full)),
            score=round(max(14.0, min(55.0, 8.0 + 4.0 * p90d + 2.0 * p90c)), 3),
            mean_score=round(max(p90d, p90c), 3),
            seam_overlap_frac=0.0,
            reason="kept_compact_border_speck",
        ))
        out[full > 0] = 255
    new.sort(key=lambda d: d.score, reverse=True)
    return new, out

def append_edge_material_components(
    img_bgr: np.ndarray,
    comps: list[DefectComponent],
    kept_mask: np.ndarray,
) -> tuple[list[DefectComponent], np.ndarray]:
    """Recover real defects clipped by the device/crop border.

    The regular detector is intentionally harsh near borders because the gold
    frame and the last few vertical fingers otherwise create false positives.
    This pass uses a dark/brown *seed* near the border, not bright frame pixels,
    so ordinary frame material stays ignored while bottom/right edge defects are
    recovered.  Edge detections are expanded inward/to the crop edge because the
    visible delamination halo is often much larger than the high-contrast core.
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    edge_margin = max(55, int(round(0.085 * min(h, w))))
    edge_zone = np.zeros((h, w), dtype=bool)
    edge_zone[:edge_margin, :] = True
    edge_zone[-edge_margin:, :] = True
    edge_zone[:, :edge_margin] = True
    edge_zone[:, -edge_margin:] = True

    # Avoid the ordinary top frame and absolute crop edge.  Bottom/right defects
    # are allowed through because they are common real failure modes.
    valid = edge_zone.copy()
    valid[:max(20, int(0.028 * h)), :] = False
    valid[:, :4] = False
    valid[:, -2:] = False
    valid[-2:, :] = False

    # Seeds are dark or muted brown/gold material.  Bright yellow frame/fingers
    # are deliberately not seeds; they may only be included later as halo after
    # a real seed exists.
    seed = (
        ((V < 118) & (S > 4))
        | ((H >= 5) & (H <= 62) & (S > 32) & (V > 30) & (V < 178))
        | ((B > 145) & (L < 172) & (S > 18))
    ) & valid

    seed_u8 = seed.astype(np.uint8) * 255
    seed_u8 = cv2.morphologyEx(seed_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    seed_u8 = cv2.morphologyEx(seed_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(seed_u8, connectivity=8)

    new = list(comps)
    out_mask = kept_mask.copy()
    for i in range(1, n):
        x, y, ww, hh, area = stats[i].tolist()
        if area < 8:
            continue
        aspect = max(ww / max(hh, 1), hh / max(ww, 1))
        # Reject normal frame strips and vertical finger remnants.
        if hh > 0.35 * h and ww < 0.055 * w:
            continue
        if ww > 0.35 * w and hh < 0.055 * h:
            continue
        if aspect > 15.0 and area < 0.005 * h * w:
            continue

        comp = labels[y:y + hh, x:x + ww] == i
        vals_v = V[y:y + hh, x:x + ww][comp]
        vals_s = S[y:y + hh, x:x + ww][comp]
        if vals_v.size == 0:
            continue
        if float(np.min(vals_v)) > 145 and float(np.max(vals_s)) < 28:
            continue

        near_left = x <= edge_margin
        near_right = (x + ww) >= w - edge_margin
        near_bottom = (y + hh) >= h - edge_margin
        near_top = y <= edge_margin
        if not (near_left or near_right or near_bottom or near_top):
            continue

        # Bounded expansion: enough to cover the visible halo, not enough to eat
        # the whole cell.  Right-edge defects need a wider inward allowance.
        inward = max(42, int(round(0.075 * min(h, w))))
        side_pad = max(14, int(round(0.024 * min(h, w))))
        if near_right:
            rx0 = max(0, x - inward)
            rx1 = w
            ry0 = max(0, y - side_pad)
            ry1 = min(h, y + hh + side_pad)
        elif near_left:
            rx0 = 0
            rx1 = min(w, x + ww + inward)
            ry0 = max(0, y - side_pad)
            ry1 = min(h, y + hh + side_pad)
        elif near_bottom:
            rx0 = max(0, x - side_pad)
            rx1 = min(w, x + ww + side_pad)
            ry0 = max(0, y - max(28, int(round(0.055 * min(h, w)))))
            ry1 = h
        else:  # near_top, but top frame was suppressed so this is rare.
            rx0 = max(0, x - side_pad)
            rx1 = min(w, x + ww + side_pad)
            ry0 = 0
            ry1 = min(h, y + hh + max(36, inward // 2))

        rww, rhh = rx1 - rx0, ry1 - ry0
        if rww <= 0 or rhh <= 0:
            continue
        if rww * rhh > 0.10 * h * w:
            continue
        # Avoid duplicates with an existing component.
        if np.count_nonzero(out_mask[ry0:ry1, rx0:rx1] > 0) > 0.55 * rww * rhh:
            continue

        full = np.ones((rhh, rww), dtype=np.uint8) * 255
        poly = _component_polygon(full, x=rx0, y=ry0)
        score_val = 34.0 if (near_right or near_bottom) else 22.0
        new.append(DefectComponent(
            bbox_px=[int(rx0), int(ry0), int(rww), int(rhh)],
            polygon_px=poly,
            area_px=int(rww * rhh),
            score=round(score_val, 3),
            mean_score=round(score_val, 3),
            seam_overlap_frac=0.0,
            reason="kept_edge_material",
        ))
        out_mask[ry0:ry1, rx0:rx1] = np.maximum(out_mask[ry0:ry1, rx0:rx1], full)

    new.sort(key=lambda d: d.score, reverse=True)
    return new, out_mask

def refine_edge_component_coverage(
    img_bgr: np.ndarray,
    comps: list[DefectComponent],
    kept_mask: np.ndarray,
) -> tuple[list[DefectComponent], np.ndarray]:
    """Expand already-detected edge defects to cover their full visible extent.

    This avoids the common failure where only the black/gold core of a border
    defect is boxed while the gray/yellow delamination halo connected to it is
    left outside the subtraction polygon.  It only modifies components already
    detected near an edge, so it does not create a new false-positive pathway.
    """
    if not comps:
        return comps, kept_mask
    h, w = img_bgr.shape[:2]
    edge_margin = max(55, int(round(0.085 * min(h, w))))
    valid = np.ones((h, w), dtype=bool)
    pad = 3
    valid[:pad, :] = valid[-pad:, :] = valid[:, :pad] = valid[:, -pad:] = False
    feats = _edge_local_features(img_bgr, valid)
    H, S, V = feats["H"], feats["S"], feats["V"]
    dark, chroma, edge, dL = feats["dark"], feats["chroma"], feats["edge"], feats["dL"]
    yellow_brown = (H >= 7) & (H <= 58) & (S > 24) & (V > 38)
    very_dark = (V < 92) & (S > 3)
    weak = (yellow_brown | very_dark | ((dark > 1.05) & (V < 205)) | ((chroma > 1.55) & (S > 5)) | ((edge > 3.2) & (V < 205)) | ((dL < -5.0) & (V < 205))) & valid
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    new_comps: list[DefectComponent] = []
    new_mask = kept_mask.copy()
    for comp_obj in comps:
        x, y, ww, hh = [int(v) for v in comp_obj.bbox_px]
        near_left = x <= edge_margin
        near_top = y <= edge_margin
        near_right = (x + ww) >= w - edge_margin
        near_bottom = (y + hh) >= h - edge_margin
        if not (near_left or near_top or near_right or near_bottom):
            new_comps.append(comp_obj)
            continue
        # Only expand plausible material defects, not skinny seam/frame ticks.
        crop_mask = kept_mask[y:y + hh, x:x + ww] > 0
        if np.count_nonzero(crop_mask) == 0:
            new_comps.append(comp_obj)
            continue
        margin = max(28, int(round(0.050 * min(h, w))))
        gx0, gy0 = max(0, x - margin), max(0, y - margin)
        gx1, gy1 = min(w, x + ww + margin), min(h, y + hh + margin)
        base = np.zeros((gy1 - gy0, gx1 - gx0), dtype=np.uint8)
        base[y - gy0:y - gy0 + hh, x - gx0:x - gx0 + ww] = (crop_mask.astype(np.uint8) * 255)
        weak_win = weak[gy0:gy1, gx0:gx1].astype(np.uint8) * 255
        grown = base.copy()
        for _ in range(7):
            nxt = cv2.bitwise_and(cv2.dilate(grown, kernel, iterations=1), weak_win)
            nxt = cv2.bitwise_or(nxt, base)
            if np.array_equal(nxt, grown):
                break
            grown = nxt
        ys, xs = np.nonzero(grown)
        if xs.size == 0:
            new_comps.append(comp_obj)
            continue
        rx0, ry0 = int(xs.min()) + gx0, int(ys.min()) + gy0
        rx1, ry1 = int(xs.max()) + gx0 + 1, int(ys.max()) + gy0 + 1
        # If the original component is essentially clipped by the crop/device
        # edge, extend the box to the image boundary for review/subtraction.
        if near_right and rx1 >= w - edge_margin // 2:
            rx1 = w
        if near_left and rx0 <= edge_margin // 2:
            rx0 = 0
        if near_bottom and ry1 >= h - edge_margin // 2:
            ry1 = h
        if near_top and ry0 <= edge_margin // 2:
            ry0 = 0

        # Large material defects clipped by an edge usually have a faint gray/blue
        # delamination halo that is visually obvious but below the strict weak
        # mask.  Expand the review/subtraction box inward a bounded amount so the
        # whole border defect is covered instead of just its black/gold core.
        orig_box_area = max(1, ww * hh)
        strong_edge_material = (orig_box_area >= 900) or (float(comp_obj.score) >= 18.0)
        if strong_edge_material:
            inward = max(42, int(round(0.070 * min(h, w))))
            outward_y = max(14, int(round(0.022 * min(h, w))))
            outward_x = max(14, int(round(0.022 * min(h, w))))
            if near_right:
                rx0 = max(0, min(rx0, x - inward))
                rx1 = w
                ry0 = max(0, min(ry0, y - outward_y))
                ry1 = min(h, max(ry1, y + hh + outward_y))
            if near_left:
                rx0 = 0
                rx1 = min(w, max(rx1, x + ww + inward))
                ry0 = max(0, min(ry0, y - outward_y))
                ry1 = min(h, max(ry1, y + hh + outward_y))
            if near_bottom:
                ry0 = max(0, min(ry0, y - inward // 2))
                ry1 = h
                rx0 = max(0, min(rx0, x - outward_x))
                rx1 = min(w, max(rx1, x + ww + outward_x))
            if near_top:
                ry0 = 0
                ry1 = min(h, max(ry1, y + hh + inward // 2))
                rx0 = max(0, min(rx0, x - outward_x))
                rx1 = min(w, max(rx1, x + ww + outward_x))

        rww, rhh = rx1 - rx0, ry1 - ry0
        # Reject pathological expansion into a whole frame strip.
        if (rww > 0.55 * w and rhh < 0.08 * h) or (rhh > 0.55 * h and rww < 0.08 * w):
            new_comps.append(comp_obj)
            continue
        if rww * rhh > 0.08 * h * w and max(rww, rhh) > 0.45 * max(h, w):
            new_comps.append(comp_obj)
            continue
        if rww <= ww + 2 and rhh <= hh + 2:
            new_comps.append(comp_obj)
            continue
        full = np.zeros((rhh, rww), dtype=np.uint8)
        # Preserve the grown connected mask, plus a filled bbox for edge-clipped
        # defects so polygons are not undercut at the border.
        local_grown = grown[max(0, ry0 - gy0):max(0, ry0 - gy0) + rhh, max(0, rx0 - gx0):max(0, rx0 - gx0) + rww]
        if local_grown.shape == full.shape:
            full = np.maximum(full, local_grown)
        if near_right or near_left or near_bottom or near_top:
            # Slightly dilate/fill only the connected evidence by default.  For
            # strong clipped material defects, use the expanded rectangle because
            # the faint halo is visually real but often too weak to stay connected
            # in the evidence mask.
            if 'strong_edge_material' in locals() and strong_edge_material:
                full[:, :] = 255
            else:
                full = cv2.dilate(full, kernel, iterations=1)
        new_mask[ry0:ry1, rx0:rx1] = np.maximum(new_mask[ry0:ry1, rx0:rx1], full)
        comp_obj.bbox_px = [int(rx0), int(ry0), int(rww), int(rhh)]
        comp_obj.polygon_px = _mask_to_polygon(full, x=rx0, y=ry0)
        comp_obj.area_px = int(max(comp_obj.area_px, np.count_nonzero(full)))
        comp_obj.reason = (comp_obj.reason or "kept") + "+edge_grown"
        new_comps.append(comp_obj)
    return new_comps, new_mask



def refine_connected_edge_extensions(
    img_bgr: np.ndarray,
    comps: list[DefectComponent],
    kept_mask: np.ndarray,
    parts: dict[str, np.ndarray],
) -> tuple[list[DefectComponent], np.ndarray]:
    """Follow thin material that is physically connected to an edge defect.

    This is deliberately seed-constrained: it never proposes a new defect.  It
    starts from an already-detected component near a crop edge and performs a
    bounded geodesic growth through dark/chromatic/transverse evidence.  It is
    meant for hair-like contamination, clipped sheet protrusions, and faint edge
    halos that are connected to a strong core but extend far beyond its bbox.
    """
    if not comps:
        return comps, kept_mask
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    dark_local = parts.get("dark_local", np.zeros((h, w), dtype=np.float32))
    chroma_local = parts.get("chroma_local", np.zeros((h, w), dtype=np.float32))
    transverse = parts.get("transverse_score", np.zeros((h, w), dtype=np.float32))
    large_score = parts.get("large_score", np.zeros((h, w), dtype=np.float32))
    force = parts.get("force_mask", np.zeros((h, w), dtype=np.float32)) > 0

    edge_margin = max(48, int(round(0.080 * min(h, w))))
    bright_frame = (H >= 13) & (H <= 48) & (S > 38) & (V > 185)

    # Two extension graphs are used.  Thin/medium edge defects grow only through
    # line-like y-gradient evidence, while catastrophic sheets may use the broader
    # material graph.  This prevents a hair/fiber from turning the area between it
    # and the crop border into a filled wedge.
    L32 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    gy_abs = np.abs(cv2.Sobel(L32, cv2.CV_32F, 0, 1, ksize=3))
    grad_valid = np.ones((h, w), dtype=bool)
    grad_valid[:3, :] = grad_valid[-3:, :] = False
    grad_valid[:, :3] = grad_valid[:, -3:] = False
    gy_z = robust_z(gy_abs, grad_valid, floor=2.0)
    line_allowed = (
        ((gy_z >= 2.45) & (dark_local >= 0.48) & (V < 208))
        | ((dark_local >= 1.85) & (V < 192) & (gy_z >= 0.85))
        | ((transverse >= 2.55) & (dark_local >= 0.40) & (V < 190))
        | force
    )
    line_allowed &= (~bright_frame) | (dark_local >= 1.75) | (gy_z >= 3.6) | force

    broad_allowed = (
        ((dark_local >= 1.15) & (V < 218))
        | ((chroma_local >= 1.55) & (S > 5) & ((dark_local >= 0.45) | (transverse >= 1.8)))
        | ((transverse >= 2.05) & ((dark_local >= 0.35) | (chroma_local >= 0.65) | (V < 175)))
        | ((large_score >= 2.0) & (S > 8) & (V < 225) & ((dark_local >= 0.25) | (transverse >= 1.4)))
        | force
    )
    broad_allowed &= (~bright_frame) | (dark_local >= 1.55) | (transverse >= 2.35) | force

    def prep_graph(a: np.ndarray) -> np.ndarray:
        a = a.copy()
        a[0, :] = a[-1, :] = False
        a[:, 0] = a[:, -1] = False
        u8 = cv2.morphologyEx(
            a.astype(np.uint8) * 255, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        vlen = odd_kernel(max(25, int(round(0.050 * h))))
        long_vertical = cv2.morphologyEx(
            u8, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, vlen)),
        ) > 0
        preserve_vertical = (transverse >= 3.6) | (dark_local >= 2.1) | force
        u8[long_vertical & ~preserve_vertical] = 0
        return u8

    line_allowed_u8 = prep_graph(line_allowed)
    broad_allowed_u8 = prep_graph(broad_allowed)

    def obj_mask(d: DefectComponent) -> np.ndarray:
        m = np.zeros((h, w), dtype=np.uint8)
        if d.polygon_px and len(d.polygon_px) >= 3:
            pts = np.asarray(d.polygon_px, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(m, [pts], 255)
        else:
            x, y, ww, hh = [int(round(v)) for v in d.bbox_px[:4]]
            m[max(0, y):min(h, y + hh), max(0, x):min(w, x + ww)] = 255
        return m

    new: list[DefectComponent] = []
    out_mask = kept_mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for d in comps:
        x, y, ww, hh = [int(round(v)) for v in d.bbox_px[:4]]
        near_edge = x <= edge_margin or y <= edge_margin or (x + ww) >= w - edge_margin or (y + hh) >= h - edge_margin
        substantial = ww * hh >= 280 or float(d.score) >= 18.0
        if not near_edge or not substantial:
            new.append(d)
            continue

        base_full = obj_mask(d)
        if not np.any(base_full):
            new.append(d)
            continue
        near_left = x <= edge_margin
        near_top = y <= edge_margin
        near_right = (x + ww) >= w - edge_margin
        near_bottom = (y + hh) >= h - edge_margin
        if not near_top:
            # The regression case requiring long connected growth is a top-edge
            # fiber.  Bottom/side growth tended to follow frame remnants; compact
            # side specks are handled by append_compact_border_specks instead.
            new.append(d)
            continue
        margin_x = max(52, int(round(0.15 * min(h, w))))
        margin_y = max(38, int(round(0.10 * min(h, w))))
        gx0, gx1 = max(0, x - margin_x), min(w, x + ww + margin_x)
        gy0, gy1 = max(0, y - margin_y), min(h, y + hh + margin_y)
        # Directional cap: a top-clipped contaminant may extend far sideways, but
        # should not drag vertical fingers deep into the cell.  Analogous caps are
        # used for the other edges.
        cap = max(16, int(round(0.030 * min(h, w))))
        if near_top:
            gy0 = 0
            gy1 = min(gy1, y + hh + cap)
        if near_bottom:
            gy0 = max(gy0, y - cap)
            gy1 = h
        if near_left and not (near_top or near_bottom):
            gx0 = 0
            gx1 = min(gx1, x + ww + cap)
        if near_right and not (near_top or near_bottom):
            gx0 = max(gx0, x - cap)
            gx1 = w
        base = base_full[gy0:gy1, gx0:gx1]
        reason_text = str(d.reason or "").lower()
        large_parent = (
            ww * hh >= 0.025 * h * w
            or int(d.area_px) >= 0.012 * h * w
            or "large_material_density" in reason_text
            or "clustered_diffuse" in reason_text
        )
        if large_parent:
            # Catastrophic sheets are already handled by the broad material
            # reconstruction.  Emitting dozens of edge-extension satellites around
            # their perimeter is redundant and makes review unusable.
            new.append(d)
            continue
        allow = line_allowed_u8[gy0:gy1, gx0:gx1]
        seed = cv2.bitwise_and(base, cv2.bitwise_or(allow, base))
        grown = seed.copy()
        # Bounded geodesic distance.  Enough to follow the annotated top-edge
        # fibers, but not enough to wander across the full cell.
        max_iter = max(18, int(round(0.115 * min(h, w))))
        for _ in range(max_iter):
            nxt = cv2.bitwise_and(cv2.dilate(grown, kernel, iterations=1), allow)
            nxt = cv2.bitwise_or(nxt, seed)
            if np.array_equal(nxt, grown):
                break
            grown = nxt
        added = (grown > 0) & (base == 0)
        added_px = int(np.count_nonzero(added))
        if added_px < 4:
            new.append(d)
            continue

        # Require the extension to be physically modest relative to the existing
        # component.  This rejects accidental contact with a broad seam/frame.
        base_px = int(np.count_nonzero(base))
        if added_px > max(1800, 1.8 * base_px):
            new.append(d)
            continue
        combined = cv2.bitwise_or(base, grown)
        # Remove isolated growth islands that do not remain connected to the seed.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
        seed_labels = set(int(v) for v in np.unique(labels[base > 0]) if int(v) > 0)
        keep_local = np.zeros_like(combined)
        for li in seed_labels:
            keep_local[labels == li] = 255
        if np.count_nonzero(keep_local) <= base_px + 3:
            new.append(d)
            continue

        # Keep the original defect geometry intact.  Emit only the newly reached
        # material as one or more tight satellite polygons; merging a thin branch
        # into the parent outer contour can fill a large clean wedge.
        new.append(d)
        base_guard = cv2.dilate(base, kernel, iterations=1) > 0
        ext_local = (keep_local > 0) & (~base_guard)
        ext_u8 = ext_local.astype(np.uint8) * 255
        en, elabels, estats, _ = cv2.connectedComponentsWithStats(ext_u8, connectivity=8)
        for ei in range(1, en):
            ex, ey, ew, eh, earea = estats[ei].tolist()
            if earea < 4:
                continue
            em = elabels[ey:ey + eh, ex:ex + ew] == ei
            full_ext = np.zeros((h, w), dtype=np.uint8)
            full_ext[gy0 + ey:gy0 + ey + eh, gx0 + ex:gx0 + ex + ew][em] = 255
            # One-pixel halo makes the path robust in GDS without becoming a box.
            full_ext = cv2.dilate(full_ext, kernel, iterations=1)
            # Remove any renewed overlap with the parent; the two polygons may
            # touch, but neither should contain the other.
            full_ext[base_full > 0] = 0
            fys, fxs = np.nonzero(full_ext)
            if fxs.size < 4:
                continue
            rx0, rx1 = int(fxs.min()), int(fxs.max()) + 1
            ry0, ry1 = int(fys.min()), int(fys.max()) + 1
            loc = full_ext[ry0:ry1, rx0:rx1]
            vals = np.maximum.reduce([
                dark_local[full_ext > 0],
                chroma_local[full_ext > 0],
                transverse[full_ext > 0],
                large_score[full_ext > 0],
            ])
            peak = float(np.max(vals)) if vals.size else 13.0
            mean = float(np.mean(vals)) if vals.size else 0.0
            new.append(DefectComponent(
                bbox_px=[rx0, ry0, rx1 - rx0, ry1 - ry0],
                polygon_px=_component_polygon(loc, rx0, ry0),
                area_px=int(np.count_nonzero(full_ext)),
                score=round(max(13.0, min(60.0, 6.0 + 4.0 * peak)), 3),
                mean_score=round(mean, 3),
                seam_overlap_frac=0.0,
                reason="kept_edge_extension_fragment",
            ))
            out_mask[full_ext > 0] = 255
    new.sort(key=lambda c: c.score, reverse=True)
    return new, out_mask

def append_tiny_dark_components(
    img_bgr: np.ndarray,
    comps: list[DefectComponent],
    kept_mask: np.ndarray,
    interior_mask: np.ndarray,
) -> tuple[list[DefectComponent], np.ndarray]:
    """Add high-confidence microscopic dark/brown defects without using global threshold.

    This is deliberately stricter than the experimental broad tiny detector: it
    only accepts compact components with strong dark/color evidence after a
    vertical-local background subtraction.  It does not grow into weak halos, so
    it cannot create the huge horizontal-band boxes that the residual score can.
    """
    h, w = img_bgr.shape[:2]
    # If the main detector and edge-material pass found nothing, only enable the
    # aggressive tiny-defect pass when there is at least one genuinely dark local
    # particle in the interior.  This keeps the clean reference cells from turning
    # into dozens of boxes while still recovering cells like Wafer_A_cell_2-5.
    if not comps:
        hsv_probe = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        V_probe = hsv_probe[:, :, 2]
        S_probe = hsv_probe[:, :, 1]
        probe = interior_mask.copy()
        pp = max(18, int(round(0.025 * min(h, w))))
        probe[:pp, :] = probe[-pp:, :] = probe[:, :pp] = probe[:, -pp:] = False
        dark_pixels = (V_probe < 105) & (S_probe > 3) & probe
        if np.count_nonzero(dark_pixels) < 2:
            return comps, kept_mask
    valid = interior_mask.copy()
    # Let tiny defects near the device border through, but still suppress the
    # absolute crop edge.  Border blobs are handled by edge refinement above.
    pad = max(8, int(round(0.010 * min(h, w))))
    valid[:pad, :] = valid[-pad:, :] = valid[:, :pad] = valid[:, -pad:] = False
    valid &= kept_mask == 0
    feats = _edge_local_features(img_bgr, valid)
    H, S, V = feats["H"], feats["S"], feats["V"]
    dark, chroma, edge, dL = feats["dark"], feats["chroma"], feats["edge"], feats["dL"]
    yellow_brown = (H >= 7) & (H <= 58) & (S > 32) & (V > 35) & (V < 165)
    seed = (
        ((dark >= 4.65) & (V < 158))
        | ((V < 72) & (S > 3))
        | (yellow_brown & ((dark >= 1.3) | (edge >= 3.5)))
        | ((edge >= 7.2) & (V < 155) & ((dark >= 2.2) | (chroma >= 3.4)))
        | ((dL < -16.0) & (V < 150))
    ) & valid
    seed_u8_pre = seed.astype(np.uint8) * 255
    # Crowded rows are the tell-tale sign of periodic stitch/finger residuals.
    # Remove those rows before connected components; otherwise hundreds of
    # row-ripple ticks become tiny boxes in clean cells.  True particles on a
    # crowded row survive only if they are genuinely dark/strongly chromatic.
    row_count_pre = np.count_nonzero(seed_u8_pre > 0, axis=1).astype(np.float32)
    row_density_pre = cv2.GaussianBlur(row_count_pre[:, None], (0, 0), sigmaX=0.01, sigmaY=2.0).reshape(-1)
    crowded_rows = row_density_pre > max(7.0, 0.010 * w)
    if np.any(crowded_rows):
        strong_on_crowded = ((V < 66) | ((dark > 7.2) & (V < 105)) | ((chroma > 7.0) & (S > 18) & (V < 145)))
        seed[crowded_rows, :] &= strong_on_crowded[crowded_rows, :]

    seed_u8 = seed.astype(np.uint8) * 255
    seed_u8 = cv2.morphologyEx(seed_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(seed_u8, connectivity=8)
    row_count = np.count_nonzero(seed_u8 > 0, axis=1).astype(np.float32)
    row_density = cv2.GaussianBlur(row_count[:, None], (0, 0), sigmaX=0.01, sigmaY=2.0).reshape(-1)
    new = list(comps)
    out_mask = kept_mask.copy()
    for i in range(1, n):
        x, y, ww, hh, area = stats[i].tolist()
        if area < 1 or area > 260:
            continue
        comp = labels[y:y + hh, x:x + ww] == i
        aspect = max(ww / max(hh, 1), hh / max(ww, 1))
        fill = area / max(ww * hh, 1)
        vals_d = dark[y:y + hh, x:x + ww][comp]
        vals_c = chroma[y:y + hh, x:x + ww][comp]
        vals_e = edge[y:y + hh, x:x + ww][comp]
        vals_v = V[y:y + hh, x:x + ww][comp]
        if vals_d.size == 0:
            continue
        dpk = float(np.max(vals_d)); cpk = float(np.max(vals_c)); epk = float(np.max(vals_e)); minv = float(np.min(vals_v))
        if area <= 2 and minv > 66 and dpk < 6.1 and cpk < 6.2:
            continue
        if aspect > 11.0 and fill < 0.55 and dpk < 6.4:
            continue
        if hh >= max(20, 3 * ww) and ww <= 5 and dpk < 6.8 and cpk < 6.5:
            continue
        rd = float(np.max(row_density[max(0, y - 3):min(h, y + hh + 4)]))
        if rd > max(14.0, 0.020 * w) and minv > 72 and dpk < 6.3 and cpk < 6.5:
            continue
        if not (minv < 92 or dpk >= 5.0 or cpk >= 5.4 or (epk >= 8.0 and dpk >= 2.6)):
            continue
        halo = 3 if max(ww, hh) <= 9 else 2
        bx0, by0 = max(0, x - halo), max(0, y - halo)
        bx1, by1 = min(w, x + ww + halo), min(h, y + hh + halo)
        if np.any(out_mask[by0:by1, bx0:bx1] > 0):
            continue
        full = np.zeros((by1 - by0, bx1 - bx0), dtype=np.uint8)
        full[y - by0:y - by0 + hh, x - bx0:x - bx0 + ww][comp] = 255
        full = cv2.dilate(full, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        poly = _component_polygon(full, x=bx0, y=by0)
        score_val = max(9.0, dpk + 4.0, cpk + 3.5, min(18.0, epk + 2.5))
        new.append(DefectComponent(
            bbox_px=[int(bx0), int(by0), int(bx1 - bx0), int(by1 - by0)],
            polygon_px=poly,
            area_px=int(np.count_nonzero(full)),
            score=round(float(score_val), 3),
            mean_score=round(float(score_val), 3),
            seam_overlap_frac=0.0,
            reason="kept_tiny_dark",
        ))
        out_mask[by0:by1, bx0:bx1] = np.maximum(out_mask[by0:by1, bx0:bx1], full)

    # If the cell already contains a major border defect, do not flood the UI
    # with every weak microscopic seed elsewhere.  The user's pain point on
    # those cells is edge coverage; tiny recall is capped to only the strongest
    # particles so review stays sane.
    def _near_edge_box(d: DefectComponent) -> bool:
        x0, y0, ww0, hh0 = [int(v) for v in d.bbox_px]
        em = max(55, int(round(0.085 * min(h, w))))
        return x0 <= em or y0 <= em or (x0 + ww0) >= w - em or (y0 + hh0) >= h - em

    has_major_edge = any(
        (not str(d.reason).startswith("kept_tiny_dark"))
        and _near_edge_box(d)
        and (int(d.area_px) >= 350 or (int(d.bbox_px[2]) * int(d.bbox_px[3]) >= 1200))
        for d in new
    )
    has_major_defect = any(
        (not str(d.reason).startswith("kept_tiny_dark"))
        and (int(d.area_px) >= 350 or (int(d.bbox_px[2]) * int(d.bbox_px[3]) >= 1200))
        for d in new
    )
    if has_major_edge or has_major_defect:
        base = [d for d in new if not str(d.reason).startswith("kept_tiny_dark")]
        tiny = [d for d in new if str(d.reason).startswith("kept_tiny_dark") and float(d.score) >= (12.5 if has_major_edge else 13.5)]
        tiny.sort(key=lambda d: d.score, reverse=True)
        new = base + tiny[:(8 if has_major_edge else 10)]
        rebuilt = np.zeros_like(out_mask)
        for d in new:
            if d.polygon_px and len(d.polygon_px) >= 3:
                pts = np.asarray(d.polygon_px, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(rebuilt, [pts], 255)
            else:
                x0, y0, ww0, hh0 = [int(v) for v in d.bbox_px]
                rebuilt[y0:y0 + hh0, x0:x0 + ww0] = 255
        out_mask = rebuilt

    new.sort(key=lambda d: d.score, reverse=True)
    return new, out_mask



def append_salient_spot_components(
    img_bgr: np.ndarray,
    comps: list[DefectComponent],
    kept_mask: np.ndarray,
    interior_mask: np.ndarray,
) -> tuple[list[DefectComponent], np.ndarray]:
    """Add a small number of high-salience missed specks/scratches.

    The residual-map detector is intentionally conservative because the normal
    finger pattern creates a high periodic floor.  This pass does not threshold
    every pixel.  It finds only local maxima of a vertical-local salience map,
    so isolated particles and short scratches can be recovered without turning
    every stripe/ripple row into boxes.
    """
    h, w = img_bgr.shape[:2]
    # Keep this away from the ordinary device frame.  Edge-clipped material is
    # handled by refine_edge_component_coverage().
    valid = interior_mask.copy()
    pad = max(62, int(round(0.090 * min(h, w))))
    valid[:pad, :] = False
    valid[-pad:, :] = False
    valid[:, :pad] = False
    valid[:, -pad:] = False
    has_major_defect = any(
        (not str(d.reason).startswith("kept_tiny_dark"))
        and (int(d.area_px) >= 350 or (int(d.bbox_px[2]) * int(d.bbox_px[3]) >= 1200))
        for d in comps
    )
    if np.count_nonzero(valid) < 100 or len(comps) < 6:
        return comps, kept_mask

    feats = _edge_local_features(img_bgr, valid)
    H, S, V = feats["H"], feats["S"], feats["V"]
    dark, chroma, edge, dL = feats["dark"], feats["chroma"], feats["edge"], feats["dL"]

    # Local salience.  The column median subtraction knocks down persistent
    # finger/stripe columns while preserving localized spots and short scratches.
    sal = np.maximum.reduce([
        dark + 0.15 * edge,
        chroma + 0.10 * edge,
        0.65 * edge + 0.30 * dark + 0.25 * chroma,
        np.maximum(0.0, -dL) / 4.0,
    ]).astype(np.float32)
    sal[~valid] = 0.0
    with np.errstate(all="ignore"):
        col = np.nanmedian(np.where(valid, sal, np.nan), axis=0)
    med = float(np.nanmedian(col)) if np.any(np.isfinite(col)) else 0.0
    col = np.where(np.isfinite(col), col, med).astype(np.float32)
    sal = sal - 0.75 * col[None, :]
    sal[~valid] = 0.0

    local_max = sal >= (cv2.dilate(sal, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))) - 1e-3)
    gate = (
        ((dark >= 3.5) & (V < 170))
        | ((chroma >= 4.2) & (S > 6) & (V < 190))
        | ((edge >= 8.2) & (V < 175) & ((dark >= 1.3) | (chroma >= 2.7) | (dL < -5.0)))
        | ((dL < -12.0) & (V < 185))
    ) & local_max & valid & (sal >= (9.2 if has_major_defect else 8.5))

    # If a whole stitch/illumination row starts producing maxima, keep only the
    # genuinely strong/dark points on that row.
    row_count = np.count_nonzero(gate, axis=1).astype(np.float32)
    row_density = cv2.GaussianBlur(row_count[:, None], (0, 0), sigmaX=0.01, sigmaY=2.0).reshape(-1)
    crowded = row_density > max(3.0, 0.004 * w)
    if np.any(crowded):
        strong = (V < 120) | (dark > 5.4) | (chroma > 5.6) | ((edge > 10.0) & (dark > 1.6))
        gate[crowded, :] &= strong[crowded, :]

    ys, xs = np.nonzero(gate)
    if xs.size == 0:
        return comps, kept_mask
    candidates = sorted([(float(sal[y, x]), int(x), int(y)) for x, y in zip(xs, ys)], reverse=True)
    new = list(comps)
    out_mask = kept_mask.copy()
    occupied = (out_mask > 0).astype(np.uint8) * 255
    added = 0
    for score_val, x, y in candidates:
        radius = 8
        x0, y0 = max(0, x - radius), max(0, y - radius)
        x1, y1 = min(w, x + radius + 1), min(h, y + radius + 1)
        if np.any(occupied[y0:y1, x0:x1] > 0):
            continue
        # Small review/subtraction halo; these are all tiny candidates.
        full = np.ones((y1 - y0, x1 - x0), dtype=np.uint8) * 255
        poly = _component_polygon(full, x=x0, y=y0)
        new.append(DefectComponent(
            bbox_px=[int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
            polygon_px=poly,
            area_px=int(np.count_nonzero(full)),
            score=round(max(9.0, min(38.0, score_val + 3.0)), 3),
            mean_score=round(max(9.0, min(38.0, score_val + 3.0)), 3),
            seam_overlap_frac=0.0,
            reason="kept_salient_spot",
        ))
        out_mask[y0:y1, x0:x1] = np.maximum(out_mask[y0:y1, x0:x1], full)
        occupied[y0:y1, x0:x1] = 255
        added += 1
        if added >= (10 if has_major_defect else 24):
            break
    new.sort(key=lambda d: d.score, reverse=True)
    return new, out_mask



def append_sparse_particle_components(
    img_bgr: np.ndarray,
    comps: list[DefectComponent],
    kept_mask: np.ndarray,
) -> tuple[list[DefectComponent], np.ndarray]:
    """Recover isolated faint/dark particles that the interior mask misses.

    This pass is intentionally gated.  It can run on clean-looking cells only
    when there is very strong sparse-particle evidence, so clean references stay
    quiet, but cells with obvious missed dust/holes get small review boxes.
    """
    h, w = img_bgr.shape[:2]
    # Use a simple active-area mask instead of make_interior_mask(): the latter
    # can exclude exactly the dark particles we need to recover.
    valid = np.ones((h, w), dtype=bool)
    side_pad = max(18, int(round(0.024 * min(h, w))))
    top_pad = max(52, int(round(0.072 * h)))
    bottom_pad = max(28, int(round(0.040 * h)))
    valid[:top_pad, :] = False
    valid[-bottom_pad:, :] = False
    valid[:, :side_pad] = False
    valid[:, -side_pad:] = False

    if np.count_nonzero(valid) < 100:
        return comps, kept_mask

    feats = _edge_local_features(img_bgr, valid)
    H, S, V = feats["H"], feats["S"], feats["V"]
    dark, chroma, edge, dL = feats["dark"], feats["chroma"], feats["edge"], feats["dL"]

    sal = np.maximum.reduce([
        dark + 0.16 * edge,
        chroma + 0.10 * edge,
        0.68 * edge + 0.35 * dark + 0.28 * chroma,
        np.maximum(0.0, -dL) / 3.8,
    ]).astype(np.float32)
    sal[~valid] = 0.0
    with np.errstate(all="ignore"):
        col = np.nanmedian(np.where(valid, sal, np.nan), axis=0)
    med = float(np.nanmedian(col[np.isfinite(col)])) if np.any(np.isfinite(col)) else 0.0
    col = np.where(np.isfinite(col), col, med).astype(np.float32)
    sal = sal - 0.55 * col[None, :]
    sal[~valid] = 0.0

    local_max = sal >= (cv2.dilate(sal, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))) - 1e-3)
    evidence = (
        ((dark >= 8.5) & (V < 190))
        | ((chroma >= 10.5) & (S > 18) & (V < 205))
        | ((edge >= 13.5) & (V < 195) & ((dark >= 2.4) | (chroma >= 3.4) | (dL < -8.0)))
        | ((dL < -22.0) & (V < 190))
    )
    gate = local_max & valid & evidence & (sal >= 10.5)

    # Reject rows that look like periodic frame/finger junctions unless the point
    # is very strong. This is the main clean-reference guard.
    row_count = np.count_nonzero(gate, axis=1).astype(np.float32)
    row_density = cv2.GaussianBlur(row_count[:, None], (0, 0), sigmaX=0.01, sigmaY=2.0).reshape(-1)
    crowded = row_density > max(3.0, 0.004 * w)
    if np.any(crowded):
        very = (
            ((dark >= 15.0) & (V < 180))
            | ((edge >= 18.0) & (dark >= 5.0) & (V < 180))
            | ((chroma >= 18.0) & (S > 30) & (V < 190))
        )
        gate[crowded, :] &= very[crowded, :]

    ys, xs = np.nonzero(gate)
    if xs.size == 0:
        return comps, kept_mask

    raw = sorted(
        [(float(sal[y, x]), int(x), int(y), float(dark[y, x]), float(chroma[y, x]), float(edge[y, x]), int(V[y, x])) for y, x in zip(ys, xs)],
        reverse=True,
    )
    # NMS before gating/capping.
    selected: list[tuple[float, int, int, float, float, float, int]] = []
    for cand in raw:
        score_val, x, y, *_ = cand
        if any((x - sx) * (x - sx) + (y - sy) * (y - sy) < 26 * 26 for _, sx, sy, *_ in selected):
            continue
        selected.append(cand)
        if len(selected) >= 18:
            break
    if not selected:
        return comps, kept_mask

    # Clean-looking cells are only allowed into this sparse pass when the image has
    # genuinely strong evidence.  This preserves the previous 0/0/0 clean refs.
    has_existing = len(comps) > 0
    max_sparse = selected[0][0]
    strong_count = sum(1 for c in selected if c[0] >= 20.0)
    if not has_existing and max_sparse < 45.0:
        return comps, kept_mask

    min_keep = 10.5 if has_existing else 14.0
    max_add = 8 if has_existing else 12
    new = list(comps)
    out_mask = kept_mask.copy()
    occupied = (out_mask > 0).astype(np.uint8) * 255
    added = 0
    for score_val, x, y, dpk, cpk, epk, minv in selected:
        if score_val < min_keep:
            continue
        radius = 9 if score_val < 25.0 else 11
        x0, y0 = max(0, x - radius), max(0, y - radius)
        x1, y1 = min(w, x + radius + 1), min(h, y + radius + 1)
        if np.any(occupied[y0:y1, x0:x1] > 0):
            continue
        full = np.ones((y1 - y0, x1 - x0), dtype=np.uint8) * 255
        poly = _component_polygon(full, x=x0, y=y0)
        new.append(DefectComponent(
            bbox_px=[int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
            polygon_px=poly,
            area_px=int(np.count_nonzero(full)),
            score=round(float(max(9.0, min(60.0, score_val + 3.0))), 3),
            mean_score=round(float(max(9.0, min(60.0, score_val + 3.0))), 3),
            seam_overlap_frac=0.0,
            reason="kept_sparse_particle",
        ))
        out_mask[y0:y1, x0:x1] = np.maximum(out_mask[y0:y1, x0:x1], full)
        occupied[y0:y1, x0:x1] = 255
        added += 1
        if added >= max_add:
            break
    new.sort(key=lambda d: d.score, reverse=True)
    return new, out_mask



def _column_robust_z_map(arr: np.ndarray, valid: np.ndarray, floor: float = 0.8) -> np.ndarray:
    """Robustly normalize each x-column independently."""
    h, w = arr.shape
    out = np.zeros((h, w), dtype=np.float32)
    for x in range(w):
        m = valid[:, x]
        vals = arr[:, x][m]
        vals = vals[np.isfinite(vals)]
        if vals.size < 12:
            continue
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)) * 1.4826)
        if not np.isfinite(mad) or mad < floor:
            mad = float(floor)
        out[:, x] = (arr[:, x].astype(np.float32) - med) / mad
    return out


def _row_positive_outlier_map(arr: np.ndarray, valid: np.ndarray, floor: float = 1.0) -> np.ndarray:
    """Suppress row-wide stitch/illumination excursions from a positive score map."""
    h, w = arr.shape
    out = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        m = valid[y, :]
        vals = arr[y, :][m]
        vals = vals[np.isfinite(vals)]
        if vals.size < 16:
            continue
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)) * 1.4826)
        if not np.isfinite(mad) or mad < floor:
            mad = float(floor)
        out[y, :] = np.maximum(0.0, (arr[y, :].astype(np.float32) - med) / mad)
    return out


def compute_multiscale_micro_score(
    img_bgr: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vertical-context anomaly maps for dust, scratches, and faint smudges."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    valid = valid_mask.astype(bool).copy()
    responses: list[np.ndarray] = []
    for k in (9, 17, 31, 57, 97):
        if k >= h:
            continue
        kk = odd_kernel(k)
        z_channels = []
        for c, floor in ((0, 1.0), (1, 0.70), (2, 0.70)):
            bg = cv2.blur(lab[:, :, c], (1, kk), borderType=cv2.BORDER_REFLECT)
            residual = lab[:, :, c] - bg
            z_channels.append(_column_robust_z_map(residual, valid, floor=floor))
        zL, zA, zB = z_channels
        dark = np.maximum(0.0, -zL)
        bright = np.maximum(0.0, zL)
        chroma = np.sqrt(zA * zA + zB * zB)
        response = np.maximum.reduce([dark, 0.68 * bright, 0.92 * chroma]).astype(np.float32)
        response[~valid] = 0.0
        responses.append(response)
    if not responses:
        z = np.zeros((h, w), dtype=np.float32)
        return z, z, np.zeros((h, w), dtype=np.uint8)
    stack = np.stack(responses, axis=2)
    raw = np.max(stack, axis=2).astype(np.float32)
    support = np.sum(stack >= 3.0, axis=2).astype(np.uint8)
    row = _row_positive_outlier_map(raw, valid, floor=0.9)
    row = cv2.GaussianBlur(row, (0, 0), sigmaX=0.55, sigmaY=0.55)
    row[~valid] = 0.0
    return raw, row.astype(np.float32), support


def compute_periodic_neighbor_anomaly(
    img_bgr: np.ndarray,
    valid_mask: np.ndarray,
    pitch_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compare each pixel with the same phase in neighboring device fingers.

    Normal finger texture repeats horizontally.  Median comparison at ±1, ±2,
    and ±3 pitches cancels the regular pattern while retaining local defects,
    including faint marks that are not globally dark or strongly colored.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    shifted: list[np.ndarray] = []
    for mult in (1, 2, 3):
        d = max(1, int(round(float(pitch_px) * mult)))
        left = np.full_like(lab, np.nan, dtype=np.float32)
        right = np.full_like(lab, np.nan, dtype=np.float32)
        left[:, d:, :] = lab[:, :-d, :]
        right[:, :-d, :] = lab[:, d:, :]
        shifted.extend([left, right])
    expected = np.nanmedian(np.stack(shifted, axis=0), axis=0).astype(np.float32)
    residual = lab - expected
    valid = valid_mask.astype(bool) & np.isfinite(residual[:, :, 0])
    z_channels: list[np.ndarray] = []
    for c, floor in ((0, 1.25), (1, 0.75), (2, 0.75)):
        vals = residual[:, :, c][valid]
        med = float(np.median(vals)) if vals.size else 0.0
        mad = float(np.median(np.abs(vals - med)) * 1.4826) if vals.size else floor
        mad = max(float(floor), mad if np.isfinite(mad) else float(floor))
        z_channels.append((residual[:, :, c] - med) / mad)
    zL, zA, zB = z_channels
    raw = np.maximum.reduce([
        np.maximum(0.0, -zL),
        0.70 * np.maximum(0.0, zL),
        np.sqrt(zA * zA + zB * zB),
    ]).astype(np.float32)
    raw[~valid] = 0.0
    row = _row_positive_outlier_map(raw, valid, floor=0.8)
    row = cv2.GaussianBlur(row, (0, 0), sigmaX=0.55, sigmaY=0.55)
    row[~valid] = 0.0
    delta_e = np.linalg.norm(residual, axis=2).astype(np.float32)
    delta_e[~valid] = 0.0
    # Same-phase neighbors do not exist near the left/right image boundary.
    # Zero that guard band instead of allowing one-sided comparisons to create
    # a ladder of false edge candidates.
    edge_guard = min(w // 4, max(5, int(math.ceil(3.0 * float(pitch_px))) + 2))
    raw[:, :edge_guard] = 0.0; raw[:, -edge_guard:] = 0.0
    row[:, :edge_guard] = 0.0; row[:, -edge_guard:] = 0.0
    delta_e[:, :edge_guard] = 0.0; delta_e[:, -edge_guard:] = 0.0
    return raw, row.astype(np.float32), delta_e


def compute_column_residual_anomaly(
    residual_lab: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Find local outliers within each vertically persistent finger column."""
    valid = valid_mask.astype(bool)
    zL = _column_robust_z_map(residual_lab[:, :, 0], valid, floor=1.0)
    zA = _column_robust_z_map(residual_lab[:, :, 1], valid, floor=0.65)
    zB = _column_robust_z_map(residual_lab[:, :, 2], valid, floor=0.65)
    raw = np.maximum.reduce([
        np.maximum(0.0, -zL),
        0.70 * np.maximum(0.0, zL),
        0.95 * np.sqrt(zA * zA + zB * zB),
    ]).astype(np.float32)
    raw[~valid] = 0.0
    row = _row_positive_outlier_map(raw, valid, floor=0.8)
    row = cv2.GaussianBlur(row, (0, 0), sigmaX=0.45, sigmaY=0.45)
    row[~valid] = 0.0
    return raw, row.astype(np.float32)


def _nms_peaks(
    score_map: np.ndarray,
    gate: np.ndarray,
    radius: int,
    limit: int,
) -> list[tuple[float, int, int]]:
    ys, xs = np.nonzero(gate)
    if xs.size == 0:
        return []
    order = np.argsort(score_map[ys, xs])[::-1]
    radius = max(2, int(radius))
    cell = radius
    grid: dict[tuple[int, int], list[tuple[int, int]]] = {}
    selected: list[tuple[float, int, int]] = []
    for idx in order:
        x = int(xs[idx]); y = int(ys[idx]); val = float(score_map[y, x])
        gx, gy = x // cell, y // cell
        too_close = False
        for yy in range(gy - 1, gy + 2):
            for xx in range(gx - 1, gx + 2):
                for sx, sy in grid.get((xx, yy), []):
                    if (x - sx) ** 2 + (y - sy) ** 2 < radius * radius:
                        too_close = True
                        break
                if too_close:
                    break
            if too_close:
                break
        if too_close:
            continue
        selected.append((val, x, y))
        grid.setdefault((gx, gy), []).append((x, y))
        if len(selected) >= max(1, int(limit)):
            break
    return selected


def _spatially_stratified_peaks(
    score_map: np.ndarray,
    gate: np.ndarray,
    tile_size: int,
    per_tile: int,
    radius: int,
    limit: int,
) -> list[tuple[float, int, int]]:
    """Select peaks round-robin by spatial tile.

    The first candidate from every active tile is considered before any tile gets
    a second candidate.  This prevents one catastrophic defect from consuming the
    entire micro-candidate budget and hiding faint defects elsewhere.
    """
    h, w = score_map.shape
    tile_size = max(24, int(tile_size))
    tile_lists: list[list[tuple[float, int, int]]] = []
    for y0 in range(0, h, tile_size):
        for x0 in range(0, w, tile_size):
            y1 = min(h, y0 + tile_size); x1 = min(w, x0 + tile_size)
            ys, xs = np.nonzero(gate[y0:y1, x0:x1])
            if xs.size == 0:
                continue
            vals = score_map[y0 + ys, x0 + xs]
            order = np.argsort(vals)[::-1]
            chosen: list[tuple[float, int, int]] = []
            for idx in order:
                x = int(x0 + xs[idx]); y = int(y0 + ys[idx])
                if any((x - px) ** 2 + (y - py) ** 2 < radius * radius for _, px, py in chosen):
                    continue
                chosen.append((float(score_map[y, x]), x, y))
                if len(chosen) >= max(1, int(per_tile)):
                    break
            if chosen:
                tile_lists.append(chosen)
    selected: list[tuple[float, int, int]] = []
    for rank in range(max(1, int(per_tile))):
        layer = [lst[rank] for lst in tile_lists if len(lst) > rank]
        layer.sort(reverse=True)
        for cand in layer:
            _, x, y = cand
            if any((x - sx) ** 2 + (y - sy) ** 2 < radius * radius for _, sx, sy in selected):
                continue
            selected.append(cand)
            if len(selected) >= limit:
                return selected
    return selected


def append_multiscale_micro_components(
    img_bgr: np.ndarray,
    comps: list[DefectComponent],
    kept_mask: np.ndarray,
    interior_mask: np.ndarray,
    residual_lab: Optional[np.ndarray] = None,
    parts: Optional[dict[str, np.ndarray]] = None,
    design_valid_mask: Optional[np.ndarray] = None,
) -> tuple[list[DefectComponent], np.ndarray]:
    """Fused micro/cluster recovery with conservative, evidence-shaped masks.

    A candidate must be supported by independent views of the same physical
    anomaly: vertical-context deviation, same-phase periodic deviation,
    per-column deviation, compact residual, or an aligned clean-template z map.
    This recovers subtle defects without dropping one global threshold and
    turning every normal stripe residual into a box.
    """
    h, w = img_bgr.shape[:2]
    parts = parts or {}
    if residual_lab is None:
        tmp_valid = make_interior_mask(img_bgr, border_px=4, border_frac=0.0, auto_yellow_border=True)
        residual_lab, _ = per_cell_normal_residuals(img_bgr, tmp_valid)

    # Micro defects are allowed on the yellow frame and at clipped device edges.
    # Only the absolute image boundary is forbidden; frame-like candidates are
    # handled later with evidence/shape gates rather than a 40+ px dead zone.
    valid = np.ones((h, w), dtype=bool)
    abs_pad = 4
    valid[:abs_pad, :] = False; valid[-abs_pad:, :] = False
    valid[:, :abs_pad] = False; valid[:, -abs_pad:] = False
    if design_valid_mask is not None:
        dm = design_valid_mask
        if dm.shape[:2] != (h, w):
            dm = cv2.resize(dm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        valid &= dm > 0
    strict_interior = make_interior_mask(img_bgr, border_px=4, border_frac=0.0, auto_yellow_border=True)
    frame_zone = valid & ~strict_interior

    pitch = float(parts.get("pitch_px", np.full((1, 1), estimate_vertical_pitch_px(img_bgr, valid)))[0, 0])
    micro_raw, micro_row, scale_support = compute_multiscale_micro_score(img_bgr, valid)
    periodic_raw, periodic_row, periodic_delta = compute_periodic_neighbor_anomaly(img_bgr, valid, pitch)
    column_raw, column_row = compute_column_residual_anomaly(residual_lab, valid)
    compact = parts.get("compact_score", np.zeros((h, w), dtype=np.float32)).astype(np.float32)
    large = parts.get("large_score", np.zeros((h, w), dtype=np.float32)).astype(np.float32)
    template_score = parts.get("template_score", np.zeros((h, w), dtype=np.float32)).astype(np.float32)
    transverse_score = parts.get("transverse_score", np.zeros((h, w), dtype=np.float32)).astype(np.float32)

    normalized = np.stack([
        np.clip(periodic_row / 5.0, 0.0, 8.0),
        np.clip(column_row / 5.0, 0.0, 8.0),
        np.clip(micro_row / 5.0, 0.0, 8.0),
        np.clip(compact / 6.0, 0.0, 8.0),
        np.clip(template_score / 4.5, 0.0, 8.0),
        np.clip(transverse_score / 5.5, 0.0, 8.0),
    ], axis=2)
    ordered = np.sort(normalized, axis=2)
    fused = ordered[:, :, -1] + 0.55 * ordered[:, :, -2] + 0.25 * ordered[:, :, -3]
    support_count = np.sum(normalized >= 0.78, axis=2).astype(np.uint8)
    local_bg = cv2.GaussianBlur(fused.astype(np.float32), (0, 0), sigmaX=5.0, sigmaY=5.0)
    prominence = np.maximum(0.0, fused - local_bg)
    delta_prominence = np.maximum(
        0.0, periodic_delta - cv2.GaussianBlur(periodic_delta, (0, 0), sigmaX=3.0, sigmaY=3.0)
    )

    has_major = any(
        int(d.area_px) >= 350 or int(d.bbox_px[2]) * int(d.bbox_px[3]) >= 1200
        for d in comps
    )
    occupied = kept_mask > 0
    hard_occ = cv2.erode(occupied.astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
    free = valid & ~hard_occ

    local_max = fused >= (cv2.dilate(fused.astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) - 1e-5)
    if has_major:
        gate = free & local_max & (
            ((fused >= 2.12) & (support_count >= 2) & (prominence >= 0.14))
            | ((np.maximum.reduce([periodic_row, column_row, micro_row, template_score]) >= 10.0) & (fused >= 1.75))
            | ((compact >= 7.5) & (np.maximum(periodic_row, column_row) >= 3.4))
            | ((periodic_row >= 2.35) & (delta_prominence >= 5.8) & (column_row >= 1.35) & (micro_row >= 1.9))
            | ((transverse_score >= 7.5) & (np.maximum(column_row, micro_row) >= 2.0))
        )
        peak_limit = 42
    else:
        gate = free & local_max & (
            ((fused >= 3.25) & (support_count >= 3) & (prominence >= 0.34))
            | ((np.maximum.reduce([periodic_row, column_row, micro_row, template_score]) >= 15.0) & (support_count >= 2))
            | ((template_score >= 7.5) & (fused >= 2.25) & (np.maximum.reduce([periodic_row, column_row, micro_row, compact, transverse_score]) >= 2.2))
            | ((transverse_score >= 10.0) & (np.maximum(column_row, micro_row) >= 2.8))
        )
        peak_limit = 10

    # Border/frame candidates need strong localized evidence, not merely yellow color.
    frame_ok = (
        ((template_score >= 8.0) & (np.maximum.reduce([periodic_row, column_row, micro_row, transverse_score]) >= 2.4))
        | (periodic_row >= 8.0)
        | ((periodic_row >= 2.3) & (delta_prominence >= 6.0) & (column_row >= 1.3))
        | (compact >= 9.0)
    )
    gate &= (~frame_zone) | frame_ok

    # Suppress row-wide and column-repeating candidates unless independently strong.
    weak_support = (support_count >= 2) & (fused >= 1.45) & free
    row_density = cv2.blur(weak_support.astype(np.float32), (max(31, int(0.11 * w)), 1))
    col_density = cv2.blur(weak_support.astype(np.float32), (1, max(41, int(0.13 * h))))
    very_strong = np.maximum.reduce([periodic_row, column_row, micro_row, template_score, transverse_score]) >= 15.0
    gate &= ((row_density < 0.18) & (col_density < 0.24)) | very_strong

    tile_side = max(40, int(round(0.061 * min(h, w))))
    peaks = _spatially_stratified_peaks(
        fused, gate, tile_size=tile_side,
        per_tile=2 if has_major else 1, radius=8,
        limit=50 if has_major else peak_limit,
    )
    if has_major:
        # Reserve a small budget for low-scoring but independently supported
        # defects that are the third event inside a busy tile.  Remove the
        # neighborhoods of already selected peaks before the second pass.
        faint_gate = gate & (fused >= 2.40) & (fused < 4.2)
        faint_u8 = faint_gate.astype(np.uint8) * 255
        for _, px, py in peaks:
            cv2.circle(faint_u8, (int(px), int(py)), 10, 0, -1)
        faint_gate = faint_u8 > 0
        extras = _spatially_stratified_peaks(
            fused, faint_gate, tile_size=tile_side, per_tile=1, radius=9, limit=8
        )
        peaks.extend(extras)
    new = list(comps)
    out_mask = kept_mask.copy()

    def add_candidate(mask_full: np.ndarray, peak_value: float, reason: str) -> None:
        nonlocal new, out_mask
        mask_full = (mask_full > 0).astype(np.uint8) * 255
        mask_full[~valid] = 0
        mask_full[hard_occ] = 0
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_full, connectivity=8)
        for i in range(1, n):
            x, y, ww, hh, area = stats[i].tolist()
            if area < 2 or area > 0.012 * h * w:
                continue
            cmask_local = labels[y:y + hh, x:x + ww] == i
            aspect = max(ww / max(hh, 1), hh / max(ww, 1))
            fill = area / max(1, ww * hh)
            angle = _component_orientation_deg(cmask_local)
            if _is_near_vertical_angle(angle, 11.0) and aspect > 8.0 and ww <= 4 and fill < 0.70:
                continue
            full_bool = labels == i
            # At least two independent detectors must support most ordinary
            # candidates.  A very strong template or periodic event may stand alone.
            sc = support_count[full_bool]
            peak_independent = float(np.max(np.maximum.reduce([
                periodic_row[full_bool], column_row[full_bool], micro_row[full_bool], template_score[full_bool], transverse_score[full_bool]
            ])))
            if float(np.mean(sc >= 2)) < 0.25 and peak_independent < 17.0:
                continue
            if np.count_nonzero(full_bool & (out_mask > 0)) > 0.70 * max(1, area):
                continue
            # A one/two-pixel evidence core receives only a one-pixel review halo.
            halo = 1 if area < 35 else 2
            full_u8 = full_bool.astype(np.uint8) * 255
            if halo:
                full_u8 = cv2.dilate(full_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * halo + 1, 2 * halo + 1)))
            ys2, xs2 = np.nonzero(full_u8)
            if xs2.size == 0:
                continue
            bx0, bx1 = int(xs2.min()), int(xs2.max()) + 1
            by0, by1 = int(ys2.min()), int(ys2.max()) + 1
            local = full_u8[by0:by1, bx0:bx1]
            poly = _component_polygon(local, x=bx0, y=by0)
            vals = fused[full_u8 > 0]
            score_val = max(float(peak_value), float(np.max(vals)) if vals.size else 0.0)
            new.append(DefectComponent(
                bbox_px=[bx0, by0, bx1 - bx0, by1 - by0],
                polygon_px=poly,
                area_px=int(np.count_nonzero(full_u8)),
                score=round(max(8.9, min(65.0, 8.0 + 2.2 * score_val)), 3),
                mean_score=round(float(np.mean(vals)) if vals.size else score_val, 3),
                seam_overlap_frac=0.0,
                reason=reason,
            ))
            out_mask = cv2.bitwise_or(out_mask, full_u8)

    for peak_value, x, y in peaks:
        r = 9 if peak_value < 5.0 else 12
        x0, y0, x1, y1 = max(0, x - r), max(0, y - r), min(w, x + r + 1), min(h, y + r + 1)
        local_fused = fused[y0:y1, x0:x1]
        local_support = support_count[y0:y1, x0:x1]
        allowed = (
            ((local_fused >= max(1.15, 0.28 * peak_value)) & (local_support >= 2))
            | (template_score[y0:y1, x0:x1] >= max(4.0, 0.33 * float(template_score[y, x])))
            | ((periodic_row[y0:y1, x0:x1] >= max(3.2, 0.30 * float(periodic_row[y, x])))
               & (column_row[y0:y1, x0:x1] >= 2.2))
        )
        seed = np.zeros_like(allowed, dtype=np.uint8)
        seed[y - y0, x - x0] = 255
        grown = seed.copy()
        allowed_u8 = allowed.astype(np.uint8) * 255
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for _ in range(5):
            nxt = cv2.bitwise_and(cv2.dilate(grown, k3), allowed_u8)
            nxt = cv2.bitwise_or(nxt, seed)
            if np.array_equal(nxt, grown):
                break
            grown = nxt
        full = np.zeros((h, w), dtype=np.uint8)
        full[y0:y1, x0:x1] = grown
        before = len(new)
        add_candidate(full, peak_value, "kept_fused_micro")
        if len(new) == before and (
            (support_count[y, x] >= 2 and fused[y, x] >= 2.35)
            or template_score[y, x] >= 7.0
            or periodic_row[y, x] >= 12.0
        ):
            # The seed passed the fused gate but a shape filter rejected its tiny
            # one-pixel component.  Keep a minimal review proposal rather than
            # silently losing a real microscopic defect.
            tiny = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(tiny, (int(x), int(y)), 2, 255, -1)
            tiny[hard_occ] = 0
            ys3, xs3 = np.nonzero(tiny)
            if xs3.size:
                bx0, bx1 = int(xs3.min()), int(xs3.max()) + 1
                by0, by1 = int(ys3.min()), int(ys3.max()) + 1
                local = tiny[by0:by1, bx0:bx1]
                new.append(DefectComponent(
                    bbox_px=[bx0, by0, bx1 - bx0, by1 - by0],
                    polygon_px=_component_polygon(local, bx0, by0),
                    area_px=int(np.count_nonzero(tiny)),
                    score=round(max(8.9, min(65.0, 8.0 + 2.2 * float(peak_value))), 3),
                    mean_score=round(float(peak_value), 3),
                    seam_overlap_frac=0.0,
                    reason="kept_fused_micro_seed",
                ))
                out_mask = cv2.bitwise_or(out_mask, tiny)

    # Independent weak-cluster path.  Keep only a handful of genuinely dense
    # groups and connect nearby evidence with a two-pixel bridge.  The previous
    # implementation emitted every disconnected weak pixel as its own box.
    weak = free & (fused >= (1.75 if has_major else 2.30)) & (support_count >= 2)
    density = cv2.GaussianBlur(weak.astype(np.float32), (0, 0), sigmaX=6.0, sigmaY=6.0)
    core = (density >= (0.050 if has_major else 0.075)) & free
    core_u8 = cv2.morphologyEx(
        core.astype(np.uint8) * 255, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(core_u8, connectivity=8)
    cluster_candidates: list[tuple[float, np.ndarray]] = []
    for i in range(1, n):
        x, y, ww, hh, area = stats[i].tolist()
        if area < 18 or area > 0.018 * h * w:
            continue
        roi = labels[y:y + hh, x:x + ww] == i
        peak_count = sum(1 for _, px, py in peaks if x <= px < x + ww and y <= py < y + hh)
        mass = float(np.sum(np.maximum(0.0, fused[y:y + hh, x:x + ww] - 1.55)[roi]))
        if peak_count < 3 and mass < 30.0:
            continue
        supported = weak[y:y + hh, x:x + ww] & cv2.dilate(
            roi.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        ).astype(bool)
        local = supported.astype(np.uint8) * 255
        # Bridge only very nearby evidence; never fill the cluster bbox.
        local = cv2.dilate(local, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        if np.count_nonzero(local) < 8:
            continue
        full = np.zeros((h, w), dtype=np.uint8)
        full[y:y + hh, x:x + ww] = local
        cluster_candidates.append((mass, full))

    for mass, full in sorted(cluster_candidates, key=lambda t: t[0], reverse=True)[:4]:
        add_candidate(full, min(12.0, 2.0 + mass / 12.0), "kept_fused_micro_cluster")

    new.sort(key=lambda d: d.score, reverse=True)
    return new, out_mask


def append_clustered_review_region_components(
    img_bgr: np.ndarray,
    comps: list[DefectComponent],
    kept_mask: np.ndarray,
) -> tuple[list[DefectComponent], np.ndarray]:
    """Add one broad review region for clustered faint/smear evidence.

    The small-particle passes intentionally mark only high-confidence local
    evidence.  For diffuse delamination/grime fields this leaves a constellation
    of tiny boxes while the visible defect is a larger faint halo.  This pass is
    gated by the existing detections: it only adds a large region when several
    already-kept boxes form a spatial cluster.  Clean cells with no detections do
    not enter this path.
    """
    h, w = img_bgr.shape[:2]
    if len(comps) < 7 or h <= 0 or w <= 0:
        return comps, kept_mask

    image_area = float(h * w)
    seed = np.zeros((h, w), dtype=np.uint8)
    member_boxes: list[tuple[int, int, int, int, int]] = []
    grow_seed = max(18, int(round(0.038 * min(h, w))))

    for idx, d in enumerate(comps):
        x, y, ww, hh = [int(round(v)) for v in d.bbox_px[:4]]
        if ww <= 0 or hh <= 0:
            continue
        area = ww * hh
        reason = str(getattr(d, "reason", ""))
        # Do not let a single already-huge frame/edge/material box define the
        # cluster.  The cluster should be made from multiple small/medium pieces.
        if area > 0.060 * image_area:
            continue
        if "edge_material" in reason and (x <= 8 or y <= 8 or x + ww >= w - 8 or y + hh >= h - 8):
            continue
        cx = x + ww // 2
        cy = y + hh // 2
        # Skip ordinary bottom-center registration/contact target candidates.
        if (0.43 * w <= cx <= 0.57 * w) and (cy >= 0.875 * h) and area < 0.010 * image_area:
            continue
        x0 = max(0, x - grow_seed)
        y0 = max(0, y - grow_seed)
        x1 = min(w, x + ww + grow_seed)
        y1 = min(h, y + hh + grow_seed)
        cv2.rectangle(seed, (x0, y0), (x1, y1), 255, -1)
        member_boxes.append((idx, x, y, ww, hh))

    if len(member_boxes) < 7:
        return comps, kept_mask

    # Smooth the seed so a smear made of nearby specks becomes one component, but
    # widely scattered dust points stay separate.
    k = max(21, int(round(0.045 * min(h, w))) | 1)
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(seed, connectivity=8)

    new = list(comps)
    out_mask = kept_mask.copy()

    def inter_area(a, b):
        ax0, ay0, aw, ah = a
        bx0, by0, bw, bh = b
        ax1, ay1 = ax0 + aw, ay0 + ah
        bx1, by1 = bx0 + bw, by0 + bh
        return max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))

    for i in range(1, n):
        sx, sy, sww, shh, sarea = stats[i].tolist()
        if sarea < 0.010 * image_area:
            continue
        members: list[tuple[int, int, int, int, int]] = []
        for mb in member_boxes:
            _, x, y, ww, hh = mb
            cx = x + ww // 2
            cy = y + hh // 2
            inside = sx <= cx <= sx + sww and sy <= cy <= sy + shh and labels[min(max(cy, 0), h - 1), min(max(cx, 0), w - 1)] == i
            overlaps = inter_area((sx, sy, sww, shh), (x, y, ww, hh)) > 0
            if inside or overlaps:
                members.append(mb)
        if len(members) < 7:
            continue
        # Avoid turning a field of independent tiny dust particles into one huge
        # subtraction box.  Broad cluster boxes are only for areas that also have
        # medium/large smear evidence from the main detector or salient spots.
        member_reasons = [str(comps[m[0]].reason) for m in members]
        non_sparse = sum(1 for r in member_reasons if ("tiny_dark" not in r and "sparse_particle" not in r))
        core_smear = sum(1 for r in member_reasons if (r == "kept" or "edge_grown" in r or "large_material_density" in r or "edge_material" in r))
        if non_sparse < 5 or core_smear < 2:
            continue

        xs0 = [m[1] for m in members]
        ys0 = [m[2] for m in members]
        xs1 = [m[1] + m[3] for m in members]
        ys1 = [m[2] + m[4] for m in members]
        x0, y0, x1, y1 = min(xs0), min(ys0), max(xs1), max(ys1)
        span_w, span_h = x1 - x0, y1 - y0
        if span_w < max(75, int(0.095 * w)) or span_h < max(55, int(0.075 * h)):
            continue
        bbox_area = span_w * span_h
        if bbox_area < 0.012 * image_area or bbox_area > 0.52 * image_area:
            continue

        # Expand more for diffuse haze fields than for compact clusters.
        grow = max(20, int(round(0.035 * min(h, w))))
        x0 = max(0, x0 - grow)
        y0 = max(0, y0 - grow)
        x1 = min(w, x1 + grow)
        y1 = min(h, y1 + grow)

        candidate = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
        dup = False
        for d in new:
            b = [int(v) for v in d.bbox_px[:4]]
            ia = inter_area(candidate, b)
            if ia / max(1, min(candidate[2] * candidate[3], b[2] * b[3])) > 0.72 and (b[2] * b[3]) > 0.55 * (candidate[2] * candidate[3]):
                dup = True
                break
        if dup:
            continue

        full = np.ones((y1 - y0, x1 - x0), dtype=np.uint8) * 255
        poly = _component_polygon(full, x=int(x0), y=int(y0))
        score_val = min(65.0, 22.0 + 2.0 * len(members))
        new.append(DefectComponent(
            bbox_px=candidate,
            polygon_px=poly,
            area_px=int(np.count_nonzero(full)),
            score=round(float(score_val), 3),
            mean_score=round(float(score_val), 3),
            seam_overlap_frac=0.0,
            reason="kept_clustered_diffuse_region",
        ))
        out_mask[y0:y1, x0:x1] = 255

    new.sort(key=lambda d: d.score, reverse=True)
    return new, out_mask


def suppress_bottom_center_target_components(
    img_bgr: np.ndarray,
    comps: list[DefectComponent],
    kept_mask: np.ndarray,
) -> tuple[list[DefectComponent], np.ndarray]:
    """Drop small boxes on the normal bottom-center registration/contact target."""
    h, w = img_bgr.shape[:2]
    if not comps:
        return comps, kept_mask
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    keep: list[DefectComponent] = []
    for d in comps:
        x, y, ww, hh = [int(round(v)) for v in d.bbox_px[:4]]
        area = max(0, ww) * max(0, hh)
        cx = x + ww / 2.0
        cy = y + hh / 2.0
        drop = False
        reason = str(d.reason or "")
        protected_micro = ("fused_micro" in reason or "template" in reason) and float(d.score) >= 12.0
        if (not protected_micro) and (0.425 * w <= cx <= 0.575 * w) and (cy >= 0.865 * h) and area < 0.014 * h * w:
            yy0, yy1 = max(0, y), min(h, y + hh)
            xx0, xx1 = max(0, x), min(w, x + ww)
            if yy1 > yy0 and xx1 > xx0:
                roi_v = V[yy0:yy1, xx0:xx1]
                roi_s = S[yy0:yy1, xx0:xx1]
                min_v = float(np.min(roi_v)) if roi_v.size else 255.0
                med_s = float(np.median(roi_s)) if roi_s.size else 0.0
                # Real bottom-edge damage tends to include very dark/brown pixels
                # and/or spans into the actual crop edge; ordinary target rings do not.
                touches_edge = y + hh >= h - 3
                if (min_v > 72 and med_s < 75 and not touches_edge) or (area < 0.0045 * h * w and min_v > 58):
                    drop = True
        if not drop:
            keep.append(d)
    if len(keep) == len(comps):
        return comps, kept_mask
    rebuilt = np.zeros((h, w), dtype=np.uint8)
    for d in keep:
        if d.polygon_px and len(d.polygon_px) >= 3:
            pts = np.asarray(d.polygon_px, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(rebuilt, [pts], 255)
        else:
            x, y, ww, hh = [int(round(v)) for v in d.bbox_px[:4]]
            rebuilt[max(0, y):min(h, y + hh), max(0, x):min(w, x + ww)] = 255
    keep.sort(key=lambda d: d.score, reverse=True)
    return keep, rebuilt

def suppress_overlapping_edge_duplicates(
    comps: list[DefectComponent],
    kept_mask: np.ndarray,
    shape_hw: tuple[int, int],
) -> tuple[list[DefectComponent], np.ndarray]:
    """Remove duplicate boxes caused by multiple edge/halo recovery passes."""
    if len(comps) <= 1:
        return comps, kept_mask
    h, w = shape_hw

    def box_area(b):
        return max(0, int(b[2])) * max(0, int(b[3]))

    def inter_area(a, b):
        ax0, ay0, aw, ah = [int(v) for v in a]
        bx0, by0, bw, bh = [int(v) for v in b]
        ax1, ay1 = ax0 + aw, ay0 + ah
        bx1, by1 = bx0 + bw, by0 + bh
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        return max(0, ix1 - ix0) * max(0, iy1 - iy0)

    # Edge-material boxes are intentionally conservative.  Keep them and drop
    # smaller boxes whose center/area lies inside them so the review UI does not
    # show a stack of duplicate boxes around the same border defect.
    keep = [True] * len(comps)
    edge_idxs = [i for i, d in enumerate(comps) if "edge_material" in str(d.reason)]
    for ei in edge_idxs:
        eb = comps[ei].bbox_px
        ex0, ey0, ew, eh = [int(v) for v in eb]
        ex1, ey1 = ex0 + ew, ey0 + eh
        for j, d in enumerate(comps):
            if j == ei or not keep[j]:
                continue
            b = d.bbox_px
            cx = int(b[0]) + int(b[2]) / 2.0
            cy = int(b[1]) + int(b[3]) / 2.0
            ia = inter_area(eb, b)
            amin = max(1, min(box_area(eb), box_area(b)))
            center_inside = ex0 <= cx <= ex1 and ey0 <= cy <= ey1
            if center_inside or ia / amin >= 0.35:
                # Do not let a small edge-recovery box suppress a huge material
                # sheet that happens to contain/touch it.
                if box_area(b) > 4 * box_area(eb):
                    continue
                # If both are edge boxes, keep the larger one.
                if "edge_material" in str(d.reason) and box_area(b) > box_area(eb):
                    keep[ei] = False
                else:
                    keep[j] = False

    # Generic near-duplicate removal.
    order = sorted([i for i in range(len(comps)) if keep[i]], key=lambda i: (float(comps[i].score), box_area(comps[i].bbox_px)), reverse=True)
    final: list[int] = []
    for i in order:
        b = comps[i].bbox_px
        duplicate = False
        for j in final:
            ia = inter_area(b, comps[j].bbox_px)
            if ia / max(1, min(box_area(b), box_area(comps[j].bbox_px))) >= 0.70:
                # A broad material/cluster region is intentionally allowed to
                # contain smaller speck/edge boxes; otherwise high-scoring tiny
                # particles inside a delamination sheet would suppress the sheet.
                broad_i = (("large_material_density" in str(comps[i].reason)) or ("clustered_diffuse_region" in str(comps[i].reason)) or box_area(b) > 20000)
                broad_j = (("large_material_density" in str(comps[j].reason)) or ("clustered_diffuse_region" in str(comps[j].reason)) or box_area(comps[j].bbox_px) > 20000)
                if broad_i and box_area(b) > 4 * box_area(comps[j].bbox_px):
                    continue
                if broad_j and box_area(comps[j].bbox_px) > 4 * box_area(b):
                    continue
                duplicate = True
                break
        if not duplicate:
            final.append(i)
    new = [comps[i] for i in final]
    new.sort(key=lambda d: d.score, reverse=True)
    rebuilt = np.zeros((h, w), dtype=np.uint8)
    for d in new:
        if d.polygon_px and len(d.polygon_px) >= 3:
            pts = np.asarray(d.polygon_px, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(rebuilt, [pts], 255)
        else:
            x0, y0, ww0, hh0 = [int(v) for v in d.bbox_px]
            rebuilt[y0:y0 + hh0, x0:x0 + ww0] = 255
    return new, rebuilt

def cleanup_defect_geometry(
    img_bgr: np.ndarray,
    comps: list[DefectComponent],
    score: np.ndarray,
    parts: dict[str, np.ndarray],
    interior_mask: np.ndarray,
    params: DetectorParams,
    design_valid_mask: Optional[np.ndarray] = None,
) -> tuple[list[DefectComponent], np.ndarray]:
    """Convert noisy proposal boxes into tight, non-redundant defect regions.

    Detection passes are intentionally recall-heavy.  Several of them emit filled
    rectangles as *review proposals*.  Those rectangles must never be treated as
    final subtraction geometry: they create giant parent boxes, duplicate children,
    and destructive masks.  This stage treats every proposal as a seed/ROI only,
    reconstructs supported pixels from weak/strong evidence, splits disconnected
    regions, merges only genuinely adjacent fragments, and suppresses nested boxes.
    """
    h, w = img_bgr.shape[:2]
    if not comps or h <= 0 or w <= 0:
        return comps, np.zeros((h, w), dtype=np.uint8)

    valid = np.ones((h, w), dtype=bool)
    abs_pad = max(4, int(round(0.006 * min(h, w))))
    valid[:abs_pad, :] = False
    valid[-abs_pad:, :] = False
    valid[:, :abs_pad] = False
    valid[:, -abs_pad:] = False
    if design_valid_mask is not None:
        dm = design_valid_mask
        if dm.shape[:2] != (h, w):
            dm = cv2.resize(dm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        valid &= dm > 0

    feats = _edge_local_features(img_bgr, valid)
    H, S, V = feats["H"], feats["S"], feats["V"]
    dark = feats["dark"]
    chroma = feats["chroma"]
    edge = feats["edge"]
    dL = feats["dL"]

    force = parts.get("force_mask")
    force = (force > 0) if force is not None else np.zeros((h, w), dtype=bool)
    large_score = parts.get("large_score", score)
    compact_score = parts.get("compact_score", score)
    chroma_large = parts.get("chroma_large", np.zeros_like(score))
    dark_local = parts.get("dark_local", np.zeros_like(score))
    chroma_local = parts.get("chroma_local", np.zeros_like(score))
    transverse = parts.get("transverse_score", np.zeros_like(score))

    brown = (H >= 6) & (H <= 62) & (S > 18) & (V > 25) & (V < 205)
    strong = (
        (score >= max(7.6, params.min_score - 1.0))
        | force
        | ((V < 84) & (S > 2))
        | ((dark >= 4.2) & (V < 172))
        | ((chroma >= 5.0) & (S > 8) & (V < 198))
        | (brown & (dark >= 1.45))
        | ((edge >= 10.0) & (dark >= 2.0) & (V < 185))
    ) & valid

    material_seed = (
        ((V < 92) & (S > 2))
        | ((dark >= 4.6) & (V < 165))
        | ((chroma >= 5.2) & (S > 10) & (V < 190))
        | (brown & ((dark >= 1.6) | (V < 145)))
        | force
    ) & valid

    broad_weak = (
        (large_score >= 2.65)
        | (chroma_large >= 2.25)
        | ((dark >= 1.05) & (V < 205))
        | ((chroma >= 1.55) & (S > 4) & (V < 212))
        | ((edge >= 3.4) & (V < 202) & ((dark >= 0.7) | (chroma >= 1.0)))
        | ((transverse >= 2.65) & ((dark_local >= 0.55) | (chroma_local >= 0.75) | (V < 175)))
        | ((dL <= -5.0) & (V < 205))
        | (brown & ((dark >= 1.20) | (chroma >= 1.55) | (large_score >= 2.0) | (transverse >= 2.3)))
        | force
    ) & valid
    compact_weak = (
        (compact_score >= 5.4)
        | ((dark_local >= 2.4) & (V < 205))
        | ((dark >= 2.0) & (V < 190))
        | ((chroma >= 2.7) & (S > 6) & (V < 205))
        | ((edge >= 6.0) & (V < 195) & ((dark >= 1.0) | (chroma >= 1.7)))
        | ((transverse >= 3.15) & ((dark_local >= 0.85) | (chroma_local >= 1.0) | (V < 165)))
        | force
    ) & valid
    edge_allowed = (
        (score >= 5.2)
        | (large_score >= 3.45)
        | (chroma_large >= 2.85)
        | ((dark >= 1.75) & (V < 192))
        | ((chroma >= 2.45) & (S > 8) & (V < 202))
        | ((edge >= 5.4) & (V < 190) & ((dark >= 1.2) | (chroma >= 1.8)))
        | ((transverse >= 2.75) & ((dark_local >= 0.60) | (chroma_local >= 0.85) | (V < 178)))
        | ((dL <= -8.0) & (V < 190))
        | force
        | strong
    ) & valid
    extension_support = (
        ((dark_local >= 0.95) & (V < 220))
        | ((chroma_local >= 1.35) & (S > 4) & ((dark_local >= 0.35) | (transverse >= 1.55)))
        | ((transverse >= 1.95) & ((dark_local >= 0.30) | (chroma_local >= 0.55) | (V < 178)))
        | force
    ) & valid
    weak = broad_weak | compact_weak | strong

    # Remove persistent device fingers and frame strips from the weak growth mask.
    # Strong local evidence is retained even when it happens to lie on a line.
    vlen = odd_kernel(max(31, int(round(0.055 * h))))
    hlen = odd_kernel(max(41, int(round(0.070 * w))))
    weak_u8 = weak.astype(np.uint8) * 255
    long_vertical = cv2.morphologyEx(
        weak_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, vlen))
    ) > 0
    long_horizontal = cv2.morphologyEx(
        weak_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hlen, 3))
    ) > 0
    weak &= ~(long_vertical & ~strong)
    weak &= ~(long_horizontal & ~strong)
    broad_weak &= ~(long_vertical & ~strong)
    broad_weak &= ~(long_horizontal & ~strong)
    edge_allowed &= ~(long_vertical & ~strong)
    edge_allowed &= ~(long_horizontal & ~strong)

    def component_mask(d: DefectComponent) -> np.ndarray:
        m = np.zeros((h, w), dtype=np.uint8)
        if d.polygon_px and len(d.polygon_px) >= 3:
            pts = np.asarray(d.polygon_px, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(m, [pts], 255)
        else:
            x, y, ww, hh = [int(round(v)) for v in d.bbox_px[:4]]
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(w, x + ww), min(h, y + hh)
            if x1 > x0 and y1 > y0:
                m[y0:y1, x0:x1] = 255
        return m

    def bbox_from_mask(m: np.ndarray) -> list[int] | None:
        ys, xs = np.nonzero(m)
        if xs.size == 0:
            return None
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        return [x0, y0, x1 - x0, y1 - y0]

    def reconstruct(
        roi_box: list[int],
        seed_full: np.ndarray,
        allowed_full: np.ndarray,
        max_iterations: int,
        close_size: int,
    ) -> np.ndarray:
        x, y, ww, hh = [int(round(v)) for v in roi_box[:4]]
        extra = max(4, int(round(0.008 * min(h, w))))
        x0, y0 = max(0, x - extra), max(0, y - extra)
        x1, y1 = min(w, x + ww + extra), min(h, y + hh + extra)
        if x1 <= x0 or y1 <= y0:
            return np.zeros((h, w), dtype=np.uint8)

        seed = (seed_full[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
        allowed = (allowed_full[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
        seed = cv2.bitwise_and(seed, allowed)
        if not np.any(seed):
            return np.zeros((h, w), dtype=np.uint8)

        # Remove one-pixel noise from seeds, but restore strong singleton particles.
        opened = cv2.morphologyEx(
            seed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        if np.any(opened):
            seed = opened
        if close_size > 1:
            k = odd_kernel(close_size)
            allowed = cv2.morphologyEx(
                allowed, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            )

        grown = seed.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for _ in range(max(1, int(max_iterations))):
            nxt = cv2.bitwise_and(cv2.dilate(grown, kernel, iterations=1), allowed)
            nxt = cv2.bitwise_or(nxt, seed)
            if np.array_equal(nxt, grown):
                break
            grown = nxt

        out = np.zeros((h, w), dtype=np.uint8)
        out[y0:y1, x0:x1] = grown
        return out

    proposals: list[dict] = []
    cluster_parents: list[DefectComponent] = []

    for d in comps:
        reason = str(d.reason or "kept")
        original = component_mask(d)
        box_area = max(1, int(d.bbox_px[2]) * int(d.bbox_px[3]))
        fill = float(np.count_nonzero(original) / box_area)

        if "clustered_diffuse_region" in reason:
            cluster_parents.append(d)
            continue

        is_spot = any(k in reason for k in (
            "tiny_dark", "salient_spot", "sparse_particle", "clustered_tiny",
            "fused_micro", "fused_micro_cluster",
        ))
        is_broad_rectangle = (
            "edge_material" in reason
            or "edge_grown" in reason
            or "large_material_density" in reason
            or (fill >= 0.82 and box_area >= 700)
        )

        if "fused_micro" in reason:
            # The fused pass already reconstructed a tight evidence mask. Clip it
            # to the currently valid image region. If the optional GDS mask is
            # enabled, that mask participates here; it is deliberately off by
            # default and is unrelated to the annotated topology/coverage fixes.
            mask = cv2.bitwise_and(original, valid.astype(np.uint8) * 255)
        elif is_spot:
            seed = ((original > 0) & strong).astype(np.uint8) * 255
            if not np.any(seed):
                x, y, ww, hh = [int(round(v)) for v in d.bbox_px[:4]]
                x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + ww), min(h, y + hh)
                if x1 > x0 and y1 > y0:
                    local_sal = np.maximum.reduce([
                        score[y0:y1, x0:x1],
                        dark[y0:y1, x0:x1] + 0.15 * edge[y0:y1, x0:x1],
                        chroma[y0:y1, x0:x1] + 0.10 * edge[y0:y1, x0:x1],
                    ])
                    iy, ix = np.unravel_index(int(np.argmax(local_sal)), local_sal.shape)
                    seed = np.zeros((h, w), dtype=np.uint8)
                    cv2.circle(seed, (x0 + int(ix), y0 + int(iy)), 2, 255, -1)
            mask = reconstruct(d.bbox_px, seed, compact_weak | strong, max_iterations=5, close_size=3)
        elif is_broad_rectangle:
            # Rectangular proposal is only an ROI.  Seed it from actual evidence,
            # never from the filled proposal itself.
            seed = np.zeros((h, w), dtype=np.uint8)
            x, y, ww, hh = [int(round(v)) for v in d.bbox_px[:4]]
            x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + ww), min(h, y + hh)
            if x1 > x0 and y1 > y0:
                if "large_material_density" in reason:
                    local_seed = material_seed[y0:y1, x0:x1] | strong[y0:y1, x0:x1]
                else:
                    local_seed = material_seed[y0:y1, x0:x1]
                # A non-rectangular original polygon may contribute only where
                # actual material evidence exists; never seed from a filled box.
                local_seed |= (original[y0:y1, x0:x1] > 0) & material_seed[y0:y1, x0:x1]
                seed[y0:y1, x0:x1] = local_seed.astype(np.uint8) * 255
            allowed = (broad_weak | strong) if "large_material_density" in reason else edge_allowed
            grow_iters = 7 if "large_material_density" in reason else 11
            mask = reconstruct(d.bbox_px, seed, allowed, max_iterations=grow_iters, close_size=1)
            # Broad proposals are search ROIs, not permission to keep every
            # connected chromatic stripe.  Retain only material/low-frequency
            # evidence after growth; this carves clean islands around large blobs.
            pale_sheet = (
                brown & (S > 10) & (V > 78) & (V < 245)
                & ((large_score >= 1.8) | (chroma_large >= 1.25) | (transverse >= 2.15))
            )
            transverse_material = (
                (transverse >= 2.85)
                & ((dark_local >= 0.55) | (chroma_local >= 0.80) | (V < 175))
            )
            trim_material = (
                ((V < 102) & (S > 2))
                | ((dark >= 4.6) & (V < 165))
                | ((chroma >= 5.5) & (S > 15) & (V < 186))
                | (brown & ((dark >= 1.55) | (V < 145)))
                | pale_sheet
                | transverse_material
            )
            broad_support = (
                trim_material.astype(np.uint8)
                + (large_score >= 4.15).astype(np.uint8)
                + ((dark >= 1.70) & (V < 184)).astype(np.uint8)
                + ((chroma_large >= 3.65) & (S > 14) & (V < 196) & (dark >= 0.35)).astype(np.uint8)
                + transverse_material.astype(np.uint8)
            )
            broad_trim = (
                force
                | (large_score >= 6.6)
                | (broad_support >= 2)
                | (transverse_material & ("edge" in reason))
            ) & valid
            mask = cv2.bitwise_and(mask, broad_trim.astype(np.uint8) * 255)
            if np.any(mask):
                mask = cv2.morphologyEx(
                    mask, cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                )
        else:
            # Main connected-component polygons are already the best geometry.
            # Do not regrow them through a weak field; that is how periodic texture
            # gets pulled into an otherwise good defect contour.
            mask = cv2.bitwise_and(original, valid.astype(np.uint8) * 255)

        if np.count_nonzero(mask) < 3:
            continue
        proposals.append({
            "mask": mask,
            "score": float(d.score),
            "mean_score": float(d.mean_score),
            "seam": float(d.seam_overlap_frac),
            "reasons": {reason},
            "parent": False,
        })

    # Cluster-parent rectangles are deliberately not emitted.  Their children are
    # cleaned below, then mask-aware containment removes redundant fragments.
    # A filled or weakly-grown parent is exactly what created the destructive
    # upper-right mega-box in earlier versions.

    # Split disconnected masks and reject unsupported frame/line artifacts.
    # Broad/edge proposals may contain several seed islands; retain the dominant
    # compact material island (plus only substantial nearby pieces), not every
    # vertical stripe fragment inside the ROI.
    split: list[dict] = []
    for p in proposals:
        m = p["mask"]
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        local_entries: list[tuple[int, dict]] = []
        edge_reason = any("edge_material" in r or "edge_grown" in r for r in p["reasons"])
        fused_reason = any("fused_micro" in r for r in p["reasons"])
        broad_reason = edge_reason or any("large_material_density" in r for r in p["reasons"])
        for i in range(1, n):
            x, y, ww, hh, area = stats[i].tolist()
            if area < 3:
                continue
            fill = area / max(ww * hh, 1)
            aspect = max(ww / max(hh, 1), hh / max(ww, 1))
            cmask = labels == i
            strong_frac = float(np.count_nonzero(cmask & strong) / max(area, 1))
            material_frac = float(np.count_nonzero(cmask & material_seed) / max(area, 1))
            inside_frac = float(np.count_nonzero(cmask & interior_mask) / max(area, 1))

            narrow_vertical = hh >= 2.8 * max(ww, 1) and ww <= max(10, int(0.016 * w))
            if (not fused_reason) and narrow_vertical and area < 0.004 * h * w and material_frac < 0.35:
                continue
            if (not fused_reason) and aspect > 16.0 and fill < 0.38 and strong_frac < 0.12:
                continue
            if (ww > 0.48 * w and hh < 0.055 * h) or (hh > 0.48 * h and ww < 0.055 * w):
                continue

            cx = x + ww / 2.0
            cy = y + hh / 2.0
            near_outer_side = cx < 0.075 * w or cx > 0.925 * w
            # Extremely narrow side-frame detections are almost always crop/frame
            # junk, not device defects.  Real side defects extend farther inward.
            if edge_reason and cx < 0.050 * w and ww < 0.050 * w and cy > 0.60 * h:
                if material_frac < 0.30 and strong_frac < 0.14 and not force[cmask].any():
                    continue
            if edge_reason and cx < 0.050 * w and ww < 0.050 * w and not force[cmask].any():
                if material_frac < 0.26 and strong_frac < 0.12:
                    continue
            if edge_reason and cx > 0.950 * w and ww < 0.050 * w and not force[cmask].any():
                if material_frac < 0.26 and strong_frac < 0.12:
                    continue
            if edge_reason and near_outer_side and inside_frac < 0.48 and material_frac < 0.24:
                continue
            if edge_reason and cx < 0.075 * w and (x + ww) < 0.115 * w and cy > 0.62 * h and material_frac < 0.42:
                continue
            if (not fused_reason) and inside_frac < 0.08 and material_frac < 0.25 and not force[cmask].any():
                continue

            local_entries.append((area, {**p, "mask": cmask.astype(np.uint8) * 255, "bbox": [x, y, ww, hh]}))

        if not local_entries:
            continue
        local_entries.sort(key=lambda t: t[0], reverse=True)
        if not broad_reason:
            split.extend(entry for _, entry in local_entries)
            continue

        largest_area, largest = local_entries[0]
        selected = [largest]
        lx, ly, lww, lhh = largest["bbox"]
        lcx, lcy = lx + lww / 2.0, ly + lhh / 2.0
        for area, entry in local_entries[1:]:
            if area < max(24, 0.32 * largest_area):
                continue
            x, y, ww, hh = entry["bbox"]
            cx, cy = x + ww / 2.0, y + hh / 2.0
            gap = math.hypot(cx - lcx, cy - lcy)
            if gap <= max(28.0, 0.55 * max(lww, lhh)):
                selected.append(entry)

        merged_mask = np.zeros((h, w), dtype=np.uint8)
        for entry in selected:
            merged_mask = cv2.bitwise_or(merged_mask, entry["mask"])
        if len(selected) > 1:
            merged_mask = cv2.morphologyEx(
                merged_mask, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            )
        largest["mask"] = merged_mask
        split.append(largest)

    if not split:
        return [], np.zeros((h, w), dtype=np.uint8)

    # Mask-aware nested suppression.  Compare only overlapping bounding-box ROIs;
    # the earlier full-frame O(N^2) boolean operations made high-recall cells
    # needlessly slow.
    areas = [int(np.count_nonzero(p["mask"])) for p in split]
    boxes = [bbox_from_mask(p["mask"]) or [0, 0, 0, 0] for p in split]
    order = sorted(range(len(split)), key=lambda i: (areas[i], split[i]["score"], split[i]["parent"]), reverse=True)
    keep = [True] * len(split)

    def boxes_overlap(a: list[int], b: list[int], pad: int = 0) -> bool:
        ax, ay, aw, ah = a; bx, by, bw, bh = b
        return not (ax + aw + pad <= bx or bx + bw + pad <= ax or ay + ah + pad <= by or by + bh + pad <= ay)

    for oi, i in enumerate(order):
        if not keep[i]:
            continue
        large_u8 = split[i]["mask"]
        lb = boxes[i]
        for j in order[oi + 1:]:
            if not keep[j] or areas[i] < 2.2 * areas[j]:
                continue
            sb = boxes[j]
            if not boxes_overlap(lb, sb, pad=5):
                continue
            x0 = max(lb[0], sb[0]); y0 = max(lb[1], sb[1])
            x1 = min(lb[0] + lb[2], sb[0] + sb[2]); y1 = min(lb[1] + lb[3], sb[1] + sb[3])
            overlap_px = 0
            if x1 > x0 and y1 > y0:
                overlap_px = int(np.count_nonzero(
                    (large_u8[y0:y1, x0:x1] > 0) & (split[j]["mask"][y0:y1, x0:x1] > 0)
                ))
            overlap = float(overlap_px / max(areas[j], 1))
            cx = int(round(sb[0] + 0.5 * sb[2])); cy = int(round(sb[1] + 0.5 * sb[3]))
            center_supported = False
            if 0 <= cx < w and 0 <= cy < h:
                rx0, ry0 = max(0, cx - 4), max(0, cy - 4)
                rx1, ry1 = min(w, cx + 5), min(h, cy + 5)
                center_supported = bool(np.any(large_u8[ry0:ry1, rx0:rx1] > 0))
            contained = overlap >= 0.62
            center_supported = center_supported and overlap >= 0.28 and areas[i] >= 4.0 * areas[j]
            if contained or center_supported:
                keep[j] = False
                split[i]["reasons"].update(split[j]["reasons"])
                split[i]["score"] = max(split[i]["score"], split[j]["score"])

    split = [p for p, k in zip(split, keep) if k]

    # Merge nearby fragments only when at least one is substantial and their
    # local ROIs genuinely touch after a small dilation.
    changed = True
    merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    while changed:
        changed = False
        boxes_now = [bbox_from_mask(p["mask"]) or [0, 0, 0, 0] for p in split]
        areas_now = [int(np.count_nonzero(p["mask"])) for p in split]
        for i in range(len(split)):
            if changed:
                break
            ai = areas_now[i]
            bi = boxes_now[i]
            for j in range(i + 1, len(split)):
                aj = areas_now[j]
                bj = boxes_now[j]
                ext_i = any("edge_extension_fragment" in r for r in split[i]["reasons"])
                ext_j = any("edge_extension_fragment" in r for r in split[j]["reasons"])
                if ext_i != ext_j and max(ai, aj) >= 3.0 * max(1, min(ai, aj)):
                    # Keep a thin extension as a separate tight polygon instead of
                    # merging it into a large parent outer contour.
                    continue
                if max(ai, aj) < 260 and not (split[i]["parent"] or split[j]["parent"]):
                    continue
                if not boxes_overlap(bi, bj, pad=4):
                    continue
                x0 = max(0, min(bi[0], bj[0]) - 4)
                y0 = max(0, min(bi[1], bj[1]) - 4)
                x1 = min(w, max(bi[0] + bi[2], bj[0] + bj[2]) + 4)
                y1 = min(h, max(bi[1] + bi[3], bj[1] + bj[3]) + 4)
                mi = split[i]["mask"][y0:y1, x0:x1] > 0
                mj = split[j]["mask"][y0:y1, x0:x1] > 0
                di = cv2.dilate(mi.astype(np.uint8) * 255, merge_kernel) > 0
                if not np.any(di & mj):
                    continue
                union_local = mi | mj
                ys, xs = np.nonzero(union_local)
                if xs.size == 0:
                    continue
                bbox_area = (int(xs.max()) - int(xs.min()) + 1) * (int(ys.max()) - int(ys.min()) + 1)
                if bbox_area > 2.45 * max(1, ai + aj):
                    continue
                merged_local = cv2.morphologyEx(
                    union_local.astype(np.uint8) * 255, cv2.MORPH_CLOSE, merge_kernel
                )
                merged = np.zeros((h, w), dtype=np.uint8)
                merged[y0:y1, x0:x1] = merged_local
                split[i]["mask"] = merged
                split[i]["score"] = max(split[i]["score"], split[j]["score"])
                split[i]["mean_score"] = max(split[i]["mean_score"], split[j]["mean_score"])
                split[i]["seam"] = max(split[i]["seam"], split[j]["seam"])
                split[i]["parent"] = bool(split[i]["parent"] or split[j]["parent"])
                split[i]["reasons"].update(split[j]["reasons"])
                split.pop(j)
                changed = True
                break

    final: list[DefectComponent] = []
    rebuilt = np.zeros((h, w), dtype=np.uint8)
    for p in split:
        m = p["mask"]
        m = _complete_supported_internal_regions(
            m, score=score, parts=parts, strong=strong, material_seed=material_seed,
            valid=valid, reasons=set(p["reasons"]),
        )
        m, periodic_tightened = _tighten_periodic_texture_only_region(
            m,
            img_bgr=img_bgr,
            score=score,
            parts=parts,
            strong=strong,
            material_seed=material_seed,
            valid=valid,
            reasons=set(p["reasons"]),
        )
        if periodic_tightened:
            p["reasons"].add("periodic_texture_guard")
        if not np.any(m):
            continue
        # Small, shape-preserving subtraction/review halo.  Never fill the bbox.
        area0 = int(np.count_nonzero(m))
        halo = 1 if area0 < 40 else (2 if area0 < 1000 else 3)
        if halo > 0:
            k = odd_kernel(2 * halo + 1)
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            x, y, ww, hh, area = stats[i].tolist()
            if area < 3:
                continue
            local = (labels[y:y + hh, x:x + ww] == i).astype(np.uint8) * 255
            # Preserve internal clean islands in the simple-polygon output by
            # opening each hole to the exterior with a very narrow notch.
            local = _open_mask_holes_with_notches(local, min_hole_area=max(8, int(0.00002 * h * w)))
            contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            # A notch can occasionally split the component.  Emit each resulting
            # simple polygon separately so subtraction never fills a clean island.
            for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                c_area = int(round(cv2.contourArea(contour)))
                if c_area < 3:
                    continue
                perimeter = cv2.arcLength(contour, True)
                eps = max(0.75, 0.0035 * perimeter)
                contour = cv2.approxPolyDP(contour, eps, True)
                poly = [[int(pt[0][0] + x), int(pt[0][1] + y)] for pt in contour]
                piece = np.zeros_like(local)
                cv2.drawContours(piece, [contour], -1, 255, -1)
                full_piece = np.zeros((h, w), dtype=bool)
                full_piece[y:y + hh, x:x + ww] = piece > 0
                vals = score[full_piece]
                peak = max(float(p["score"]), float(np.max(vals)) if vals.size else 0.0)
                mean = float(np.mean(vals)) if vals.size else float(p["mean_score"])
                reasons = sorted(p["reasons"])
                reason = "cleaned:" + "+".join(reasons[:4])
                px, py, pww, phh = cv2.boundingRect(contour)
                final.append(DefectComponent(
                    bbox_px=[int(x + px), int(y + py), int(pww), int(phh)],
                    polygon_px=poly,
                    area_px=int(np.count_nonzero(piece)),
                    score=round(peak, 3),
                    mean_score=round(mean, 3),
                    seam_overlap_frac=round(float(p["seam"]), 3),
                    reason=reason,
                ))
                rebuilt[full_piece] = 255

    final.sort(key=lambda d: d.score, reverse=True)
    return final, rebuilt

# ---------------------------------------------------------------------------
# Preview overlays
# ---------------------------------------------------------------------------

def score_to_heat(score: np.ndarray, clip: float = 12.0) -> np.ndarray:
    s = np.clip(score / clip * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(s, cv2.COLORMAP_TURBO)


def draw_overlay(img_bgr: np.ndarray, comps: list[DefectComponent], kept_mask: np.ndarray, score: np.ndarray) -> np.ndarray:
    overlay = img_bgr.copy()
    heat = score_to_heat(score)
    alpha = (kept_mask.astype(np.float32) / 255.0) * 0.45
    overlay = (overlay.astype(np.float32) * (1 - alpha[:, :, None]) + heat.astype(np.float32) * alpha[:, :, None]).astype(np.uint8)

    for idx, comp in enumerate(comps, start=1):
        x, y, w, h = comp.bbox_px
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if len(comp.polygon_px) >= 3:
            pts = np.array(comp.polygon_px, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [pts], True, (0, 255, 255), 1)
        cv2.putText(
            overlay,
            f"{idx}:{comp.score:.1f}",
            (x, max(12, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return overlay


# ---------------------------------------------------------------------------
# Main detection API
# ---------------------------------------------------------------------------

def find_seam_mask_for_image(image_path: Path, seam_mask_dir: Optional[Path]) -> Optional[np.ndarray]:
    if seam_mask_dir is None:
        return None
    candidates = [
        seam_mask_dir / f"{image_path.stem}_seam_mask.png",
        seam_mask_dir / f"{image_path.stem}_seam.png",
        image_path.with_name(f"{image_path.stem}_seam_mask.png"),
    ]
    for c in candidates:
        if c.exists():
            return read_gray(c)
    return None


def detect_cell_defects(
    image_path: Path | str,
    params: DetectorParams,
    template: Optional[MedianTemplate] = None,
    seam_mask: Optional[np.ndarray] = None,
    max_width: int = 1400,
    design_mask: Optional[np.ndarray] = None,
    design_mask_provider=None,
) -> dict:
    image_path = Path(image_path)
    original_img = read_bgr(image_path)
    original_h, original_w = original_img.shape[:2]
    img = resize_bgr_max_width(original_img, max_width=max_width)
    detect_h, detect_w = img.shape[:2]
    if design_mask is None and design_mask_provider is not None:
        try:
            design_mask = design_mask_provider.mask_for_image(
                image_path.name, (original_h, original_w), (detect_h, detect_w)
            )
        except Exception:
            design_mask = None
    scale_to_original_x = original_w / float(max(detect_w, 1))
    scale_to_original_y = original_h / float(max(detect_h, 1))
    interior = make_interior_mask(
        img,
        border_px=params.border_px,
        border_frac=params.border_frac,
        auto_yellow_border=True,
    )

    template_z = None
    if template is not None:
        residual, expected, template_z = template_residuals(img, template)
    else:
        residual, expected = per_cell_normal_residuals(img, interior)

    score, parts = compute_defect_score(residual, img, interior)

    # Optional GDS-derived validity mask. This is an opt-in suppression aid for
    # projects that explicitly want it; it is not used to solve the annotated
    # missed-interior or missed-protrusion problems. Those are handled later by
    # evidence-supported topology completion and connected-edge recovery.
    design_valid: Optional[np.ndarray] = None
    design_boundary = np.zeros_like(score, dtype=bool)
    if design_mask is not None:
        dm = design_mask
        if dm.shape[:2] != (detect_h, detect_w):
            dm = cv2.resize(dm.astype(np.uint8), (detect_w, detect_h), interpolation=cv2.INTER_NEAREST)
        pitch = float(parts.get("pitch_px", np.full((1, 1), 7.0, dtype=np.float32))[0, 0])
        radius = max(2, int(round(0.58 * pitch)))
        k = odd_kernel(2 * radius + 1)
        design_valid_u8 = cv2.dilate(
            (dm > 0).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
            iterations=1,
        )
        # Close only sub-pitch gaps.  Large intentional voids remain holes.
        design_valid_u8 = cv2.morphologyEx(
            design_valid_u8, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, k // 2 * 2 + 1), max(3, k // 2 * 2 + 1))),
        )
        design_valid = design_valid_u8 > 0
        boundary_k = odd_kernel(max(3, int(round(0.85 * pitch))))
        design_boundary = cv2.morphologyEx(
            design_valid_u8, cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (boundary_k, boundary_k)),
        ) > 0
        parts["design_valid_mask"] = design_valid_u8.astype(np.float32)
        parts["design_boundary_mask"] = design_boundary.astype(np.float32)

    if template_z is not None:
        tzL = template_z[:, :, 0].astype(np.float32)
        tzA = template_z[:, :, 1].astype(np.float32)
        tzB = template_z[:, :, 2].astype(np.float32)
        template_score = np.maximum.reduce([
            np.maximum(0.0, -tzL),
            0.72 * np.maximum(0.0, tzL),
            np.sqrt(tzA * tzA + tzB * tzB),
        ]).astype(np.float32)
        template_score[~interior] = 0.0
        # Broad template differences are often legitimate device-layout changes.
        # Keep only localized template excess unless an image-domain detector
        # corroborates the same region.
        template_bg = cv2.GaussianBlur(template_score, (0, 0), sigmaX=7.0, sigmaY=7.0)
        template_local = np.maximum(0.0, template_score - 0.78 * template_bg)
        corroborated = (
            (parts.get("compact_score", 0) >= 2.6)
            | (parts.get("large_score", 0) >= 1.9)
            | (parts.get("transverse_score", 0) >= 3.2)
            | (parts.get("dark_local", 0) >= 1.8)
            | (parts.get("chroma_local", 0) >= 1.8)
        )
        template_promoted = np.where(
            ((template_local >= 1.15) & (template_score >= 4.8)) | corroborated,
            np.minimum(template_score, 18.0),
            0.0,
        ).astype(np.float32)
        if design_valid is not None:
            template_promoted[~design_valid] = 0.0
            template_score[~design_valid] = 0.0
            template_local[~design_valid] = 0.0
        score = np.maximum(score, template_promoted).astype(np.float32)
        parts["template_score"] = template_score
        parts["template_local"] = template_local.astype(np.float32)
    else:
        parts["template_score"] = np.zeros_like(score, dtype=np.float32)
        parts["template_local"] = np.zeros_like(score, dtype=np.float32)

    force_mask = parts.get("force_mask")
    physical_strong = (
        (parts.get("compact_score", np.zeros_like(score)) >= 6.2)
        | (parts.get("large_score", np.zeros_like(score)) >= 5.0)
        | (parts.get("dark_local", np.zeros_like(score)) >= 4.0)
        | (parts.get("chroma_local", np.zeros_like(score)) >= 4.2)
        | ((force_mask > 0) if force_mask is not None else False)
    )
    if design_valid is not None:
        score[~design_valid] = 0.0
        # Expected GDS boundaries are a weak prior, not an absolute veto: genuinely
        # dark/chromatic material can still win there.
        score[design_boundary & ~physical_strong] *= 0.32

    raw_mask = morphology_score_mask(score, params)
    if force_mask is not None:
        valid_detection = interior | (force_mask > 0)
    else:
        valid_detection = interior
    if design_valid is not None:
        valid_detection &= design_valid | physical_strong
    raw_mask[~valid_detection] = 0

    comps, kept_mask = find_components(
        score=score,
        raw_mask=raw_mask,
        interior_mask=interior,
        params=params,
        seam_mask=seam_mask,
        parts=parts,
    )
    comps, kept_mask = refine_edge_component_coverage(img, comps, kept_mask)
    comps, kept_mask = append_edge_material_components(img, comps, kept_mask)
    comps, kept_mask = refine_connected_edge_extensions(img, comps, kept_mask, parts=parts)
    comps, kept_mask = append_large_material_density_components(img, comps, kept_mask)
    comps, kept_mask = suppress_overlapping_edge_duplicates(comps, kept_mask, img.shape[:2])
    comps, kept_mask = append_tiny_dark_components(img, comps, kept_mask, interior)
    comps, kept_mask = append_salient_spot_components(img, comps, kept_mask, interior)
    comps, kept_mask = append_sparse_particle_components(img, comps, kept_mask)
    comps, kept_mask = append_multiscale_micro_components(
        img, comps, kept_mask, interior, residual_lab=residual, parts=parts,
        design_valid_mask=(design_valid.astype(np.uint8) * 255) if design_valid is not None else None,
    )
    comps, kept_mask = append_compact_border_specks(img, comps, kept_mask, parts=parts)
    comps, kept_mask = append_clustered_review_region_components(img, comps, kept_mask)
    comps, kept_mask = suppress_bottom_center_target_components(img, comps, kept_mask)
    comps, kept_mask = suppress_overlapping_edge_duplicates(comps, kept_mask, img.shape[:2])
    comps, kept_mask = cleanup_defect_geometry(
        img_bgr=img,
        comps=comps,
        score=score,
        parts=parts,
        interior_mask=interior,
        params=params,
        design_valid_mask=(design_valid.astype(np.uint8) * 255) if design_valid is not None else None,
    )

    return {
        "image": str(image_path.name),
        "image_path": str(image_path),
        "shape": [int(img.shape[0]), int(img.shape[1])],
        "original_shape": [int(original_h), int(original_w)],
        "scale_to_original_px": [float(scale_to_original_x), float(scale_to_original_y)],
        "defects": [asdict(c) for c in comps],
        "_debug": {
            "score": score,
            "kept_mask": kept_mask,
            "raw_mask": raw_mask,
            "interior_mask": interior.astype(np.uint8) * 255,
            "residual_lab": residual,
            "expected_lab": expected,
            "design_valid_mask": (design_valid.astype(np.uint8) * 255) if design_valid is not None else np.zeros_like(raw_mask),
            "design_boundary_mask": design_boundary.astype(np.uint8) * 255,
        },
    }


def strip_debug(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "_debug"}



# ---------------------------------------------------------------------------
# Autodetected image defects -> subtraction-ready GDS JSON
# ---------------------------------------------------------------------------

def _load_json(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _round_pair(pt: tuple[float, float], ndigits: int = 3) -> list[float]:
    return [round(float(pt[0]), ndigits), round(float(pt[1]), ndigits)]


def _fit_canvas_to_gds_affine(canvas_pts: np.ndarray, gds_pts: np.ndarray) -> np.ndarray:
    """Least-squares affine map from canvas/downscaled-canvas px to GDS microns."""
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

    # Fast crops are made from the downscaled wafer canvas, so use the saved
    # downscaled canvas corners. Native crops use full-resolution canvas corners.
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
    """Map original analysis-crop pixel coordinates to canvas/downscaled-canvas px.

    Handles both fast downscaled crops and native full-resolution crops using the
    transform metadata written by wafer_alignment_and_extraction.py.
    """
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


def _find_metadata_for_image(image_name: str, metadata_dir: Path) -> Optional[Path]:
    stem = Path(image_name).stem
    candidates = [
        metadata_dir / f"{stem}.json",
        metadata_dir / f"{stem.replace('_preview', '')}.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    hits = list(metadata_dir.rglob(f"{stem}.json"))
    return hits[0] if hits else None




class GDSDesignMaskProvider:
    """Rasterize expected selected-layer GDS geometry into each analysis crop.

    The optical cells are not all geometrically identical: some devices contain
    intentional line terminations, openings, and large voids.  A global median
    image therefore treats legitimate design geometry as a defect.  This provider
    uses the extraction metadata to map the real GDS polygons into crop pixels.
    The detector then ignores expected voids while retaining a small alignment
    tolerance around the active device geometry.
    """

    def __init__(
        self,
        gds_path: Path | str,
        metadata_dir: Path | str,
        layers: Iterable[int] = (1, 4),
        cache_dir: Path | str | None = None,
        rebuild_cache: bool = False,
    ) -> None:
        self.gds_path = Path(gds_path)
        self.metadata_dir = Path(metadata_dir)
        self.layers = {int(v) for v in layers}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.rebuild_cache = bool(rebuild_cache)
        self.records: list[tuple[np.ndarray, tuple[float, float, float, float]]] = []
        self.available = False
        self.error = ""

        try:
            import gdstk  # type: ignore

            if not self.gds_path.exists():
                raise FileNotFoundError(self.gds_path)
            # Reuse the project's filtered reader when possible; a full wafer GDS
            # can otherwise spend minutes loading irrelevant layers.
            if self.layers.issubset({1, 2, 4}):
                try:
                    from gds_parser import read_gds_filtered  # type: ignore
                    lib = read_gds_filtered(str(self.gds_path))
                except Exception:
                    lib = gdstk.read_gds(str(self.gds_path))
            else:
                lib = gdstk.read_gds(str(self.gds_path))
            tops = lib.top_level()
            if not tops:
                raise ValueError("GDS contains no top-level cell")
            flat = tops[0].copy("__DEFECT_DESIGN_MASK_FLAT__")
            flat.flatten()
            for poly in flat.polygons:
                if int(poly.layer) not in self.layers:
                    continue
                pts = np.asarray(poly.points, dtype=np.float64)
                if pts.ndim != 2 or pts.shape[0] < 3:
                    continue
                lo = np.min(pts, axis=0)
                hi = np.max(pts, axis=0)
                self.records.append((pts, (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))))
            if not self.records:
                raise ValueError(f"No polygons found on GDS layers {sorted(self.layers)}")
            if self.cache_dir:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.available = True
        except Exception as exc:
            self.error = str(exc)
            self.available = False

    def _cache_path(self, image_name: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        layer_tag = "-".join(str(v) for v in sorted(self.layers))
        return self.cache_dir / f"{Path(image_name).stem}_gds_layers_{layer_tag}.png"

    def mask_for_image(
        self,
        image_name: str,
        original_shape: tuple[int, int],
        output_shape: tuple[int, int],
    ) -> Optional[np.ndarray]:
        if not self.available:
            return None
        original_h, original_w = [int(v) for v in original_shape]
        out_h, out_w = [int(v) for v in output_shape]
        cache_path = self._cache_path(image_name)
        active: Optional[np.ndarray] = None
        if cache_path is not None and cache_path.exists() and not self.rebuild_cache:
            try:
                cached = read_gray(cache_path)
                if cached.shape[:2] == (original_h, original_w):
                    active = cached
            except Exception:
                active = None

        if active is None:
            meta_path = _find_metadata_for_image(image_name, self.metadata_dir)
            if meta_path is None:
                return None
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                crop_pts = np.asarray(
                    [[0.0, 0.0], [float(original_w - 1), 0.0],
                     [float(original_w - 1), float(original_h - 1)], [0.0, float(original_h - 1)]],
                    dtype=np.float64,
                )
                gds_pts = np.asarray(
                    [crop_px_to_gds(float(x), float(y), meta) for x, y in crop_pts],
                    dtype=np.float64,
                )
                crop_to_gds = _fit_canvas_to_gds_affine(crop_pts, gds_pts)
                gds_to_crop = cv2.invertAffineTransform(crop_to_gds.astype(np.float64))
                gx0, gy0 = np.min(gds_pts, axis=0)
                gx1, gy1 = np.max(gds_pts, axis=0)
                # A small GDS-coordinate margin protects against rounded metadata.
                span = max(float(gx1 - gx0), float(gy1 - gy0), 1.0)
                margin = 0.015 * span
                active = np.zeros((original_h, original_w), dtype=np.uint8)
                for pts, (px0, py0, px1, py1) in self.records:
                    if px1 < gx0 - margin or px0 > gx1 + margin or py1 < gy0 - margin or py0 > gy1 + margin:
                        continue
                    hom = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
                    mapped = (gds_to_crop @ hom.T).T
                    if not np.all(np.isfinite(mapped)):
                        continue
                    ipts = np.round(mapped).astype(np.int32).reshape(-1, 1, 2)
                    cv2.fillPoly(active, [ipts], 255)
                if cache_path is not None and np.any(active):
                    imwrite(cache_path, active)
            except Exception:
                return None

        if active is None or not np.any(active):
            return None
        if active.shape[:2] != (out_h, out_w):
            active = cv2.resize(active, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        return (active > 0).astype(np.uint8) * 255


def _resolve_design_gds_path(config_path: Path | str, explicit_path: str = "") -> Optional[Path]:
    if explicit_path:
        p = Path(explicit_path)
        return p if p.is_absolute() else Path.cwd() / p
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        raw = str(cfg.get("gds_path", "") or "").strip()
        if not raw:
            return None
        p = Path(raw)
        return p if p.is_absolute() else cfg_path.parent / p
    except Exception:
        return None


def _scale_polygon_to_original(poly: list[list[float]], sx: float, sy: float) -> list[tuple[float, float]]:
    out = []
    for p in poly:
        if len(p) >= 2:
            out.append((float(p[0]) * sx, float(p[1]) * sy))
    return out


def _scale_bbox_to_original(bbox: list[float], sx: float, sy: float) -> list[float]:
    x, y, w, h = [float(v) for v in bbox[:4]]
    return [x * sx, y * sy, w * sx, h * sy]


def convert_algo_results_to_gds_json(
    algo_results: dict,
    metadata_dir: Path | str,
    output_json: Path | str,
    default_type: str = "auto_defect",
    min_score: float = 0.0,
    use_polygons: bool = True,
) -> dict:
    """Convert detector output into the JSON schema consumed by subtract_defects.py.

    The output keeps the manual-labeling fields (`box_px`, `center_x_um`,
    `width_um`, `height_um`, `corners_gds`) and additionally writes
    `polygon_gds` for exact component-shaped subtraction. Updated
    subtract_defects.py will prefer `polygon_gds` for the actual cut and use
    `corners_gds` for scale/error-margin estimation.
    """
    metadata_dir = Path(metadata_dir)
    if not metadata_dir.exists():
        raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")

    out: dict[str, list[dict]] = {}
    converted = 0
    skipped = 0
    images = algo_results.get("images", [])
    bar = ProgressBar("GDS JSON", len(images))

    for idx, image_entry in enumerate(images, start=1):
        image_name = image_entry.get("image") or Path(image_entry.get("image_path", "")).name
        meta_path = _find_metadata_for_image(image_name, metadata_dir)
        if meta_path is None:
            skipped += len(image_entry.get("defects", []))
            bar.update(idx, extra=f"missing metadata for {image_name}")
            continue
        meta = _load_json(meta_path)
        canvas_to_gds = _metadata_canvas_to_gds_affine(meta)

        det_shape = image_entry.get("shape") or []
        orig_shape = image_entry.get("original_shape") or []
        if len(orig_shape) >= 2:
            orig_h, orig_w = float(orig_shape[0]), float(orig_shape[1])
        else:
            crop_size = meta.get("crop_size_px") or [0, 0]
            orig_w, orig_h = float(crop_size[0]), float(crop_size[1])
        if len(det_shape) >= 2:
            det_h, det_w = float(det_shape[0]), float(det_shape[1])
        else:
            det_h, det_w = orig_h, orig_w
        sx = orig_w / max(det_w, 1.0)
        sy = orig_h / max(det_h, 1.0)

        legacy_name = Path(meta.get("legacy_jpg", "")).name or f"{Path(image_name).stem}.jpg"
        cell_defects = out.setdefault(legacy_name, [])

        for defect in image_entry.get("defects", []):
            score = float(defect.get("score", 0.0) or 0.0)
            if score < float(min_score):
                continue
            bbox = defect.get("bbox_px") or defect.get("box_px")
            if not bbox or len(bbox) < 4:
                skipped += 1
                continue
            x, y, w, h = _scale_bbox_to_original(bbox, sx, sy)
            bbox_corners_px = [
                (x, y),
                (x + w, y),
                (x + w, y + h),
                (x, y + h),
            ]
            bbox_corners_gds = [_round_pair(crop_px_to_gds(px, py, meta, canvas_to_gds)) for px, py in bbox_corners_px]
            cx_gds, cy_gds = crop_px_to_gds(x + w / 2.0, y + h / 2.0, meta, canvas_to_gds)
            xs = [p[0] for p in bbox_corners_gds]
            ys = [p[1] for p in bbox_corners_gds]

            polygon_gds = None
            polygon_px_orig = _scale_polygon_to_original(defect.get("polygon_px") or [], sx, sy)
            if use_polygons and len(polygon_px_orig) >= 3:
                polygon_gds = [_round_pair(crop_px_to_gds(px, py, meta, canvas_to_gds)) for px, py in polygon_px_orig]

            record = {
                "type": str(defect.get("type") or default_type),
                "box_px": [int(round(x)), int(round(y)), int(round(w)), int(round(h))],
                "center_x_um": round(float(cx_gds), 3),
                "center_y_um": round(float(cy_gds), 3),
                "width_um": round(float(max(xs) - min(xs)), 3),
                "height_um": round(float(max(ys) - min(ys)), 3),
                "corners_gds": bbox_corners_gds,
                "score": round(score, 3),
                "area_px": int(round(float(defect.get("area_px", 0) or 0) * sx * sy)),
                "source": "algorithmic_detector",
            }
            if polygon_gds is not None:
                record["polygon_gds"] = polygon_gds
                # Keep the exact detector polygon in native crop pixels too.
                # The review UI uses this to show the real subtraction geometry;
                # box_px remains only the enclosing rectangle used for selection.
                record["polygon_px"] = [
                    [round(float(px), 3), round(float(py), 3)]
                    for px, py in polygon_px_orig
                ]
            if "seam_overlap_frac" in defect:
                record["seam_overlap_frac"] = defect.get("seam_overlap_frac")
            cell_defects.append(record)
            converted += 1

        bar.update(idx, extra=f"{image_name}: {len(image_entry.get('defects', []))} candidates")

    bar.done(extra=f"converted {converted}, skipped {skipped}")
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return {"converted": converted, "skipped": skipped, "output": str(output_json)}

# ---------------------------------------------------------------------------
# Review UI launcher
# ---------------------------------------------------------------------------

def launch_review_ui(
    image_dir: Path | str,
    annotations_json: Path | str,
    metadata_dir: Path | str,
    preview_dir: Path | str | None = None,
    wafer_id: str = "",
    review_state: Path | str | None = None,
    quick_review: bool = False,
    review_cache_size: int = 12,
    review_autosave_seconds: float = 4.0,
    review_hide_labels: bool = False,
) -> None:
    """Open the manual labeling UI with auto-detected boxes preloaded.

    The UI edits the same subtraction-ready JSON that subtract_defects.py reads.
    This makes the second command a detect -> review -> save stage.
    """
    restore_normal_priority()
    try:
        import defect_mapper_gui
    except Exception as exc:
        raise RuntimeError(f"Could not import defect_mapper_gui.py for review UI: {exc}") from exc

    gui_version = str(getattr(defect_mapper_gui, "GUI_VERSION", ""))
    if not gui_version:
        print("[Review UI] WARNING: defect_mapper_gui.py has no GUI_VERSION marker; replace it with the supplied updated file.")
    else:
        print(f"[Review UI] gui_version={gui_version}")

    print(
        f"[Review UI] mode={'quick/untyped' if quick_review else 'typed'}, "
        f"cache={max(2, int(review_cache_size))} cells, autosave={max(1.0, float(review_autosave_seconds)):.1f}s"
    )
    tool = defect_mapper_gui.AutoLabelReviewTool(
        image_dir=image_dir,
        annotations_json=annotations_json,
        metadata_dir=metadata_dir,
        preview_dir=preview_dir,
        wafer_id=wafer_id,
        resume_state_path=review_state,
        quick_label=bool(quick_review),
        default_defect_type="defect",
        image_cache_size=max(2, int(review_cache_size)),
        autosave_seconds=max(1.0, float(review_autosave_seconds)),
        show_annotation_labels=not bool(review_hide_labels or quick_review),
    )
    tool.run()
    # REVIEWED_WAFER_STITCH_AFTER_REVIEW_V1
    # The review UI edits annotations_json in place. Once it closes, rebuild a
    # GDS-positioned wafer overview from the final reviewed JSON so retained
    # automatic polygons and manually added regions appear together.
    try:
        from reviewed_defect_wafer_stitch import build_reviewed_wafer_stitch

        reviewed_stitch = build_reviewed_wafer_stitch(
            image_dir=Path(image_dir),
            annotations_json=Path(annotations_json),
            metadata_dir=Path(metadata_dir),
            wafer_id=str(wafer_id or ""),
        )
        print(
            "[Reviewed Wafer Stitch] Saved: "
            f"{reviewed_stitch['outputs']['composite']}"
        )
    except Exception as exc:
        # Do not destroy reviewed labels just because overview generation fails.
        print(
            "[Reviewed Wafer Stitch] WARNING: "
            f"overview generation failed: {exc}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Algorithmic h2p cell defect detector")
    p.add_argument("--input", default="extracted_cells/analysis_png", help="Cell image or directory containing cell crops. Default: extracted_cells/analysis_png")
    p.add_argument("--output", default="extracted_cells/algo_defects.json", help="Output JSON path. Default: extracted_cells/algo_defects.json")
    p.add_argument("--preview-dir", default="extracted_cells/algo_previews", help="Optional directory for overlay previews. Default: extracted_cells/algo_previews")
    p.add_argument("--seam-mask-dir", default="", help="Optional directory containing *_seam_mask.png files")
    p.add_argument("--template", default="", help="Optional .npz median normal template")
    p.add_argument("--save-template", default="", help="Build and save an aligned median+MAD normal template from --input, then continue")
    p.add_argument("--auto-template", dest="auto_template", action="store_true", default=True, help="Automatically build/use an aligned median+MAD template when the input contains enough cells. Default: on")
    p.add_argument("--no-auto-template", dest="auto_template", action="store_false", help="Disable automatic dataset template construction")
    p.add_argument("--auto-template-cache", default="", help="Optional .npz cache path for the automatic template. Default: <input>/../normal_template_auto.npz")
    p.add_argument("--auto-template-min-images", type=int, default=8, help="Minimum cell count for automatic template construction. Default: 8")
    p.add_argument("--max-template-images", type=int, default=80)
    p.add_argument("--max-image-width", type=int, default=3000, help="Resize images to this max width before detecting. 0 keeps full resolution. Default: 3000 for sharper balanced crops")
    p.add_argument("--cv-threads", type=int, default=1, help="OpenCV CPU threads. Default 1 so your PC stays usable")
    p.add_argument("--normal-priority", action="store_true", help="Do not lower process priority")
    p.add_argument("--save-heatmaps", action="store_true", help="Also save *_score_heat.png tuning images; disabled by default for speed")
    p.add_argument("--metadata-dir", default="extracted_cells/metadata", help="Metadata directory from extraction. Default: extracted_cells/metadata")
    p.add_argument("--config", default="config.json", help="Project config used to auto-discover gds_path. Default: config.json")
    p.add_argument("--design-gds", default="", help="Optional GDS path for opt-in design-aware suppression")
    p.add_argument("--design-layers", type=int, nargs="+", default=[1, 4], help="GDS layers defining active device geometry. Default: 1 4")
    p.add_argument("--design-mask-cache-dir", default="extracted_cells/design_masks", help="Cache directory for per-cell GDS design masks")
    p.add_argument("--use-design-mask", action="store_true", help="Opt in to GDS-aware suppression. Disabled by default because physical defects can occupy intentional GDS openings.")
    p.add_argument("--no-design-mask", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--rebuild-design-masks", action="store_true", help="Regenerate cached GDS design masks when --use-design-mask is enabled")
    p.add_argument("--gds-output-json", default="Wafer_A_device_defects.json", help="Also write subtraction-ready *_device_defects.json with corners_gds/polygon_gds. Default: Wafer_A_device_defects.json")
    p.add_argument("--gds-min-score", type=float, default=0.0, help="Minimum detector score to include in the GDS JSON. Default: include all kept candidates")
    p.add_argument("--no-polygon-gds", action="store_true", help="Only export bbox corners_gds; do not export exact polygon_gds")
    p.add_argument("--review-ui", action="store_true", help="After detection/GDS JSON export, open the labeling UI with auto-detected boxes loaded")
    p.add_argument(
        "--review-only", "--review-existing",
        dest="review_only", action="store_true",
        help="Open the existing --gds-output-json in the labeling UI without rerunning detection or rewriting either detector JSON",
    )
    p.add_argument(
        "--review-state", default="",
        help="Optional review progress file. Default: <gds-output-json>.review_state.json",
    )
    p.add_argument("--clean", dest="clean_outputs", action="store_true", default=True, help="Delete output JSONs and preview folder before running. Default: on")
    p.add_argument("--no-clean", dest="clean_outputs", action="store_false", help="Do not delete existing output JSONs or preview folder before running")
    p.add_argument("--review-preview-dir", default="", help="Optional original crop preview dir for review UI. Default: sibling previews directory next to --input")
    p.add_argument("--wafer-id", default="", help="Optional wafer id label for the review UI")
    p.add_argument(
        "--quick-review", "--untyped-review",
        dest="quick_review", action="store_true",
        help="Fast labeling mode: newly drawn boxes are saved immediately as generic type 'defect'; no 1-5 classification prompt",
    )
    p.add_argument(
        "--review-cache-size", type=int, default=12,
        help="Number of display-sized cell images kept in RAM for fast navigation. Default: 12",
    )
    p.add_argument(
        "--review-autosave-seconds", type=float, default=4.0,
        help="Save edited review JSON after this many idle seconds. Navigation never rewrites an unchanged JSON. Default: 4",
    )
    p.add_argument(
        "--review-hide-labels", action="store_true",
        help="Hide per-polygon type/score text for faster, cleaner rendering. Quick review implies this",
    )

    p.add_argument("--threshold", type=float, default=8.8)
    p.add_argument("--min-score", type=float, default=8.8)
    p.add_argument("--min-area", type=int, default=18)
    p.add_argument("--max-area-frac", type=float, default=0.22)
    p.add_argument("--border-px", type=int, default=42)
    p.add_argument("--border-frac", type=float, default=0.015)
    p.add_argument("--open-radius", type=int, default=1)
    p.add_argument("--close-radius", type=int, default=5)
    p.add_argument("--no-seam-suppress", action="store_true")
    return p.parse_args()



def clean_detector_outputs(output_path: Path, preview_dir: Optional[Path], gds_output_path: Optional[Path]) -> None:
    """Remove stale detector artifacts for repeatable h2p runs."""
    targets: list[Path] = []
    if output_path:
        targets.append(output_path)
    if gds_output_path:
        targets.append(gds_output_path)

    seen: set[Path] = set()
    for target in targets:
        try:
            resolved = target.resolve()
        except Exception:
            resolved = target
        if resolved in seen:
            continue
        seen.add(resolved)
        if target.exists() and target.is_file():
            target.unlink()
            print(f"[Clean] Removed {target}")

    if preview_dir and preview_dir.exists():
        if preview_dir.is_dir():
            shutil.rmtree(preview_dir)
            print(f"[Clean] Removed {preview_dir}")
        else:
            raise SystemExit(f"--preview-dir exists but is not a directory: {preview_dir}")

def main() -> None:
    args = parse_args()
    configure_runtime(cv_threads=int(args.cv_threads), low_priority=not bool(args.normal_priority))
    input_path = Path(args.input)
    gds_output_path = Path(args.gds_output_json) if args.gds_output_json else None

    # Resume labeling without touching detector outputs.  This is deliberately
    # handled before image discovery, template building, cleaning, or detection.
    if bool(args.review_only):
        if gds_output_path is None:
            raise SystemExit("--review-only requires --gds-output-json")
        if not gds_output_path.exists():
            raise SystemExit(f"Existing review JSON not found: {gds_output_path}")
        if not input_path.exists():
            raise SystemExit(f"Review image path not found: {input_path}")
        metadata_path = Path(args.metadata_dir) if args.metadata_dir else None
        if metadata_path is None or not metadata_path.exists():
            raise SystemExit(f"Review metadata directory not found: {args.metadata_dir}")
        review_preview_dir = Path(args.review_preview_dir) if args.review_preview_dir else input_path.parent / "previews"
        state_path = Path(args.review_state) if args.review_state else Path(str(gds_output_path) + ".review_state.json")
        print(f"[Review UI] Reopening existing labels without detection: {gds_output_path}")
        launch_review_ui(
            image_dir=input_path,
            annotations_json=gds_output_path,
            metadata_dir=metadata_path,
            preview_dir=review_preview_dir,
            wafer_id=str(args.wafer_id or ""),
            review_state=state_path,
            quick_review=bool(args.quick_review),
            review_cache_size=int(args.review_cache_size),
            review_autosave_seconds=float(args.review_autosave_seconds),
            review_hide_labels=bool(args.review_hide_labels),
        )
        print(f"[Review UI] Saved reviewed labels to {gds_output_path}")
        print(f"[Review UI] Saved resume position to {state_path}")
        return

    image_paths = list_images(input_path)
    if not image_paths:
        raise SystemExit(f"No cell images found under: {input_path}")

    template = None
    if args.save_template:
        print(f"[Template] Building median template from {len(image_paths)} images...")
        template = build_median_template(
            image_paths,
            max_images=int(args.max_template_images),
            max_width=int(args.max_image_width),
            show_progress=True,
        )
        save_template(template, args.save_template)
        print(f"[Template] Saved {args.save_template} from {template.source_count} images")

    if args.template:
        template = load_template(args.template)
        print(f"[Template] Loaded {args.template} from {template.source_count} images")

    if template is None and bool(args.auto_template) and len(image_paths) >= int(args.auto_template_min_images):
        if args.auto_template_cache:
            auto_cache = Path(args.auto_template_cache)
        elif input_path.is_dir():
            auto_cache = input_path.parent / "normal_template_auto.npz"
        else:
            auto_cache = input_path.parent / "normal_template_auto.npz"
        load_ok = False
        if auto_cache.exists():
            try:
                cached = load_template(auto_cache)
                if cached.source_count >= int(args.auto_template_min_images):
                    template = cached
                    load_ok = True
                    print(f"[Template] Loaded automatic template {auto_cache} from {template.source_count} images")
            except Exception as exc:
                print(f"[Template] Ignoring stale/invalid cache {auto_cache}: {exc}")
        if not load_ok:
            print(f"[Template] Building automatic aligned median+MAD template from {len(image_paths)} cells...")
            template = build_median_template(
                image_paths,
                max_images=int(args.max_template_images),
                max_width=int(args.max_image_width),
                show_progress=True,
            )
            save_template(template, auto_cache)
            print(f"[Template] Saved automatic template {auto_cache} from {template.source_count} images")

    params = DetectorParams(
        threshold=float(args.threshold),
        min_score=float(args.min_score),
        min_area=int(args.min_area),
        max_area_frac=float(args.max_area_frac),
        border_px=int(args.border_px),
        border_frac=float(args.border_frac),
        open_radius=int(args.open_radius),
        close_radius=int(args.close_radius),
        seam_suppress=not bool(args.no_seam_suppress),
    )

    design_provider = None
    if bool(args.use_design_mask) and not bool(args.no_design_mask):
        design_gds = _resolve_design_gds_path(Path(args.config), str(args.design_gds or ""))
        if design_gds is not None:
            design_provider = GDSDesignMaskProvider(
                gds_path=design_gds,
                metadata_dir=Path(args.metadata_dir),
                layers=args.design_layers,
                cache_dir=Path(args.design_mask_cache_dir) if args.design_mask_cache_dir else None,
                rebuild_cache=bool(args.rebuild_design_masks),
            )
            if design_provider.available:
                print(
                    f"[Design Mask] GDS-aware suppression enabled: {design_gds} "
                    f"layers={sorted(design_provider.layers)} polygons={len(design_provider.records)}"
                )
            else:
                print(f"[Design Mask] WARNING: disabled because GDS loading failed: {design_provider.error}")
                design_provider = None
        else:
            print("[Design Mask] WARNING: no GDS path found; continuing without design-aware suppression")

    seam_dir = Path(args.seam_mask_dir) if args.seam_mask_dir else None
    preview_dir = Path(args.preview_dir) if args.preview_dir else None
    out_path = Path(args.output)

    if bool(args.clean_outputs):
        clean_detector_outputs(out_path, preview_dir, gds_output_path)

    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total = 0
    print(
        f"[Runtime] version={DETECTOR_VERSION}, images={len(image_paths)}, max_width={int(args.max_image_width)}, "
        f"cv_threads={int(args.cv_threads)}, low_priority={not bool(args.normal_priority)}, "
        f"heatmaps={bool(args.save_heatmaps)}"
    )
    bar = ProgressBar("Detect", len(image_paths))
    for i, pth in enumerate(image_paths, start=1):
        seam = find_seam_mask_for_image(pth, seam_dir)
        res = detect_cell_defects(
            pth,
            params=params,
            template=template,
            seam_mask=seam,
            max_width=int(args.max_image_width),
            design_mask_provider=design_provider,
        )
        debug = res["_debug"]
        count = len(res["defects"])
        total += count

        if preview_dir:
            img = read_bgr_resized(pth, max_width=int(args.max_image_width))
            overlay = draw_overlay(img, [DefectComponent(**d) for d in res["defects"]], debug["kept_mask"], debug["score"])
            imwrite(preview_dir / f"{pth.stem}_algo_overlay.png", overlay)
            if args.save_heatmaps:
                imwrite(preview_dir / f"{pth.stem}_score_heat.png", score_to_heat(debug["score"]))
                if np.any(debug.get("design_valid_mask", 0)):
                    imwrite(preview_dir / f"{pth.stem}_design_valid.png", debug["design_valid_mask"])
                    imwrite(preview_dir / f"{pth.stem}_design_boundary.png", debug["design_boundary_mask"])

        results.append(strip_debug(res))
        bar.update(i, extra=f"{pth.name}: {count} candidates")
    bar.done(extra=f"total candidates {total}")

    out = {
        "summary": {
            "detector_version": DETECTOR_VERSION,
            "input": str(input_path),
            "image_count": len(image_paths),
            "candidate_count": int(total),
            "params": asdict(params),
            "template": str(args.template or args.save_template or ("auto" if template is not None else "")),
            "template_source_count": int(template.source_count) if template is not None else 0,
            "design_mask_enabled": bool(design_provider is not None),
            "design_gds": str(design_provider.gds_path) if design_provider is not None else "",
            "design_layers": sorted(design_provider.layers) if design_provider is not None else [],
        },
        "images": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[Done] Wrote {out_path} with {total} total candidates")

    if args.gds_output_json:
        if not args.metadata_dir:
            raise SystemExit("--gds-output-json requires --metadata-dir so image pixels can be mapped into GDS coordinates")
        conv = convert_algo_results_to_gds_json(
            out,
            metadata_dir=Path(args.metadata_dir),
            output_json=gds_output_path,
            min_score=float(args.gds_min_score),
            use_polygons=not bool(args.no_polygon_gds),
        )
        print(f"[Done] Wrote subtraction-ready GDS defect JSON: {conv['output']} ({conv['converted']} defects)")

    if args.review_ui:
        if not args.gds_output_json:
            raise SystemExit("--review-ui requires --gds-output-json because the UI edits the subtraction-ready JSON")
        if not args.metadata_dir:
            raise SystemExit("--review-ui requires --metadata-dir so new/manual boxes can be mapped into GDS coordinates")
        review_preview_dir = Path(args.review_preview_dir) if args.review_preview_dir else Path(args.input).parent / "previews"
        print("[Review UI] Opening labeling UI with detected boxes preloaded...")
        state_path = Path(args.review_state) if args.review_state else Path(str(gds_output_path) + ".review_state.json")
        launch_review_ui(
            image_dir=Path(args.input),
            annotations_json=gds_output_path,
            metadata_dir=Path(args.metadata_dir),
            preview_dir=review_preview_dir,
            wafer_id=str(args.wafer_id or ""),
            review_state=state_path,
            quick_review=bool(args.quick_review),
            review_cache_size=int(args.review_cache_size),
            review_autosave_seconds=float(args.review_autosave_seconds),
            review_hide_labels=bool(args.review_hide_labels),
        )
        print(f"[Review UI] Saved reviewed labels to {args.gds_output_json}")


if __name__ == "__main__":
    main()
