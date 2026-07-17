"""
Illumination-normalized tile stitching utilities for h2p_device_viewer.

This is a drop-in replacement for the existing illumination_stitching.py, but it
adds the two things that matter for your current stitching artifacts:

1. A shared tile-coordinate flat-field model.
   The center-bright / edge-dark flash pattern is fixed relative to each raw
   tile image, so the correction should be learned across many tiles in tile
   coordinates instead of inferred independently from each device-containing
   tile.

2. Overlap luma leveling.
   After flat-field correction, the tile being inserted is compared against the
   pixels already present in true overlap regions. A robust LAB-L offset is
   applied before feather blending. This specifically attacks the blurred dark
   seam bands that remain after ordinary flat-field correction.

The public API is intentionally backward-compatible:
    generate_downscaled_stitch(folder, config) -> (canvas, tile_ext)
    stitch_local_canvas_from_overlapping_tiles(...) -> canvas

For extraction code that wants seam metadata, call:
    stitch_local_canvas_from_overlapping_tiles(..., return_masks=True)
which returns:
    (canvas, masks)
where masks contains coverage_count, seam_mask, weight_sum, and max_weight.
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

_TILE_RE = re.compile(r"tile_x(\d+)_y(\d+)", re.IGNORECASE)



def _progress_line(label: str, current: int, total: int, start: float, extra: str = "", width: int = 28) -> None:
    total = max(1, int(total))
    current = max(0, min(int(current), total))
    frac = current / total
    filled = int(round(frac * width))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.time() - start
    eta = elapsed * (total - current) / max(current, 1) if current else 0.0
    msg = f"\r[{label}] |{bar}| {current}/{total} {frac*100:5.1f}% elapsed {elapsed/60:4.1f}m"
    if current:
        msg += f" ETA {eta/60:4.1f}m"
    if extra:
        msg += f" | {extra}"
    sys.stdout.write(msg)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _cfg_bool(config: dict, *keys: str, default: bool) -> bool:
    for key in keys:
        if key in config:
            return bool(config[key])
    return default


def _cfg_float(config: dict, *keys: str, default: float) -> float:
    for key in keys:
        if key in config:
            return float(config[key])
    return default


def _cfg_int(config: dict, *keys: str, default: int) -> int:
    for key in keys:
        if key in config:
            return int(config[key])
    return default


def _cfg_str(config: dict, *keys: str, default: str) -> str:
    for key in keys:
        if key in config:
            return str(config[key])
    return default


def _config_illumination_params(config: dict) -> dict:
    """Collect all illumination parameters, supporting old and new keys."""
    return {
        # Existing behavior toggles.
        "illumination_enabled": _cfg_bool(
            config, "_illumination_enabled", "illumination_normalize", default=True
        ),
        "brightness_match_enabled": _cfg_bool(
            config, "_brightness_match_enabled", "brightness_match", default=True
        ),
        "illumination_strength": _cfg_float(
            config,
            "_illumination_strength",
            "illumination_strength",
            default=1.0,
        ),
        "per_tile_blur_sigma_frac": _cfg_float(
            config,
            "_illumination_blur_sigma_frac",
            "illumination_blur_sigma_frac",
            "blur_sigma_frac",
            default=0.18,
        ),
        "brightness_match_strength": _cfg_float(
            config,
            "_brightness_match_strength",
            "brightness_match_strength",
            default=0.65,
        ),
        "brightness_samples": _cfg_int(
            config,
            "_illumination_brightness_samples",
            "illumination_brightness_samples",
            default=250,
        ),
        # New shared flat-field model.
        "shared_flatfield_enabled": _cfg_bool(
            config,
            "_shared_flatfield_enabled",
            "shared_flatfield_enabled",
            default=True,
        ),
        "shared_flatfield_samples": _cfg_int(
            config,
            "_shared_flatfield_samples",
            "shared_flatfield_samples",
            default=400,
        ),
        "shared_flatfield_model_side": _cfg_int(
            config,
            "_shared_flatfield_model_side",
            "shared_flatfield_model_side",
            default=384,
        ),
        "shared_flatfield_smooth_sigma_frac": _cfg_float(
            config,
            "_shared_flatfield_smooth_sigma_frac",
            "shared_flatfield_smooth_sigma_frac",
            default=0.055,
        ),
        "shared_flatfield_clip_low": _cfg_float(
            config,
            "_shared_flatfield_clip_low",
            "shared_flatfield_clip_low",
            default=0.55,
        ),
        "shared_flatfield_clip_high": _cfg_float(
            config,
            "_shared_flatfield_clip_high",
            "shared_flatfield_clip_high",
            default=1.65,
        ),
        "shared_flatfield_cache": _cfg_str(
            config,
            "_shared_flatfield_cache",
            "shared_flatfield_cache",
            default="",
        ),
        # Overlap leveling.
        "overlap_leveling_enabled": _cfg_bool(
            config,
            "_overlap_leveling_enabled",
            "overlap_leveling_enabled",
            default=True,
        ),
        "overlap_leveling_strength": _cfg_float(
            config,
            "_overlap_leveling_strength",
            "overlap_leveling_strength",
            default=0.85,
        ),
        "overlap_leveling_max_delta": _cfg_float(
            config,
            "_overlap_leveling_max_delta",
            "overlap_leveling_max_delta",
            default=18.0,
        ),
        "overlap_leveling_min_pixels": _cfg_int(
            config,
            "_overlap_leveling_min_pixels",
            "overlap_leveling_min_pixels",
            default=2500,
        ),
    }


# ---------------------------------------------------------------------------
# IO and basic image helpers
# ---------------------------------------------------------------------------

def read_bgr(path: Path | str) -> np.ndarray:
    """Read an image as BGR uint8, with Windows-path-safe np.fromfile decoding."""
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    return img



def read_bgr_resized_fast(
    path: Path | str,
    target_w: int,
    target_h: int,
    *,
    expected_w: int | None = None,
    expected_h: int | None = None,
    enabled: bool = True,
) -> np.ndarray:
    """Decode a JPEG near its final size before the normal resize.

    OpenCV's reduced-DCT JPEG modes decode fewer frequency blocks instead of
    decoding the full camera frame and immediately throwing most pixels away.
    A 1/4 decode is used for the 1/12 coarse stitch; non-JPEG inputs and any
    decoder failure fall back to the original full decode path.
    """
    path = Path(path)
    target_w = max(1, int(target_w))
    target_h = max(1, int(target_h))
    img = None
    if enabled and path.suffix.lower() in {".jpg", ".jpeg", ".jpe"}:
        ew = max(1, int(expected_w or target_w))
        eh = max(1, int(expected_h or target_h))
        ratio = max(target_w / float(ew), target_h / float(eh))
        flag = None
        if ratio <= 0.30:
            flag = cv2.IMREAD_REDUCED_COLOR_4
        elif ratio <= 0.60:
            flag = cv2.IMREAD_REDUCED_COLOR_2
        if flag is not None:
            try:
                data = np.fromfile(str(path), dtype=np.uint8)
                img = cv2.imdecode(data, flag)
            except Exception:
                img = None
    if img is None:
        img = read_bgr(path)
    return resize_if_needed(img, target_w, target_h)

def imwrite(path: Path | str, img: np.ndarray, params: Optional[list[int]] = None) -> bool:
    """cv2.imwrite replacement that is safe with Windows paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def resize_if_needed(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == target_w and h == target_h:
        return img
    interp = cv2.INTER_AREA if target_w < w or target_h < h else cv2.INTER_LINEAR
    return cv2.resize(img, (int(target_w), int(target_h)), interpolation=interp)


def _as_lab_l(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    return lab[:, :, 0].astype(np.float32)


def _replace_lab_l(img_bgr: np.ndarray, L_new: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = np.clip(L_new, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Tile discovery and global luminance
# ---------------------------------------------------------------------------

def discover_tile_files(folder: Path | str) -> tuple[dict[tuple[int, int], Path], str]:
    folder = Path(folder)
    tile_files = sorted(folder.glob("tile_x*_y*.*"))
    if not tile_files:
        raise ValueError(f"No grid tile files found in: {folder}")

    tiles: dict[tuple[int, int], Path] = {}
    for p in tile_files:
        m = _TILE_RE.search(p.stem)
        if not m:
            continue
        col, row = int(m.group(1)), int(m.group(2))
        tiles[(col, row)] = p

    if not tiles:
        raise ValueError(f"No valid tile_x###_y### files found in: {folder}")

    tile_ext = next(iter(tiles.values())).suffix
    return tiles, tile_ext


def estimate_global_luma_from_paths(
    tile_paths: Iterable[Path],
    max_samples: int = 250,
    sample_side: int = 256,
) -> float:
    """Estimate global target luminance from low-resolution tile samples."""
    paths = list(tile_paths)
    if not paths:
        return 128.0

    if len(paths) > max_samples:
        step = max(1, len(paths) // max_samples)
        paths = paths[::step][:max_samples]

    medians: list[float] = []
    for p in paths:
        try:
            img = read_bgr(p)
        except Exception:
            continue
        h, w = img.shape[:2]
        scale = min(1.0, float(sample_side) / max(h, w))
        if scale < 1.0:
            img = cv2.resize(
                img,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        L = _as_lab_l(img)
        hh, ww = L.shape
        y0, y1 = int(hh * 0.20), int(hh * 0.80)
        x0, x1 = int(ww * 0.20), int(ww * 0.80)
        crop = L[y0:y1, x0:x1]
        if crop.size:
            medians.append(float(np.median(crop)))
    return float(np.median(medians)) if medians else 128.0


def estimate_global_luma_for_folder(
    folder: Path | str,
    max_samples: int = 250,
    sample_side: int = 256,
) -> float:
    tiles, _ = discover_tile_files(folder)
    return estimate_global_luma_from_paths(
        sorted(tiles.values()), max_samples=max_samples, sample_side=sample_side
    )


# ---------------------------------------------------------------------------
# Shared flat-field model
# ---------------------------------------------------------------------------

@dataclass
class SharedFlatfieldModel:
    """Tile-coordinate multiplicative illumination field for LAB-L.

    field is normalized so median(field) ~= 1. Dark tile-edge regions should
    have field < 1; bright center regions should have field > 1.
    """

    field: np.ndarray
    tile_width: int
    tile_height: int
    source_count: int = 0

    def resized_to(self, width: int, height: int) -> np.ndarray:
        if self.field.shape[1] == width and self.field.shape[0] == height:
            return self.field.astype(np.float32, copy=False)
        return cv2.resize(
            self.field.astype(np.float32),
            (int(width), int(height)),
            interpolation=cv2.INTER_CUBIC,
        )


def _default_flatfield_cache_path(folder: Path | str) -> Path:
    return Path(folder) / ".h2p_shared_flatfield_labL.npz"


def save_shared_flatfield_model(model: SharedFlatfieldModel, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        field=model.field.astype(np.float32),
        tile_width=np.array([model.tile_width], dtype=np.int32),
        tile_height=np.array([model.tile_height], dtype=np.int32),
        source_count=np.array([model.source_count], dtype=np.int32),
    )


def load_shared_flatfield_model(path: Path | str) -> SharedFlatfieldModel:
    data = np.load(str(path))
    return SharedFlatfieldModel(
        field=data["field"].astype(np.float32),
        tile_width=int(data["tile_width"][0]),
        tile_height=int(data["tile_height"][0]),
        source_count=int(data.get("source_count", np.array([0]))[0]),
    )


def _flatfield_model_shape(tile_width: int, tile_height: int, model_side: int) -> tuple[int, int]:
    scale = float(model_side) / max(tile_width, tile_height)
    model_w = max(32, int(round(tile_width * scale)))
    model_h = max(32, int(round(tile_height * scale)))
    return model_w, model_h


def _sample_path_subset(paths: list[Path], max_samples: int) -> list[Path]:
    if len(paths) <= max_samples:
        return paths
    step = max(1, len(paths) // max_samples)
    subset = paths[::step][:max_samples]
    return subset


def _normalized_smooth_luma_sample(
    img_bgr: np.ndarray,
    model_w: int,
    model_h: int,
    smooth_sigma_frac: float,
) -> Optional[np.ndarray]:
    """Return one smooth normalized LAB-L illumination sample.

    The smoothing is deliberate: this should represent the camera/flash field,
    not the vertical device lines.
    """
    img_small = cv2.resize(img_bgr, (model_w, model_h), interpolation=cv2.INTER_AREA)
    L = _as_lab_l(img_small)

    med = float(np.median(L))
    if med <= 1.0 or not np.isfinite(med):
        return None

    sample = L / med

    # Kill small device-line remnants before stacking.
    # Median blur handles thin lines; Gaussian blur handles the smooth field.
    k = max(3, int(round(min(model_w, model_h) * 0.015)) | 1)
    if k >= 3:
        sample = cv2.medianBlur(sample.astype(np.float32), k)

    sigma = max(model_w, model_h) * float(smooth_sigma_frac)
    sample = cv2.GaussianBlur(
        sample.astype(np.float32),
        ksize=(0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT,
    )
    return sample.astype(np.float32)


def build_shared_flatfield_model(
    folder: Path | str,
    config: dict,
    max_samples: Optional[int] = None,
    model_side: Optional[int] = None,
    smooth_sigma_frac: Optional[float] = None,
    verbose: bool = True,
) -> SharedFlatfieldModel:
    """Build a robust shared LAB-L flat-field model from many raw tiles.

    This directly targets your flash pattern: every tile contributes one smooth
    normalized estimate in the *tile's own coordinate system*, then the median
    across tiles becomes the camera/illumination field.
    """
    folder = Path(folder)
    tiles, _ = discover_tile_files(folder)
    params = _config_illumination_params(config)

    tile_width = int(config["tile_width"])
    tile_height = int(config["tile_height"])
    max_samples = int(max_samples or params["shared_flatfield_samples"])
    model_side = int(model_side or params["shared_flatfield_model_side"])
    smooth_sigma_frac = float(
        smooth_sigma_frac or params["shared_flatfield_smooth_sigma_frac"]
    )

    model_w, model_h = _flatfield_model_shape(tile_width, tile_height, model_side)
    paths = _sample_path_subset(sorted(tiles.values()), max_samples)

    samples: list[np.ndarray] = []
    start = time.time()
    for i, p in enumerate(paths, start=1):
        try:
            img = read_bgr(p)
            img = resize_if_needed(img, tile_width, tile_height)
            sample = _normalized_smooth_luma_sample(
                img, model_w=model_w, model_h=model_h, smooth_sigma_frac=smooth_sigma_frac
            )
        except Exception:
            sample = None
        if sample is not None:
            samples.append(sample)

        if verbose and (i == 1 or i % 50 == 0 or i == len(paths)):
            pct = 100.0 * i / max(1, len(paths))
            sys.stdout.write(
                f"\r[Flatfield] samples {i}/{len(paths)} ({pct:5.1f}%) | "
                f"kept {len(samples)} | {(time.time() - start) / 60:4.1f} min"
            )
            sys.stdout.flush()

    if verbose:
        sys.stdout.write("\n")

    if not samples:
        # Neutral model so the rest of the pipeline still works.
        field = np.ones((model_h, model_w), dtype=np.float32)
        return SharedFlatfieldModel(field, tile_width, tile_height, source_count=0)

    stack = np.stack(samples, axis=0).astype(np.float32)
    field = np.median(stack, axis=0).astype(np.float32)

    # Final aggressive smoothing: this field should not know about device lines.
    sigma = max(model_w, model_h) * smooth_sigma_frac
    field = cv2.GaussianBlur(
        field,
        ksize=(0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT,
    )

    med = float(np.median(field))
    if med <= 0 or not np.isfinite(med):
        med = 1.0
    field = field / med

    low = float(params["shared_flatfield_clip_low"])
    high = float(params["shared_flatfield_clip_high"])
    field = np.clip(field, low, high).astype(np.float32)

    return SharedFlatfieldModel(
        field=field,
        tile_width=tile_width,
        tile_height=tile_height,
        source_count=len(samples),
    )


def get_or_build_shared_flatfield_model(
    folder: Path | str,
    config: dict,
    verbose: bool = True,
) -> Optional[SharedFlatfieldModel]:
    params = _config_illumination_params(config)
    if not params["illumination_enabled"] or not params["shared_flatfield_enabled"]:
        return None

    cache_raw = params["shared_flatfield_cache"].strip()
    cache_path = Path(cache_raw) if cache_raw else _default_flatfield_cache_path(folder)

    if cache_path.exists():
        try:
            model = load_shared_flatfield_model(cache_path)
            if (
                model.tile_width == int(config["tile_width"])
                and model.tile_height == int(config["tile_height"])
            ):
                if verbose:
                    print(
                        f"[Flatfield] Loaded shared model: {cache_path} "
                        f"({model.field.shape[1]}x{model.field.shape[0]}, "
                        f"sources={model.source_count})"
                    )
                config["_shared_flatfield_cache_resolved"] = str(cache_path)
                return model
        except Exception as exc:
            if verbose:
                print(f"[Flatfield] Could not load cache {cache_path}: {exc}")

    model = build_shared_flatfield_model(folder, config, verbose=verbose)
    try:
        save_shared_flatfield_model(model, cache_path)
        config["_shared_flatfield_cache_resolved"] = str(cache_path)
        if verbose:
            print(f"[Flatfield] Saved shared model: {cache_path}")
    except Exception as exc:
        if verbose:
            print(f"[Flatfield] Warning: could not save model cache: {exc}")
    return model


# ---------------------------------------------------------------------------
# Per-tile and shared illumination correction
# ---------------------------------------------------------------------------

def normalize_illumination_bgr(
    img: np.ndarray,
    strength: float = 1.0,
    blur_sigma_frac: float = 0.18,
) -> np.ndarray:
    """Legacy per-tile flat-field correction.

    Kept as a fallback. It is useful, but less correct than the shared model
    because it estimates illumination from a tile that contains device content.
    """
    if strength <= 0.0:
        return img

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    h, w = L.shape

    max_bg_side = 512
    bg_scale = min(1.0, max_bg_side / max(h, w))
    if bg_scale < 1.0:
        small = cv2.resize(
            L,
            (max(16, int(w * bg_scale)), max(16, int(h * bg_scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = L

    small_h, small_w = small.shape
    sigma = max(small_h, small_w) * float(blur_sigma_frac)
    bg_small = cv2.GaussianBlur(
        small,
        ksize=(0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT,
    )
    bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_CUBIC)

    median_bg = float(np.median(bg))
    bg = np.maximum(bg, 1.0)
    corrected_L = L / bg * median_bg
    corrected_L = (1.0 - strength) * L + strength * corrected_L

    out_lab = lab.copy()
    out_lab[:, :, 0] = np.clip(corrected_L, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)


def apply_shared_flatfield_bgr(
    img: np.ndarray,
    model: SharedFlatfieldModel,
    strength: float = 1.0,
) -> np.ndarray:
    """Apply shared tile-coordinate flat-field correction to LAB-L."""
    if strength <= 0.0 or model is None:
        return img

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    h, w = L.shape
    field = model.resized_to(w, h).astype(np.float32)
    field = np.maximum(field, 1e-3)

    corrected_L = L / field

    # Preserve the tile's median L so brightness matching can do global leveling.
    med_before = float(np.median(L))
    med_after = float(np.median(corrected_L))
    if med_after > 1e-3 and np.isfinite(med_after):
        corrected_L *= med_before / med_after

    L2 = (1.0 - strength) * L + strength * corrected_L
    lab[:, :, 0] = np.clip(L2, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def match_tile_luma_bgr(
    img: np.ndarray,
    target_luma: float,
    strength: float = 0.65,
) -> np.ndarray:
    """Nudge a tile toward the global median LAB-L luminance."""
    if strength <= 0.0 or target_luma is None:
        return img

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    current = float(np.median(L))
    delta = float(target_luma) - current
    L2 = L + strength * delta
    lab[:, :, 0] = np.clip(L2, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def normalize_tile_bgr(
    img: np.ndarray,
    target_luma: Optional[float],
    illumination_enabled: bool = True,
    brightness_match_enabled: bool = True,
    illumination_strength: float = 1.0,
    blur_sigma_frac: float = 0.18,
    brightness_match_strength: float = 0.65,
    shared_flatfield_model: Optional[SharedFlatfieldModel] = None,
) -> np.ndarray:
    """Apply illumination correction and global brightness match to one tile."""
    if illumination_enabled:
        if shared_flatfield_model is not None:
            img = apply_shared_flatfield_bgr(
                img, model=shared_flatfield_model, strength=illumination_strength
            )
        else:
            img = normalize_illumination_bgr(
                img,
                strength=illumination_strength,
                blur_sigma_frac=blur_sigma_frac,
            )

    if brightness_match_enabled and target_luma is not None:
        img = match_tile_luma_bgr(
            img,
            target_luma=target_luma,
            strength=brightness_match_strength,
        )
    return img


# ---------------------------------------------------------------------------
# Blending and overlap leveling
# ---------------------------------------------------------------------------

def make_feather_weight(
    h: int,
    w: int,
    overlap_x: int,
    overlap_y: int,
    has_left: bool,
    has_right: bool,
    has_top: bool,
    has_bottom: bool,
) -> np.ndarray:
    """Create a separable 2D feather mask for overlap blending."""
    wx = np.ones(int(w), dtype=np.float32)
    wy = np.ones(int(h), dtype=np.float32)

    overlap_x = int(min(max(overlap_x, 0), max(1, w // 2)))
    overlap_y = int(min(max(overlap_y, 0), max(1, h // 2)))

    if overlap_x > 1:
        if has_left:
            wx[:overlap_x] *= np.linspace(0.0, 1.0, overlap_x, endpoint=False, dtype=np.float32)
        if has_right:
            wx[-overlap_x:] *= np.linspace(1.0, 0.0, overlap_x, endpoint=False, dtype=np.float32)

    if overlap_y > 1:
        if has_top:
            wy[:overlap_y] *= np.linspace(0.0, 1.0, overlap_y, endpoint=False, dtype=np.float32)
        if has_bottom:
            wy[-overlap_y:] *= np.linspace(1.0, 0.0, overlap_y, endpoint=False, dtype=np.float32)

    return (wy[:, None] * wx[None, :]).astype(np.float32)


def _normalize_accumulator_to_uint8(acc: np.ndarray, wacc: np.ndarray, chunk_rows: int = 1024) -> np.ndarray:
    """Convert blended float accumulator to uint8 without giant temporaries."""
    h, w = wacc.shape
    out = np.empty((h, w, 3), dtype=np.uint8)
    for y0 in range(0, h, chunk_rows):
        y1 = min(h, y0 + chunk_rows)
        weights = wacc[y0:y1]
        safe_w = np.maximum(weights, 1e-6)
        chunk = acc[y0:y1] / safe_w[:, :, None]
        chunk = np.clip(chunk, 0, 255).astype(np.uint8)
        no_coverage = weights <= 1e-6
        chunk[no_coverage] = 0
        out[y0:y1] = chunk
    return out


def _robust_overlap_delta_l(
    tile_crop_bgr: np.ndarray,
    acc_region: np.ndarray,
    wacc_region: np.ndarray,
    candidate_weight: Optional[np.ndarray] = None,
    min_pixels: int = 2500,
) -> Optional[float]:
    """Median LAB-L difference between existing canvas and new tile in overlap."""
    existing_mask = wacc_region > 1e-4
    if candidate_weight is not None:
        existing_mask &= candidate_weight > 1e-4

    if int(np.count_nonzero(existing_mask)) < int(min_pixels):
        return None

    # Compare low-frequency L only. This makes the estimate insensitive to the
    # vertical device lines and most defects.
    existing = acc_region / np.maximum(wacc_region[:, :, None], 1e-6)
    existing_u8 = np.clip(existing, 0, 255).astype(np.uint8)

    L_existing = _as_lab_l(existing_u8)
    L_tile = _as_lab_l(tile_crop_bgr)

    sigma = max(3.0, max(tile_crop_bgr.shape[:2]) * 0.015)
    L_existing_s = cv2.GaussianBlur(L_existing, (0, 0), sigmaX=sigma, sigmaY=sigma)
    L_tile_s = cv2.GaussianBlur(L_tile, (0, 0), sigmaX=sigma, sigmaY=sigma)

    diff = (L_existing_s - L_tile_s)[existing_mask]
    if diff.size < min_pixels:
        return None

    lo, hi = np.percentile(diff, [10, 90])
    trimmed = diff[(diff >= lo) & (diff <= hi)]
    if trimmed.size < min_pixels // 4:
        trimmed = diff
    delta = float(np.median(trimmed))
    if not np.isfinite(delta):
        return None
    return delta


def _apply_luma_delta_bgr(img_bgr: np.ndarray, delta_l: float, strength: float, max_delta: float) -> np.ndarray:
    if delta_l is None or strength <= 0.0:
        return img_bgr
    delta = float(np.clip(delta_l, -max_delta, max_delta)) * float(strength)
    if abs(delta) < 0.05:
        return img_bgr
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    lab[:, :, 0] = np.clip(L + delta, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _level_tile_against_accumulator(
    tile_crop_bgr: np.ndarray,
    acc_region: np.ndarray,
    wacc_region: np.ndarray,
    weight_crop: np.ndarray,
    strength: float,
    max_delta: float,
    min_pixels: int,
) -> np.ndarray:
    delta = _robust_overlap_delta_l(
        tile_crop_bgr=tile_crop_bgr,
        acc_region=acc_region,
        wacc_region=wacc_region,
        candidate_weight=weight_crop,
        min_pixels=min_pixels,
    )
    if delta is None:
        return tile_crop_bgr
    return _apply_luma_delta_bgr(tile_crop_bgr, delta, strength=strength, max_delta=max_delta)


# ---------------------------------------------------------------------------
# Downscaled full-wafer stitch
# ---------------------------------------------------------------------------

def generate_downscaled_stitch(
    folder: Path | str,
    config: dict,
    verbose: bool = True,
) -> tuple[np.ndarray, str]:
    """Drop-in replacement for wafer_metrology.generate_downscaled_stitch.

    Keeps the configured geometry, but applies shared flat-field correction,
    overlap luma leveling, and feather blending.
    """
    folder_path = Path(folder)
    tiles, tile_ext = discover_tile_files(folder_path)

    cols, rows = int(config["tile_cols"]), int(config["tile_rows"])
    tw, th = int(config["tile_width"]), int(config["tile_height"])

    # The CLI/config value named "downscale" is a divisor: 4 means 1/4-size
    # tiles. The stitcher works with the multiplier/factor: 0.25. Prefer the
    # explicit divisor when present so stale downscale_factor values cannot
    # accidentally keep the canvas at 20x downsampled resolution.
    if "_stitch_downscale_divisor" in config and float(config["_stitch_downscale_divisor"]) > 0:
        ds = 1.0 / float(config["_stitch_downscale_divisor"])
    elif "downscale" in config and float(config["downscale"]) > 0:
        ds = 1.0 / float(config["downscale"])
    else:
        ds = float(config.get("downscale_factor", 1.0 / 20.0))
    config["downscale_factor"] = float(ds)
    config["downscale"] = float(1.0 / ds) if ds > 0 else 20.0

    step_x = tw * (1.0 - float(config["overlap_x_percent"]) / 100.0)
    step_y = th * (1.0 - float(config["overlap_y_percent"]) / 100.0)

    canvas_w = int(((cols - 1) * step_x + tw) * ds)
    canvas_h = int(((rows - 1) * step_y + th) * ds)
    tile_w_ds = max(1, int(tw * ds))
    tile_h_ds = max(1, int(th * ds))

    step_x_ds = step_x * ds
    step_y_ds = step_y * ds
    overlap_x_ds = max(1, tile_w_ds - int(round(step_x_ds)))
    overlap_y_ds = max(1, tile_h_ds - int(round(step_y_ds)))

    params = _config_illumination_params(config)
    flatfield_model = get_or_build_shared_flatfield_model(folder_path, config, verbose=verbose)

    target_luma = config.get("_illumination_target_luma")
    if target_luma is None:
        target_luma = estimate_global_luma_from_paths(
            sorted(tiles.values()),
            max_samples=params["brightness_samples"],
        )
        config["_illumination_target_luma"] = float(target_luma)

    if verbose:
        print(
            f"[Illumination Stitch] Downscaled stitch: {canvas_w}x{canvas_h}, "
            f"tile {tile_w_ds}x{tile_h_ds}, target L={float(target_luma):.2f}, "
            f"shared_flatfield={flatfield_model is not None}"
        )

    acc = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    wacc = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    items = sorted(tiles.items(), key=lambda kv: (kv[0][1], kv[0][0]))
    start = time.time()

    for i, ((col, row), tile_file) in enumerate(items, start=1):
        if col < 1 or col > cols or row < 1 or row > rows:
            continue
        try:
            img_ds = read_bgr_resized_fast(
                tile_file,
                tile_w_ds,
                tile_h_ds,
                expected_w=tw,
                expected_h=th,
                enabled=bool(config.get("_fast_jpeg_decode", True)),
            )
        except Exception:
            continue
        img_ds = normalize_tile_bgr(
            img_ds,
            target_luma=target_luma,
            illumination_enabled=params["illumination_enabled"],
            brightness_match_enabled=params["brightness_match_enabled"],
            illumination_strength=params["illumination_strength"],
            blur_sigma_frac=params["per_tile_blur_sigma_frac"],
            brightness_match_strength=params["brightness_match_strength"],
            shared_flatfield_model=flatfield_model,
        )

        x_can = max(0, int(((col - 1) * step_x) * ds))
        y_can = max(0, int(((row - 1) * step_y) * ds))
        h_ds, w_ds = img_ds.shape[:2]
        h_clamp = min(h_ds, canvas_h - y_can)
        w_clamp = min(w_ds, canvas_w - x_can)
        if h_clamp <= 0 or w_clamp <= 0:
            continue

        has_left = (col - 1, row) in tiles
        has_right = (col + 1, row) in tiles
        has_top = (col, row - 1) in tiles
        has_bottom = (col, row + 1) in tiles
        weight = make_feather_weight(
            h_ds,
            w_ds,
            overlap_x_ds,
            overlap_y_ds,
            has_left=has_left,
            has_right=has_right,
            has_top=has_top,
            has_bottom=has_bottom,
        )

        tile_crop = img_ds[:h_clamp, :w_clamp]
        weight_crop = weight[:h_clamp, :w_clamp]

        if params["overlap_leveling_enabled"]:
            tile_crop = _level_tile_against_accumulator(
                tile_crop_bgr=tile_crop,
                acc_region=acc[y_can:y_can + h_clamp, x_can:x_can + w_clamp],
                wacc_region=wacc[y_can:y_can + h_clamp, x_can:x_can + w_clamp],
                weight_crop=weight_crop,
                strength=params["overlap_leveling_strength"],
                max_delta=params["overlap_leveling_max_delta"],
                min_pixels=max(64, int(params["overlap_leveling_min_pixels"] * ds * ds)),
            )

        acc[y_can:y_can + h_clamp, x_can:x_can + w_clamp] += (
            tile_crop.astype(np.float32) * weight_crop[:, :, None]
        )
        wacc[y_can:y_can + h_clamp, x_can:x_can + w_clamp] += weight_crop

        if verbose and (i == 1 or i % 100 == 0 or i == len(items)):
            elapsed = time.time() - start
            pct = 100.0 * i / max(1, len(items))
            sys.stdout.write(
                f"\r[Illumination Stitch] {i}/{len(items)} ({pct:5.1f}%) | "
                f"{elapsed / 60:4.1f} min"
            )
            sys.stdout.flush()

    if verbose:
        sys.stdout.write("\n")

    ds_canvas = _normalize_accumulator_to_uint8(acc, wacc, chunk_rows=1024)
    return ds_canvas, tile_ext


# ---------------------------------------------------------------------------
# Native tile cache and local crop stitch
# ---------------------------------------------------------------------------

class NormalizedTileCache:
    """Small LRU cache for full-resolution normalized tiles during extraction."""

    def __init__(
        self,
        max_items: int = 24,
        target_luma: Optional[float] = None,
        illumination_enabled: bool = True,
        brightness_match_enabled: bool = True,
        illumination_strength: float = 1.0,
        blur_sigma_frac: float = 0.18,
        brightness_match_strength: float = 0.65,
        shared_flatfield_model: Optional[SharedFlatfieldModel] = None,
    ):
        self.max_items = max(0, int(max_items))
        self.target_luma = target_luma
        self.illumination_enabled = illumination_enabled
        self.brightness_match_enabled = brightness_match_enabled
        self.illumination_strength = illumination_strength
        self.blur_sigma_frac = blur_sigma_frac
        self.brightness_match_strength = brightness_match_strength
        self.shared_flatfield_model = shared_flatfield_model
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(
        self,
        path: Path | str,
        expected_w: Optional[int] = None,
        expected_h: Optional[int] = None,
    ) -> np.ndarray:
        key = str(Path(path))
        if self.max_items > 0 and key in self._cache:
            img = self._cache.pop(key)
            self._cache[key] = img
            return img

        img = read_bgr(path)
        if expected_w is not None and expected_h is not None:
            img = resize_if_needed(img, int(expected_w), int(expected_h))

        img = normalize_tile_bgr(
            img,
            target_luma=self.target_luma,
            illumination_enabled=self.illumination_enabled,
            brightness_match_enabled=self.brightness_match_enabled,
            illumination_strength=self.illumination_strength,
            blur_sigma_frac=self.blur_sigma_frac,
            brightness_match_strength=self.brightness_match_strength,
            shared_flatfield_model=self.shared_flatfield_model,
        )

        if self.max_items > 0:
            self._cache[key] = img
            while len(self._cache) > self.max_items:
                self._cache.popitem(last=False)
        return img

    def clear(self) -> None:
        self._cache.clear()


def _build_local_masks(
    coverage_count: np.ndarray,
    wacc: np.ndarray,
    max_weight: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build seam metadata masks from local stitch accumulators."""
    coverage_u8 = np.clip(coverage_count, 0, 255).astype(np.uint8)
    weight_sum = np.clip(wacc / max(float(np.max(wacc)), 1e-6) * 255.0, 0, 255).astype(np.uint8)
    max_weight_u8 = np.clip(max_weight * 255.0, 0, 255).astype(np.uint8)

    seam = (coverage_count > 1.0).astype(np.uint8) * 255
    if np.any(seam):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        seam = cv2.dilate(seam, kernel, iterations=1)

    return {
        "coverage_count": coverage_u8,
        "weight_sum": weight_sum,
        "max_weight": max_weight_u8,
        "seam_mask": seam,
    }


def stitch_local_canvas_from_overlapping_tiles(
    folder: Path | str,
    tile_ext: str,
    overlapping_tiles: list[tuple[int, int, int, int, int, int]],
    local_origin: tuple[int, int],
    local_size: tuple[int, int],
    config: dict,
    tile_cache: Optional[NormalizedTileCache] = None,
    excluded_tile_names: Optional[set[str]] = None,
    return_masks: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Native-resolution local stitch used before rotating/cropping a device.

    overlapping_tiles entries are:
        (col, row, tile_x1, tile_y1, tile_x2, tile_y2)
    where tile coordinates are global stitched-canvas pixels.
    """
    folder = Path(folder)
    x1, y1 = map(int, local_origin)
    local_w, local_h = map(int, local_size)

    tw, th = int(config["tile_width"]), int(config["tile_height"])
    cols, rows = int(config["tile_cols"]), int(config["tile_rows"])
    overlap_x = max(1, int(round(tw * float(config["overlap_x_percent"]) / 100.0)))
    overlap_y = max(1, int(round(th * float(config["overlap_y_percent"]) / 100.0)))
    params = _config_illumination_params(config)

    excluded_tile_names = excluded_tile_names or set()

    def tile_available(c: int, r: int) -> bool:
        if c < 1 or c > cols or r < 1 or r > rows:
            return False
        name = f"tile_x{c:03d}_y{r:03d}{tile_ext}"
        if name in excluded_tile_names:
            return False
        return (folder / name).exists()

    acc = np.zeros((local_h, local_w, 3), dtype=np.float32)
    wacc = np.zeros((local_h, local_w), dtype=np.float32)
    coverage_count = np.zeros((local_h, local_w), dtype=np.uint16)
    max_weight = np.zeros((local_h, local_w), dtype=np.float32)

    for c_col, r_row, tx1, ty1, tx2, ty2 in overlapping_tiles:
        tile_path = folder / f"tile_x{c_col:03d}_y{r_row:03d}{tile_ext}"
        if not tile_path.exists():
            continue

        try:
            if tile_cache is not None:
                tile_img = tile_cache.get(tile_path, expected_w=tw, expected_h=th)
            else:
                tile_img = read_bgr(tile_path)
                tile_img = resize_if_needed(tile_img, tw, th)
                target_luma = config.get("_illumination_target_luma")
                shared_model = config.get("_shared_flatfield_model_obj")
                tile_img = normalize_tile_bgr(
                    tile_img,
                    target_luma=target_luma,
                    illumination_enabled=params["illumination_enabled"],
                    brightness_match_enabled=params["brightness_match_enabled"],
                    illumination_strength=params["illumination_strength"],
                    blur_sigma_frac=params["per_tile_blur_sigma_frac"],
                    brightness_match_strength=params["brightness_match_strength"],
                    shared_flatfield_model=shared_model,
                )
        except Exception:
            continue

        loc_tx1, loc_ty1 = tx1 - x1, ty1 - y1
        loc_tx2, loc_ty2 = tx2 - x1, ty2 - y1

        ox1, oy1 = max(0, loc_tx1), max(0, loc_ty1)
        ox2, oy2 = min(local_w, loc_tx2), min(local_h, loc_ty2)
        if ox2 <= ox1 or oy2 <= oy1:
            continue

        sx1, sy1 = ox1 - loc_tx1, oy1 - loc_ty1
        sx2, sy2 = sx1 + (ox2 - ox1), sy1 + (oy2 - oy1)

        has_left = tile_available(c_col - 1, r_row)
        has_right = tile_available(c_col + 1, r_row)
        has_top = tile_available(c_col, r_row - 1)
        has_bottom = tile_available(c_col, r_row + 1)
        weight = make_feather_weight(
            th,
            tw,
            overlap_x,
            overlap_y,
            has_left=has_left,
            has_right=has_right,
            has_top=has_top,
            has_bottom=has_bottom,
        )

        tile_crop = tile_img[sy1:sy2, sx1:sx2]
        weight_crop = weight[sy1:sy2, sx1:sx2]

        if params["overlap_leveling_enabled"]:
            tile_crop = _level_tile_against_accumulator(
                tile_crop_bgr=tile_crop,
                acc_region=acc[oy1:oy2, ox1:ox2],
                wacc_region=wacc[oy1:oy2, ox1:ox2],
                weight_crop=weight_crop,
                strength=params["overlap_leveling_strength"],
                max_delta=params["overlap_leveling_max_delta"],
                min_pixels=params["overlap_leveling_min_pixels"],
            )

        acc[oy1:oy2, ox1:ox2] += tile_crop.astype(np.float32) * weight_crop[:, :, None]
        wacc[oy1:oy2, ox1:ox2] += weight_crop
        coverage_count[oy1:oy2, ox1:ox2] += (weight_crop > 1e-4).astype(np.uint16)
        max_weight[oy1:oy2, ox1:ox2] = np.maximum(max_weight[oy1:oy2, ox1:ox2], weight_crop)

    canvas = _normalize_accumulator_to_uint8(acc, wacc, chunk_rows=1024)
    if not return_masks:
        return canvas

    masks = _build_local_masks(coverage_count=coverage_count, wacc=wacc, max_weight=max_weight)
    return canvas, masks


# ---------------------------------------------------------------------------
# Optional metadata helpers for extraction integration
# ---------------------------------------------------------------------------

def write_stitch_masks(mask_dir: Path | str, stem: str, masks: dict[str, np.ndarray]) -> None:
    """Write seam/coverage masks beside cell crops."""
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in masks.items():
        imwrite(mask_dir / f"{stem}_{name}.png", arr)


def save_cell_metadata_json(path: Path | str, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
