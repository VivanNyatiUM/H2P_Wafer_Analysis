"""
Illumination-normalized tile stitching utilities for h2p_device_viewer.

This module is intentionally dependency-light: numpy + opencv only.
It provides two integration points:

1. generate_downscaled_stitch(folder, config)
   Drop-in replacement for wafer_metrology.generate_downscaled_stitch.

2. stitch_local_canvas_from_overlapping_tiles(...)
   Native-resolution local stitching for device crop extraction.

The important changes compared with the old code are:
- flat-field correction per tile to reduce center-bright / edge-dark artifacts
- global brightness matching across tiles
- feather blending in horizontal/vertical overlap zones
- normalized tile caching for fast repeated native crop extraction
"""

from __future__ import annotations

import math
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np

_TILE_RE = re.compile(r"tile_x(\d+)_y(\d+)", re.IGNORECASE)


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


def read_bgr(path: Path | str) -> np.ndarray:
    """Read an image as BGR uint8, with Windows-path-safe np.fromfile decoding."""
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    return img


def resize_if_needed(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == target_w and h == target_h:
        return img
    interp = cv2.INTER_AREA if target_w < w or target_h < h else cv2.INTER_LINEAR
    return cv2.resize(img, (int(target_w), int(target_h)), interpolation=interp)


def normalize_illumination_bgr(
    img: np.ndarray,
    strength: float = 1.0,
    blur_sigma_frac: float = 0.18,
) -> np.ndarray:
    """
    Flat-field illumination correction.

    This estimates a smooth illumination field from the LAB lightness channel,
    then divides by that field. It is designed specifically for center-bright /
    edge-dark tile artifacts.
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

    out_lab = lab.copy()
    out_lab[:, :, 0] = np.clip(L2, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)


def normalize_tile_bgr(
    img: np.ndarray,
    target_luma: Optional[float],
    illumination_enabled: bool = True,
    brightness_match_enabled: bool = True,
    illumination_strength: float = 1.0,
    blur_sigma_frac: float = 0.18,
    brightness_match_strength: float = 0.65,
) -> np.ndarray:
    """Apply illumination correction and global brightness match to one tile."""
    if illumination_enabled:
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
    """Estimate global target luminance from a low-resolution sample of tiles."""
    paths = list(tile_paths)
    if not paths:
        return 128.0

    if len(paths) > max_samples:
        step = max(1, len(paths) // max_samples)
        paths = paths[::step][:max_samples]

    medians = []
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

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float32)
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
        sorted(tiles.values()),
        max_samples=max_samples,
        sample_side=sample_side,
    )


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
    """Convert blended float accumulator to uint8 without allocating giant temp arrays."""
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


def _config_illumination_params(config: dict) -> dict:
    return {
        "illumination_enabled": _cfg_bool(config, "_illumination_enabled", "illumination_normalize", default=True),
        "brightness_match_enabled": _cfg_bool(config, "_brightness_match_enabled", "brightness_match", default=True),
        "illumination_strength": _cfg_float(config, "_illumination_strength", "illumination_strength", default=1.0),
        "blur_sigma_frac": _cfg_float(config, "_illumination_blur_sigma_frac", "illumination_blur_sigma_frac", "blur_sigma_frac", default=0.18),
        "brightness_match_strength": _cfg_float(config, "_brightness_match_strength", "brightness_match_strength", default=0.65),
        "brightness_samples": _cfg_int(config, "_illumination_brightness_samples", "illumination_brightness_samples", default=250),
    }


def generate_downscaled_stitch(folder: Path | str, config: dict, verbose: bool = True) -> tuple[np.ndarray, str]:
    """
    Drop-in replacement for wafer_metrology.generate_downscaled_stitch.

    It keeps the exact configured downscale geometry, but replaces hard overwrite
    stitching with normalized feather blending.
    """
    folder_path = Path(folder)
    tiles, tile_ext = discover_tile_files(folder_path)

    cols, rows = int(config["tile_cols"]), int(config["tile_rows"])
    tw, th = int(config["tile_width"]), int(config["tile_height"])
    ds = float(config["downscale_factor"])
    step_x = tw * (1.0 - float(config["overlap_x_percent"]) / 100.0)
    step_y = th * (1.0 - float(config["overlap_y_percent"]) / 100.0)

    canvas_w = int(((cols - 1) * step_x + tw) * ds)
    canvas_h = int(((rows - 1) * step_y + th) * ds)
    tile_w_ds = max(1, int(tw * ds))
    tile_h_ds = max(1, int(th * ds))

    # Match old geometry while making overlap masks robust.
    step_x_ds = step_x * ds
    step_y_ds = step_y * ds
    overlap_x_ds = max(1, tile_w_ds - int(round(step_x_ds)))
    overlap_y_ds = max(1, tile_h_ds - int(round(step_y_ds)))

    params = _config_illumination_params(config)
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
            f"tile {tile_w_ds}x{tile_h_ds}, target L={target_luma:.2f}"
        )

    acc = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    wacc = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    items = sorted(tiles.items(), key=lambda kv: (kv[0][1], kv[0][0]))
    start = time.time()
    for i, ((col, row), tile_file) in enumerate(items, start=1):
        if col < 1 or col > cols or row < 1 or row > rows:
            continue

        try:
            img = read_bgr(tile_file)
        except Exception:
            continue

        img_ds = resize_if_needed(img, tile_w_ds, tile_h_ds)
        img_ds = normalize_tile_bgr(
            img_ds,
            target_luma=target_luma,
            illumination_enabled=params["illumination_enabled"],
            brightness_match_enabled=params["brightness_match_enabled"],
            illumination_strength=params["illumination_strength"],
            blur_sigma_frac=params["blur_sigma_frac"],
            brightness_match_strength=params["brightness_match_strength"],
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

        acc[y_can:y_can + h_clamp, x_can:x_can + w_clamp] += (
            img_ds[:h_clamp, :w_clamp].astype(np.float32) * weight[:h_clamp, :w_clamp, None]
        )
        wacc[y_can:y_can + h_clamp, x_can:x_can + w_clamp] += weight[:h_clamp, :w_clamp]

        if verbose and (i == 1 or i % 100 == 0 or i == len(items)):
            elapsed = time.time() - start
            pct = 100.0 * i / max(1, len(items))
            sys.stdout.write(f"\r[Illumination Stitch] {i}/{len(items)} ({pct:5.1f}%) | {elapsed/60:4.1f} min")
            sys.stdout.flush()

    if verbose:
        sys.stdout.write("\n")

    ds_canvas = _normalize_accumulator_to_uint8(acc, wacc, chunk_rows=1024)
    return ds_canvas, tile_ext


class NormalizedTileCache:
    """Small LRU cache for full-resolution normalized tiles during crop extraction."""

    def __init__(
        self,
        max_items: int = 24,
        target_luma: Optional[float] = None,
        illumination_enabled: bool = True,
        brightness_match_enabled: bool = True,
        illumination_strength: float = 1.0,
        blur_sigma_frac: float = 0.18,
        brightness_match_strength: float = 0.65,
    ):
        self.max_items = max(0, int(max_items))
        self.target_luma = target_luma
        self.illumination_enabled = illumination_enabled
        self.brightness_match_enabled = brightness_match_enabled
        self.illumination_strength = illumination_strength
        self.blur_sigma_frac = blur_sigma_frac
        self.brightness_match_strength = brightness_match_strength
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, path: Path | str, expected_w: Optional[int] = None, expected_h: Optional[int] = None) -> np.ndarray:
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
        )

        if self.max_items > 0:
            self._cache[key] = img
            while len(self._cache) > self.max_items:
                self._cache.popitem(last=False)

        return img

    def clear(self) -> None:
        self._cache.clear()


def stitch_local_canvas_from_overlapping_tiles(
    folder: Path | str,
    tile_ext: str,
    overlapping_tiles: list[tuple[int, int, int, int, int, int]],
    local_origin: tuple[int, int],
    local_size: tuple[int, int],
    config: dict,
    tile_cache: Optional[NormalizedTileCache] = None,
    excluded_tile_names: Optional[set[str]] = None,
) -> np.ndarray:
    """
    Native-resolution local stitch used before rotating/cropping a device.

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
        acc[oy1:oy2, ox1:ox2] += tile_crop.astype(np.float32) * weight_crop[:, :, None]
        wacc[oy1:oy2, ox1:ox2] += weight_crop

    return _normalize_accumulator_to_uint8(acc, wacc, chunk_rows=1024)
