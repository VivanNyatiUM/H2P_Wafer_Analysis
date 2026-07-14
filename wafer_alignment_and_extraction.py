import argparse
import copy
import json
import shutil
import math
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import centroid_algorithm
import coordinate_transformer
import defect_mapper_gui
import gds_parser
import illumination_stitching
import large_wafer_tester
import wafer_align_gui
import wafer_metrology


WAFER_EXTRACTION_VERSION = "simple-cli-create-2026-07-13"


# ===========================================================================
# 1. IO / CONFIG UTILITIES
# ===========================================================================


def load_config(config_path="config.json"):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_defect_json(json_path):
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Defect JSON file not found at: {json_path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_grid_size(tile_folder):
    folder = Path(tile_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Tile folder does not exist at: {tile_folder}")

    pattern = re.compile(r"tile_x(\d+)_y(\d+)")
    max_col = 0
    max_row = 0
    for file in folder.iterdir():
        if not file.is_file():
            continue
        match = pattern.search(file.name)
        if not match:
            continue
        col = int(match.group(1))
        row = int(match.group(2))
        max_col = max(max_col, col)
        max_row = max(max_row, row)

    if max_col == 0 or max_row == 0:
        raise ValueError(f"No valid tile files matched inside folder: {tile_folder}")
    return max_col, max_row


def load_exclusions(exclusions_path):
    path = Path(exclusions_path)
    if not path.exists():
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return set(json.load(f))
    except Exception:
        return set()


def apply_stitch_downscale_to_config(config_run: dict, args) -> None:
    """Normalize downscale settings.

    CLI --stitch-downscale is a DIVISOR: 4 means raw 3000px tile -> 750px
    tile in the stitched canvas. The stitcher uses downscale_factor, which is
    the multiplier: 4 -> 0.25. Keep both values synchronized so the CLI cannot
    silently fall back to the old config value, e.g. 20 -> 0.05.
    """
    stitch_downscale = float(getattr(args, "stitch_downscale", 0) or 0)
    if stitch_downscale > 0:
        config_run["downscale"] = stitch_downscale
        config_run["downscale_factor"] = 1.0 / stitch_downscale
        config_run["_stitch_downscale_divisor"] = stitch_downscale
        return

    # No CLI override. Accept either old-style divisor or direct factor.
    if "downscale" in config_run and float(config_run["downscale"]) > 0:
        config_run["downscale"] = float(config_run["downscale"])
        config_run["downscale_factor"] = 1.0 / float(config_run["downscale"])
        config_run["_stitch_downscale_divisor"] = float(config_run["downscale"])
    elif "downscale_factor" in config_run and float(config_run["downscale_factor"]) > 0:
        config_run["downscale_factor"] = float(config_run["downscale_factor"])
        config_run["downscale"] = 1.0 / float(config_run["downscale_factor"])
        config_run["_stitch_downscale_divisor"] = float(config_run["downscale"])
    else:
        config_run["downscale"] = 20.0
        config_run["downscale_factor"] = 1.0 / 20.0
        config_run["_stitch_downscale_divisor"] = 20.0


def apply_illumination_cli_to_config(config_run: dict, args) -> None:
    """Push all stitching/illumination CLI settings into config_run.

    wafer_metrology.generate_downscaled_stitch() and the full-resolution local
    crop stitcher both read from this dict, so keeping the values here prevents
    the coarse preview and native cell crops from silently using different
    correction parameters.
    """
    config_run["_illumination_enabled"] = not args.no_illumination_normalize
    config_run["_brightness_match_enabled"] = not args.no_brightness_match
    config_run["_illumination_strength"] = float(args.illumination_strength)
    config_run["_illumination_blur_sigma_frac"] = float(args.illumination_blur_sigma_frac)
    config_run["_brightness_match_strength"] = float(args.brightness_match_strength)
    config_run["_illumination_brightness_samples"] = int(args.illumination_brightness_samples)

    # Balanced-resolution fast mode: lower divisor means a larger stitched
    # canvas and sharper cell crops. Example: --stitch-downscale 4 means
    # downscale_factor 0.25, so a 3000px tile becomes 750px in the stitch.
    apply_stitch_downscale_to_config(config_run, args)

    # Newer stitcher controls. Defaults are ON because this file is intended for
    # the algorithmic-defect workflow where seam suppression matters.
    config_run["_shared_flatfield_enabled"] = not args.no_shared_flatfield
    config_run["_overlap_leveling_enabled"] = not args.no_overlap_leveling
    config_run["_shared_flatfield_samples"] = int(args.shared_flatfield_samples)
    config_run["_shared_flatfield_model_side"] = int(args.shared_flatfield_model_side)
    config_run["_overlap_leveling_strength"] = float(args.overlap_leveling_strength)


def _as_float_or_zero(value, key_name: str) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Config value {key_name!r} must be numeric, got {value!r}") from exc


def get_auto_alignment_translation_correction_um(config: dict) -> tuple[float, float]:
    """Return the hardcoded translation correction added to SVD alignment.

    Supported config names:
      - auto_alignment_translation_correction_um: {"x": ..., "y": ...}
      - auto_alignment_translation_offset_um: {"x": ..., "y": ...}
      - alignment_translation_correction_um: {"x": ..., "y": ...}
      - top-level *_x_um / *_y_um aliases
    """
    x_corr = 0.0
    y_corr = 0.0

    for block_name in (
        "auto_alignment_translation_correction_um",
        "auto_alignment_translation_offset_um",
        "alignment_translation_correction_um",
    ):
        block = config.get(block_name)
        if isinstance(block, dict):
            if "x" in block:
                x_corr = _as_float_or_zero(block.get("x"), f"{block_name}.x")
            if "x_um" in block:
                x_corr = _as_float_or_zero(block.get("x_um"), f"{block_name}.x_um")
            if "y" in block:
                y_corr = _as_float_or_zero(block.get("y"), f"{block_name}.y")
            if "y_um" in block:
                y_corr = _as_float_or_zero(block.get("y_um"), f"{block_name}.y_um")

    top_level_aliases = [
        ("auto_alignment_translation_correction_x_um", "x"),
        ("auto_alignment_translation_correction_y_um", "y"),
        ("auto_alignment_translation_offset_x_um", "x"),
        ("auto_alignment_translation_offset_y_um", "y"),
        ("alignment_translation_correction_x_um", "x"),
        ("alignment_translation_correction_y_um", "y"),
    ]
    for key, axis in top_level_aliases:
        if key not in config:
            continue
        if axis == "x":
            x_corr = _as_float_or_zero(config.get(key), key)
        else:
            y_corr = _as_float_or_zero(config.get(key), key)

    return x_corr, y_corr


def apply_auto_alignment_translation_correction(
    flat_angle: float,
    x_offset_um: float,
    y_offset_um: float,
    scale_mult: float,
    config: dict,
    out_stem: str,
) -> tuple[float, float, float, float]:
    x_corr, y_corr = get_auto_alignment_translation_correction_um(config)
    if abs(x_corr) < 1e-12 and abs(y_corr) < 1e-12:
        return flat_angle, x_offset_um, y_offset_um, scale_mult

    corrected_x = float(x_offset_um) + x_corr
    corrected_y = float(y_offset_um) + y_corr
    print(f"[{out_stem} Auto-Align] Applying config translation correction:")
    print(f" Correction Added: X={x_corr:+.3f} um, Y={y_corr:+.3f} um")
    print(f" Corrected Translation: X={corrected_x:.3f} um, Y={corrected_y:.3f} um")
    print(f" GUI Initial Translation: X={-corrected_x:.3f} um, Y={-corrected_y:.3f} um")
    return flat_angle, corrected_x, corrected_y, scale_mult


# ===========================================================================
# 2. ALIGNMENT / EXTRACTION CORE
# ===========================================================================


def _resolve_alignment_from_centroid_tester(
    tester,
    markers,
    gds_R,
    canvas_xc,
    canvas_yc,
    canvas_R,
    gds_xc,
    gds_yc,
    out_stem,
):
    """Resolve global alignment from centroid marker boxes using SVD/Umeyama."""
    phys_list = []
    nom_list = []

    left_squares = [m for m in markers["left"] if m.get("type") == "square"]
    right_squares = [m for m in markers["right"] if m.get("type") == "square"]

    left_source = left_squares if left_squares else markers["left"]
    right_source = right_squares if right_squares else markers["right"]

    left_gds_cx = np.mean([m["center"][0] for m in left_source])
    left_gds_cy = np.mean([m["center"][1] for m in left_source])
    right_gds_cx = np.mean([m["center"][0] for m in right_source])
    right_gds_cy = np.mean([m["center"][1] for m in right_source])

    if getattr(tester, "left_boxes_global", None):
        for (row, col), corners in tester.left_boxes_global.items():
            if len(corners) != 4:
                continue
            target_x = left_gds_cx + centroid_algorithm.NOMINAL_COORDS[(row, col)][0]
            target_y = left_gds_cy - centroid_algorithm.NOMINAL_COORDS[(row, col)][1]
            best_match = None
            min_dist = float("inf")
            for m in left_squares:
                dist = math.hypot(m["center"][0] - target_x, m["center"][1] - target_y)
                if dist < min_dist:
                    min_dist = dist
                    best_match = m
            if best_match is not None and min_dist < 200.0:
                min_x, min_y, max_x, max_y = best_match["bbox"]
                nom_corners = [
                    (min_x, max_y),
                    (max_x, max_y),
                    (max_x, min_y),
                    (min_x, min_y),
                ]
                for i in range(4):
                    phys_list.append(corners[i])
                    nom_list.append(nom_corners[i])

    if getattr(tester, "right_boxes_global", None):
        for (row, col), corners in tester.right_boxes_global.items():
            if len(corners) != 4:
                continue
            target_x = right_gds_cx + centroid_algorithm.NOMINAL_COORDS[(row, col)][0]
            target_y = right_gds_cy - centroid_algorithm.NOMINAL_COORDS[(row, col)][1]
            best_match = None
            min_dist = float("inf")
            for m in right_squares:
                dist = math.hypot(m["center"][0] - target_x, m["center"][1] - target_y)
                if dist < min_dist:
                    min_dist = dist
                    best_match = m
            if best_match is not None and min_dist < 200.0:
                min_x, min_y, max_x, max_y = best_match["bbox"]
                nom_corners = [
                    (min_x, max_y),
                    (max_x, max_y),
                    (max_x, min_y),
                    (min_x, min_y),
                ]
                for i in range(4):
                    phys_list.append(corners[i])
                    nom_list.append(nom_corners[i])

    if len(phys_list) < 4:
        print(f"[{out_stem} Auto-Align] Warning: Not enough points resolved for SVD pre-alignment.")
        return None

    phys_arr = np.array(phys_list, dtype=np.float64)
    nom_arr = np.array(nom_list, dtype=np.float64)

    s_0 = gds_R / canvas_R
    x_cart = (phys_arr[:, 0] - canvas_xc) * s_0
    y_cart = (canvas_yc - phys_arr[:, 1]) * s_0
    base_cart_arr = np.column_stack((x_cart, y_cart))

    scale_mult, r_mat, t_vec, rmsd = coordinate_transformer.umeyama_rigid_registration(base_cart_arr, nom_arr)
    flat_angle = math.atan2(r_mat[1, 0], r_mat[0, 0])
    x_offset_um = float(t_vec[0, 0]) - gds_xc
    y_offset_um = float(t_vec[1, 0]) - gds_yc

    print(f"[{out_stem} Auto-Align] Global Rigid SVD registration complete:")
    print(f" RMSD: {rmsd:.3f} um over {len(phys_arr)} point pairs")
    print(f" Solved Flat Angle: {flat_angle * 180 / np.pi:.4f}°")
    print(f" Solved Scale Multiplier: {scale_mult:.6f}")
    print(f" Solved Translation: X={x_offset_um:.1f} um, Y={y_offset_um:.1f} um")
    return flat_angle, x_offset_um, y_offset_um, scale_mult


def _safe_int(value) -> int:
    return int(round(float(value)))


def _rotate_masks(local_masks: dict[str, np.ndarray], matrix, local_w: int, local_h: int) -> dict[str, np.ndarray]:
    return {
        name: cv2.warpAffine(mask, matrix, (local_w, local_h), flags=cv2.INTER_NEAREST)
        for name, mask in local_masks.items()
    }


def _save_crop_artifacts(
    *,
    out_dir: Path,
    preview_dir: Path,
    analysis_dir: Path,
    mask_dir: Path,
    meta_dir: Path,
    out_stem: str,
    row: int,
    col: int,
    cell_crop: np.ndarray,
    rotated_local_masks: dict[str, np.ndarray],
    crop_bounds: tuple[int, int, int, int],
    metadata: dict,
    preview_width: int = 2000,
) -> None:
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bounds
    cell_stem = f"{out_stem}_cell_{row}-{col}"

    # Keep legacy JPGs in out_dir so the existing GUI and any old scripts still work.
    cv2.imwrite(
        str(out_dir / f"{cell_stem}.jpg"),
        cell_crop,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )

    # Lossless image for the algorithmic detector.
    cv2.imwrite(str(analysis_dir / f"{cell_stem}.png"), cell_crop)

    # All masks are in final cell-crop coordinates. The detector currently uses
    # *_seam_mask.png, but the extra masks are useful for debugging.
    for mask_name, rotated_mask in rotated_local_masks.items():
        cell_mask = rotated_mask[crop_y1:crop_y2, crop_x1:crop_x2]
        cv2.imwrite(str(mask_dir / f"{cell_stem}_{mask_name}.png"), cell_mask)

    preview_w = int(preview_width or 0)
    if preview_w <= 0:
        preview_w = int(cell_crop.shape[1])
    preview_w = max(1, min(preview_w, int(cell_crop.shape[1])))  # never upscale previews
    preview_h = max(1, int(round(preview_w * cell_crop.shape[0] / max(cell_crop.shape[1], 1))))
    cell_preview = cv2.resize(cell_crop, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(
        str(preview_dir / f"{cell_stem}_preview.jpg"),
        cell_preview,
        [cv2.IMWRITE_JPEG_QUALITY, 94],
    )

    illumination_stitching.save_cell_metadata_json(meta_dir / f"{cell_stem}.json", metadata)



class ProgressBar:
    """Small dependency-free progress bar that works in PowerShell."""

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
        if current > 0:
            eta = elapsed * (self.total - current) / max(current, 1)
            eta_s = f" ETA {eta/60:4.1f}m"
        else:
            eta_s = " ETA  --.-m"
        msg = f"\r[{self.label}] |{bar}| {current}/{self.total} {frac*100:5.1f}% elapsed {elapsed/60:4.1f}m{eta_s}"
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


def _resize_crop_and_masks_if_needed(
    cell_crop: np.ndarray,
    rotated_local_masks: dict[str, np.ndarray],
    crop_bounds: tuple[int, int, int, int],
    max_width: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], tuple[int, int, int, int], float]:
    """Resize final crop/masks for fast detector work while preserving matching sizes."""
    max_width = int(max_width or 0)
    if max_width <= 0 or cell_crop.shape[1] <= max_width:
        return cell_crop, rotated_local_masks, crop_bounds, 1.0

    scale = max_width / float(cell_crop.shape[1])
    new_w = int(round(cell_crop.shape[1] * scale))
    new_h = int(round(cell_crop.shape[0] * scale))
    resized_crop = cv2.resize(cell_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bounds
    resized_masks = {}
    for name, rotated_mask in rotated_local_masks.items():
        cell_mask = rotated_mask[crop_y1:crop_y2, crop_x1:crop_x2]
        resized_masks[name] = cv2.resize(cell_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # The masks are now already final-crop masks, so use bounds covering the whole resized image.
    return resized_crop, resized_masks, (0, 0, new_w, new_h), scale


def _make_downscaled_seam_mask_region(
    *,
    local_origin_ds: tuple[int, int],
    local_size_ds: tuple[int, int],
    ds_factor: float,
    config_run: dict,
) -> np.ndarray:
    """Approximate seam/overlap mask for crops pulled from the downscaled stitch.

    The full native seam mask is expensive because it comes from local full-res
    tile blending. For the fast crop path, this marks the repeated overlap bands
    implied by tile geometry, downscaled into the same coordinates as ds_canvas.
    """
    local_w, local_h = map(int, local_size_ds)
    if local_w <= 0 or local_h <= 0:
        return np.zeros((0, 0), dtype=np.uint8)

    ds = float(ds_factor)
    tw = float(config_run["tile_width"])
    th = float(config_run["tile_height"])
    step_x = tw * (1.0 - float(config_run["overlap_x_percent"]) / 100.0)
    step_y = th * (1.0 - float(config_run["overlap_y_percent"]) / 100.0)
    if ds <= 0 or step_x <= 0 or step_y <= 0:
        return np.zeros((local_h, local_w), dtype=np.uint8)

    x0_ds, y0_ds = local_origin_ds
    xs_raw = (np.arange(local_w, dtype=np.float32) + float(x0_ds) + 0.5) / ds
    ys_raw = (np.arange(local_h, dtype=np.float32) + float(y0_ds) + 0.5) / ds

    mx = np.mod(xs_raw, step_x)
    my = np.mod(ys_raw, step_y)
    seam_x = mx >= step_x
    seam_y = my >= step_y

    # The modulo test above cannot hit step_x..tile_width when the period is step_x,
    # so explicitly mark tile boundaries and overlap bands using repeated starts.
    seam = np.zeros((local_h, local_w), dtype=np.uint8)
    x0_raw = float(x0_ds) / ds
    x1_raw = float(x0_ds + local_w) / ds
    y0_raw = float(y0_ds) / ds
    y1_raw = float(y0_ds + local_h) / ds
    overlap_x = max(1.0, tw - step_x)
    overlap_y = max(1.0, th - step_y)

    # Mark broad overlap regions between neighboring tiles.
    start_col = int(math.floor((x0_raw - tw) / step_x))
    end_col = int(math.ceil(x1_raw / step_x)) + 1
    for c in range(start_col, end_col + 1):
        raw_a = c * step_x + step_x
        raw_b = c * step_x + tw
        a = int(math.floor(raw_a * ds - x0_ds))
        b = int(math.ceil(raw_b * ds - x0_ds))
        if b > 0 and a < local_w:
            seam[:, max(0, a):min(local_w, b)] = 255

    start_row = int(math.floor((y0_raw - th) / step_y))
    end_row = int(math.ceil(y1_raw / step_y)) + 1
    for r in range(start_row, end_row + 1):
        raw_a = r * step_y + step_y
        raw_b = r * step_y + th
        a = int(math.floor(raw_a * ds - y0_ds))
        b = int(math.ceil(raw_b * ds - y0_ds))
        if b > 0 and a < local_h:
            seam[max(0, a):min(local_h, b), :] = 255

    # Slight dilation covers rounding and affine-rotation interpolation.
    k = max(3, int(round(max(overlap_x, overlap_y) * ds * 0.20)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.dilate(seam, kernel, iterations=1)


def _extract_fast_crops_from_downscaled_canvas(
    *,
    ds_canvas: np.ndarray,
    ds_factor: float,
    cells: list[dict],
    transformer,
    config_run: dict,
    args,
    out_stem: str,
    out_dir: Path,
    preview_dir: Path,
    analysis_dir: Path,
    mask_dir: Path,
    meta_dir: Path,
    flat_angle: float,
) -> tuple[int, list[dict]]:
    """Fast path: crop cells directly from the already-built downscaled stitch.

    This avoids re-reading and re-normalizing full-resolution source tiles for
    every cell. On the 40x58 wafer case, this turns the old multi-hour native
    crop stage into a seconds-to-minutes crop stage after the coarse stitch.
    """
    h_ds_canvas, w_ds_canvas = ds_canvas.shape[:2]
    ds = float(ds_factor)
    if ds <= 0:
        raise ValueError(f"Invalid downscale_factor: {ds_factor!r}")

    saved_count = 0
    records: list[dict] = []
    pad_raw = int(getattr(args, "fast_crop_pad", 200))
    pad_ds = max(4, int(round(pad_raw * ds)))
    shave_ds = max(0, int(round(int(args.shave) * ds)))
    max_width = int(getattr(args, "fast_crop_width", 1600))

    progress = ProgressBar(f"{out_stem} Fast Cell Crops", len(cells))
    for idx, cell in enumerate(cells, start=1):
        row = int(cell["row"])
        col = int(cell["col"])
        min_x, min_y, max_x, max_y = cell["bbox"]
        gds_corners = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
        pts_canvas = np.array([transformer.gds_to_canvas(gx, gy) for gx, gy in gds_corners], dtype=np.float64)
        # In normal operation gds_to_canvas returns full-resolution canvas pixels,
        # so multiplying by downscale_factor maps into ds_canvas. If a future
        # transformer already returns downscaled coordinates, auto-detect that.
        max_canvas_coord = float(np.nanmax(np.abs(pts_canvas))) if pts_canvas.size else 0.0
        max_ds_coord = float(max(w_ds_canvas, h_ds_canvas))
        scale_to_ds = ds if max_canvas_coord > max_ds_coord * 1.35 else 1.0
        pts_ds = pts_canvas * scale_to_ds

        cx_min, cy_min = np.min(pts_ds, axis=0)
        cx_max, cy_max = np.max(pts_ds, axis=0)
        x1 = max(0, int(np.floor(cx_min)) - pad_ds)
        y1 = max(0, int(np.floor(cy_min)) - pad_ds)
        x2 = min(w_ds_canvas, int(np.ceil(cx_max)) + pad_ds)
        y2 = min(h_ds_canvas, int(np.ceil(cy_max)) + pad_ds)
        if x2 <= x1 or y2 <= y1:
            progress.update(idx, extra=f"skip {row}-{col}")
            continue

        local_canvas = ds_canvas[y1:y2, x1:x2]
        local_h, local_w = local_canvas.shape[:2]
        crop_center = (float(np.mean(pts_ds[:, 0]) - x1), float(np.mean(pts_ds[:, 1]) - y1))
        m_rot = cv2.getRotationMatrix2D(crop_center, flat_angle * 180.0 / np.pi, 1.0)
        rotated_local_canvas = cv2.warpAffine(local_canvas, m_rot, (local_w, local_h), flags=cv2.INTER_LINEAR)

        seam_local = _make_downscaled_seam_mask_region(
            local_origin_ds=(x1, y1),
            local_size_ds=(local_w, local_h),
            ds_factor=ds,
            config_run=config_run,
        )
        rotated_local_masks = {
            "seam_mask": cv2.warpAffine(seam_local, m_rot, (local_w, local_h), flags=cv2.INTER_NEAREST)
        }

        pts_local_hom = np.column_stack([pts_ds[:, 0] - x1, pts_ds[:, 1] - y1, np.ones(4)])
        pts_rotated_local = (m_rot @ pts_local_hom.T).T
        rx_min, ry_min = np.min(pts_rotated_local, axis=0)
        rx_max, ry_max = np.max(pts_rotated_local, axis=0)

        crop_x1 = max(0, min(int(round(rx_min)) + shave_ds, local_w - 1))
        crop_x2 = max(0, min(int(round(rx_max)) - shave_ds, local_w - 1))
        crop_y1 = max(0, min(int(round(ry_min)) + shave_ds, local_h - 1))
        crop_y2 = max(0, min(int(round(ry_max)) - shave_ds, local_h - 1))
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            progress.update(idx, extra=f"skip {row}-{col}")
            continue

        cell_crop = rotated_local_canvas[crop_y1:crop_y2, crop_x1:crop_x2]
        cell_crop, rotated_local_masks, final_bounds, resize_scale = _resize_crop_and_masks_if_needed(
            cell_crop=cell_crop,
            rotated_local_masks=rotated_local_masks,
            crop_bounds=(crop_x1, crop_y1, crop_x2, crop_y2),
            max_width=max_width,
        )
        cell_stem = f"{out_stem}_cell_{row}-{col}"

        metadata = {
            "wafer_id": out_stem,
            "cell_row": row,
            "cell_col": col,
            "cell_stem": cell_stem,
            "crop_source": "downscaled_fast",
            "downscale_factor": float(ds),
            "canvas_to_downscaled_scale_used": float(scale_to_ds),
            "output_resize_scale": float(resize_scale),
            "gds_bbox_um": [float(min_x), float(min_y), float(max_x), float(max_y)],
            "gds_corners_um": [[float(a), float(b)] for a, b in gds_corners],
            "canvas_corners_px_fullres": pts_canvas.tolist(),
            "canvas_corners_px_downscaled": pts_ds.tolist(),
            "local_origin_px_downscaled": [int(x1), int(y1)],
            "local_size_px_downscaled": [int(local_w), int(local_h)],
            "rotation_matrix_2x3_downscaled": m_rot.tolist(),
            "rotated_local_corners_px_downscaled": pts_rotated_local.tolist(),
            "crop_bounds_local_px_downscaled_before_resize": [int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)],
            "crop_size_px": [int(cell_crop.shape[1]), int(cell_crop.shape[0])],
            "flat_angle_rad": float(flat_angle),
            "flat_angle_deg": float(flat_angle * 180.0 / np.pi),
            "shave_px_raw_requested": int(args.shave),
            "shave_px_downscaled_used": int(shave_ds),
            "analysis_png": str((analysis_dir / f"{cell_stem}.png").as_posix()),
            "legacy_jpg": str((out_dir / f"{cell_stem}.jpg").as_posix()),
            "seam_mask": str((mask_dir / f"{cell_stem}_seam_mask.png").as_posix()),
        }

        _save_crop_artifacts(
            out_dir=out_dir,
            preview_dir=preview_dir,
            analysis_dir=analysis_dir,
            mask_dir=mask_dir,
            meta_dir=meta_dir,
            out_stem=out_stem,
            row=row,
            col=col,
            cell_crop=cell_crop,
            rotated_local_masks=rotated_local_masks,
            crop_bounds=final_bounds,
            metadata=metadata,
            preview_width=int(getattr(args, "preview_width", 2000)),
        )
        records.append(metadata)
        saved_count += 1
        progress.update(idx, extra=f"saved {row}-{col}")

    progress.done(extra=f"saved {saved_count}")
    return saved_count, records

def process_wafer_cells(folder, json_file, config, args, wafer_id):
    config_run = copy.deepcopy(config)
    apply_illumination_cli_to_config(config_run, args)
    out_stem = wafer_id

    if json_file and Path(json_file).exists():
        try:
            defect_data = load_defect_json(json_file)
            summary_block = defect_data.get("summary", {})
            for param in ["overlap_x_percent", "overlap_y_percent", "downscale"]:
                if param in summary_block:
                    config_run[param] = float(summary_block[param])
            # Re-normalize after summary overrides. If the user supplied
            # --stitch-downscale, it must win over stale JSON/config values.
            apply_stitch_downscale_to_config(config_run, args)
        except Exception as e:
            print(f"[{out_stem}] Dynamic override parsing warning: {e}")

    try:
        detected_cols, detected_rows = detect_grid_size(folder)
        config_run["tile_cols"] = detected_cols
        config_run["tile_rows"] = detected_rows
    except Exception as e:
        print(f"[{out_stem}] Layout scanning error: {e}")
        return False

    try:
        gds_xc, gds_yc, gds_R = gds_parser.parse_gds_wafer_boundary(
            config_run["gds_path"],
            layer=config_run.get("gds_layer", 2),
            datatype=config_run.get("gds_datatype", 0),
        )
        gds_R = float(gds_R)
        gds_polygons = gds_parser.get_gds_overlay_polygons(config_run["gds_path"], config_run)
    except Exception as e:
        print(f"[{out_stem}] Critical error reading GDS data: {e}")
        return False

    try:
        print(f"[{out_stem}] Using stitch downscale divisor={float(config_run.get('downscale', 1.0/float(config_run.get('downscale_factor', 0.05)))):.3g}, factor={float(config_run.get('downscale_factor', 0.05)):.6g}")
        ds_canvas, tile_ext = wafer_metrology.generate_downscaled_stitch(folder, config_run)
    except Exception as e:
        print(f"[{out_stem}] Coarse-stitch canvas generation failed: {e}")
        return False

    ds_factor = config_run["downscale_factor"]
    x_offset_um = 0.0
    y_offset_um = 0.0
    scale_mult = 1.0

    try:
        canvas_xc, canvas_yc, canvas_R, flat_angle = wafer_metrology.detect_wafer_on_canvas(ds_canvas, ds_factor)
        markers = gds_parser.parse_alignment_markers(config_run["gds_path"])

        if args.manual:
            print(f"\n[{out_stem}] Launching automated Centroid Snapping UI on tiles...")
            try:
                tester = large_wafer_tester.LargeWaferTester(
                    image_path=folder,
                    display_height=800,
                    debug=args.centroid_debug,
                )
                tester.run()

                solved = _resolve_alignment_from_centroid_tester(
                    tester=tester,
                    markers=markers,
                    gds_R=gds_R,
                    canvas_xc=canvas_xc,
                    canvas_yc=canvas_yc,
                    canvas_R=canvas_R,
                    gds_xc=gds_xc,
                    gds_yc=gds_yc,
                    out_stem=out_stem,
                )
                if solved is not None:
                    flat_angle, x_offset_um, y_offset_um, scale_mult = solved
                    flat_angle, x_offset_um, y_offset_um, scale_mult = apply_auto_alignment_translation_correction(
                        flat_angle=flat_angle,
                        x_offset_um=x_offset_um,
                        y_offset_um=y_offset_um,
                        scale_mult=scale_mult,
                        config=config_run,
                        out_stem=out_stem,
                    )
            except Exception as e:
                print(f"[{out_stem} Auto-Align] Warning: SVD alignment calculation bypassed ({e}). Using metrology defaults.")

            flat_angle, x_offset_um, y_offset_um, scale_mult = wafer_align_gui.run_manual_alignment(
                ds_canvas,
                config_run,
                canvas_xc * ds_factor,
                canvas_yc * ds_factor,
                canvas_R * ds_factor,
                ds_factor,
                tile_ext,
                flat_angle,
                gds_polygons,
                gds_R,
                map_mode=True,
                gds_center=(gds_xc, gds_yc),
                shear=float(config_run.get("shear", 0.0)),
                markers=markers,
                initial_tx=-x_offset_um,
                initial_ty=-y_offset_um,
                initial_scale=scale_mult,
            )

        flat_angle = float(flat_angle)
        canvas_xc = float(canvas_xc)
        canvas_yc = float(canvas_yc)
        canvas_R = float(canvas_R)
    except Exception as e:
        print(f"[{out_stem}] Wafer metrology alignment failed: {e}")
        return False

    exclusions = load_exclusions("manual_exclusions.json")
    transformer = coordinate_transformer.WaferTransformer(
        canvas_center=(canvas_xc, canvas_yc),
        canvas_radius=canvas_R,
        canvas_flat_angle=flat_angle,
        gds_radius=gds_R,
        config=config_run,
        ext=tile_ext,
        exclusions=exclusions,
        shear=float(config_run.get("shear", 0.0)),
        x_offset=x_offset_um,
        y_offset=y_offset_um,
        map_mode=True,
        gds_center=(gds_xc, gds_yc),
    )
    transformer.S_x *= scale_mult
    transformer.S_y *= scale_mult
    transformer.S *= scale_mult

    cells = gds_parser.get_gds_cells_list(gds_polygons, gds_R)
    if not cells:
        print(f"[{out_stem}] Critical Error: No device cells identified inside design GDS layer.")
        return False

    run_create = args.create or (not args.create and not args.label)
    run_label = args.label or (not args.create and not args.label)
    out_dir = Path(args.out_dir)

    # -----------------------------------------------------------------------
    # NATIVE DEVICE CROP EXTRACTION
    # -----------------------------------------------------------------------
    if run_create:
        out_dir.mkdir(parents=True, exist_ok=True)
        preview_dir = out_dir / "previews"
        preview_dir.mkdir(exist_ok=True)
        analysis_dir = out_dir / "analysis_png"
        analysis_dir.mkdir(exist_ok=True)
        mask_dir = out_dir / "seam_masks"
        mask_dir.mkdir(exist_ok=True)
        meta_dir = out_dir / "metadata"
        meta_dir.mkdir(exist_ok=True)

        if str(getattr(args, "crop_source", "fast")).lower() == "fast":
            saved_count, cell_index_records = _extract_fast_crops_from_downscaled_canvas(
                ds_canvas=ds_canvas,
                ds_factor=float(config_run["downscale_factor"]),
                cells=cells,
                transformer=transformer,
                config_run=config_run,
                args=args,
                out_stem=out_stem,
                out_dir=out_dir,
                preview_dir=preview_dir,
                analysis_dir=analysis_dir,
                mask_dir=mask_dir,
                meta_dir=meta_dir,
                flat_angle=flat_angle,
            )
            illumination_stitching.save_cell_metadata_json(
                meta_dir / f"{out_stem}_cell_index.json",
                {"wafer_id": out_stem, "count": int(saved_count), "crop_source": "downscaled_fast", "cells": cell_index_records},
            )
            print(f"[{out_stem}] Fast slicing complete. Extracted {saved_count} cells.")
            print(f"[{out_stem}] Lossless detector inputs: {analysis_dir}")
            print(f"[{out_stem}] Seam masks: {mask_dir}")
            print(f"[{out_stem}] Crop metadata: {meta_dir}")
            return True
        else:

                tile_width = int(config_run["tile_width"])

                tile_height = int(config_run["tile_height"])

                step_x = tile_width * (1.0 - config_run["overlap_x_percent"] / 100.0)

                step_y = tile_height * (1.0 - config_run["overlap_y_percent"] / 100.0)



                target_luma = config_run.get("_illumination_target_luma")
        if target_luma is None:
            print(f"[{out_stem}] Estimating luminance for native crop extraction...")
            target_luma = illumination_stitching.estimate_global_luma_for_folder(
                folder,
                max_samples=int(args.illumination_brightness_samples),
            )
            config_run["_illumination_target_luma"] = float(target_luma)

        shared_flatfield_model = illumination_stitching.get_or_build_shared_flatfield_model(
            folder,
            config_run,
            verbose=True,
        )
        config_run["_shared_flatfield_model_obj"] = shared_flatfield_model

        tile_cache = illumination_stitching.NormalizedTileCache(
            max_items=int(args.tile_cache_size),
            target_luma=target_luma,
            illumination_enabled=not args.no_illumination_normalize,
            brightness_match_enabled=not args.no_brightness_match,
            illumination_strength=float(args.illumination_strength),
            blur_sigma_frac=float(args.illumination_blur_sigma_frac),
            brightness_match_strength=float(args.brightness_match_strength),
            shared_flatfield_model=shared_flatfield_model,
        )

        saved_count = 0
        pad = 200
        cell_index_records = []

        for idx, cell in enumerate(cells):
            row = int(cell["row"])
            col = int(cell["col"])
            min_x, min_y, max_x, max_y = cell["bbox"]

            gds_corners = [
                (min_x, min_y),
                (max_x, min_y),
                (max_x, max_y),
                (min_x, max_y),
            ]
            pts_canvas = np.array([transformer.gds_to_canvas(gx, gy) for gx, gy in gds_corners], dtype=np.float64)

            cx_min, cy_min = np.min(pts_canvas, axis=0)
            cx_max, cy_max = np.max(pts_canvas, axis=0)
            x1 = int(np.floor(cx_min)) - pad
            y1 = int(np.floor(cy_min)) - pad
            x2 = int(np.ceil(cx_max)) + pad
            y2 = int(np.ceil(cy_max)) + pad

            overlapping_tiles = []
            for c_col in range(1, int(config_run["tile_cols"]) + 1):
                for r_row in range(1, int(config_run["tile_rows"]) + 1):
                    tile_key = f"tile_x{c_col:03d}_y{r_row:03d}{tile_ext}"
                    if tile_key in transformer.exclusions:
                        continue
                    tile_x1 = int(round((c_col - 1) * step_x))
                    tile_y1 = int(round((r_row - 1) * step_y))
                    tile_x2 = tile_x1 + tile_width
                    tile_y2 = tile_y1 + tile_height
                    if max(x1, tile_x1) < min(x2, tile_x2) and max(y1, tile_y1) < min(y2, tile_y2):
                        overlapping_tiles.append((c_col, r_row, tile_x1, tile_y1, tile_x2, tile_y2))

            if not overlapping_tiles:
                continue

            local_w = x2 - x1
            local_h = y2 - y1
            if local_w <= 0 or local_h <= 0:
                continue

            local_canvas, local_masks = illumination_stitching.stitch_local_canvas_from_overlapping_tiles(
                folder=folder,
                tile_ext=tile_ext,
                overlapping_tiles=overlapping_tiles,
                local_origin=(x1, y1),
                local_size=(local_w, local_h),
                config=config_run,
                tile_cache=tile_cache,
                excluded_tile_names=transformer.exclusions,
                return_masks=True,
            )

            crop_center = (
                float(np.mean(pts_canvas, axis=0)[0] - x1),
                float(np.mean(pts_canvas, axis=0)[1] - y1),
            )
            m_rot = cv2.getRotationMatrix2D(crop_center, flat_angle * 180.0 / np.pi, 1.0)
            rotated_local_canvas = cv2.warpAffine(
                local_canvas,
                m_rot,
                (local_w, local_h),
                flags=cv2.INTER_LINEAR,
            )
            rotated_local_masks = _rotate_masks(local_masks, m_rot, local_w, local_h)

            pts_local_hom = np.column_stack([
                pts_canvas[:, 0] - x1,
                pts_canvas[:, 1] - y1,
                np.ones(4),
            ])
            pts_rotated_local = (m_rot @ pts_local_hom.T).T
            rx_min, ry_min = np.min(pts_rotated_local, axis=0)
            rx_max, ry_max = np.max(pts_rotated_local, axis=0)

            shave = int(args.shave)
            crop_x1 = max(0, min(int(round(rx_min)) + shave, local_w - 1))
            crop_x2 = max(0, min(int(round(rx_max)) - shave, local_w - 1))
            crop_y1 = max(0, min(int(round(ry_min)) + shave, local_h - 1))
            crop_y2 = max(0, min(int(round(ry_max)) - shave, local_h - 1))

            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                continue

            cell_crop = rotated_local_canvas[crop_y1:crop_y2, crop_x1:crop_x2]
            cell_stem = f"{out_stem}_cell_{row}-{col}"

            metadata = {
                "wafer_id": out_stem,
                "cell_row": row,
                "cell_col": col,
                "cell_stem": cell_stem,
                "gds_bbox_um": [float(min_x), float(min_y), float(max_x), float(max_y)],
                "gds_corners_um": [[float(a), float(b)] for a, b in gds_corners],
                "canvas_corners_px": pts_canvas.tolist(),
                "local_origin_px": [int(x1), int(y1)],
                "local_size_px": [int(local_w), int(local_h)],
                "rotation_matrix_2x3": m_rot.tolist(),
                "rotated_local_corners_px": pts_rotated_local.tolist(),
                "crop_bounds_local_px": [int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)],
                "crop_size_px": [int(cell_crop.shape[1]), int(cell_crop.shape[0])],
                "overlapping_tiles": [list(map(int, t)) for t in overlapping_tiles],
                "flat_angle_rad": float(flat_angle),
                "flat_angle_deg": float(flat_angle * 180.0 / np.pi),
                "shave_px": int(shave),
                "analysis_png": str((analysis_dir / f"{cell_stem}.png").as_posix()),
                "legacy_jpg": str((out_dir / f"{cell_stem}.jpg").as_posix()),
                "seam_mask": str((mask_dir / f"{cell_stem}_seam_mask.png").as_posix()),
            }

            _save_crop_artifacts(
                out_dir=out_dir,
                preview_dir=preview_dir,
                analysis_dir=analysis_dir,
                mask_dir=mask_dir,
                meta_dir=meta_dir,
                out_stem=out_stem,
                row=row,
                col=col,
                cell_crop=cell_crop,
                rotated_local_masks=rotated_local_masks,
                crop_bounds=(crop_x1, crop_y1, crop_x2, crop_y2),
                metadata=metadata,
            )

            cell_index_records.append(metadata)
            saved_count += 1
            sys.stdout.write(
                f"\r[{out_stem} Normalized Crop] Process: {idx + 1}/{len(cells)} | "
                f"Saved cell {row}-{col} -> analysis_png\\{cell_stem}.png\033[K"
            )
            sys.stdout.flush()

        illumination_stitching.save_cell_metadata_json(
            meta_dir / f"{out_stem}_cell_index.json",
            {
                "wafer_id": out_stem,
                "count": int(saved_count),
                "cells": cell_index_records,
            },
        )
        print(f"\n[{out_stem}] Slicing complete. Extracted {saved_count} cells.")
        print(f"[{out_stem}] Lossless detector inputs: {analysis_dir}")
        print(f"[{out_stem}] Seam masks: {mask_dir}")
        print(f"[{out_stem}] Crop metadata: {meta_dir}")
    else:
        print(f"[{out_stem}] Slicing skipped. Target cells loaded directly.")

    # -----------------------------------------------------------------------
    # MANUAL DEFECT ANNOTATION REVIEW PASS
    # -----------------------------------------------------------------------
    if args.device and run_label:
        mapper = defect_mapper_gui.DeviceDefectMapperTool(
            wafer_id=out_stem,
            cells=cells,
            out_dir=args.out_dir,
            transformer=transformer,
            gds_R=gds_R,
            config=config_run,
            shave=args.shave,
            pad=200,
        )
        mapper.run()

    return True


# ===========================================================================
# 3. BATCH PARSING / CLI
# ===========================================================================


def parse_batch_file(filepath: str) -> list[dict]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Batch definitions config file not found: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    wafers = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.endswith(":"):
            wafer_id = line[:-1].strip()
            if i + 3 < len(lines):
                after_folder = lines[i + 1].strip('"').strip("'")
                before_folder = lines[i + 2].strip('"').strip("'")
                defect_json = lines[i + 3].strip('"').strip("'")
                wafers.append(
                    {
                        "id": wafer_id,
                        "after_folder": after_folder,
                        "before_folder": before_folder,
                        "defect_json": defect_json,
                    }
                )
                i += 4
            else:
                i += 1
        else:
            i += 1
    return wafers


def _safe_remove_tree(path: Path) -> None:
    """Remove a generated output folder, with guardrails against bad paths."""
    resolved = path.expanduser().resolve()
    cwd = Path.cwd().resolve()
    forbidden = {cwd, cwd.parent, Path(resolved.anchor).resolve()}
    if resolved in forbidden or str(resolved) in ("", "."):
        raise RuntimeError(f"Refusing to delete unsafe output path: {path}")
    if not resolved.exists():
        return
    if not resolved.is_dir():
        raise RuntimeError(f"Refusing to delete non-directory output path: {path}")
    print(f"[Cleanup] Removing stale extraction folder: {resolved}")
    shutil.rmtree(resolved)


def _apply_simple_create_defaults(args) -> None:
    """Make `python wafer_alignment_and_extraction.py -c` match the normal H2P workflow."""
    # Simple create mode should launch the alignment GUI, matching the old long
    # command that explicitly passed --manual. Users can opt out with --no-manual.
    if args.create and not args.no_manual:
        args.manual = True

    if args.create and not args.no_clean:
        _safe_remove_tree(Path(args.out_dir))


def main():
    parser = argparse.ArgumentParser(description="Standalone Metrology Core and GDS Extraction Service")
    parser.add_argument("--batch", type=str, default="batch_wafers.txt", help="Path to wafer batch definitions txt config file. Default: batch_wafers.txt")
    parser.add_argument("--manual", action="store_true", help="Launch interactive manual adjustment dashboard beforehand")
    parser.add_argument("--no-manual", action="store_true", help="Do not launch manual alignment GUI in simple -c mode")
    parser.add_argument("--no-clean", action="store_true", help="Keep existing output folder instead of deleting it before -c")
    parser.add_argument("--shave", type=int, default=10, help="Crop outer buffer width inside rotated cell frame")
    parser.add_argument("--out-dir", type=str, default="extracted_cells", help="Target output subdirectory to store cropped files")
    parser.add_argument("-d", "--device", action="store_true", help="Enable defect inspection annotation dashboard review")
    parser.add_argument("-c", "--create", action="store_true", help="Stage 1: Generate native cropped die formats")
    parser.add_argument("-l", "--label", action="store_true", help="Stage 2: Label anomalies on extracted images")
    parser.add_argument("--centroid-debug", action="store_true", help="Enable debug mode in automated alignment")

    parser.add_argument("--no-illumination-normalize", action="store_true", help="Disable flat-field tile illumination normalization")
    parser.add_argument("--no-brightness-match", action="store_true", help="Disable global tile brightness matching")
    parser.add_argument("--illumination-strength", type=float, default=1.0, help="Flat-field correction strength, 0..1. Default: 1.0")
    parser.add_argument("--illumination-blur-sigma-frac", type=float, default=0.18, help="Per-tile smooth background blur sigma fraction. Default: 0.18")
    parser.add_argument("--brightness-match-strength", type=float, default=0.65, help="Global brightness matching strength. Default: 0.65")
    parser.add_argument("--illumination-brightness-samples", type=int, default=250, help="Tiles sampled for global luminance. Default: 250")
    parser.add_argument("--tile-cache-size", type=int, default=96, help="Full-resolution normalized tile cache size. Default: 96")
    parser.add_argument("--crop-source", choices=["fast", "native"], default="fast", help="fast crops from the already-built downscaled stitch; native re-stitches full-res tiles per cell. Default: fast")
    parser.add_argument("--stitch-downscale", type=float, default=12.0, help="Downscale divisor for the stitched wafer canvas. Smaller = sharper/slower. Default: 12")
    parser.add_argument("--fast-crop-width", type=int, default=0, help="Maximum width for fast analysis PNG crops. 0 keeps the stitched-canvas crop resolution. Default: 0")
    parser.add_argument("--fast-crop-pad", type=int, default=200, help="Raw-pixel padding around cell before fast rotate/crop. Default: 200")
    parser.add_argument("--preview-width", type=int, default=0, help="Maximum width of JPG review previews. 0 keeps crop width. Default: 0 for best review quality")

    parser.add_argument("--no-shared-flatfield", action="store_true", help="Disable shared tile-coordinate flat-field model")
    parser.add_argument("--no-overlap-leveling", action="store_true", help="Disable overlap-based LAB-L seam leveling")
    parser.add_argument("--shared-flatfield-samples", type=int, default=400, help="Tiles sampled to build shared flat-field")
    parser.add_argument("--shared-flatfield-model-side", type=int, default=384, help="Max side length of shared flat-field model")
    parser.add_argument("--overlap-leveling-strength", type=float, default=0.85, help="Strength of overlap LAB-L leveling")

    args = parser.parse_args()
    print(f"[Runtime] version={WAFER_EXTRACTION_VERSION}")
    _apply_simple_create_defaults(args)

    try:
        config = load_config("config.json")
    except Exception as e:
        print(f"Error reading parameters config file: {e}")
        sys.exit(1)

    try:
        wafers = parse_batch_file(args.batch)
        print(f"Discovered {len(wafers)} configuration sequences configured in batch run.")
    except Exception as e:
        print(f"Error parsing instruction batch file: {e}")
        sys.exit(1)

    for idx, wafer in enumerate(wafers):
        wafer_id = wafer["id"]
        after_folder = wafer["after_folder"]
        defect_json = wafer["defect_json"]

        print("\n" + "=" * 70)
        print(f" WAFER RUN [{idx + 1}/{len(wafers)}]: {wafer_id}")
        print("=" * 70)

        if after_folder.lower() == "none" or not after_folder:
            continue
        if not Path(after_folder).exists():
            print(f"[{wafer_id}] Skipping missing after-folder: {after_folder}")
            continue

        try:
            process_wafer_cells(
                folder=after_folder,
                json_file=defect_json,
                config=config,
                args=args,
                wafer_id=wafer_id,
            )
        except Exception as e:
            print(f"[{wafer_id}] Script crashed during operational pipeline execution: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print(" BATCH EXECUTION RUN COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
