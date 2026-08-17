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
import cell_boundary_alignment
import coordinate_transformer
import defect_mapper_gui
import design_geometry as gds_parser
import illumination_stitching
import large_wafer_tester
import wafer_align_gui
import wafer_metrology
import design_alignment
import design_geometry
from batch_wafers_parser import parse_batch_file
WAFER_EXTRACTION_VERSION = 'future-design-only-refactor-v2-2026-08-10+targeted-extraction-v1+folder-selector-v1'

def load_config(path='config.json'):
    config_path = Path(path)
    with config_path.open('r', encoding='utf-8') as handle:
        config = json.load(handle)
    config['gds_path'] = str(design_geometry.resolve_design_path())
    config['gds_layer'] = 2
    config['gds_datatype'] = 0
    return config

def load_defect_json(json_path):
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f'Defect JSON file not found at: {json_path}')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def detect_grid_size(tile_folder):
    folder = Path(tile_folder)
    if not folder.exists():
        raise FileNotFoundError(f'Tile folder does not exist at: {tile_folder}')
    pattern = re.compile('tile_x(\\d+)_y(\\d+)')
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
        raise ValueError(f'No valid tile files matched inside folder: {tile_folder}')
    return (max_col, max_row)

def load_exclusions(exclusions_path):
    path = Path(exclusions_path)
    if not path.exists():
        return set()
    try:
        with open(path, 'r', encoding='utf-8') as f:
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
    stitch_downscale = float(getattr(args, 'stitch_downscale', 0) or 0)
    if stitch_downscale > 0:
        config_run['downscale'] = stitch_downscale
        config_run['downscale_factor'] = 1.0 / stitch_downscale
        config_run['_stitch_downscale_divisor'] = stitch_downscale
        return
    if 'downscale' in config_run and float(config_run['downscale']) > 0:
        config_run['downscale'] = float(config_run['downscale'])
        config_run['downscale_factor'] = 1.0 / float(config_run['downscale'])
        config_run['_stitch_downscale_divisor'] = float(config_run['downscale'])
    elif 'downscale_factor' in config_run and float(config_run['downscale_factor']) > 0:
        config_run['downscale_factor'] = float(config_run['downscale_factor'])
        config_run['downscale'] = 1.0 / float(config_run['downscale_factor'])
        config_run['_stitch_downscale_divisor'] = float(config_run['downscale'])
    else:
        config_run['downscale'] = 20.0
        config_run['downscale_factor'] = 1.0 / 20.0
        config_run['_stitch_downscale_divisor'] = 20.0

def apply_illumination_cli_to_config(config_run: dict, args) -> None:
    """Push all stitching/illumination CLI settings into config_run.

    wafer_metrology.generate_downscaled_stitch() and the full-resolution local
    crop stitcher both read from this dict, so keeping the values here prevents
    the coarse preview and native cell crops from silently using different
    correction parameters.
    """
    config_run['_illumination_enabled'] = not args.no_illumination_normalize
    config_run['_brightness_match_enabled'] = not args.no_brightness_match
    config_run['_illumination_strength'] = float(args.illumination_strength)
    config_run['_illumination_blur_sigma_frac'] = float(args.illumination_blur_sigma_frac)
    config_run['_brightness_match_strength'] = float(args.brightness_match_strength)
    config_run['_illumination_brightness_samples'] = int(args.illumination_brightness_samples)
    apply_stitch_downscale_to_config(config_run, args)
    config_run['_shared_flatfield_enabled'] = not args.no_shared_flatfield
    config_run['_overlap_leveling_enabled'] = not args.no_overlap_leveling
    config_run['_shared_flatfield_samples'] = int(args.shared_flatfield_samples)
    config_run['_shared_flatfield_model_side'] = int(args.shared_flatfield_model_side)
    config_run['_overlap_leveling_strength'] = float(args.overlap_leveling_strength)
    config_run['_fast_jpeg_decode'] = not bool(getattr(args, 'bound_exact_jpeg_decode', False))

def _safe_int(value) -> int:
    return int(round(float(value)))

def _rotate_masks(local_masks: dict[str, np.ndarray], matrix, local_w: int, local_h: int) -> dict[str, np.ndarray]:
    return {name: cv2.warpAffine(mask, matrix, (local_w, local_h), flags=cv2.INTER_NEAREST) for name, mask in local_masks.items()}
DEFAULT_DEVICE_ZOOM_X = 1.0377104590638329
DEFAULT_DEVICE_ZOOM_Y = 1.0714417075665796
DEFAULT_DEVICE_BOTTOM_TRIM_PX = 14

def _normalize_device_crop_cli(args, parser=None) -> None:
    """Resolve the opt-in device crop switches into the legacy internal fields.

    Default: no measured zoom and zero bottom trim.
    --zoom: enable the measured X/Y crop using DEFAULT_DEVICE_ZOOM_X/Y.
    --trim: enable DEFAULT_DEVICE_BOTTOM_TRIM_PX without implicitly enabling zoom.
    The older low-level options remain available for explicit overrides.
    """
    zoom_requested = bool(getattr(args, 'zoom', False))
    force_no_zoom = bool(getattr(args, 'no_device_zoom', False))
    if zoom_requested and force_no_zoom:
        message = '--zoom and --no-device-zoom cannot be used together'
        if parser is not None:
            parser.error(message)
        raise ValueError(message)
    args.no_device_zoom = not zoom_requested
    requested_trim = getattr(args, 'device_bottom_trim_px', None)
    if requested_trim is None:
        requested_trim = DEFAULT_DEVICE_BOTTOM_TRIM_PX if bool(getattr(args, 'trim', False)) else 0
    requested_trim = int(round(float(requested_trim)))
    if requested_trim < 0:
        message = f'device bottom trim must be >= 0 pixels, got {requested_trim}'
        if parser is not None:
            parser.error(message)
        raise ValueError(message)
    args.device_bottom_trim_px = requested_trim

def _apply_device_zoom_to_bounds(crop_bounds: tuple[int, int, int, int], *, enabled: bool, zoom_x: float, zoom_y: float, bottom_trim_px: int=0, max_width: int, max_height: int) -> tuple[tuple[int, int, int, int], dict]:
    """Optionally apply measured zoom and/or an independent bottom-only trim.

    Both operations are crops, never resizes, so source pixels and seam masks
    remain registered. Zoom is centered. Bottom trim preserves the top edge.
    """
    x1, y1, x2, y2 = (int(v) for v in crop_bounds)
    max_width = max(1, int(max_width))
    max_height = max(1, int(max_height))
    x1 = max(0, min(x1, max_width - 1))
    y1 = max(0, min(y1, max_height - 1))
    x2 = max(x1 + 1, min(x2, max_width))
    y2 = max(y1 + 1, min(y2, max_height))
    input_w = x2 - x1
    input_h = y2 - y1
    zx = float(zoom_x)
    zy = float(zoom_y)
    requested_bottom_trim = int(round(float(bottom_trim_px)))
    if not math.isfinite(zx) or zx < 1.0:
        raise ValueError(f'device zoom X must be finite and >= 1.0, got {zoom_x!r}')
    if not math.isfinite(zy) or zy < 1.0:
        raise ValueError(f'device zoom Y must be finite and >= 1.0, got {zoom_y!r}')
    if requested_bottom_trim < 0:
        raise ValueError(f'device bottom trim must be >= 0 pixels, got {bottom_trim_px!r}')
    zoom_applied = bool(enabled) and (not (abs(zx - 1.0) < 1e-12 and abs(zy - 1.0) < 1e-12))
    if zoom_applied:
        target_w = max(16, min(input_w, int(round(input_w / zx))))
        target_h = max(16, min(input_h, int(round(input_h / zy))))
    else:
        target_w = input_w
        target_h = input_h
    trim_x = input_w - target_w
    trim_y = input_h - target_h
    trim_left = trim_x // 2
    trim_right = trim_x - trim_left
    trim_top = trim_y // 2
    trim_bottom = trim_y - trim_top
    applied_bottom_trim = min(requested_bottom_trim, max(0, target_h - 16))
    output_h = target_h - applied_bottom_trim
    trim_bottom += applied_bottom_trim
    result = (x1 + trim_left, y1 + trim_top, x2 - trim_right, y2 - trim_bottom)
    metadata = {'enabled': bool(zoom_applied or applied_bottom_trim), 'applied': bool(trim_x or trim_y or applied_bottom_trim), 'zoom_enabled': bool(enabled), 'zoom_applied': zoom_applied, 'trim_enabled': bool(requested_bottom_trim), 'zoom_x': zx, 'zoom_y': zy, 'bottom_trim_px': applied_bottom_trim, 'requested_bottom_trim_px': requested_bottom_trim, 'input_size_px': [input_w, input_h], 'output_size_px': [target_w, output_h], 'trim_px': {'left': trim_left, 'right': trim_right, 'top': trim_top, 'bottom': trim_bottom}}
    return (result, metadata)

def _save_crop_artifacts(*, out_dir: Path, preview_dir: Path, analysis_dir: Path, mask_dir: Path, meta_dir: Path, out_stem: str, row: int, col: int, cell_crop: np.ndarray, rotated_local_masks: dict[str, np.ndarray], crop_bounds: tuple[int, int, int, int], metadata: dict, preview_width: int=2000) -> None:
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bounds
    cell_stem = f'{out_stem}_cell_{row}-{col}'
    cv2.imwrite(str(out_dir / f'{cell_stem}.jpg'), cell_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(analysis_dir / f'{cell_stem}.png'), cell_crop)
    for mask_name, rotated_mask in rotated_local_masks.items():
        cell_mask = rotated_mask[crop_y1:crop_y2, crop_x1:crop_x2]
        cv2.imwrite(str(mask_dir / f'{cell_stem}_{mask_name}.png'), cell_mask)
    preview_w = int(preview_width or 0)
    if preview_w <= 0:
        preview_w = int(cell_crop.shape[1])
    preview_w = max(1, min(preview_w, int(cell_crop.shape[1])))
    preview_h = max(1, int(round(preview_w * cell_crop.shape[0] / max(cell_crop.shape[1], 1))))
    cell_preview = cv2.resize(cell_crop, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(preview_dir / f'{cell_stem}_preview.jpg'), cell_preview, [cv2.IMWRITE_JPEG_QUALITY, 94])
    illumination_stitching.save_cell_metadata_json(meta_dir / f'{cell_stem}.json', metadata)

class ProgressBar:
    """Small dependency-free progress bar that works in PowerShell."""

    def __init__(self, label: str, total: int, width: int=28):
        self.label = str(label)
        self.total = max(1, int(total))
        self.width = max(8, int(width))
        self.start = time.time()
        self.last_len = 0

    def update(self, current: int, extra: str='') -> None:
        current = max(0, min(int(current), self.total))
        frac = current / self.total
        filled = int(round(frac * self.width))
        bar = '#' * filled + '-' * (self.width - filled)
        elapsed = time.time() - self.start
        if current > 0:
            eta = elapsed * (self.total - current) / max(current, 1)
            eta_s = f' ETA {eta / 60:4.1f}m'
        else:
            eta_s = ' ETA  --.-m'
        msg = f'\r[{self.label}] |{bar}| {current}/{self.total} {frac * 100:5.1f}% elapsed {elapsed / 60:4.1f}m{eta_s}'
        if extra:
            msg += f' | {extra}'
        pad = max(0, self.last_len - len(msg))
        sys.stdout.write(msg + ' ' * pad)
        sys.stdout.flush()
        self.last_len = len(msg)

    def done(self, extra: str='') -> None:
        self.update(self.total, extra=extra)
        sys.stdout.write('\n')
        sys.stdout.flush()

def _resize_crop_and_masks_if_needed(cell_crop: np.ndarray, rotated_local_masks: dict[str, np.ndarray], crop_bounds: tuple[int, int, int, int], max_width: int) -> tuple[np.ndarray, dict[str, np.ndarray], tuple[int, int, int, int], float]:
    """Resize final crop/masks for fast detector work while preserving matching sizes."""
    max_width = int(max_width or 0)
    if max_width <= 0 or cell_crop.shape[1] <= max_width:
        return (cell_crop, rotated_local_masks, crop_bounds, 1.0)
    scale = max_width / float(cell_crop.shape[1])
    new_w = int(round(cell_crop.shape[1] * scale))
    new_h = int(round(cell_crop.shape[0] * scale))
    resized_crop = cv2.resize(cell_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bounds
    resized_masks = {}
    for name, rotated_mask in rotated_local_masks.items():
        cell_mask = rotated_mask[crop_y1:crop_y2, crop_x1:crop_x2]
        resized_masks[name] = cv2.resize(cell_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return (resized_crop, resized_masks, (0, 0, new_w, new_h), scale)

def _make_downscaled_seam_mask_region(*, local_origin_ds: tuple[int, int], local_size_ds: tuple[int, int], ds_factor: float, config_run: dict) -> np.ndarray:
    """Approximate seam/overlap mask for crops pulled from the downscaled stitch.

    The full native seam mask is expensive because it comes from local full-res
    tile blending. For the fast crop path, this marks the repeated overlap bands
    implied by tile geometry, downscaled into the same coordinates as ds_canvas.
    """
    local_w, local_h = map(int, local_size_ds)
    if local_w <= 0 or local_h <= 0:
        return np.zeros((0, 0), dtype=np.uint8)
    ds = float(ds_factor)
    tw = float(config_run['tile_width'])
    th = float(config_run['tile_height'])
    step_x = tw * (1.0 - float(config_run['overlap_x_percent']) / 100.0)
    step_y = th * (1.0 - float(config_run['overlap_y_percent']) / 100.0)
    if ds <= 0 or step_x <= 0 or step_y <= 0:
        return np.zeros((local_h, local_w), dtype=np.uint8)
    x0_ds, y0_ds = local_origin_ds
    xs_raw = (np.arange(local_w, dtype=np.float32) + float(x0_ds) + 0.5) / ds
    ys_raw = (np.arange(local_h, dtype=np.float32) + float(y0_ds) + 0.5) / ds
    mx = np.mod(xs_raw, step_x)
    my = np.mod(ys_raw, step_y)
    seam_x = mx >= step_x
    seam_y = my >= step_y
    seam = np.zeros((local_h, local_w), dtype=np.uint8)
    x0_raw = float(x0_ds) / ds
    x1_raw = float(x0_ds + local_w) / ds
    y0_raw = float(y0_ds) / ds
    y1_raw = float(y0_ds + local_h) / ds
    overlap_x = max(1.0, tw - step_x)
    overlap_y = max(1.0, th - step_y)
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
    k = max(3, int(round(max(overlap_x, overlap_y) * ds * 0.2)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.dilate(seam, kernel, iterations=1)

def _extract_fast_crops_from_downscaled_canvas(*, ds_canvas: np.ndarray, ds_factor: float, cells: list[dict], transformer, config_run: dict, args, out_stem: str, out_dir: Path, preview_dir: Path, analysis_dir: Path, mask_dir: Path, meta_dir: Path, flat_angle: float) -> tuple[int, list[dict]]:
    """Fast path: crop cells directly from the already-built downscaled stitch.

    This avoids re-reading and re-normalizing full-resolution source tiles for
    every cell. On the 40x58 wafer case, this turns the old multi-hour native
    crop stage into a seconds-to-minutes crop stage after the coarse stitch.
    """
    h_ds_canvas, w_ds_canvas = ds_canvas.shape[:2]
    ds = float(ds_factor)
    if ds <= 0:
        raise ValueError(f'Invalid downscale_factor: {ds_factor!r}')
    saved_count = 0
    records: list[dict] = []
    pad_raw = int(getattr(args, 'fast_crop_pad', 200))
    pad_ds = max(4, int(round(pad_raw * ds)))
    shave_ds = max(0, int(round(int(args.shave) * ds)))
    max_width = int(getattr(args, 'fast_crop_width', 1600))
    progress = ProgressBar(f'{out_stem} Fast Cell Crops', len(cells))
    for idx, cell in enumerate(cells, start=1):
        row = int(cell['row'])
        col = int(cell['col'])
        min_x, min_y, max_x, max_y = cell['bbox']
        gds_corners = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
        pts_canvas = np.array([transformer.gds_to_canvas(gx, gy) for gx, gy in gds_corners], dtype=np.float64)
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
            progress.update(idx, extra=f'skip {row}-{col}')
            continue
        local_canvas = ds_canvas[y1:y2, x1:x2]
        local_h, local_w = local_canvas.shape[:2]
        crop_center = (float(np.mean(pts_ds[:, 0]) - x1), float(np.mean(pts_ds[:, 1]) - y1))
        m_rot = cv2.getRotationMatrix2D(crop_center, flat_angle * 180.0 / np.pi, 1.0)
        rotated_local_canvas = cv2.warpAffine(local_canvas, m_rot, (local_w, local_h), flags=cv2.INTER_LINEAR)
        seam_local = _make_downscaled_seam_mask_region(local_origin_ds=(x1, y1), local_size_ds=(local_w, local_h), ds_factor=ds, config_run=config_run)
        rotated_local_masks = {'seam_mask': cv2.warpAffine(seam_local, m_rot, (local_w, local_h), flags=cv2.INTER_NEAREST)}
        pts_local_hom = np.column_stack([pts_ds[:, 0] - x1, pts_ds[:, 1] - y1, np.ones(4)])
        pts_rotated_local = (m_rot @ pts_local_hom.T).T
        rx_min, ry_min = np.min(pts_rotated_local, axis=0)
        rx_max, ry_max = np.max(pts_rotated_local, axis=0)
        crop_x1 = max(0, min(int(round(rx_min)) + shave_ds, local_w - 1))
        crop_x2 = max(0, min(int(round(rx_max)) - shave_ds, local_w - 1))
        crop_y1 = max(0, min(int(round(ry_min)) + shave_ds, local_h - 1))
        crop_y2 = max(0, min(int(round(ry_max)) - shave_ds, local_h - 1))
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            progress.update(idx, extra=f'skip {row}-{col}')
            continue
        (crop_x1, crop_y1, crop_x2, crop_y2), device_zoom_metadata = _apply_device_zoom_to_bounds((crop_x1, crop_y1, crop_x2, crop_y2), enabled=not bool(getattr(args, 'no_device_zoom', False)), zoom_x=float(getattr(args, 'device_zoom_x', DEFAULT_DEVICE_ZOOM_X)), zoom_y=float(getattr(args, 'device_zoom_y', DEFAULT_DEVICE_ZOOM_Y)), bottom_trim_px=int(getattr(args, 'device_bottom_trim_px', DEFAULT_DEVICE_BOTTOM_TRIM_PX)), max_width=local_w, max_height=local_h)
        cell_crop = rotated_local_canvas[crop_y1:crop_y2, crop_x1:crop_x2]
        cell_crop, rotated_local_masks, final_bounds, resize_scale = _resize_crop_and_masks_if_needed(cell_crop=cell_crop, rotated_local_masks=rotated_local_masks, crop_bounds=(crop_x1, crop_y1, crop_x2, crop_y2), max_width=max_width)
        cell_stem = f'{out_stem}_cell_{row}-{col}'
        metadata = {'wafer_id': out_stem, 'cell_row': row, 'cell_col': col, 'cell_stem': cell_stem, 'crop_source': 'downscaled_fast', 'downscale_factor': float(ds), 'canvas_to_downscaled_scale_used': float(scale_to_ds), 'output_resize_scale': float(resize_scale), 'gds_bbox_um': [float(min_x), float(min_y), float(max_x), float(max_y)], 'gds_corners_um': [[float(a), float(b)] for a, b in gds_corners], 'canvas_corners_px_fullres': pts_canvas.tolist(), 'canvas_corners_px_downscaled': pts_ds.tolist(), 'local_origin_px_downscaled': [int(x1), int(y1)], 'local_size_px_downscaled': [int(local_w), int(local_h)], 'rotation_matrix_2x3_downscaled': m_rot.tolist(), 'rotated_local_corners_px_downscaled': pts_rotated_local.tolist(), 'crop_bounds_local_px_downscaled_before_resize': [int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)], 'crop_size_px': [int(cell_crop.shape[1]), int(cell_crop.shape[0])], 'flat_angle_rad': float(flat_angle), 'flat_angle_deg': float(flat_angle * 180.0 / np.pi), 'shave_px_raw_requested': int(args.shave), 'shave_px_downscaled_used': int(shave_ds), 'analysis_png': str((analysis_dir / f'{cell_stem}.png').as_posix()), 'legacy_jpg': str((out_dir / f'{cell_stem}.jpg').as_posix()), 'seam_mask': str((mask_dir / f'{cell_stem}_seam_mask.png').as_posix())}
        metadata['device_zoom'] = dict(device_zoom_metadata)
        _save_crop_artifacts(out_dir=out_dir, preview_dir=preview_dir, analysis_dir=analysis_dir, mask_dir=mask_dir, meta_dir=meta_dir, out_stem=out_stem, row=row, col=col, cell_crop=cell_crop, rotated_local_masks=rotated_local_masks, crop_bounds=final_bounds, metadata=metadata, preview_width=int(getattr(args, 'preview_width', 2000)))
        records.append(metadata)
        saved_count += 1
        progress.update(idx, extra=f'saved {row}-{col}')
    progress.done(extra=f'saved {saved_count}')
    return (saved_count, records)

def process_wafer_cells(folder, json_file, config, args, wafer_id):
    if getattr(args, 'bound', False):
        _bound_threads = max(1, int(getattr(args, 'bound_opencv_threads', 2)))
        try:
            cv2.setNumThreads(_bound_threads)
        except Exception:
            pass
        print(f'[Runtime] --bound selected: using hybrid boundary detection with bounded scaled-native local crops (OpenCV threads={_bound_threads}).')
    config_run = copy.deepcopy(config)
    apply_illumination_cli_to_config(config_run, args)
    out_stem = wafer_id
    if json_file and Path(json_file).exists():
        try:
            defect_data = load_defect_json(json_file)
            summary_block = defect_data.get('summary', {})
            for param in ['overlap_x_percent', 'overlap_y_percent', 'downscale']:
                if param in summary_block:
                    config_run[param] = float(summary_block[param])
            apply_stitch_downscale_to_config(config_run, args)
        except Exception as e:
            print(f'[{out_stem}] Dynamic override parsing warning: {e}')
    try:
        detected_cols, detected_rows = detect_grid_size(folder)
        config_run['tile_cols'] = detected_cols
        config_run['tile_rows'] = detected_rows
    except Exception as e:
        print(f'[{out_stem}] Layout scanning error: {e}')
        return False
    try:
        gds_xc, gds_yc, gds_R = gds_parser.parse_gds_wafer_boundary(config_run['gds_path'], layer=config_run.get('gds_layer', 2), datatype=config_run.get('gds_datatype', 0))
        gds_R = float(gds_R)
        gds_polygons = gds_parser.get_gds_overlay_polygons(config_run['gds_path'], config_run)
    except Exception as e:
        print(f'[{out_stem}] Critical error reading GDS data: {e}')
        return False
    try:
        print(f"[{out_stem}] Using stitch downscale divisor={float(config_run.get('downscale', 1.0 / float(config_run.get('downscale_factor', 0.05)))):.3g}, factor={float(config_run.get('downscale_factor', 0.05)):.6g}")
        ds_canvas, tile_ext = wafer_metrology.generate_downscaled_stitch(folder, config_run)
    except Exception as e:
        print(f'[{out_stem}] Coarse-stitch canvas generation failed: {e}')
        return False
    ds_factor = config_run['downscale_factor']
    x_offset_um = 0.0
    y_offset_um = 0.0
    scale_mult = 1.0
    try:
        canvas_xc, canvas_yc, canvas_R, flat_angle = wafer_metrology.detect_wafer_on_canvas(ds_canvas, ds_factor)
        geometry = design_geometry.load_design_geometry(config_run['gds_path'])
        alignment_runtime = {'ds_canvas': ds_canvas, 'tile_folder': str(folder), 'stitch_config_run': copy.deepcopy(config_run), 'ds_factor': float(ds_factor), 'canvas_center_full': (float(canvas_xc), float(canvas_yc)), 'canvas_radius_full': float(canvas_R), 'canvas_center_ds': (float(canvas_xc) * float(ds_factor), float(canvas_yc) * float(ds_factor)), 'canvas_radius_ds': float(canvas_R) * float(ds_factor)}
        markers = gds_parser.parse_alignment_markers(config_run['gds_path'])
        if args.manual:
            print(f'\n[{out_stem}] Launching automated Centroid Snapping UI on tiles...')
            try:
                tester = design_alignment.create_marker_reviewer(large_wafer_tester.LargeWaferTester, geometry, alignment_runtime)(image_path=folder, display_height=800, debug=args.centroid_debug)
                tester.run()
                solved = design_alignment.resolve_alignment(tester=tester, markers=markers, gds_R=gds_R, canvas_xc=canvas_xc, canvas_yc=canvas_yc, canvas_R=canvas_R, gds_xc=gds_xc, gds_yc=gds_yc, out_stem=out_stem, geometry=geometry, runtime_state=alignment_runtime)
                if solved is not None:
                    flat_angle, x_offset_um, y_offset_um, scale_mult = solved
            except Exception as e:
                print(f'[{out_stem} Auto-Align] Warning: SVD alignment calculation bypassed ({e}). Using metrology defaults.')
            flat_angle, x_offset_um, y_offset_um, scale_mult = wafer_align_gui.run_manual_alignment(ds_canvas, config_run, canvas_xc * ds_factor, canvas_yc * ds_factor, canvas_R * ds_factor, ds_factor, tile_ext, flat_angle, gds_polygons, gds_R, map_mode=True, gds_center=(gds_xc, gds_yc), shear=float(config_run.get('shear', 0.0)), markers=markers, initial_tx=-x_offset_um, initial_ty=-y_offset_um, initial_scale=scale_mult)
        flat_angle = float(flat_angle)
        canvas_xc = float(canvas_xc)
        canvas_yc = float(canvas_yc)
        canvas_R = float(canvas_R)
    except Exception as e:
        print(f'[{out_stem}] Wafer metrology alignment failed: {e}')
        return False
    exclusions = load_exclusions('manual_exclusions.json')
    transformer = coordinate_transformer.WaferTransformer(canvas_center=(canvas_xc, canvas_yc), canvas_radius=canvas_R, canvas_flat_angle=flat_angle, gds_radius=gds_R, config=config_run, ext=tile_ext, exclusions=exclusions, shear=float(config_run.get('shear', 0.0)), x_offset=x_offset_um, y_offset=y_offset_um, map_mode=True, gds_center=(gds_xc, gds_yc))
    transformer.S_x *= scale_mult
    transformer.S_y *= scale_mult
    transformer.S *= scale_mult
    cells = gds_parser.get_gds_cells_list(gds_polygons, gds_R)
    if not cells:
        print(f'[{out_stem}] Critical Error: No device cells identified inside design GDS layer.')
        return False
    cells = _filter_selected_cells(cells, getattr(args, 'cell', None), out_stem)
    run_create = args.create or (not args.create and (not args.label))
    run_label = args.label or (not args.create and (not args.label))
    out_dir = Path(args.out_dir) / out_stem
    if run_create:
        out_dir.mkdir(parents=True, exist_ok=True)
        preview_dir = out_dir / 'previews'
        preview_dir.mkdir(exist_ok=True)
        analysis_dir = out_dir / 'analysis_png'
        analysis_dir.mkdir(exist_ok=True)
        mask_dir = out_dir / 'seam_masks'
        mask_dir.mkdir(exist_ok=True)
        meta_dir = out_dir / 'metadata'
        meta_dir.mkdir(exist_ok=True)
        if getattr(args, 'bound', False):
            saved_count, cell_index_records = cell_boundary_alignment.extract_bound_crops(folder=folder, tile_ext=tile_ext, cells=cells, transformer=transformer, config_run=config_run, args=args, out_stem=out_stem, out_dir=out_dir, preview_dir=preview_dir, analysis_dir=analysis_dir, mask_dir=mask_dir, meta_dir=meta_dir, save_crop_artifacts=_save_crop_artifacts)
            _save_cell_index_payload(meta_dir / f'{out_stem}_cell_index.json', {'wafer_id': out_stem, 'count': int(saved_count), 'crop_source': 'scaled_native_physical_boundary', 'cells': cell_index_records}, merge=bool(getattr(args, 'cell', None)))
            print(f'[{out_stem}] Boundary slicing complete. Extracted {saved_count} cells.')
            print(f'[{out_stem}] Lossless detector inputs: {analysis_dir}')
            print(f"[{out_stem}] Boundary diagnostics: {out_dir / 'boundary_debug'}")
            return True
        if str(getattr(args, 'crop_source', 'fast')).lower() == 'fast':
            saved_count, cell_index_records = _extract_fast_crops_from_downscaled_canvas(ds_canvas=ds_canvas, ds_factor=float(config_run['downscale_factor']), cells=cells, transformer=transformer, config_run=config_run, args=args, out_stem=out_stem, out_dir=out_dir, preview_dir=preview_dir, analysis_dir=analysis_dir, mask_dir=mask_dir, meta_dir=meta_dir, flat_angle=flat_angle)
            _save_cell_index_payload(meta_dir / f'{out_stem}_cell_index.json', {'wafer_id': out_stem, 'count': int(saved_count), 'crop_source': 'downscaled_fast', 'cells': cell_index_records}, merge=bool(getattr(args, 'cell', None)))
            print(f'[{out_stem}] Fast slicing complete. Extracted {saved_count} cells.')
            print(f'[{out_stem}] Lossless detector inputs: {analysis_dir}')
            print(f'[{out_stem}] Seam masks: {mask_dir}')
            print(f'[{out_stem}] Crop metadata: {meta_dir}')
            return True
        else:
            tile_width = int(config_run['tile_width'])
            tile_height = int(config_run['tile_height'])
            step_x = tile_width * (1.0 - config_run['overlap_x_percent'] / 100.0)
            step_y = tile_height * (1.0 - config_run['overlap_y_percent'] / 100.0)
            target_luma = config_run.get('_illumination_target_luma')
        if target_luma is None:
            print(f'[{out_stem}] Estimating luminance for native crop extraction...')
            target_luma = illumination_stitching.estimate_global_luma_for_folder(folder, max_samples=int(args.illumination_brightness_samples))
            config_run['_illumination_target_luma'] = float(target_luma)
        shared_flatfield_model = illumination_stitching.get_or_build_shared_flatfield_model(folder, config_run, verbose=True)
        config_run['_shared_flatfield_model_obj'] = shared_flatfield_model
        tile_cache = illumination_stitching.NormalizedTileCache(max_items=int(args.tile_cache_size), target_luma=target_luma, illumination_enabled=not args.no_illumination_normalize, brightness_match_enabled=not args.no_brightness_match, illumination_strength=float(args.illumination_strength), blur_sigma_frac=float(args.illumination_blur_sigma_frac), brightness_match_strength=float(args.brightness_match_strength), shared_flatfield_model=shared_flatfield_model)
        saved_count = 0
        pad = 200
        cell_index_records = []
        for idx, cell in enumerate(cells):
            row = int(cell['row'])
            col = int(cell['col'])
            min_x, min_y, max_x, max_y = cell['bbox']
            gds_corners = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
            pts_canvas = np.array([transformer.gds_to_canvas(gx, gy) for gx, gy in gds_corners], dtype=np.float64)
            if getattr(args, 'bound', False):
                _bound_top = float(np.linalg.norm(pts_canvas[1] - pts_canvas[0]))
                _bound_bottom = float(np.linalg.norm(pts_canvas[2] - pts_canvas[3]))
                _bound_left = float(np.linalg.norm(pts_canvas[3] - pts_canvas[0]))
                _bound_right = float(np.linalg.norm(pts_canvas[2] - pts_canvas[1]))
                _bound_long_side = max(_bound_top, _bound_bottom, _bound_left, _bound_right, 1.0)
                pad = max(int(getattr(args, 'bound_pad', 400)), int(round(_bound_long_side * float(getattr(args, 'bound_expand', 1.25)))))
            cx_min, cy_min = np.min(pts_canvas, axis=0)
            cx_max, cy_max = np.max(pts_canvas, axis=0)
            x1 = int(np.floor(cx_min)) - pad
            y1 = int(np.floor(cy_min)) - pad
            x2 = int(np.ceil(cx_max)) + pad
            y2 = int(np.ceil(cy_max)) + pad
            overlapping_tiles = []
            for c_col in range(1, int(config_run['tile_cols']) + 1):
                for r_row in range(1, int(config_run['tile_rows']) + 1):
                    tile_key = f'tile_x{c_col:03d}_y{r_row:03d}{tile_ext}'
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
            local_canvas, local_masks = illumination_stitching.stitch_local_canvas_from_overlapping_tiles(folder=folder, tile_ext=tile_ext, overlapping_tiles=overlapping_tiles, local_origin=(x1, y1), local_size=(local_w, local_h), config=config_run, tile_cache=tile_cache, excluded_tile_names=transformer.exclusions, return_masks=True)
            crop_center = (float(np.mean(pts_canvas, axis=0)[0] - x1), float(np.mean(pts_canvas, axis=0)[1] - y1))
            m_rot = cv2.getRotationMatrix2D(crop_center, flat_angle * 180.0 / np.pi, 1.0)
            rotated_local_canvas = cv2.warpAffine(local_canvas, m_rot, (local_w, local_h), flags=cv2.INTER_LINEAR)
            rotated_local_masks = _rotate_masks(local_masks, m_rot, local_w, local_h)
            pts_local_hom = np.column_stack([pts_canvas[:, 0] - x1, pts_canvas[:, 1] - y1, np.ones(4)])
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
            (crop_x1, crop_y1, crop_x2, crop_y2), device_zoom_metadata = _apply_device_zoom_to_bounds((crop_x1, crop_y1, crop_x2, crop_y2), enabled=not bool(getattr(args, 'no_device_zoom', False)), zoom_x=float(getattr(args, 'device_zoom_x', DEFAULT_DEVICE_ZOOM_X)), zoom_y=float(getattr(args, 'device_zoom_y', DEFAULT_DEVICE_ZOOM_Y)), bottom_trim_px=int(getattr(args, 'device_bottom_trim_px', DEFAULT_DEVICE_BOTTOM_TRIM_PX)), max_width=local_w, max_height=local_h)
            cell_crop = rotated_local_canvas[crop_y1:crop_y2, crop_x1:crop_x2]
            boundary_metadata = {'enabled': bool(getattr(args, 'bound', False)), 'applied': False}
            if getattr(args, 'bound', False):
                approximate_local_quad = pts_canvas.astype(np.float32) - np.array([x1, y1], dtype=np.float32)
                debug_path = None
                if getattr(args, 'bound_debug', False):
                    debug_path = out_dir / 'boundary_debug' / f'{out_stem}_cell_{row}-{col}_boundary.jpg'
                boundary_result = cell_boundary_alignment.refine_and_rectify_cell(local_canvas, approximate_local_quad, masks=local_masks, search_inward_fraction=float(getattr(args, 'bound_search_inward_fraction', 0.08)), min_confidence=float(getattr(args, 'bound_min_confidence', 0.22)), output_scale=float(getattr(args, 'bound_output_scale', 1.0)), shave_px=int(args.shave), debug_path=debug_path)
                boundary_metadata = dict(boundary_result.metadata or {})
                boundary_metadata['enabled'] = True
                boundary_metadata['applied'] = bool(boundary_result.success)
                boundary_metadata['fallback_reason'] = '' if boundary_result.success else boundary_result.reason
                if boundary_result.success:
                    cell_crop = boundary_result.image
                    rotated_local_masks = boundary_result.masks
                    crop_x1 = 0
                    crop_y1 = 0
                    crop_x2 = int(cell_crop.shape[1])
                    crop_y2 = int(cell_crop.shape[0])
                    device_zoom_metadata = {'enabled': not bool(getattr(args, 'no_device_zoom', False)), 'applied': False, 'reason': 'physical_boundary_alignment_applied', 'zoom_x': float(getattr(args, 'device_zoom_x', DEFAULT_DEVICE_ZOOM_X)), 'zoom_y': float(getattr(args, 'device_zoom_y', DEFAULT_DEVICE_ZOOM_Y)), 'input_size_px': [int(cell_crop.shape[1]), int(cell_crop.shape[0])], 'output_size_px': [int(cell_crop.shape[1]), int(cell_crop.shape[0])], 'trim_px': {'left': 0, 'right': 0, 'top': 0, 'bottom': 0}}
                else:
                    print(f'\n[{out_stem} Bound] Cell {row}-{col}: {boundary_result.reason}; using the original GDS crop fallback.')
            cell_stem = f'{out_stem}_cell_{row}-{col}'
            metadata = {'wafer_id': out_stem, 'cell_row': row, 'cell_col': col, 'cell_stem': cell_stem, 'gds_bbox_um': [float(min_x), float(min_y), float(max_x), float(max_y)], 'gds_corners_um': [[float(a), float(b)] for a, b in gds_corners], 'canvas_corners_px': pts_canvas.tolist(), 'local_origin_px': [int(x1), int(y1)], 'local_size_px': [int(local_w), int(local_h)], 'rotation_matrix_2x3': m_rot.tolist(), 'rotated_local_corners_px': pts_rotated_local.tolist(), 'crop_bounds_local_px': [int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)], 'crop_size_px': [int(cell_crop.shape[1]), int(cell_crop.shape[0])], 'overlapping_tiles': [list(map(int, t)) for t in overlapping_tiles], 'flat_angle_rad': float(flat_angle), 'flat_angle_deg': float(flat_angle * 180.0 / np.pi), 'shave_px': int(shave), 'analysis_png': str((analysis_dir / f'{cell_stem}.png').as_posix()), 'legacy_jpg': str((out_dir / f'{cell_stem}.jpg').as_posix()), 'seam_mask': str((mask_dir / f'{cell_stem}_seam_mask.png').as_posix())}
            metadata['boundary_alignment'] = boundary_metadata
            metadata['crop_source'] = 'native_physical_boundary' if boundary_metadata.get('applied') else 'native_gds_fallback'
            metadata['device_zoom'] = dict(device_zoom_metadata)
            _save_crop_artifacts(out_dir=out_dir, preview_dir=preview_dir, analysis_dir=analysis_dir, mask_dir=mask_dir, meta_dir=meta_dir, out_stem=out_stem, row=row, col=col, cell_crop=cell_crop, rotated_local_masks=rotated_local_masks, crop_bounds=(crop_x1, crop_y1, crop_x2, crop_y2), metadata=metadata)
            cell_index_records.append(metadata)
            saved_count += 1
            sys.stdout.write(f'\r[{out_stem} Normalized Crop] Process: {idx + 1}/{len(cells)} | Saved cell {row}-{col} -> analysis_png\\{cell_stem}.png\x1b[K')
            sys.stdout.flush()
        _save_cell_index_payload(meta_dir / f'{out_stem}_cell_index.json', {'wafer_id': out_stem, 'count': int(saved_count), 'cells': cell_index_records}, merge=bool(getattr(args, 'cell', None)))
        print(f'\n[{out_stem}] Slicing complete. Extracted {saved_count} cells.')
        print(f'[{out_stem}] Lossless detector inputs: {analysis_dir}')
        print(f'[{out_stem}] Seam masks: {mask_dir}')
        print(f'[{out_stem}] Crop metadata: {meta_dir}')
    else:
        print(f'[{out_stem}] Slicing skipped. Target cells loaded directly.')
    if args.device and run_label:
        mapper = defect_mapper_gui.DeviceDefectMapperTool(wafer_id=out_stem, cells=cells, out_dir=out_dir, transformer=transformer, gds_R=gds_R, config=config_run, shave=args.shave, pad=200)
        mapper.run()
    return True
H2P_TARGETED_EXTRACTION_V1 = True

def _parse_cell_selector(value: str) -> tuple[int, int]:
    text = str(value).strip()
    match = re.fullmatch('(\\d+)\\s*-\\s*(\\d+)', text)
    if not match:
        raise argparse.ArgumentTypeError(f'invalid cell {value!r}; expected ROW-COL, for example 3-7')
    return (int(match.group(1)), int(match.group(2)))

def _select_batch_records(records: list[dict], wafer_selectors, folder_selectors) -> list[dict]:
    wafer_values = [str(value).strip() for value in wafer_selectors or [] if str(value).strip()]
    folder_values = [str(value).strip() for value in folder_selectors or [] if str(value).strip()]
    if not wafer_values and not folder_values:
        return list(records)

    requested_wafers = {value.casefold() for value in wafer_values}
    requested_folders = {value.casefold() for value in folder_values}
    matched_wafers: set[str] = set()
    matched_folders: set[str] = set()
    selected: list[dict] = []

    for record in records:
        wafer_id = str(record['id']).strip()
        wafer_key = wafer_id.casefold()
        aliases = {wafer_key}
        if wafer_key.startswith('wafer_'):
            aliases.add(wafer_id[6:].casefold())

        group = str(record.get('source_group') or '').strip()
        group_key = group.casefold() if group else ''

        wafer_hits = requested_wafers & aliases
        folder_hits = {group_key} if group_key and group_key in requested_folders else set()
        if wafer_hits or folder_hits:
            selected.append(record)
            matched_wafers.update(wafer_hits)
            matched_folders.update(folder_hits)

    missing_wafers = sorted(requested_wafers - matched_wafers)
    missing_folders = sorted(requested_folders - matched_folders)
    if missing_wafers or missing_folders:
        available_wafers = ', '.join(str(record['id']) for record in records)
        available_groups = sorted(
            {str(record.get('source_group')).strip() for record in records if record.get('source_group')},
            key=str.casefold,
        )
        group_text = ', '.join(available_groups) if available_groups else '(none)'
        group_keys = {group.casefold() for group in available_groups}
        errors = []
        for missing in missing_wafers:
            if missing in group_keys:
                errors.append(f"{missing!r} is a folder_name group; use --folder {missing}")
            else:
                errors.append(f"unknown wafer selector {missing!r}")
        for missing in missing_folders:
            errors.append(f"unknown folder selector {missing!r}")
        raise ValueError(
            '; '.join(errors)
            + f". Available wafers: {available_wafers or '(none)'}. "
            + f'Available folder_name groups: {group_text}.'
        )
    return selected

def _filter_selected_cells(cells: list[dict], requested_cells, wafer_id: str) -> list[dict]:
    if not requested_cells:
        return cells
    wanted = {tuple(map(int, value)) for value in requested_cells}
    available = {(int(cell['row']), int(cell['col'])) for cell in cells}
    missing = sorted(wanted - available)
    if missing:
        missing_text = ', '.join((f'{row}-{col}' for row, col in missing))
        available_text = ', '.join((f'{row}-{col}' for row, col in sorted(available)))
        raise ValueError(f'[{wafer_id}] Unknown device cell(s): {missing_text}. Available cells: {available_text}')
    selected = [cell for cell in cells if (int(cell['row']), int(cell['col'])) in wanted]
    labels = ', '.join((f"{int(cell['row'])}-{int(cell['col'])}" for cell in selected))
    print(f'[{wafer_id}] Selected device cells: {labels}')
    return selected

def _save_cell_index_payload(path, payload: dict, *, merge: bool) -> None:
    path = Path(path)
    if not merge or not path.exists():
        illumination_stitching.save_cell_metadata_json(path, payload)
        return
    try:
        existing = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        existing = {}
    if not isinstance(existing, dict):
        existing = {}
    existing_cells = existing.get('cells', [])
    new_cells = payload.get('cells', [])
    if not isinstance(existing_cells, list):
        existing_cells = []
    if not isinstance(new_cells, list):
        new_cells = []

    def record_key(record):
        if not isinstance(record, dict):
            return None
        try:
            return (int(record['row']), int(record['col']))
        except Exception:
            return None
    replacements = {record_key(record): record for record in new_cells if record_key(record) is not None}
    merged = []
    used: set[tuple[int, int]] = set()
    for record in existing_cells:
        key = record_key(record)
        if key is not None and key in replacements:
            merged.append(replacements[key])
            used.add(key)
        else:
            merged.append(record)
            if key is not None:
                used.add(key)
    for record in new_cells:
        key = record_key(record)
        if key is None or key not in used:
            merged.append(record)
            if key is not None:
                used.add(key)
    output = dict(existing)
    output.update(payload)
    output['cells'] = merged
    output['count'] = len(merged)
    old_source = existing.get('crop_source')
    new_source = payload.get('crop_source')
    if old_source and new_source and (old_source != new_source):
        output.pop('crop_source', None)
    illumination_stitching.save_cell_metadata_json(path, output)

def _safe_remove_tree(path: Path) -> None:
    """Remove a generated output folder, with guardrails against bad paths."""
    resolved = path.expanduser().resolve()
    cwd = Path.cwd().resolve()
    forbidden = {cwd, cwd.parent, Path(resolved.anchor).resolve()}
    if resolved in forbidden or str(resolved) in ('', '.'):
        raise RuntimeError(f'Refusing to delete unsafe output path: {path}')
    if not resolved.exists():
        return
    if not resolved.is_dir():
        raise RuntimeError(f'Refusing to delete non-directory output path: {path}')
    print(f'[Cleanup] Removing stale extraction folder: {resolved}')
    shutil.rmtree(resolved)

def _apply_simple_create_defaults(args) -> None:
    """Make `python wafer_alignment_and_extraction.py -c` match the normal H2P workflow."""
    if args.create and (not args.no_manual):
        args.manual = True
    if (args.create and (not args.no_clean)) and (not getattr(args, '_targeted_extraction', False)):
        _safe_remove_tree(Path(args.out_dir))

def main():
    parser = argparse.ArgumentParser(description='Standalone Metrology Core and GDS Extraction Service')
    parser.add_argument('--batch', type=str, default='batch_wafers.txt', help='Path to wafer batch definitions txt config file. Default: batch_wafers.txt')
    parser.add_argument('--manual', action='store_true', help='Launch interactive manual adjustment dashboard beforehand')
    parser.add_argument('--no-manual', action='store_true', help='Do not launch manual alignment GUI in simple -c mode')
    parser.add_argument('--no-clean', action='store_true', help='Keep existing output folder instead of deleting it before -c')
    parser.add_argument('--shave', type=int, default=10, help='Crop outer buffer width inside rotated cell frame')
    parser.add_argument('--zoom', action='store_true', help='Enable the measured centered device crop zoom. Default: off')
    parser.add_argument('--trim', action='store_true', help='Trim 14 pixels from the bottom of each cell crop. Default: off')
    parser.add_argument('--device-zoom-x', type=float, default=DEFAULT_DEVICE_ZOOM_X, help='Zoom X used by --zoom. Default: 1.037710')
    parser.add_argument('--device-zoom-y', type=float, default=DEFAULT_DEVICE_ZOOM_Y, help='Zoom Y used by --zoom. Default: 1.071442')
    parser.add_argument('--device-bottom-trim-px', type=int, default=None, help='Explicit bottom trim in pixels. Overrides --trim; default: 0 (14 with --trim)')
    parser.add_argument('--out-dir', type=str, default='extracted_cells', help='Target output subdirectory to store cropped files')
    parser.add_argument('-d', '--device', action='store_true', help='Enable defect inspection annotation dashboard review')
    parser.add_argument('-c', '--create', action='store_true', help='Stage 1: Generate native cropped die formats')
    parser.add_argument('-l', '--label', action='store_true', help='Stage 2: Label anomalies on extracted images')
    parser.add_argument('--centroid-debug', action='store_true', help='Enable debug mode in automated alignment')
    parser.add_argument('--no-illumination-normalize', action='store_true', help='Disable flat-field tile illumination normalization')
    parser.add_argument('--no-brightness-match', action='store_true', help='Disable global tile brightness matching')
    parser.add_argument('--illumination-strength', type=float, default=1.0, help='Flat-field correction strength, 0..1. Default: 1.0')
    parser.add_argument('--illumination-blur-sigma-frac', type=float, default=0.18, help='Per-tile smooth background blur sigma fraction. Default: 0.18')
    parser.add_argument('--brightness-match-strength', type=float, default=0.65, help='Global brightness matching strength. Default: 0.65')
    parser.add_argument('--illumination-brightness-samples', type=int, default=250, help='Tiles sampled for global luminance. Default: 250')
    parser.add_argument('--tile-cache-size', type=int, default=96, help='Full-resolution normalized tile cache size. Default: 96')
    parser.add_argument('--crop-source', choices=['fast', 'native'], default='fast', help='fast crops from the already-built downscaled stitch; native re-stitches full-res tiles per cell. Default: fast')
    parser.add_argument('--stitch-downscale', type=float, default=12.0, help='Downscale divisor for the stitched wafer canvas. Smaller = sharper/slower. Default: 12')
    parser.add_argument('--fast-crop-width', type=int, default=0, help='Maximum width for fast analysis PNG crops. 0 keeps the stitched-canvas crop resolution. Default: 0')
    parser.add_argument('--fast-crop-pad', type=int, default=200, help='Raw-pixel padding around cell before fast rotate/crop. Default: 200')
    parser.add_argument('--preview-width', type=int, default=0, help='Maximum width of JPG review previews. 0 keeps crop width. Default: 0 for best review quality')
    parser.add_argument('--no-shared-flatfield', action='store_true', help='Disable shared tile-coordinate flat-field model')
    parser.add_argument('--no-overlap-leveling', action='store_true', help='Disable overlap-based LAB-L seam leveling')
    parser.add_argument('--shared-flatfield-samples', type=int, default=400, help='Tiles sampled to build shared flat-field')
    parser.add_argument('--shared-flatfield-model-side', type=int, default=384, help='Max side length of shared flat-field model')
    parser.add_argument('--overlap-leveling-strength', type=float, default=0.85, help='Strength of overlap LAB-L leveling')
    parser.add_argument('--bound', action='store_true', help='Use the transformed GDS cell only as a seed, detect the four physical cell borders, and perspective-rectify a native-resolution crop.')
    parser.add_argument('--bound-pad', type=int, default=250, help='Minimum raw pixels searched beyond each approximate GDS cell. Default: 250')
    parser.add_argument('--bound-expand', type=float, default=0.18, help="Search padding as a fraction of the approximate cell's longer side. Default: 0.18")
    parser.add_argument('--bound-search-inward-fraction', type=float, default=0.08, help='Permit border search slightly inside the GDS seed edge. Default: 0.08')
    parser.add_argument('--bound-min-confidence', type=float, default=0.22, help='Minimum border-fit confidence before falling back to the GDS crop. Default: 0.22')
    parser.add_argument('--bound-output-scale', type=float, default=1.0, help='Scale applied to the rectified native crop. Default: 1.0')
    parser.add_argument('--bound-debug', action='store_true', help='Write orange seed and green detected-border overlays under boundary_debug.')
    parser.add_argument('--bound-native-scale', type=float, default=0.2, help='Resolution of each bounded local stitch relative to raw tiles. 0.20 is 2.4x sharper than a 1/12 stitch when the memory cap allows it. Default: 0.20')
    parser.add_argument('--bound-max-local-megapixels', type=float, default=8.0, help='Maximum pixels in one scaled local stitch. Default: 8 MP')
    parser.add_argument('--bound-detect-max-side', type=int, default=1600, help='Maximum side used by the expensive boundary feature pass; the final crop still uses the higher-resolution local stitch. Default: 1600')
    parser.add_argument('--bound-tile-cache-size', type=int, default=64, help='Number of scaled normalized tiles reused across neighboring cells. Default: 64')
    parser.add_argument('--bound-opencv-threads', type=int, default=1, help='OpenCV worker threads while --bound is active. Default: 2')
    parser.add_argument('--bound-workers', type=int, default=3, help='Number of cells processed concurrently by --bound. Default: 3. Use 1 for minimum RAM or exact v4 execution order.')
    parser.add_argument('--bound-exact-jpeg-decode', action='store_true', help='Disable the faster reduced-DCT JPEG decode path. This reproduces v4 tile decoding at the cost of longer runtime.')
    parser.add_argument('--wafer', action='append', default=[], metavar='NAME', help='Stage-1 wafer selector. Accepts a declared wafer name or canonical wafer ID. Repeat to select multiple.')
    parser.add_argument('--folder', action='append', default=[], metavar='NAME', help='Stage-1 folder_name group selector. Repeat to select multiple folder groups.')
    parser.add_argument('--cell', action='append', default=[], type=_parse_cell_selector, metavar='ROW-COL', help='Create only this device cell, for example 3-7. Repeat --cell to select multiple device cells.')
    args = parser.parse_args()
    args._targeted_extraction = bool(args.wafer or args.folder or args.cell)
    print(f'[Runtime] version={WAFER_EXTRACTION_VERSION}')
    _apply_simple_create_defaults(args)
    _normalize_device_crop_cli(args, parser)
    if args.create:
        if args.no_device_zoom:
            print('[Extraction] Physical-device crop zoom: disabled (use --zoom to enable)')
        else:
            print(f'[Extraction] Physical-device crop zoom: x={args.device_zoom_x:.6f}, y={args.device_zoom_y:.6f}')
        if int(args.device_bottom_trim_px) > 0:
            print(f'[Extraction] Device bottom trim: {int(args.device_bottom_trim_px)} px')
        else:
            print('[Extraction] Device bottom trim: disabled (use --trim to enable)')
    try:
        config = load_config('config.json')
    except Exception as e:
        print(f'Error reading parameters config file: {e}')
        sys.exit(1)
    try:
        wafers = parse_batch_file(args.batch)
        print(f'Discovered {len(wafers)} configuration sequences configured in batch run.')
    except Exception as e:
        print(f'Error parsing instruction batch file: {e}')
        sys.exit(1)
    wafers = _select_batch_records(wafers, args.wafer, args.folder)
    print('[Batch] Selected wafers: ' + ', '.join((str(wafer['id']) for wafer in wafers)))
    if args.create and (not args.no_clean) and (args.wafer or args.folder) and (not args.cell):
        for wafer in wafers:
            _safe_remove_tree(Path(args.out_dir) / str(wafer['id']))
    elif args.create and args.cell and (not args.no_clean):
        print('[Extraction] Partial-cell mode: preserving existing wafer outputs; selected cell artifacts will be overwritten/updated in place.')
    for idx, wafer in enumerate(wafers):
        wafer_id = wafer['id']
        after_folder = wafer['after_folder']
        defect_json = wafer['defect_json']
        print('\n' + '=' * 70)
        print(f' WAFER RUN [{idx + 1}/{len(wafers)}]: {wafer_id}')
        print('=' * 70)
        if after_folder.lower() == 'none' or not after_folder:
            continue
        if not Path(after_folder).exists():
            print(f'[{wafer_id}] Skipping missing after-folder: {after_folder}')
            continue
        try:
            process_wafer_cells(folder=after_folder, json_file=defect_json, config=config, args=args, wafer_id=wafer_id)
        except Exception as e:
            print(f'[{wafer_id}] Script crashed during operational pipeline execution: {e}')
            import traceback
            traceback.print_exc()
    print('\n' + '=' * 70)
    print(' BATCH EXECUTION RUN COMPLETE')
    print('=' * 70)
if __name__ == '__main__':
    main()
_PER_WAFER_OUTPUT_LAYOUT_V22 = 'per-wafer-output-v22-2026-07-23'
