"""ROI-aware launcher for defect_detector.py.

The extraction remains the authoritative, full layer-8 crop.  This launcher lets
automatic detection operate on the recurring layer-5 device rectangle, then maps
all detected boxes and polygons back into full-crop pixels before the normal GDS
conversion and review UI run.

It is intentionally a launcher instead of a replacement for defect_detector.py,
so existing detector changes remain untouched.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path
import design_geometry
from typing import Any, Optional

import cv2
import numpy as np

import defect_detector as detector


WRAPPER_VERSION = "layer5-analysis-roi-inverse-map-v1.2-border-padding-2026-07-27"
DEFAULT_ANALYSIS_ROI_PADDING_UM = 100.0
def _arg_value(argv: list[str], name: str, default: str = "") -> str:
    for i, token in enumerate(argv):
        if token == name and i + 1 < len(argv):
            return argv[i + 1]
        prefix = name + "="
        if token.startswith(prefix):
            return token[len(prefix):]
    return default


def _pop_flag(argv: list[str], name: str) -> bool:
    found = False
    while name in argv:
        argv.remove(name)
        found = True
    return found


def _pop_option(argv: list[str], name: str, default: str) -> str:
    value = default
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == name:
            if i + 1 >= len(argv):
                raise SystemExit(f"{name} requires a value")
            value = argv[i + 1]
            del argv[i:i + 2]
            continue
        prefix = name + "="
        if token.startswith(prefix):
            value = token[len(prefix):]
            del argv[i]
            continue
        i += 1
    return value


def _resolve_gds_path(argv: list[str], config_path: Path, explicit: str) -> Path:
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else Path.cwd() / path

    design_gds = _arg_value(argv, "--design-gds", "")
    if design_gds:
        path = Path(design_gds)
        return path if path.is_absolute() else Path.cwd() / path

    preferred = design_geometry.resolve_design_path()
    if preferred.exists():
        return preferred

    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            raw = str(cfg.get("gds_path", "") or "").strip()
            if raw:
                path = Path(raw)
                return path if path.is_absolute() else config_path.parent / path
        except Exception:
            pass
    return preferred


def _fit_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    A: list[list[float]] = []
    b: list[float] = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1.0, 0.0, 0.0, 0.0])
        b.append(u)
        A.append([0.0, 0.0, 0.0, x, y, 1.0])
        b.append(v)
    coeff, *_ = np.linalg.lstsq(np.asarray(A), np.asarray(b), rcond=None)
    return coeff.reshape(2, 3)


def _apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    hom = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    return (np.asarray(matrix, dtype=np.float64) @ hom.T).T


class AnalysisRoiManager:
    def __init__(
        self,
        *,
        metadata_dir: Path,
        gds_path: Path,
        layer: int = 5,
        datatype: int = 0,
        padding_um: float = DEFAULT_ANALYSIS_ROI_PADDING_UM,
    ) -> None:
        self.metadata_dir = Path(metadata_dir)
        self.gds_path = Path(gds_path)
        self.layer = int(layer)
        self.datatype = int(datatype)
        self.padding_um = max(0.0, float(padding_um))
        self._rectangles: list[dict[str, float]] = []
        self._transform_cache: dict[str, dict[str, Any]] = {}
        self._last_path: Optional[Path] = None
        self._last_full_image: Optional[np.ndarray] = None
        self.available = False
        self.error = ""
        self._load_gds_rectangles()

    def _load_gds_rectangles(self) -> None:
        try:
            import gdstk  # type: ignore

            if not self.gds_path.exists():
                raise FileNotFoundError(self.gds_path)
            lib = gdstk.read_gds(str(self.gds_path))
            tops = lib.top_level()
            if not tops:
                raise ValueError("GDS contains no top-level cell")
            flat = tops[0].copy("__H2P_ANALYSIS_ROI_FLAT__")
            flat.flatten()
            records: list[dict[str, float]] = []
            for poly in flat.polygons:
                if int(poly.layer) != self.layer or int(poly.datatype) != self.datatype:
                    continue
                pts = np.asarray(poly.points, dtype=np.float64)
                if pts.ndim != 2 or len(pts) < 3:
                    continue
                lo = np.min(pts, axis=0)
                hi = np.max(pts, axis=0)
                width = float(hi[0] - lo[0])
                height = float(hi[1] - lo[1])
                if width <= 0 or height <= 0:
                    continue
                area = abs(float(poly.area()))
                bbox_area = width * height
                fill = area / max(bbox_area, 1e-12)
                if fill < 0.97:
                    continue
                records.append({
                    "x0": float(lo[0]), "y0": float(lo[1]),
                    "x1": float(hi[0]), "y1": float(hi[1]),
                    "cx": float((lo[0] + hi[0]) * 0.5),
                    "cy": float((lo[1] + hi[1]) * 0.5),
                    "width": width, "height": height, "area": area,
                })
            if not records:
                raise ValueError(f"No rectangular polygons found on layer {self.layer}/{self.datatype}")
            self._rectangles = records
            self.available = True
        except Exception as exc:
            self.error = str(exc)
            self.available = False

    def _metadata_path(self, image_path: Path) -> Optional[Path]:
        finder = getattr(detector, "_find_metadata_for_image", None)
        if callable(finder):
            found = finder(image_path.name, self.metadata_dir)
            if found is not None:
                return Path(found)
        direct = self.metadata_dir / f"{image_path.stem}.json"
        return direct if direct.exists() else None

    def _select_gds_roi(self, meta: dict[str, Any]) -> Optional[dict[str, float]]:
        bbox = meta.get("gds_bbox_um")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None
        x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
        cw = x1 - x0
        ch = y1 - y0
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5
        candidates: list[tuple[float, float, dict[str, float]]] = []
        for rec in self._rectangles:
            if rec["width"] < 0.70 * cw or rec["height"] < 0.70 * ch:
                continue
            if rec["width"] > 1.05 * cw or rec["height"] > 1.05 * ch:
                continue
            dist = math.hypot(rec["cx"] - cx, rec["cy"] - cy)
            if dist > 0.25 * max(cw, ch):
                continue
            # Prefer the nearest center, then the largest enclosing-device rectangle.
            candidates.append((dist, -rec["area"], rec))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        rec = dict(candidates[0][2])
        pad = self.padding_um
        rec["x0"] = max(x0, rec["x0"] - pad)
        rec["y0"] = max(y0, rec["y0"] - pad)
        rec["x1"] = min(x1, rec["x1"] + pad)
        rec["y1"] = min(y1, rec["y1"] + pad)
        return rec

    def transform_for(self, image_path: Path, full_shape: Optional[tuple[int, int]] = None) -> Optional[dict[str, Any]]:
        image_path = Path(image_path)
        key = str(image_path.resolve()).lower()
        cached = self._transform_cache.get(key)
        if cached is not None:
            return cached
        if not self.available:
            return None
        meta_path = self._metadata_path(image_path)
        if meta_path is None:
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if full_shape is None:
                size = meta.get("crop_size_px") or []
                if len(size) >= 2:
                    full_w, full_h = int(size[0]), int(size[1])
                else:
                    raw = detector.read_bgr.__wrapped__(image_path) if hasattr(detector.read_bgr, "__wrapped__") else None
                    if raw is None:
                        return None
                    full_h, full_w = raw.shape[:2]
            else:
                full_h, full_w = [int(v) for v in full_shape]
            roi_gds = self._select_gds_roi(meta)
            if roi_gds is None:
                return None

            crop_corners = np.asarray([
                [0.0, 0.0], [float(full_w), 0.0],
                [float(full_w), float(full_h)], [0.0, float(full_h)],
            ], dtype=np.float64)
            gds_corners = np.asarray([
                detector.crop_px_to_gds(float(x), float(y), meta)
                for x, y in crop_corners
            ], dtype=np.float64)
            crop_to_gds = _fit_affine(crop_corners, gds_corners)
            gds_to_crop = cv2.invertAffineTransform(crop_to_gds.astype(np.float64))
            roi_corners_gds = np.asarray([
                [roi_gds["x0"], roi_gds["y0"]],
                [roi_gds["x1"], roi_gds["y0"]],
                [roi_gds["x1"], roi_gds["y1"]],
                [roi_gds["x0"], roi_gds["y1"]],
            ], dtype=np.float64)
            roi_crop = _apply_affine(gds_to_crop, roi_corners_gds)
            x0 = max(0, int(math.floor(float(np.min(roi_crop[:, 0])))))
            y0 = max(0, int(math.floor(float(np.min(roi_crop[:, 1])))))
            x1 = min(full_w, int(math.ceil(float(np.max(roi_crop[:, 0])))))
            y1 = min(full_h, int(math.ceil(float(np.max(roi_crop[:, 1])))))
            if x1 - x0 < 32 or y1 - y0 < 32:
                return None
            transform = {
                "enabled": True,
                "version": WRAPPER_VERSION,
                "layer": self.layer,
                "datatype": self.datatype,
                "padding_um": self.padding_um,
                "full_size_px": [full_w, full_h],
                "roi_bounds_full_px": [x0, y0, x1, y1],
                "roi_size_px": [x1 - x0, y1 - y0],
                "roi_bbox_gds_um": [roi_gds["x0"], roi_gds["y0"], roi_gds["x1"], roi_gds["y1"]],
                "metadata_path": str(meta_path),
            }
            self._transform_cache[key] = transform
            return transform
        except Exception:
            return None

    def crop_image(self, image_path: Path, full: np.ndarray, *, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        self._last_path = Path(image_path)
        self._last_full_image = full
        transform = self.transform_for(Path(image_path), full.shape[:2])
        if transform is None:
            return full
        x0, y0, x1, y1 = transform["roi_bounds_full_px"]
        return full[y0:y1, x0:x1].copy()

    def crop_mask(self, image_path: Path, mask: np.ndarray) -> np.ndarray:
        transform = self.transform_for(Path(image_path), mask.shape[:2])
        if transform is None:
            return mask
        x0, y0, x1, y1 = transform["roi_bounds_full_px"]
        return mask[y0:y1, x0:x1].copy()

    def map_detection_result_to_full(self, image_path: Path, result: dict[str, Any]) -> dict[str, Any]:
        transform = self.transform_for(Path(image_path))
        if transform is None:
            return result
        x0, y0, x1, y1 = [float(v) for v in transform["roi_bounds_full_px"]]
        roi_w = x1 - x0
        roi_h = y1 - y0
        det_shape = result.get("shape") or []
        analysis_shape = result.get("original_shape") or []
        if len(det_shape) < 2 or len(analysis_shape) < 2:
            return result
        det_h, det_w = float(det_shape[0]), float(det_shape[1])
        ana_h, ana_w = float(analysis_shape[0]), float(analysis_shape[1])
        det_to_analysis_x = ana_w / max(det_w, 1.0)
        det_to_analysis_y = ana_h / max(det_h, 1.0)
        analysis_to_full_x = roi_w / max(ana_w, 1.0)
        analysis_to_full_y = roi_h / max(ana_h, 1.0)
        sx = det_to_analysis_x * analysis_to_full_x
        sy = det_to_analysis_y * analysis_to_full_y

        mapped_defects: list[dict[str, Any]] = []
        for defect in result.get("defects", []):
            d = dict(defect)
            bbox = d.get("bbox_px") or []
            if len(bbox) >= 4:
                bx, by, bw, bh = [float(v) for v in bbox[:4]]
                fx0 = x0 + bx * sx
                fy0 = y0 + by * sy
                fx1 = x0 + (bx + bw) * sx
                fy1 = y0 + (by + bh) * sy
                d["bbox_px"] = [
                    int(round(fx0)), int(round(fy0)),
                    max(1, int(round(fx1 - fx0))), max(1, int(round(fy1 - fy0))),
                ]
            polygon = d.get("polygon_px") or []
            if polygon:
                d["polygon_px"] = [
                    [int(round(x0 + float(p[0]) * sx)), int(round(y0 + float(p[1]) * sy))]
                    for p in polygon if len(p) >= 2
                ]
            d["area_px"] = int(round(float(d.get("area_px", 0) or 0) * sx * sy))
            mapped_defects.append(d)

        full_w, full_h = [int(v) for v in transform["full_size_px"]]
        debug = result.get("_debug") or {}
        mapped_debug: dict[str, Any] = {}
        for name, value in debug.items():
            if not isinstance(value, np.ndarray) or value.ndim not in (2, 3):
                mapped_debug[name] = value
                continue
            interpolation = cv2.INTER_NEAREST if value.dtype == np.uint8 and name.endswith("mask") else cv2.INTER_LINEAR
            resized = cv2.resize(value, (int(round(roi_w)), int(round(roi_h))), interpolation=interpolation)
            out_shape = (full_h, full_w) if value.ndim == 2 else (full_h, full_w, value.shape[2])
            canvas = np.zeros(out_shape, dtype=value.dtype)
            ix0, iy0, ix1, iy1 = [int(v) for v in transform["roi_bounds_full_px"]]
            canvas[iy0:iy1, ix0:ix1] = resized[:iy1 - iy0, :ix1 - ix0]
            mapped_debug[name] = canvas

        result = dict(result)
        result["defects"] = mapped_defects
        result["analysis_shape"] = list(result.get("shape") or [])
        result["analysis_original_shape"] = list(result.get("original_shape") or [])
        result["shape"] = [full_h, full_w]
        result["original_shape"] = [full_h, full_w]
        result["scale_to_original_px"] = [1.0, 1.0]
        result["analysis_roi_transform"] = transform
        result["_debug"] = mapped_debug
        return result

    def full_image_for_preview(self) -> Optional[np.ndarray]:
        return self._last_full_image


class RoiAwareDesignMaskProvider:
    def __init__(self, base_provider: Any, manager: AnalysisRoiManager) -> None:
        self._base = base_provider
        self._manager = manager
        for name in ("gds_path", "metadata_dir", "layers", "records", "available", "error"):
            if hasattr(base_provider, name):
                setattr(self, name, getattr(base_provider, name))

    def mask_for_image(self, image_name: str, original_shape: tuple[int, int], output_shape: tuple[int, int]) -> Optional[np.ndarray]:
        meta_path = self._manager._metadata_path(Path(image_name))
        image_path = Path(image_name)
        if meta_path is not None:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                raw_path = str(meta.get("analysis_png", "") or "")
                if raw_path:
                    image_path = Path(raw_path)
            except Exception:
                pass
        transform = self._manager.transform_for(image_path)
        if transform is None:
            return self._base.mask_for_image(image_name, original_shape, output_shape)
        full_w, full_h = transform["full_size_px"]
        full_mask = self._base.mask_for_image(image_name, (full_h, full_w), (full_h, full_w))
        if full_mask is None:
            return None
        roi = self._manager.crop_mask(image_path, full_mask)
        out_h, out_w = [int(v) for v in output_shape]
        if roi.shape[:2] != (out_h, out_w):
            roi = cv2.resize(roi, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        return roi


def install_roi_hooks(manager: AnalysisRoiManager) -> None:
    original_read_bgr = detector.read_bgr
    original_find_seam = detector.find_seam_mask_for_image
    original_detect = detector.detect_cell_defects
    original_draw_overlay = detector.draw_overlay
    original_provider_class = detector.GDSDesignMaskProvider

    def read_bgr_roi(path: Path | str) -> np.ndarray:
        full = original_read_bgr(path)
        return manager.crop_image(Path(path), full)

    read_bgr_roi.__wrapped__ = original_read_bgr  # type: ignore[attr-defined]
    detector.read_bgr = read_bgr_roi

    def find_seam_roi(image_path: Path, seam_mask_dir: Optional[Path]) -> Optional[np.ndarray]:
        mask = original_find_seam(image_path, seam_mask_dir)
        if mask is None:
            return None
        return manager.crop_mask(Path(image_path), mask)

    detector.find_seam_mask_for_image = find_seam_roi

    def detect_roi(*args: Any, **kwargs: Any) -> dict[str, Any]:
        image_path = Path(args[0] if args else kwargs.get("image_path"))
        result = original_detect(*args, **kwargs)
        return manager.map_detection_result_to_full(image_path, result)

    detector.detect_cell_defects = detect_roi

    def draw_full_overlay(img_bgr: np.ndarray, comps: list[Any], kept_mask: np.ndarray, score: np.ndarray) -> np.ndarray:
        full = manager.full_image_for_preview()
        if full is not None and full.shape[:2] == kept_mask.shape[:2]:
            return original_draw_overlay(full, comps, kept_mask, score)
        return original_draw_overlay(img_bgr, comps, kept_mask, score)

    detector.draw_overlay = draw_full_overlay

    class PatchedProvider(original_provider_class):  # type: ignore[misc, valid-type]
        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            base = original_provider_class(*args, **kwargs)
            return RoiAwareDesignMaskProvider(base, manager)

    detector.GDSDesignMaskProvider = PatchedProvider


def main() -> None:
    argv = list(sys.argv[1:])
    disabled = _pop_flag(argv, "--no-analysis-roi")
    layer = int(_pop_option(argv, "--analysis-roi-layer", "5"))
    datatype = int(_pop_option(argv, "--analysis-roi-datatype", "0"))
    padding_um = float(_pop_option(argv, "--analysis-roi-padding-um", str(DEFAULT_ANALYSIS_ROI_PADDING_UM)))
    explicit_gds = _pop_option(argv, "--analysis-roi-gds", "")

    metadata_dir = Path(_arg_value(argv, "--metadata-dir", "extracted_cells/metadata"))
    config_path = Path(_arg_value(argv, "--config", "config.json"))
    gds_path = _resolve_gds_path(argv, config_path, explicit_gds)

    if not disabled:
        manager = AnalysisRoiManager(
            metadata_dir=metadata_dir,
            gds_path=gds_path,
            layer=layer,
            datatype=datatype,
            padding_um=padding_um,
        )
        if manager.available:
            install_roi_hooks(manager)
            print(
                f"[Analysis ROI] version={WRAPPER_VERSION}; GDS={gds_path}; "
                f"layer={layer}/{datatype}; padding={padding_um:.1f} um"
            )
            print("[Analysis ROI] Detection uses the inner device ROI; saved polygons are inverse-mapped to full layer-8 crop pixels.")

            if "--auto-template-cache" not in argv and not any(a.startswith("--auto-template-cache=") for a in argv):
                input_value = _arg_value(argv, "--input", "extracted_cells/analysis_png")
                input_path = Path(input_value)
                parent = input_path.parent if input_path.suffix else input_path.parent
                padding_key = (
                    str(int(round(padding_um)))
                    if abs(padding_um - round(padding_um)) < 1e-9
                    else (f"{padding_um:.3f}".rstrip("0").rstrip(".").replace(".", "p"))
                )
                cache = parent / f"normal_template_auto_roi_l{layer}_{datatype}_p{padding_key}um.npz"
                argv.extend(["--auto-template-cache", str(cache)])
        else:
            print(f"[Analysis ROI] WARNING: disabled because ROI setup failed: {manager.error}")
    else:
        print("[Analysis ROI] Disabled by --no-analysis-roi; detector uses full crops directly.")

    sys.argv = [sys.argv[0], *argv]
    detector.main()


if __name__ == "__main__":
    main()
