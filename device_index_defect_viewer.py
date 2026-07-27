r"""Production-style cross-wafer device-index defect viewer for H2P.

The first screen presents the GDS-derived wafer device map. Selecting a device
opens the matching crop from every wafer, with reviewed or automatic defect
geometry overlaid. The app intentionally remains a single-file Tkinter tool so
it can be dropped into the existing H2P repository without adding new runtime
dependencies beyond Pillow, OpenCV, and NumPy.

Typical use from the repository root::

    python .\device_index_defect_viewer.py

Useful variants::

    python .\device_index_defect_viewer.py --source reviewed
    python .\device_index_defect_viewer.py --source auto
    python .\device_index_defect_viewer.py --summary
    python .\device_index_defect_viewer.py --logo C:\path\to\logo.png
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk
import tkinter as tk
from tkinter import messagebox, ttk


APP_TITLE = "H2P Device Index Defect Viewer"
APP_VERSION = "production-ui-v2-2026-07-27"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
CELL_RE = re.compile(r"(?:^|_)cell_(\d+)-(\d+)$", re.IGNORECASE)
DEFAULT_LOGO_NAME = "h2pLogo.png"


class C:
    """Visual design tokens."""

    BG = "#0B1020"
    SURFACE = "#11182A"
    SURFACE_2 = "#161F34"
    SURFACE_3 = "#1C2740"
    CARD = "#151E31"
    CARD_HOVER = "#1A2640"
    BORDER = "#273654"
    BORDER_SOFT = "#202C46"
    TEXT = "#F5F8FF"
    TEXT_2 = "#B7C2D8"
    TEXT_3 = "#7F8CA7"
    ACCENT = "#38D6C5"
    ACCENT_HOVER = "#54E6D7"
    ACCENT_DARK = "#123C40"
    BLUE = "#6EA8FE"
    BLUE_DARK = "#20365E"
    ORANGE = "#FFB547"
    ORANGE_DARK = "#4A3017"
    RED = "#FF6B72"
    RED_DARK = "#4A2029"
    GREEN = "#66D19E"
    GREEN_DARK = "#19382C"
    IMAGE_BG = "#080C16"
    WHITE = "#FFFFFF"


FONT_FAMILY = "Segoe UI Variable"
FONT_FALLBACK = "Segoe UI"


def _font(size: int, weight: str = "normal") -> tuple[str, int, str]:
    return (FONT_FAMILY, size, weight)


def _enable_windows_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


@dataclass(frozen=True, order=True)
class DeviceIndex:
    row: int
    col: int

    @property
    def label(self) -> str:
        return f"{self.row}-{self.col}"


@dataclass
class WaferDeviceEntry:
    wafer_id: str
    device: DeviceIndex
    image_path: Path
    defects: list[dict[str, Any]]
    annotation_source: str
    image_size: tuple[int, int]
    gds_size_um: tuple[float, float] | None


@dataclass
class Catalog:
    wafer_ids: list[str]
    by_device: dict[DeviceIndex, list[WaferDeviceEntry]]
    warnings: list[str]

    @property
    def devices(self) -> list[DeviceIndex]:
        return sorted(self.by_device)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _read_bgr(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _normalize_stem(value: str | Path) -> str:
    return Path(str(value)).stem.strip().casefold()


def _device_from_stem(value: str | Path) -> DeviceIndex | None:
    stem = Path(str(value)).stem.strip()
    match = CELL_RE.search(stem)
    if not match:
        return None
    return DeviceIndex(int(match.group(1)), int(match.group(2)))


def _fallback_parse_batch(batch_path: Path) -> list[str]:
    result: list[str] = []
    if not batch_path.exists():
        return result
    lines: list[str] = []
    for raw in batch_path.read_text(encoding="utf-8-sig").splitlines():
        value = raw.split("#", 1)[0].strip().strip('"').strip("'")
        if value:
            lines.append(value)
    i = 0
    while i < len(lines):
        value = lines[i]
        if value.endswith(":"):
            result.append(value[:-1].strip())
            i += 4
        else:
            result.append(value if value.casefold().startswith("wafer_") else f"Wafer_{value}")
            i += 1
    return result


def load_wafer_ids(batch_path: Path, extracted_root: Path) -> list[str]:
    wafer_ids: list[str] = []
    try:
        from batch_wafers_parser import parse_batch_file  # type: ignore

        if batch_path.exists():
            wafer_ids = [str(item["id"]) for item in parse_batch_file(batch_path)]
    except Exception:
        wafer_ids = _fallback_parse_batch(batch_path)

    discovered: list[str] = []
    if extracted_root.exists():
        discovered = sorted(
            p.name
            for p in extracted_root.iterdir()
            if p.is_dir() and p.name.casefold().startswith("wafer_")
        )

    seen: set[str] = set()
    merged: list[str] = []
    for wafer_id in [*wafer_ids, *discovered]:
        key = wafer_id.casefold()
        if key not in seen:
            seen.add(key)
            merged.append(wafer_id)
    return merged


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not read {path}: {exc}") from exc


def _index_annotation_data(data: Any) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    if isinstance(data, dict) and isinstance(data.get("images"), list):
        for item in data["images"]:
            if not isinstance(item, dict):
                continue
            image_name = item.get("image") or item.get("filename") or item.get("name")
            defects = item.get("defects", [])
            if image_name and isinstance(defects, list):
                result[_normalize_stem(image_name)] = [d for d in defects if isinstance(d, dict)]
        return result

    if isinstance(data, dict):
        for image_name, defects in data.items():
            if isinstance(defects, list):
                result[_normalize_stem(image_name)] = [d for d in defects if isinstance(d, dict)]
    return result


def _annotation_file_for_wafer(wafer_dir: Path, wafer_id: str, source: str) -> tuple[Path | None, str]:
    reviewed = wafer_dir / f"{wafer_id}_device_defects.json"
    auto = wafer_dir / "algo_defects.json"
    if source == "reviewed":
        return (reviewed if reviewed.exists() else None, "reviewed")
    if source == "auto":
        return (auto if auto.exists() else None, "automatic")
    if reviewed.exists():
        return reviewed, "reviewed"
    if auto.exists():
        return auto, "automatic"
    return None, "none"


def _iter_crop_images(wafer_dir: Path) -> Iterable[Path]:
    search_roots = [wafer_dir / "analysis_png", wafer_dir / "previews", wafer_dir]
    seen: set[str] = set()
    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
                continue
            if _device_from_stem(path) is None:
                continue
            key = _normalize_stem(path)
            if key in seen:
                continue
            seen.add(key)
            yield path


def _load_gds_size_um(wafer_dir: Path, image_path: Path) -> tuple[float, float] | None:
    metadata_path = wafer_dir / "metadata" / f"{image_path.stem}.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = _load_json(metadata_path)
        bbox = metadata.get("gds_bbox_um") if isinstance(metadata, dict) else None
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
            width = abs(x1 - x0)
            height = abs(y1 - y0)
            if width > 0.0 and height > 0.0:
                return width, height
    except Exception:
        return None
    return None


def build_catalog(extracted_root: Path, batch_path: Path, source: str) -> Catalog:
    wafer_ids = load_wafer_ids(batch_path, extracted_root)
    by_device: dict[DeviceIndex, list[WaferDeviceEntry]] = defaultdict(list)
    warnings: list[str] = []

    for wafer_id in wafer_ids:
        wafer_dir = extracted_root / wafer_id
        if not wafer_dir.exists():
            warnings.append(f"Missing extracted wafer directory: {wafer_dir}")
            continue

        annotation_path, annotation_source = _annotation_file_for_wafer(wafer_dir, wafer_id, source)
        annotations: dict[str, list[dict[str, Any]]] = {}
        if annotation_path is not None:
            try:
                annotations = _index_annotation_data(_load_json(annotation_path))
            except Exception as exc:
                warnings.append(str(exc))
        else:
            warnings.append(f"No {source} annotation JSON found for {wafer_id}")

        crop_paths = list(_iter_crop_images(wafer_dir))
        if not crop_paths:
            warnings.append(f"No cell crops found for {wafer_id}")
            continue

        for image_path in crop_paths:
            device = _device_from_stem(image_path)
            if device is None:
                continue
            bgr = _read_bgr(image_path)
            if bgr is None:
                warnings.append(f"Could not decode image: {image_path}")
                continue
            h, w = bgr.shape[:2]
            defects = annotations.get(_normalize_stem(image_path), [])
            by_device[device].append(
                WaferDeviceEntry(
                    wafer_id=wafer_id,
                    device=device,
                    image_path=image_path,
                    defects=defects,
                    annotation_source=annotation_source,
                    image_size=(w, h),
                    gds_size_um=_load_gds_size_um(wafer_dir, image_path),
                )
            )

    order = {wafer.casefold(): i for i, wafer in enumerate(wafer_ids)}
    for entries in by_device.values():
        entries.sort(key=lambda e: (order.get(e.wafer_id.casefold(), 10**9), e.wafer_id.casefold()))
    return Catalog(wafer_ids=wafer_ids, by_device=dict(by_device), warnings=warnings)


# ---------------------------------------------------------------------------
# Image rendering
# ---------------------------------------------------------------------------


def _points_from_defect(defect: dict[str, Any]) -> np.ndarray | None:
    points = defect.get("polygon_px")
    if isinstance(points, (list, tuple)) and len(points) >= 3:
        try:
            arr = np.asarray([[float(x), float(y)] for x, y in points], dtype=np.float32)
            if arr.shape[0] >= 3:
                return arr
        except Exception:
            pass

    box = defect.get("box_px") or defect.get("bbox_px")
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        try:
            x, y, w, h = [float(v) for v in box[:4]]
            return np.asarray(
                [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                dtype=np.float32,
            )
        except Exception:
            return None
    return None


def draw_defect_overlay(
    image_bgr: np.ndarray,
    defects: list[dict[str, Any]],
    *,
    opacity: float = 0.18,
    enabled: bool = True,
) -> np.ndarray:
    out = image_bgr.copy()
    if not enabled or not defects:
        return out

    fill_layer = out.copy()
    valid_polygons: list[np.ndarray] = []
    for defect in defects:
        points = _points_from_defect(defect)
        if points is None:
            continue
        pts = np.round(points).astype(np.int32).reshape((-1, 1, 2))
        valid_polygons.append(pts)
        cv2.fillPoly(fill_layer, [pts], (40, 174, 255), lineType=cv2.LINE_AA)

    opacity = float(np.clip(opacity, 0.0, 0.75))
    cv2.addWeighted(fill_layer, opacity, out, 1.0 - opacity, 0.0, dst=out)
    for pts in valid_polygons:
        cv2.polylines(out, [pts], True, (8, 15, 30), 6, cv2.LINE_AA)
        cv2.polylines(out, [pts], True, (40, 174, 255), 3, cv2.LINE_AA)
    return out


def _fit_image(image_bgr: np.ndarray, max_width: int, max_height: int | None = None) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if max_height is None:
        max_height = max_width
    scale = min(1.0, float(max_width) / max(w, 1), float(max_height) / max(h, 1))
    if scale >= 0.999:
        return image_bgr
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA)


def _to_photo(image_bgr: np.ndarray) -> ImageTk.PhotoImage:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


# ---------------------------------------------------------------------------
# UI primitives
# ---------------------------------------------------------------------------


def _bind_click_recursive(widget: tk.Widget, callback: Callable[[], None]) -> None:
    widget.configure(cursor="hand2")
    widget.bind("<Button-1>", lambda _e: callback())
    for child in widget.winfo_children():
        _bind_click_recursive(child, callback)


class HoverButton(tk.Button):
    def __init__(
        self,
        master: tk.Misc,
        *,
        text: str,
        command: Callable[[], None],
        bg: str = C.SURFACE_3,
        hover_bg: str = C.BORDER,
        fg: str = C.TEXT,
        padx: int = 15,
        pady: int = 9,
        font: tuple[str, int, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            master,
            text=text,
            command=command,
            bg=bg,
            activebackground=hover_bg,
            fg=fg,
            activeforeground=fg,
            relief="flat",
            bd=0,
            highlightthickness=0,
            padx=padx,
            pady=pady,
            font=font or _font(10, "bold"),
            cursor="hand2",
            **kwargs,
        )
        self._normal_bg = bg
        self._hover_bg = hover_bg
        self.bind("<Enter>", lambda _e: self.configure(bg=self._hover_bg))
        self.bind("<Leave>", lambda _e: self.configure(bg=self._normal_bg))


class TogglePill(tk.Frame):
    def __init__(self, master: tk.Misc, text: str, variable: tk.BooleanVar, command: Callable[[], None]) -> None:
        super().__init__(master, bg=C.SURFACE_2)
        self.variable = variable
        self.command = command
        self.button = tk.Button(
            self,
            text="",
            command=self._toggle,
            relief="flat",
            bd=0,
            highlightthickness=0,
            cursor="hand2",
            font=_font(9, "bold"),
            padx=11,
            pady=7,
        )
        self.button.pack()
        self.label = text
        self._refresh()

    def _toggle(self) -> None:
        self.variable.set(not self.variable.get())
        self._refresh()
        self.command()

    def _refresh(self) -> None:
        on = self.variable.get()
        self.button.configure(
            text=f"{'â—' if on else 'â—‹'}  {self.label}",
            bg=C.ACCENT_DARK if on else C.SURFACE_3,
            activebackground=C.ACCENT_DARK if on else C.BORDER,
            fg=C.ACCENT if on else C.TEXT_2,
            activeforeground=C.ACCENT if on else C.TEXT,
        )


class StatCard(tk.Frame):
    def __init__(self, master: tk.Misc, label: str, value: str, accent: str = C.ACCENT) -> None:
        super().__init__(master, bg=C.CARD, highlightthickness=1, highlightbackground=C.BORDER_SOFT)
        stripe = tk.Frame(self, bg=accent, width=4)
        stripe.pack(side="left", fill="y")
        body = tk.Frame(self, bg=C.CARD, padx=14, pady=10)
        body.pack(side="left", fill="both", expand=True)
        tk.Label(body, text=value, bg=C.CARD, fg=C.TEXT, font=_font(18, "bold")).pack(anchor="w")
        tk.Label(body, text=label.upper(), bg=C.CARD, fg=C.TEXT_3, font=_font(8, "bold")).pack(anchor="w", pady=(1, 0))


class ScrollableFrame(tk.Frame):
    def __init__(self, master: tk.Misc, *, bg: str = C.BG) -> None:
        super().__init__(master, bg=bg)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.inner.bind("<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.window_id, width=e.width))
        self.canvas.bind("<Enter>", lambda _e: self._bind_wheel())
        self.canvas.bind("<Leave>", lambda _e: self._unbind_wheel())

    def _bind_wheel(self) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)

    def _unbind_wheel(self) -> None:
        self.canvas.unbind_all("<MouseWheel>")

    def _on_wheel(self, event: tk.Event) -> None:
        delta = getattr(event, "delta", 0)
        if delta:
            self.canvas.yview_scroll(int(-delta / 120), "units")


class DeviceTile(tk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        *,
        device: DeviceIndex,
        crop_count: int,
        wafers_with_defects: int,
        defect_count: int,
        command: Callable[[], None],
    ) -> None:
        self.has_defects = defect_count > 0
        bg = C.ORANGE_DARK if self.has_defects else C.CARD
        border = C.ORANGE if self.has_defects else C.BORDER
        hover = "#57391C" if self.has_defects else C.CARD_HOVER
        super().__init__(master, bg=bg, highlightthickness=1, highlightbackground=border, padx=12, pady=10)
        self._base_bg = bg
        self._hover_bg = hover
        self._command = command

        top = tk.Frame(self, bg=bg)
        top.pack(fill="x")
        tk.Label(top, text=device.label, bg=bg, fg=C.TEXT, font=_font(16, "bold")).pack(side="left")
        badge_text = str(defect_count) if self.has_defects else "CLEAN"
        badge_bg = C.ORANGE if self.has_defects else C.GREEN_DARK
        badge_fg = C.BG if self.has_defects else C.GREEN
        badge = tk.Label(
            top,
            text=badge_text,
            bg=badge_bg,
            fg=badge_fg,
            font=_font(8, "bold"),
            padx=7,
            pady=3,
        )
        badge.pack(side="right")

        coverage = f"{wafers_with_defects}/{crop_count} wafers flagged" if crop_count else "No crops"
        tk.Label(self, text=coverage, bg=bg, fg=C.TEXT_2, font=_font(8)).pack(anchor="w", pady=(8, 0))
        progress = tk.Frame(self, bg=C.BORDER_SOFT, height=3)
        progress.pack(fill="x", pady=(7, 0))
        if crop_count:
            ratio = max(0.0, min(1.0, wafers_with_defects / crop_count))
            fill = tk.Frame(progress, bg=C.ORANGE if self.has_defects else C.GREEN, height=3)
            fill.place(relx=0, rely=0, relwidth=max(0.03, ratio) if self.has_defects else 0.03, relheight=1)

        self.bind("<Enter>", lambda _e: self._paint(self._hover_bg))
        self.bind("<Leave>", lambda _e: self._paint(self._base_bg))
        _bind_click_recursive(self, command)

    def _paint(self, bg: str) -> None:
        self.configure(bg=bg)
        for child in self.winfo_children():
            if isinstance(child, (tk.Frame, tk.Label)) and child.cget("bg") in {self._base_bg, self._hover_bg}:
                child.configure(bg=bg)
                for grandchild in child.winfo_children():
                    if isinstance(grandchild, (tk.Frame, tk.Label)) and grandchild.cget("bg") in {self._base_bg, self._hover_bg}:
                        grandchild.configure(bg=bg)


# ---------------------------------------------------------------------------
# Full image viewer
# ---------------------------------------------------------------------------


class FullImageViewer(tk.Toplevel):
    def __init__(self, parent: tk.Tk, entry: WaferDeviceEntry, overlay_opacity: float) -> None:
        super().__init__(parent)
        self.entry = entry
        self.overlay_enabled = tk.BooleanVar(value=True)
        self.opacity = overlay_opacity
        self.scale = 1.0
        self._photo: ImageTk.PhotoImage | None = None
        self._image_id: int | None = None
        self._drag_origin: tuple[int, int] | None = None
        self._original_bgr = _read_bgr(entry.image_path)
        if self._original_bgr is None:
            self.destroy()
            messagebox.showerror(APP_TITLE, f"Could not decode {entry.image_path}")
            return

        self.title(f"{entry.wafer_id} Â· device {entry.device.label}")
        self.configure(bg=C.BG)
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        self.geometry(f"{max(1100, int(screen_w * 0.86))}x{max(760, int(screen_h * 0.84))}")
        self.minsize(900, 640)

        header = tk.Frame(self, bg=C.SURFACE, height=64, padx=16, pady=12)
        header.pack(fill="x")
        tk.Label(
            header,
            text=f"{entry.wafer_id}  Â·  Device {entry.device.label}",
            bg=C.SURFACE,
            fg=C.TEXT,
            font=_font(15, "bold"),
        ).pack(side="left")
        tk.Label(
            header,
            text=f"{len(entry.defects)} defect regions",
            bg=C.ORANGE_DARK if entry.defects else C.GREEN_DARK,
            fg=C.ORANGE if entry.defects else C.GREEN,
            font=_font(9, "bold"),
            padx=10,
            pady=5,
        ).pack(side="left", padx=12)

        HoverButton(header, text="Fit", command=self._fit, padx=12, pady=6).pack(side="right", padx=(6, 0))
        HoverButton(header, text="100%", command=lambda: self._set_scale(1.0), padx=12, pady=6).pack(side="right", padx=6)
        HoverButton(header, text="ï¼‹", command=lambda: self._zoom(1.22), padx=11, pady=6).pack(side="right", padx=3)
        HoverButton(header, text="ï¼", command=lambda: self._zoom(1 / 1.22), padx=11, pady=6).pack(side="right", padx=3)
        TogglePill(header, "Overlay", self.overlay_enabled, self._refresh_image).pack(side="right", padx=10)

        body = tk.Frame(self, bg=C.BG)
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(body, bg=C.IMAGE_BG, highlightthickness=0, bd=0)
        self.canvas.pack(side="left", fill="both", expand=True)

        side = tk.Frame(body, bg=C.SURFACE_2, width=270, padx=18, pady=18)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)
        tk.Label(side, text="IMAGE DETAILS", bg=C.SURFACE_2, fg=C.TEXT_3, font=_font(9, "bold")).pack(anchor="w")
        self._detail_row(side, "Filename", entry.image_path.name)
        self._detail_row(side, "Resolution", f"{entry.image_size[0]} Ã— {entry.image_size[1]} px")
        self._detail_row(side, "Annotations", entry.annotation_source.title())
        self._detail_row(side, "Defects", str(len(entry.defects)))
        if entry.gds_size_um:
            self._detail_row(side, "GDS cell", f"{entry.gds_size_um[0]:.1f} Ã— {entry.gds_size_um[1]:.1f} Âµm")
        tk.Frame(side, bg=C.BORDER_SOFT, height=1).pack(fill="x", pady=18)
        tk.Label(
            side,
            text="Mouse wheel to zoom\nDrag to pan\nEsc to close",
            bg=C.SURFACE_2,
            fg=C.TEXT_3,
            justify="left",
            font=_font(9),
        ).pack(anchor="w")

        self.canvas.bind("<Configure>", lambda _e: self._fit())
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<ButtonPress-1>", self._pan_start)
        self.canvas.bind("<B1-Motion>", self._pan_move)
        self.bind("<Escape>", lambda _e: self.destroy())
        self.after(80, self._fit)

    @staticmethod
    def _detail_row(master: tk.Misc, label: str, value: str) -> None:
        row = tk.Frame(master, bg=C.SURFACE_2)
        row.pack(fill="x", pady=(14, 0))
        tk.Label(row, text=label, bg=C.SURFACE_2, fg=C.TEXT_3, font=_font(8, "bold")).pack(anchor="w")
        tk.Label(row, text=value, bg=C.SURFACE_2, fg=C.TEXT, font=_font(10), wraplength=230, justify="left").pack(anchor="w", pady=(2, 0))

    def _render_bgr(self) -> np.ndarray:
        return draw_defect_overlay(
            self._original_bgr,
            self.entry.defects,
            opacity=self.opacity,
            enabled=self.overlay_enabled.get(),
        )

    def _fit(self) -> None:
        if not self.winfo_exists():
            return
        h, w = self._original_bgr.shape[:2]
        cw = max(100, self.canvas.winfo_width() - 36)
        ch = max(100, self.canvas.winfo_height() - 36)
        self.scale = min(cw / w, ch / h)
        self._refresh_image(center=True)

    def _set_scale(self, scale: float) -> None:
        self.scale = float(np.clip(scale, 0.08, 8.0))
        self._refresh_image(center=True)

    def _zoom(self, factor: float) -> None:
        self.scale = float(np.clip(self.scale * factor, 0.08, 8.0))
        self._refresh_image(center=False)

    def _on_wheel(self, event: tk.Event) -> None:
        self._zoom(1.14 if event.delta > 0 else 1 / 1.14)

    def _refresh_image(self, center: bool = False) -> None:
        if not hasattr(self, "canvas"):
            return
        bgr = self._render_bgr()
        h, w = bgr.shape[:2]
        nw = max(1, int(round(w * self.scale)))
        nh = max(1, int(round(h * self.scale)))
        interp = cv2.INTER_LINEAR if self.scale > 1.0 else cv2.INTER_AREA
        shown = cv2.resize(bgr, (nw, nh), interpolation=interp)
        self._photo = _to_photo(shown)

        if self._image_id is None:
            self._image_id = self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        else:
            self.canvas.itemconfigure(self._image_id, image=self._photo)
        self.canvas.configure(scrollregion=(0, 0, nw, nh))
        if center:
            self.canvas.xview_moveto(max(0.0, (nw - self.canvas.winfo_width()) / max(1, nw) / 2))
            self.canvas.yview_moveto(max(0.0, (nh - self.canvas.winfo_height()) / max(1, nh) / 2))

    def _pan_start(self, event: tk.Event) -> None:
        self.canvas.scan_mark(event.x, event.y)

    def _pan_move(self, event: tk.Event) -> None:
        self.canvas.scan_dragto(event.x, event.y, gain=1)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


class DeviceViewerApp:
    def __init__(
        self,
        root: tk.Tk,
        catalog: Catalog,
        *,
        thumb_size: int,
        only_with_defects: bool,
        logo_path: Path | None,
    ) -> None:
        self.root = root
        self.catalog = catalog
        self.thumb_size = max(260, int(thumb_size))
        self.only_with_defects_var = tk.BooleanVar(value=bool(only_with_defects))
        self.detail_only_defects_var = tk.BooleanVar(value=False)
        self.overlay_enabled_var = tk.BooleanVar(value=True)
        self.overlay_opacity_var = tk.DoubleVar(value=0.18)
        self.search_var = tk.StringVar(value="")
        self.selected_device: DeviceIndex | None = None
        self._photos: list[ImageTk.PhotoImage] = []
        self._logo_photo: ImageTk.PhotoImage | None = None
        self._search_entry: tk.Entry | None = None
        self._detail_entries: list[WaferDeviceEntry] = []
        self._detail_content: tk.Frame | None = None
        self._detail_columns = 0
        self._render_job: str | None = None
        self.logo_path = logo_path

        root.title(APP_TITLE)
        root.geometry("1580x960")
        root.minsize(1040, 700)
        root.configure(bg=C.BG)
        root.protocol("WM_DELETE_WINDOW", root.destroy)
        root.bind("<Escape>", self._on_escape)
        root.bind("q", lambda _event: root.destroy())
        root.bind("Q", lambda _event: root.destroy())
        root.bind("<Left>", lambda _event: self._step_device(-1))
        root.bind("<Right>", lambda _event: self._step_device(1))
        root.bind("/", self._focus_search)
        root.bind("o", lambda _event: self._toggle_overlay())
        root.bind("O", lambda _event: self._toggle_overlay())

        self._configure_styles()
        self._build_shell()
        self.show_grid()

    def _configure_styles(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Vertical.TScrollbar",
            background=C.SURFACE_3,
            troughcolor=C.BG,
            bordercolor=C.BG,
            arrowcolor=C.TEXT_3,
            lightcolor=C.SURFACE_3,
            darkcolor=C.SURFACE_3,
        )

    def _build_shell(self) -> None:
        self.header = tk.Frame(self.root, bg=C.SURFACE, height=86, padx=22, pady=12)
        self.header.pack(fill="x")
        self.header.pack_propagate(False)

        brand = tk.Frame(self.header, bg=C.SURFACE)
        brand.pack(side="left", fill="y")
        self._build_logo(brand)
        title_box = tk.Frame(brand, bg=C.SURFACE)
        title_box.pack(side="left", padx=(14, 0), pady=3)
        tk.Label(title_box, text="H2P DEVICE INSPECTOR", bg=C.SURFACE, fg=C.TEXT, font=_font(17, "bold")).pack(anchor="w")
        tk.Label(title_box, text="Cross-wafer defect intelligence", bg=C.SURFACE, fg=C.TEXT_3, font=_font(9)).pack(anchor="w", pady=(2, 0))

        right = tk.Frame(self.header, bg=C.SURFACE)
        right.pack(side="right", fill="y")
        self._header_chip(right, "ANNOTATIONS", self._annotation_mode_label(), C.BLUE, C.BLUE_DARK).pack(side="right", padx=(8, 0), pady=9)
        self._header_chip(right, "DATA HEALTH", self._data_health_label(), C.GREEN if not self.catalog.warnings else C.ORANGE, C.GREEN_DARK if not self.catalog.warnings else C.ORANGE_DARK).pack(side="right", padx=(8, 0), pady=9)

        self.page = tk.Frame(self.root, bg=C.BG)
        self.page.pack(fill="both", expand=True)

        self.footer = tk.Frame(self.root, bg=C.SURFACE, height=32, padx=18)
        self.footer.pack(fill="x")
        self.footer.pack_propagate(False)
        self.status_label = tk.Label(self.footer, text="", bg=C.SURFACE, fg=C.TEXT_3, font=_font(8))
        self.status_label.pack(side="left", pady=7)
        tk.Label(self.footer, text=f"{APP_VERSION}  Â·  Esc back  Â·  â†/â†’ navigate  Â·  O overlay  Â·  / search", bg=C.SURFACE, fg=C.TEXT_3, font=_font(8)).pack(side="right", pady=7)

    def _build_logo(self, master: tk.Misc) -> None:
        holder = tk.Frame(master, bg=C.WHITE, width=106, height=56, padx=8, pady=6)
        holder.pack(side="left", fill="y")
        holder.pack_propagate(False)
        if self.logo_path and self.logo_path.exists():
            try:
                image = Image.open(self.logo_path).convert("RGBA")
                image.thumbnail((90, 44), Image.Resampling.LANCZOS)
                self._logo_photo = ImageTk.PhotoImage(image)
                tk.Label(holder, image=self._logo_photo, bg=C.WHITE).pack(expand=True)
                return
            except Exception:
                pass
        fallback = tk.Frame(holder, bg=C.WHITE)
        fallback.pack(expand=True)
        tk.Label(fallback, text="H2P", bg=C.WHITE, fg="#102743", font=_font(17, "bold")).pack()
        tk.Label(fallback, text="HEAT â†’ POWER", bg=C.WHITE, fg="#52627A", font=_font(6, "bold")).pack()

    @staticmethod
    def _header_chip(master: tk.Misc, label: str, value: str, fg: str, bg: str) -> tk.Frame:
        frame = tk.Frame(master, bg=bg, padx=11, pady=7)
        tk.Label(frame, text=label, bg=bg, fg=fg, font=_font(7, "bold")).pack(anchor="w")
        tk.Label(frame, text=value, bg=bg, fg=C.TEXT, font=_font(9, "bold")).pack(anchor="w")
        return frame

    def _annotation_mode_label(self) -> str:
        sources = {entry.annotation_source for entries in self.catalog.by_device.values() for entry in entries}
        if sources == {"reviewed"}:
            return "Reviewed"
        if sources == {"automatic"}:
            return "Automatic"
        if not sources:
            return "None"
        return "Mixed"

    def _data_health_label(self) -> str:
        return "Ready" if not self.catalog.warnings else f"{len(self.catalog.warnings)} warnings"

    def _clear_page(self) -> None:
        self._photos.clear()
        if self._render_job:
            try:
                self.root.after_cancel(self._render_job)
            except Exception:
                pass
            self._render_job = None
        for child in self.page.winfo_children():
            child.destroy()

    def _set_status(self, text: str) -> None:
        self.status_label.configure(text=text)

    def _visible_devices(self) -> list[DeviceIndex]:
        devices = self.catalog.devices
        if self.only_with_defects_var.get():
            devices = [
                device
                for device in devices
                if any(entry.defects for entry in self.catalog.by_device.get(device, []))
            ]
        query = self.search_var.get().strip().casefold().replace("device", "").strip()
        if query:
            digits = re.findall(r"\d+", query)
            if "-" in query and len(digits) >= 2:
                row, col = int(digits[0]), int(digits[1])
                devices = [d for d in devices if d.row == row and d.col == col]
            elif digits:
                token = digits[0]
                devices = [d for d in devices if token in d.label or int(token) in {d.row, d.col}]
            else:
                devices = [d for d in devices if query in d.label.casefold()]
        return devices

    def show_grid(self) -> None:
        self.selected_device = None
        self._clear_page()
        self._set_status("Device map ready")

        outer = tk.Frame(self.page, bg=C.BG, padx=24, pady=20)
        outer.pack(fill="both", expand=True)

        hero = tk.Frame(outer, bg=C.BG)
        hero.pack(fill="x")
        title_box = tk.Frame(hero, bg=C.BG)
        title_box.pack(side="left")
        tk.Label(title_box, text="Device map", bg=C.BG, fg=C.TEXT, font=_font(24, "bold")).pack(anchor="w")
        tk.Label(
            title_box,
            text="Select a device index to compare the same physical location across every wafer.",
            bg=C.BG,
            fg=C.TEXT_2,
            font=_font(10),
        ).pack(anchor="w", pady=(4, 0))

        controls = tk.Frame(hero, bg=C.BG)
        controls.pack(side="right", pady=(4, 0))
        search_shell = tk.Frame(controls, bg=C.SURFACE_2, highlightthickness=1, highlightbackground=C.BORDER, padx=10, pady=6)
        search_shell.pack(side="left", padx=(0, 10))
        tk.Label(search_shell, text="âŒ•", bg=C.SURFACE_2, fg=C.TEXT_3, font=_font(15)).pack(side="left", padx=(0, 7))
        self._search_entry = tk.Entry(
            search_shell,
            textvariable=self.search_var,
            bg=C.SURFACE_2,
            fg=C.TEXT,
            insertbackground=C.ACCENT,
            relief="flat",
            bd=0,
            width=18,
            font=_font(10),
        )
        self._search_entry.pack(side="left")
        self._search_entry.bind("<KeyRelease>", self._schedule_grid_refresh)
        TogglePill(controls, "Defects only", self.only_with_defects_var, self.show_grid).pack(side="left")

        entries = [entry for values in self.catalog.by_device.values() for entry in values]
        total_defects = sum(len(entry.defects) for entry in entries)
        flagged_devices = sum(any(entry.defects for entry in self.catalog.by_device[d]) for d in self.catalog.devices)
        stats = tk.Frame(outer, bg=C.BG)
        stats.pack(fill="x", pady=(18, 18))
        stat_data = [
            ("Wafers", str(len(self.catalog.wafer_ids)), C.ACCENT),
            ("Device indices", str(len(self.catalog.devices)), C.BLUE),
            ("Flagged devices", str(flagged_devices), C.ORANGE),
            ("Defect regions", f"{total_defects:,}", C.RED),
        ]
        for col, (label, value, accent) in enumerate(stat_data):
            stats.grid_columnconfigure(col, weight=1, uniform="stats")
            card = StatCard(stats, label, value, accent)
            card.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else 6, 0 if col == len(stat_data) - 1 else 6))

        map_shell = tk.Frame(outer, bg=C.SURFACE, highlightthickness=1, highlightbackground=C.BORDER_SOFT)
        map_shell.pack(fill="both", expand=True)
        map_header = tk.Frame(map_shell, bg=C.SURFACE, padx=18, pady=13)
        map_header.pack(fill="x")
        tk.Label(map_header, text="WAFER DEVICE LAYOUT", bg=C.SURFACE, fg=C.TEXT_2, font=_font(9, "bold")).pack(side="left")
        legend = tk.Frame(map_header, bg=C.SURFACE)
        legend.pack(side="right")
        self._legend_item(legend, C.GREEN, "No defects").pack(side="left", padx=(0, 14))
        self._legend_item(legend, C.ORANGE, "Defects present").pack(side="left")

        scroll = ScrollableFrame(map_shell, bg=C.SURFACE)
        scroll.pack(fill="both", expand=True, padx=2, pady=(0, 2))
        grid_frame = scroll.inner
        grid_frame.configure(bg=C.SURFACE, padx=18, pady=18)

        devices = self._visible_devices()
        if not devices:
            empty = tk.Frame(grid_frame, bg=C.SURFACE, pady=80)
            empty.pack(fill="both", expand=True)
            tk.Label(empty, text="No device indices match this filter", bg=C.SURFACE, fg=C.TEXT, font=_font(16, "bold")).pack()
            tk.Label(empty, text="Try clearing the search or showing all devices.", bg=C.SURFACE, fg=C.TEXT_3, font=_font(10)).pack(pady=(5, 0))
            self._set_status("No matching devices")
            return

        by_row: dict[int, list[DeviceIndex]] = defaultdict(list)
        for device in devices:
            by_row[device.row].append(device)
        max_cols = max(len(row_devices) for row_devices in by_row.values())

        for visual_row, row_num in enumerate(sorted(by_row)):
            row_devices = sorted(by_row[row_num], key=lambda d: d.col)
            offset = max(0, (max_cols - len(row_devices)) // 2)
            tk.Label(grid_frame, text=f"R{row_num:02d}", bg=C.SURFACE, fg=C.TEXT_3, font=_font(8, "bold"), width=5).grid(
                row=visual_row, column=0, padx=(0, 12), pady=7
            )
            for device in row_devices:
                entries_for_device = self.catalog.by_device.get(device, [])
                crop_count = len(entries_for_device)
                defect_count = sum(len(entry.defects) for entry in entries_for_device)
                wafers_with_defects = sum(bool(entry.defects) for entry in entries_for_device)
                tile = DeviceTile(
                    grid_frame,
                    device=device,
                    crop_count=crop_count,
                    wafers_with_defects=wafers_with_defects,
                    defect_count=defect_count,
                    command=lambda d=device: self.show_device(d),
                )
                tile.grid(
                    row=visual_row,
                    column=1 + offset + device.col - 1,
                    padx=6,
                    pady=6,
                    sticky="nsew",
                )

        self._set_status(f"Showing {len(devices)} of {len(self.catalog.devices)} device indices")

    @staticmethod
    def _legend_item(master: tk.Misc, color: str, text: str) -> tk.Frame:
        frame = tk.Frame(master, bg=C.SURFACE)
        tk.Label(frame, text="â—", bg=C.SURFACE, fg=color, font=_font(10)).pack(side="left")
        tk.Label(frame, text=text, bg=C.SURFACE, fg=C.TEXT_3, font=_font(8)).pack(side="left", padx=(4, 0))
        return frame

    def _schedule_grid_refresh(self, _event: tk.Event | None = None) -> None:
        if self._render_job:
            try:
                self.root.after_cancel(self._render_job)
            except Exception:
                pass
        self._render_job = self.root.after(140, self.show_grid)

    def _focus_search(self, _event: tk.Event | None = None) -> str:
        if self.selected_device is None and self._search_entry:
            self._search_entry.focus_set()
            self._search_entry.select_range(0, "end")
        return "break"

    def show_device(self, device: DeviceIndex) -> None:
        self.selected_device = device
        self._clear_page()
        self._set_status(f"Device {device.label}")

        outer = tk.Frame(self.page, bg=C.BG, padx=24, pady=18)
        outer.pack(fill="both", expand=True)

        nav = tk.Frame(outer, bg=C.BG)
        nav.pack(fill="x")
        HoverButton(nav, text="â†  Device map", command=self.show_grid, bg=C.SURFACE_2, hover_bg=C.SURFACE_3).pack(side="left")
        HoverButton(nav, text="â€¹", command=lambda: self._step_device(-1), padx=13).pack(side="left", padx=(10, 4))
        HoverButton(nav, text="â€º", command=lambda: self._step_device(1), padx=13).pack(side="left", padx=4)

        title_box = tk.Frame(nav, bg=C.BG)
        title_box.pack(side="left", padx=18)
        tk.Label(title_box, text=f"Device {device.label}", bg=C.BG, fg=C.TEXT, font=_font(23, "bold")).pack(anchor="w")
        tk.Label(title_box, text="Cross-wafer comparison", bg=C.BG, fg=C.TEXT_3, font=_font(9)).pack(anchor="w")

        tool = tk.Frame(nav, bg=C.BG)
        tool.pack(side="right")
        TogglePill(tool, "Overlay", self.overlay_enabled_var, lambda: self._render_detail_cards(force=True)).pack(side="right", padx=(8, 0))
        TogglePill(tool, "Defects only", self.detail_only_defects_var, lambda: self.show_device(device)).pack(side="right", padx=(8, 0))

        entries = list(self.catalog.by_device.get(device, []))
        if self.detail_only_defects_var.get():
            entries = [entry for entry in entries if entry.defects]
        self._detail_entries = entries
        total_defects = sum(len(entry.defects) for entry in entries)
        flagged_wafers = sum(bool(entry.defects) for entry in entries)

        stats = tk.Frame(outer, bg=C.BG)
        stats.pack(fill="x", pady=(16, 14))
        detail_stats = [
            ("Crops shown", str(len(entries)), C.ACCENT),
            ("Flagged wafers", str(flagged_wafers), C.ORANGE),
            ("Defect regions", str(total_defects), C.RED),
            ("Coverage", f"{len(entries)}/{len(self.catalog.wafer_ids)}", C.BLUE),
        ]
        for col, (label, value, accent) in enumerate(detail_stats):
            stats.grid_columnconfigure(col, weight=1, uniform="detailstats")
            StatCard(stats, label, value, accent).grid(
                row=0,
                column=col,
                sticky="ew",
                padx=(0 if col == 0 else 6, 0 if col == len(detail_stats) - 1 else 6),
            )

        scroll_shell = tk.Frame(outer, bg=C.SURFACE, highlightthickness=1, highlightbackground=C.BORDER_SOFT)
        scroll_shell.pack(fill="both", expand=True)
        scroll = ScrollableFrame(scroll_shell, bg=C.SURFACE)
        scroll.pack(fill="both", expand=True, padx=2, pady=2)
        self._detail_content = scroll.inner
        self._detail_content.configure(bg=C.SURFACE, padx=10, pady=10)
        scroll.canvas.bind("<Configure>", lambda e: self._schedule_detail_render(e.width))

        if not entries:
            empty = tk.Frame(self._detail_content, bg=C.SURFACE, pady=100)
            empty.pack(fill="both", expand=True)
            tk.Label(empty, text="No wafer crops match this filter", bg=C.SURFACE, fg=C.TEXT, font=_font(16, "bold")).pack()
            tk.Label(empty, text="Turn off â€˜Defects onlyâ€™ to show clean wafers.", bg=C.SURFACE, fg=C.TEXT_3, font=_font(10)).pack(pady=(5, 0))
            return

        self.root.update_idletasks()
        self._render_detail_cards(force=True)

    def _detail_column_count(self, width: int) -> int:
        card_width = self.thumb_size + 58
        return max(1, min(4, int(max(1, width - 18) // card_width)))

    def _schedule_detail_render(self, width: int) -> None:
        columns = self._detail_column_count(width)
        if columns == self._detail_columns:
            return
        if self._render_job:
            try:
                self.root.after_cancel(self._render_job)
            except Exception:
                pass
        self._render_job = self.root.after(120, lambda: self._render_detail_cards(force=True))

    def _render_detail_cards(self, *, force: bool = False) -> None:
        content = self._detail_content
        if content is None or not content.winfo_exists():
            return
        width = max(1, content.winfo_width())
        columns = self._detail_column_count(width)
        if not force and columns == self._detail_columns:
            return
        self._detail_columns = columns
        self._photos.clear()
        for child in content.winfo_children():
            child.destroy()
        for col in range(columns):
            content.grid_columnconfigure(col, weight=1, uniform="cards")

        for index, entry in enumerate(self._detail_entries):
            row, col = divmod(index, columns)
            card = self._create_wafer_card(content, entry)
            card.grid(row=row, column=col, sticky="nsew", padx=8, pady=8)
        self._set_status(f"Device {self.selected_device.label if self.selected_device else ''}: {len(self._detail_entries)} crops")

    def _create_wafer_card(self, master: tk.Misc, entry: WaferDeviceEntry) -> tk.Frame:
        card = tk.Frame(master, bg=C.CARD, highlightthickness=1, highlightbackground=C.BORDER, padx=0, pady=0)
        header = tk.Frame(card, bg=C.CARD, padx=14, pady=12)
        header.pack(fill="x")
        tk.Label(header, text=entry.wafer_id, bg=C.CARD, fg=C.TEXT, font=_font(13, "bold")).pack(side="left")
        if entry.defects:
            badge_text = f"{len(entry.defects)} DEFECTS"
            badge_bg, badge_fg = C.ORANGE_DARK, C.ORANGE
        else:
            badge_text = "CLEAN"
            badge_bg, badge_fg = C.GREEN_DARK, C.GREEN
        tk.Label(header, text=badge_text, bg=badge_bg, fg=badge_fg, font=_font(8, "bold"), padx=9, pady=4).pack(side="right")

        meta = tk.Frame(card, bg=C.CARD, padx=14)
        meta.pack(fill="x")
        w, h = entry.image_size
        gds = ""
        if entry.gds_size_um:
            gds = f"  Â·  {entry.gds_size_um[0]:.0f}Ã—{entry.gds_size_um[1]:.0f} Âµm"
        tk.Label(
            meta,
            text=f"{w}Ã—{h} px{gds}  Â·  {entry.annotation_source.title()}",
            bg=C.CARD,
            fg=C.TEXT_3,
            font=_font(8),
        ).pack(anchor="w", pady=(0, 10))

        image_shell = tk.Frame(card, bg=C.IMAGE_BG, padx=1, pady=1)
        image_shell.pack(fill="both", expand=True, padx=12)
        bgr = _read_bgr(entry.image_path)
        if bgr is None:
            tk.Label(image_shell, text="Image decode failed", bg=C.IMAGE_BG, fg=C.RED, font=_font(10, "bold")).pack(pady=80)
        else:
            overlay = draw_defect_overlay(
                bgr,
                entry.defects,
                opacity=self.overlay_opacity_var.get(),
                enabled=self.overlay_enabled_var.get(),
            )
            thumb = _fit_image(overlay, self.thumb_size, self.thumb_size)
            photo = _to_photo(thumb)
            self._photos.append(photo)
            image_label = tk.Label(image_shell, image=photo, bg=C.IMAGE_BG, cursor="hand2")
            image_label.pack(expand=True)
            image_label.bind("<Button-1>", lambda _e, item=entry: FullImageViewer(self.root, item, self.overlay_opacity_var.get()))

        footer = tk.Frame(card, bg=C.CARD, padx=14, pady=11)
        footer.pack(fill="x")
        tk.Label(footer, text=entry.image_path.name, bg=C.CARD, fg=C.TEXT_3, font=_font(8)).pack(side="left")
        open_label = tk.Label(footer, text="OPEN  â†—", bg=C.CARD, fg=C.ACCENT, font=_font(8, "bold"), cursor="hand2")
        open_label.pack(side="right")
        open_label.bind("<Button-1>", lambda _e, item=entry: FullImageViewer(self.root, item, self.overlay_opacity_var.get()))

        def enter(_event: tk.Event) -> None:
            card.configure(highlightbackground=C.ACCENT)

        def leave(_event: tk.Event) -> None:
            card.configure(highlightbackground=C.BORDER)

        card.bind("<Enter>", enter)
        card.bind("<Leave>", leave)
        return card

    def _toggle_overlay(self) -> None:
        if self.selected_device is None:
            return
        self.overlay_enabled_var.set(not self.overlay_enabled_var.get())
        self._render_detail_cards(force=True)

    def _step_device(self, direction: int) -> None:
        if self.selected_device is None:
            return
        devices = self.catalog.devices
        if not devices:
            return
        try:
            index = devices.index(self.selected_device)
        except ValueError:
            index = 0
        self.show_device(devices[(index + direction) % len(devices)])

    def _on_escape(self, _event: tk.Event) -> None:
        if self.selected_device is not None:
            self.show_grid()
        else:
            self.root.destroy()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def print_summary(catalog: Catalog) -> None:
    entries = [entry for items in catalog.by_device.values() for entry in items]
    sizes = Counter(entry.image_size for entry in entries)
    square_count = sum(w == h for w, h in (entry.image_size for entry in entries))
    print(f"Wafers discovered: {len(catalog.wafer_ids)}")
    print(f"Device indices discovered: {len(catalog.by_device)}")
    print(f"Cell crops discovered: {len(entries)}")
    print(f"Square crops: {square_count}/{len(entries)}")
    print("Crop resolutions:")
    for (w, h), count in sizes.most_common():
        print(f"  {w} x {h}: {count}")
    gds_sizes = Counter(
        (round(entry.gds_size_um[0], 3), round(entry.gds_size_um[1], 3))
        for entry in entries
        if entry.gds_size_um is not None
    )
    if gds_sizes:
        gds_square = sum(count for (w, h), count in gds_sizes.items() if abs(w - h) <= 1e-6)
        print(f"GDS-square metadata records: {gds_square}/{sum(gds_sizes.values())}")
        print("GDS cell dimensions:")
        for (w, h), count in gds_sizes.most_common():
            print(f"  {w:g} x {h:g} um: {count}")
    print(f"Defect regions: {sum(len(entry.defects) for entry in entries)}")
    if catalog.warnings:
        print("Warnings:")
        for warning in catalog.warnings:
            print(f"  - {warning}")


def _default_logo_path() -> Path:
    return Path(__file__).resolve().parent / "assets" / DEFAULT_LOGO_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare one device index across all H2P wafer crops with defect overlays."
    )
    parser.add_argument("--extracted-root", default="extracted_cells", help="Per-wafer extraction root. Default: extracted_cells")
    parser.add_argument("--batch", default="batch_wafers.txt", help="Wafer batch file. Default: batch_wafers.txt")
    parser.add_argument(
        "--source",
        choices=("prefer-reviewed", "reviewed", "auto"),
        default="prefer-reviewed",
        help="Annotation source. Default: reviewed JSON when available, otherwise algo_defects.json",
    )
    parser.add_argument("--thumb-size", type=int, default=440, help="Maximum thumbnail side in pixels. Default: 440")
    parser.add_argument("--only-with-defects", action="store_true", help="Start with the device grid filtered to indices with defects")
    parser.add_argument("--summary", action="store_true", help="Print crop resolutions/catalog information and exit without opening the UI")
    parser.add_argument(
        "--logo",
        default=str(_default_logo_path()),
        help=f"H2P logo path. Default: ./assets/{DEFAULT_LOGO_NAME}",
    )
    return parser.parse_args()


def main() -> int:
    _enable_windows_dpi_awareness()
    args = parse_args()
    extracted_root = Path(args.extracted_root)
    batch_path = Path(args.batch)
    if not extracted_root.exists():
        print(f"ERROR: extracted root not found: {extracted_root}", file=sys.stderr)
        return 2

    catalog = build_catalog(extracted_root, batch_path, args.source)
    if args.summary:
        print_summary(catalog)
        return 0
    if not catalog.by_device:
        print_summary(catalog)
        print("ERROR: no device crops were found.", file=sys.stderr)
        return 2

    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", max(1.0, float(root.winfo_fpixels("1i")) / 96.0))
    except Exception:
        pass
    logo_path = Path(args.logo).expanduser() if args.logo else None
    DeviceViewerApp(
        root,
        catalog,
        thumb_size=args.thumb_size,
        only_with_defects=args.only_with_defects,
        logo_path=logo_path,
    )
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


