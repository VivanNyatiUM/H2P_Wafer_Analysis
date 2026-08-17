"""Repository-wide H2P UI branding.

UI modules import this shared hook before creating their windows.  It gives Tk
windows an H2P icon and a compact visible
logo badge, and it overlays the same logo on images passed to ``cv2.imshow``.
The OpenCV overlay is applied to a copy used for display only; source arrays and
saved image files are never changed.
"""

from __future__ import annotations

import os
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable


_DISABLED = os.environ.get("H2P_DISABLE_UI_BRANDING", "").strip().casefold() in {
    "1",
    "true",
    "yes",
    "on",
}
_INSTALLED = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _logo_path() -> Path:
    return _repo_root() / "assets" / "h2pLogo.png"


def _is_calibration_window(window: Any) -> bool:
    try:
        return "gds alignment calibration" in str(window.title()).casefold()
    except Exception:
        return False


def _tk_badge_place_options(window: Any) -> dict[str, Any]:
    """Return stable in-client badge placement for Tk windows."""
    if _is_calibration_window(window):
        # Keep the badge owned by the calibration window so it moves/minimizes
        # with the UI.  y=0 places it a little higher than the previous v5
        # placement without creating a detached borderless Toplevel.
        return {"relx": 0.5, "x": 0, "y": 0, "anchor": "n"}
    return {"relx": 1.0, "x": -12, "y": 12, "anchor": "ne"}


def _brand_tk_window(window: Any) -> None:
    if _DISABLED or getattr(window, "_h2p_branding_scheduled", False):
        return
    window._h2p_branding_scheduled = True

    def apply() -> None:
        try:
            if not window.winfo_exists():
                return
            logo = _logo_path()
            if not logo.exists():
                return

            from PIL import Image, ImageTk
            import tkinter as tk

            original = Image.open(logo).convert("RGBA")
            icon_image = original.copy()
            icon_image.thumbnail((64, 64), Image.Resampling.LANCZOS)
            icon_photo = ImageTk.PhotoImage(icon_image, master=window)
            window.iconphoto(True, icon_photo)

            images = getattr(window, "_h2p_branding_images", [])
            images.append(icon_photo)
            window._h2p_branding_images = images

            # A UI that already renders its own H2P logo can opt out of the
            # badge while still receiving the native window icon.
            if getattr(window, "_h2p_embedded_branding", False):
                return

            badge_image = original.copy()
            badge_image.thumbnail((86, 42), Image.Resampling.LANCZOS)
            badge_photo = ImageTk.PhotoImage(badge_image, master=window)
            images.append(badge_photo)

            badge = tk.Frame(
                window,
                bg="#FFFFFF",
                highlightthickness=1,
                highlightbackground="#D7DEE8",
                padx=7,
                pady=5,
            )
            badge._h2p_branding_badge = True
            label = tk.Label(badge, image=badge_photo, bg="#FFFFFF", bd=0)
            label.pack()
            badge.place(**_tk_badge_place_options(window))
            badge.lift()
            window._h2p_branding_badge_widget = badge
        except Exception:
            # Branding must never prevent a scientific workflow from opening.
            return

    try:
        window.after(50, apply)
    except Exception:
        pass


def _patch_tkinter() -> None:
    try:
        import tkinter as tk
    except Exception:
        return

    for cls in (tk.Tk, tk.Toplevel):
        if getattr(cls, "_h2p_branding_patched", False):
            continue
        original_init = cls.__init__

        @wraps(original_init)
        def branded_init(self: Any, *args: Any, __original: Callable[..., Any] = original_init, **kwargs: Any) -> None:
            __original(self, *args, **kwargs)
            _brand_tk_window(self)

        cls.__init__ = branded_init  # type: ignore[method-assign]
        cls._h2p_branding_patched = True


@lru_cache(maxsize=32)
def _opencv_logo_for_height(target_height: int) -> Any:
    import cv2

    source = cv2.imread(str(_logo_path()), cv2.IMREAD_UNCHANGED)
    if source is None or source.ndim not in (2, 3):
        return None

    height, width = source.shape[:2]
    if height <= 0 or width <= 0:
        return None
    scale = target_height / float(height)
    target_width = max(1, int(round(width * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(source, (target_width, target_height), interpolation=interpolation)


def _overlay_opencv_logo(image: Any) -> Any:
    try:
        import cv2
        import numpy as np

        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            return image
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            return image

        height, width = image.shape[:2]
        if height < 120 or width < 180:
            return image

        target_height = max(28, min(64, int(round(height * 0.075))))
        logo = _opencv_logo_for_height(target_height)
        if logo is None:
            return image

        logo_h, logo_w = logo.shape[:2]
        margin = max(8, int(round(target_height * 0.22)))
        if logo_h + 2 * margin >= height or logo_w + 2 * margin >= width:
            return image

        shown = image.copy()
        y0 = margin
        x0 = width - logo_w - margin
        roi = shown[y0 : y0 + logo_h, x0 : x0 + logo_w]

        if logo.ndim == 3 and logo.shape[2] == 4:
            logo_rgb = logo[:, :, :3]
            alpha = logo[:, :, 3:4].astype(np.float32) / 255.0
        else:
            logo_rgb = cv2.cvtColor(logo, cv2.COLOR_GRAY2BGR) if logo.ndim == 2 else logo[:, :, :3]
            alpha = np.full((logo_h, logo_w, 1), 0.92, dtype=np.float32)

        if roi.shape[2] == 4:
            roi_rgb = roi[:, :, :3]
            blended = logo_rgb.astype(np.float32) * alpha + roi_rgb.astype(np.float32) * (1.0 - alpha)
            roi[:, :, :3] = np.clip(blended, 0, 255).astype(np.uint8)
        else:
            blended = logo_rgb.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)
            roi[:] = np.clip(blended, 0, 255).astype(np.uint8)
        return shown
    except Exception:
        return image


def _patch_opencv() -> None:
    try:
        import cv2
    except Exception:
        return

    if getattr(cv2, "_h2p_branding_patched", False):
        return
    original_imshow = cv2.imshow

    @wraps(original_imshow)
    def branded_imshow(window_name: str, image: Any) -> Any:
        return original_imshow(window_name, _overlay_opencv_logo(image))

    cv2.imshow = branded_imshow
    cv2._h2p_branding_patched = True


def install_global_branding() -> None:
    """Install idempotent Tk and OpenCV display branding hooks."""
    global _INSTALLED
    if _DISABLED or _INSTALLED:
        return
    _INSTALLED = True
    _patch_tkinter()
    _patch_opencv()
