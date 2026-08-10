
from h2p_ui_branding import install_global_branding as _install_h2p_ui_branding
_install_h2p_ui_branding()

import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class ManualAlignApp:
    def __init__(self, root, ds_image, xc_ds, yc_ds, R_ds, ds_factor, gds_polygons, gds_R, initial_angle_rad, map_mode=False, gds_center=(0.0, 0.0), shear=0.0, markers=None, initial_tx=0.0, initial_ty=0.0, initial_scale=1.0):
        self.root = root
        self.ds_image = ds_image
        self.xc_ds = xc_ds
        self.yc_ds = yc_ds
        self.R_ds = R_ds
        self.ds_factor = ds_factor
        self.gds_polygons = gds_polygons
        self.Rg = gds_R
        self.map_mode = map_mode
        self.xg_c = float(gds_center[0])
        self.yg_c = float(gds_center[1])
        self.shear = float(shear)
        self.markers = markers if markers is not None else {"left": [], "right": []}

        self.initial_angle_rad = initial_angle_rad
        self.initial_angle_deg = initial_angle_rad * 180.0 / np.pi
        self.current_angle_deg = self.initial_angle_deg
        self.current_angle_rad = self.initial_angle_rad

        self.offset_x = float(initial_tx)
        self.offset_y = float(initial_ty)
        self.scale_mult = float(initial_scale)

        self.V_w, self.V_h = 800, 600
        self.zoom, self.pan_x, self.pan_y = 1.0, 0.0, 0.0
        self.is_dragging = False

        self.show_gds = True
        self.setup_ui()
        self.setup_bindings()
        self.render()

    def setup_ui(self):
        self.root._h2p_embedded_branding = True
        self.root._h2p_alignment_header_v1 = True
        self.root.title("Interactive GDS Alignment Calibration Workspace")
        self.root.geometry("1420x724")
        self.root.resizable(False, False)

        style = ttk.Style()
        style.theme_use("clam")

        # Dedicated logo header: it consumes its own row and never overlays the wafer.
        header = tk.Frame(self.root, height=44, bg="#FFFFFF")
        header.grid(row=0, column=0, columnspan=3, sticky="ew")
        header.grid_propagate(False)
        try:
            from pathlib import Path as _Path
            logo_path = _Path(__file__).resolve().parent / "assets" / "h2pLogo.png"
            logo_image = Image.open(logo_path).convert("RGBA")
            logo_image.thumbnail((76, 36), Image.Resampling.LANCZOS)
            self._h2p_alignment_logo = ImageTk.PhotoImage(logo_image, master=self.root)
            tk.Label(header, image=self._h2p_alignment_logo, bg="#FFFFFF", bd=0).pack(pady=(3, 0))
        except Exception:
            pass

        self.sidebar = ttk.Frame(self.root, padding=15, width=340)
        self.sidebar.grid(row=1, column=0, sticky="ns", padx=5, pady=5)
        self.sidebar.grid_propagate(False)

        self.canvas = tk.Canvas(self.root, width=self.V_w, height=self.V_h, bg="#1e1e1e", highlightthickness=0)
        self.canvas.grid(row=1, column=1, padx=5, pady=10)

        self.marker_panel = ttk.Frame(self.root, padding=10, width=250)
        self.marker_panel.grid(row=1, column=2, sticky="ns", padx=5, pady=5)
        self.marker_panel.grid_propagate(False)

        ttk.Label(self.marker_panel, text="Marker Alignment Assistant", font=("Helvetica", 11, "bold")).pack(anchor="n", pady=(0, 10))

        ttk.Label(self.marker_panel, text="Left Marker (GDS Layer 4)", font=("Helvetica", 9, "bold")).pack(anchor="w", pady=(5, 2))
        self.left_zoom_canvas = tk.Canvas(self.marker_panel, width=220, height=220, bg="#151515", highlightthickness=1, highlightbackground="#444")
        self.left_zoom_canvas.pack(anchor="w", pady=(0, 15))

        ttk.Label(self.marker_panel, text="Right Marker (GDS Layer 4)", font=("Helvetica", 9, "bold")).pack(anchor="w", pady=(5, 2))
        self.right_zoom_canvas = tk.Canvas(self.marker_panel, width=220, height=220, bg="#151515", highlightthickness=1, highlightbackground="#444")
        self.right_zoom_canvas.pack(anchor="w", pady=(0, 15))

        # GUI controls setup
        ttk.Label(self.sidebar, text="1. Rotation (Degrees)", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 2))
        rot_frame = ttk.Frame(self.sidebar)
        rot_frame.pack(fill="x", pady=(0, 10))
        self.rot_var = tk.StringVar(value=f"{self.current_angle_deg:.3f}")
        self.rot_entry = ttk.Entry(rot_frame, textvariable=self.rot_var, width=12)
        self.rot_entry.pack(side="left", padx=(0, 5))
        ttk.Button(rot_frame, text="Apply", command=self.apply_rotation).pack(side="left")

        rot_btn_frame = ttk.Frame(self.sidebar)
        rot_btn_frame.pack(fill="x", pady=(0, 15))
        for idx, (txt, val) in enumerate([("-0.10°", -0.1), ("-0.01°", -0.01), ("+0.01°", 0.01), ("+0.10°", 0.1)]):
            ttk.Button(rot_btn_frame, text=txt, width=7, command=lambda v=val: self.adjust_angle(v)).grid(row=0, column=idx, padx=2)

        ttk.Label(self.sidebar, text="2. Translation X (microns)", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 2))
        tx_frame = ttk.Frame(self.sidebar)
        tx_frame.pack(fill="x", pady=(0, 10))
        self.tx_var = tk.StringVar(value=f"{self.offset_x:.1f}")
        self.tx_entry = ttk.Entry(tx_frame, textvariable=self.tx_var, width=12)
        self.tx_entry.pack(side="left", padx=(0, 5))
        ttk.Button(tx_frame, text="Apply", command=self.apply_rotation).pack(side="left")

        tx_btn_frame = ttk.Frame(self.sidebar)
        tx_btn_frame.pack(fill="x", pady=(0, 15))
        for idx, (txt, val) in enumerate([("-50µm", -50.0), ("-5µm", -5.0), ("+5µm", 5.0), ("+50µm", 50.0)]):
            ttk.Button(tx_btn_frame, text=txt, width=7, command=lambda v=val: self.adjust_tx(v)).grid(row=0, column=idx, padx=2)

        ttk.Label(self.sidebar, text="3. Translation Y (microns)", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 2))
        ty_frame = ttk.Frame(self.sidebar)
        ty_frame.pack(fill="x", pady=(0, 10))
        self.ty_var = tk.StringVar(value=f"{self.offset_y:.1f}")
        self.ty_entry = ttk.Entry(ty_frame, textvariable=self.ty_var, width=12)
        self.ty_entry.pack(side="left", padx=(0, 5))
        ttk.Button(ty_frame, text="Apply", command=self.apply_rotation).pack(side="left")

        ty_btn_frame = ttk.Frame(self.sidebar)
        ty_btn_frame.pack(fill="x", pady=(0, 15))
        for idx, (txt, val) in enumerate([("-50µm", -50.0), ("-5µm", -5.0), ("+5µm", 5.0), ("+50µm", 50.0)]):
            ttk.Button(ty_btn_frame, text=txt, width=7, command=lambda v=val: self.adjust_ty(v)).grid(row=0, column=idx, padx=2)

        ttk.Label(self.sidebar, text="4. Scale Factor Multiplier", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 2))
        sc_frame = ttk.Frame(self.sidebar)
        sc_frame.pack(fill="x", pady=(0, 10))
        self.scale_var = tk.StringVar(value=f"{self.scale_mult:.5f}")
        self.scale_entry = ttk.Entry(sc_frame, textvariable=self.scale_var, width=12)
        self.scale_entry.pack(side="left", padx=(0, 5))
        ttk.Button(sc_frame, text="Apply", command=self.apply_rotation).pack(side="left")

        sc_btn_frame = ttk.Frame(self.sidebar)
        sc_btn_frame.pack(fill="x", pady=(0, 20))
        for idx, (txt, val) in enumerate([("-0.50%", -0.005), ("-0.05%", -0.0005), ("+0.05%", 0.0005), ("+0.50%", 0.005)]):
            ttk.Button(sc_btn_frame, text=txt, width=7, command=lambda v=val: self.adjust_scale(v)).grid(row=0, column=idx, padx=2)

        self.status_label = ttk.Label(self.sidebar, text="", font=("Helvetica", 9, "bold"), justify="left")
        self.status_label.pack(anchor="w", pady=(0, 25))

        ttk.Button(self.sidebar, text="✔ Confirm Settings", command=self.confirm).pack(fill="x", pady=4)
        ttk.Button(self.sidebar, text="✖ Cancel (No Calibration)", command=self.cancel).pack(fill="x", pady=4)

    def setup_bindings(self):
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, 'is_dragging', False))
        self.canvas.bind("<MouseWheel>", self.on_scroll)
        self.canvas.bind("<Button-4>", lambda e: self.adjust_zoom(1.15))
        self.canvas.bind("<Button-5>", lambda e: self.adjust_zoom(1.0 / 1.15))

        for entry in [self.rot_entry, self.tx_entry, self.ty_entry, self.scale_entry]:
            entry.bind("<Return>", lambda e: self.apply_rotation())

        self.root.bind("<Control-equal>", lambda e: self.adjust_zoom(1.15))
        self.root.bind("<Control-plus>", lambda e: self.adjust_zoom(1.15))
        self.root.bind("<Control-minus>", lambda e: self.adjust_zoom(1.0 / 1.15))
        self.root.bind("<KeyPress-r>", lambda e: self.reset_viewport())

    def apply_rotation(self):
        try:
            self.current_angle_deg = float(self.rot_var.get())
            self.current_angle_rad = self.current_angle_deg * np.pi / 180.0
            self.offset_x = float(self.tx_var.get())
            self.offset_y = float(self.ty_var.get())
            self.scale_mult = float(self.scale_var.get())
            self.render()
        except ValueError:
            pass

    def adjust_angle(self, offset):
        self.current_angle_deg = np.clip(self.current_angle_deg + offset, -45.0, 45.0)
        self.current_angle_rad = self.current_angle_deg * np.pi / 180.0
        self.rot_var.set(f"{self.current_angle_deg:.3f}")
        self.render()

    def adjust_tx(self, offset):
        self.offset_x += offset
        self.tx_var.set(f"{self.offset_x:.1f}")
        self.render()

    def adjust_ty(self, offset):
        self.offset_y += offset
        self.ty_var.set(f"{self.offset_y:.1f}")
        self.render()

    def adjust_scale(self, offset):
        self.scale_mult = np.clip(self.scale_mult + offset, 0.5, 2.0)
        self.scale_var.set(f"{self.scale_mult:.5f}")
        self.render()

    def on_drag_start(self, event):
        self.is_dragging = True
        self.drag_start_x, self.drag_start_y = event.x, event.y
        self.orig_pan_x, self.orig_pan_y = self.pan_x, self.pan_y

    def on_drag(self, event):
        if self.is_dragging:
            dx, dy = event.x - self.drag_start_x, event.y - self.drag_start_y
            H, W = self.ds_image.shape[:2]
            S = min(self.V_w / W, self.V_h / H) * self.zoom
            self.pan_x = self.orig_pan_x - dx / S
            self.pan_y = self.orig_pan_y - dy / S
            self.render()

    def on_scroll(self, event):
        self.adjust_zoom(1.15 if event.delta > 0 else 1.0 / 1.15)

    def adjust_zoom(self, factor):
        self.zoom = np.clip(self.zoom * factor, 1.0, 15.0)
        self.render()

    def reset_viewport(self):
        self.zoom, self.pan_x, self.pan_y = 1.0, 0.0, 0.0
        self.render()

    def get_zoomed_marker_image(self, center_gds: tuple[float, float], marker_polys: list[np.ndarray], size: int = 220) -> np.ndarray:
        gx, gy = center_gds
        S_base = self.Rg / self.R_ds

        M_rs = cv2.getRotationMatrix2D((self.xc_ds, self.yc_ds), self.current_angle_deg, self.scale_mult)
        M_rs[0, 2] += -self.offset_x / S_base
        M_rs[1, 2] += self.offset_y / S_base

        H, W = self.ds_image.shape[:2]
        ds_img_aligned = cv2.warpAffine(self.ds_image, M_rs, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(30, 30, 30))

        gx_centered, gy_centered = gx - self.xg_c, gy - self.yg_c
        x_can_marker = (gx_centered / S_base) + self.xc_ds
        y_can_marker = self.yc_ds - (gy_centered / S_base)

        crop_size = max(30, min(int(round(5000.0 / S_base)), 400))
        half_crop = crop_size / 2.0

        x1, y1 = int(round(x_can_marker - half_crop)), int(round(y_can_marker - half_crop))
        x2, y2 = x1 + crop_size, y1 + crop_size

        crop_img = np.zeros((crop_size, crop_size, 3), dtype=np.uint8) + 30
        src_x1, src_y1 = max(0, x1), max(0, y1)
        src_x2, src_y2 = min(W, x2), min(H, y2)

        dst_x1, dst_y1 = src_x1 - x1, src_y1 - y1
        dst_x2, dst_y2 = dst_x1 + (src_x2 - src_x1), dst_y1 + (src_y2 - src_y1)

        if (src_x2 > src_x1) and (src_y2 > src_y1):
            crop_img[dst_y1:dst_y2, dst_x1:dst_x2] = ds_img_aligned[src_y1:src_y2, src_x1:src_x2]

        resized_crop = cv2.resize(crop_img, (size, size), interpolation=cv2.INTER_LINEAR)
        overlay = resized_crop.copy()
        pixels_per_micron = size / float(crop_size * S_base)

        # Draw anti-aliased, 1px GDS overlays using OpenCV's sub-pixel shift parameters
        shift = 5
        factor = 1 << shift

        for poly in marker_polys:
            pts_scr = []
            for gx_p, gy_p in poly:
                dg_x, dg_y = gx_p - gx, gy_p - gy
                sx = int(round((size / 2.0 + dg_x * pixels_per_micron) * factor))
                sy = int(round((size / 2.0 - dg_y * pixels_per_micron) * factor))
                pts_scr.append([sx, sy])
                
            if len(pts_scr) >= 2:
                pts_scr = np.array(pts_scr, dtype=np.int32)
                cv2.polylines(overlay, [pts_scr], isClosed=True, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA, shift=shift)

        cv2.addWeighted(overlay, 0.35, resized_crop, 0.65, 0, resized_crop)
        cv2.drawMarker(resized_crop, (size // 2, size // 2), (0, 255, 255), cv2.MARKER_CROSS, 15, 1)
        return resized_crop

    def update_marker_zooms(self):
        if not hasattr(self, "markers") or not self.markers:
            self.draw_placeholder(self.left_zoom_canvas, "No GDS Markers")
            self.draw_placeholder(self.right_zoom_canvas, "No GDS Markers")
            return

        for side, canvas, cache in [("left", self.left_zoom_canvas, "left_tk"), ("right", self.right_zoom_canvas, "right_tk")]:
            m_list = self.markers.get(side, [])
            if m_list:
                # Terminology corrected: `"circle"` changed to `"square"`
                squares = [m for m in m_list if m["type"] == "square"]
                targets = squares if squares else m_list
                cx = np.mean([m["center"][0] for m in targets])
                cy = np.mean([m["center"][1] for m in targets])
                
                polys = [np.array(m["polygon"]) for m in m_list]
                img = self.get_zoomed_marker_image((cx, cy), polys, size=220)
                self.display_on_canvas(canvas, img, cache)
            else:
                self.draw_placeholder(canvas, f"No {side.title()} Markers")

    def display_on_canvas(self, canvas: tk.Canvas, cv_img: np.ndarray, cache_name: str):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        tk_img = ImageTk.PhotoImage(Image.fromarray(rgb_img))
        setattr(self, cache_name, tk_img)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)

    def draw_placeholder(self, canvas: tk.Canvas, text: str):
        canvas.delete("all")
        canvas.create_text(110, 110, text=text, fill="#888888", font=("Helvetica", 9))

    def render(self):
        H, W = self.ds_image.shape[:2]
        S_base_canvas = min(self.V_w / W, self.V_h / H)
        S = S_base_canvas * self.zoom

        M_rs = cv2.getRotationMatrix2D((self.xc_ds, self.yc_ds), self.current_angle_deg, self.scale_mult)
        S_base = self.Rg / self.R_ds
        M_rs[0, 2] += -self.offset_x / S_base
        M_rs[1, 2] += self.offset_y / S_base

        T_align = np.eye(3)
        T_align[:2, :] = M_rs
        tx = self.V_w / 2.0 - S * (self.xc_ds + self.pan_x)
        ty = self.V_h / 2.0 - S * (self.yc_ds + self.pan_y)
        T_view = np.array([[S, 0, tx], [0, S, ty], [0, 0, 1]])

        display_img = cv2.warpAffine(self.ds_image, (T_view @ T_align)[:2, :], (self.V_w, self.V_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(30, 30, 30))

        if self.show_gds and self.gds_polygons:
            for poly in self.gds_polygons:
                if len(poly) > 300:
                    poly = poly[::max(1, len(poly) // 300)]
                pts_scr = []
                for gx, gy in poly:
                    x_can = ((gx - self.xg_c) / S_base) + self.xc_ds
                    y_can = self.yc_ds - ((gy - self.yg_c) / S_base)
                    scr_x, scr_y = int(S * x_can + tx), int(S * y_can + ty)
                    if -500 <= scr_x < self.V_w + 500 and -500 <= scr_y < self.V_h + 500:
                        pts_scr.append([scr_x, scr_y])
                if len(pts_scr) >= 2:
                    cv2.polylines(display_img, [np.array(pts_scr, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

        for y_line in [180, 300, 420]:
            cv2.line(display_img, (0, y_line), (self.V_w, y_line), (0, 255, 255), 1, cv2.LINE_AA)

        self.display_on_canvas(self.canvas, display_img, "tk_img")
        self.status_label.config(text=f"Angle: {self.current_angle_deg:.3f}°\nZoom: {self.zoom:.2f}x\nPan: ({int(self.pan_x)}, {int(self.pan_y)})\nMode: Alignment Active")
        self.update_marker_zooms()

    def confirm(self):
        self.result_angle_rad = self.current_angle_rad
        self.result_tx = -self.offset_x
        self.result_ty = -self.offset_y
        self.result_scale_mult = self.scale_mult
        self.root.destroy()

    def cancel(self):
        self.result_angle_rad = self.initial_angle_rad
        self.result_tx = 0.0
        self.result_ty = 0.0
        self.result_scale_mult = 1.0
        self.root.destroy()


def run_manual_alignment(ds_canvas, config, xc_ds, yc_ds, R_ds, ds_factor, tile_ext, initial_angle_rad, gds_polygons, gds_R, map_mode=False, gds_center=(0.0, 0.0), shear=0.0, markers=None, initial_tx=0.0, initial_ty=0.0, initial_scale=1.0):
    root = tk.Tk()
    app = ManualAlignApp(root, ds_canvas, xc_ds, yc_ds, R_ds, ds_factor, gds_polygons, gds_R, initial_angle_rad, map_mode=map_mode, gds_center=gds_center, shear=shear, markers=markers, initial_tx=initial_tx, initial_ty=initial_ty, initial_scale=initial_scale)
    root.protocol("WM_DELETE_WINDOW", app.cancel)
    root.mainloop()
    return app.result_angle_rad, app.result_tx, app.result_ty, app.result_scale_mult