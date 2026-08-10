
from h2p_ui_branding import install_global_branding as _install_h2p_ui_branding
_install_h2p_ui_branding()

import os
import re
import json
import math
import numpy as np
import cv2
from PIL import Image
import centroid_algorithm

def draw_translucent_polyline(img: np.ndarray, pts: list[tuple[int, int]], color: tuple[int, int, int], thickness: int, alpha: float = 0.5):
    overlay = img.copy()
    poly_pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(overlay, [poly_pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


def save_layout_cache(cache_path, grid_map, tile_w, tile_h, step_x, step_y, min_r_act, min_c_act, num_rows, num_cols):
    serializable_map = {f"{r},{c}": path for (r, c), path in grid_map.items()}
    data = {
        "grid_map": serializable_map, "tile_w": tile_w, "tile_h": tile_h, "step_x": step_x, "step_y": step_y,
        "min_r_act": min_r_act, "min_c_act": min_c_act, "num_rows": num_rows, "num_cols": num_cols
    }
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=4)


def load_layout_cache(cache_path):
    with open(cache_path, "r") as f:
        data = json.load(f)
    grid_map = {}
    for k, path in data["grid_map"].items():
        parts = k.split(",")
        grid_map[(int(parts[0]), int(parts[1]))] = path
    return grid_map, data["tile_w"], data["tile_h"], data["step_x"], data["step_y"], data["min_r_act"], data["min_c_act"], data["num_rows"], data["num_cols"]


class CompressedStitchedImage:
    def __init__(self, folder_path, grid_map, tile_w, tile_h, step_x, step_y, min_r_act, min_c_act, num_rows, num_cols, compressed_img_path):
        self.folder_path, self.grid_map, self.tile_w, self.tile_h = folder_path, grid_map, tile_w, tile_h
        self.step_x, self.step_y, self.min_r_act, self.min_c_act = step_x, step_y, min_r_act, min_c_act
        self.width = (num_cols - 1) * step_x + tile_w
        self.height = (num_rows - 1) * step_y + tile_h
        self.size = (self.width, self.height)
        self.compressed_img = Image.open(compressed_img_path)

    def reduce(self, factor: int) -> Image.Image:
        return self.compressed_img.resize((self.width // factor, self.height // factor), Image.Resampling.BILINEAR)

    def crop(self, box: tuple[int, int, int, int]) -> Image.Image:
        x1, y1, x2, y2 = box
        crop_canvas = Image.new("RGB", (x2 - x1, y2 - y1), (0, 0, 0))
        for (r, c), tile_path in self.grid_map.items():
            tile_x1 = (c - self.min_c_act) * self.step_x
            tile_y1 = (r - self.min_r_act) * self.step_y
            tile_x2, tile_y2 = tile_x1 + self.tile_w, tile_y1 + self.tile_h
            if max(x1, tile_x1) < min(x2, tile_x2) and max(y1, tile_y1) < min(y2, tile_y2):
                if os.path.exists(tile_path):
                    with Image.open(tile_path) as tile_im:
                        if tile_im.mode != "RGB":
                            tile_im = tile_im.convert("RGB")
                        ix1, iy1 = max(x1, tile_x1), max(y1, tile_y1)
                        ix2, iy2 = min(x2, tile_x2), min(y2, tile_y2)
                        tile_crop = tile_im.crop((ix1 - tile_x1, iy1 - tile_y1, ix2 - tile_x1, iy2 - tile_y1))
                        crop_canvas.paste(tile_crop, (ix1 - x1, iy1 - y1))
        return crop_canvas


class LargeWaferTester:
    def __init__(self, image_path: str, display_height: int = 800, debug: bool = False):
        self.image_path, self.target_height, self.debug = image_path, display_height, debug
        if os.path.isdir(image_path):
            self.im = self.stitch_tiles(image_path)
        else:
            dir_name = os.path.dirname(image_path)
            cache_path = os.path.join(dir_name, "large_wafer_layout_cache.json")
            if os.path.exists(cache_path):
                grid_map, tile_w, tile_h, step_x, step_y, min_r, min_c, rows, cols = load_layout_cache(cache_path)
                self.im = CompressedStitchedImage(dir_name, grid_map, tile_w, tile_h, step_x, step_y, min_r, min_c, rows, cols, image_path)
            else:
                self.im = Image.open(image_path)

        self.orig_w, self.orig_h = self.im.size
        self.sidebar_w, self.top_bar_h, self.bottom_bar_h = 320, 60, 80
        master_reduce = max(1, self.orig_h // (self.target_height * 2))
        self.master_gray = np.array(self.im.reduce(master_reduce).convert("L"))
        self.recompute_display_sizes()

        self.STATE_IDLE, self.STATE_WAIT_LEFT, self.STATE_WAIT_RIGHT, self.STATE_FINISHED = 0, 1, 2, 3
        self.current_state = self.STATE_IDLE
        self.status_text = "STATUS: IDLE. CLICK 'START ALIGNMENT' TO BEGIN"
        self.status_bg_color = (15, 15, 15)

        self.left_click_global = self.right_click_global = None
        self.left_marker_global = self.right_marker_global = None
        self.left_resolved_success = self.right_resolved_success = False
        self.left_squares_global = self.right_squares_global = None
        self.right_squares_global = self.right_squares_global = None
        self.left_boxes_global = self.right_boxes_global = None
        self.calibrated_theta = self.calibrated_scale_x = self.calibrated_scale_y = self.expected_col_spacing = None
        self.redraw_gui()

    def stitch_tiles(self, folder_path: str) -> CompressedStitchedImage:
        cache_path = os.path.join(folder_path, "large_wafer_layout_cache.json")
        compressed_img_path = os.path.join(folder_path, "stitched_master.jpg")
        if os.path.exists(cache_path) and os.path.exists(compressed_img_path):
            grid_map, tile_w, tile_h, step_x, step_y, min_r, min_c, rows, cols = load_layout_cache(cache_path)
            return CompressedStitchedImage(folder_path, grid_map, tile_w, tile_h, step_x, step_y, min_r, min_c, rows, cols, compressed_img_path)

        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(exts) and "stitched_master" not in f.lower()]
        grid_map, rows, cols = {}, set(), set()
        for f in files:
            lower = f.lower()
            match = re.search(r'tile_x(\d+)_y(\d+)', lower)
            if match:
                c, r = int(match.group(1)), int(match.group(2))
                grid_map[(r, c)] = os.path.join(folder_path, f)
                rows.add(r)
                cols.add(c)

        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        num_rows, num_cols = max_r - min_r + 1, max_c - min_c + 1

        with Image.open(next(iter(grid_map.values()))) as sample_im:
            tile_w, tile_h = sample_im.size

        step_x, step_y = int(round(tile_w * 0.90)), int(round(tile_h * 0.90))
        stitched_w, stitched_h = (num_cols - 1) * step_x + tile_w, (num_rows - 1) * step_y + tile_h
        ds_factor = 10
        stitched_downscaled = Image.new("RGB", (stitched_w // ds_factor, stitched_h // ds_factor), (0, 0, 0))

        for (r, c), tile_path in grid_map.items():
            if os.path.exists(tile_path):
                s_pos_x = ((c - min_c) * step_x) // ds_factor
                s_pos_y = ((r - min_r) * step_y) // ds_factor
                with Image.open(tile_path) as tile_im:
                    tile_small = tile_im.resize((max(1, tile_w // ds_factor), max(1, tile_h // ds_factor)), Image.Resampling.BILINEAR)
                    stitched_downscaled.paste(tile_small.convert("RGB"), (s_pos_x, s_pos_y))

        stitched_downscaled.save(compressed_img_path, "JPEG", quality=60)
        save_layout_cache(cache_path, grid_map, tile_w, tile_h, step_x, step_y, min_r, min_c, num_rows, num_cols)
        return CompressedStitchedImage(folder_path, grid_map, tile_w, tile_h, step_x, step_y, min_r, min_c, num_rows, num_cols, compressed_img_path)

    def recompute_display_sizes(self):
        self.scale = self.target_height / float(self.orig_h)
        self.display_width = int(self.orig_w * self.scale)
        self.preview_color = cv2.cvtColor(cv2.resize(self.master_gray, (self.display_width, self.target_height)), cv2.COLOR_GRAY2BGR)
        self.canvas_w = self.display_width + self.sidebar_w
        self.canvas_h = self.target_height + self.top_bar_h + self.bottom_bar_h
        self.canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        self.init_dashboard_buttons()

    def init_dashboard_buttons(self):
        panel_y = self.target_height + self.top_bar_h
        self.btn_start = {"box": (20, panel_y + 15, 220, panel_y + 65), "label": "START ALIGNMENT"}
        self.btn_reset = {"box": (240, panel_y + 15, 440, panel_y + 65), "label": "RESET SYSTEM"}
        self.btn_exit  = {"box": (self.display_width - 220, panel_y + 15, self.display_width - 20, panel_y + 65), "label": "EXIT TESTER"}

    def redraw_gui(self):
        cv2.rectangle(self.canvas, (0, 0), (self.canvas_w, self.top_bar_h), self.status_bg_color, -1)
        tx = (self.canvas_w - cv2.getTextSize(self.status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0][0]) // 2
        cv2.putText(self.canvas, self.status_text, (tx, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255) if self.current_state != self.STATE_FINISHED else (0, 255, 0), 2, cv2.LINE_AA)

        self.canvas[self.top_bar_h: self.top_bar_h + self.target_height, 0:self.display_width] = self.preview_color.copy()

        for click in [self.left_click_global, self.right_click_global]:
            if click is not None:
                cv2.circle(self.canvas, (int(click[0] * self.scale), int(click[1] * self.scale) + self.top_bar_h), 6, (0, 255, 0), 2)

        for squares in [self.left_squares_global, self.right_squares_global]:
            if squares is not None:
                for (row, col), (gx, gy) in squares.items():
                    cv2.circle(self.canvas, (int(gx * self.scale), int(gy * self.scale) + self.top_bar_h), 3, (0, 255, 255), -1)

        for marker, resolved in [(self.left_marker_global, self.left_resolved_success), (self.right_marker_global, self.right_resolved_success)]:
            if marker is not None:
                dx, dy = int(marker[0] * self.scale), int(marker[1] * self.scale) + self.top_bar_h
                cv2.circle(self.canvas, (dx, dy), 12, (0, 0, 255) if resolved else (255, 0, 0), -1)

        sidebar_x = self.display_width
        cv2.rectangle(self.canvas, (sidebar_x, self.top_bar_h), (self.canvas_w, self.canvas_h - self.bottom_bar_h), (35, 35, 35), -1)

        active_pt = self.left_marker_global if self.current_state == self.STATE_WAIT_RIGHT else (self.right_marker_global if self.current_state == self.STATE_FINISHED else None)
        active_boxes = self.left_boxes_global if self.current_state == self.STATE_WAIT_RIGHT else (self.right_boxes_global if self.current_state == self.STATE_FINISHED else None)
        active_squares = self.left_squares_global if self.current_state == self.STATE_WAIT_RIGHT else (self.right_squares_global if self.current_state == self.STATE_FINISHED else None)

        sidebar_y = self.top_bar_h + 30
        if active_pt is not None:
            gx, gy = active_pt
            x1, y1 = max(0, int(gx) - 1000), max(0, int(gy) - 1000)
            crop_resized = cv2.resize(cv2.cvtColor(np.array(self.im.crop((x1, y1, min(self.orig_w, x1+2000), min(self.orig_h, y1+2000)))), cv2.COLOR_RGB2BGR), (320, 320))
            if active_boxes:
                for corners in active_boxes.values():
                    pts = [(int((bx - x1) * 320.0 / 2000.0), int((by - y1) * 320.0 / 2000.0)) for bx, by in corners]
                    draw_translucent_polyline(crop_resized, pts, (0, 0, 255), 2, alpha=0.55)
            if active_squares:
                for (r, c), (cx, cy) in active_squares.items():
                    px, py = int((cx - x1) * 320.0 / 2000.0), int((cy - y1) * 320.0 / 2000.0)
                    cv2.circle(crop_resized, (px, py), 5, (0, 255, 255), -1)

            cv2.circle(crop_resized, (160, 160), 25, (0, 255, 0), 2)
            self.canvas[sidebar_y: sidebar_y + 320, sidebar_x: self.canvas_w] = crop_resized
        else:
            cv2.rectangle(self.canvas, (sidebar_x + 10, sidebar_y), (self.canvas_w - 10, sidebar_y + 320), (20, 20, 20), -1)

        # Draw panel control frames
        panel_y = self.target_height + self.top_bar_h
        cv2.rectangle(self.canvas, (0, panel_y), (self.canvas_w, self.canvas_h), (25, 25, 25), -1)
        for btn in [self.btn_start, self.btn_reset, self.btn_exit]:
            x1, y1, x2, y2 = btn["box"]
            # Dynamic color changes depending on alignment state returned to active controls
            if btn["label"] == "START ALIGNMENT":
                color = (0, 140, 0) if self.current_state == self.STATE_IDLE else (50, 50, 50)
            elif btn["label"] == "RESET SYSTEM":
                color = (0, 120, 120) if self.current_state != self.STATE_IDLE else (50, 50, 50)
            else:
                color = (0, 0, 140)
            cv2.rectangle(self.canvas, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(self.canvas, (x1, y1), (x2, y2), (90, 90, 90), 1)
            tx = x1 + ((x2 - x1) - cv2.getTextSize(btn["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]) // 2
            cv2.putText(self.canvas, btn["label"], (tx, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Large Wafer Tester", self.canvas)

    def process_wafer_click(self, x: int, y: int):
        if self.current_state in (self.STATE_IDLE, self.STATE_FINISHED):
            return
        orig_x, orig_y = int(x / self.scale), int(y / self.scale)
        x1, y1 = max(0, orig_x - 1000), max(0, orig_y - 1000)
        crop_gray = np.array(self.im.crop((x1, y1, min(self.orig_w, x1 + 2000), min(self.orig_h, y1 + 2000))).convert("L"))

        success, lx, ly, scale_x, scale_y, circles_local, boxes_local, theta = centroid_algorithm.process_marker_detection(
            crop_gray, float(orig_x - x1), float(orig_y - y1), self.expected_col_spacing, self.calibrated_theta, self.calibrated_scale_x, self.calibrated_scale_y
        )

        global_cx, global_cy = x1 + lx, y1 + ly
        global_squares = {(r, c): (x1 + cx, y1 + cy) for (r, c), (cx, cy) in circles_local.items()} if success else {}
        global_boxes = {(r, c): [(x1 + bx, y1 + by) for bx, by in corners] for (r, c), corners in boxes_local.items()} if success else {}

        if self.current_state == self.STATE_WAIT_LEFT:
            self.left_click_global = (orig_x, orig_y)
            self.left_marker_global = (global_cx, global_cy)
            self.left_squares_global = global_squares
            self.left_boxes_global = global_boxes
            self.left_resolved_success = success
            self.expected_col_spacing = scale_x * 800.0
            self.calibrated_theta, self.calibrated_scale_x, self.calibrated_scale_y = theta, scale_x, scale_y
            self.current_state, self.status_text = self.STATE_WAIT_RIGHT, "LEFT: SNAPPED! CLICK RIGHT APPROXIMATE CENTER"
        else:
            self.right_click_global = (orig_x, orig_y)
            self.right_marker_global = (global_cx, global_cy)
            self.right_squares_global = global_squares
            self.right_boxes_global = global_boxes
            self.right_resolved_success = success
            self.current_state, self.status_text = self.STATE_FINISHED, "STATUS: COMPLETED! PRESS EXIT TO COMPLETE PRESET"
        self.redraw_gui()

    def handle_click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if y >= self.top_bar_h + self.target_height:
            if self.btn_start["box"][0] <= x <= self.btn_start["box"][2] and self.btn_start["box"][1] <= y <= self.btn_start["box"][3]:
                if self.current_state == self.STATE_IDLE:
                    self.current_state, self.status_text = self.STATE_WAIT_LEFT, "STATUS: [LEFT MARKER] CLICK APPROXIMATE CENTER"
            elif self.btn_reset["box"][0] <= x <= self.btn_reset["box"][2] and self.btn_reset["box"][1] <= y <= self.btn_reset["box"][3]:
                self.__init__(self.image_path, self.target_height, self.debug)
            elif self.btn_exit["box"][0] <= x <= self.btn_exit["box"][2] and self.btn_exit["box"][1] <= y <= self.btn_exit["box"][3]:
                self.running = False
            self.redraw_gui()
        elif self.top_bar_h <= y < self.top_bar_h + self.target_height and x < self.display_width:
            self.process_wafer_click(x, y - self.top_bar_h)

    def run(self):
        cv2.namedWindow("Large Wafer Tester", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Large Wafer Tester", self.canvas_w, self.canvas_h)
        cv2.setMouseCallback("Large Wafer Tester", self.handle_click)
        self.running = True
        while self.running:
            key = cv2.waitKeyEx(20)
            if key in (27, ord("q"), ord("Q")):
                self.running = False
        cv2.destroyWindow("Large Wafer Tester")