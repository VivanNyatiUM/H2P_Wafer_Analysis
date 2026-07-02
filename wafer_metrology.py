import re
from pathlib import Path

import cv2
import numpy as np

import illumination_stitching


def get_aligned_tile(src_img, alpha, scale_factor=1.0):
    h, w = src_img.shape[:2]
    center = (w / 2.0, h / 2.0)
    angle_deg = alpha * 180.0 / np.pi
    M = cv2.getRotationMatrix2D(center, angle_deg, scale_factor)
    return cv2.warpAffine(
        src_img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def fit_circle_least_squares(points):
    pts = np.array(points, dtype=np.float64)
    A = np.column_stack((2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))))
    B = pts[:, 0] ** 2 + pts[:, 1] ** 2
    res = np.linalg.lstsq(A, B, rcond=None)[0]
    xc, yc = res[0], res[1]
    val = res[2] + xc ** 2 + yc ** 2
    if val < 0:
        raise ValueError("Invalid circle geometry detected during least squares fit.")
    R = np.sqrt(val)
    return xc, yc, R


def robust_circle_fit(points, outlier_pct=0.012, max_iters=5):
    pts = np.array(points, dtype=np.float64)
    for _ in range(max_iters):
        xc, yc, R = fit_circle_least_squares(pts)
        dists = np.linalg.norm(pts - np.array([xc, yc]), axis=1)
        err = np.abs(dists - R)
        threshold = outlier_pct * R
        inliers = pts[err < threshold]
        if len(inliers) < 3:
            break
        pts = inliers
    return xc, yc, R, float(np.mean(np.abs(dists - R)))


def find_dynamic_clipping_bounds(contour_pts, ds_factor, base_tolerance_px=3):
    tolerance_px = int(np.clip(base_tolerance_px / ds_factor, 1, 20))
    x_coords = contour_pts[:, 0]
    y_coords = contour_pts[:, 1]

    left_edge, right_edge = np.min(x_coords), np.max(x_coords)
    top_edge, bottom_edge = np.min(y_coords), np.max(y_coords)

    left_clip_count = np.sum(x_coords < (left_edge + tolerance_px))
    right_clip_count = np.sum(x_coords > (right_edge - tolerance_px))
    top_clip_count = np.sum(y_coords < (top_edge + tolerance_px))

    dynamic_left_limit = left_edge + tolerance_px if left_clip_count > 15 else left_edge - 1
    dynamic_right_limit = right_edge - tolerance_px if right_clip_count > 15 else right_edge + 1
    dynamic_top_limit = top_edge + tolerance_px if top_clip_count > 15 else top_edge - 1
    dynamic_bottom_limit = bottom_edge + 1

    return dynamic_left_limit, dynamic_right_limit, dynamic_top_limit, dynamic_bottom_limit


def robust_vertical_profile_flat(gray_img, xc_ds, yc_ds, R_ds, h_img, w_img, is_inverted=False):
    profile_pts = []
    x_start = int(xc_ds - 0.20 * R_ds)
    x_end = int(xc_ds + 0.20 * R_ds)
    y_start_scan = int(yc_ds + 0.75 * R_ds)
    y_end_scan = min(int(yc_ds + 0.94 * R_ds), h_img - 15)

    for x in range(x_start, x_end, 2):
        if x < 0 or x >= w_img:
            continue
        column_pixels = gray_img[y_start_scan:y_end_scan, x].astype(np.float64)
        if len(column_pixels) < 10:
            continue
        gradients = np.diff(column_pixels)
        for i in range(len(gradients)):
            intensity = column_pixels[i]
            next_intensity = column_pixels[i + 1]
            grad = gradients[i]
            if not is_inverted:
                if grad < -5.0 and intensity > 90 and (15 <= next_intensity <= 75):
                    profile_pts.append([x, y_start_scan + i + 1])
                    break
            else:
                if grad > 5.0 and intensity < 70 and (90 <= next_intensity <= 160):
                    profile_pts.append([x, y_start_scan + i + 1])
                    break

    return np.array(profile_pts, dtype=np.float64)


def _weighted_line_fit(pts):
    if len(pts) < 2:
        raise ValueError("Need at least 2 points for line fit.")
    cx = np.mean(pts[:, 0])
    weights = 1.0 / (1.0 + np.abs(pts[:, 0] - cx) / (np.ptp(pts[:, 0]) + 1e-9))
    coeffs = np.polyfit(pts[:, 0], pts[:, 1], 1, w=weights)
    slope, intercept = coeffs
    predicted = slope * pts[:, 0] + intercept
    return slope, intercept, float(np.mean(np.abs(pts[:, 1] - predicted)))


def geometry_based_flat_fit(circular_pts, xc_ds, yc_ds, R_ds):
    dx = circular_pts[:, 0] - xc_ds
    dy = circular_pts[:, 1] - yc_ds
    dists = np.sqrt(dx ** 2 + dy ** 2)
    angles = np.arctan2(dy, dx)

    best_alpha = None
    best_residual = np.inf
    best_flat_pts = None

    for sector_half in np.arange(0.44, 0.19, -0.04):
        bottom_sector_mask = (angles >= np.pi / 2.0 - sector_half) & (angles <= np.pi / 2.0 + sector_half)
        flat_chord_mask = bottom_sector_mask & (dists >= 0.91 * R_ds) & (dists <= 0.995 * R_ds)
        flat_pts = circular_pts[flat_chord_mask]
        if len(flat_pts) < 10:
            continue

        for _ in range(5):
            if len(flat_pts) < 10:
                break
            slope, intercept, mean_err = _weighted_line_fit(flat_pts)
            predicted_y = slope * flat_pts[:, 0] + intercept
            errors = np.abs(flat_pts[:, 1] - predicted_y)
            inliers = errors < 0.8
            flat_pts = flat_pts[inliers]

        if len(flat_pts) < 10:
            continue

        slope, intercept, residual = _weighted_line_fit(flat_pts)
        alpha_candidate = np.arctan(slope)
        if residual < best_residual:
            best_residual = residual
            best_alpha = alpha_candidate
            best_flat_pts = flat_pts.copy()

    return best_flat_pts, best_alpha, best_residual


def detect_wafer_on_canvas(ds_image, ds_factor):
    h_img, w_img = ds_image.shape[:2]
    gray = cv2.cvtColor(ds_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_val = float(max(15.0, otsu_thresh))
    _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

    cw, ch = int(w_img * 0.05), int(h_img * 0.05)
    corners = [
        thresh[0:ch, 0:cw],
        thresh[0:ch, w_img - cw:w_img],
        thresh[h_img - ch:h_img, 0:cw],
        thresh[h_img - ch:h_img, w_img - cw:w_img],
    ]
    corner_pixels = np.concatenate([c.flatten() for c in corners])
    corner_white_pct = (np.sum(corner_pixels == 255) / corner_pixels.size) * 100.0

    is_inverted = False
    if corner_white_pct > 50.0:
        thresh = cv2.bitwise_not(thresh)
        is_inverted = True

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    if num_labels >= 2:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        wafer_mask = (labels == largest_label).astype(np.uint8) * 255
    else:
        wafer_mask = thresh

    k_size = int(np.clip(15 * (ds_factor / 0.05), 5, 31))
    if k_size % 2 == 0:
        k_size += 1
    kernel_seg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    eroded_mask = cv2.erode(wafer_mask, kernel_seg)
    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    solid_mask = np.zeros_like(wafer_mask)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(solid_mask, [largest_contour], -1, 255, -1)
    else:
        solid_mask = wafer_mask

    solid_mask = cv2.dilate(solid_mask, kernel_seg)
    solid_mask = cv2.morphologyEx(solid_mask, cv2.MORPH_CLOSE, kernel_seg)
    solid_mask = cv2.morphologyEx(solid_mask, cv2.MORPH_OPEN, kernel_seg)

    contours, _ = cv2.findContours(solid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Could not find wafer boundary contour on stitched canvas.")

    wafer_contour = max(contours, key=cv2.contourArea)
    contour_pts = wafer_contour.squeeze(axis=1).astype(np.float64)

    left_lim, right_lim, top_lim, bottom_lim = find_dynamic_clipping_bounds(contour_pts, ds_factor)
    boundary_mask = (
        (contour_pts[:, 0] > left_lim)
        & (contour_pts[:, 0] < right_lim)
        & (contour_pts[:, 1] > top_lim)
        & (contour_pts[:, 1] < bottom_lim)
    )
    circular_pts = contour_pts[boundary_mask]
    if len(circular_pts) < 10:
        circular_pts = contour_pts

    rough_yc = np.mean(circular_pts[:, 1])
    top_half_pts = circular_pts[circular_pts[:, 1] < rough_yc]
    if len(top_half_pts) < 3:
        top_half_pts = circular_pts

    xc_ds, yc_ds, R_ds, _ = robust_circle_fit(top_half_pts, outlier_pct=0.012, max_iters=5)

    profile_pts = robust_vertical_profile_flat(gray, xc_ds, yc_ds, R_ds, h_img, w_img, is_inverted=is_inverted)
    if len(profile_pts) >= 20:
        for _ in range(5):
            slope, intercept = np.polyfit(profile_pts[:, 0], profile_pts[:, 1], 1)
            predicted_y = slope * profile_pts[:, 0] + intercept
            inliers = np.abs(profile_pts[:, 1] - predicted_y) < 1.2
            if np.sum(inliers) < 15:
                break
            profile_pts = profile_pts[inliers]
        return xc_ds / ds_factor, yc_ds / ds_factor, R_ds / ds_factor, np.arctan(
            np.polyfit(profile_pts[:, 0], profile_pts[:, 1], 1)[0]
        )

    flat_pts, alpha, _ = geometry_based_flat_fit(circular_pts, xc_ds, yc_ds, R_ds)
    if flat_pts is not None and len(flat_pts) >= 10:
        return xc_ds / ds_factor, yc_ds / ds_factor, R_ds / ds_factor, alpha

    dists = np.sqrt((circular_pts[:, 0] - xc_ds) ** 2 + (circular_pts[:, 1] - yc_ds) ** 2)
    angles_all = np.arctan2(circular_pts[:, 1] - yc_ds, circular_pts[:, 0] - xc_ds)
    search_angles = np.linspace(np.pi / 2.0 - 0.44, np.pi / 2.0 + 0.44, 360)
    avg_dists = np.array([
        np.mean(dists[np.minimum(np.abs(angles_all - a), 2 * np.pi - np.abs(angles_all - a)) <= 0.22])
        for a in search_angles
    ])
    return xc_ds / ds_factor, yc_ds / ds_factor, R_ds / ds_factor, search_angles[np.argmin(avg_dists)] - np.pi / 2.0


def generate_downscaled_stitch(folder, config):
    """
    Illumination-normalized feather-blended coarse stitch.

    Same public API as the original function:
        return ds_canvas, tile_ext
    """
    return illumination_stitching.generate_downscaled_stitch(folder, config, verbose=True)
