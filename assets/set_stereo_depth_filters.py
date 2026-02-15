#!/usr/bin/env python3

"""
Interactive stereo depth filter tuning tool.

@file    set_stereo_depth_filters.py
@author  Michael Venz
@version 1.0
@date    2025-02-14

@brief   Real-time stereo depth parameter tuning with side-by-side disparity view.

Window 1: Live stereo view (raw + filtered disparity side by side, color mapped).
Window 2: Trackbar controls for stereo settings and all filter parameters,
           plus a text readout of the current settings you can copy into code.

All stereo modes (subpixel, LR check, extended disparity) can be toggled at
runtime via trackbars - no restart needed.  Filter delta/threshold trackbars
auto-rescale when subpixel mode or fractional bits change.

Press 'q' to quit.  Press 'p' to print current settings to the terminal.

Usage examples:

  # Default (1280x800, no subpixel/lr_check/extended, all togglable at runtime)
  python3 set_stereo_depth_filters.py

  # Start with subpixel and left-right check enabled
  python3 set_stereo_depth_filters.py --subpixel --lr_check

  # Subpixel + LR check with custom threshold and fractional bits
  python3 set_stereo_depth_filters.py --subpixel --subpixel_fractional_bits 4 --lr_check --lrc_threshold 3

  # Extended disparity (for very close objects <35cm)
  python3 set_stereo_depth_filters.py --extended_disparity

  # All features enabled
  python3 set_stereo_depth_filters.py --subpixel --lr_check --extended_disparity

  # Lower resolution for faster processing
  python3 set_stereo_depth_filters.py --width 640 --height 400

  # Scale display to 50% so side-by-side view fits on screen
  python3 set_stereo_depth_filters.py --display_scale 0.5

  # Typical tuning session: subpixel + lr_check, scaled to fit screen
  python3 set_stereo_depth_filters.py --subpixel --lr_check --display_scale 0.5
"""

import argparse

import cv2
import depthai as dai
import numpy as np

FPS: int = 45

# ── Help descriptions for each setting ────────────────────────────────────────
# Shown as tooltips when hovering over the "?" icon next to each setting.

HELP_TEXT: dict[str, str] = {
    "Subpixel":
        "Computes disparity at sub-pixel level for finer depth precision."
        " Especially useful for long-range measurements.",
    "Subpixel Frac Bits":
        "Number of fractional bits (3-5) for subpixel accuracy."
        " Higher = finer precision but reduces maximum depth range.",
    "LR Check":
        "Validates depth by comparing left-to-right and right-to-left disparity."
        " Removes incorrect matches and occlusions.",
    "LRC Threshold":
        "Max disparity difference allowed between L-R and R-L checks (0-10)."
        " Lower = stricter validation, fewer false matches.",
    "Extended Disparity":
        "Doubles disparity search range (96->192) to detect very close objects."
        " Useful for objects closer than ~35 cm.",
    "Confidence Thresh":
        "Minimum stereo matching confidence (0-255)."
        " Higher values filter out uncertain / low-quality depth estimates.",
    "Bilateral Sigma":
        "Smoothing strength of the bilateral filter (0-250)."
        " Reduces noise while preserving depth edges. 0 = disabled.",
    "Max Disparity":
        "Maximum possible disparity value (read-only)."
        " Determined by subpixel mode, fractional bits, and extended disparity.",
    "Speckle Enable":
        "Removes speckle noise - small regions with high disparity variance"
        " that appear as noisy blobs in the depth map.",
    "Speckle Range":
        "Max speckle region size in pixels (0-200)."
        " Regions smaller than this are considered noise and removed.",
    "Speckle Diff Thresh":
        "Max disparity difference within a valid region."
        " Neighboring pixels exceeding this difference are filtered out.",
    "Spatial Enable":
        "Edge-preserving spatial filter that fills invalid depth pixels"
        " using valid neighbors. Smooths while keeping edges sharp.",
    "Spatial Hole Radius":
        "Max search radius (0-16 pixels) for finding valid pixels to fill holes."
        " Larger = fills bigger gaps but may reduce accuracy.",
    "Spatial Alpha x100":
        "Smoothing strength (0-100, representing 0.00-1.00)."
        " Higher = stronger smoothing but less edge preservation.",
    "Spatial Delta":
        "Edge threshold for spatial filter."
        " Pixels with disparity difference above this are not smoothed together.",
    "Spatial Iterations":
        "Number of horizontal+vertical filter passes (1-5)."
        " More iterations = stronger smoothing effect.",
    "Temporal Enable":
        "Smooths depth over time using previous frames."
        " Improves stability but adds latency. Best for static scenes.",
    "Temporal Persist":
        "Persistence mode controlling how many valid frames are needed"
        " to accept a depth value. Stricter = less noise, more holes.",
    "Temporal Alpha x100":
        "Temporal smoothing factor (0-100, representing 0.00-1.00)."
        " Higher = more weight on current frame, less temporal smoothing.",
    "Temporal Delta":
        "Validity threshold for temporal filter."
        " Depth changes exceeding this delta trigger re-evaluation.",
    "Median (0=Off 1=3x3 2=5x5)":
        "Fast noise reduction using median of neighboring pixels."
        " Larger kernel = more smoothing. Runs on host (device uses 7x7).",
}

# ── Trackbar ↔ enum mappings ────────────────────────────────────────────────

MEDIAN_OPTIONS: list[tuple[str, object]] = [
    ("MEDIAN_OFF", dai.node.ImageFilters.MedianFilterParams.MEDIAN_OFF),
    ("KERNEL_3x3", dai.node.ImageFilters.MedianFilterParams.KERNEL_3x3),
    ("KERNEL_5x5", dai.node.ImageFilters.MedianFilterParams.KERNEL_5x5),
]

PERSISTENCY_OPTIONS: list[tuple[str, object]] = [
    ("PERSISTENCY_OFF", dai.filters.params.TemporalFilter.PersistencyMode.PERSISTENCY_OFF),
    ("VALID_8_OUT_OF_8", dai.filters.params.TemporalFilter.PersistencyMode.VALID_8_OUT_OF_8),
    ("VALID_2_IN_LAST_3", dai.filters.params.TemporalFilter.PersistencyMode.VALID_2_IN_LAST_3),
    ("VALID_2_IN_LAST_4", dai.filters.params.TemporalFilter.PersistencyMode.VALID_2_IN_LAST_4),
    ("VALID_2_OUT_OF_8", dai.filters.params.TemporalFilter.PersistencyMode.VALID_2_OUT_OF_8),
    ("VALID_1_IN_LAST_2", dai.filters.params.TemporalFilter.PersistencyMode.VALID_1_IN_LAST_2),
    ("VALID_1_IN_LAST_5", dai.filters.params.TemporalFilter.PersistencyMode.VALID_1_IN_LAST_5),
    ("VALID_1_IN_LAST_8", dai.filters.params.TemporalFilter.PersistencyMode.VALID_1_IN_LAST_8),
    ("PERSISTENCY_INDEFINITELY", dai.filters.params.TemporalFilter.PersistencyMode.PERSISTENCY_INDEFINITELY),
]

# Trackbar names for delta/threshold values that need subpixel scaling
DELTA_TRACKBARS: list[str] = ["Speckle Diff Thresh", "Spatial Delta", "Temporal Delta"]

# ── Default trackbar values ─────────────────────────────────────────────────
# Edit these to change the starting position of every trackbar.
# CLI args (--subpixel, --lr_check, etc.) override the stereo defaults.

DEFAULTS: dict[str, int] = {
    # Stereo settings (BOTH) - also used as CLI arg defaults
    # Based on ROBOTICS preset (FAST_DENSITY base profile)
    "subpixel":             1,      # 0=off, 1=on (CLI: --subpixel)
    "subpixel_frac_bits":   3,      # 3-5, clamped to min 3 at runtime
    "lr_check":             1,      # 0=off, 1=on (CLI: --lr_check)
    "lrc_threshold":        10,     # 0-10 (FAST_DENSITY base = 10)
    "extended_disparity":   1,      # 0=off, 1=on (CLI: --extended_disparity)
    "confidence":           15,     # 0-255 (FAST_DENSITY base = 15)
    "bilateral":            0,      # 0-250
    # Filter settings (FILTERED only)
    "speckle_enable":       1,
    "speckle_range":        200,
    "speckle_diff":         2,      # auto-scaled by init_scale (->16 on trackbar)
    "spatial_enable":       1,
    "spatial_hole_rad":     2,
    "spatial_alpha":        50,     # stored as x100 (50 = 0.50)
    "spatial_delta":        20,     # auto-scaled by init_scale (->160 on trackbar)
    "spatial_iters":        1,
    "temporal_enable":      0,      # DISABLED in ROBOTICS preset
    "temporal_persist":     0,      # index into PERSISTENCY_OPTIONS
    "temporal_alpha":       0,      # stored as x100 (40 = 0.40)
    "temporal_delta":       0,      # auto-scaled by init_scale
    "median":               2,      # 0=Off, 1=3x3, 2=5x5 (ROBOTICS uses 7x7 device-side, 5x5 is host-side max)
}

# ── Helpers ──────────────────────────────────────────────────────────────────

CTRL_WIN: str = "Filter Controls"

# Map display labels (from settings_rows) to HELP_TEXT keys.
# Most match directly; a few panel labels differ from trackbar names.
LABEL_TO_HELP: dict[str, str] = {
    "Subpixel": "Subpixel",
    "LR Check": "LR Check",
    "Extended Disparity": "Extended Disparity",
    "Confidence Thresh": "Confidence Thresh",
    "Bilateral Sigma": "Bilateral Sigma",
    "Max Disparity": "Max Disparity",
    "Speckle Enable": "Speckle Enable",
    "Speckle Range": "Speckle Range",
    "Speckle Diff Thresh": "Speckle Diff Thresh",
    "Spatial Enable": "Spatial Enable",
    "Spatial Hole Radius": "Spatial Hole Radius",
    "Spatial Alpha": "Spatial Alpha x100",
    "Spatial Delta": "Spatial Delta",
    "Spatial Iterations": "Spatial Iterations",
    "Temporal Enable": "Temporal Enable",
    "Temporal Persist": "Temporal Persist",
    "Temporal Alpha": "Temporal Alpha x100",
    "Temporal Delta": "Temporal Delta",
    "Median Filter": "Median (0=Off 1=3x3 2=5x5)",
}


def noop(_: int) -> None:
    """Trackbar callback -- we poll values in the main loop instead."""
    pass


def compute_scale(subpixel: bool, fractional_bits: int) -> int:
    """Return the disparity scale factor for the current subpixel mode.

    Args:
        subpixel:        Whether subpixel mode is enabled.
        fractional_bits: Number of fractional bits (3-5).

    Returns:
        Scale factor (1 when subpixel is off, 2**fractional_bits otherwise).
    """
    return (2 ** fractional_bits) if subpixel else 1


def compute_max_disparity(subpixel: bool, fractional_bits: int, extended: bool) -> int:
    """Compute the maximum disparity value for normalization.

    Args:
        subpixel:        Whether subpixel mode is enabled.
        fractional_bits: Number of fractional bits (3-5).
        extended:        Whether extended disparity mode is enabled.

    Returns:
        Maximum possible disparity value.
    """
    base = 192 if extended else 96
    if subpixel:
        return base * (2 ** fractional_bits)
    return base


def create_trackbars(args: argparse.Namespace) -> None:
    """Create all trackbars and register mouse callback.

    CLI args override DEFAULTS for stereo settings.

    Args:
        args: Parsed command-line arguments.
    """
    cv2.namedWindow(CTRL_WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(CTRL_WIN, _mouse_callback)
    d = DEFAULTS
    init_scale = compute_scale(args.subpixel, args.subpixel_fractional_bits)

    # --- Stereo settings (CLI flags override DEFAULTS) -----------------------
    cv2.createTrackbar("Subpixel",              CTRL_WIN, int(args.subpixel), 1, noop)
    cv2.createTrackbar("Subpixel Frac Bits",    CTRL_WIN, args.subpixel_fractional_bits, 5, noop)
    cv2.createTrackbar("LR Check",              CTRL_WIN, int(args.lr_check), 1, noop)
    cv2.createTrackbar("LRC Threshold",         CTRL_WIN, args.lrc_threshold, 10, noop)
    cv2.createTrackbar("Extended Disparity",    CTRL_WIN, int(args.extended_disparity), 1, noop)
    cv2.createTrackbar("Confidence Thresh",     CTRL_WIN, args.confidence, 255, noop)
    cv2.createTrackbar("Bilateral Sigma",       CTRL_WIN, args.bilateral, 250, noop)

    # --- Filter controls (use DEFAULTS) -------------------------------------
    # Speckle
    cv2.createTrackbar("Speckle Enable",        CTRL_WIN, d["speckle_enable"], 1, noop)
    cv2.createTrackbar("Speckle Range",         CTRL_WIN, d["speckle_range"], 200, noop)
    cv2.createTrackbar("Speckle Diff Thresh",   CTRL_WIN, d["speckle_diff"] * init_scale, 250, noop)

    # Spatial
    cv2.createTrackbar("Spatial Enable",        CTRL_WIN, d["spatial_enable"], 1, noop)
    cv2.createTrackbar("Spatial Hole Radius",   CTRL_WIN, d["spatial_hole_rad"], 16, noop)
    cv2.createTrackbar("Spatial Alpha x100",    CTRL_WIN, d["spatial_alpha"], 100, noop)
    cv2.createTrackbar("Spatial Delta",         CTRL_WIN, d["spatial_delta"] * init_scale, 250, noop)
    cv2.createTrackbar("Spatial Iterations",    CTRL_WIN, d["spatial_iters"], 5, noop)

    # Temporal
    cv2.createTrackbar("Temporal Enable",       CTRL_WIN, d["temporal_enable"], 1, noop)
    cv2.createTrackbar("Temporal Persist",      CTRL_WIN, d["temporal_persist"], 8, noop)
    cv2.createTrackbar("Temporal Alpha x100",   CTRL_WIN, d["temporal_alpha"], 100, noop)
    cv2.createTrackbar("Temporal Delta",        CTRL_WIN, d["temporal_delta"] * init_scale, 250, noop)

    # Median (0=Off, 1=3x3, 2=5x5)
    cv2.createTrackbar("Median (0=Off 1=3x3 2=5x5)", CTRL_WIN, d["median"], 2, noop)


def read_stereo_trackbars() -> dict:
    """Read stereo setting trackbars and return a dict.

    Returns:
        Dict with keys: subpixel, subpixel_fractional_bits, lr_check,
        lrc_threshold, extended_disparity, confidence, bilateral.
    """
    return {
        "subpixel":              bool(cv2.getTrackbarPos("Subpixel", CTRL_WIN)),
        "subpixel_fractional_bits": max(3, cv2.getTrackbarPos("Subpixel Frac Bits", CTRL_WIN)),
        "lr_check":              bool(cv2.getTrackbarPos("LR Check", CTRL_WIN)),
        "lrc_threshold":         cv2.getTrackbarPos("LRC Threshold", CTRL_WIN),
        "extended_disparity":    bool(cv2.getTrackbarPos("Extended Disparity", CTRL_WIN)),
        "confidence":            cv2.getTrackbarPos("Confidence Thresh", CTRL_WIN),
        "bilateral":             cv2.getTrackbarPos("Bilateral Sigma", CTRL_WIN),
    }


def read_filter_trackbars() -> dict:
    """Read filter setting trackbars and return a dict.

    Returns:
        Dict with keys: speckle_enable, speckle_range, speckle_diff,
        spatial_enable, spatial_hole_rad, spatial_alpha, spatial_delta,
        spatial_iters, temporal_enable, temporal_persist, temporal_alpha,
        temporal_delta, median.
    """
    return {
        "speckle_enable":     cv2.getTrackbarPos("Speckle Enable",      CTRL_WIN),
        "speckle_range":      cv2.getTrackbarPos("Speckle Range",       CTRL_WIN),
        "speckle_diff":       cv2.getTrackbarPos("Speckle Diff Thresh", CTRL_WIN),
        "spatial_enable":     cv2.getTrackbarPos("Spatial Enable",      CTRL_WIN),
        "spatial_hole_rad":   cv2.getTrackbarPos("Spatial Hole Radius", CTRL_WIN),
        "spatial_alpha":      cv2.getTrackbarPos("Spatial Alpha x100",  CTRL_WIN),
        "spatial_delta":      cv2.getTrackbarPos("Spatial Delta",       CTRL_WIN),
        "spatial_iters":      cv2.getTrackbarPos("Spatial Iterations",  CTRL_WIN),
        "temporal_enable":    cv2.getTrackbarPos("Temporal Enable",     CTRL_WIN),
        "temporal_persist":   cv2.getTrackbarPos("Temporal Persist",    CTRL_WIN),
        "temporal_alpha":     cv2.getTrackbarPos("Temporal Alpha x100", CTRL_WIN),
        "temporal_delta":     cv2.getTrackbarPos("Temporal Delta",      CTRL_WIN),
        "median":             cv2.getTrackbarPos("Median (0=Off 1=3x3 2=5x5)", CTRL_WIN),
    }


def rescale_delta_trackbars(old_scale: int, new_scale: int) -> None:
    """Rescale delta/threshold trackbar values when subpixel mode changes.

    Args:
        old_scale: Previous subpixel scale factor.
        new_scale: New subpixel scale factor.
    """
    if old_scale == new_scale:
        return
    for name in DELTA_TRACKBARS:
        old_val = cv2.getTrackbarPos(name, CTRL_WIN)
        new_val = min(250, max(0, int(old_val * new_scale / old_scale)))
        cv2.setTrackbarPos(name, CTRL_WIN, new_val)


def build_filter_params(filter_vals: dict) -> tuple:
    """Convert filter trackbar values into depthai filter param objects.

    Args:
        filter_vals: Dict from read_filter_trackbars().

    Returns:
        Tuple of (speckle, temporal, spatial, median) param objects.
    """
    assert 0 <= filter_vals["temporal_persist"] < len(PERSISTENCY_OPTIONS), \
        f"temporal_persist index {filter_vals['temporal_persist']} out of range"
    assert 0 <= filter_vals["median"] < len(MEDIAN_OPTIONS), \
        f"median index {filter_vals['median']} out of range"

    speckle = dai.node.ImageFilters.SpeckleFilterParams()
    speckle.enable = bool(filter_vals["speckle_enable"])
    speckle.speckleRange = filter_vals["speckle_range"]
    speckle.differenceThreshold = filter_vals["speckle_diff"]

    spatial = dai.node.ImageFilters.SpatialFilterParams()
    spatial.enable = bool(filter_vals["spatial_enable"])
    spatial.holeFillingRadius = filter_vals["spatial_hole_rad"]
    spatial.alpha = filter_vals["spatial_alpha"] / 100.0
    spatial.delta = filter_vals["spatial_delta"]
    spatial.numIterations = max(1, filter_vals["spatial_iters"])

    temporal = dai.node.ImageFilters.TemporalFilterParams()
    temporal.enable = bool(filter_vals["temporal_enable"])
    temporal.persistencyMode = PERSISTENCY_OPTIONS[filter_vals["temporal_persist"]][1]
    temporal.alpha = filter_vals["temporal_alpha"] / 100.0
    temporal.delta = filter_vals["temporal_delta"]

    median = MEDIAN_OPTIONS[filter_vals["median"]][1]

    return speckle, temporal, spatial, median


def build_stereo_config(stereo_vals: dict) -> dai.StereoDepthConfig:
    """Build a StereoDepthConfig from stereo trackbar values.

    Args:
        stereo_vals: Dict from read_stereo_trackbars().

    Returns:
        Configured StereoDepthConfig ready to send to the device.
    """
    cfg = dai.StereoDepthConfig()
    cfg.algorithmControl.enableSubpixel = stereo_vals["subpixel"]
    cfg.algorithmControl.subpixelFractionalBits = stereo_vals["subpixel_fractional_bits"]
    cfg.algorithmControl.enableLeftRightCheck = stereo_vals["lr_check"]
    cfg.algorithmControl.leftRightCheckThreshold = stereo_vals["lrc_threshold"]
    cfg.algorithmControl.enableExtended = stereo_vals["extended_disparity"]
    cfg.costMatching.confidenceThreshold = stereo_vals["confidence"]
    cfg.postProcessing.bilateralSigmaValue = stereo_vals["bilateral"]
    return cfg


def settings_rows(stereo_vals: dict, filter_vals: dict) -> list:
    """Return settings as (label, value, applies_to) tuples.

    applies_to is "BOTH" for stereo settings (affect raw + filtered)
    or "FILTERED" for filter settings (affect filtered only).
    None entries act as visual separators.

    Args:
        stereo_vals: Dict from read_stereo_trackbars().
        filter_vals: Dict from read_filter_trackbars().

    Returns:
        List of (label, value, applies_to) tuples with None separators.
    """
    subpixel = stereo_vals["subpixel"]
    frac_bits = stereo_vals["subpixel_fractional_bits"]
    extended = stereo_vals["extended_disparity"]
    scale = compute_scale(subpixel, frac_bits)
    max_disp = compute_max_disparity(subpixel, frac_bits, extended)

    rows: list = []

    # Stereo settings (affect both images)
    sp_val = f"True  ({frac_bits} frac bits, {scale}x)" if subpixel else "False"
    rows.append(("Subpixel", sp_val, "BOTH"))
    lr_val = f"True  (threshold={stereo_vals['lrc_threshold']})" if stereo_vals["lr_check"] else "False"
    rows.append(("LR Check", lr_val, "BOTH"))
    rows.append(("Extended Disparity", str(extended), "BOTH"))
    rows.append(("Confidence Thresh", str(stereo_vals["confidence"]), "BOTH"))
    rows.append(("Bilateral Sigma", str(stereo_vals["bilateral"]), "BOTH"))
    rows.append(("Max Disparity", str(max_disp), "BOTH"))
    rows.append(None)  # separator

    # Filter settings (affect filtered image only)
    rows.append(("Speckle Enable", str(bool(filter_vals["speckle_enable"])), "FILTERED"))
    rows.append(("Speckle Range", str(filter_vals["speckle_range"]), "FILTERED"))
    rows.append(("Speckle Diff Thresh", str(filter_vals["speckle_diff"]), "FILTERED"))
    rows.append(None)
    rows.append(("Spatial Enable", str(bool(filter_vals["spatial_enable"])), "FILTERED"))
    rows.append(("Spatial Hole Radius", str(filter_vals["spatial_hole_rad"]), "FILTERED"))
    rows.append(("Spatial Alpha", f"{filter_vals['spatial_alpha'] / 100.0:.2f}", "FILTERED"))
    rows.append(("Spatial Delta", str(filter_vals["spatial_delta"]), "FILTERED"))
    rows.append(("Spatial Iterations", str(max(1, filter_vals["spatial_iters"])), "FILTERED"))
    rows.append(None)
    rows.append(("Temporal Enable", str(bool(filter_vals["temporal_enable"])), "FILTERED"))
    rows.append(("Temporal Persist", PERSISTENCY_OPTIONS[filter_vals["temporal_persist"]][0], "FILTERED"))
    rows.append(("Temporal Alpha", f"{filter_vals['temporal_alpha'] / 100.0:.2f}", "FILTERED"))
    rows.append(("Temporal Delta", str(filter_vals["temporal_delta"]), "FILTERED"))
    rows.append(None)
    rows.append(("Median Filter", MEDIAN_OPTIONS[filter_vals["median"]][0], "FILTERED"))

    return rows


def settings_text(stereo_vals: dict, filter_vals: dict) -> list[str]:
    """Return all current settings as plain text lines (for terminal printing).

    Args:
        stereo_vals: Dict from read_stereo_trackbars().
        filter_vals: Dict from read_filter_trackbars().

    Returns:
        List of formatted strings, one per line.
    """
    rows = settings_rows(stereo_vals, filter_vals)
    subpixel = stereo_vals["subpixel"]
    scale = compute_scale(subpixel, stereo_vals["subpixel_fractional_bits"])

    lines: list[str] = []
    lines.append(f"{'Setting':<22} {'Value':<30} {'Applies To'}")
    lines.append("-" * 65)
    for row in rows:
        if row is None:
            lines.append("")
            continue
        label, value, applies = row
        lines.append(f"{label:<22} {value:<30} {applies}")
    if subpixel:
        lines.append("")
        lines.append(f"  Note: delta/thresh values include {scale}x subpixel scale")
    return lines


# ── Settings panel (color-coded table + help tooltips) ────────────────────────

# Colors for the settings panel (BGR)
COLOR_BOTH = (0, 255, 255)       # yellow - changes BOTH images
COLOR_FILTERED = (255, 255, 0)   # cyan - changes filtered only
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (160, 160, 160)
COLOR_HEADER = (100, 200, 100)   # green
COLOR_HELP_BG = (50, 50, 50)     # dark gray tooltip background
COLOR_HELP_BORDER = (180, 180, 180)
COLOR_HELP_ICON = (200, 200, 200)
COLOR_HELP_HOVER = (100, 220, 255)  # bright highlight when hovered

# Tooltip state: stores help icon hit-zones and current hover
_help_zones: list[tuple[int, int, int, str]] = []
_hover_label: list[str | None] = [None]  # mutable container for mouse callback

# Panel layout constants
_PANEL_WIDTH = 620
_LINE_HEIGHT = 22
_PAD_TOP = 10
_COL_SETTING = 10
_COL_VALUE = 230
_COL_APPLIES = 470
_HELP_RADIUS = 7
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45


def _mouse_callback(event: int, mx: int, my: int, flags: int, param: object) -> None:
    """Mouse callback for the settings panel - detect hover over '?' icons.

    Registered via cv2.setMouseCallback on the control window.

    Args:
        event: OpenCV mouse event type.
        mx:    Mouse x position.
        my:    Mouse y position.
        flags: OpenCV mouse event flags (unused, required by API).
        param: User data (unused, required by API).
    """
    if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
        for (cx, cy, r, label) in _help_zones:
            if (mx - cx) ** 2 + (my - cy) ** 2 <= (r + 4) ** 2:
                _hover_label[0] = label
                return
        _hover_label[0] = None


def _draw_help_icon(img: np.ndarray, cx: int, cy: int, radius: int, hovered: bool) -> None:
    """Draw a small circled '?' icon.

    Args:
        img:     Image to draw on (modified in place).
        cx:      Icon center x.
        cy:      Icon center y.
        radius:  Circle radius in pixels.
        hovered: Whether the icon is currently hovered.
    """
    color = COLOR_HELP_HOVER if hovered else COLOR_HELP_ICON
    cv2.circle(img, (cx, cy), radius, color, 1, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.32
    tw, th = cv2.getTextSize("?", font, fs, 1)[0]
    cv2.putText(img, "?", (cx - tw // 2, cy + th // 2),
                font, fs, color, 1, cv2.LINE_AA)


def _draw_tooltip(img: np.ndarray, text: str, anchor_x: int, anchor_y: int) -> None:
    """Draw a word-wrapped tooltip box near the given anchor position.

    Args:
        img:      Image to draw on (modified in place).
        text:     Tooltip text content.
        anchor_x: X coordinate to anchor the tooltip near.
        anchor_y: Y coordinate to anchor the tooltip near.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.38
    th = 1
    max_w = 380  # max tooltip text width before wrapping

    # Word-wrap the text
    words = text.split()
    lines: list[str] = []
    current_line = ""
    for word in words:
        test = f"{current_line} {word}".strip()
        tw = cv2.getTextSize(test, font, fs, th)[0][0]
        if tw > max_w and current_line:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test
    if current_line:
        lines.append(current_line)

    line_h = 16
    pad = 8
    box_w = max(cv2.getTextSize(line_text, font, fs, th)[0][0] for line_text in lines) + pad * 2
    box_h = line_h * len(lines) + pad * 2

    # Position tooltip to the right of anchor, clamped to image bounds
    tx = min(anchor_x + 15, img.shape[1] - box_w - 5)
    ty = max(anchor_y - box_h // 2, 5)
    if ty + box_h > img.shape[0] - 5:
        ty = img.shape[0] - box_h - 5

    # Background box with border
    cv2.rectangle(img, (tx, ty), (tx + box_w, ty + box_h), COLOR_HELP_BG, -1)
    cv2.rectangle(img, (tx, ty), (tx + box_w, ty + box_h), COLOR_HELP_BORDER, 1)

    # Text lines
    for i, line_text in enumerate(lines):
        cv2.putText(img, line_text, (tx + pad, ty + pad + line_h * (i + 1) - 3),
                    font, fs, COLOR_WHITE, th, cv2.LINE_AA)


def _draw_panel_rows(img: np.ndarray, y: int, rows: list) -> tuple[int, list]:
    """Draw setting rows with color coding and help icons onto the panel image.

    Args:
        img:  Panel image to draw on (modified in place).
        y:    Starting y position for the first row.
        rows: Output from settings_rows() (list of tuples and None separators).

    Returns:
        Tuple of (new_y position, list of help zone tuples).
    """
    help_zones: list[tuple[int, int, int, str]] = []

    for row in rows:
        if row is None:
            y += int(_LINE_HEIGHT * 0.4)
            continue
        label, value, applies = row
        color = COLOR_BOTH if applies == "BOTH" else COLOR_FILTERED
        tag = applies

        cv2.putText(img, label, (_COL_SETTING, y), _FONT, _FONT_SCALE, color, 1, cv2.LINE_AA)
        cv2.putText(img, value, (_COL_VALUE, y), _FONT, _FONT_SCALE, COLOR_WHITE, 1, cv2.LINE_AA)
        cv2.putText(img, tag, (_COL_APPLIES, y), _FONT, _FONT_SCALE, color, 1, cv2.LINE_AA)

        # Draw "?" help icon next to the label
        help_key = LABEL_TO_HELP.get(label)
        if help_key and help_key in HELP_TEXT:
            text_w = cv2.getTextSize(label, _FONT, _FONT_SCALE, 1)[0][0]
            icon_x = _COL_SETTING + text_w + 12
            icon_y = y - 5
            hovered = (_hover_label[0] == help_key)
            _draw_help_icon(img, icon_x, icon_y, _HELP_RADIUS, hovered)
            help_zones.append((icon_x, icon_y, _HELP_RADIUS, help_key))

        y += _LINE_HEIGHT

    return y, help_zones


def _draw_panel_legend(img: np.ndarray, y: int, subpixel: bool, scale: int) -> None:
    """Draw the scale note and color legend at the bottom of the panel.

    Args:
        img:      Panel image to draw on (modified in place).
        y:        Starting y position.
        subpixel: Whether subpixel mode is active (controls scale note).
        scale:    Current subpixel scale factor.
    """
    if subpixel:
        y += int(_LINE_HEIGHT * 0.3)
        cv2.putText(img, f"delta/thresh values include {scale}x subpixel scale",
                    (_COL_SETTING, y), _FONT, _FONT_SCALE, COLOR_GRAY, 1, cv2.LINE_AA)
        y += _LINE_HEIGHT
        y += _LINE_HEIGHT

    y += int(_LINE_HEIGHT * 0.5)
    cv2.line(img, (_COL_SETTING, y - 8), (_PANEL_WIDTH - 10, y - 8), COLOR_GRAY, 1)
    cv2.putText(img, "BOTH", (_COL_SETTING, y + _LINE_HEIGHT),
                _FONT, _FONT_SCALE, COLOR_BOTH, 1, cv2.LINE_AA)
    cv2.putText(img, "= changes Raw + Filtered", (_COL_SETTING + 55, y + _LINE_HEIGHT),
                _FONT, _FONT_SCALE, COLOR_GRAY, 1, cv2.LINE_AA)
    cv2.putText(img, "FILTERED", (_COL_SETTING, y + 2 * _LINE_HEIGHT),
                _FONT, _FONT_SCALE, COLOR_FILTERED, 1, cv2.LINE_AA)
    cv2.putText(img, "= changes Filtered only    [p] print  [q] quit",
                (_COL_SETTING + 85, y + 2 * _LINE_HEIGHT),
                _FONT, _FONT_SCALE, COLOR_GRAY, 1, cv2.LINE_AA)


def draw_settings_panel(stereo_vals: dict, filter_vals: dict) -> np.ndarray:
    """Render settings as a color-coded table on a black image.

    Yellow rows = setting affects BOTH raw and filtered images.
    Cyan rows   = setting affects FILTERED image only.
    Hover over the (?) icon next to a setting to see what it does.

    Args:
        stereo_vals: Dict from read_stereo_trackbars().
        filter_vals: Dict from read_filter_trackbars().

    Returns:
        BGR image of the rendered settings panel.
    """
    global _help_zones
    rows = settings_rows(stereo_vals, filter_vals)
    subpixel = stereo_vals["subpixel"]
    scale = compute_scale(subpixel, stereo_vals["subpixel_fractional_bits"])

    # Compute panel height
    extra = 1 if subpixel else 0
    total_lines = 2 + len(rows) + 3 + extra
    height = _PAD_TOP + _LINE_HEIGHT * total_lines + 10
    img = np.zeros((height, _PANEL_WIDTH, 3), dtype=np.uint8)

    # Header
    y = _PAD_TOP + _LINE_HEIGHT
    cv2.putText(img, "Setting", (_COL_SETTING, y), _FONT, _FONT_SCALE, COLOR_HEADER, 1, cv2.LINE_AA)
    cv2.putText(img, "Value", (_COL_VALUE, y), _FONT, _FONT_SCALE, COLOR_HEADER, 1, cv2.LINE_AA)
    cv2.putText(img, "Applies To", (_COL_APPLIES, y), _FONT, _FONT_SCALE, COLOR_HEADER, 1, cv2.LINE_AA)
    y += _LINE_HEIGHT
    cv2.line(img, (_COL_SETTING, y - 8), (_PANEL_WIDTH - 10, y - 8), COLOR_GRAY, 1)
    y += _LINE_HEIGHT

    # Setting rows with help icons
    y, help_zones = _draw_panel_rows(img, y, rows)

    # Scale note and legend
    _draw_panel_legend(img, y, subpixel, scale)

    # Update global help zones for mouse callback
    _help_zones = help_zones

    # Draw tooltip if hovering over a "?" icon
    if _hover_label[0] and _hover_label[0] in HELP_TEXT:
        for (cx, cy, r, lbl) in help_zones:
            if lbl == _hover_label[0]:
                _draw_tooltip(img, HELP_TEXT[lbl], cx + r, cy)
                break

    return img


# ── Pipeline setup ────────────────────────────────────────────────────────────

def _create_pipeline(args: argparse.Namespace) -> tuple:
    """Build the depthai pipeline with stereo depth and image filter nodes.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Tuple of (pipeline, filter_config_queue, stereo_config_queue,
        filter_output_queue, depth_queue).
    """
    pipeline = dai.Pipeline()

    mono_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    mono_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    out_left = mono_left.requestOutput((args.width, args.height), fps=FPS)
    out_right = mono_right.requestOutput((args.width, args.height), fps=FPS)

    depth = pipeline.create(dai.node.StereoDepth)

    # Enable runtime mode switching so subpixel/lr_check/extended can
    # be toggled at runtime via StereoDepthConfig without a restart.
    depth.setRuntimeModeSwitch(True)

    # Set initial modes from CLI args
    depth.setLeftRightCheck(args.lr_check)
    depth.setExtendedDisparity(args.extended_disparity)
    depth.setSubpixel(args.subpixel)
    if args.subpixel:
        depth.initialConfig.setSubpixelFractionalBits(args.subpixel_fractional_bits)
    if args.lr_check:
        depth.initialConfig.setLeftRightCheckThreshold(args.lrc_threshold)
    depth.inputConfig.setBlocking(False)

    out_left.link(depth.left)
    out_right.link(depth.right)

    # Host-side ImageFilters node for post-processing
    filter_node = pipeline.create(dai.node.ImageFilters)
    filter_node.setRunOnHost(True)
    filter_node.build(depth.disparity)

    filter_node.initialConfig.filterIndices = []
    filter_node.initialConfig.filterParams = [
        dai.node.ImageFilters.SpeckleFilterParams(),
        dai.node.ImageFilters.TemporalFilterParams(),
        dai.node.ImageFilters.SpatialFilterParams(),
        dai.node.ImageFilters.MedianFilterParams.MEDIAN_OFF,
    ]

    # Queues for runtime config and frame output
    filter_config_queue = filter_node.inputConfig.createInputQueue()
    stereo_config_queue = depth.inputConfig.createInputQueue()
    filter_output_queue = filter_node.output.createOutputQueue()
    depth_queue = depth.disparity.createOutputQueue()

    return pipeline, filter_config_queue, stereo_config_queue, filter_output_queue, depth_queue


def _render_disparity_view(
    raw_frame: np.ndarray,
    filtered_frame: np.ndarray,
    stereo_vals: dict,
) -> np.ndarray:
    """Render a side-by-side color-mapped disparity view with labels.

    Args:
        raw_frame:      Raw disparity frame from the device.
        filtered_frame: Filtered disparity frame from ImageFilters.
        stereo_vals:    Current stereo settings dict.

    Returns:
        BGR image with raw and filtered views side by side.
    """
    max_disp = compute_max_disparity(
        stereo_vals["subpixel"], stereo_vals["subpixel_fractional_bits"],
        stereo_vals["extended_disparity"]
    )
    norm = 255.0 / max_disp if max_disp > 0 else 1.0
    raw = (raw_frame * norm).astype(np.uint8)
    filt = (filtered_frame * norm).astype(np.uint8)

    raw_color = cv2.applyColorMap(raw, cv2.COLORMAP_JET)
    filt_color = cv2.applyColorMap(filt, cv2.COLORMAP_JET)
    combined = np.hstack([raw_color, filt_color])

    cv2.putText(combined, "Raw Disparity", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(combined, "Filtered Disparity", (raw_color.shape[1] + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Status line showing active stereo modes
    parts: list[str] = []
    if stereo_vals["subpixel"]:
        parts.append(f"subpixel({stereo_vals['subpixel_fractional_bits']}bit)")
    if stereo_vals["lr_check"]:
        parts.append(f"lr_check(thr={stereo_vals['lrc_threshold']})")
    if stereo_vals["extended_disparity"]:
        parts.append("extended")
    mode_str = "Mode: " + ", ".join(parts) if parts else "Mode: defaults"
    cv2.putText(combined, mode_str, (10, combined.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return combined


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    """Run the interactive stereo depth tuning loop.

    Args:
        args: Parsed command-line arguments.
    """
    create_trackbars(args)

    pipeline, filter_config_queue, stereo_config_queue, \
        filter_output_queue, depth_queue = _create_pipeline(args)

    with pipeline:
        pipeline.start()

        # Set up resizable display window
        cv2.namedWindow("Stereo Depth", cv2.WINDOW_NORMAL)
        display_w = int(args.width * 2 * args.display_scale)
        display_h = int(args.height * args.display_scale)
        cv2.resizeWindow("Stereo Depth", display_w, display_h)

        # Print initial mode
        print(f"Resolution: {args.width}x{args.height}")
        if args.display_scale != 1.0:
            print(f"Display scale: {args.display_scale}x ({display_w}x{display_h})")
        print("Runtime mode switching enabled - toggle stereo settings via trackbars")

        prev_stereo: dict | None = None
        prev_filter: dict | None = None
        current_scale = compute_scale(args.subpixel, args.subpixel_fractional_bits)

        while pipeline.isRunning():
            # Grab frames
            in_disparity = depth_queue.get()
            filter_frame = filter_output_queue.get()

            # Read current trackbar values
            stereo_vals = read_stereo_trackbars()
            filter_vals = read_filter_trackbars()

            # Auto-rescale filter delta/threshold when subpixel scale changes
            new_scale = compute_scale(stereo_vals["subpixel"], stereo_vals["subpixel_fractional_bits"])
            if new_scale != current_scale:
                rescale_delta_trackbars(current_scale, new_scale)
                current_scale = new_scale
                filter_vals = read_filter_trackbars()  # re-read after rescale

            # Send stereo config when any stereo setting changes
            if stereo_vals != prev_stereo:
                stereo_config_queue.send(build_stereo_config(stereo_vals))
                prev_stereo = stereo_vals.copy()

            # Send filter config when any filter setting changes
            if filter_vals != prev_filter:
                speckle, temporal, spatial, median = build_filter_params(filter_vals)
                cfg = dai.ImageFiltersConfig()
                cfg = cfg.updateFilterAtIndex(0, speckle)
                cfg = cfg.updateFilterAtIndex(1, temporal)
                cfg = cfg.updateFilterAtIndex(2, spatial)
                cfg = cfg.updateFilterAtIndex(3, median)
                filter_config_queue.send(cfg)
                prev_filter = filter_vals.copy()

            # Render and display
            combined = _render_disparity_view(
                in_disparity.getFrame(), filter_frame.getFrame(), stereo_vals
            )
            cv2.imshow("Stereo Depth", combined)

            panel = draw_settings_panel(stereo_vals, filter_vals)
            cv2.imshow(CTRL_WIN, panel)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            elif key == ord("p"):
                print("\n".join(settings_text(stereo_vals, filter_vals)))
                print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive stereo depth filter tuning tool"
    )
    d = DEFAULTS
    parser.add_argument(
        "--extended_disparity", action="store_true",
        default=bool(d["extended_disparity"]),
        help="Start with extended disparity (togglable at runtime)"
    )
    parser.add_argument(
        "--subpixel", action="store_true",
        default=bool(d["subpixel"]),
        help="Start with subpixel mode (togglable at runtime)"
    )
    parser.add_argument(
        "--subpixel_fractional_bits", type=int,
        default=d["subpixel_frac_bits"],
        help="Initial subpixel fractional bits 3-5 (adjustable at runtime)"
    )
    parser.add_argument(
        "--lr_check", action="store_true",
        default=bool(d["lr_check"]),
        help="Start with left-right check (togglable at runtime)"
    )
    parser.add_argument(
        "--lrc_threshold", type=int,
        default=d["lrc_threshold"],
        help="Initial LR check threshold 0-10 (adjustable at runtime)"
    )
    parser.add_argument(
        "--confidence", type=int,
        default=d["confidence"],
        help="Initial confidence threshold 0-255 (adjustable at runtime)"
    )
    parser.add_argument(
        "--bilateral", type=int,
        default=d["bilateral"],
        help="Initial bilateral sigma 0-250 (adjustable at runtime)"
    )
    parser.add_argument(
        "--width", type=int, default=1280,
        help="Stereo resolution width (default=1280)"
    )
    parser.add_argument(
        "--height", type=int, default=800,
        help="Stereo resolution height (default=800)"
    )
    parser.add_argument(
        "--display_scale", type=float, default=1.0,
        help="Scale factor for display window, e.g. 0.5 to halve (default=1.0)"
    )
    args = parser.parse_args()
    main(args)
