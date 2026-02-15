# Interactive Stereo Depth Filter Tuning Tool

Interactive tool for tuning DepthAI stereo depth parameters in real time. Displays raw and filtered disparity side by side with a trackbar control panel, so you can visually dial in the best settings for your scene.

**Window 1 — Stereo Depth:** Live side-by-side view of raw (unfiltered) and filtered disparity, both color-mapped with JET colormap.

![Python Depthai Stereo Preview](assets/stereo_preview.png)

**Window 2 — Filter Controls:** Trackbar sliders for every stereo and filter parameter, plus a color-coded settings readout you can copy into code. Hover over the **(?)** icon next to any setting to see a description of what it does.

![Python Depthai Stereo Settings](assets/stereo_settings.png)

- **Yellow** rows affect **both** raw and filtered images (stereo algorithm settings).
- **Cyan** rows affect the **filtered** image only (post-processing filters).

## Quick Setup for DepthAI-Core on Ubuntu

### Ubuntu DepthAI-Core Installation

```bash
sudo apt update && sudo apt install -y
git clone https://github.com/luxonis/depthai-core.git && cd depthai-core
python3 -m venv venv
source venv/bin/activate
# Installs library and requirements
python3 examples/python/install_requirements.py
echo "export OPENBLAS_CORETYPE=ARMV8" >> ~/.bashrc && source ~/.bashrc
```

### Installing Stereo Tuning Tool

```bash
cd examples/python/StereoDepth
git clone git@github.com:roboticsmick/stereo_tuning_depthai_V3.git
```

### Quick Start

```bash
# Default settings (1280x800, subpixel + LR check + extended enabled)
python3 set_stereo_depth_filters.py

# Typical tuning session — scaled to fit screen
python3 set_stereo_depth_filters.py --subpixel --lr_check --display_scale 0.5
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `p` | Print current settings to terminal (copy-paste ready) |

## Command-Line Arguments

All stereo mode flags (`--subpixel`, `--lr_check`, `--extended_disparity`) set the **initial** state only — you can toggle any of them at runtime via trackbars without restarting.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--subpixel` | flag | on | Start with subpixel mode enabled. Computes disparity at sub-pixel level for finer depth precision. |
| `--subpixel_fractional_bits` | int | 3 | Initial subpixel fractional bits (3–5). Higher values give finer precision but reduce the maximum depth range. |
| `--lr_check` | flag | on | Start with left-right consistency check enabled. Validates depth by comparing L→R and R→L disparity maps. |
| `--lrc_threshold` | int | 10 | Initial LR check threshold (0–10). Maximum allowed disparity difference between L-R and R-L checks. Lower = stricter. |
| `--extended_disparity` | flag | on | Start with extended disparity mode. Doubles the search range (96→192) to detect very close objects (<35 cm). |
| `--confidence` | int | 15 | Initial confidence threshold (0–255). Minimum stereo matching confidence; higher values discard uncertain estimates. |
| `--bilateral` | int | 0 | Initial bilateral filter sigma (0–250). Edge-preserving smoothing strength. 0 = disabled. |
| `--width` | int | 1280 | Stereo input resolution width. |
| `--height` | int | 800 | Stereo input resolution height. |
| `--display_scale` | float | 1.0 | Scale factor for the display window, e.g. `0.5` to halve. Useful when the side-by-side view is too wide for your monitor. |

## Usage Examples

```bash
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
```

## Stereo Algorithm Settings

These settings control the stereo matching algorithm itself and affect **both** the raw and filtered disparity outputs.

### Subpixel Mode

Computes disparity at sub-pixel level for finer depth precision. Especially useful for long-range measurements where integer-pixel disparity gives coarse depth steps.

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Subpixel | 0–1 (toggle) | 1 (on) | Enable/disable subpixel disparity computation. |
| Subpixel Frac Bits | 3–5 | 3 | Number of fractional bits for subpixel accuracy. Higher = finer precision but reduces maximum representable disparity range. |

When subpixel is enabled, all disparity values (and filter delta/threshold trackbars) are internally scaled by `2^fractional_bits`. The tool auto-rescales filter trackbars when you change subpixel mode or fractional bits.

### Left-Right Consistency Check

Validates depth by computing disparity in both left-to-right and right-to-left directions and comparing results. Pixels where the two disagree beyond the threshold are invalidated. This effectively removes incorrect matches and reveals occlusion regions.

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| LR Check | 0–1 (toggle) | 1 (on) | Enable/disable left-right consistency check. |
| LRC Threshold | 0–10 | 10 | Maximum allowed disparity difference (in pixels) between the L→R and R→L passes. Lower values are stricter — fewer false matches but more holes. |

### Extended Disparity

Doubles the disparity search range from 96 to 192 by combining results from full-resolution and downscaled image pairs. This allows detecting objects much closer to the camera (down to ~20 cm depending on baseline) at the cost of some additional compute.

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Extended Disparity | 0–1 (toggle) | 1 (on) | Enable/disable extended disparity range. |

### Confidence Threshold

Sets the minimum stereo matching confidence required to accept a disparity value. The stereo algorithm produces a confidence score (0–255) for each pixel. Pixels below this threshold are marked as invalid.

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Confidence Thresh | 0–255 | 15 | Minimum confidence. Higher values are more aggressive at filtering uncertain depth, producing cleaner maps with more holes. |

### Bilateral Filter

An edge-preserving smoothing filter applied to the disparity map. Reduces noise while keeping depth edges sharp. Applied on-device before post-processing filters.

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Bilateral Sigma | 0–250 | 0 | Smoothing strength. 0 = disabled. Higher values produce smoother depth but may blur fine details. |

## Post-Processing Filters

These filters run on the host and affect only the **filtered** disparity output (right side of the display). They are applied in this order: Speckle → Temporal → Spatial → Median.

### Speckle Filter

Removes speckle noise — small isolated regions with high disparity variance that appear as noisy blobs in the depth map. Works by examining connected regions of similar disparity and removing those that are too small.

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Speckle Enable | 0–1 (toggle) | 1 (on) | Enable/disable speckle filtering. |
| Speckle Range | 0–200 | 200 | Maximum speckle region size in pixels. Connected regions smaller than this value are removed as noise. |
| Speckle Diff Thresh | 0–250 | 2 (×scale) | Maximum disparity difference between neighboring pixels within a valid region. Pixels exceeding this are classified as speckle noise. When subpixel is enabled, the trackbar value includes the subpixel scale factor. |

### Temporal Filter

Smooths depth over time using previous frames to improve stability and fill transient holes. Best suited for static or slowly-changing scenes. Adds some latency because it relies on temporal history.

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Temporal Enable | 0–1 (toggle) | 0 (off) | Enable/disable temporal filtering. Disabled by default in the ROBOTICS preset. |
| Temporal Persist | 0–8 (mode index) | 0 | Persistency mode controlling how many recent valid frames are required to accept a depth value. Higher persistence = more gap-filling but more ghosting on moving objects. |
| Temporal Alpha ×100 | 0–100 | 0 | Smoothing factor (0.00–1.00). Higher values give more weight to the current frame and less temporal smoothing. |
| Temporal Delta | 0–250 | 0 (×scale) | Validity threshold. Depth changes exceeding this delta between frames trigger re-evaluation of the pixel value. |

**Persistency modes** (Temporal Persist trackbar values):

| Index | Mode | Description |
|-------|------|-------------|
| 0 | PERSISTENCY_OFF | No persistence — each frame is independent. |
| 1 | VALID_8_OUT_OF_8 | Require all 8 of the last 8 frames to be valid. Most strict. |
| 2 | VALID_2_IN_LAST_3 | Require 2 valid frames in the last 3. |
| 3 | VALID_2_IN_LAST_4 | Require 2 valid frames in the last 4. |
| 4 | VALID_2_OUT_OF_8 | Require 2 valid frames in the last 8. |
| 5 | VALID_1_IN_LAST_2 | Require 1 valid frame in the last 2. |
| 6 | VALID_1_IN_LAST_5 | Require 1 valid frame in the last 5. |
| 7 | VALID_1_IN_LAST_8 | Require 1 valid frame in the last 8. |
| 8 | PERSISTENCY_INDEFINITELY | Once a pixel gets a valid depth, keep it indefinitely. Most aggressive gap-filling. |

### Spatial Filter

Edge-preserving spatial filter that fills invalid depth pixels using valid neighbors. Performs alternating horizontal and vertical passes to smooth the depth map while preserving sharp depth edges.

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Spatial Enable | 0–1 (toggle) | 1 (on) | Enable/disable spatial filtering. |
| Spatial Hole Radius | 0–16 | 2 | Maximum search radius (in pixels) for finding valid pixels to fill holes. Larger values fill bigger gaps but may reduce edge accuracy. |
| Spatial Alpha ×100 | 0–100 | 50 | Smoothing strength (0.00–1.00). Higher values produce stronger smoothing but less edge preservation. |
| Spatial Delta | 0–250 | 20 (×scale) | Edge threshold. Neighboring pixels with a disparity difference above this are not smoothed together, preserving depth discontinuities. |
| Spatial Iterations | 1–5 | 1 | Number of horizontal + vertical filter passes. More iterations produce stronger smoothing. |

### Median Filter

Fast noise reduction that replaces each pixel with the median of its neighbors. Effective at removing salt-and-pepper noise while preserving edges. The device runs a 7×7 median internally; this host-side filter supports up to 5×5.

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| Median | 0 = Off | 2 (5×5) | Select median filter kernel size. |
| | 1 = 3×3 | | Smaller kernel, less smoothing, faster. |
| | 2 = 5×5 | | Larger kernel, more smoothing. |

## Subpixel Scaling

When subpixel mode is enabled, disparity values are represented with fractional precision. This means the raw disparity values are scaled by `2^fractional_bits` (e.g., 8× with 3 fractional bits). The filter delta and threshold parameters are similarly scaled.

The tool handles this automatically:
- When you toggle subpixel or change fractional bits, all delta/threshold trackbars are rescaled proportionally.
- The settings panel shows the scaled values (as sent to the hardware).
- A note at the bottom of the panel reminds you of the current scale factor.

## Architecture

The tool uses two DepthAI nodes:

1. **StereoDepth** (runs on device) — Computes raw disparity from the stereo camera pair. Stereo settings (subpixel, LR check, extended disparity, confidence, bilateral) are sent as runtime config updates.

2. **ImageFilters** (runs on host) — Applies post-processing filters (speckle, temporal, spatial, median) to the raw disparity. Filter parameters are sent as runtime config updates.

Runtime mode switching is enabled (`setRuntimeModeSwitch(True)`), allowing subpixel, LR check, and extended disparity to be toggled without restarting the pipeline.

## References

- [StereoDepth Node Documentation](https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes/stereo_depth/)
- [Configuring Stereo Depth](https://docs.luxonis.com/hardware/platform/depth/configuring-stereo-depth/)
- [StereoDepthConfig Message](https://docs.luxonis.com/software-v3/depthai/depthai-components/messages/stereo_depth_config/)
- [Depth Post-Processing Filters](https://docs.luxonis.com/projects/api/en/latest/samples/StereoDepth/depth_post_processing/)
