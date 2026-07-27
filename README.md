# H2P Wafer Defect Detection and GDS Subtraction

This project processes stitched microscope images of H2P wafer device cells, detects physical defects, lets a reviewer correct the detections, maps the reviewed regions into GDS coordinates, and subtracts those regions from selected GDS layers.

The Windows/PowerShell workflow is organized into three stages:

1. Align each wafer and extract its device-cell images.
2. Detect and review defects separately for each wafer.
3. Inspect matching device indices across wafers. <br>

and/or

3. Subtract the reviewed defect regions from selected GDS layers.

## Workflow

### 1. Alignment and device-image extraction

```powershell
python .\wafer_alignment_and_extraction.py -c
```

The default extraction keeps the full, unzoomed, untrimmed device-cell crop so its pixel coordinates remain aligned with the GDS.

For higher-quality, exact-boundary JPEG extraction (purely for individual device viewing and not for mask creation or defect detection):

```powershell
python .\wafer_alignment_and_extraction.py -c --bound --bound-exact-jpeg-decode --bound-workers 3
```

A batch device creation run writes every wafer into its own output directory:

```text
extracted_cells\
├─ Wafer_topcontact\
│  ├─ analysis_png\
│  ├─ metadata\
│  ├─ seam_masks\
│  └─ previews\
├─ Wafer_mesaetch\
├─ Wafer_substrateetch\
├─ Wafer_LOR1\
└─ Wafer_LOR2\
```

### 2. Defect detection and review

Run detection and review for every wafer listed in `batch_wafers.txt`:

```powershell
python .\review_batch_wafers.py --quick-review
```

The batch launcher uses `defect_detector_analysis_roi.py`. Automatic detection operates on a padded layer-5 device ROI, then inverse-maps every detected region into the original full layer-8 crop before GDS conversion.


To process only one wafer:

```powershell
python .\review_batch_wafers.py --wafer mesaetch --quick-review
```

#### Review existing detections without rerunning detection

For every wafer:

```powershell
python .\review_batch_wafers.py --no-review --review-only --quick-review
```

For one wafer:

```powershell
python .\review_batch_wafers.py --wafer mesaetch --no-review --review-only --quick-review
```

In this launcher, `--no-review` means “do not automatically add `--review-ui`.” The forwarded `--review-only` argument still opens the review-only UI.

To inspect the generated detector commands without running them:

```powershell
python .\review_batch_wafers.py --quick-review --dry-run
```

### 3. Cross-wafer device inspection

Launch the device-index defect viewer:

```powershell
python .\device_index_defect_viewer.py
```

The viewer groups the same device index across all extracted wafers and displays the available images, defect overlays, wafer labels, image dimensions, and defect counts.

To print a dataset summary without opening the UI:

```powershell
python .\device_index_defect_viewer.py --summary
```

Useful UI controls include:

```text
/               Focus device-index search
Left/Right      Previous or next device index
O               Toggle defect overlays
Mouse wheel     Zoom the selected image
Mouse drag      Pan the selected image
```

### 3. GDS subtraction

```powershell
python .\subtract_defects.py .\extracted_cells\Wafer_{name}\Wafer_{name}_device_defects.json -l num1 num2 ...
```

Example:

```powershell
python .\subtract_defects.py .\extracted_cells\Wafer_topcontact\Wafer_topcontact_device_defects.json -l 1 4
```

The reviewed-wafer stitcher writes wafer-level inspection images beside the reviewed JSON after the review UI closes.

## Detection behavior

The detector is recall-oriented. It is designed to find subtle particles, scratches, smears, holes, delamination, edge-clipped damage, and diffuse contamination while suppressing dense vertical device-line texture.

Automatic detection uses an expanded device ROI for better border coverage. The full unzoomed crop remains the coordinate reference, and reviewed regions are mapped back into that coordinate system before GDS conversion.

A human review step is still expected before GDS subtraction.

Automatic proposals and manually added regions are preserved in the final reviewed JSON. The reviewed-wafer stitch distinguishes retained automatic detections from manually added detections.

## Requirements

The project is primarily used on Windows with PowerShell.

Install the Python dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install numpy opencv-python pillow gdstk
```

`tkinter` is used by the alignment, review, and device-viewer interfaces and is normally included with the standard Windows Python installer.

## Repository layout

The main files are:

```text
wafer_alignment_and_extraction.py       Stage 1 entry point
wafer_alignment_and_extraction_old.py   Underlying extraction implementation
future_design_adapter.py                Future-design alignment adapter
future_alignment.py                     Future-design alignment helpers
centroid_algorithm.py                   Automatic wafer/cell feature alignment
coordinate_transformer.py               Pixel-to-GDS coordinate transforms
gds_parser.py                           GDS boundary and overlay parsing
illumination_stitching.py               Flat-field correction and tile stitching
large_wafer_tester.py                   Large stitched-wafer inspection helpers
wafer_align_gui.py                      Manual alignment interface
wafer_metrology.py                      Wafer geometry and metrology helpers

defect_detector.py                      Core automatic defect detector
defect_detector_analysis_roi.py         Padded analysis ROI and inverse mapping
defect_mapper_gui.py                    Fast review and labeling UI
review_batch_wafers.py                  Runs the detector separately per wafer
reviewed_defect_wafer_stitch.py         Creates reviewed wafer-level images
device_index_defect_viewer.py           Cross-wafer device-index inspection UI

subtract_defects.py                     Stage 4 GDS subtraction
batch_wafers_parser.py                  Compact and legacy batch-file parser
wafer_run_layout.py                     Per-wafer output path helpers
migrate_combined_extracted_cells.py     Migrates older combined extraction output

config.json                             Geometry and stitching configuration
batch_wafers.txt                        Compact wafer-name list
future_design.gds                       Active future-design GDS
semiconductor_design.gds                Legacy GDS used by the old workflow
```

## Configuration

```json
{
  "gds_path": "future_design.gds",
  "gds_layer": 0,
  "gds_datatype": 0,
  "tile_cols": 40,
  "tile_rows": 58,
  "tile_width": 3000,
  "tile_height": 1992,
  "overlap_x_percent": 10.0,
  "overlap_y_percent": 10.0,
  "output_image_size": 4000,
  "downscale_factor": 0.05,
  "auto_alignment_translation_correction_um": {
    "x": 0.0,
    "y": 0.0
  }
}
```

The tile-grid size is detected from filenames when extraction runs. Tile filenames must contain coordinates in this form:

```text
tile_x001_y001.jpg
tile_x001_y002.jpg
...
```

## Batch file

The preferred `batch_wafers.txt` format is one wafer/process name per line:

```text
topcontact
mesaetch
substrateetch
LOR1
LOR2
```

The corresponding tile directories should be located in the repository as:

```text
folder_of_tiles_topcontact\
folder_of_tiles_mesaetch\
folder_of_tiles_substrateetch\
folder_of_tiles_LOR1\
folder_of_tiles_LOR2\
```

Blank lines and comments beginning with `#` are ignored.

## Manual alignment controls

After pressing **START ALIGNMENT**:

- Click the full-wafer image to set the coarse absolute left and right marker positions.
- Drag inside a marker template to translate the full rectangle and square grid.
- Drag a corner handle to scale the complete template uniformly.
- Drag the circular handle to rotate the complete template around its center.
- Use **RESET SYSTEM** to restore the automatic marker positions and geometry.

## Defect-review controls

| Control | Action |
|---|---|
| Left-drag | Draw a new rectangular defect region |
| Lasso mode | Draw an irregular defect region |
| Right-click a defect | Delete it |
| `N`, Right Arrow, or Space | Next cell |
| `P` or Left Arrow | Previous cell |
| Up/Down Arrow | Jump to the cell above/below |
| `C` | Clear all annotations on the current cell |
| `X` | Toggle the current cell as excluded/damaged |
| `Ctrl+Z` | Undo |
| `Ctrl+Shift+Z` or `Ctrl+Y` | Redo |
| `L` | Toggle annotation labels |
| `K` | Copy the current view to the Windows clipboard |
| `Q` or `Esc` | Save and quit |
| `1`–`5` | Assign a class in standard typed mode only |

## Typical output for one wafer

```text
extracted_cells\
└─ Wafer_topcontact\
   ├─ analysis_png\
   ├─ metadata\
   ├─ seam_masks\
   ├─ previews\
   ├─ algo_previews\
   ├─ normal_template_auto_roi_*.npz
   ├─ algo_defects.json
   ├─ Wafer_topcontact_device_defects.json
   ├─ Wafer_topcontact_device_defects.json.review_state.json
   └─ Wafer_topcontact_reviewed_wafer\
      ├─ Wafer_topcontact_reviewed_wafer_clean.png
      ├─ Wafer_topcontact_reviewed_wafer_defects.png
      ├─ Wafer_topcontact_reviewed_wafer_outline.png
      ├─ Wafer_topcontact_reviewed_wafer_defect_mask.png
      ├─ Wafer_topcontact_reviewed_wafer_auto_mask.png
      ├─ Wafer_topcontact_reviewed_wafer_manual_mask.png
      └─ Wafer_topcontact_reviewed_wafer_report.json
```
