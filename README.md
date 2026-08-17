# H2P Wafer Defect Detection and GDS Subtraction

This project processes stitched microscope images of H2P wafer device cells, detects physical defects, lets a reviewer correct the detections, maps the reviewed regions into GDS coordinates, and subtracts those regions from selected GDS layers.



## Batch file

Each active entry in `batch_wafers.txt` is an exact two-line `name`/`path` or `folder_name`/`path` pair:

```text
name: mesaetch
path: C:\Users\reala\Documents\code\h2p_device_view_temp\wafer_n_mesaetch

name: substrateetch
path: C:\Users\reala\Documents\code\h2p_device_view_temp\wafer_n_substrateetch

folder_name: LORs
path: C:\Users\reala\Documents\code\h2p_device_view_temp\LOR
```

`name` points to one folder of tiles. Plain names such as `mesaetch` are output as `Wafer_mesaetch`; names that already contain a `Wafer_` token are kept as written. The tile path may be outside the repository.

`folder_name` points to a folder containing one direct child folder per wafer. The parent may contain only directories, and those child folders may contain files but no subfolders. Each wafer is named `{folder_name}_{child_folder}` (for example, `LORs_2-LOR`). Image extensions and image counts are not restricted.

Blank lines and full-line comments beginning with `#` are ignored. Every other line must follow the declaration/path pairing above; if not, the parser reports the first bad physical line and stops before processing any wafers.

The Windows/PowerShell workflow is organized into three stages:

1. Align each wafer and extract its device-cell images.
2. Detect and review defects separately for each wafer.
3. Inspect matching device indices across wafers. <br>

and/or

3. Subtract the reviewed defect regions from selected GDS layers.

## Workflow

If no names are given, the program defaults to all wafers in the batch file.

Otherwise, the arguments `--wafer {wafer_name}` and `--folder {folder_name}` can be used to select specific wafers or folders of wafers and can be used in succession in the same command line input (for all stages aside from the device index viewer).

Using `--dry-run` basically acts as a runtime test.

### 1. Alignment and device-image extraction

```powershell
python .\wafer_alignment_and_extraction.py -c
```


The default extraction keeps the full, unzoomed, untrimmed device-cell crop so its pixel coordinates remain aligned with the GDS.

### 2. Defect detection and review

Run detection and review for every wafer listed in `batch_wafers.txt`:

```powershell
python .\review_batch_wafers.py --quick-review
```

Removing `--quick-review` forces the user to select the type of defect.

#### Review existing detections without rerunning detection

```powershell
python .\review_batch_wafers.py --no-review --review-only --quick-review
```

Note that `--no-review` means “do not automatically add `--review-ui`.” The forwarded `--review-only` argument still opens the review-only UI.

### 3. Cross-wafer device inspection

Launch the device-index defect viewer:

```powershell
python .\device_index_defect_viewer.py
```

The viewer groups the same device index across all extracted wafers and displays the available images, defect overlays, wafer labels, image dimensions, and defect counts. If `folder_name` entries are present in `batch_wafers.txt`, a **Folder group** dropdown can limit the view to one declared group; **All folders** shows everything.

To print a dataset summary without opening the UI:

```powershell
python .\device_index_defect_viewer.py --summary
```

### 3. GDS subtraction

```powershell
python .\subtract_defects.py -l num1 num2 ...
```

Subtracts the defects from the combined layers of num1, num2, etc.

Example:

```powershell
python .\subtract_defects.py -l 3
```

## Requirements

The project is primarily used on Windows with PowerShell.

Install the Python dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install numpy opencv-python pillow gdstk shapely
```

`tkinter` is used by the alignment, review, and device-viewer interfaces and is normally included with the standard Windows Python installer.

## Repository layout

The main files are:

```text
1.  wafer_alignment_and_extraction.py       Stage 1 entry point
2.  wafer_alignment_and_extraction_old.py   Underlying extraction implementation
3.  future_design_adapter.py                Future-design alignment adapter
4.  future_alignment.py                     Future-design alignment helpers
5.  alignment_marker_ui_upgrade.py          Interactive alignment-marker manipulation UI
6.  cell_boundary_alignment.py              Exact device-boundary alignment helpers
7.  centroid_algorithm.py                   Automatic wafer/cell feature alignment
8.  coordinate_transformer.py               Pixel-to-GDS coordinate transforms
9.  gds_parser.py                           GDS boundary and overlay parsing
10. illumination_stitching.py               Flat-field correction and tile stitching
11. large_wafer_tester.py                   Large stitched-wafer inspection helpers
12. wafer_align_gui.py                      Manual alignment interface
13. wafer_metrology.py                      Wafer geometry and metrology helpers

14. defect_detector.py                      Core automatic defect detector
15. defect_detector_analysis_roi.py         Padded analysis ROI and inverse mapping
16. defect_mapper_gui.py                    Fast review and labeling UI
17. review_batch_wafers.py                  Runs the detector separately per wafer
18. reviewed_defect_wafer_stitch.py         Creates reviewed wafer-level images
19. device_index_defect_viewer.py           Cross-wafer device-index inspection UI

20. subtract_defects.py                     Stage 3 GDS subtraction
21. remove_hanging_gridlines.py             Removes floating grid line parts
22. batch_wafers_parser.py                  Strict path-based batch-file parser
23. wafer_run_layout.py                     Per-wafer output path helpers
24. h2p_progress.py                         Shared command-line progress reporting
25. migrate_combined_extracted_cells.py     Migrates older combined extraction output

26. config.json                             Geometry and stitching configuration
27. batch_wafers.txt                        Wafer-name/path definitions
28. future_design.gds                       Active future-design GDS
29. semiconductor_design.gds                Legacy GDS used by the old workflow

30. assets\h2pLogo.png                      H2P logo used by the UIs
```

## Configuration
Should be changed to semiconductor_design.gds if using the legacy pipeline.

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

## Detection behavior

The detector is recall-oriented. It is designed to find subtle particles, scratches, smears, holes, delamination, edge-clipped damage, and diffuse contamination while suppressing dense vertical device-line texture.

Automatic detection uses an expanded device ROI for better border coverage. The full unzoomed crop remains the coordinate reference, and reviewed regions are mapped back into that coordinate system before GDS conversion.

A human review step is still expected before GDS subtraction.

Automatic proposals and manually added regions are preserved in the final reviewed JSON. The reviewed-wafer stitch distinguishes retained automatic detections from manually added detections.
