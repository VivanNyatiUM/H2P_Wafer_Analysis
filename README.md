# H2P Wafer Defect Detection and GDS Subtraction

This project processes stitched microscope images of H2P wafer device cells, detects physical defects, lets a reviewer correct the detections, maps the reviewed regions into GDS coordinates, and subtracts those regions from selected GDS layers.

The intended workflow is (PowerShell-based):

Alignment and Device Image Creation:
```powershell
python .\wafer_alignment_and_extraction.py -c
```

Detection and Labeling (skipping detection with --no-clean, and optionally --use-design-mask, which may help reduce false positives, though unlikely):
```powershell
python .\defect_detector.py --clean --review-ui --quick-review
```
```powershell
python .\defect_detector.py --no-clean --review-ui --quick-review
```

Mask Creation (e.g. num1 = 1, num2 = 4):
```powershell
python .\subtract_defects.py .\Wafer_A_device_defects.json -l num1 num2 ...
```

The detector is intentionally recall-oriented. It is designed to find subtle particles, scratches, smears, holes, delamination, edge-clipped damage, and diffuse contamination while suppressing the wafer's dense vertical device-line texture. A human review step is still expected before GDS subtraction.

## Requirements

The project is primarily used on Windows with PowerShell.

Install the Python dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install numpy opencv-python pillow gdstk
```

`tkinter` is used by the alignment GUI and is normally included with the standard Windows Python installer.

## Repository layout

The main files are:

```text
wafer_alignment_and_extraction.py   Stage 1: align wafer and extract cells
centroid_algorithm.py               Automatic wafer/cell feature alignment
coordinate_transformer.py           Pixel-to-GDS coordinate transforms
gds_parser.py                       GDS boundary and overlay parsing
illumination_stitching.py           Flat-field correction and tile stitching
large_wafer_tester.py               Large stitched-wafer inspection helpers
wafer_align_gui.py                  Manual alignment interface
wafer_metrology.py                  Wafer geometry and metrology helpers

defect_detector.py                 Stage 2: automatic defect proposals
defect_mapper_gui.py                Fast review and labeling UI

subtract_defects.py                 Stage 3: subtract reviewed masks from GDS
config.json                         Project geometry and stitching configuration
batch_wafers.txt                    Wafer input definitions
```

## Configuration

A minimal `config.json` looks like:

```json
{
  "gds_path": "semiconductor_design.gds",
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
    "x": 10.0,
    "y": -20.0
  }
}
```

The tile grid size is detected from filenames when extraction runs. Tile filenames must contain coordinates in this form:

```text
tile_x1_y1.tif
tile_x2_y1.tif
...
```

### Batch file

`batch_wafers.txt` uses four lines per wafer:

```text
Wafer_A:
C:\path\to\Wafer_A\after_tiles
none
none
```

The fields are:

1. Wafer ID followed by `:`
2. Microscope tile folder
3. Optional before-image folder; currently unused by the simplified workflow
4. Optional previous summary/defect JSON used for extraction parameter overrides; use `none` when not needed

Comments beginning with `#` and blank lines are ignored.

### UI controls

| Control | Action |
|---|---|
| Left-drag | Draw a new defect box |
| Right-click a defect | Delete it |
| `N`, Right Arrow, or Space | Next cell |
| `P` or Left Arrow | Previous cell |
| Up/Down Arrow | Jump to the cell above/below |
| `C` | Clear all annotations on the current cell |
| `X` | Toggle current cell as excluded/damaged |
| `Ctrl+Z` | Undo |
| `Ctrl+Shift+Z` or `Ctrl+Y` | Redo |
| `L` | Toggle annotation labels |
| `K` | Copy current view to the Windows clipboard |
| `Q` or `Esc` | Save and quit |
| `1`–`5` | Assign class in standard typed mode only |

Typed-mode classes are:

```text
1 blister
2 tear
3 delamination
4 particulate
5 hole
```
