# Wafer Alignment & Extraction Tool

This tool aligns raw physical wafer tiles to a GDSII design file, extracts individual devices at native resolution, and maps labeled defect coordinates to design microns ($\mu\text{m}$).

### Prerequisites
```bash
pip install numpy opencv-python pillow gdstk
```

### Directory Tree
```text
h2p_device_view/
├── config.json                       # Main configuration file (overlaps, dimensions, GDS paths)
├── manual_exclusions.json            # Tracking list for manually excluded camera tiles
├── semiconductor_design.gds          # Reference GDS CAD layout design file
├── centroid_algorithm.py             # Sub-pixel OBB snapping mathematics and feature search
├── coordinate_transformer.py         # Handles GDS/Canvas coordinate mapping & SVD solver
├── gds_parser.py                     # Reads GDS files, flattens layers, clusters cells
├── wafer_metrology.py                # Wafer segmentation, circle fitting, flat angle profiling
├── wafer_align_gui.py                # Tkinter-based manual alignment slider GUI
├── defect_mapper_gui.py              # OpenCV-based interactive defect annotation tool
├── large_wafer_tester.py             # OpenCV-based automated centroid snap interface
├── wafer_alignment_and_extraction.py # Core orchestrator and main execution script
└── subtract_defects.py               # Flat GDS boolean subtraction tool
```

### Batch Configuration (`batch_wafers.txt`)
Create a file formatted in 4-line groupings:
```text
Wafer_Sample_01:
"folder_of_tiles"
"none"
"none"
Wafer_B:
"folder_for_B"
"none"
"none"
.
.
.
```

### Command Line Interface (CLI)

#### **Stage 1: Alignment & Native Crop Extraction Into Labeling**
```bash
python wafer_alignment_and_extraction.py --batch batch_wafers.txt --manual --device -c -l
```
*Optional additions:*
*   `--manual` : Launch the interactive overlay UI to align GDS wireframes manually before cropping.
*   `--shave 10` : Crop margin buffer size inside the rotated frame to eliminate edge artifacts.
*   `--out-dir extracted_cells` : Customize the destination directory for crops.

#### **Interactive Controls:**
*   **`Mouse Drag`**: Draw a bounding box around a defect.
*   **`1` - `5` keys**: Assign a defect class (`1`: blister, `2`: tear, `3`: delamination, `4`: particulate, `5`: hole).
*   **`Space` / `Right Arrow`**: Save the current cell and advance.
*   **`Left Arrow`**: Go back one cell.
*   **`X`**: Toggle exclusion for damaged/empty devices.
*   **`Esc` / `Q`**: Save data to `{wafer_id}_device_defects.json` and export the final annotated layout to `{wafer_id}_stitched_devices.jpg`.

#### **Stage 2: Mask Generation**
Create the Subtracted Mask File
```bash
python subtract_defects.py --json Wafer_A_device_defects.json --out repaired_design.gds --layers [layers]
```
[layers] should be represented as numbers with spaces in between. Examples:
1
2 4
