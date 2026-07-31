#!/usr/bin/env python3
"""Remove disconnected (hanging) grid-line pieces from a flat GDSII file.

The script reproduces the useful effect of this KLayout workflow:

1. Merge the target layer conceptually (for connectivity only).
2. Delete shapes fully contained in the left 80% of each cell.
3. Delete shapes fully contained in the right 80% of each cell.

Equivalently, only connected components that span from the leftmost 20% to the
rightmost 20% of a cell are retained.  The output is deliberately conservative:
it copies every retained GDS record byte-for-byte and only omits target-layer
BOUNDARY elements belonging to disconnected components.  It never creates,
merges, expands, clips, or shunts geometry.

Cell locations are inferred from the regular vertical finger array on a reference
layer (layer 7 by default).  This matches the supplied wafer GDS.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from shapely.geometry import Polygon
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires Shapely. Install it with: python -m pip install shapely"
    ) from exc

try:
    from shapely.validation import make_valid as _make_valid
except ImportError:  # Shapely < 1.8 fallback
    _make_valid = None


BOUNDARY = 0x08
LAYER = 0x0D
DATATYPE = 0x0E
XY = 0x10
ENDEL = 0x11


@dataclass
class Boundary:
    element_index: int
    layer: int
    datatype: int
    points: list[tuple[int, int]]
    raw_records: list[bytes]

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return min(xs), min(ys), max(xs), max(ys)


@dataclass
class Cell:
    row: int
    column: int
    center_x: float
    center_y: float
    window: tuple[float, float, float, float]
    reference_line_count: int


@dataclass
class ParsedGDS:
    chunks: list[bytes | Boundary]
    boundaries: list[Boundary]


def _decode_int2(data: bytes) -> int:
    if len(data) != 2:
        raise ValueError(f"Expected 2 bytes for GDS INT2, got {len(data)}")
    return struct.unpack(">h", data)[0]


def _decode_xy(data: bytes) -> list[tuple[int, int]]:
    if len(data) % 8:
        raise ValueError("Malformed GDS XY record length")
    values = struct.unpack(">" + "i" * (len(data) // 4), data)
    return list(zip(values[::2], values[1::2]))


def parse_flat_gds(path: Path) -> ParsedGDS:
    """Parse records while retaining exact bytes for lossless filtering.

    The supplied file is flat and contains BOUNDARY elements only.  Other record
    types are still copied unchanged.  A clear error is raised if a targetable
    BOUNDARY lacks layer or XY data.
    """
    chunks: list[bytes | Boundary] = []
    boundaries: list[Boundary] = []
    current_records: list[bytes] | None = None
    current_layer: int | None = None
    current_datatype = 0
    current_points: list[tuple[int, int]] | None = None

    with path.open("rb") as stream:
        while True:
            header = stream.read(4)
            if not header:
                break
            if len(header) != 4:
                raise ValueError("Truncated GDS record header")
            length, record_type, _data_type = struct.unpack(">HBB", header)
            if length < 4:
                raise ValueError(f"Invalid GDS record length: {length}")
            payload = stream.read(length - 4)
            if len(payload) != length - 4:
                raise ValueError("Truncated GDS record payload")
            raw = header + payload

            if current_records is None:
                if record_type == BOUNDARY:
                    current_records = [raw]
                    current_layer = None
                    current_datatype = 0
                    current_points = None
                else:
                    chunks.append(raw)
                continue

            current_records.append(raw)
            if record_type == LAYER:
                current_layer = _decode_int2(payload)
            elif record_type == DATATYPE:
                current_datatype = _decode_int2(payload)
            elif record_type == XY:
                current_points = _decode_xy(payload)
            elif record_type == ENDEL:
                if current_layer is None or current_points is None:
                    raise ValueError("BOUNDARY element is missing LAYER or XY")
                boundary = Boundary(
                    element_index=len(boundaries),
                    layer=current_layer,
                    datatype=current_datatype,
                    points=current_points,
                    raw_records=current_records,
                )
                boundaries.append(boundary)
                chunks.append(boundary)
                current_records = None

    if current_records is not None:
        raise ValueError("Truncated GDS: BOUNDARY without ENDEL")
    return ParsedGDS(chunks=chunks, boundaries=boundaries)


def _mode(values: Iterable[int]) -> int:
    counts = collections.Counter(values)
    if not counts:
        raise ValueError("Cannot calculate mode of an empty sequence")
    return counts.most_common(1)[0][0]


def _positive_median(values: Iterable[float]) -> float:
    positive = sorted(v for v in values if v > 0)
    if not positive:
        raise ValueError("Could not infer a positive pitch")
    middle = len(positive) // 2
    if len(positive) % 2:
        return float(positive[middle])
    return 0.5 * (positive[middle - 1] + positive[middle])


def infer_cells(boundaries: list[Boundary], reference_layer: int) -> tuple[list[Cell], float, float]:
    """Infer cell centers from the modal vertical rectangles on reference_layer."""
    refs = [b for b in boundaries if b.layer == reference_layer]
    if not refs:
        raise ValueError(f"No BOUNDARY elements found on reference layer {reference_layer}")

    sizes = []
    for boundary in refs:
        x0, y0, x1, y1 = boundary.bounds
        sizes.append((x1 - x0, y1 - y0))

    modal_width = _mode(width for width, _height in sizes)
    modal_height = _mode(height for _width, height in sizes)
    fingers = [
        boundary
        for boundary in refs
        if (boundary.bounds[2] - boundary.bounds[0]) == modal_width
        and (boundary.bounds[3] - boundary.bounds[1]) == modal_height
        and modal_height > modal_width
    ]
    if len(fingers) < 4:
        raise ValueError(
            f"Reference layer {reference_layer} does not contain enough regular vertical fingers"
        )

    rows: dict[tuple[int, int], list[Boundary]] = collections.defaultdict(list)
    for finger in fingers:
        _x0, y0, _x1, y1 = finger.bounds
        rows[(y0, y1)].append(finger)

    provisional: list[tuple[float, float, int, int]] = []
    row_centers = []
    for row_number, ((y0, y1), row_fingers) in enumerate(sorted(rows.items()), start=1):
        ordered = sorted(row_fingers, key=lambda b: 0.5 * (b.bounds[0] + b.bounds[2]))
        centers = [0.5 * (b.bounds[0] + b.bounds[2]) for b in ordered]
        differences = [b - a for a, b in zip(centers, centers[1:])]
        # Most differences are the finger pitch; the larger differences are cell gaps.
        finger_pitch = _mode(int(round(value)) for value in differences if value > 0)
        split_gap = 3.0 * finger_pitch

        groups: list[list[Boundary]] = []
        group: list[Boundary] = []
        previous_center: float | None = None
        for boundary, center in zip(ordered, centers):
            if previous_center is not None and center - previous_center > split_gap:
                groups.append(group)
                group = []
            group.append(boundary)
            previous_center = center
        if group:
            groups.append(group)

        cy = 0.5 * (y0 + y1)
        row_centers.append(cy)
        for column_number, group in enumerate(groups, start=1):
            first_center = 0.5 * (group[0].bounds[0] + group[0].bounds[2])
            last_center = 0.5 * (group[-1].bounds[0] + group[-1].bounds[2])
            cx = 0.5 * (first_center + last_center)
            provisional.append((cx, cy, row_number, column_number))

    unique_x = sorted(set(cx for cx, _cy, _row, _column in provisional))
    unique_y = sorted(set(cy for _cx, cy, _row, _column in provisional))
    x_pitch = _positive_median(b - a for a, b in zip(unique_x, unique_x[1:]))
    y_pitch = _positive_median(b - a for a, b in zip(unique_y, unique_y[1:]))

    finger_counts: dict[tuple[float, float], int] = collections.Counter()
    for (y0, y1), row_fingers in rows.items():
        ordered = sorted(row_fingers, key=lambda b: 0.5 * (b.bounds[0] + b.bounds[2]))
        centers = [0.5 * (b.bounds[0] + b.bounds[2]) for b in ordered]
        differences = [b - a for a, b in zip(centers, centers[1:])]
        finger_pitch = _mode(int(round(value)) for value in differences if value > 0)
        split_gap = 3.0 * finger_pitch
        group: list[Boundary] = []
        previous_center = None
        for boundary, center in zip(ordered, centers):
            if previous_center is not None and center - previous_center > split_gap:
                first = 0.5 * (group[0].bounds[0] + group[0].bounds[2])
                last = 0.5 * (group[-1].bounds[0] + group[-1].bounds[2])
                finger_counts[(0.5 * (first + last), 0.5 * (y0 + y1))] = len(group)
                group = []
            group.append(boundary)
            previous_center = center
        if group:
            first = 0.5 * (group[0].bounds[0] + group[0].bounds[2])
            last = 0.5 * (group[-1].bounds[0] + group[-1].bounds[2])
            finger_counts[(0.5 * (first + last), 0.5 * (y0 + y1))] = len(group)

    cells = []
    for cx, cy, row, column in provisional:
        cells.append(
            Cell(
                row=row,
                column=column,
                center_x=cx,
                center_y=cy,
                window=(
                    cx - 0.5 * x_pitch,
                    cy - 0.5 * y_pitch,
                    cx + 0.5 * x_pitch,
                    cy + 0.5 * y_pitch,
                ),
                reference_line_count=finger_counts[(cx, cy)],
            )
        )
    return cells, x_pitch, y_pitch


def _shape(boundary: Boundary):
    points = boundary.points
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    geometry = Polygon(points)
    if geometry.is_valid:
        return geometry
    if _make_valid is not None:
        geometry = _make_valid(geometry)
    else:
        geometry = geometry.buffer(0)
    if geometry.is_empty:
        raise ValueError(f"Boundary {boundary.element_index} became empty during validation")
    return geometry


def _union_find_components(geometries: list) -> list[list[int]]:
    """Return touching/overlapping connected components.

    A bounding-box sweep keeps this portable across Shapely 1.x and 2.x and is
    fast enough for the few hundred polygons present in a device cell.
    """
    count = len(geometries)
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    bounds = [geometry.bounds for geometry in geometries]
    order = sorted(range(count), key=lambda index: bounds[index][0])
    for order_position, i in enumerate(order):
        min_x_i, min_y_i, max_x_i, max_y_i = bounds[i]
        for j in order[order_position + 1 :]:
            min_x_j, min_y_j, max_x_j, max_y_j = bounds[j]
            if min_x_j > max_x_i:
                break
            if max_y_j < min_y_i or min_y_j > max_y_i:
                continue
            if geometries[i].intersects(geometries[j]):
                join(i, j)

    components: dict[int, list[int]] = collections.defaultdict(list)
    for index in range(count):
        components[find(index)].append(index)
    return list(components.values())


def find_hanging_boundaries(
    boundaries: list[Boundary],
    cells: list[Cell],
    target_layer: int,
    x_pitch: float,
    y_pitch: float,
    span_fraction: float,
) -> tuple[set[int], list[dict], int]:
    if not 0.5 < span_fraction <= 1.0:
        raise ValueError("span_fraction must be greater than 0.5 and at most 1.0")

    assigned: dict[int, list[Boundary]] = collections.defaultdict(list)
    outside_count = 0

    for boundary in boundaries:
        if boundary.layer != target_layer:
            continue
        x0, y0, x1, y1 = boundary.bounds
        width = x1 - x0
        height = y1 - y0
        # Wafer frames and other large layer geometry are not device-cell grid pieces.
        if width > 1.2 * x_pitch or height > 1.2 * y_pitch:
            outside_count += 1
            continue
        center_x = 0.5 * (x0 + x1)
        center_y = 0.5 * (y0 + y1)
        matches = [
            index
            for index, cell in enumerate(cells)
            if cell.window[0] <= center_x < cell.window[2]
            and cell.window[1] <= center_y < cell.window[3]
        ]
        if len(matches) == 1:
            assigned[matches[0]].append(boundary)
        else:
            outside_count += 1

    removed: set[int] = set()
    reports: list[dict] = []
    margin_fraction = 0.5 * (1.0 - span_fraction)

    for cell_index, cell in enumerate(cells):
        cell_boundaries = assigned.get(cell_index, [])
        geometries = [_shape(boundary) for boundary in cell_boundaries]
        components = _union_find_components(geometries)

        x0, _y0, x1, _y1 = cell.window
        left_threshold = x0 + margin_fraction * (x1 - x0)
        right_threshold = x1 - margin_fraction * (x1 - x0)

        component_reports = []
        spanning_components = 0
        for component in components:
            component_bounds = [geometries[index].bounds for index in component]
            min_x = min(bounds[0] for bounds in component_bounds)
            min_y = min(bounds[1] for bounds in component_bounds)
            max_x = max(bounds[2] for bounds in component_bounds)
            max_y = max(bounds[3] for bounds in component_bounds)
            spans = min_x <= left_threshold and max_x >= right_threshold
            if spans:
                spanning_components += 1
            else:
                removed.update(cell_boundaries[index].element_index for index in component)
            component_reports.append(
                {
                    "boundary_count": len(component),
                    "spans_cell": spans,
                    "bounds_dbu": [min_x, min_y, max_x, max_y],
                }
            )

        # Safety valve: never erase all target geometry from a populated cell.
        if cell_boundaries and spanning_components == 0:
            candidates = []
            for component in components:
                component_bounds = [geometries[index].bounds for index in component]
                min_x = min(bounds[0] for bounds in component_bounds)
                max_x = max(bounds[2] for bounds in component_bounds)
                candidates.append((max_x - min_x, component))
            _span, fallback = max(candidates, key=lambda item: item[0])
            for index in fallback:
                removed.discard(cell_boundaries[index].element_index)
            spanning_components = 1
            fallback_used = True
        else:
            fallback_used = False

        reports.append(
            {
                "row": cell.row,
                "column": cell.column,
                "center_dbu": [cell.center_x, cell.center_y],
                "reference_line_count": cell.reference_line_count,
                "target_boundary_count": len(cell_boundaries),
                "connected_component_count": len(components),
                "removed_boundary_count": sum(
                    1 for boundary in cell_boundaries if boundary.element_index in removed
                ),
                "kept_spanning_component_count": spanning_components,
                "fallback_used": fallback_used,
                "components": component_reports,
            }
        )

    return removed, reports, outside_count


def write_filtered_gds(parsed: ParsedGDS, output_path: Path, removed: set[int]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        for chunk in parsed.chunks:
            if isinstance(chunk, bytes):
                stream.write(chunk)
            elif chunk.element_index not in removed:
                for raw_record in chunk.raw_records:
                    stream.write(raw_record)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove disconnected target-layer grid pieces while preserving every "
            "retained GDS boundary byte-for-byte."
        )
    )
    parser.add_argument("input_gds", type=Path)
    parser.add_argument("output_gds", type=Path)
    parser.add_argument("--target-layer", type=int, default=3)
    parser.add_argument("--reference-layer", type=int, default=7)
    parser.add_argument(
        "--span-fraction",
        type=float,
        default=0.60,
        help=(
            "Fraction of the cell width a connected component must span. "
            "0.60 exactly matches the overlapping left-80%%/right-80%% deletion trick."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path (default: <output>_cleanup_report.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.input_gds.exists():
        raise FileNotFoundError(args.input_gds)
    if args.input_gds.resolve() == args.output_gds.resolve():
        raise ValueError("Input and output paths must be different")

    parsed = parse_flat_gds(args.input_gds)
    cells, x_pitch, y_pitch = infer_cells(parsed.boundaries, args.reference_layer)
    removed, cell_reports, outside_count = find_hanging_boundaries(
        parsed.boundaries,
        cells,
        target_layer=args.target_layer,
        x_pitch=x_pitch,
        y_pitch=y_pitch,
        span_fraction=args.span_fraction,
    )
    write_filtered_gds(parsed, args.output_gds, removed)

    report_path = args.report or args.output_gds.with_name(
        f"{args.output_gds.stem}_cleanup_report.json"
    )
    report = {
        "input_gds": str(args.input_gds),
        "output_gds": str(args.output_gds),
        "target_layer": args.target_layer,
        "reference_layer": args.reference_layer,
        "span_fraction": args.span_fraction,
        "cell_count": len(cells),
        "cell_pitch_dbu": [x_pitch, y_pitch],
        "removed_boundary_count": len(removed),
        "unchanged_target_boundaries_outside_cells": outside_count,
        "cells_with_removals": sum(
            1 for report_entry in cell_reports if report_entry["removed_boundary_count"] > 0
        ),
        "fallback_cell_count": sum(1 for report_entry in cell_reports if report_entry["fallback_used"]),
        "cells": cell_reports,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Detected {len(cells)} cells from layer {args.reference_layer}.")
    print(f"Cell pitch: x={x_pitch:g} DBU, y={y_pitch:g} DBU.")
    print(f"Removed {len(removed)} hanging layer-{args.target_layer} BOUNDARY elements.")
    print(f"Preserved {outside_count} layer-{args.target_layer} boundaries outside device cells.")
    print(f"Output: {args.output_gds}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
