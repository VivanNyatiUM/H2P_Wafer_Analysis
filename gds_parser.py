
"""Single-design GDS API.

Kept as a small import surface for modules that historically imported gds_parser.
All implementation lives in design_geometry.py and uses the configured active GDS.
"""
from design_geometry import (
    DesignGeometry,
    cell_records,
    clean_boundary_polygon,
    get_gds_cells_list,
    get_gds_overlay_polygons,
    load_design_geometry,
    marker_records,
    overlay_polygons,
    parse_alignment_markers,
    parse_gds_wafer_boundary,
    resolve_design_path,
)

__all__ = [
    "DesignGeometry", "cell_records", "clean_boundary_polygon",
    "get_gds_cells_list", "get_gds_overlay_polygons", "load_design_geometry",
    "marker_records", "overlay_polygons", "parse_alignment_markers",
    "parse_gds_wafer_boundary", "resolve_design_path",
]
