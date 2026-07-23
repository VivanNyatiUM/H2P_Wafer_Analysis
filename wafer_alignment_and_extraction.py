#!/usr/bin/env python3
"""H2P future-design entry point.

Default behavior uses future_design.gds and the compact 3 x 4 inner fiducials.
Pass --old to execute the untouched automatic-branch pipeline.
"""

from __future__ import annotations

import sys

H2P_FUTURE_DESIGN_WRAPPER_V1 = True


def main() -> None:
    old_mode = "--old" in sys.argv[1:]
    if old_mode:
        sys.argv = [sys.argv[0], *[arg for arg in sys.argv[1:] if arg != "--old"]]

    try:
        import wafer_alignment_and_extraction_old as legacy_pipeline
    except ImportError as exc:
        raise SystemExit(
            "Missing wafer_alignment_and_extraction_old.py. Run "
            "`python .\\install_future_design_update.py` from the automatic-branch folder."
        ) from exc

    if old_mode:
        print("[H2P] --old selected: using semiconductor_design.gds and the original detector.")
    else:
        try:
            import future_design_adapter

            future_design_adapter.install(legacy_pipeline)
        except Exception as exc:
            raise SystemExit(f"Future-design initialization failed: {exc}") from exc

    legacy_pipeline.main()


if __name__ == "__main__":
    main()
