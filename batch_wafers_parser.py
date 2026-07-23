"""Batch wafer definitions for the H2P extraction pipeline.

Accepted formats
----------------

Compact format (recommended):

    topcontact
    mesaetch
    substrateetch
    LOR1
    LOR2

Each compact name expands to:

    id            = Wafer_<name>
    after_folder  = folder_of_tiles_<name>
    before_folder = none
    defect_json   = none

The legacy four-line format remains supported:

    Wafer_topcontact:
    "folder_of_tiles_topcontact"
    "none"
    "none"

Blank lines and comments beginning with ``#`` are ignored.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any


_COMPACT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _logical_lines(path: Path) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            lexer = shlex.shlex(stripped, posix=True)
            lexer.whitespace_split = True
            lexer.commenters = "#"
            try:
                tokens = list(lexer)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: malformed quoting: {exc}"
                ) from exc

            if not tokens:
                continue
            if len(tokens) != 1:
                raise ValueError(
                    f"{path}:{line_number}: expected one value per line, "
                    f"got {len(tokens)} tokens"
                )
            result.append((line_number, tokens[0].strip()))
    return result


def _normalize_optional(value: str) -> str:
    cleaned = str(value).strip()
    return "none" if not cleaned or cleaned.lower() == "none" else cleaned


def _compact_record(name: str, *, path: Path, line_number: int) -> dict[str, str]:
    if not _COMPACT_NAME.fullmatch(name):
        raise ValueError(
            f"{path}:{line_number}: invalid compact wafer name {name!r}. "
            "Use letters, numbers, dots, underscores, or hyphens."
        )

    if name.lower().startswith("wafer_"):
        suffix = name[6:]
        wafer_id = name
    else:
        suffix = name
        wafer_id = f"Wafer_{name}"

    if not suffix:
        raise ValueError(
            f"{path}:{line_number}: wafer name cannot be only 'Wafer_'."
        )

    return {
        "id": wafer_id,
        "after_folder": f"folder_of_tiles_{suffix}",
        "before_folder": "none",
        "defect_json": "none",
    }


def parse_batch_file(filepath: str | Path) -> list[dict[str, str]]:
    """Parse compact names and legacy four-line wafer blocks.

    The return schema intentionally matches the historical pipeline:
    ``id``, ``after_folder``, ``before_folder``, and ``defect_json``.
    """

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Batch definitions config file not found: {filepath}"
        )
    if not path.is_file():
        raise ValueError(f"Batch path is not a file: {filepath}")

    lines = _logical_lines(path)
    if not lines:
        raise ValueError(f"Batch file contains no wafer definitions: {filepath}")

    wafers: list[dict[str, str]] = []
    i = 0
    while i < len(lines):
        line_number, value = lines[i]

        if value.endswith(":"):
            wafer_id = value[:-1].strip()
            if not wafer_id:
                raise ValueError(
                    f"{path}:{line_number}: empty wafer header."
                )
            if i + 3 >= len(lines):
                raise ValueError(
                    f"{path}:{line_number}: legacy wafer block {wafer_id!r} "
                    "requires three following lines: after folder, before "
                    "folder, and defect JSON."
                )

            block = lines[i + 1 : i + 4]
            for child_line, child_value in block:
                if child_value.endswith(":"):
                    raise ValueError(
                        f"{path}:{child_line}: encountered a new wafer header "
                        f"before legacy block {wafer_id!r} was complete."
                    )

            wafers.append(
                {
                    "id": wafer_id,
                    "after_folder": _normalize_optional(block[0][1]),
                    "before_folder": _normalize_optional(block[1][1]),
                    "defect_json": _normalize_optional(block[2][1]),
                }
            )
            i += 4
            continue

        wafers.append(
            _compact_record(
                value,
                path=path,
                line_number=line_number,
            )
        )
        i += 1

    seen: dict[str, int] = {}
    for index, wafer in enumerate(wafers, start=1):
        key = wafer["id"].casefold()
        if key in seen:
            raise ValueError(
                f"{path}: duplicate wafer id {wafer['id']!r} at definitions "
                f"{seen[key]} and {index}."
            )
        seen[key] = index

    return wafers


def compact_batch_text(wafers: list[dict[str, Any]]) -> str:
    """Return compact names when records follow the inferred folder pattern."""
    names: list[str] = []
    for wafer in wafers:
        wafer_id = str(wafer["id"])
        suffix = wafer_id[6:] if wafer_id.lower().startswith("wafer_") else wafer_id
        expected_folder = f"folder_of_tiles_{suffix}"
        if (
            str(wafer.get("after_folder", "")) != expected_folder
            or _normalize_optional(str(wafer.get("before_folder", "none"))) != "none"
            or _normalize_optional(str(wafer.get("defect_json", "none"))) != "none"
        ):
            raise ValueError(
                f"Wafer {wafer_id!r} cannot be represented losslessly in "
                "compact format."
            )
        names.append(suffix)
    return "\n".join(names) + "\n"
