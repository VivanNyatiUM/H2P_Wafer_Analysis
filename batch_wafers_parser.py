"""Strict parser for H2P batch wafer definitions.

The active syntax is a sequence of two-line records::

    name: topcontact_Wafer_1
    path: "C:\\data\\folder_of_tiles_topcontact"

or a grouped folder-of-folders record::

    folder_name: Wafer_3
    path: "C:\\data\\wafer_3_tile_folders"

Blank lines and full-line comments beginning with ``#`` are ignored.  Every
other line must participate in an exact ``name/folder_name`` + ``path`` pair.
The parser stops at the first error and reports the physical line number.

``folder_name`` records expand one level: every direct child directory becomes
one wafer whose id is ``<folder_name>_<child-directory-name>``.  The parent may
contain directories only, and those direct child directories may not contain
subdirectories.  Files inside a child directory are intentionally not checked.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable, NoReturn


_DECLARATION_RE = re.compile(r"^(name|folder_name):[ \t]*(.*)$")
_PATH_RE = re.compile(r"^path:[ \t]*(.*)$")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_WAFER_TOKEN_RE = re.compile(r"(?:^|_)wafer_", re.IGNORECASE)
_INVALID_NAME_CHARS_RE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


class BatchWafersFormatError(ValueError):
    """Raised when ``batch_wafers.txt`` is malformed or semantically invalid."""


def _fail(path: Path, line_number: int | None, message: str) -> "NoReturn":
    location = f"{path}:{line_number}" if line_number is not None else str(path)
    raise BatchWafersFormatError(f"{location}: {message}")


def _logical_lines(path: Path) -> list[tuple[int, str]]:
    """Return nonblank, noncomment physical lines without altering their text."""
    result: list[tuple[int, str]] = []
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise OSError(f"Could not read batch definitions file {path}: {exc}") from exc

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        result.append((line_number, stripped))
    return result


def _parse_value(
    raw_value: str,
    *,
    path: Path,
    line_number: int,
    field_name: str,
) -> str:
    value = raw_value.strip()
    if not value:
        _fail(path, line_number, f"{field_name} cannot be empty.")

    starts_quote = value[:1] in {'"', "'"}
    ends_quote = value[-1:] in {'"', "'"}
    if starts_quote or ends_quote:
        if len(value) < 2 or value[0] != value[-1] or value[0] not in {'"', "'"}:
            _fail(path, line_number, f"{field_name} has unmatched or mixed quotes.")
        value = value[1:-1]
        if not value:
            _fail(path, line_number, f"{field_name} cannot be empty.")

    return value


def _validate_name(value: str, *, path: Path, line_number: int, field_name: str) -> str:
    if value in {".", ".."}:
        _fail(path, line_number, f"{field_name} cannot be {value!r}.")
    invalid = _INVALID_NAME_CHARS_RE.search(value)
    if invalid:
        _fail(
            path,
            line_number,
            f"{field_name} contains invalid filename character {invalid.group(0)!r}.",
        )
    if value.endswith((" ", ".")):
        _fail(path, line_number, f"{field_name} cannot end with a space or period.")
    return value


def _resolve_directory(raw_path: str, *, batch_path: Path, line_number: int) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(raw_path))
    candidate = Path(expanded)

    # On Windows, pathlib running on Windows recognizes drive-letter paths.
    # The regex also prevents such a value from being joined to the batch-file
    # directory when this parser is unit-tested on another operating system.
    is_windows_absolute = bool(_WINDOWS_ABSOLUTE_RE.match(expanded))
    if not candidate.is_absolute() and not is_windows_absolute:
        candidate = batch_path.parent / candidate

    candidate = candidate.resolve(strict=False)
    if not candidate.exists():
        _fail(batch_path, line_number, f"path does not exist: {candidate}")
    if not candidate.is_dir():
        _fail(batch_path, line_number, f"path is not a directory: {candidate}")
    return candidate


def _normalize_direct_id(name: str) -> str:
    """Keep explicit Wafer-labelled names; prefix plain legacy-style names."""
    return name if _WAFER_TOKEN_RE.search(name) else f"Wafer_{name}"


def _base_record(
    *,
    wafer_id: str,
    declared_name: str,
    tile_path: Path,
    source_type: str,
    declaration_line: int,
    folder_group: str = "",
    folder_group_path: Path | None = None,
    tile_subfolder: str = "",
) -> dict[str, str]:
    resolved = str(tile_path)
    return {
        "id": wafer_id,
        "name": declared_name,
        "path": resolved,
        # Historical consumers use after_folder.  Keeping this key means the
        # extraction/review/subtraction pipeline does not need a parallel schema.
        "after_folder": resolved,
        "before_folder": "none",
        "defect_json": "none",
        "source_type": source_type,
        "folder_group": folder_group,
        "folder_group_path": str(folder_group_path) if folder_group_path else "",
        "tile_subfolder": tile_subfolder,
        "declaration_line": str(declaration_line),
    }


def _expand_folder_group(
    group_name: str,
    parent: Path,
    *,
    batch_path: Path,
    declaration_line: int,
    path_line: int,
) -> list[dict[str, str]]:
    try:
        children = sorted(parent.iterdir(), key=lambda item: item.name.casefold())
    except OSError as exc:
        _fail(batch_path, path_line, f"could not inspect folder_name path {parent}: {exc}")
    if not children:
        _fail(
            batch_path,
            path_line,
            f"folder_name path contains no wafer folders: {parent}",
        )

    non_directories = [item.name for item in children if not item.is_dir()]
    if non_directories:
        first = non_directories[0]
        _fail(
            batch_path,
            path_line,
            "folder_name paths may contain direct child directories only; "
            f"found file {first!r} in {parent}",
        )

    records: list[dict[str, str]] = []
    for child in children:
        _validate_name(
            child.name,
            path=batch_path,
            line_number=path_line,
            field_name="wafer child folder name",
        )
        try:
            nested = sorted(
                (item for item in child.iterdir() if item.is_dir()),
                key=lambda item: item.name.casefold(),
            )
        except OSError as exc:
            _fail(batch_path, path_line, f"could not inspect wafer folder {child}: {exc}")
        if nested:
            _fail(
                batch_path,
                path_line,
                "each direct child of a folder_name path must be one flat folder "
                f"of tiles; {child} contains subfolder {nested[0].name!r}",
            )

        wafer_id = f"{group_name}_{child.name}"
        records.append(
            _base_record(
                wafer_id=wafer_id,
                declared_name=wafer_id,
                tile_path=child.resolve(strict=False),
                source_type="folder_name",
                declaration_line=declaration_line,
                folder_group=group_name,
                folder_group_path=parent,
                tile_subfolder=child.name,
            )
        )
    return records


def parse_batch_file(filepath: str | Path) -> list[dict[str, str]]:
    """Parse and validate a strict H2P batch file.

    Validation completes before the caller receives any records, so downstream
    processing cannot partially run after a malformed declaration.
    """
    path = Path(filepath).expanduser().resolve(strict=False)
    if not path.exists():
        raise FileNotFoundError(f"Batch definitions config file not found: {path}")
    if not path.is_file():
        raise BatchWafersFormatError(f"Batch path is not a file: {path}")

    lines = _logical_lines(path)
    if not lines:
        raise BatchWafersFormatError(f"Batch file contains no wafer definitions: {path}")

    records: list[dict[str, str]] = []
    declared_groups: dict[str, int] = {}
    i = 0
    while i < len(lines):
        declaration_line, declaration_text = lines[i]
        match = _DECLARATION_RE.fullmatch(declaration_text)
        if match is None:
            _fail(
                path,
                declaration_line,
                "expected exactly 'name: <value>' or 'folder_name: <value>'.",
            )

        declaration_type = match.group(1)
        declared_name = _parse_value(
            match.group(2),
            path=path,
            line_number=declaration_line,
            field_name=declaration_type,
        )
        declared_name = _validate_name(
            declared_name,
            path=path,
            line_number=declaration_line,
            field_name=declaration_type,
        )
        if declaration_type == "folder_name":
            group_key = declared_name.casefold()
            if group_key in declared_groups:
                _fail(
                    path,
                    declaration_line,
                    f"duplicate folder_name {declared_name!r}; the first declaration "
                    f"is on line {declared_groups[group_key]}.",
                )
            declared_groups[group_key] = declaration_line

        if i + 1 >= len(lines):
            _fail(
                path,
                declaration_line,
                f"{declaration_type} declaration must be followed by 'path: <directory>'.",
            )

        path_line, path_text = lines[i + 1]
        path_match = _PATH_RE.fullmatch(path_text)
        if path_match is None:
            _fail(
                path,
                path_line,
                f"expected exactly 'path: <directory>' after {declaration_type} "
                f"declared on line {declaration_line}.",
            )

        raw_directory = _parse_value(
            path_match.group(1),
            path=path,
            line_number=path_line,
            field_name="path",
        )
        directory = _resolve_directory(raw_directory, batch_path=path, line_number=path_line)

        if declaration_type == "name":
            records.append(
                _base_record(
                    wafer_id=_normalize_direct_id(declared_name),
                    declared_name=declared_name,
                    tile_path=directory,
                    source_type="name",
                    declaration_line=declaration_line,
                )
            )
        else:
            records.extend(
                _expand_folder_group(
                    declared_name,
                    directory,
                    batch_path=path,
                    declaration_line=declaration_line,
                    path_line=path_line,
                )
            )
        i += 2

    seen: dict[str, tuple[str, str]] = {}
    for record in records:
        key = record["id"].casefold()
        if key in seen:
            prior_id, prior_line = seen[key]
            _fail(
                path,
                int(record["declaration_line"]),
                f"duplicate wafer id {record['id']!r}; it conflicts with "
                f"{prior_id!r} declared on line {prior_line}.",
            )
        seen[key] = (record["id"], record["declaration_line"])

    return records


def compact_batch_text(wafers: Iterable[dict[str, Any]]) -> str:
    """Serialize parsed records into the strict two-line syntax.

    Expanded records from the same ``folder_name`` declaration are collapsed
    back to one group record.  Direct records are emitted individually.
    """
    lines: list[str] = []
    emitted_groups: set[tuple[str, str]] = set()

    for wafer in wafers:
        source_type = str(wafer.get("source_type", "name"))
        if source_type == "folder_name":
            group = str(wafer.get("folder_group", "")).strip()
            group_path = str(wafer.get("folder_group_path", "")).strip()
            if not group or not group_path:
                raise ValueError("folder_name record is missing folder_group metadata")
            key = (group.casefold(), os.path.normcase(group_path))
            if key in emitted_groups:
                continue
            emitted_groups.add(key)
            lines.extend((f"folder_name: {group}", f'path: "{group_path}"', ""))
            continue

        declared_name = str(wafer.get("name") or wafer.get("id") or "").strip()
        tile_path = str(wafer.get("path") or wafer.get("after_folder") or "").strip()
        if not declared_name or not tile_path:
            raise ValueError("name record is missing a name or path")
        lines.extend((f"name: {declared_name}", f'path: "{tile_path}"', ""))

    if lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"

