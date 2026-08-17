r"""Strict path-based batch wafer parser.

Supported records are exactly::

    name: mesaetch
    path: C:\data\wafer_n_mesaetch

or::

    folder_name: LORs
    path: C:\data\LOR

Canonical wafer IDs:
- ``name: mesaetch`` -> ``Wafer_mesaetch``
- ``name: Wafer_mesaetch`` -> ``Wafer_mesaetch``
- ``folder_name: LORs`` child ``2-LOR`` -> ``LORs_2-LOR``

Blank lines and full-line ``#`` comments are ignored. No legacy compact or
four-line batch formats are supported.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class BatchEntry:
    id: str
    after_folder: str
    before_folder: str = "none"
    defect_json: str = "none"
    source_kind: str = "name"
    source_group: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "id": self.id,
            "after_folder": self.after_folder,
            "before_folder": self.before_folder,
            "defect_json": self.defect_json,
            "source_kind": self.source_kind,
            "source_group": self.source_group,
        }


def _meaningful_lines(lines: Iterable[str]) -> list[tuple[int, str]]:
    output: list[tuple[int, str]] = []
    for line_number, raw in enumerate(lines, start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        output.append((line_number, text))
    return output


def _split_field(line_number: int, text: str) -> tuple[str, str]:
    if ":" not in text:
        raise ValueError(f"batch_wafers.txt line {line_number}: expected 'key: value'")
    key, value = text.split(":", 1)
    key = key.strip().lower()
    value = value.strip()
    if not value:
        raise ValueError(f"batch_wafers.txt line {line_number}: {key!r} has no value")
    return key, value


def _validate_leaf_directory(path: Path, *, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label}: directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label}: expected a directory: {path}")
    nested_dirs = [child for child in path.iterdir() if child.is_dir()]
    if nested_dirs:
        names = ", ".join(child.name for child in nested_dirs[:5])
        raise ValueError(f"{label}: wafer directory must not contain subdirectories: {names}")


def _canonical_named_wafer_id(name: str) -> str:
    cleaned = str(name).strip()
    if not cleaned:
        raise ValueError("wafer name cannot be empty")
    if cleaned.casefold().startswith("wafer_"):
        return cleaned
    return f"Wafer_{cleaned}"


def parse_batch_file(path: str | Path = "batch_wafers.txt") -> list[dict[str, str | None]]:
    batch_path = Path(path)
    if not batch_path.exists():
        raise FileNotFoundError(batch_path)

    lines = _meaningful_lines(batch_path.read_text(encoding="utf-8-sig").splitlines())
    if len(lines) % 2:
        line_number, text = lines[-1]
        raise ValueError(
            f"batch_wafers.txt line {line_number}: declaration {text!r} is missing its following path"
        )

    entries: list[BatchEntry] = []
    seen_ids: set[str] = set()

    for index in range(0, len(lines), 2):
        declaration_line, declaration_text = lines[index]
        path_line, path_text = lines[index + 1]
        kind, name = _split_field(declaration_line, declaration_text)
        path_key, path_value = _split_field(path_line, path_text)

        if kind not in {"name", "folder_name"}:
            raise ValueError(
                f"batch_wafers.txt line {declaration_line}: expected 'name:' or 'folder_name:', got {kind!r}"
            )
        if path_key != "path":
            raise ValueError(
                f"batch_wafers.txt line {path_line}: expected 'path:' after {kind}:, got {path_key!r}"
            )

        source = Path(path_value).expanduser()

        if kind == "name":
            _validate_leaf_directory(source, label=name)
            candidates = [
                BatchEntry(
                    id=_canonical_named_wafer_id(name),
                    after_folder=str(source),
                    source_kind="name",
                )
            ]
        else:
            if not source.exists():
                raise FileNotFoundError(f"{name}: group directory does not exist: {source}")
            if not source.is_dir():
                raise NotADirectoryError(f"{name}: expected a group directory: {source}")

            loose_files = [child.name for child in source.iterdir() if child.is_file()]
            if loose_files:
                raise ValueError(
                    f"{name}: folder_name parent may contain only directories; found file {loose_files[0]!r}"
                )

            children = sorted(
                (child for child in source.iterdir() if child.is_dir()),
                key=lambda p: p.name.casefold(),
            )
            if not children:
                raise ValueError(f"{name}: group directory contains no wafer subdirectories: {source}")

            candidates = []
            for child in children:
                _validate_leaf_directory(child, label=f"{name}/{child.name}")
                candidates.append(
                    BatchEntry(
                        id=f"{name}_{child.name}",
                        after_folder=str(child),
                        source_kind="folder_name",
                        source_group=name,
                    )
                )

        for entry in candidates:
            key = entry.id.casefold()
            if key in seen_ids:
                raise ValueError(f"Duplicate wafer id produced by batch file: {entry.id!r}")
            seen_ids.add(key)
            entries.append(entry)

    if not entries:
        raise ValueError("batch_wafers.txt contains no wafer entries")

    return [entry.as_dict() for entry in entries]

