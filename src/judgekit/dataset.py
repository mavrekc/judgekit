"""JSONL loaders for datasets and system outputs."""

import json
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ValidationError

from judgekit import hashing
from judgekit.errors import DatasetError
from judgekit.models import Case, CaseRecord, Dataset, Label, OutputRecord, RatingRecord


def _read_lines(path: Path) -> list[tuple[int, str]]:
    """Read non-blank lines from a JSONL file, keeping original 1-based line numbers."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise DatasetError(f"{path}: {exc}") from exc
    # split("\n"), not splitlines(): U+2028 and friends are legal inside JSON strings.
    return [(n, line) for n, line in enumerate(text.split("\n"), start=1) if line.strip()]


def _validation_detail(exc: ValidationError) -> str:
    first = exc.errors()[0]
    loc = ".".join(str(part) for part in first["loc"])
    return f"{loc}: {first['msg']}" if loc else first["msg"]


def _load_records[M: BaseModel](
    path: Path,
    model: type[M],
    key_fn: Callable[[M], str],
    key_label: str,
) -> list[M]:
    errors: list[tuple[int, str]] = []
    seen: dict[str, int] = {}
    records: list[M] = []
    for line_no, raw in _read_lines(path):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append((line_no, f"invalid JSON: {exc.msg}"))
            continue
        if not isinstance(value, dict):
            errors.append((line_no, f"expected a JSON object, got {type(value).__name__}"))
            continue
        try:
            record = model.model_validate(value)
        except ValidationError as exc:
            errors.append((line_no, _validation_detail(exc)))
            continue
        key = key_fn(record)
        if key in seen:
            errors.append(
                (line_no, f'duplicate {key_label} "{key}" (first seen at line {seen[key]})')
            )
            continue
        seen[key] = line_no
        records.append(record)
    if errors:
        body = "\n".join(f"line {n}: {msg}" for n, msg in sorted(errors, key=lambda e: e[0]))
        raise DatasetError(f"{path}\n{body}")
    if not records:
        raise DatasetError(f"{path}: no records found")
    return records


def load_dataset(path: Path) -> Dataset:
    """Load a JSONL dataset file into a content-versioned Dataset."""
    records = _load_records(path, CaseRecord, lambda r: r.id, "id")
    version = hashing.dataset_version(records)
    cases = tuple(Case(**record.model_dump(), dataset_version=version) for record in records)
    return Dataset(dataset_version=version, cases=cases)


def load_outputs(path: Path) -> dict[str, str]:
    """Load a JSONL outputs file into a case_id -> output mapping."""
    records = _load_records(path, OutputRecord, lambda r: r.case_id, "case_id")
    return {record.case_id: record.output for record in records}


def load_ratings(path: Path) -> dict[str, Label]:
    """Load a JSONL ratings file into a case_id -> label mapping."""
    records = _load_records(path, RatingRecord, lambda r: r.case_id, "case_id")
    return {record.case_id: record.label for record in records}
