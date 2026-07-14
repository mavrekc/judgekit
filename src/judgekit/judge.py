"""Judge config loading and validation."""

import json
from pathlib import Path

from pydantic import ValidationError

from judgekit import hashing
from judgekit.errors import JudgeError
from judgekit.models import JudgeConfig, JudgeConfigRecord, Label
from judgekit.stats import category


def _validation_detail(exc: ValidationError) -> str:
    first = exc.errors()[0]
    loc = ".".join(str(part) for part in first["loc"])
    return f"{loc}: {first['msg']}" if loc else first["msg"]


def load_judge_config(path: Path) -> JudgeConfig:
    """Load a single-JSON-object judge config file into a version-hashed JudgeConfig."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise JudgeError(f"{path}: {exc}") from exc

    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JudgeError(f"{path}: invalid JSON: {exc.msg}") from exc

    if not isinstance(value, dict):
        raise JudgeError(f"{path}: expected a JSON object, got {type(value).__name__}")

    try:
        record = JudgeConfigRecord.model_validate(value)
    except ValidationError as exc:
        raise JudgeError(f"{path}: {_validation_detail(exc)}") from exc

    dupes = _duplicate_label_categories(record.labels)
    if dupes:
        raise JudgeError(f"{path}: duplicate labels: {', '.join(dupes)}")

    version_hash = hashing.judge_config_version_hash(record)
    return JudgeConfig(**record.model_dump(), version_hash=version_hash)


def _duplicate_label_categories(labels: tuple[Label, ...]) -> list[str]:
    """Return the sorted category strings of labels that appear more than once."""
    seen: set[str] = set()
    dupes: set[str] = set()
    for label in labels:
        cat = category(label)
        if cat in seen:
            dupes.add(cat)
        seen.add(cat)
    return sorted(dupes)
