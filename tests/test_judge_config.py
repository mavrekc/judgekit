import codecs
import json
from pathlib import Path

import pytest

from judgekit import hashing
from judgekit.errors import JudgeError
from judgekit.judge import load_judge_config
from judgekit.models import JudgeConfigRecord

MINIMAL_CONFIG: dict[str, object] = {
    "id": "example-judge",
    "provider": "anthropic",
    "model": "test-model",
    "rubric": "Rate this support reply.\n\n$input\n\nReply with JSON.",
    "labels": ["good", "bad"],
}


def _write(tmp_path: Path, name: str, payload: object) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_judge_config_valid(tmp_path: Path) -> None:
    path = _write(tmp_path, "judge.json", MINIMAL_CONFIG)

    config = load_judge_config(path)

    assert config.id == "example-judge"
    assert config.provider == "anthropic"
    assert config.model == "test-model"
    assert config.labels == ("good", "bad")
    assert config.version_hash.startswith("sha256:")
    expected = hashing.judge_config_version_hash(JudgeConfigRecord.model_validate(MINIMAL_CONFIG))
    assert config.version_hash == expected


def test_load_judge_config_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"

    with pytest.raises(JudgeError):
        load_judge_config(path)


def test_load_judge_config_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "judge.json"
    path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(JudgeError) as exc:
        load_judge_config(path)
    assert "invalid JSON" in str(exc.value)


def test_load_judge_config_top_level_array(tmp_path: Path) -> None:
    path = _write(tmp_path, "judge.json", ["not", "an", "object"])

    with pytest.raises(JudgeError) as exc:
        load_judge_config(path)
    assert "expected a JSON object" in str(exc.value)


def test_load_judge_config_unknown_key(tmp_path: Path) -> None:
    payload = {**MINIMAL_CONFIG, "unknown_field": "x"}
    path = _write(tmp_path, "judge.json", payload)

    with pytest.raises(JudgeError):
        load_judge_config(path)


def test_load_judge_config_rejects_human_id(tmp_path: Path) -> None:
    payload = {**MINIMAL_CONFIG, "id": "human"}
    path = _write(tmp_path, "judge.json", payload)

    with pytest.raises(JudgeError):
        load_judge_config(path)


def test_load_judge_config_duplicate_labels(tmp_path: Path) -> None:
    payload = {**MINIMAL_CONFIG, "labels": ["good", "good"]}
    path = _write(tmp_path, "judge.json", payload)

    with pytest.raises(JudgeError) as exc:
        load_judge_config(path)
    assert '"good"' in str(exc.value)


def test_load_judge_config_distinct_int_and_bool_labels(tmp_path: Path) -> None:
    payload = {**MINIMAL_CONFIG, "labels": [1, True]}
    path = _write(tmp_path, "judge.json", payload)

    config = load_judge_config(path)

    assert config.labels == (1, True)


def test_load_judge_config_distinct_str_and_int_labels(tmp_path: Path) -> None:
    payload = {**MINIMAL_CONFIG, "labels": ["1", 1]}
    path = _write(tmp_path, "judge.json", payload)

    config = load_judge_config(path)

    assert config.labels == ("1", 1)


def test_load_judge_config_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "judge.json"
    content = json.dumps(MINIMAL_CONFIG)
    path.write_bytes(codecs.BOM_UTF8 + content.encode("utf-8"))

    config = load_judge_config(path)

    assert config.id == "example-judge"
