import codecs
import json
from pathlib import Path

import pytest

from judgekit import hashing
from judgekit.dataset import load_dataset, load_outputs
from judgekit.errors import DatasetError
from judgekit.models import CaseRecord


def _write(tmp_path: Path, name: str, lines: list[str]) -> Path:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_dataset_valid(tmp_path: Path) -> None:
    lines = [
        json.dumps({"id": "c1", "input": "q1"}),
        json.dumps({"id": "c2", "input": "q2", "reference": "r2"}),
    ]
    path = _write(tmp_path, "data.jsonl", lines)

    dataset = load_dataset(path)

    assert len(dataset.cases) == 2
    assert [c.id for c in dataset.cases] == ["c1", "c2"]
    assert all(c.dataset_version == dataset.dataset_version for c in dataset.cases)
    expected = hashing.dataset_version(
        [CaseRecord(id="c1", input="q1"), CaseRecord(id="c2", input="q2", reference="r2")]
    )
    assert dataset.dataset_version == expected


def test_load_dataset_malformed_json(tmp_path: Path) -> None:
    lines = [
        json.dumps({"id": "c1", "input": "q1"}),
        "{not valid json",
        json.dumps({"id": "c2", "input": "q2"}),
    ]
    path = _write(tmp_path, "data.jsonl", lines)

    with pytest.raises(DatasetError) as exc:
        load_dataset(path)
    assert "line 2" in str(exc.value)


def test_load_dataset_missing_required_field(tmp_path: Path) -> None:
    lines = [json.dumps({"id": "c1"})]
    path = _write(tmp_path, "data.jsonl", lines)

    with pytest.raises(DatasetError) as exc:
        load_dataset(path)
    assert "line 1" in str(exc.value)


def test_load_dataset_nested_metadata_dict(tmp_path: Path) -> None:
    lines = [json.dumps({"id": "c1", "input": "q1", "metadata": {"a": {"b": 1}}})]
    path = _write(tmp_path, "data.jsonl", lines)

    with pytest.raises(DatasetError) as exc:
        load_dataset(path)
    assert "line 1" in str(exc.value)


def test_load_dataset_json_array_line(tmp_path: Path) -> None:
    lines = [json.dumps(["not", "an", "object"])]
    path = _write(tmp_path, "data.jsonl", lines)

    with pytest.raises(DatasetError) as exc:
        load_dataset(path)
    assert "line 1" in str(exc.value)


def test_load_dataset_multiple_bad_lines(tmp_path: Path) -> None:
    lines = ["{bad", json.dumps({"id": "c1"}), json.dumps([1, 2])]
    path = _write(tmp_path, "data.jsonl", lines)

    with pytest.raises(DatasetError) as exc:
        load_dataset(path)
    message = str(exc.value)
    assert "line 1" in message
    assert "line 2" in message
    assert "line 3" in message


def test_load_dataset_rejects_dataset_version_key(tmp_path: Path) -> None:
    lines = [json.dumps({"id": "c1", "input": "q1", "dataset_version": "sha256:x"})]
    path = _write(tmp_path, "data.jsonl", lines)

    with pytest.raises(DatasetError) as exc:
        load_dataset(path)
    assert "line 1" in str(exc.value)


def test_load_dataset_duplicate_id(tmp_path: Path) -> None:
    lines = [
        json.dumps({"id": "c1", "input": "q1"}),
        json.dumps({"id": "c2", "input": "q2"}),
        json.dumps({"id": "c1", "input": "q3"}),
    ]
    path = _write(tmp_path, "data.jsonl", lines)

    with pytest.raises(DatasetError) as exc:
        load_dataset(path)
    message = str(exc.value)
    assert "line 1" in message
    assert "line 3" in message
    assert "c1" in message


def test_load_dataset_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(DatasetError):
        load_dataset(path)


def test_load_dataset_whitespace_only_file(tmp_path: Path) -> None:
    path = tmp_path / "blank.jsonl"
    path.write_text("   \n\n\t\n", encoding="utf-8")

    with pytest.raises(DatasetError):
        load_dataset(path)


def test_load_dataset_blank_interior_lines(tmp_path: Path) -> None:
    lines = [
        json.dumps({"id": "c1", "input": "q1"}),
        "",
        "   ",
        "{bad json",
    ]
    path = _write(tmp_path, "data.jsonl", lines)

    with pytest.raises(DatasetError) as exc:
        load_dataset(path)
    assert "line 4" in str(exc.value)


def test_load_dataset_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "bom.jsonl"
    content = json.dumps({"id": "c1", "input": "q1"}) + "\n"
    path.write_bytes(codecs.BOM_UTF8 + content.encode("utf-8"))

    dataset = load_dataset(path)

    assert len(dataset.cases) == 1
    assert dataset.cases[0].id == "c1"


def test_load_outputs_valid(tmp_path: Path) -> None:
    lines = [
        json.dumps({"case_id": "c1", "output": "a1"}),
        json.dumps({"case_id": "c2", "output": "a2"}),
    ]
    path = _write(tmp_path, "outputs.jsonl", lines)

    result = load_outputs(path)

    assert result == {"c1": "a1", "c2": "a2"}


def test_load_outputs_duplicate_case_id(tmp_path: Path) -> None:
    lines = [
        json.dumps({"case_id": "c1", "output": "a1"}),
        json.dumps({"case_id": "c2", "output": "a2"}),
        json.dumps({"case_id": "c1", "output": "a3"}),
    ]
    path = _write(tmp_path, "outputs.jsonl", lines)

    with pytest.raises(DatasetError) as exc:
        load_outputs(path)
    message = str(exc.value)
    assert "line 1" in message
    assert "line 3" in message
    assert "c1" in message


def test_load_outputs_ignores_extra_fields(tmp_path: Path) -> None:
    lines = [json.dumps({"case_id": "c1", "output": "a1", "extra": "ignored"})]
    path = _write(tmp_path, "outputs.jsonl", lines)

    result = load_outputs(path)

    assert result == {"c1": "a1"}


def test_load_outputs_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(DatasetError):
        load_outputs(path)


def test_load_dataset_unicode_line_separator_inside_string(tmp_path: Path) -> None:
    record = {"id": "c1", "input": "before\u2028after"}
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    dataset = load_dataset(path)

    assert dataset.cases[0].input == "before\u2028after"
