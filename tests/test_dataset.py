import codecs
import json
from pathlib import Path

import pytest

from judgekit import hashing
from judgekit.dataset import load_dataset, load_outputs, load_rater, load_ratings
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


def test_load_ratings_valid_mixed_label_types(tmp_path: Path) -> None:
    lines = [
        json.dumps({"case_id": "c1", "label": "yes"}),
        json.dumps({"case_id": "c2", "label": 1}),
        json.dumps({"case_id": "c3", "label": 0.5}),
        json.dumps({"case_id": "c4", "label": True}),
    ]
    path = _write(tmp_path, "ratings.jsonl", lines)

    result = load_ratings(path)

    assert result == {"c1": "yes", "c2": 1, "c3": 0.5, "c4": True}


def test_load_ratings_duplicate_case_id(tmp_path: Path) -> None:
    lines = [
        json.dumps({"case_id": "c1", "label": "yes"}),
        json.dumps({"case_id": "c2", "label": "no"}),
        json.dumps({"case_id": "c1", "label": "no"}),
    ]
    path = _write(tmp_path, "ratings.jsonl", lines)

    with pytest.raises(DatasetError) as exc:
        load_ratings(path)
    message = str(exc.value)
    assert "line 1" in message
    assert "line 3" in message
    assert "c1" in message


def test_load_ratings_invalid_json_and_schema_error_together(tmp_path: Path) -> None:
    lines = [
        "{not valid json",
        json.dumps({"case_id": "c2"}),
        json.dumps({"case_id": "c3", "label": "yes"}),
    ]
    path = _write(tmp_path, "ratings.jsonl", lines)

    with pytest.raises(DatasetError) as exc:
        load_ratings(path)
    message = str(exc.value)
    assert "line 1" in message
    assert "line 2" in message


def test_load_ratings_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(DatasetError):
        load_ratings(path)


def test_load_ratings_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "bom.jsonl"
    content = json.dumps({"case_id": "c1", "label": "yes"}) + "\n"
    path.write_bytes(codecs.BOM_UTF8 + content.encode("utf-8"))

    result = load_ratings(path)

    assert result == {"c1": "yes"}


def test_load_ratings_rejects_nan_label(tmp_path: Path) -> None:
    lines = ['{"case_id": "c1", "label": NaN}']
    path = _write(tmp_path, "ratings.jsonl", lines)

    with pytest.raises(DatasetError) as exc:
        load_ratings(path)
    assert "line 1" in str(exc.value)


def _manifest_dict(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "kind": "judge_run",
        "schema_version": 1,
        "run_id": "run1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "dataset_path": "dataset.jsonl",
        "dataset_version": "sha256:abc",
        "judge_config_path": "config.json",
        "judge_config_id": "judge1",
        "judge_config_hash": "sha256:def",
        "provider": "anthropic",
        "model": "test-model",
        "cache_dir": ".judgekit/cache",
        "max_cost": None,
        "totals": {
            "n_cases": 2,
            "n_cached": 2,
            "n_live": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
        },
    }
    base.update(overrides)
    return base


def _verdict_dict(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "case_id": "c1",
        "label": "good",
        "input_tokens": 10,
        "output_tokens": 5,
        "cost": 0.0,
        "cached": True,
        "n_attempts": 1,
    }
    base.update(overrides)
    return base


def test_load_rater_plain_ratings_file(tmp_path: Path) -> None:
    lines = [
        json.dumps({"case_id": "c1", "label": "yes"}),
        json.dumps({"case_id": "c2", "label": "no"}),
    ]
    path = _write(tmp_path, "r1.jsonl", lines)

    rater_id, labels = load_rater(path)

    assert rater_id == path.stem
    assert labels == load_ratings(path)


def test_load_rater_plain_file_invalid_line_matches_load_ratings(tmp_path: Path) -> None:
    lines = [
        json.dumps({"case_id": "c1", "label": "yes"}),
        json.dumps({"case_id": "c2"}),
    ]
    path = _write(tmp_path, "r1.jsonl", lines)

    with pytest.raises(DatasetError) as rater_exc:
        load_rater(path)
    with pytest.raises(DatasetError) as ratings_exc:
        load_ratings(path)
    assert str(rater_exc.value) == str(ratings_exc.value)


def test_load_rater_judge_artifact(tmp_path: Path) -> None:
    lines = [
        json.dumps(_manifest_dict()),
        json.dumps(_verdict_dict(case_id="c1", label="good")),
        json.dumps(_verdict_dict(case_id="c2", label="bad")),
    ]
    path = _write(tmp_path, "judge.jsonl", lines)

    rater_id, labels = load_rater(path)

    assert rater_id == "judge1"
    assert labels == {"c1": "good", "c2": "bad"}


def test_load_rater_judge_artifact_duplicate_case_id(tmp_path: Path) -> None:
    lines = [
        json.dumps(_manifest_dict()),
        json.dumps(_verdict_dict(case_id="c1", label="good")),
        json.dumps(_verdict_dict(case_id="c1", label="bad")),
    ]
    path = _write(tmp_path, "judge.jsonl", lines)

    with pytest.raises(DatasetError) as exc:
        load_rater(path)
    message = str(exc.value)
    assert "duplicate case_id" in message
    assert "line 3" in message
    assert "first seen at line 2" in message


def test_load_rater_judge_artifact_corrupt_verdict_line(tmp_path: Path) -> None:
    lines = [
        json.dumps(_manifest_dict()),
        "{not valid json",
    ]
    path = _write(tmp_path, "judge.jsonl", lines)

    with pytest.raises(DatasetError) as exc:
        load_rater(path)
    assert "line 2" in str(exc.value)


def test_load_rater_judge_artifact_manifest_only(tmp_path: Path) -> None:
    lines = [json.dumps(_manifest_dict())]
    path = _write(tmp_path, "judge.jsonl", lines)

    with pytest.raises(DatasetError) as exc:
        load_rater(path)
    assert "no verdicts found" in str(exc.value)


def test_load_rater_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(DatasetError) as exc:
        load_rater(path)
    assert "no records found" in str(exc.value)


def test_load_rater_non_judge_run_kind_falls_through_to_plain(tmp_path: Path) -> None:
    lines = [json.dumps({"kind": "calibration_report", "case_id": "c1"})]
    path = _write(tmp_path, "ratings.jsonl", lines)

    with pytest.raises(DatasetError) as rater_exc:
        load_rater(path)
    with pytest.raises(DatasetError) as ratings_exc:
        load_ratings(path)
    assert str(rater_exc.value) == str(ratings_exc.value)
