import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from judgekit.cli import app

runner = CliRunner()


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _run_id_from_stdout(stdout: str) -> str:
    line = next(line for line in stdout.splitlines() if line.startswith("run_id"))
    return line[16:].strip()


def test_happy_path(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    outputs = tmp_path / "outputs.jsonl"
    out = tmp_path / "out.jsonl"
    _write_jsonl(
        dataset,
        [
            {"id": "c1", "input": "q1", "reference": "a"},
            {"id": "c2", "input": "q2", "reference": "b"},
            {"id": "c3", "input": "q3", "reference": "c"},
            {"id": "c4", "input": "q4", "reference": "d"},
        ],
    )
    _write_jsonl(
        outputs,
        [
            {"case_id": "c1", "output": "a"},
            {"case_id": "c2", "output": "b"},
            {"case_id": "c3", "output": "c"},
            {"case_id": "c4", "output": "x"},
        ],
    )

    result = runner.invoke(
        app,
        [
            "run",
            "--dataset",
            str(dataset),
            "--outputs",
            str(outputs),
            "--scorer",
            "exact",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0
    assert out.exists()
    for key in (
        "run_id",
        "dataset_version",
        "scorer",
        "n_cases",
        "mean_score",
        "pass_rate",
        "artifact",
    ):
        assert key in result.stdout
    assert result.stdout.count("0.7500") == 2
    assert "sha256:" in result.stdout
    assert str(out) in result.stdout

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    from judgekit.models import RunManifest, RunResult

    RunManifest.model_validate(json.loads(lines[0]))
    for line in lines[1:]:
        RunResult.model_validate(json.loads(line))


def test_default_out_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = tmp_path / "dataset.jsonl"
    outputs = tmp_path / "outputs.jsonl"
    _write_jsonl(dataset, [{"id": "c1", "input": "q1", "reference": "a"}])
    _write_jsonl(outputs, [{"case_id": "c1", "output": "a"}])
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        ["run", "--dataset", str(dataset), "--outputs", str(outputs), "--scorer", "exact"],
    )

    assert result.exit_code == 0
    run_id = _run_id_from_stdout(result.stdout)
    runs_dir = tmp_path / "runs"
    files = list(runs_dir.iterdir())
    assert len(files) == 1
    assert files[0].name == f"{run_id}.jsonl"


def test_bad_dataset_malformed_json(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    outputs = tmp_path / "outputs.jsonl"
    dataset.write_text(
        json.dumps({"id": "c1", "input": "q1", "reference": "a"}) + "\n{not valid json\n",
        encoding="utf-8",
    )
    _write_jsonl(outputs, [{"case_id": "c1", "output": "a"}])

    result = runner.invoke(
        app,
        ["run", "--dataset", str(dataset), "--outputs", str(outputs), "--scorer", "exact"],
    )

    assert result.exit_code == 1
    assert "line" in result.stderr


def test_unknown_scorer(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    outputs = tmp_path / "outputs.jsonl"
    _write_jsonl(dataset, [{"id": "c1", "input": "q1", "reference": "a"}])
    _write_jsonl(outputs, [{"case_id": "c1", "output": "a"}])

    result = runner.invoke(
        app,
        ["run", "--dataset", str(dataset), "--outputs", str(outputs), "--scorer", "bogus"],
    )

    assert result.exit_code == 1
    assert "exact" in result.stderr
    assert "regex" in result.stderr
    assert "structured" in result.stderr


def test_empty_dataset_file(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    outputs = tmp_path / "outputs.jsonl"
    dataset.write_text("", encoding="utf-8")
    _write_jsonl(outputs, [{"case_id": "c1", "output": "a"}])

    result = runner.invoke(
        app,
        ["run", "--dataset", str(dataset), "--outputs", str(outputs), "--scorer", "exact"],
    )

    assert result.exit_code == 1


def test_missing_output_for_case(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    outputs = tmp_path / "outputs.jsonl"
    _write_jsonl(
        dataset,
        [
            {"id": "c1", "input": "q1", "reference": "a"},
            {"id": "c2", "input": "q2", "reference": "b"},
        ],
    )
    _write_jsonl(outputs, [{"case_id": "c1", "output": "a"}])

    result = runner.invoke(
        app,
        ["run", "--dataset", str(dataset), "--outputs", str(outputs), "--scorer", "exact"],
    )

    assert result.exit_code == 1
    assert "c2" in result.stderr


def test_extra_outputs_warns_but_succeeds(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    outputs = tmp_path / "outputs.jsonl"
    out = tmp_path / "out.jsonl"
    _write_jsonl(dataset, [{"id": "c1", "input": "q1", "reference": "a"}])
    _write_jsonl(
        outputs,
        [
            {"case_id": "c1", "output": "a"},
            {"case_id": "unknown", "output": "z"},
        ],
    )

    result = runner.invoke(
        app,
        [
            "run",
            "--dataset",
            str(dataset),
            "--outputs",
            str(outputs),
            "--scorer",
            "exact",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0
    assert "warning: 1 output(s) have no matching case id and will be ignored" in result.stderr
    assert "run_id" in result.stdout


def test_existing_out_file_refuses_overwrite(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    outputs = tmp_path / "outputs.jsonl"
    out = tmp_path / "out.jsonl"
    _write_jsonl(dataset, [{"id": "c1", "input": "q1", "reference": "a"}])
    _write_jsonl(outputs, [{"case_id": "c1", "output": "a"}])
    out.write_text("existing", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "run",
            "--dataset",
            str(dataset),
            "--outputs",
            str(outputs),
            "--scorer",
            "exact",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 1
    assert "refusing to overwrite" in result.stderr


def test_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert "judgekit 0.1.0" in result.stdout


def test_nonexistent_dataset_path(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs.jsonl"
    _write_jsonl(outputs, [{"case_id": "c1", "output": "a"}])

    result = runner.invoke(
        app,
        [
            "run",
            "--dataset",
            str(tmp_path / "missing.jsonl"),
            "--outputs",
            str(outputs),
            "--scorer",
            "exact",
        ],
    )

    assert result.exit_code == 2
