import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from judgekit import stats
from judgekit.cli import app

runner = CliRunner()


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _run_id_from_stdout(stdout: str) -> str:
    line = next(line for line in stdout.splitlines() if line.startswith("run_id"))
    return line[16:].strip()


def _report_id_from_stdout(stdout: str) -> str:
    line = next(line for line in stdout.splitlines() if line.startswith("report_id"))
    return line[16:].strip()


def _line(stdout: str, key: str) -> str:
    return next(line for line in stdout.splitlines() if line.startswith(key))


def _stat_value(line: str) -> tuple[float, float, float, int]:
    value_str, rest = line[16:].strip().split(" [", 1)
    bounds, n_part = rest.split("] (n=", 1)
    low_str, high_str = bounds.split(", ")
    return float(value_str), float(low_str), float(high_str), int(n_part.rstrip(")"))


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
    assert "judgekit 0.2.0" in result.stdout


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


def test_calibrate_happy_path_single_rater(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    ratings = tmp_path / "r1.jsonl"
    out = tmp_path / "report.json"
    _write_jsonl(
        dataset,
        [
            {"id": "c1", "input": "q1", "human_label": "yes"},
            {"id": "c2", "input": "q2", "human_label": "yes"},
            {"id": "c3", "input": "q3", "human_label": "no"},
            {"id": "c4", "input": "q4", "human_label": "no"},
        ],
    )
    _write_jsonl(
        ratings,
        [
            {"case_id": "c1", "label": "yes"},
            {"case_id": "c2", "label": "no"},
            {"case_id": "c3", "label": "no"},
            {"case_id": "c4", "label": "no"},
        ],
    )

    result = runner.invoke(
        app,
        [
            "calibrate",
            "--dataset",
            str(dataset),
            "--ratings",
            str(ratings),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0
    for key in (
        "report_id",
        "dataset_version",
        "level",
        "n_cases",
        "n_labeled",
        "kappa",
        "alpha",
        "artifact",
    ):
        assert key in result.stdout
    assert str(out) in result.stdout

    kappa_value, kappa_low, kappa_high, kappa_n = _stat_value(_line(result.stdout, "kappa"))
    alpha_value, alpha_low, alpha_high, alpha_n = _stat_value(_line(result.stdout, "alpha"))

    expected_kappa = stats.cohen_kappa(["yes", "yes", "no", "no"], ["yes", "no", "no", "no"])
    expected_alpha = stats.krippendorff_alpha(
        [["yes", "yes"], ["yes", "no"], ["no", "no"], ["no", "no"]], "nominal"
    )
    assert kappa_value == pytest.approx(expected_kappa, abs=1e-4)
    assert kappa_n == 4
    assert kappa_low <= kappa_value <= kappa_high
    assert alpha_value == pytest.approx(expected_alpha, abs=1e-4)
    assert alpha_n == 4
    assert alpha_low <= alpha_value <= alpha_high

    from judgekit.models import CalibrationReport

    report = CalibrationReport.model_validate_json(out.read_text(encoding="utf-8"))
    assert report.kappa is not None
    assert report.kappa.n_used == 4
    assert report.alpha.n_used == 4


def test_calibrate_multi_rater_has_no_kappa(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    r1 = tmp_path / "r1.jsonl"
    r2 = tmp_path / "r2.jsonl"
    out = tmp_path / "report.json"
    _write_jsonl(
        dataset,
        [
            {"id": "c1", "input": "q1", "human_label": "yes"},
            {"id": "c2", "input": "q2", "human_label": "no"},
        ],
    )
    _write_jsonl(r1, [{"case_id": "c1", "label": "yes"}, {"case_id": "c2", "label": "no"}])
    _write_jsonl(r2, [{"case_id": "c1", "label": "yes"}, {"case_id": "c2", "label": "yes"}])

    result = runner.invoke(
        app,
        [
            "calibrate",
            "--dataset",
            str(dataset),
            "--ratings",
            str(r1),
            "--ratings",
            str(r2),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0
    assert "kappa" not in result.stdout
    assert "alpha" in result.stdout

    from judgekit.models import CalibrationReport

    report = CalibrationReport.model_validate_json(out.read_text(encoding="utf-8"))
    assert report.kappa is None


def test_calibrate_custom_out_path_refuses_overwrite(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    ratings = tmp_path / "r1.jsonl"
    out = tmp_path / "custom" / "report.json"
    _write_jsonl(
        dataset,
        [
            {"id": "c1", "input": "q1", "human_label": "yes"},
            {"id": "c2", "input": "q2", "human_label": "no"},
        ],
    )
    _write_jsonl(ratings, [{"case_id": "c1", "label": "yes"}, {"case_id": "c2", "label": "no"}])
    args = ["calibrate", "--dataset", str(dataset), "--ratings", str(ratings), "--out", str(out)]

    first = runner.invoke(app, args)
    assert first.exit_code == 0
    assert out.exists()

    second = runner.invoke(app, args)
    assert second.exit_code == 1
    assert "refusing to overwrite" in second.stderr


def test_calibrate_default_out_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = tmp_path / "dataset.jsonl"
    ratings = tmp_path / "r1.jsonl"
    _write_jsonl(
        dataset,
        [
            {"id": "c1", "input": "q1", "human_label": "yes"},
            {"id": "c2", "input": "q2", "human_label": "no"},
        ],
    )
    _write_jsonl(ratings, [{"case_id": "c1", "label": "yes"}, {"case_id": "c2", "label": "no"}])
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["calibrate", "--dataset", str(dataset), "--ratings", str(ratings)])

    assert result.exit_code == 0
    report_id = _report_id_from_stdout(result.stdout)
    reports_dir = tmp_path / "reports"
    files = list(reports_dir.iterdir())
    assert len(files) == 1
    assert files[0].name == f"{report_id}.json"


def test_calibrate_unknown_case_id_warns(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    ratings = tmp_path / "r1.jsonl"
    out = tmp_path / "report.json"
    _write_jsonl(
        dataset,
        [
            {"id": "c1", "input": "q1", "human_label": "yes"},
            {"id": "c2", "input": "q2", "human_label": "no"},
        ],
    )
    _write_jsonl(
        ratings,
        [
            {"case_id": "c1", "label": "yes"},
            {"case_id": "c2", "label": "no"},
            {"case_id": "unknown", "label": "yes"},
        ],
    )

    result = runner.invoke(
        app,
        ["calibrate", "--dataset", str(dataset), "--ratings", str(ratings), "--out", str(out)],
    )

    assert result.exit_code == 0
    assert (
        "warning: 1 rating(s) from r1 have no matching case id and will be ignored" in result.stderr
    )


def test_calibrate_reproducible_same_seed(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    ratings = tmp_path / "r1.jsonl"
    out1 = tmp_path / "report1.json"
    out2 = tmp_path / "report2.json"
    _write_jsonl(
        dataset,
        [
            {"id": "c1", "input": "q1", "human_label": "yes"},
            {"id": "c2", "input": "q2", "human_label": "yes"},
            {"id": "c3", "input": "q3", "human_label": "no"},
            {"id": "c4", "input": "q4", "human_label": "no"},
        ],
    )
    _write_jsonl(
        ratings,
        [
            {"case_id": "c1", "label": "yes"},
            {"case_id": "c2", "label": "no"},
            {"case_id": "c3", "label": "no"},
            {"case_id": "c4", "label": "no"},
        ],
    )

    first = runner.invoke(
        app, ["calibrate", "--dataset", str(dataset), "--ratings", str(ratings), "--out", str(out1)]
    )
    second = runner.invoke(
        app, ["calibrate", "--dataset", str(dataset), "--ratings", str(ratings), "--out", str(out2)]
    )

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert _line(first.stdout, "kappa") == _line(second.stdout, "kappa")
    assert _line(first.stdout, "alpha") == _line(second.stdout, "alpha")


def test_calibrate_ordinal_level_accepts_int_labels(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    ratings = tmp_path / "r1.jsonl"
    out = tmp_path / "report.json"
    _write_jsonl(
        dataset,
        [
            {"id": "c1", "input": "q1", "human_label": 1},
            {"id": "c2", "input": "q2", "human_label": 2},
            {"id": "c3", "input": "q3", "human_label": 3},
            {"id": "c4", "input": "q4", "human_label": 4},
        ],
    )
    _write_jsonl(
        ratings,
        [
            {"case_id": "c1", "label": 1},
            {"case_id": "c2", "label": 2},
            {"case_id": "c3", "label": 2},
            {"case_id": "c4", "label": 4},
        ],
    )

    result = runner.invoke(
        app,
        [
            "calibrate",
            "--dataset",
            str(dataset),
            "--ratings",
            str(ratings),
            "--level",
            "ordinal",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0


def test_calibrate_ordinal_level_rejects_string_labels(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    ratings = tmp_path / "r1.jsonl"
    _write_jsonl(
        dataset,
        [
            {"id": "c1", "input": "q1", "human_label": "lo"},
            {"id": "c2", "input": "q2", "human_label": "hi"},
        ],
    )
    _write_jsonl(
        ratings,
        [
            {"case_id": "c1", "label": "lo"},
            {"case_id": "c2", "label": "hi"},
        ],
    )

    result = runner.invoke(
        app,
        ["calibrate", "--dataset", str(dataset), "--ratings", str(ratings), "--level", "ordinal"],
    )

    assert result.exit_code == 1
    assert "c1" in result.stderr
    assert "c2" in result.stderr


def test_calibrate_invalid_level_exits_2(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    ratings = tmp_path / "r1.jsonl"
    _write_jsonl(dataset, [{"id": "c1", "input": "q1", "human_label": "yes"}])
    _write_jsonl(ratings, [{"case_id": "c1", "label": "yes"}])

    result = runner.invoke(
        app,
        ["calibrate", "--dataset", str(dataset), "--ratings", str(ratings), "--level", "bogus"],
    )

    assert result.exit_code == 2


def test_calibrate_missing_ratings_exits_2(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_jsonl(dataset, [{"id": "c1", "input": "q1", "human_label": "yes"}])

    result = runner.invoke(app, ["calibrate", "--dataset", str(dataset)])

    assert result.exit_code == 2


def test_calibrate_bad_dataset_malformed_json(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    ratings = tmp_path / "r1.jsonl"
    dataset.write_text(
        json.dumps({"id": "c1", "input": "q1", "human_label": "yes"}) + "\n{not valid json\n",
        encoding="utf-8",
    )
    _write_jsonl(ratings, [{"case_id": "c1", "label": "yes"}])

    result = runner.invoke(app, ["calibrate", "--dataset", str(dataset), "--ratings", str(ratings)])

    assert result.exit_code == 1
    assert "line" in result.stderr
