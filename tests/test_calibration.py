import json
from datetime import timedelta
from pathlib import Path

import pytest

from judgekit import stats
from judgekit.calibration import RaterRatings, execute_calibration, write_report
from judgekit.errors import CalibrationError, StatsError
from judgekit.models import CalibrationReport, Case, Dataset, Label, MetaValue

DATASET_VERSION = "sha256:" + "0" * 64


def _case(
    case_id: str,
    human_label: Label | None = None,
    metadata: dict[str, MetaValue] | None = None,
) -> Case:
    return Case(
        id=case_id,
        input="q",
        human_label=human_label,
        metadata=metadata or {},
        dataset_version=DATASET_VERSION,
    )


def _dataset(*cases: Case) -> Dataset:
    return Dataset(dataset_version=DATASET_VERSION, cases=tuple(cases))


def _happy_path_dataset() -> Dataset:
    return _dataset(
        _case("c1", "yes", {"difficulty": "easy"}),
        _case("c2", "yes", {"difficulty": "easy"}),
        _case("c3", "no", {"difficulty": "easy"}),
        _case("c4", "no", {"difficulty": "easy"}),
        _case("c5", "yes", {"difficulty": "hard"}),
        _case("c6", "no", {"difficulty": "hard"}),
        _case("c7", "yes", {"difficulty": "hard"}),
        _case("c8", "no", {"difficulty": "hard"}),
        _case("c9", None, {"difficulty": "easy"}),
        _case("c10", None, {"difficulty": "hard"}),
    )


def _happy_path_rater() -> RaterRatings:
    return RaterRatings(
        rater_id="r1",
        path="data/r1.jsonl",
        labels={
            "c1": "yes",
            "c2": "no",
            "c3": "no",
            "c4": "no",
            "c5": "yes",
            "c6": "yes",
            "c9": "yes",
        },
    )


def test_two_rater_happy_path() -> None:
    dataset = _happy_path_dataset()
    rater = _happy_path_rater()

    report = execute_calibration(dataset, [rater], dataset_path="data/cases.jsonl")

    assert report.n_cases == 10
    assert report.n_labeled == 8

    expected_humans = ["yes", "yes", "no", "no", "yes", "no"]
    expected_raters = ["yes", "no", "no", "no", "yes", "yes"]
    assert report.kappa is not None
    assert report.kappa.n_used == 6
    assert report.kappa.value == pytest.approx(stats.cohen_kappa(expected_humans, expected_raters))
    assert report.kappa.percent_agreement == pytest.approx(
        stats.percent_agreement(expected_humans, expected_raters)
    )

    expected_units = [
        ["yes", "yes"],
        ["yes", "no"],
        ["no", "no"],
        ["no", "no"],
        ["yes", "yes"],
        ["no", "yes"],
        ["yes"],
        ["no"],
        ["yes"],
        [],
    ]
    assert report.alpha.value == pytest.approx(stats.krippendorff_alpha(expected_units, "nominal"))
    assert report.alpha.n_used == 6

    assert report.confusion is not None
    total = sum(sum(row.values()) for row in report.confusion.values())
    assert total == report.kappa.n_used

    assert report.confusion_by_slice is not None
    assert set(report.confusion_by_slice) == {'difficulty="easy"', 'difficulty="hard"'}
    easy = report.confusion_by_slice['difficulty="easy"']
    hard = report.confusion_by_slice['difficulty="hard"']
    assert easy.n == 4
    assert hard.n == 2
    assert easy.percent_agreement == pytest.approx(0.75)
    assert hard.percent_agreement == pytest.approx(0.5)

    assert len(report.raters) == 1
    summary = report.raters[0]
    assert summary.rater_id == "r1"
    assert summary.n_ratings == 7
    assert summary.n_matched == 7


def test_multi_rater_has_no_kappa_but_alpha_counts_shared_unlabeled_case() -> None:
    dataset = _dataset(
        _case("c1", "a"),
        _case("c2", "b"),
        _case("c3", None),
        _case("c4", None),
    )
    rater_a = RaterRatings(rater_id="a", path="a.jsonl", labels={"c1": "a", "c2": "a", "c3": "a"})
    rater_b = RaterRatings(rater_id="b", path="b.jsonl", labels={"c1": "a", "c2": "b", "c3": "b"})

    report = execute_calibration(dataset, [rater_a, rater_b], dataset_path="d.jsonl")

    assert report.kappa is None
    assert report.confusion is None
    assert report.confusion_by_slice is None
    assert report.alpha.n_used == 3


def test_reproducibility_same_seed_identical_different_seed_differs() -> None:
    dataset = _happy_path_dataset()
    rater = _happy_path_rater()

    first = execute_calibration(dataset, [rater], dataset_path="d.jsonl", seed=0)
    second = execute_calibration(dataset, [rater], dataset_path="d.jsonl", seed=0)

    assert first.kappa == second.kappa
    assert first.alpha == second.alpha

    third = execute_calibration(dataset, [rater], dataset_path="d.jsonl", seed=1)
    assert third.kappa is not None
    assert first.kappa is not None
    ci_changed = (
        third.kappa.ci_low != first.kappa.ci_low
        or third.kappa.ci_high != first.kappa.ci_high
        or third.alpha.ci_low != first.alpha.ci_low
        or third.alpha.ci_high != first.alpha.ci_high
    )
    assert ci_changed


def test_ci_brackets_point_estimate_and_resamples_used_bounded() -> None:
    dataset = _happy_path_dataset()
    rater = _happy_path_rater()

    report = execute_calibration(dataset, [rater], dataset_path="d.jsonl", n_resamples=500)

    assert report.kappa is not None
    assert report.kappa.ci_low <= report.kappa.value <= report.kappa.ci_high
    assert report.alpha.ci_low <= report.alpha.value <= report.alpha.ci_high
    assert report.kappa.n_resamples_used <= 500
    assert report.alpha.n_resamples_used <= 500


def test_empty_raters_raises() -> None:
    dataset = _dataset(_case("c1", "yes"))
    with pytest.raises(CalibrationError):
        execute_calibration(dataset, [], dataset_path="d.jsonl")


def test_duplicate_rater_ids_raises() -> None:
    dataset = _dataset(_case("c1", "yes"))
    rater1 = RaterRatings(rater_id="r1", path="a.jsonl", labels={"c1": "yes"})
    rater2 = RaterRatings(rater_id="r1", path="b.jsonl", labels={"c1": "yes"})
    with pytest.raises(CalibrationError) as exc_info:
        execute_calibration(dataset, [rater1, rater2], dataset_path="d.jsonl")
    assert "r1" in str(exc_info.value)


def test_reserved_human_rater_id_raises() -> None:
    dataset = _dataset(_case("c1", "yes"))
    rater = RaterRatings(rater_id="human", path="a.jsonl", labels={"c1": "yes"})
    with pytest.raises(CalibrationError):
        execute_calibration(dataset, [rater], dataset_path="d.jsonl")


def test_zero_overlap_between_human_and_rater_raises() -> None:
    dataset = _dataset(_case("c1", "yes"), _case("c2", None))
    rater = RaterRatings(rater_id="r1", path="a.jsonl", labels={"c2": "yes"})
    with pytest.raises(CalibrationError):
        execute_calibration(dataset, [rater], dataset_path="d.jsonl")


def test_no_rater_labels_at_all_raises() -> None:
    dataset = _dataset(_case("c1", "yes"), _case("c2", "no"))
    rater = RaterRatings(rater_id="r1", path="a.jsonl", labels={})
    with pytest.raises(CalibrationError):
        execute_calibration(dataset, [rater], dataset_path="d.jsonl")


def test_ordinal_level_rejects_string_labels_naming_all_offenders() -> None:
    dataset = _dataset(_case("c1", "bad"), _case("c2", 3))
    rater = RaterRatings(rater_id="r1", path="a.jsonl", labels={"c1": 5, "c2": "bad_string"})
    with pytest.raises(CalibrationError) as exc_info:
        execute_calibration(dataset, [rater], dataset_path="d.jsonl", level="ordinal")
    message = str(exc_info.value)
    assert "c1" in message
    assert "c2" in message


def test_degenerate_single_category_data_propagates_stats_error() -> None:
    dataset = _dataset(_case("c1", "X"), _case("c2", "X"), _case("c3", "X"))
    rater = RaterRatings(rater_id="r1", path="a.jsonl", labels={"c1": "X", "c2": "X", "c3": "X"})
    with pytest.raises(StatsError):
        execute_calibration(dataset, [rater], dataset_path="d.jsonl")


def test_write_report_round_trips(tmp_path: Path) -> None:
    dataset = _happy_path_dataset()
    rater = _happy_path_rater()
    report = execute_calibration(dataset, [rater], dataset_path="d.jsonl")
    out_path = tmp_path / "report.json"

    write_report(report, out_path)

    parsed = json.loads(out_path.read_text(encoding="utf-8"))
    restored = CalibrationReport.model_validate(parsed)
    assert restored == report
    assert restored.created_at.tzinfo is not None
    assert restored.created_at.utcoffset() == timedelta(0)


def test_write_report_refuses_existing_path(tmp_path: Path) -> None:
    dataset = _dataset(_case("c1", "yes"), _case("c2", "no"))
    rater = RaterRatings(rater_id="r1", path="a.jsonl", labels={"c1": "yes", "c2": "no"})
    report = execute_calibration(dataset, [rater], dataset_path="d.jsonl", n_resamples=10)
    out_path = tmp_path / "report.json"
    out_path.write_text("existing", encoding="utf-8")

    with pytest.raises(CalibrationError):
        write_report(report, out_path)


def test_write_report_creates_missing_parent_directories(tmp_path: Path) -> None:
    dataset = _dataset(_case("c1", "yes"), _case("c2", "no"))
    rater = RaterRatings(rater_id="r1", path="a.jsonl", labels={"c1": "yes", "c2": "no"})
    report = execute_calibration(dataset, [rater], dataset_path="d.jsonl", n_resamples=10)
    out_path = tmp_path / "a" / "b" / "report.json"

    write_report(report, out_path)

    assert out_path.exists()
