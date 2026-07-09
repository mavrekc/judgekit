"""Builds and persists calibration reports comparing human labels against raters."""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple
from uuid import uuid4

from judgekit import stats
from judgekit.errors import CalibrationError
from judgekit.models import (
    AgreementEstimate,
    CalibrationReport,
    Case,
    Dataset,
    KappaEstimate,
    Label,
    RaterSummary,
    SliceAgreement,
)


class RaterRatings(NamedTuple):
    """One rater's id, source path, and case_id -> label mapping."""

    rater_id: str
    path: str
    labels: dict[str, Label]


def _is_numeric(value: Label) -> bool:
    """Return True if value is an int or float and not a bool."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def _validate_raters(raters: Sequence[RaterRatings]) -> None:
    """Reject an empty rater list, duplicate rater ids, or the reserved "human" id."""
    if not raters:
        raise CalibrationError("at least one rater is required")
    counts: dict[str, int] = {}
    for rater in raters:
        counts[rater.rater_id] = counts.get(rater.rater_id, 0) + 1
    duplicates = sorted(rater_id for rater_id, n in counts.items() if n > 1)
    if duplicates:
        raise CalibrationError(f"duplicate rater ids: {', '.join(duplicates)}")
    if "human" in counts:
        raise CalibrationError('rater_id "human" is reserved for the dataset human_label anchor')


def _validate_numeric_labels(
    dataset: Dataset,
    raters: Sequence[RaterRatings],
    level: stats.Level,
    case_ids: set[str],
) -> None:
    """For ordinal/interval levels, raise one CalibrationError naming every non-numeric label."""
    if level not in ("ordinal", "interval"):
        return
    offenses: list[str] = []
    for case in dataset.cases:
        if case.human_label is not None and not _is_numeric(case.human_label):
            offenses.append(f"{case.id} (human_label)")
    for rater in raters:
        for case_id, label in rater.labels.items():
            if case_id in case_ids and not _is_numeric(label):
                offenses.append(f"{case_id} (rater {rater.rater_id})")
    if offenses:
        raise CalibrationError(
            f"{level} level requires int or float labels, not bool; offending: "
            + ", ".join(offenses)
        )


def _rater_summaries(
    raters: Sequence[RaterRatings], case_ids: set[str]
) -> tuple[RaterSummary, ...]:
    """Build a RaterSummary per rater, in input order."""
    return tuple(
        RaterSummary(
            rater_id=rater.rater_id,
            ratings_path=rater.path,
            n_ratings=len(rater.labels),
            n_matched=sum(1 for case_id in rater.labels if case_id in case_ids),
        )
        for rater in raters
    )


def _alpha_units(dataset: Dataset, raters: Sequence[RaterRatings]) -> list[list[Label]]:
    """Build one unit per case: the human label plus each rater's label, if present."""
    units: list[list[Label]] = []
    for case in dataset.cases:
        unit: list[Label] = []
        if case.human_label is not None:
            unit.append(case.human_label)
        for rater in raters:
            if case.id in rater.labels:
                unit.append(rater.labels[case.id])
        units.append(unit)
    return units


def _kappa_pairs(dataset: Dataset, rater: RaterRatings) -> list[tuple[Case, Label, Label]]:
    """Pair each case's human label with the rater's label, keeping only cases both cover."""
    pairs: list[tuple[Case, Label, Label]] = []
    for case in dataset.cases:
        if case.human_label is None or case.id not in rater.labels:
            continue
        pairs.append((case, case.human_label, rater.labels[case.id]))
    return pairs


def _build_confusion(pairs: Sequence[tuple[Case, Label, Label]]) -> dict[str, dict[str, int]]:
    """Count human-category to rater-category pairs into a nested confusion matrix."""
    confusion: dict[str, dict[str, int]] = {}
    for _, human_label, rater_label in pairs:
        human_cat = stats.category(human_label)
        rater_cat = stats.category(rater_label)
        row = confusion.setdefault(human_cat, {})
        row[rater_cat] = row.get(rater_cat, 0) + 1
    return confusion


def _confusion_by_slice(pairs: Sequence[tuple[Case, Label, Label]]) -> dict[str, SliceAgreement]:
    """Break the confusion matrix down by every metadata key=value combination present."""
    combos: dict[str, tuple[str, Label]] = {}
    for case, _, _ in pairs:
        for key, value in case.metadata.items():
            combos[f"{key}={stats.category(value)}"] = (key, value)

    result: dict[str, SliceAgreement] = {}
    for slice_key in sorted(combos):
        key, value = combos[slice_key]
        target = stats.category(value)
        slice_pairs = [
            pair
            for pair in pairs
            if key in pair[0].metadata and stats.category(pair[0].metadata[key]) == target
        ]
        humans = [pair[1] for pair in slice_pairs]
        rater_labels = [pair[2] for pair in slice_pairs]
        result[slice_key] = SliceAgreement(
            n=len(slice_pairs),
            percent_agreement=stats.percent_agreement(humans, rater_labels),
            confusion=_build_confusion(slice_pairs),
        )
    return result


def execute_calibration(
    dataset: Dataset,
    raters: Sequence[RaterRatings],
    *,
    dataset_path: str,
    level: stats.Level = "nominal",
    seed: int = 0,
    n_resamples: int = 1000,
    confidence: float = 0.95,
) -> CalibrationReport:
    """Align human labels with one or more raters and compute kappa and alpha agreement."""
    _validate_raters(raters)
    case_ids = {case.id for case in dataset.cases}
    _validate_numeric_labels(dataset, raters, level, case_ids)

    n_cases = len(dataset.cases)
    n_labeled = sum(1 for case in dataset.cases if case.human_label is not None)
    rater_summaries = _rater_summaries(raters, case_ids)

    units = _alpha_units(dataset, raters)
    retained_units = [unit for unit in units if len(unit) >= 2]
    n_used_alpha = len(retained_units)
    if n_used_alpha == 0:
        raise CalibrationError("no case has two or more labels")

    def _alpha_of(us: Sequence[Sequence[Label]]) -> float:
        return stats.krippendorff_alpha(us, level)

    alpha_value = stats.krippendorff_alpha(units, level)
    alpha_ci = stats.bootstrap_ci(
        retained_units, _alpha_of, n_resamples=n_resamples, seed=seed, confidence=confidence
    )
    alpha = AgreementEstimate(
        value=alpha_value,
        ci_low=alpha_ci.low,
        ci_high=alpha_ci.high,
        n_used=n_used_alpha,
        n_resamples_used=alpha_ci.n_resamples_used,
    )

    kappa: KappaEstimate | None = None
    confusion: dict[str, dict[str, int]] | None = None
    confusion_by_slice: dict[str, SliceAgreement] | None = None

    if len(raters) == 1:
        rater = raters[0]
        pairs = _kappa_pairs(dataset, rater)
        n_used_kappa = len(pairs)
        if n_used_kappa == 0:
            raise CalibrationError(f"no overlap between human labels and rater {rater.rater_id!r}")

        humans = [pair[1] for pair in pairs]
        rater_labels = [pair[2] for pair in pairs]
        kappa_value = stats.cohen_kappa(humans, rater_labels)
        pct_agreement = stats.percent_agreement(humans, rater_labels)

        def _kappa_of(label_pairs: Sequence[tuple[Case, Label, Label]]) -> float:
            h = [pair[1] for pair in label_pairs]
            r = [pair[2] for pair in label_pairs]
            return stats.cohen_kappa(h, r)

        kappa_ci = stats.bootstrap_ci(
            pairs, _kappa_of, n_resamples=n_resamples, seed=seed, confidence=confidence
        )
        kappa = KappaEstimate(
            value=kappa_value,
            ci_low=kappa_ci.low,
            ci_high=kappa_ci.high,
            n_used=n_used_kappa,
            n_resamples_used=kappa_ci.n_resamples_used,
            percent_agreement=pct_agreement,
        )
        confusion = _build_confusion(pairs)
        confusion_by_slice = _confusion_by_slice(pairs)

    return CalibrationReport(
        report_id=uuid4().hex,
        created_at=datetime.now(UTC),
        dataset_path=dataset_path,
        dataset_version=dataset.dataset_version,
        judge_config_id=None,
        level=level,
        raters=rater_summaries,
        n_cases=n_cases,
        n_labeled=n_labeled,
        bootstrap_seed=seed,
        bootstrap_resamples=n_resamples,
        confidence_level=confidence,
        kappa=kappa,
        alpha=alpha,
        confusion=confusion,
        confusion_by_slice=confusion_by_slice,
    )


def write_report(report: CalibrationReport, out_path: Path) -> None:
    """Write a calibration report as a single, fully formed indented JSON object."""
    if out_path.exists():
        raise CalibrationError(f"{out_path}: report already exists; refusing to overwrite")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
