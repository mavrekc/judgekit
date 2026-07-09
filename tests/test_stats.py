import random
from collections import Counter
from collections.abc import Sequence

import krippendorff
import numpy as np
import pytest

from judgekit.errors import StatsError
from judgekit.stats import (
    BootstrapCI,
    bootstrap_ci,
    category,
    cohen_kappa,
    krippendorff_alpha,
    percent_agreement,
)


def test_kappa_2x2_fixture_exact() -> None:
    both_yes, both_no, a_yes_b_no, a_no_b_yes = 20, 15, 5, 10
    a = ["yes"] * both_yes + ["no"] * both_no + ["yes"] * a_yes_b_no + ["no"] * a_no_b_yes
    b = ["yes"] * both_yes + ["no"] * both_no + ["no"] * a_yes_b_no + ["yes"] * a_no_b_yes
    n = both_yes + both_no + a_yes_b_no + a_no_b_yes
    po = (both_yes + both_no) / n
    n_a_yes = both_yes + a_yes_b_no
    n_a_no = both_no + a_no_b_yes
    n_b_yes = both_yes + a_no_b_yes
    n_b_no = both_no + a_yes_b_no
    pe = (n_a_yes / n) * (n_b_yes / n) + (n_a_no / n) * (n_b_no / n)
    expected = (po - pe) / (1 - pe)
    assert po == pytest.approx(0.7)
    assert pe == pytest.approx(0.5)
    assert expected == pytest.approx(0.4)
    assert cohen_kappa(a, b) == pytest.approx(expected)


def test_kappa_three_category_fixture_exact() -> None:
    a = ["A", "A", "A", "B", "B", "B", "C", "C", "C"]
    b = ["A", "A", "B", "B", "B", "C", "A", "C", "C"]
    n = len(a)
    po = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    freq_a = Counter(a)
    freq_b = Counter(b)
    pe = sum((freq_a[c] / n) * (freq_b.get(c, 0) / n) for c in freq_a)
    expected = (po - pe) / (1 - pe)
    assert po == pytest.approx(2 / 3)
    assert pe == pytest.approx(1 / 3)
    assert expected == pytest.approx(0.5)
    assert cohen_kappa(a, b) == pytest.approx(expected)


def test_kappa_perfect_agreement_is_one() -> None:
    a = ["x", "y", "x", "y", "z"]
    assert cohen_kappa(a, list(a)) == pytest.approx(1.0)


def test_kappa_chance_level_is_zero() -> None:
    a = ["x", "x", "y", "y"]
    b = ["x", "y", "x", "y"]
    assert cohen_kappa(a, b) == 0.0


def test_kappa_symmetric() -> None:
    a = ["A", "A", "A", "B", "B", "B", "C", "C", "C"]
    b = ["A", "A", "B", "B", "B", "C", "A", "C", "C"]
    assert cohen_kappa(a, b) == cohen_kappa(b, a)


def test_kappa_invariant_to_pair_order() -> None:
    a = ["A", "A", "A", "B", "B", "B", "C", "C", "C"]
    b = ["A", "A", "B", "B", "B", "C", "A", "C", "C"]
    pairs = list(zip(a, b, strict=True))
    rng = random.Random(7)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    shuffled_a = [pair[0] for pair in shuffled]
    shuffled_b = [pair[1] for pair in shuffled]
    assert cohen_kappa(shuffled_a, shuffled_b) == pytest.approx(cohen_kappa(a, b))


def test_kappa_bool_and_int_are_distinct_categories() -> None:
    assert cohen_kappa([True], [1]) == 0.0


def test_category_distinguishes_bool_int_float_and_string() -> None:
    values = {category(True), category(1), category("1"), category(1.0)}
    assert len(values) == 4


def test_kappa_single_shared_category_raises() -> None:
    with pytest.raises(StatsError):
        cohen_kappa(["Q", "Q", "Q"], ["Q", "Q", "Q"])


def test_kappa_length_mismatch_raises() -> None:
    with pytest.raises(StatsError):
        cohen_kappa(["a", "b"], ["a"])


def test_kappa_empty_raises() -> None:
    with pytest.raises(StatsError):
        cohen_kappa([], [])


def test_kappa_non_finite_label_raises() -> None:
    with pytest.raises(StatsError):
        cohen_kappa([float("nan")], [1.0])


def test_category_non_finite_raises() -> None:
    with pytest.raises(StatsError):
        category(float("inf"))


def test_percent_agreement_matches_kappa_po() -> None:
    a = ["x", "x", "y", "y"]
    b = ["x", "y", "x", "y"]
    assert percent_agreement(a, b) == 0.5


def test_percent_agreement_length_mismatch_raises() -> None:
    with pytest.raises(StatsError):
        percent_agreement(["a"], ["a", "b"])


def test_percent_agreement_empty_raises() -> None:
    with pytest.raises(StatsError):
        percent_agreement([], [])


# Krippendorff (2011) "Computing Krippendorff's Alpha-Reliability", section C/D worked
# example (4 observers by 12 units, nominal alpha 0.743, ordinal 0.815, interval 0.849).
PAPER_UNITS: list[list[float]] = [
    [1, 1, 1],
    [2, 2, 3, 2],
    [3, 3, 3, 3],
    [3, 3, 3, 3],
    [2, 2, 2, 2],
    [1, 2, 3, 4],
    [4, 4, 4, 4],
    [1, 1, 2, 1],
    [2, 2, 2, 2],
    [5, 5, 5],
    [1, 1],
    [3],
]


def test_alpha_paper_worked_example_nominal() -> None:
    assert krippendorff_alpha(PAPER_UNITS, "nominal") == pytest.approx(0.743, abs=1e-3)


def test_alpha_paper_worked_example_ordinal() -> None:
    assert krippendorff_alpha(PAPER_UNITS, "ordinal") == pytest.approx(0.815, abs=1e-3)


def test_alpha_paper_worked_example_interval() -> None:
    assert krippendorff_alpha(PAPER_UNITS, "interval") == pytest.approx(0.849, abs=1e-3)


def test_alpha_perfect_agreement_all_levels() -> None:
    units = [[1, 1], [2, 2, 2], [3, 3]]
    for level in ("nominal", "ordinal", "interval"):
        assert krippendorff_alpha(units, level) == pytest.approx(1.0)


def test_alpha_discards_units_with_fewer_than_two_ratings() -> None:
    with_lone_value = krippendorff_alpha([[1, 1], [2, 2], [9]], "nominal")
    without_lone_value = krippendorff_alpha([[1, 1], [2, 2]], "nominal")
    assert with_lone_value == pytest.approx(without_lone_value)


def test_alpha_all_units_too_short_raises() -> None:
    with pytest.raises(StatsError):
        krippendorff_alpha([[1], [], [2]], "nominal")


def test_alpha_single_category_raises() -> None:
    with pytest.raises(StatsError):
        krippendorff_alpha([[1, 1], [1, 1], [1, 1]], "nominal")


def test_alpha_ordinal_rejects_bool() -> None:
    with pytest.raises(StatsError):
        krippendorff_alpha([[True, False], [True, True]], "ordinal")


def test_alpha_interval_rejects_string() -> None:
    with pytest.raises(StatsError):
        krippendorff_alpha([["a", "b"], ["a", "a"]], "interval")


def test_alpha_interval_differs_from_nominal() -> None:
    units = [[1.0, 2.0], [1.0, 3.0], [2.0, 3.0], [1.0, 1.0]]
    nominal = krippendorff_alpha(units, "nominal")
    interval = krippendorff_alpha(units, "interval")
    assert nominal != pytest.approx(interval)


CROSS_CHECK_CATEGORIES = (0.0, 1.0, 2.0, 3.0)


def _generate_reliability_matrix(seed: int) -> list[list[float | None]]:
    rng = random.Random(seed)
    n_raters = rng.randint(2, 5)
    n_units = rng.randint(5, 30)
    missing_rate = rng.uniform(0.0, 0.4)
    matrix: list[list[float | None]] = []
    for _ in range(n_raters):
        row: list[float | None] = []
        for _ in range(n_units):
            value = None if rng.random() < missing_rate else rng.choice(CROSS_CHECK_CATEGORIES)
            row.append(value)
        matrix.append(row)
    return matrix


def _matrix_to_units(matrix: list[list[float | None]]) -> list[list[float]]:
    n_units = len(matrix[0])
    return [[row[u] for row in matrix if row[u] is not None] for u in range(n_units)]


def _matrix_to_reliability_data(matrix: list[list[float | None]]) -> np.ndarray:
    return np.array([[np.nan if v is None else v for v in row] for row in matrix])


def test_alpha_matches_reference_implementation() -> None:
    successes = 0
    for seed in range(10):
        matrix = _generate_reliability_matrix(seed)
        units = _matrix_to_units(matrix)
        reliability_data = _matrix_to_reliability_data(matrix)
        for level in ("nominal", "ordinal", "interval"):
            try:
                ours = krippendorff_alpha(units, level)
            except StatsError:
                continue
            theirs = krippendorff.alpha(
                reliability_data=reliability_data, level_of_measurement=level
            )
            assert ours == pytest.approx(theirs, abs=1e-9)
            successes += 1
    assert successes >= 20


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def test_bootstrap_ci_deterministic_for_same_seed() -> None:
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 3.5, 1.5]
    first = bootstrap_ci(data, _mean, seed=42, n_resamples=200)
    second = bootstrap_ci(data, _mean, seed=42, n_resamples=200)
    assert first == second


def test_bootstrap_ci_differs_across_seeds() -> None:
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 3.5, 1.5, 9.0, 0.5]
    first = bootstrap_ci(data, _mean, seed=1, n_resamples=200)
    second = bootstrap_ci(data, _mean, seed=2, n_resamples=200)
    assert first != second


def test_bootstrap_ci_brackets_point_estimate() -> None:
    data = [float(i) + 0.37 * (i % 5) for i in range(30)]
    point_estimate = _mean(data)
    ci = bootstrap_ci(data, _mean, seed=3, n_resamples=1000)
    assert ci.low <= point_estimate <= ci.high


def _kappa_of_pairs(pairs: Sequence[tuple[str, str]]) -> float:
    a = [pair[0] for pair in pairs]
    b = [pair[1] for pair in pairs]
    return cohen_kappa(a, b)


def test_bootstrap_ci_skips_degenerate_resamples() -> None:
    data = [("X", "X")] * 8 + [("Y", "Y"), ("Y", "X")]
    result = bootstrap_ci(data, _kappa_of_pairs, seed=5, n_resamples=500)
    assert result.n_resamples_used < 500


def test_bootstrap_ci_all_degenerate_raises() -> None:
    data = [("X", "X")] * 5
    with pytest.raises(StatsError):
        bootstrap_ci(data, _kappa_of_pairs, seed=5, n_resamples=50)


def test_bootstrap_ci_zero_resamples_raises() -> None:
    with pytest.raises(StatsError):
        bootstrap_ci([1.0, 2.0], _mean, n_resamples=0)


def test_bootstrap_ci_confidence_zero_raises() -> None:
    with pytest.raises(StatsError):
        bootstrap_ci([1.0, 2.0], _mean, confidence=0.0)


def test_bootstrap_ci_confidence_one_raises() -> None:
    with pytest.raises(StatsError):
        bootstrap_ci([1.0, 2.0], _mean, confidence=1.0)


def test_bootstrap_ci_empty_data_raises() -> None:
    with pytest.raises(StatsError):
        bootstrap_ci([], _mean)


def test_bootstrap_ci_is_namedtuple() -> None:
    result = bootstrap_ci([1.0, 2.0, 3.0], _mean, seed=0, n_resamples=50)
    assert isinstance(result, BootstrapCI)
