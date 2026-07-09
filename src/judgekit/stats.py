"""Model-agnostic agreement statistics: percent agreement, kappa, alpha, and bootstrap CIs."""

import json
import math
import random
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Literal, NamedTuple

from judgekit.errors import StatsError
from judgekit.models import Label

type Level = Literal["nominal", "ordinal", "interval"]


def category(value: Label) -> str:
    """Return the canonical JSON string that identifies value's category."""
    if isinstance(value, float) and not math.isfinite(value):
        raise StatsError(f"non-finite label has no category: {value!r}")
    return json.dumps(value, ensure_ascii=False)


def percent_agreement(a: Sequence[Label], b: Sequence[Label]) -> float:
    """Return the fraction of paired positions where a and b share a category."""
    if len(a) != len(b):
        raise StatsError("percent_agreement: a and b must have equal length")
    n = len(a)
    if n == 0:
        raise StatsError("percent_agreement: sequences must be non-empty")
    matches = sum(1 for x, y in zip(a, b, strict=True) if category(x) == category(y))
    return matches / n


def cohen_kappa(a: Sequence[Label], b: Sequence[Label]) -> float:
    """Return unweighted Cohen's kappa for two raters over paired nominal labels."""
    if len(a) != len(b):
        raise StatsError("cohen_kappa: a and b must have equal length")
    n = len(a)
    if n == 0:
        raise StatsError("cohen_kappa: sequences must be non-empty")
    po = percent_agreement(a, b)
    freq_a = Counter(category(x) for x in a)
    freq_b = Counter(category(x) for x in b)
    categories = set(freq_a) | set(freq_b)
    pe = sum((freq_a.get(c, 0) / n) * (freq_b.get(c, 0) / n) for c in categories)
    if pe == 1.0:
        raise StatsError("kappa undefined: raters use a single category")
    return (po - pe) / (1 - pe)


def _coincidence[C](
    coded: Sequence[Sequence[C]],
) -> tuple[dict[C, dict[C, float]], dict[C, float]]:
    """Build the Krippendorff coincidence matrix and category totals for coded units."""
    o: dict[C, dict[C, float]] = {}
    for unit in coded:
        m = len(unit)
        weight = 1.0 / (m - 1)
        for i in range(m):
            row = o.setdefault(unit[i], {})
            for j in range(m):
                if i == j:
                    continue
                row[unit[j]] = row.get(unit[j], 0.0) + weight
    n_c = {c: sum(row.values()) for c, row in o.items()}
    return o, n_c


def _ordinal_delta2(c: float, k: float, n_c: dict[float, float]) -> float:
    """Return Krippendorff's ordinal difference-squared between ranks c and k."""
    lo, hi = (c, k) if c <= k else (k, c)
    within = sum(value for value_, value in n_c.items() if lo <= value_ <= hi)
    return (within - (n_c[c] + n_c[k]) / 2) ** 2


def _numeric_category(value: Label, level: Level) -> float:
    """Return value as a float category, raising StatsError for bool or non-numeric input."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise StatsError(f"{level} level requires int or float values, got {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise StatsError(f"non-finite value has no category: {value!r}")
    return float(value)


def _alpha_from_coincidence[C](
    o: dict[C, dict[C, float]], n_c: dict[C, float], delta2: Callable[[C, C], float]
) -> float:
    """Combine a coincidence matrix and difference function into Krippendorff's alpha."""
    n = sum(n_c.values())
    categories = list(n_c)
    numerator = 0.0
    denominator = 0.0
    for i, c in enumerate(categories):
        for k in categories[i + 1 :]:
            d2 = delta2(c, k)
            numerator += o[c].get(k, 0.0) * d2
            denominator += n_c[c] * n_c[k] * d2
    if denominator == 0:
        raise StatsError("krippendorff_alpha: zero expected disagreement, alpha undefined")
    return 1 - (n - 1) * numerator / denominator


def krippendorff_alpha(units: Sequence[Sequence[Label]], level: Level = "nominal") -> float:
    """Return Krippendorff's alpha for units of ratings at the given level of measurement."""
    retained = [unit for unit in units if len(unit) >= 2]
    if not retained:
        raise StatsError("krippendorff_alpha: no unit has 2 or more ratings")

    if level == "nominal":
        coded_nominal = [[category(v) for v in unit] for unit in retained]
        o_nom, n_c_nom = _coincidence(coded_nominal)
        return _alpha_from_coincidence(o_nom, n_c_nom, lambda c, k: 1.0)

    coded_numeric = [[_numeric_category(v, level) for v in unit] for unit in retained]
    o_num, n_c_num = _coincidence(coded_numeric)
    if level == "interval":
        return _alpha_from_coincidence(o_num, n_c_num, lambda c, k: (c - k) ** 2)
    return _alpha_from_coincidence(o_num, n_c_num, lambda c, k: _ordinal_delta2(c, k, n_c_num))


class BootstrapCI(NamedTuple):
    """A percentile-method bootstrap confidence interval."""

    low: float
    high: float
    n_resamples_used: int


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Return the q-quantile of sorted_values via linear interpolation."""
    m = len(sorted_values)
    if m == 1:
        return sorted_values[0]
    pos = q * (m - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def bootstrap_ci[T](
    data: Sequence[T],
    statistic: Callable[[Sequence[T]], float],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> BootstrapCI:
    """Return a nonparametric percentile bootstrap CI for statistic over data, seeded."""
    if n_resamples < 1:
        raise StatsError("bootstrap_ci: n_resamples must be >= 1")
    if not 0 < confidence < 1:
        raise StatsError("bootstrap_ci: confidence must be strictly between 0 and 1")
    if len(data) == 0:
        raise StatsError("bootstrap_ci: data must be non-empty")

    rng = random.Random(seed)
    n = len(data)
    values: list[float] = []
    for _ in range(n_resamples):
        resample = [data[rng.randrange(n)] for _ in range(n)]
        try:
            values.append(statistic(resample))
        except StatsError:
            continue

    if not values:
        raise StatsError("bootstrap_ci: every resample was degenerate")

    values.sort()
    low = _percentile(values, (1 - confidence) / 2)
    high = _percentile(values, 1 - (1 - confidence) / 2)
    return BootstrapCI(low=low, high=high, n_resamples_used=len(values))
