"""How well a confidence score separates right values from wrong ones.

All of these read a score as "higher means more likely right". Intervals resample whole receipts,
because fields of one receipt fail together.
"""
from __future__ import annotations

import random
from typing import Callable, Sequence


def auroc(scores: Sequence[float], correct: Sequence[bool]) -> float | None:
    """Chance that a right value outscores a wrong one (ties count half). None if only one class."""
    pairs = sorted(zip(scores, correct))
    n_pos = sum(correct)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        mid = (i + j + 1) / 2                       # average 1-based rank of the tie group
        rank_sum += mid * sum(1 for k in range(i, j) if pairs[k][1])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def risk_coverage(scores: Sequence[float], correct: Sequence[bool]) -> list[tuple[float, float, float]]:
    """(coverage, error rate among approved, threshold) as the cut loosens. Tied scores move together."""
    order = sorted(zip(scores, correct), key=lambda p: -p[0])
    n, out, wrong, i = len(order), [], 0, 0
    while i < n:
        j = i
        while j < n and order[j][0] == order[i][0]:
            wrong += not order[j][1]
            j += 1
        out.append((j / n, wrong / j, order[i][0]))
        i = j
    return out


def aurc(scores: Sequence[float], correct: Sequence[bool]) -> float:
    """Area under the risk-coverage curve: mean error rate over every coverage level. Lower is better."""
    order = sorted(zip(scores, correct), key=lambda p: -p[0])
    n, i, wrong, area = len(order), 0, 0, 0.0
    while i < n:                                     # within a tie group, errors are spread evenly
        j = i
        group_wrong = 0
        while j < n and order[j][0] == order[i][0]:
            group_wrong += not order[j][1]
            j += 1
        size = j - i
        for step in range(1, size + 1):
            area += (wrong + group_wrong * step / size) / (i + step)
        wrong += group_wrong
        i = j
    return area / n if n else 0.0


def coverage_at_risk(scores: Sequence[float], correct: Sequence[bool], alpha: float) -> float:
    """Largest share that can be approved with the observed error rate at or below `alpha`. Optimistic: it is
    read off the same data it is reported on."""
    best = 0.0
    for coverage, risk, _ in risk_coverage(scores, correct):
        if risk <= alpha:
            best = coverage
    return best


def bootstrap(rows: Sequence[dict], stat: Callable[[Sequence[dict]], float | None], draws: int = 1000,
              seed: int = 0, level: float = 0.95) -> tuple[float, float] | None:
    """Percentile interval for `stat`, resampling receipts (the `doc` key) with replacement."""
    by_doc: dict[str, list[dict]] = {}
    for r in rows:
        by_doc.setdefault(r["doc"], []).append(r)
    docs = sorted(by_doc)
    if not docs:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(draws):
        sample = [row for d in rng.choices(docs, k=len(docs)) for row in by_doc[d]]
        v = stat(sample)
        if v is not None:
            values.append(v)
    if not values:
        return None
    values.sort()
    lo = values[int((1 - level) / 2 * (len(values) - 1))]
    hi = values[int((1 + level) / 2 * (len(values) - 1))]
    return lo, hi


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)
