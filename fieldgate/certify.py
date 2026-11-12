"""Choose an auto-approval threshold with a finite-sample guarantee on the error rate.

A confidence score ranks decisions. The business question is where to cut: decisions scoring at or
above the cut are approved without review. Picking the loosest cut whose error rate looked fine on
a labelled sample overfits that sample, and the realised error rate lands above target far more
often than people expect (see `naive` below and the simulation in the tests).

`certify` follows Learn-then-Test (Angelopoulos, Bates, Candes, Jordan, Lei, 2021) with
fixed-sequence testing. For each cut on a fixed grid it tests the hypothesis "the error rate among
approved decisions exceeds alpha" with an exact binomial test, walking from strict to loose and
stopping at the first cut it cannot reject. The guarantee:

    with probability at least 1 - delta over the labelled sample,
    the error rate among decisions approved at the returned cut is at most alpha,

provided the labelled decisions and the future ones are independent draws from the same
population. One decision per document satisfies independence; several fields of one document do
not, which is why certificates here are per field or per document. The guarantee says nothing
once the documents change, and `fieldgate.audit` exists to notice when they have.
"""
from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Sequence

from scipy.stats import beta, binom

# Cuts on a probability scale, strict to loose. Fixed in advance: a grid chosen after looking at the
# labelled errors would void the guarantee.
GRID: tuple[float, ...] = tuple(sorted(
    {round(1 - x, 6) for x in (0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075)} |
    {round(1 - k / 100, 2) for k in range(1, 51)}, reverse=True))


def upper_bound(errors: int, n: int, delta: float) -> float:
    """Clopper-Pearson upper confidence bound on an error rate, at confidence 1 - delta."""
    if n == 0:
        return 1.0
    if errors >= n:
        return 1.0
    return float(beta.ppf(1 - delta, errors + 1, n - errors))


def p_value(errors: int, n: int, alpha: float) -> float:
    """Exact p-value for the hypothesis "true error rate > alpha", having seen `errors` in `n`."""
    if n == 0:
        return 1.0
    return float(binom.cdf(errors, n, alpha))


def needed(alpha: float, delta: float, errors: int = 0) -> int:
    """Fewest approved, labelled decisions that can certify `alpha` when `errors` of them are wrong."""
    if errors == 0:
        return math.ceil(math.log(delta) / math.log(1 - alpha))
    n = errors + 1
    while p_value(errors, n, alpha) > delta:
        n += max(1, n // 200)
    while n > errors + 1 and p_value(errors, n - 1, alpha) <= delta:
        n -= 1
    return n


@dataclass
class Certificate:
    alpha: float
    delta: float
    threshold: float | None        # None: no cut could be certified, so everything is reviewed
    approved: int                  # labelled decisions at or above the cut
    errors: int                    # of those, how many were wrong
    total: int                     # labelled decisions offered
    bound: float                   # Clopper-Pearson upper bound on the error rate at the cut
    steps: int                     # cuts certified before stopping
    start: float | None            # where the walk began

    @property
    def coverage(self) -> float:
        return self.approved / self.total if self.total else 0.0

    def approves(self, score: float) -> bool:
        return self.threshold is not None and score >= self.threshold


class _Tally:
    """Approved count and error count at any cut, from one sort."""

    def __init__(self, scores: Sequence[float], wrong: Sequence[bool]):
        pairs = sorted(zip(scores, wrong), key=lambda p: p[0])
        self.scores = [p[0] for p in pairs]
        self.errors_from = [0] * (len(pairs) + 1)          # errors among pairs[i:]
        for i in range(len(pairs) - 1, -1, -1):
            self.errors_from[i] = self.errors_from[i + 1] + int(bool(pairs[i][1]))

    def at(self, cut: float) -> tuple[int, int]:
        i = bisect_left(self.scores, cut)
        return len(self.scores) - i, self.errors_from[i]


def start_cut(reference_scores: Sequence[float], n_calibration: int, alpha: float, delta: float,
              grid: Sequence[float] = GRID) -> float | None:
    """Where to begin the walk: the strictest cut expected to approve enough decisions to certify anything.

    `reference_scores` must come from data other than the calibration sample (the scores of the
    sample the scorer was fitted on, taken out of fold). Starting stricter than this wastes the
    walk on a cut too small to pass even with zero errors; starting here costs nothing in validity
    because the choice never sees a calibration label.
    """
    if not len(reference_scores):
        return None
    need = needed(alpha, delta)
    for cut in grid:
        share = sum(s >= cut for s in reference_scores) / len(reference_scores)
        if share * n_calibration >= 1.25 * need:
            return cut
    return None


def certify(scores: Sequence[float], wrong: Sequence[bool], alpha: float, delta: float = 0.05,
            start: float | None = None, grid: Sequence[float] = GRID) -> Certificate:
    """The loosest cut on `grid` certified by fixed-sequence testing, beginning at `start`."""
    if len(scores) != len(wrong):
        raise ValueError("scores and wrong must have the same length")
    cuts = [c for c in grid if start is None or c <= start]
    chosen: tuple[float, int, int] | None = None
    steps = 0
    tally = _Tally(scores, wrong)
    for cut in cuts:
        n, k = tally.at(cut)
        if p_value(k, n, alpha) > delta:
            break
        chosen = (cut, n, k)
        steps += 1
    if chosen is None:
        return Certificate(alpha, delta, None, 0, 0, len(scores), 1.0, 0, cuts[0] if cuts else None)
    cut, n, k = chosen
    return Certificate(alpha, delta, cut, n, k, len(scores), upper_bound(k, n, delta), steps, cuts[0])


def naive(scores: Sequence[float], wrong: Sequence[bool], alpha: float, grid: Sequence[float] = GRID) -> float | None:
    """What teams usually do: the loosest cut whose observed error rate is within target. No guarantee."""
    best = None
    tally = _Tally(scores, wrong)
    for cut in grid:
        n, k = tally.at(cut)
        if n and k / n <= alpha:
            best = cut
    return best
