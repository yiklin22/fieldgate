"""Notice when a certified cut has stopped holding.

A certificate is only as good as the assumption that tomorrow's documents look like the labelled
ones. A new supplier template or a scanner change breaks that silently. The defence is cheap: have a
person check a small random sample of the approved decisions, and watch the running error count.

`Monitor` is a sequential likelihood-ratio test of "the error rate among approved decisions is at
most alpha" against "it is `ratio` times alpha". It can be checked after every audited decision
without inflating false alarms: if the error rate really is at most alpha, the chance it ever raises
the alarm is at most `false_alarm` (Ville's inequality).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class Monitor:
    alpha: float
    ratio: float = 3.0
    false_alarm: float = 0.05
    audited: int = 0
    errors: int = 0
    log_wealth: float = 0.0
    alarm_at: int | None = None
    history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.alternative = min(0.999, self.alpha * self.ratio)
        if not 0 < self.alpha < self.alternative:
            raise ValueError("need 0 < alpha and ratio > 1")

    def record(self, wrong: bool) -> bool:
        """Add one audited decision. Returns True once the alarm has been raised."""
        self.audited += 1
        self.errors += bool(wrong)
        if wrong:
            self.log_wealth += math.log(self.alternative / self.alpha)
        else:
            self.log_wealth += math.log((1 - self.alternative) / (1 - self.alpha))
        self.history.append(self.log_wealth)
        if self.alarm_at is None and self.log_wealth >= math.log(1 / self.false_alarm):
            self.alarm_at = self.audited
        return self.alarm_at is not None

    @property
    def alarmed(self) -> bool:
        return self.alarm_at is not None


def choose_for_audit(decision_id: str, rate: float, salt: str = "fieldgate") -> bool:
    """Whether an approved decision goes to audit. Deterministic in its id, so a re-run picks the same ones,
    and independent of the score, so the audited sample represents everything approved."""
    return random.Random(f"{salt}:{decision_id}").random() < rate
