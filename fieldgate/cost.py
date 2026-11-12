"""Turn an approval policy into money, so a target error rate can be argued about in the customer's units.

Approving a field without review saves the seconds a person would spend on it and risks shipping a
wrong value. So the error-rate target is not a matter of taste: approving pays only while

    error rate among approved  <  cost of checking one field / cost of one wrong value

(`break_even`). A cheap check and an expensive error mean almost nothing should skip review; an
expensive check and a cheap error mean most things can. Every number in `Costs` is an assumption to
replace with the customer's own, and the comparison between policies is the output, not the absolute figure.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Costs:
    wage_per_hour: float = 18.0            # loaded cost of the person who reviews
    seconds_to_open: float = 12.0          # open a document and find your place
    seconds_per_field: float = 6.0         # check one field against the image

    @property
    def per_field(self) -> float:
        return self.seconds_per_field / 3600 * self.wage_per_hour

    @property
    def per_open(self) -> float:
        return self.seconds_to_open / 3600 * self.wage_per_hour


def break_even(cost_per_error: float, costs: Costs = Costs()) -> float:
    """The error rate among approved fields below which skipping review saves money."""
    return min(1.0, costs.per_field / cost_per_error)


def policy(rows: list[dict], approve, costs: Costs = Costs(), reviewer_catches: float = 1.0) -> dict:
    """What routing `rows` (signal rows carrying `correct`) with `approve(row) -> bool` costs per document.

    A reviewed field is corrected with probability `reviewer_catches`; an approved field ships as it is.
    Returns labour per document and wrong values shipped per document; `total` prices the errors.
    """
    docs: dict[str, list[dict]] = {}
    for r in rows:
        docs.setdefault(r["doc"], []).append(r)
    opened = reviewed = 0
    shipped = 0.0
    for fields in docs.values():
        sent = [r for r in fields if not approve(r)]
        opened += bool(sent)
        reviewed += len(sent)
        shipped += sum(not r["correct"] for r in fields if approve(r))
        shipped += (1 - reviewer_catches) * sum(not r["correct"] for r in sent)
    n = max(1, len(docs))
    return {"labour": (opened * costs.per_open + reviewed * costs.per_field) / n,
            "shipped": shipped / n,
            "touchless": 1 - opened / n,
            "fields_reviewed": reviewed / max(1, len(rows))}


def total(outcome: dict, cost_per_error: float) -> float:
    """Cost per document of a policy outcome when each wrong value that ships costs `cost_per_error`."""
    return outcome["labour"] + outcome["shipped"] * cost_per_error
