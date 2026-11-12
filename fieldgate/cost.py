"""Turn an approval policy into money, so a threshold can be argued about in the customer's units.

Every number in `Costs` is an assumption to replace with the customer's own. The comparison between
policies is the output, not the absolute figure.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Costs:
    wage_per_hour: float = 18.0            # loaded cost of the person who reviews
    seconds_to_open: float = 12.0          # open a document and find your place
    seconds_per_field: float = 6.0         # check one field against the image
    cost_per_error: float = 15.0           # a wrong value that reaches the ledger: rework, credit note, dispute


def per_document(docs: int, docs_opened: int, fields_reviewed: int, errors_shipped: int, costs: Costs = Costs()) -> dict:
    """Cost per document of a policy that had people open `docs_opened` documents and check `fields_reviewed`
    fields, and let `errors_shipped` wrong values through."""
    labour = (docs_opened * costs.seconds_to_open + fields_reviewed * costs.seconds_per_field) / 3600 * costs.wage_per_hour
    errors = errors_shipped * costs.cost_per_error
    return {"labour": labour / docs, "errors": errors / docs, "total": (labour + errors) / docs,
            "touchless": 1 - docs_opened / docs}


def policy(rows: list[dict], approve, costs: Costs = Costs(), reviewer_catches: float = 1.0) -> dict:
    """Cost of routing `rows` (signal rows carrying `correct`) with `approve(row) -> bool`.

    A reviewed field is corrected with probability `reviewer_catches`; an approved field ships as it is.
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
    out = per_document(len(docs), opened, reviewed, shipped, costs)
    out["fields_reviewed"] = reviewed / max(1, len(rows))
    out["errors_shipped_per_100_docs"] = 100 * shipped / max(1, len(docs))
    return out
