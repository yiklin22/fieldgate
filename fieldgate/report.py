"""Render the analysis as markdown."""
from __future__ import annotations

from . import certify
from .evaluate import wilson


def pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def render(r: dict) -> str:
    d, delta = r["docs"], r["delta"]
    out = [f"# Results on CORD receipts: {r['model']}, `{r['view']}` view", ""]
    out += [f"Receipts: {d['fit']} to fit the scorer, {d['calibrate']} to choose approval cuts, {d['test']} held out for "
            f"every number below. Extraction took {r['seconds']:.0f} s per receipt on an 8-core CPU, including three "
            "re-samples and the stated-confidence question.", ""]

    out += ["## How good is the extraction before any routing", "",
            f"Field accuracy {pct(r['field_accuracy'])}; receipts with every field right {pct(r['document_accuracy'])}.", "",
            "| field | accuracy | receipts | accuracy when the field is printed | printed on |", "|---|---|---|---|---|"]
    for name, a in r["accuracy"].items():
        out.append(f"| {name} | {pct(a['accuracy'])} | {a['n']} | {pct(a['accuracy_present'])} | {a['present']} |")
    out.append("")

    out += ["## Which score finds the wrong values", "",
            "AUROC is the chance a right value outscores a wrong one. AURC is the average error rate over all approval "
            "levels. The last column is read off the test set itself, so it is optimistic; the certified numbers are below.", ""]
    out += r["signal_table"]
    out.append("")
    if r["weights"]:
        out += ["Largest standardised weights in the combined scorer: " +
                ", ".join(f"`{n}` {w:+.2f}" for n, w in r["weights"][:8]) + ".", ""]

    out += ["## Certified auto-approval", "",
            f"For each target, the cut was chosen on the calibration receipts so that, with {pct(1 - delta, 0)} confidence, "
            "the error rate among approved decisions is at most the target. \"Approved on test\" and \"wrong on test\" are "
            "what then happened on receipts the cut never saw.", ""]
    for alpha in sorted({c["alpha"] for c in r["certificates"]}):
        out += [f"### Target: at most {pct(alpha, 0)} wrong among approved", "",
                "| decision | error rate with no routing | cut | approved on test | wrong on test (95% interval) |",
                "|---|---|---|---|---|"]
        for c in (c for c in r["certificates"] if c["alpha"] == alpha):
            label = c["name"] if c["unit"] == "field" else "**whole receipt**"
            if c["threshold"] is None:
                out.append(f"| {label} | {pct(c['test_error'])} | none certified | 0 of {c['test_n']} | review all |")
                continue
            lo, hi = wilson(c["test_errors"], c["test_approved"])
            rate = c["test_errors"] / c["test_approved"] if c["test_approved"] else 0.0
            out.append(f"| {label} | {pct(c['test_error'])} | {c['threshold']:.3g} | {c['test_approved']} of {c['test_n']} "
                       f"({pct(c['test_approved'] / c['test_n'], 0)}) | {c['test_errors']} = {pct(rate)} ({pct(lo)}-{pct(hi)}) |")
        out.append("")

    out += ["## Certified cut against the usual way of picking one", "",
            f"The receipts were re-split {r['repeats']} times. Each time a whole-receipt cut was chosen on the calibration part "
            "two ways and checked on the held-out part. \"Missed target\" is the share of re-splits where the held-out "
            "error rate among approved receipts came out above target.", "",
            "| target | method | missed target | mean share approved | mean error among approved | nothing approved |",
            "|---|---|---|---|---|---|"]
    for alpha, both in r["resplit"].items():
        for how, label in (("naive", "loosest cut that looked fine on calibration"), ("certified", "certified cut")):
            s = both[how]
            out.append(f"| {pct(float(alpha), 0)} | {label} | {pct(s['over_target'], 0)} | {pct(s['mean_coverage'], 0)} | "
                       f"{pct(s['mean_error'])} | {pct(s['nothing_certified'], 0)} |")
    out.append("")

    out += ["## How many labelled receipts a target needs", "",
            f"Fewest approved, labelled decisions that can certify a target at {pct(1 - delta, 0)} confidence. "
            "This is arithmetic, not a measurement, and it is the first thing to tell a customer who asks for 99.9%.", "",
            "| target error rate | with 0 wrong | with 1 wrong | with 3 wrong |", "|---|---|---|---|"]
    for alpha in (0.10, 0.05, 0.02, 0.01, 0.005, 0.001):
        out.append(f"| {pct(alpha, 1)} | {certify.needed(alpha, delta, 0):,} | {certify.needed(alpha, delta, 1):,} | "
                   f"{certify.needed(alpha, delta, 3):,} |")
    out.append("")
    return "\n".join(out)
