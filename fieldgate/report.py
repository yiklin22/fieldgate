"""Render the analysis as markdown."""
from __future__ import annotations

from . import certify
from .evaluate import wilson


def pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def render(r: dict) -> str:
    d, delta = r["docs"], r["delta"]
    out = [f"# Results on CORD receipts: {r['model']}, `{r['view']}` view", ""]
    extras = r.get("extras") or {"samples": 3, "stated": True}
    did = [f"{extras['samples']} re-samples"] if extras["samples"] else []
    did += ["the stated-confidence question"] if extras["stated"] else []
    out += [f"Receipts: {d['fit']} to fit the scorer, {d['calibrate']} to choose approval cuts, {d['test']} held out for "
            f"every number below. Extraction took {r['seconds']:.0f} s per receipt on an 8-core CPU, " +
            ("including " + " and ".join(did) if did else "with no re-samples and no stated-confidence question, so those "
             "signals are absent from this run") + ".", ""]

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
    if r.get("chosen_on_fit"):
        out += [f"The combined scorer is {r['kind_label']}, picked on the fit receipts alone by out-of-fold AURC (" +
                ", ".join(f"{k} {v:.4f}" for k, v in r["chosen_on_fit"].items()) + ").", ""]
    if r["weights"]:
        out += ["Largest standardised weights in the logistic-regression scorer, the version whose reasoning can be read out: " +
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
            f"The receipts were re-split {r['repeats']} times. Each time, every field got a cut chosen on the calibration "
            "part two ways and checked on the held-out part. \"Missed target\" is the share of those cuts whose held-out "
            "error rate among approved values came out above target. The held-out part is small, so a cut whose true "
            "error rate is under target can still miss by chance; the comparison between the two rows is the point.", "",
            "| target | method | missed target | mean share approved | mean error among approved | nothing approved |",
            "|---|---|---|---|---|---|"]
    for alpha, both in r["resplit"].items():
        for how, label in (("naive", "loosest cut that looked fine on calibration"), ("certified", "certified cut")):
            s = both["fields"][how]
            out.append(f"| {pct(float(alpha), 0)} | {label} | {pct(s['over_target'], 0)} | {pct(s['mean_coverage'], 0)} | "
                       f"{pct(s['mean_error'])} | {pct(s['nothing_certified'], 0)} |")
    out.append("")
    out += ["The same comparison for each decision on its own, as missed target / mean share approved:", "",
            "| decision | " + " | ".join(f"usual way, {pct(float(a), 0)}" + " | " + f"certified, {pct(float(a), 0)}"
                                         for a in r["resplit"]) + " |",
            "|---|" + "---|---|" * len(r["resplit"])]
    units = list(next(iter(r["resplit"].values()))["by_unit"])
    for u in units:
        cells = []
        for a in r["resplit"]:
            for how in ("naive", "certified"):
                s = r["resplit"][a]["by_unit"][u][how]
                cells.append(f"{pct(s['over_target'], 0)} / {pct(s['mean_coverage'], 0)}")
        out.append(f"| {'**whole receipt**' if u == 'document' else u} | " + " | ".join(cells) + " |")
    out.append("")

    c = r["cost_assumptions"]
    prices = list(r["break_even"])
    out += ["## What the routing is worth", "",
            f"Per receipt, on the test receipts. Labour assumes a reviewer at ${c['wage_per_hour']:.0f} an hour who takes "
            f"{c['seconds_to_open']:.0f} s to open a receipt and {c['seconds_per_field']:.0f} s per field and catches every "
            "error in what they check. The last columns price each wrong value that ships at three levels, because that "
            "price decides everything: skipping review only pays while the error rate among approved fields is below "
            "the cost of one check divided by the cost of one error (" +
            ", ".join(f"{pct(r['break_even'][p], 1)} at ${float(p):.2f}" for p in prices) +
            "). These are placeholders for a customer's own figures.", "",
            "| policy | receipts no one opens | fields checked | wrong values shipped per 100 receipts | labour | " +
            " | ".join(f"total at ${float(p):.2f} an error" for p in prices) + " |",
            "|---|---|---|---|---|" + "---|" * len(prices)]
    for row in r["costs"]:
        out.append(f"| {row['policy']} | {pct(row['touchless'], 0)} | {pct(row['fields_reviewed'], 0)} | "
                   f"{100 * row['shipped']:.1f} | ${row['labour']:.3f} | " +
                   " | ".join(f"${row['total'][p]:.3f}" for p in prices) + " |")
    out.append("")

    if r.get("views"):
        v = r["views"]
        names = [k for k in v if k != "receipts"]
        out += ["## Does rebuilding the rows matter", "",
                f"The same {v['receipts']} receipts, extracted from two renderings of the same OCR: docling's reading-order "
                "export (`reading`) and rows rebuilt from the OCR boxes (`rows`).", "",
                "| view | field accuracy | receipts with every field right | " + " | ".join(v[names[0]]["by_field"]) + " |",
                "|---|---|---|" + "---|" * len(v[names[0]]["by_field"])]
        for name in names:
            out.append(f"| `{name}` | {pct(v[name]['field'])} | {pct(v[name]['document'])} | " +
                       " | ".join(pct(x, 0) for x in v[name]["by_field"].values()) + " |")
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
