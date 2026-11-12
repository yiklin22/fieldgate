"""The experiment: which signals tell right values from wrong ones, and how much can be auto-approved
at a certified error rate.

    python -m fieldgate.analyze --model Qwen3-1.7B --view rows --out docs/results/cord.md

Receipts are split three ways and each part has one job:
  fit        the scorer's weights are learned here
  calibrate  the approval cut is chosen here, with the guarantee
  test       everything reported is measured here, and nothing was tuned on it
Fit and calibrate are the two halves of CORD's train split; test is CORD's validation and test splits.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean

from . import certify, cost, data, evaluate, ocr, router, signals
from .schema import FIELD_NAMES

RUNS = Path(".cache/runs")
BASELINES = {
    "model probability of the value": "p_value",
    "model's stated confidence": "stated",
    "agreement of 3 re-samples": "agree",
}
CHARTED = ("model probability of the value", "model's stated confidence", "agreement of 3 re-samples", "combined, all signals")
GROUPS = {
    "model": ("is_null", "p_value", "min_token_p", "first_token_p", "p_null", "tokens", "malformed"),
    "agreement": ("agree", "stated", "stated_missing"),
    "source": ("printed", "printed_twice", "ocr_conf", "label_on_row", "label_in_doc", "doc_ocr_conf", "doc_lines", "doc_tilt"),
    "rules": ("sum_ok", "sum_bad"),
}


def load_rows(model: str, view: str, splits=("train", "validation", "test"), data_dir: str = ".cache/data") -> list[dict]:
    """Signal rows for every receipt that has both OCR and an extraction on disk."""
    cache = RUNS / "rows" / f"{model}-{view}.json"
    extract_dir = RUNS / "extract" / model / view
    have = sorted(p.stem for p in extract_dir.glob("*.json")) if extract_dir.exists() else []
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached["docs"] == have:
            return cached["rows"]
    rows: list[dict] = []
    wanted = set(have)
    for split in splits:
        if not any(d.startswith(split) for d in wanted):
            continue
        for receipt in data.load(split, data_dir):
            if receipt.id not in wanted:
                continue
            page = ocr.Page.from_json(json.loads((RUNS / "ocr" / f"{receipt.id}.json").read_text()))
            extraction = json.loads((extract_dir / f"{receipt.id}.json").read_text())
            rows.extend(signals.rows_for(receipt.id, page, extraction, receipt.gold, receipt.unusable))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"docs": have, "rows": rows}))
    return rows


def split_rows(rows: list[dict], seed: int | None = None, shares=(0.4, 0.4)) -> tuple[list[dict], list[dict], list[dict]]:
    """Official split when `seed` is None (train halves, then validation + test); a random re-split otherwise."""
    docs = sorted({r["doc"] for r in rows})
    if seed is None:
        train = [d for d in docs if d.startswith("train")]
        fit_docs, cal_docs = set(train[0::2]), set(train[1::2])
    else:
        rng = random.Random(seed)
        rng.shuffle(docs)
        a, b = int(shares[0] * len(docs)), int((shares[0] + shares[1]) * len(docs))
        fit_docs, cal_docs = set(docs[:a]), set(docs[a:b])
    fit = [r for r in rows if r["doc"] in fit_docs]
    cal = [r for r in rows if r["doc"] in cal_docs]
    test = [r for r in rows if r["doc"] not in fit_docs and r["doc"] not in cal_docs]
    return fit, cal, test


def by_document(rows: list[dict], scores: list[float]) -> tuple[list[float], list[bool], list[str]]:
    """One decision per receipt: its score is the chance every field is right; it is wrong if any field is."""
    docs: dict[str, list] = {}
    for r, s in zip(rows, scores):
        entry = docs.setdefault(r["doc"], [1.0, True])
        entry[0] *= s
        entry[1] = entry[1] and r["correct"]
    names = sorted(docs)
    return [docs[d][0] for d in names], [docs[d][1] for d in names], names


def fmt_pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def fmt_ci(ci: tuple[float, float] | None, digits: int = 1, pct: bool = True) -> str:
    if ci is None:
        return ""
    if pct:
        return f" ({100 * ci[0]:.{digits}f}-{100 * ci[1]:.{digits}f})"
    return f" ({ci[0]:.{digits}f}-{ci[1]:.{digits}f})"


def signal_table(test: list[dict], scored: dict[str, list[float]], alpha: float, draws: int) -> list[str]:
    lines = ["| score | AUROC (95% CI) | AURC, lower is better | approved at ≤" + fmt_pct(alpha, 0) + " observed error |",
             "|---|---|---|---|"]
    for name, scores in scored.items():
        keyed = [dict(r, _s=s) for r, s in zip(test, scores)]
        correct = [r["correct"] for r in test]
        au = evaluate.auroc(scores, correct)
        au_ci = evaluate.bootstrap(keyed, lambda rs: evaluate.auroc([r["_s"] for r in rs], [r["correct"] for r in rs]), draws)
        ar = evaluate.aurc(scores, correct)
        ar_ci = evaluate.bootstrap(keyed, lambda rs: evaluate.aurc([r["_s"] for r in rs], [r["correct"] for r in rs]), draws)
        cov = evaluate.coverage_at_risk(scores, correct, alpha)
        cov_ci = evaluate.bootstrap(keyed, lambda rs: evaluate.coverage_at_risk([r["_s"] for r in rs], [r["correct"] for r in rs], alpha), draws)
        lines.append(f"| {name} | {au:.3f}{fmt_ci(au_ci, 3, pct=False)} | {ar:.4f}{fmt_ci(ar_ci, 4, pct=False)} | "
                     f"{fmt_pct(cov)}{fmt_ci(cov_ci)} |")
    return lines


def certificates(fit_scores, fit_rows, cal_scores, cal_rows, test_scores, test_rows, alphas, delta) -> list[dict]:
    """Per-field and per-document certificates chosen on calibration, with what happened on test."""
    out = []
    units = [(f, [i for i, r in enumerate(fit_rows) if r["field"] == f], [i for i, r in enumerate(cal_rows) if r["field"] == f],
              [i for i, r in enumerate(test_rows) if r["field"] == f]) for f in FIELD_NAMES]
    for alpha in alphas:
        for name, fi, ci, ti in units:
            ref, rw = [fit_scores[i] for i in fi], [not fit_rows[i]["correct"] for i in fi]
            cs, cw = [cal_scores[i] for i in ci], [not cal_rows[i]["correct"] for i in ci]
            cert = certify.certify(cs, cw, alpha, delta, start=certify.start_cut(ref, len(cs), alpha, delta, rw))
            ts, tw = [test_scores[i] for i in ti], [not test_rows[i]["correct"] for i in ti]
            out.append(_outcome("field", name, alpha, cert, cs, cw, ts, tw))
        ref, rc, _ = by_document(fit_rows, fit_scores)
        cs, cc, _ = by_document(cal_rows, cal_scores)
        ts, tc, _ = by_document(test_rows, test_scores)
        cw, tw = [not c for c in cc], [not c for c in tc]
        cert = certify.certify(cs, cw, alpha, delta,
                               start=certify.start_cut(ref, len(cs), alpha, delta, [not c for c in rc]))
        out.append(_outcome("document", "all fields right", alpha, cert, cs, cw, ts, tw))
    return out


def _outcome(unit, name, alpha, cert, cs, cw, ts, tw) -> dict:
    approved = [w for s, w in zip(ts, tw) if cert.approves(s)]
    return {"unit": unit, "name": name, "alpha": alpha, "threshold": cert.threshold,
            "cal_n": len(cs), "cal_error": mean(cw) if cw else 0.0, "cal_approved": cert.approved, "cal_errors": cert.errors,
            "bound": cert.bound, "test_n": len(ts), "test_error": mean(tw) if tw else 0.0,
            "test_approved": len(approved), "test_errors": sum(approved)}


ERROR_COSTS = (0.25, 1.0, 5.0)


def cost_table(certs: list[dict], test: list[dict], test_scores: list[float], alphas) -> list[dict]:
    """Labour and shipped errors per receipt on the test receipts: check everything, check nothing, and
    the per-field cuts certified for each target."""
    score_of = {(r["doc"], r["field"]): s for r, s in zip(test, test_scores)}
    policies = [("a person checks every field", lambda r: False), ("nothing is checked", lambda r: True)]
    for alpha in alphas:
        cuts = {c["name"]: c["threshold"] for c in certs if c["unit"] == "field" and c["alpha"] == alpha}

        def routed(r: dict, cuts=cuts) -> bool:
            cut = cuts.get(r["field"])
            return cut is not None and score_of[(r["doc"], r["field"])] >= cut

        policies.append((f"certified cuts, target {100 * alpha:.0f}%", routed))
    out = []
    for label, approve in policies:
        outcome = cost.policy(test, approve)
        out.append({"policy": label, **outcome, "total": {str(c): cost.total(outcome, c) for c in ERROR_COSTS}})
    return out


def compare_views(model: str, base_view: str, other_view: str) -> dict | None:
    """Accuracy of two text views on the receipts both were run on."""
    a, b = load_rows(model, base_view), load_rows(model, other_view)
    shared = {r["doc"] for r in a} & {r["doc"] for r in b}
    if len(shared) < 20:
        return None
    out = {"receipts": len(shared)}
    for view, rows in ((base_view, a), (other_view, b)):
        rows = [r for r in rows if r["doc"] in shared]
        _, ok, _ = by_document(rows, [1.0] * len(rows))
        out[view] = {"field": mean(r["correct"] for r in rows), "document": mean(ok),
                     "by_field": {f: mean(r["correct"] for r in rows if r["field"] == f) for f in FIELD_NAMES}}
    return out


def _decisions(rows: list[dict], scores: list[float], unit: str) -> tuple[list[float], list[bool]]:
    """Scores and wrong flags for one kind of decision: a field name, or "document" for the whole receipt."""
    if unit == "document":
        s, ok, _ = by_document(rows, scores)
        return s, [not c for c in ok]
    keep = [i for i, r in enumerate(rows) if r["field"] == unit]
    return [scores[i] for i in keep], [not rows[i]["correct"] for i in keep]


def resplit_study(rows: list[dict], alphas, delta: float, repeats: int, kind: str) -> dict:
    """Re-split the receipts many times and count how often each way of picking a cut misses the target.

    Every field and the whole receipt get their own cut in every re-split, chosen two ways on the
    calibration part and checked on the held-out part. The held-out parts overlap between repeats
    and are small, so a cut whose true error rate is under target can still come out above it by
    chance. This describes how the two methods behave; the proof is the simulation in the tests.
    """
    units = list(FIELD_NAMES) + ["document"]
    tally = {str(a): {u: {how: {"over": 0, "coverage": [], "risk": [], "none": 0} for how in ("naive", "certified")}
                      for u in units} for a in alphas}
    for seed in range(repeats):
        fit, cal, test = split_rows(rows, seed=seed)
        model = router.fit(fit, kind)
        fit_scores = router.cross_fit(fit, kind=kind, seed=seed)
        cal_scores, test_scores = router.score(model, cal), router.score(model, test)
        for u in units:
            ref, rw = _decisions(fit, fit_scores, u)
            cs, cw = _decisions(cal, cal_scores, u)
            ts, tw = _decisions(test, test_scores, u)
            for a in alphas:
                cuts = {"certified": certify.certify(cs, cw, a, delta,
                                                     start=certify.start_cut(ref, len(cs), a, delta, rw)).threshold,
                        "naive": certify.naive(cs, cw, a)}
                for how, cut in cuts.items():
                    approved = [w for s, w in zip(ts, tw) if cut is not None and s >= cut]
                    t = tally[str(a)][u][how]
                    t["none"] += cut is None
                    t["coverage"].append(len(approved) / len(ts))
                    risk = sum(approved) / len(approved) if approved else 0.0
                    t["risk"].append(risk)
                    t["over"] += risk > a

    def summary(parts: list[dict]) -> dict:
        n = sum(len(t["coverage"]) for t in parts)
        return {"over_target": sum(t["over"] for t in parts) / n,
                "mean_coverage": sum(sum(t["coverage"]) for t in parts) / n,
                "mean_error": sum(sum(t["risk"]) for t in parts) / n,
                "nothing_certified": sum(t["none"] for t in parts) / n}

    out = {}
    for a, per_unit in tally.items():
        out[a] = {"fields": {how: summary([per_unit[f][how] for f in FIELD_NAMES]) for how in ("naive", "certified")},
                  "by_unit": {u: {how: summary([per_unit[u][how]]) for how in ("naive", "certified")} for u in units}}
    return out


def choose_kind(fit: list[dict]) -> tuple[str, dict]:
    """Pick the scorer on the fit receipts alone: lowest out-of-fold AURC."""
    correct = [r["correct"] for r in fit]
    found = {kind: evaluate.aurc(router.cross_fit(fit, kind=kind), correct) for kind in ("logistic", "boosted")}
    return min(found, key=found.get), found


def _thin(curve: list[tuple[float, float, float]], keep: int = 240) -> list[tuple[float, float]]:
    """A risk-coverage curve at no more than `keep` points, always including both ends."""
    step = max(1, len(curve) // keep)
    picked = curve[::step] + ([curve[-1]] if (len(curve) - 1) % step else [])
    return [(round(c, 4), round(r, 4)) for c, r, _ in picked]


def _seconds(model: str, view: str) -> float:
    files = sorted((RUNS / "extract" / model / view).glob("*.json"))[:200]
    return mean(json.loads(p.read_text())["seconds"] for p in files) if files else 0.0


def _extras(model: str, view: str) -> dict:
    """What the extraction run asked for beyond the values: re-samples and the stated-confidence question."""
    files = sorted((RUNS / "extract" / model / view).glob("*.json"))[:1]
    if not files:
        return {"samples": 0, "stated": False}
    first = json.loads(files[0].read_text())
    return {"samples": len(first.get("samples") or []), "stated": bool(first.get("stated"))}


def analyse(model: str, view: str, alphas=(0.02, 0.05, 0.10), delta: float = 0.05, draws: int = 1000,
            repeats: int = 200, kind: str = "auto") -> dict:
    rows = load_rows(model, view)
    fit, cal, test = split_rows(rows)
    if not fit or not cal or not test:
        raise SystemExit(f"not enough extracted receipts yet: fit {len(fit)}, calibrate {len(cal)}, test {len(test)} rows")
    chosen_on_fit = None
    if kind == "auto":
        kind, chosen_on_fit = choose_kind(fit)
    scorer = router.fit(fit, kind)
    fit_scores = router.cross_fit(fit, kind=kind)
    cal_scores, test_scores = router.score(scorer, cal), router.score(scorer, test)

    scored = {name: [r[key] for r in test] for name, key in BASELINES.items()}
    scored = {name: scores for name, scores in scored.items() if len(set(scores)) > 1}   # a run without re-samples has no such score
    for group, names in GROUPS.items():
        if group != "model" and any(len({r[n] for r in fit}) > 1 for n in names):    # skip a group this run never collected
            scored[f"combined, {group} signals only"] = router.score(router.fit(fit, kind, names), test)
    scored["combined, model signals only"] = router.score(router.fit(fit, kind, GROUPS["model"]), test)
    other_kind = "boosted" if kind == "logistic" else "logistic"
    labels = {"logistic": "logistic regression", "boosted": "boosted trees"}
    scored["combined, all signals"] = test_scores
    scored[f"combined, all signals ({labels[other_kind]} instead)"] = router.score(router.fit(fit, other_kind), test)

    accuracy = {}
    for f in FIELD_NAMES:
        sub = [r for r in test if r["field"] == f]
        present = [r for r in sub if (r["gold"] or 0) != 0]
        accuracy[f] = {"n": len(sub), "accuracy": mean(r["correct"] for r in sub) if sub else None,
                       "present": len(present), "accuracy_present": mean(r["correct"] for r in present) if present else None}
    _, doc_ok, _ = by_document(test, test_scores)
    certs = certificates(fit_scores, fit, cal_scores, cal, test_scores, test, alphas, delta)
    other = "reading" if view == "rows" else "rows"

    return {
        "model": model, "view": view, "kind": kind, "kind_label": labels[kind], "chosen_on_fit": chosen_on_fit, "delta": delta,
        "docs": {"fit": len({r["doc"] for r in fit}), "calibrate": len({r["doc"] for r in cal}), "test": len({r["doc"] for r in test})},
        "rows": {"fit": len(fit), "calibrate": len(cal), "test": len(test)},
        "accuracy": accuracy,
        "field_accuracy": mean(r["correct"] for r in test),
        "document_accuracy": mean(doc_ok),
        "signal_table": signal_table(test, scored, alphas[0], draws),
        "curves": {name: _thin(evaluate.risk_coverage(scores, [r["correct"] for r in test]))
                   for name, scores in scored.items() if name in CHARTED},
        "signal_alpha": alphas[0],
        "certificates": certs,
        "costs": cost_table(certs, test, test_scores, alphas),
        "cost_assumptions": cost.Costs().__dict__,
        "break_even": {str(c): cost.break_even(c) for c in ERROR_COSTS},
        "views": compare_views(model, view, other),
        "resplit": resplit_study(rows, alphas, delta, repeats, kind),
        "repeats": repeats,
        "weights": router.weights(scorer if kind == "logistic" else router.fit(fit, "logistic"))[:12],
        "seconds": _seconds(model, view),
        "extras": _extras(model, view),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fieldgate.analyze")
    parser.add_argument("--model", default="Qwen3-1.7B")
    parser.add_argument("--view", default="rows")
    parser.add_argument("--kind", default="auto", choices=["auto", "logistic", "boosted"],
                        help="scorer; auto picks by out-of-fold AURC on the fit receipts")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--json", default=None, help="write the raw results here")
    parser.add_argument("--out", default=None, help="write the markdown report here")
    args = parser.parse_args(argv)
    result = analyse(args.model, args.view, repeats=args.repeats, draws=args.draws, kind=args.kind)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=1))
    from .report import render
    text = render(result)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
