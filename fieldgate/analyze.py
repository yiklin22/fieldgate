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

from . import certify, data, evaluate, ocr, router, signals
from .schema import FIELD_NAMES

RUNS = Path(".cache/runs")
BASELINES = {
    "model probability of the value": "p_value",
    "model's stated confidence": "stated",
    "agreement of 3 re-samples": "agree",
}
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
            ref = [fit_scores[i] for i in fi]
            cs, cw = [cal_scores[i] for i in ci], [not cal_rows[i]["correct"] for i in ci]
            cert = certify.certify(cs, cw, alpha, delta, start=certify.start_cut(ref, len(cs), alpha, delta))
            ts, tw = [test_scores[i] for i in ti], [not test_rows[i]["correct"] for i in ti]
            out.append(_outcome("field", name, alpha, cert, cs, cw, ts, tw))
        ref, _, _ = by_document(fit_rows, fit_scores)
        cs, cc, _ = by_document(cal_rows, cal_scores)
        ts, tc, _ = by_document(test_rows, test_scores)
        cw, tw = [not c for c in cc], [not c for c in tc]
        cert = certify.certify(cs, cw, alpha, delta, start=certify.start_cut(ref, len(cs), alpha, delta))
        out.append(_outcome("document", "all fields right", alpha, cert, cs, cw, ts, tw))
    return out


def _outcome(unit, name, alpha, cert, cs, cw, ts, tw) -> dict:
    approved = [w for s, w in zip(ts, tw) if cert.approves(s)]
    return {"unit": unit, "name": name, "alpha": alpha, "threshold": cert.threshold,
            "cal_n": len(cs), "cal_error": mean(cw) if cw else 0.0, "cal_approved": cert.approved, "cal_errors": cert.errors,
            "bound": cert.bound, "test_n": len(ts), "test_error": mean(tw) if tw else 0.0,
            "test_approved": len(approved), "test_errors": sum(approved)}


def resplit_study(rows: list[dict], alpha: float, delta: float, repeats: int, kind: str) -> dict:
    """Re-split the receipts many times and count how often each way of picking a cut misses the target on test.

    Per document, because that is one independent decision per receipt. The test parts overlap
    between repeats, so this is a description of stability, not a second proof.
    """
    tally = {"certified": {"over": 0, "coverage": [], "risk": [], "none": 0},
             "naive": {"over": 0, "coverage": [], "risk": [], "none": 0}}
    for seed in range(repeats):
        fit, cal, test = split_rows(rows, seed=seed)
        model = router.fit(fit, kind)
        ref, _, _ = by_document(fit, router.cross_fit(fit, kind=kind, seed=seed))
        cs, cc, _ = by_document(cal, router.score(model, cal))
        ts, tc, _ = by_document(test, router.score(model, test))
        cw = [not c for c in cc]
        cuts = {"certified": certify.certify(cs, cw, alpha, delta, start=certify.start_cut(ref, len(cs), alpha, delta)).threshold,
                "naive": certify.naive(cs, cw, alpha)}
        for how, cut in cuts.items():
            approved = [not c for s, c in zip(ts, tc) if cut is not None and s >= cut]
            t = tally[how]
            t["none"] += cut is None
            t["coverage"].append(len(approved) / len(ts))
            risk = sum(approved) / len(approved) if approved else 0.0
            t["risk"].append(risk)
            t["over"] += risk > alpha
    return {how: {"over_target": t["over"] / repeats, "mean_coverage": mean(t["coverage"]), "mean_error": mean(t["risk"]),
                  "nothing_certified": t["none"] / repeats} for how, t in tally.items()}


def analyse(model: str, view: str, alphas=(0.02, 0.05, 0.10), delta: float = 0.05, draws: int = 1000,
            repeats: int = 200, kind: str = "logistic") -> dict:
    rows = load_rows(model, view)
    fit, cal, test = split_rows(rows)
    if not fit or not cal or not test:
        raise SystemExit(f"not enough extracted receipts yet: fit {len(fit)}, calibrate {len(cal)}, test {len(test)} rows")
    scorer = router.fit(fit, kind)
    fit_scores = router.cross_fit(fit, kind=kind)
    cal_scores, test_scores = router.score(scorer, cal), router.score(scorer, test)

    scored = {name: [r[key] for r in test] for name, key in BASELINES.items()}
    for group, names in GROUPS.items():
        if group != "model":
            scored[f"combined, {group} signals only"] = router.score(router.fit(fit, kind, names), test)
    scored["combined, model signals only"] = router.score(router.fit(fit, kind, GROUPS["model"]), test)
    scored["combined, all signals"] = test_scores
    scored["combined, all signals (boosted trees)"] = router.score(router.fit(fit, "boosted"), test)

    accuracy = {}
    for f in FIELD_NAMES:
        sub = [r for r in test if r["field"] == f]
        present = [r for r in sub if (r["gold"] or 0) != 0]
        accuracy[f] = {"n": len(sub), "accuracy": mean(r["correct"] for r in sub) if sub else None,
                       "present": len(present), "accuracy_present": mean(r["correct"] for r in present) if present else None}
    _, doc_ok, _ = by_document(test, test_scores)

    return {
        "model": model, "view": view, "kind": kind, "delta": delta,
        "docs": {"fit": len({r["doc"] for r in fit}), "calibrate": len({r["doc"] for r in cal}), "test": len({r["doc"] for r in test})},
        "rows": {"fit": len(fit), "calibrate": len(cal), "test": len(test)},
        "accuracy": accuracy,
        "field_accuracy": mean(r["correct"] for r in test),
        "document_accuracy": mean(doc_ok),
        "signal_table": signal_table(test, scored, alphas[0], draws),
        "signal_alpha": alphas[0],
        "certificates": certificates(fit_scores, fit, cal_scores, cal, test_scores, test, alphas, delta),
        "resplit": {str(a): resplit_study(rows, a, delta, repeats, kind) for a in alphas[1:]},
        "repeats": repeats,
        "weights": router.weights(scorer)[:12],
        "seconds": mean(json.loads(p.read_text())["seconds"] for p in list((RUNS / "extract" / model / view).glob("*.json"))[:200]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fieldgate.analyze")
    parser.add_argument("--model", default="Qwen3-1.7B")
    parser.add_argument("--view", default="rows")
    parser.add_argument("--kind", default="logistic", choices=["logistic", "boosted"])
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
