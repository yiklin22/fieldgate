"""The review queue: people check only the fields the certified cuts did not approve.

    python -m fieldgate.app build --model Qwen3-1.7B --view rows --alpha 0.05    # score the test receipts
    python -m fieldgate.app serve                                                # http://127.0.0.1:8770

`build` fits the scorer and the per-field cuts on CORD's train split exactly as `fieldgate.analyze`
does, then routes the held-out receipts. `serve` shows each receipt beside its fields, with the OCR
boxes that support a value outlined on the image. Approving or correcting a field is appended to
`decisions.jsonl`; a random tenth of the auto-approved fields is also shown for audit, and those
answers feed the drift monitor.

Local, single-user demo: it binds to 127.0.0.1 and refuses requests addressed to any other host.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import analyze, audit, certify, data, ocr, router, signals
from .schema import BY_NAME, FIELD_NAMES, canonical

HOME = Path(".cache/review")
PORT = 8770
AUDIT_RATE = 0.10


def build(model: str, view: str, alpha: float, delta: float = 0.05, limit: int | None = None) -> dict:
    rows = analyze.load_rows(model, view)
    fit, cal, test = analyze.split_rows(rows)
    if not fit or not cal or not test:
        raise SystemExit("extraction has not covered the train and test splits yet")
    scorer = router.fit(fit)
    fit_scores, cal_scores, test_scores = router.cross_fit(fit), router.score(scorer, cal), router.score(scorer, test)
    cuts: dict[str, float | None] = {}
    for name in FIELD_NAMES:
        ref = [s for r, s in zip(fit, fit_scores) if r["field"] == name]
        cs = [s for r, s in zip(cal, cal_scores) if r["field"] == name]
        cw = [not r["correct"] for r in cal if r["field"] == name]
        cuts[name] = certify.certify(cs, cw, alpha, delta, start=certify.start_cut(ref, len(cs), alpha, delta)).threshold

    by_doc: dict[str, dict] = {}
    for r, s in zip(test, test_scores):
        cut = cuts[r["field"]]
        approved = cut is not None and s >= cut
        doc = by_doc.setdefault(r["doc"], {"id": r["doc"], "fields": {}})
        doc["fields"][r["field"]] = {
            "value": r["value"], "score": round(s, 4), "approved": approved,
            "audit": approved and audit.choose_for_audit(f"{r['doc']}:{r['field']}", AUDIT_RATE),
            "why": _why(r), "gold": r["gold"], "correct": r["correct"],
        }
    docs = sorted(by_doc.values(), key=lambda d: d["id"])[:limit]

    HOME.mkdir(parents=True, exist_ok=True)
    (HOME / "images").mkdir(exist_ok=True)
    wanted = {d["id"]: d for d in docs}
    for split in ("validation", "test"):
        for receipt in data.load(split):
            if receipt.id not in wanted:
                continue
            (HOME / "images" / f"{receipt.id}.png").write_bytes(receipt.image)
            page = ocr.Page.from_json(json.loads((analyze.RUNS / "ocr" / f"{receipt.id}.json").read_text()))
            doc = wanted[receipt.id]
            doc["width"], doc["height"] = page.width, page.height
            for name, f in doc["fields"].items():
                f["boxes"] = _boxes(page, name, f["value"])
    queue = {"model": model, "view": view, "alpha": alpha, "delta": delta, "cuts": cuts, "docs": docs,
             "built": time.strftime("%Y-%m-%d %H:%M")}
    (HOME / "queue.json").write_text(json.dumps(queue))
    return queue


def _why(row: dict) -> list[str]:
    """Plain reasons a reviewer can act on."""
    notes = []
    if row["sum_bad"]:
        notes.append("the amounts do not add up")
    if not row["is_null"] and not row["printed"] and row["field"] != "item_count":
        notes.append("this value was not read anywhere on the receipt")
    if row["is_null"] and row["label_in_doc"]:
        notes.append("the receipt has a label for this field but no value was extracted")
    if row["agree"] < 1:
        notes.append("the model gave a different answer when asked again")
    if row["p_value"] < 0.8:
        notes.append(f"the model put {100 * row['p_value']:.0f}% on this value")
    if row["malformed"]:
        notes.append("the model's output was malformed")
    return notes


def _boxes(page: ocr.Page, field: str, value: object) -> list[list[float]]:
    """Upright boxes (x, y, width, height in image pixels) of OCR lines that print this value."""
    target = canonical(field, value)
    if target is None or BY_NAME[field].kind == "count":
        return []
    out = []
    for line in page.lines:
        for token in signals._NUMBER.findall(line.text):
            number = signals.parse_amount(token)
            if number is not None and abs(abs(number) - abs(target)) < 0.005:
                xs, ys = line.quad[0::2], line.quad[1::2]
                out.append([min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)])
                break
    return out


def load_decisions() -> dict[str, dict]:
    path = HOME / "decisions.jsonl"
    latest: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                latest[f"{d['doc']}:{d['field']}"] = d
    return latest


def monitor_state(queue: dict, decisions: dict[str, dict]) -> dict:
    """Replay audited answers, in the order they were given, through the drift monitor."""
    monitor = audit.Monitor(alpha=queue["alpha"])
    index = {d["id"]: d for d in queue["docs"]}
    for d in sorted(decisions.values(), key=lambda d: d["at"]):
        field = index.get(d["doc"], {}).get("fields", {}).get(d["field"])
        if field and field["audit"]:
            monitor.record(d["action"] == "correct")
    return {"audited": monitor.audited, "errors": monitor.errors, "alarm": monitor.alarmed, "alarm_at": monitor.alarm_at}


def create_app():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

    queue_path = HOME / "queue.json"
    if not queue_path.exists():
        raise SystemExit("no queue yet: run `python -m fieldgate.app build` first")
    queue = json.loads(queue_path.read_text())
    index = {d["id"]: d for d in queue["docs"]}
    app = FastAPI(title="fieldgate review", docs_url=None, redoc_url=None, openapi_url=None)
    allowed_hosts = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        if request.headers.get("host") not in allowed_hosts:
            return JSONResponse({"detail": "this app answers on localhost only"}, status_code=403)
        if request.method != "GET" and request.headers.get("origin") not in (None, *(f"http://{h}" for h in allowed_hosts)):
            return JSONResponse({"detail": "cross-site request refused"}, status_code=403)
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self'"
        return response

    def public(doc: dict, decisions: dict) -> dict:
        fields = {}
        for name, f in doc["fields"].items():
            shown = {k: f[k] for k in ("value", "score", "approved", "audit", "why", "boxes")}   # labels stay server-side
            shown["decision"] = decisions.get(f"{doc['id']}:{name}")
            fields[name] = shown
        return {"id": doc["id"], "width": doc["width"], "height": doc["height"], "fields": fields}

    @app.get("/", response_class=HTMLResponse)
    def home():
        return (Path(__file__).parent / "static" / "review.html").read_text()

    @app.get("/static/review.js")
    def script():
        return FileResponse(Path(__file__).parent / "static" / "review.js", media_type="text/javascript")

    @app.get("/api/queue")
    def get_queue():
        decisions = load_decisions()
        docs = []
        for d in queue["docs"]:
            todo = [n for n, f in d["fields"].items() if (not f["approved"] or f["audit"]) and f"{d['id']}:{n}" not in decisions]
            docs.append({"id": d["id"], "to_review": len(todo),
                         "flagged": sum(not f["approved"] for f in d["fields"].values())})
        fields = [f for d in queue["docs"] for f in d["fields"].values()]
        return {"model": queue["model"], "alpha": queue["alpha"], "delta": queue["delta"], "cuts": queue["cuts"],
                "docs": docs, "fields": len(fields), "approved": sum(f["approved"] for f in fields),
                "touchless": sum(all(f["approved"] for f in d["fields"].values()) for d in queue["docs"]),
                "monitor": monitor_state(queue, decisions)}

    @app.get("/api/doc/{doc_id}")
    def get_doc(doc_id: str):
        if doc_id not in index:
            raise HTTPException(404, "no such receipt")
        return public(index[doc_id], load_decisions())

    @app.get("/image/{doc_id}")
    def image(doc_id: str):
        if doc_id not in index:                      # ids come from the queue, never from the path
            raise HTTPException(404, "no such receipt")
        return FileResponse(HOME / "images" / f"{doc_id}.png", media_type="image/png")

    @app.post("/api/doc/{doc_id}/{field}")
    async def decide(doc_id: str, field: str, request: Request):
        if doc_id not in index or field not in index[doc_id]["fields"]:
            raise HTTPException(404, "no such field")
        body = await request.json()
        action = body.get("action")
        if action not in ("accept", "correct"):
            raise HTTPException(400, "action must be accept or correct")
        value = body.get("value")
        if action == "correct":
            if value is not None and (not isinstance(value, str) or len(value) > 40):
                raise HTTPException(400, "value must be a short string or null")
            if value not in (None, "") and canonical(field, value) is None:
                raise HTTPException(400, "that does not read as a number")
        record = {"doc": doc_id, "field": field, "action": action,
                  "value": (value or None) if action == "correct" else index[doc_id]["fields"][field]["value"],
                  "was": index[doc_id]["fields"][field]["value"], "at": time.time()}
        with (HOME / "decisions.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        return {"saved": record, "monitor": monitor_state(queue, load_decisions())}

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fieldgate.app")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--model", default="Qwen3-1.7B")
    b.add_argument("--view", default="rows")
    b.add_argument("--alpha", type=float, default=0.05)
    b.add_argument("--limit", type=int, default=None)
    sub.add_parser("serve")
    args = parser.parse_args(argv)
    if args.command == "build":
        queue = build(args.model, args.view, args.alpha, limit=args.limit)
        fields = [f for d in queue["docs"] for f in d["fields"].values()]
        print(f"{len(queue['docs'])} receipts, {sum(f['approved'] for f in fields)} of {len(fields)} fields auto-approved; "
              f"cuts: {queue['cuts']}")
        return 0
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=PORT, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
