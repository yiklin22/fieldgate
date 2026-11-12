"""Run the pipeline stages. Every stage caches per receipt, so a stopped run resumes where it left off.

    python -m fieldgate.run ocr      --splits train validation test
    python -m fieldgate.run extract  --model .cache/models/Qwen3-1.7B --view rows --samples 3 --splits validation
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import data, layout, ocr

RUNS = Path(".cache/runs")


def run_ocr(splits: list[str], limit: int | None) -> None:
    for split in splits:
        start, done = time.time(), 0
        for receipt in data.load(split, limit=limit):
            ocr.read_cached(receipt.id, receipt.image, RUNS / "ocr")
            done += 1
            if done % 25 == 0:
                print(f"ocr {split}: {done} receipts, {time.time() - start:.0f}s", flush=True)
        print(f"ocr {split}: finished {done} receipts in {time.time() - start:.0f}s", flush=True)


def run_extract(model_dir: str, view: str, splits: list[str], limit: int | None, samples: int, threads: int | None,
                stated: bool = True) -> None:
    from .extract import Extractor

    extractor = Extractor(model_dir, threads=threads)
    out_dir = RUNS / "extract" / Path(model_dir).name / view
    out_dir.mkdir(parents=True, exist_ok=True)
    render = layout.VIEWS[view]
    for split in splits:
        start, done = time.time(), 0
        for receipt in data.load(split, limit=limit):
            path = out_dir / f"{receipt.id}.json"
            if path.exists():
                continue
            page = ocr.read_cached(receipt.id, receipt.image, RUNS / "ocr")
            result = extractor.extract(render(page), samples=samples, ask_confidence=stated)
            path.write_text(json.dumps(result, ensure_ascii=False))
            done += 1
            if done % 10 == 0:
                rate = (time.time() - start) / done
                print(f"extract {split} [{view}]: {done} new receipts, {rate:.1f}s each", flush=True)
        print(f"extract {split} [{view}]: finished, {done} new receipts in {time.time() - start:.0f}s", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fieldgate.run")
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("ocr", "extract"):
        p = sub.add_parser(name)
        p.add_argument("--splits", nargs="+", default=["validation"], choices=list(data.SPLITS))
        p.add_argument("--limit", type=int, default=None, help="receipts per split")
        if name == "extract":
            p.add_argument("--model", required=True, help="local directory of a Hugging Face causal language model")
            p.add_argument("--view", default="rows", choices=list(layout.VIEWS))
            p.add_argument("--samples", type=int, default=0, help="extra sampled extractions per receipt, for self-consistency")
            p.add_argument("--threads", type=int, default=None)
            p.add_argument("--no-stated", action="store_true", help="skip the stated-confidence question (faster)")
    args = parser.parse_args(argv)
    if args.stage == "ocr":
        run_ocr(args.splits, args.limit)
    else:
        run_extract(args.model, args.view, args.splits, args.limit, args.samples, args.threads, not args.no_stated)
    return 0


if __name__ == "__main__":
    sys.exit(main())
