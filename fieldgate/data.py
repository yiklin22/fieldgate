"""Load CORD v2 (Park et al., 2019; CC BY 4.0) receipts from the parquet files published on Hugging Face.

    https://huggingface.co/datasets/naver-clova-ix/cord-v2

Labels are read from `gt_parse`. A field whose label cannot be read as one number (two totals on one
receipt, free text in a price slot) is marked unusable for that receipt and left out of scoring; the
count of those is reported, not hidden.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .schema import FIELDS, canonical

SPLITS = {
    "train": ["cord-train-0.parquet", "cord-train-1.parquet", "cord-train-2.parquet", "cord-train-3.parquet"],
    "validation": ["cord-validation.parquet"],
    "test": ["cord-test.parquet"],
}
HF_BASE = "https://huggingface.co/datasets/naver-clova-ix/cord-v2/resolve/main/data/"


@dataclass
class Receipt:
    id: str                               # for example "validation-0007"
    image: bytes
    gold: dict[str, float | None]         # canonical label per field; None means absent
    unusable: set[str] = field(default_factory=set)
    words: list[dict] = field(default_factory=list)   # the dataset's own word boxes: text, quad


def _dig(tree: object, path: str) -> list[object]:
    """Every leaf reached by `path`, looking through lists (CORD uses a dict for one entry, a list for several)."""
    nodes = [tree]
    for key in path.split("."):
        nxt: list[object] = []
        for node in nodes:
            for one in (node if isinstance(node, list) else [node]):
                if isinstance(one, dict) and key in one:
                    nxt.append(one[key])
        nodes = nxt
    leaves: list[object] = []
    for node in nodes:
        leaves.extend(node if isinstance(node, list) else [node])
    return leaves


def labels(gt_parse: dict) -> tuple[dict[str, float | None], set[str]]:
    gold: dict[str, float | None] = {}
    unusable: set[str] = set()
    for f in FIELDS:
        if f.name == "item_count":
            menu = gt_parse.get("menu")
            entries = menu if isinstance(menu, list) else ([menu] if menu else [])
            lines = [m for m in entries if isinstance(m, dict) and (m.get("nm") or m.get("price"))]
            gold[f.name] = float(len(lines)) if lines else None
            continue
        leaves = [v for v in _dig(gt_parse, f.cord_path) if v not in (None, "")]
        if not leaves:
            gold[f.name] = None
        elif len(leaves) > 1 or isinstance(leaves[0], (dict, list)):
            gold[f.name] = None
            unusable.add(f.name)
        else:
            value = canonical(f.name, leaves[0])
            gold[f.name] = value
            if value is None:
                unusable.add(f.name)
    return gold, unusable


def load(split: str, data_dir: str | Path = ".cache/data", limit: int | None = None) -> Iterator[Receipt]:
    import pyarrow.parquet as pq

    n = 0
    for name in SPLITS[split]:
        path = Path(data_dir) / name
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing. Run `python -m fieldgate.fetch` to download CORD.")
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=16):
            for row in batch.to_pylist():
                if limit is not None and n >= limit:
                    return
                truth = json.loads(row["ground_truth"])
                gold, unusable = labels(truth["gt_parse"])
                words = [{"text": w["text"], "quad": w["quad"]} for line in truth.get("valid_line", [])
                         for w in line.get("words", [])]
                yield Receipt(id=f"{split}-{n:04d}", image=row["image"]["bytes"], gold=gold,
                              unusable=unusable, words=words)
                n += 1
