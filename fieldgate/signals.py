"""Evidence about whether an extracted value is right, computed without the label.

One row per (receipt, field). The signals come from four independent places, which is the point:
a value can look certain to the model and still fail a check the model never ran.

  model      token probabilities of the value, and how much the model leaned towards "absent"
  agreement  share of re-sampled extractions that gave the same value; the model's stated confidence
  source     is the value printed on the receipt, how often, how cleanly was it read, and does a
             label for this field sit on the same row
  rules      do the amounts add up (cash - total = change; subtotal + tax + service - discount = total)
"""
from __future__ import annotations

import math
import re
from statistics import mean

from . import layout
from .ocr import Page
from .schema import FIELD_NAMES, canonical, parse_amount, same

# Words that label each field on these receipts (English and Indonesian). Deployment-specific:
# for another document type this table is the first thing to rewrite.
KEYWORDS = {
    "total": r"(?<!sub)(?<!sub )(?<!sub-)\btotal\b|grand\s*total|\bjumlah\b|\bamount\b",
    "subtotal": r"sub\s*-?\s*total|sub\s*ttl|\bnet\b",
    "tax": r"\btax\b|pajak|p\.?\s?b\.?\s?1|\bppn\b|\bvat\b",
    "service": r"service|servis|\bsvc\b|\bserv\b",
    "discount": r"disc|diskon|potongan|voucher",
    "cash": r"\bcash\b|tunai",
    "change": r"change|kembali",
    "card": r"card|debit|credit|kartu|\bedc\b|visa|master",
    "item_count": r"$^",
}
_KEY = {k: re.compile(v, re.I) for k, v in KEYWORDS.items()}
_NUMBER = re.compile(r"-?\d[\d.,]*")

SIGNAL_NAMES = (
    "is_null", "p_value", "min_token_p", "first_token_p", "p_null", "tokens", "malformed",
    "agree", "stated", "stated_missing",
    "printed", "printed_twice", "ocr_conf", "label_on_row", "label_in_doc",
    "sum_ok", "sum_bad", "doc_ocr_conf", "doc_lines", "doc_tilt",
)


def printed_numbers(page: Page) -> list[dict]:
    """Every number OCR read, with the row it sits on and how confidently it was read."""
    found = []
    for r, row in enumerate(layout.rows(page)):
        row_text = " ".join(l.text for l in row)
        for line in row:
            for token in _NUMBER.findall(line.text):
                value = parse_amount(token)
                if value is not None:
                    found.append({"value": abs(value), "conf": line.conf, "row": r, "row_text": row_text})
    return found


def _checks(values: dict[str, float | None]) -> dict[str, int]:
    """+1 for each field in an arithmetic identity that holds, -1 for each in one that fails."""
    verdict = dict.fromkeys(FIELD_NAMES, 0)
    v = {k: values.get(k) for k in FIELD_NAMES}

    def judge(fields: tuple[str, ...], holds: bool) -> None:
        for f in fields:
            if verdict[f] >= 0:                      # a failed identity is not rescued by another that holds
                verdict[f] = 1 if holds else -1

    if v["cash"] is not None and v["change"] is not None and v["total"] is not None:
        judge(("cash", "change", "total"), abs(v["cash"] - v["total"] - v["change"]) < 1)
    if v["subtotal"] is not None and v["total"] is not None and any(v[k] for k in ("tax", "service", "discount")):
        expect = v["subtotal"] + (v["tax"] or 0) + (v["service"] or 0) - (v["discount"] or 0)
        involved = ("subtotal", "total") + tuple(k for k in ("tax", "service", "discount") if v[k] is not None)
        judge(involved, abs(expect - v["total"]) <= 2)       # receipts round to the unit
    return verdict


def rows_for(doc_id: str, page: Page, extraction: dict, gold: dict | None = None,
             unusable: set[str] | None = None) -> list[dict]:
    """Signal rows for one receipt. With `gold`, each row also carries `correct`."""
    numbers = printed_numbers(page)
    all_text = " ".join(l.text for l in page.lines)
    fields = extraction["fields"]
    values = {k: canonical(k, fields[k]["value"]) for k in FIELD_NAMES}
    checks = _checks(values)
    doc_conf = mean(l.conf for l in page.lines) if page.lines else 0.0
    tilt = abs(math.degrees(layout.skew(page.lines)))
    out = []
    for name in FIELD_NAMES:
        f = fields[name]
        value = values[name]
        samples = extraction.get("samples") or []
        agree = mean(same(name, s.get(name), f["value"], zero_is_absent=False) for s in samples) if samples else 1.0
        stated = (extraction.get("stated") or {}).get(name)
        hits = [n for n in numbers if value is not None and abs(n["value"] - abs(value)) < 0.005]
        row = {
            "doc": doc_id, "field": name, "value": f["value"],
            "is_null": float(value is None),
            "p_value": math.exp(f["sum_logp"]),
            "min_token_p": math.exp(f["min_logp"]),
            "first_token_p": math.exp(f["first_logp"]),
            "p_null": f["p_null"],
            "tokens": float(f["tokens"]),
            "malformed": float(bool(f["malformed"])),
            "agree": float(agree),
            "stated": (stated if stated is not None else 50) / 100,
            "stated_missing": float(stated is None),
            "printed": float(bool(hits)),
            "printed_twice": float(len(hits) >= 2),
            "ocr_conf": max((h["conf"] for h in hits), default=0.0),
            "label_on_row": float(any(_KEY[name].search(h["row_text"]) for h in hits)),
            "label_in_doc": float(bool(_KEY[name].search(all_text))),
            "sum_ok": float(checks[name] > 0),
            "sum_bad": float(checks[name] < 0),
            "doc_ocr_conf": doc_conf,
            "doc_lines": float(len(page.lines)),
            "doc_tilt": tilt,
        }
        if gold is not None:
            if unusable and name in unusable:
                continue
            row["gold"] = gold[name]
            row["correct"] = bool(same(name, f["value"], gold[name]))
        out.append(row)
    return out
