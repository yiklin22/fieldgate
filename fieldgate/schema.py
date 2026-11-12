"""The fields extracted from a receipt, and how a predicted value is compared with the label.

Every field is a money amount or a count that may be absent from the receipt. "Absent" is a real
answer: predicting a value for a field the receipt does not show is an error, and so is missing one
that is there.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Field:
    name: str
    kind: str          # "money" or "count"
    describe: str      # shown to the model
    cord_path: str     # where the label lives in CORD's gt_parse


FIELDS: tuple[Field, ...] = (
    Field("total", "money", "the final amount due, after tax, service and discounts", "total.total_price"),
    Field("subtotal", "money", "the sum of the items before tax and service", "sub_total.subtotal_price"),
    Field("tax", "money", "the tax amount (for example PB1, PPN, VAT, Tax)", "sub_total.tax_price"),
    Field("service", "money", "the service charge amount", "sub_total.service_price"),
    Field("discount", "money", "the total discount amount", "sub_total.discount_price"),
    Field("cash", "money", "the cash the customer handed over", "total.cashprice"),
    Field("change", "money", "the change returned to the customer", "total.changeprice"),
    Field("card", "money", "the amount paid by credit or debit card", "total.creditcardprice"),
    Field("item_count", "count", "the number of separate item lines sold (not the quantity of units)", "menu"),
)
FIELD_NAMES = tuple(f.name for f in FIELDS)
BY_NAME = {f.name: f for f in FIELDS}

_NUM = re.compile(r"-?\d[\d.,]*")


def parse_amount(text: object) -> float | None:
    """Read a printed amount such as "Rp 45.500", "45,500", "12.50" or "-5,000" as a number.

    A separator followed by exactly three digits is a thousands separator; a final separator
    followed by one or two digits is a decimal point. Returns None when no single number can be read.
    """
    if text is None:
        return None
    s = str(text).replace(" ", "")
    found = _NUM.findall(s)
    if len(found) != 1:
        return None
    raw = found[0].rstrip(".,")
    negative = raw.startswith("-") or bool(re.search(r"\(\s*\d", str(text)))
    raw = raw.lstrip("-")
    parts = re.split(r"[.,]", raw)
    if len(parts) == 1:
        value = float(parts[0])
    elif all(len(p) == 3 for p in parts[1:]):
        value = float("".join(parts))
    elif all(len(p) == 3 for p in parts[1:-1]) and len(parts[-1]) in (1, 2):
        value = float("".join(parts[:-1]) + "." + parts[-1])
    else:
        return None
    return -value if negative else value


def canonical(field: str, value: object) -> float | None:
    """The comparable form of a value for `field`, or None for absent or unreadable."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
    else:
        text = str(value).strip()
        if text == "" or text.lower() in ("null", "none", "n/a", "-"):
            return None
        number = parse_amount(text)
        if number is None:
            return None
    if BY_NAME[field].kind == "count":
        return float(round(number))
    return abs(number) if field == "discount" else number     # receipts print discounts with either sign


def same(field: str, predicted: object, gold: object, zero_is_absent: bool = True) -> bool:
    """True when the prediction matches the label.

    By default an absent amount and a printed zero count as the same answer, because they post the
    same number downstream. Pass `zero_is_absent=False` to require the model to tell them apart.
    """
    a, b = canonical(field, predicted), canonical(field, gold)
    if zero_is_absent:
        a, b = a or 0.0, b or 0.0
        return abs(a - b) < 0.005
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) < 0.005
