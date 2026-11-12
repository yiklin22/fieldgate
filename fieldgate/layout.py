"""Rebuild the printed rows of a receipt from OCR line boxes.

A receipt prints a label and its amount on the same row, far apart. Layout models read columns, so a
reading-order export can list every label first and every amount after, and the pairing is lost.
Here the page is de-skewed from the boxes themselves and lines that share a row are joined left to right.
"""
from __future__ import annotations

import math
from statistics import median

from .ocr import Line, Page


def _centre(q: list[float]) -> tuple[float, float]:
    return (q[0] + q[2] + q[4] + q[6]) / 4, (q[1] + q[3] + q[5] + q[7]) / 4


def _height(q: list[float]) -> float:
    return (math.dist((q[0], q[1]), (q[6], q[7])) + math.dist((q[2], q[3]), (q[4], q[5]))) / 2


def _width(q: list[float]) -> float:
    return (math.dist((q[0], q[1]), (q[2], q[3])) + math.dist((q[6], q[7]), (q[4], q[5]))) / 2


def _rotate(x: float, y: float, angle: float) -> tuple[float, float]:
    """The point as it would sit on the page turned back by `angle`."""
    cos, sin = math.cos(angle), math.sin(angle)
    return x * cos + y * sin, -x * sin + y * cos


def skew(lines: list[Line], limit_deg: float = 10.0, step_deg: float = 0.25, prior: float = 0.6) -> float:
    """Page rotation in radians.

    The OCR boxes are upright rectangles even on a tilted photo, so the tilt cannot be read from a
    single box. Instead, try each angle and keep the one at which the most pairs of boxes line up
    on a shared row. Ties go to the smaller rotation.
    """
    if len(lines) < 4:
        return 0.0
    centres = [_centre(l.quad) for l in lines]
    spans = [(min(l.quad[0::2]), max(l.quad[0::2])) for l in lines]
    close = 0.5 * median(_height(l.quad) for l in lines)
    best_angle, best_score = 0.0, -1.0
    steps = int(limit_deg / step_deg)
    for k in sorted(range(-steps, steps + 1), key=abs):
        angle = math.radians(k * step_deg)
        ys = [_rotate(x, y, angle)[1] for x, y in centres]
        score = 0.0
        for i in range(len(ys)):
            for j in range(i + 1, len(ys)):
                apart = spans[i][1] < spans[j][0] or spans[j][1] < spans[i][0]
                if apart:
                    score += max(0.0, 1 - abs(ys[i] - ys[j]) / close) ** 2
        # A tilt of a few degrees moves a wide receipt by one whole row, which lines boxes up just
        # as well with the wrong neighbours. Ask a larger rotation to be clearly better.
        score *= 1 - prior * abs(k) / steps
        if score > best_score:
            best_angle, best_score = angle, score
    return best_angle


def rows(page: Page, tolerance: float = 0.6) -> list[list[Line]]:
    """Group OCR lines into printed rows, top to bottom, each row left to right."""
    if not page.lines:
        return []
    angle = skew(page.lines)
    placed = []
    for l in page.lines:
        x, y = _rotate(*_centre(l.quad), angle)
        placed.append((y, x, _height(l.quad), l))
    placed.sort(key=lambda p: p[0])
    grouped: list[list[tuple]] = []
    for item in placed:
        if grouped:
            row = grouped[-1]
            row_y = sum(p[0] for p in row) / len(row)
            row_h = median(p[2] for p in row)
            if abs(item[0] - row_y) <= tolerance * min(row_h, item[2]):
                row.append(item)
                continue
        grouped.append([item])
    return [[p[3] for p in sorted(row, key=lambda p: p[1])] for row in grouped]


def as_rows(page: Page) -> str:
    """One printed row per text line; a tab between boxes that sat apart on the row."""
    return "\n".join("\t".join(l.text for l in row) for row in rows(page))


def as_reading(page: Page) -> str:
    return page.reading


VIEWS = {"rows": as_rows, "reading": as_reading}
