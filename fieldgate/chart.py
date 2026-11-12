"""Draw the risk-coverage chart for the README from the analysis JSON.

    python -m fieldgate.analyze --json docs/results/cord.json --out docs/results/cord.md
    python -m fieldgate.chart docs/results/cord.json docs/img/risk-coverage

Writes `<out>-light.svg` and `<out>-dark.svg`. Reading the chart: move right to approve more without
review; the height of a line is the share of those approvals that are wrong. A better score stays
low for longer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

THEMES = {
    "light": {"bg": "#ffffff", "ink": "#0c1230", "soft": "#525c7e", "rule": "#d3d8e8",
              "series": ["#1a4fd6", "#a36100", "#0b7a6c", "#c4382f"]},
    "dark": {"bg": "#0d1117", "ink": "#eef1fa", "soft": "#a3add0", "rule": "#2b3558",
             "series": ["#6cb4ff", "#ffbe5c", "#5fd6c4", "#ff7a7a"]},
}
ORDER = [("combined, all signals", "All signals combined", "", 3.0),
         ("model probability of the value", "Token probability", "", 2.0),
         ("agreement of 3 re-samples", "3 re-samples agree", "7 4", 2.0),
         ("model's stated confidence", "Stated confidence", "2 4", 2.0)]
W, H, X0, X1, TOP, BOTTOM = 860, 430, 64, 836, 78, 372
FONT = "-apple-system, Segoe UI, Helvetica, Arial, sans-serif"


def render(result: dict, theme: str) -> str:
    t = THEMES[theme]
    curves = result["curves"]
    base = max(points[-1][1] for points in curves.values())
    top = max(0.05, min(1.0, 1.6 * base))
    ticks = [k * (0.05 if top > 0.12 else 0.02) for k in range(0, 40) if k * (0.05 if top > 0.12 else 0.02) <= top + 1e-9]

    def x(c: float) -> float:
        return X0 + c * (X1 - X0)

    def y(r: float) -> float:
        return BOTTOM - min(r, top) / top * (BOTTOM - TOP)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
           f'aria-label="Error rate among approved fields against the share of fields approved, for four confidence scores">',
           f'<rect width="{W}" height="{H}" fill="{t["bg"]}"/>',
           f'<text x="{X0}" y="26" font-family="{FONT}" font-size="16" font-weight="600" fill="{t["ink"]}">'
           "Combining independent evidence keeps approved fields cleaner for longer</text>",
           f'<text x="{X0}" y="46" font-family="{FONT}" font-size="12" fill="{t["soft"]}">Share of approved fields that are wrong '
           f'(lower is better), {result["rows"]["test"]:,} fields from {result["docs"]["test"]} held-out receipts</text>']
    lx = X0
    for i, (key, label, dash, width) in enumerate(ORDER):
        if key not in curves:
            continue
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<path d="M{lx} 62 h26" stroke="{t["series"][i]}" stroke-width="{width}"{dash_attr} fill="none"/>'
                   f'<text x="{lx + 32}" y="66" font-family="{FONT}" font-size="12" fill="{t["ink"]}">{label}</text>')
        lx += 32 + 6.4 * len(label) + 24
    for tick in ticks:
        out.append(f'<path d="M{X0} {y(tick):.1f} H{X1}" stroke="{t["rule"]}" stroke-width="1" fill="none"/>'
                   f'<text x="{X0 - 8}" y="{y(tick) + 4:.1f}" text-anchor="end" font-family="{FONT}" font-size="12" '
                   f'fill="{t["soft"]}">{100 * tick:.0f}%</text>')
    for c in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<text x="{x(c):.1f}" y="{BOTTOM + 20}" text-anchor="middle" font-family="{FONT}" font-size="12" '
                   f'fill="{t["soft"]}">{100 * c:.0f}%</text>')
    out.append(f'<text x="{(X0 + X1) / 2:.0f}" y="{BOTTOM + 42}" text-anchor="middle" font-family="{FONT}" font-size="12" '
               f'fill="{t["soft"]}">share of fields approved without review, most confident first</text>')
    for i, (key, label, dash, width) in reversed(list(enumerate(ORDER))):
        if key not in curves:
            continue
        points = curves[key]
        path = f"M{x(0):.1f} {y(points[0][1]):.1f} L" + " L".join(f"{x(c):.1f} {y(r):.1f}" for c, r in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<path d="{path}" stroke="{t["series"][i]}" stroke-width="{width}"{dash_attr} fill="none" stroke-linejoin="round"/>')
    note = f"approve everything: {100 * base:.1f}% wrong"
    out.append(f'<rect x="{x(1) - 16 - 6.6 * len(note):.1f}" y="{y(base) - 31:.1f}" width="{6.6 * len(note) + 12:.1f}" height="22" rx="3" '
               f'fill="{t["bg"]}" fill-opacity="0.9"/>'
               f'<text x="{x(1) - 10:.1f}" y="{y(base) - 16:.1f}" text-anchor="end" font-family="{FONT}" font-size="12" '
               f'fill="{t["ink"]}">{note}</text>'
               f'<circle cx="{x(1):.1f}" cy="{y(base):.1f}" r="4.5" fill="{t["ink"]}" stroke="{t["bg"]}" stroke-width="2"/>')
    out.append("</svg>")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(__doc__)
        return 2
    result = json.loads(Path(args[0]).read_text())
    for theme in THEMES:
        path = Path(f"{args[1]}-{theme}.svg")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render(result, theme))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
