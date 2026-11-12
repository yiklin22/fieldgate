"""Offline tests: no model, no dataset, no network."""
import math
import random
import unittest

from unittest import mock

from fieldgate import analyze, audit, cost, data, evaluate, layout, report, router, signals
from fieldgate.extract import parse_value
from fieldgate.ocr import Line, Page
from fieldgate.schema import FIELD_NAMES, canonical, parse_amount, same


def box(text, x, y, w=60, h=20, conf=0.99):
    """An upright OCR box whose top-left corner is (x, y)."""
    return Line(text, conf, [x, y + h, x + w, y + h, x + w, y, x, y])


def receipt(tilt_deg=0.0):
    """Label on the left, amount far to the right, one pair per row; optionally photographed at a tilt."""
    pairs = [("TOTAL", "45,500"), ("CASH", "50,000"), ("CHANGE", "4,500"), ("Subtotal", "41,364"), ("PB1", "4,136")]
    lines = []
    for i, (label, amount) in enumerate(pairs):
        for text, x in ((label, 40), (amount, 420)):
            y = 100 + 30 * i + x * math.tan(math.radians(tilt_deg))
            lines.append(box(text, x, y))
    random.Random(0).shuffle(lines)
    return Page(600, 400, lines, reading="TOTAL CASH CHANGE Subtotal PB1\n\n45,500 50,000 4,500 41,364 4,136")


def extraction(values, p=0.99, samples=None, stated=None):
    fields = {}
    for name in FIELD_NAMES:
        fields[name] = {"value": values.get(name), "tokens": 4, "sum_logp": math.log(p), "min_logp": math.log(p),
                        "first_logp": math.log(p), "p_null": 0.0, "malformed": False}
    return {"fields": fields, "samples": samples or [], "stated": stated or {}}


class Amounts(unittest.TestCase):
    def test_reads_both_separator_conventions(self):
        cases = {"45,500": 45500, "45.500": 45500, "Rp 1.250.000": 1250000, "12.50": 12.5, "1,234.5": 1234.5,
                 "-5,000": -5000, "0": 0, "8.00": 8.0, "7": 7, "45 500": 45500}
        for text, value in cases.items():
            self.assertEqual(parse_amount(text), value, text)

    def test_refuses_what_is_not_one_number(self):
        for text in ("", "abc", "10/20", "2 x 5,000", None, "1,23,4"):
            self.assertIsNone(parse_amount(text), text)

    def test_absent_and_zero_match_only_when_asked(self):
        self.assertTrue(same("tax", None, 0.0))
        self.assertTrue(same("tax", "0", None))
        self.assertFalse(same("tax", "0", None, zero_is_absent=False))
        self.assertFalse(same("total", "45,500", 45000.0))
        self.assertTrue(same("total", "Rp 45.500", 45500.0))
        self.assertTrue(same("discount", "-5,000", 5000.0))             # discounts print with either sign
        self.assertEqual(canonical("item_count", "3"), 3.0)

    def test_labels_mark_ambiguous_fields_unusable(self):
        gold, unusable = data.labels({"menu": [{"nm": "A", "price": "1"}, {"nm": "B", "price": "2"}],
                                      "total": {"total_price": "3,000", "creditcardprice": ["1,000", "2,000"]},
                                      "sub_total": {"tax_price": "ten percent"}})
        self.assertEqual(gold["total"], 3000.0)
        self.assertEqual(gold["item_count"], 2.0)
        self.assertIsNone(gold["cash"])
        self.assertEqual(unusable, {"card", "tax"})


class ValueParsing(unittest.TestCase):
    def test_waits_for_a_complete_scalar(self):
        self.assertEqual(parse_value(' "45,5'), (False, None, 0, False))
        self.assertEqual(parse_value(" nu"), (False, None, 0, False))
        self.assertEqual(parse_value(' "45,500"'), (True, "45,500", 9, False))
        self.assertEqual(parse_value(' "45,500",'), (True, "45,500", 9, False))      # an over-run token is cut off
        self.assertEqual(parse_value(" null,"), (True, None, 5, False))

    def test_flags_what_is_not_json(self):
        self.assertEqual(parse_value(" 45500,"), (False, None, 0, False))           # a comma can still be part of the number
        self.assertEqual(parse_value(" 45500, "), (True, "45500", 6, True))
        done, value, end, malformed = parse_value(" sorry")
        self.assertTrue(done and malformed and value is None and end == 0)


class Rows(unittest.TestCase):
    def test_pairs_labels_with_amounts(self):
        text = layout.as_rows(receipt())
        self.assertEqual(text.splitlines()[0], "TOTAL\t45,500")
        self.assertIn("PB1\t4,136", text)

    def test_a_tilted_photo_is_straightened(self):
        page = receipt(tilt_deg=3.0)                 # 380 px between columns moves the amount most of a row down
        self.assertNotEqual(layout.as_rows(Page(600, 400, page.lines, "")).splitlines()[0], "")
        self.assertAlmostEqual(math.degrees(layout.skew(page.lines)), 3.0, delta=0.5)
        self.assertEqual(layout.as_rows(page).splitlines()[0], "TOTAL\t45,500")
        level = layout.rows(Page(600, 400, page.lines, ""), tolerance=0.6)
        self.assertEqual(len(level), 5)

    def test_a_tilt_of_one_whole_row_cannot_be_told_from_the_boxes(self):
        # Known limit: at 5 degrees this receipt's amounts sit level with the next row's label, and upright
        # boxes carry no other clue. The estimate settles on the small rotation and pairs the rows wrongly.
        page = receipt(tilt_deg=5.0)
        self.assertLess(math.degrees(layout.skew(page.lines)), 1.0)
        self.assertNotIn("TOTAL\t45,500", layout.as_rows(page))

    def test_empty_page(self):
        self.assertEqual(layout.as_rows(Page(10, 10, [], "")), "")


class Signals(unittest.TestCase):
    def test_source_and_rule_signals(self):
        values = {"total": "45,500", "cash": "50,000", "change": "4,500", "subtotal": "41,364", "tax": "4,136"}
        rows = {r["field"]: r for r in signals.rows_for("d1", receipt(), extraction(values))}
        self.assertEqual(rows["total"]["printed"], 1.0)
        self.assertEqual(rows["total"]["label_on_row"], 1.0)
        self.assertEqual(rows["total"]["sum_ok"], 1.0)
        self.assertEqual(rows["service"]["is_null"], 1.0)
        self.assertEqual(rows["service"]["label_in_doc"], 0.0)
        self.assertEqual(rows["tax"]["label_on_row"], 1.0)                # PB1 is the tax label on these receipts
        self.assertNotIn("correct", rows["total"])                        # no label given, none invented

    def test_a_value_that_is_not_on_the_page_or_breaks_the_sum(self):
        values = {"total": "54,500", "cash": "50,000", "change": "4,500"}
        rows = {r["field"]: r for r in signals.rows_for("d1", receipt(), extraction(values))}
        self.assertEqual(rows["total"]["printed"], 0.0)
        self.assertEqual(rows["total"]["sum_bad"], 1.0)
        self.assertEqual(rows["cash"]["sum_bad"], 1.0)

    def test_subtotal_is_not_taken_for_total(self):
        self.assertIsNone(signals._KEY["total"].search("Subtotal 41,364"))
        self.assertIsNotNone(signals._KEY["total"].search("GRAND TOTAL 45,500"))

    def test_agreement_and_labels(self):
        ex = extraction({"total": "45,500"}, samples=[{"total": "45,500"}, {"total": "45.500"}, {"total": "50,000"}],
                        stated={"total": 80})
        gold = dict.fromkeys(FIELD_NAMES)
        gold["total"] = 45500.0
        rows = {r["field"]: r for r in signals.rows_for("d1", receipt(), ex, gold, unusable={"card"})}
        self.assertAlmostEqual(rows["total"]["agree"], 2 / 3)
        self.assertEqual(rows["total"]["stated"], 0.8)
        self.assertTrue(rows["total"]["correct"])
        self.assertNotIn("card", rows)
        self.assertEqual(rows["tax"]["stated_missing"], 1.0)


class Metrics(unittest.TestCase):
    def test_auroc(self):
        self.assertEqual(evaluate.auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]), 1.0)
        self.assertEqual(evaluate.auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]), 0.0)
        self.assertEqual(evaluate.auroc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]), 0.5)
        self.assertIsNone(evaluate.auroc([0.5, 0.6], [True, True]))

    def test_risk_coverage_and_aurc(self):
        scores, correct = [0.9, 0.8, 0.7, 0.6], [True, True, False, True]
        self.assertEqual([(c, round(r, 3)) for c, r, _ in evaluate.risk_coverage(scores, correct)],
                         [(0.25, 0.0), (0.5, 0.0), (0.75, 0.333), (1.0, 0.25)])
        self.assertAlmostEqual(evaluate.aurc(scores, correct), (0 + 0 + 1 / 3 + 1 / 4) / 4)
        self.assertEqual(evaluate.coverage_at_risk(scores, correct, 0.0), 0.5)
        self.assertEqual(evaluate.coverage_at_risk(scores, correct, 0.25), 1.0)

    def test_a_useless_score_has_aurc_equal_to_the_error_rate(self):
        correct = [True] * 80 + [False] * 20
        self.assertAlmostEqual(evaluate.aurc([0.5] * 100, correct), 0.2, places=2)

    def test_bootstrap_resamples_receipts(self):
        rows = [{"doc": f"d{i}", "correct": i % 4 != 0} for i in range(40) for _ in range(3)]
        lo, hi = evaluate.bootstrap(rows, lambda rs: sum(r["correct"] for r in rs) / len(rs), draws=300)
        self.assertLess(lo, 0.75)
        self.assertGreater(hi, 0.75)
        self.assertGreater(hi - lo, 0.15)            # wider than treating the 120 rows as independent would give


class Scorer(unittest.TestCase):
    def make(self, n, seed):
        rng = random.Random(seed)
        rows = []
        for d in range(n):
            for name in FIELD_NAMES:
                good = rng.random() < 0.8
                row = dict.fromkeys(signals.SIGNAL_NAMES, 0.0)
                row.update(doc=f"d{d}", field=name, correct=good,
                           printed=float(rng.random() < (0.95 if good else 0.3)),
                           p_value=min(1.0, max(0.0, rng.gauss(0.9 if good else 0.7, 0.15))))
                rows.append(row)
        return rows

    def test_combining_signals_beats_either_alone(self):
        train, test = self.make(150, 1), self.make(150, 2)
        correct = [r["correct"] for r in test]
        combined = evaluate.auroc(router.score(router.fit(train), test), correct)
        self.assertGreater(combined, evaluate.auroc([r["p_value"] for r in test], correct) + 0.03)
        self.assertGreater(combined, evaluate.auroc([r["printed"] for r in test], correct) + 0.03)
        self.assertEqual(router.weights(router.fit(train))[0][0] in ("printed", "p_value"), True)

    def test_cross_fit_never_scores_a_receipt_with_its_own_model(self):
        rows = self.make(40, 3)
        for r in rows:                                # a signal that leaks the label on exactly one receipt
            r["doc_lines"] = 1000.0 if (r["doc"] == "d0" and r["correct"]) else 0.0
        out = router.cross_fit(rows, folds=5)
        self.assertEqual(len(out), len(rows))
        self.assertTrue(all(0.0 <= s <= 1.0 for s in out))

    def test_needs_both_outcomes(self):
        rows = [r for r in self.make(10, 4) if r["correct"]]
        with self.assertRaises(ValueError):
            router.fit(rows)


class Analysis(unittest.TestCase):
    """The whole experiment on made-up signal rows, so the splits, certificates and report are exercised in CI."""

    def rows(self):
        rng = random.Random(5)
        rows = []
        for split, n in (("train", 240), ("validation", 40), ("test", 40)):
            for d in range(n):
                for name in FIELD_NAMES:
                    good = rng.random() < 0.85
                    row = dict.fromkeys(signals.SIGNAL_NAMES, 0.0)
                    row.update(doc=f"{split}-{d:04d}", field=name, correct=good, gold=1.0, value="1",
                               printed=float(rng.random() < (0.97 if good else 0.15)),
                               agree=1.0 if good or rng.random() < 0.4 else 0.33,
                               p_value=min(1.0, max(0.0, rng.gauss(0.95 if good else 0.6, 0.1))))
                    rows.append(row)
        return rows

    def test_runs_end_to_end_and_keeps_its_splits_apart(self):
        rows = self.rows()
        fit, cal, test = analyze.split_rows(rows)
        docs = [{r["doc"] for r in part} for part in (fit, cal, test)]
        self.assertFalse(docs[0] & docs[1] or docs[0] & docs[2] or docs[1] & docs[2])
        self.assertTrue(all(d.startswith("train") for d in docs[0] | docs[1]))
        self.assertEqual(len(docs[2]), 80)
        with mock.patch.object(analyze, "load_rows", return_value=rows):
            result = analyze.analyse("made-up", "rows", draws=40, repeats=4)
        self.assertEqual(result["docs"], {"fit": 120, "calibrate": 120, "test": 80})
        self.assertEqual(len(result["certificates"]), 3 * (len(FIELD_NAMES) + 1))
        for c in result["certificates"]:
            self.assertLessEqual(c["test_approved"], c["test_n"])
            if c["threshold"] is not None:
                self.assertLessEqual(c["bound"], c["alpha"] + 1e-9)        # a certificate never claims more than it proved
        strict = [c for c in result["certificates"] if c["alpha"] == 0.02 and c["unit"] == "field"]
        self.assertTrue(all(c["threshold"] is None for c in strict))       # 120 receipts cannot certify 2%: that needs 149
        everything, nothing, routed = result["costs"][0], result["costs"][1], result["costs"][-1]
        self.assertLess(routed["labour"], everything["labour"])
        self.assertLess(routed["shipped"], nothing["shipped"])
        self.assertEqual(everything["shipped"], 0.0)
        text = report.render(result)
        for heading in ("Which score finds the wrong values", "Certified auto-approval", "What the routing is worth",
                        "How many labelled receipts a target needs"):
            self.assertIn(heading, text)

    def test_document_score_is_the_product_and_one_wrong_field_fails_it(self):
        rows = [{"doc": "a", "correct": True}, {"doc": "a", "correct": False}, {"doc": "b", "correct": True}]
        scores, ok, names = analyze.by_document(rows, [0.9, 0.5, 0.8])
        self.assertEqual(names, ["a", "b"])
        self.assertEqual(ok, [False, True])
        self.assertAlmostEqual(scores[0], 0.45)


class Audit(unittest.TestCase):
    def run_stream(self, error_rate, n, seed):
        rng = random.Random(seed)
        monitor = audit.Monitor(alpha=0.02)
        for _ in range(n):
            monitor.record(rng.random() < error_rate)
        return monitor

    def test_stays_quiet_when_the_cut_holds(self):
        alarms = sum(self.run_stream(0.02, 1500, s).alarmed for s in range(300))
        self.assertLessEqual(alarms / 300, 0.05 + 0.04)

    def test_raises_the_alarm_after_drift(self):
        fired = [self.run_stream(0.10, 400, s).alarm_at for s in range(100)]
        self.assertTrue(all(f is not None for f in fired))
        self.assertLess(sorted(fired)[50], 120)       # typically within about a hundred audited decisions

    def test_audit_choice_is_stable_and_near_its_rate(self):
        picks = [audit.choose_for_audit(f"doc-{i}:total", 0.1) for i in range(5000)]
        self.assertEqual(picks, [audit.choose_for_audit(f"doc-{i}:total", 0.1) for i in range(5000)])
        self.assertAlmostEqual(sum(picks) / 5000, 0.1, delta=0.02)


class Cost(unittest.TestCase):
    ROWS = [{"doc": "a", "field": "total", "correct": True}, {"doc": "a", "field": "tax", "correct": False},
            {"doc": "b", "field": "total", "correct": True}, {"doc": "b", "field": "tax", "correct": True}]

    def test_review_everything_ships_no_errors(self):
        out = cost.policy(self.ROWS, lambda r: False)
        self.assertEqual(out["shipped"], 0.0)
        self.assertEqual(out["touchless"], 0.0)
        self.assertAlmostEqual(out["labour"], (2 * 12 + 4 * 6) / 3600 * 18 / 2)

    def test_approve_everything_ships_every_error(self):
        out = cost.policy(self.ROWS, lambda r: True)
        self.assertEqual(out["labour"], 0.0)
        self.assertEqual(out["shipped"], 0.5)
        self.assertEqual(cost.total(out, 10.0), 5.0)
        self.assertEqual(out["touchless"], 1.0)

    def test_routing_the_wrong_field_is_cheapest(self):
        routed = cost.policy(self.ROWS, lambda r: r["correct"])
        self.assertLess(cost.total(routed, 5.0), cost.total(cost.policy(self.ROWS, lambda r: False), 5.0))
        self.assertLess(cost.total(routed, 5.0), cost.total(cost.policy(self.ROWS, lambda r: True), 5.0))
        self.assertEqual(routed["touchless"], 0.5)

    def test_break_even_is_check_cost_over_error_cost(self):
        self.assertAlmostEqual(cost.break_even(3.0), 0.01)           # a 3-cent check against a 3-dollar error
        self.assertEqual(cost.break_even(0.001), 1.0)
        approve_all = cost.policy([{"doc": str(i), "field": "total", "correct": i >= 2} for i in range(100)], lambda r: True)
        review_all = cost.policy([{"doc": str(i), "field": "total", "correct": i >= 2} for i in range(100)],
                                 lambda r: False, cost.Costs(seconds_to_open=0))
        self.assertLess(cost.total(approve_all, 1.0), review_all["labour"])      # 2% wrong is under the 3% break-even at $1
        self.assertGreater(cost.total(approve_all, 3.0), review_all["labour"])   # and over the 1% break-even at $3


if __name__ == "__main__":
    unittest.main()
