"""The guarantee is checked against a population whose true error rate is known in closed form."""
import random
import unittest

from fieldgate import certify


def population(n: int, rng: random.Random):
    """Scores uniform on [0, 1]; a value with score s is wrong with probability 2 * (1 - s)^3."""
    scores = [rng.random() for _ in range(n)]
    wrong = [rng.random() < 2 * (1 - s) ** 3 for s in scores]
    return scores, wrong


def true_risk(cut: float) -> float:
    """Error rate among decisions scoring at or above `cut` in that population: within target above 0.658."""
    return 0.5 * (1 - cut) ** 3


class Bounds(unittest.TestCase):
    def test_upper_bound_matches_the_rule_of_three(self):
        self.assertAlmostEqual(certify.upper_bound(0, 100, 0.05), 1 - 0.05 ** (1 / 100), places=9)
        self.assertEqual(certify.upper_bound(0, 0, 0.05), 1.0)
        self.assertEqual(certify.upper_bound(5, 5, 0.05), 1.0)

    def test_needed_is_the_smallest_sample_that_passes(self):
        for alpha, errors in ((0.01, 0), (0.02, 0), (0.02, 1), (0.05, 3)):
            n = certify.needed(alpha, 0.05, errors)
            self.assertLessEqual(certify.p_value(errors, n, alpha), 0.05)
            self.assertGreater(certify.p_value(errors, n - 1, alpha), 0.05)
        self.assertEqual(certify.needed(0.02, 0.05), 149)
        self.assertEqual(certify.needed(0.01, 0.05), 299)

    def test_grid_is_fixed_and_strict_to_loose(self):
        self.assertEqual(list(certify.GRID), sorted(certify.GRID, reverse=True))
        self.assertEqual(certify.GRID[-1], 0.5)


class Guarantee(unittest.TestCase):
    ALPHA, DELTA, N, RUNS = 0.02, 0.05, 2000, 1000

    def run_many(self, choose):
        rng = random.Random(7)
        violations = coverage = 0
        for _ in range(self.RUNS):
            scores, wrong = population(self.N, rng)
            cut = choose(scores, wrong, rng)
            if cut is not None:
                violations += true_risk(cut) > self.ALPHA
                coverage += 1 - cut
        return violations / self.RUNS, coverage / self.RUNS

    def test_certified_cut_exceeds_target_at_most_delta_of_the_time(self):
        def choose(scores, wrong, rng):
            reference, _ = population(self.N, rng)                  # independent of the labelled sample
            start = certify.start_cut(reference, self.N, self.ALPHA, self.DELTA)
            return certify.certify(scores, wrong, self.ALPHA, self.DELTA, start=start).threshold

        rate, coverage = self.run_many(choose)
        self.assertLessEqual(rate, self.DELTA + 0.02)               # 0.02 is three standard errors of the estimate
        self.assertGreater(coverage, 0.15)                          # and it is not vacuous: it approves a real share

    def test_naive_cut_exceeds_target_far_more_often(self):
        rate, _ = self.run_many(lambda scores, wrong, rng: certify.naive(scores, wrong, self.ALPHA))
        self.assertGreater(rate, 0.25)

    def test_nothing_is_certified_from_too_small_a_sample(self):
        rng = random.Random(1)
        scores, wrong = population(60, rng)
        cert = certify.certify(scores, [False] * 60, 0.02, 0.05)
        self.assertIsNone(cert.threshold)                           # 60 clean decisions cannot certify 2%
        self.assertEqual(cert.coverage, 0.0)

    def test_certificate_reports_what_it_saw(self):
        scores = [0.99] * 300 + [0.6] * 100
        wrong = [False] * 300 + [True] * 100
        self.assertIsNone(certify.certify(scores, wrong, 0.02, 0.05).threshold)     # the strictest cuts approve nothing
        cert = certify.certify(scores, wrong, 0.02, 0.05, start=0.99)
        self.assertEqual((cert.approved, cert.errors, cert.total), (300, 0, 400))
        self.assertTrue(cert.approves(0.99) and not cert.approves(0.6))
        self.assertLessEqual(cert.bound, 0.02)


if __name__ == "__main__":
    unittest.main()
