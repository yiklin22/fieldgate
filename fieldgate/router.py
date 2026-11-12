"""Turn the signals into one score: the probability that a value is right.

A logistic regression by default, because its weights can be read out to a customer and it does
not need much data. `kind="boosted"` swaps in gradient-boosted trees for comparison.

Scores used to pick or judge a threshold must come from a model that never saw those receipts,
so `cross_fit` scores each receipt with a model fitted on the other folds.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .schema import FIELD_NAMES
from .signals import SIGNAL_NAMES

FEATURES = tuple(SIGNAL_NAMES) + tuple(f"field={f}" for f in FIELD_NAMES) + tuple(f"null:{f}" for f in FIELD_NAMES)


def matrix(rows: Sequence[dict], signals: Sequence[str] = SIGNAL_NAMES) -> np.ndarray:
    """Signals, one indicator per field, and one per field for "answered absent" (fields differ in how
    often absent is the right answer)."""
    x = np.zeros((len(rows), len(signals) + 2 * len(FIELD_NAMES)))
    for i, r in enumerate(rows):
        for j, name in enumerate(signals):
            x[i, j] = r[name]
        k = FIELD_NAMES.index(r["field"])
        x[i, len(signals) + k] = 1.0
        x[i, len(signals) + len(FIELD_NAMES) + k] = r["is_null"]
    return x


def fit(rows: Sequence[dict], kind: str = "logistic", signals: Sequence[str] = SIGNAL_NAMES):
    y = np.array([r["correct"] for r in rows], dtype=int)
    if len(set(y.tolist())) < 2:
        raise ValueError("need both right and wrong examples to fit a scorer")
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    elif kind == "boosted":
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=200, l2_regularization=1.0,
                                               random_state=0)
    else:
        raise ValueError(kind)
    model.fit(matrix(rows, signals), y)
    model.signals_ = tuple(signals)
    return model


def score(model, rows: Sequence[dict]) -> list[float]:
    if not rows:
        return []
    return model.predict_proba(matrix(rows, model.signals_))[:, 1].tolist()


def cross_fit(rows: Sequence[dict], folds: int = 5, kind: str = "logistic", seed: int = 0,
              signals: Sequence[str] = SIGNAL_NAMES) -> list[float]:
    """Out-of-fold scores: each receipt is scored by a model fitted without it."""
    docs = sorted({r["doc"] for r in rows})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(docs))
    fold_of = {docs[d]: i % folds for i, d in enumerate(order)}
    out = [0.0] * len(rows)
    for k in range(folds):
        train = [r for r in rows if fold_of[r["doc"]] != k]
        held = [i for i, r in enumerate(rows) if fold_of[r["doc"]] == k]
        model = fit(train, kind, signals)
        for i, s in zip(held, score(model, [rows[i] for i in held])):
            out[i] = s
    return out


def weights(model) -> list[tuple[str, float]]:
    """Standardised logistic weights, largest first. Empty for other model kinds."""
    try:
        coef = model[-1].coef_[0]
    except (TypeError, AttributeError, IndexError):
        return []
    names = tuple(model.signals_) + FEATURES[len(SIGNAL_NAMES):]
    return sorted(zip(names, coef.tolist()), key=lambda p: -abs(p[1]))
