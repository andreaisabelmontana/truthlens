"""Trained linear entailment model for (claim, evidence) stance.

A multinomial logistic-regression classifier over the transparent lexical +
similarity features in ``features.py``. Trained on the committed seed dataset
``data/nli_seed.json``. Predicts one of SUPPORTED / CONTRADICTED / NEI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from .features import pair_features

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_SEED = DATA_DIR / "nli_seed.json"

LABELS = ("SUPPORTED", "CONTRADICTED", "NEI")


def load_seed(path: Path = DEFAULT_SEED) -> List[dict]:
    """Load the seed entailment triples from JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload["data"]


def _build_xy(rows: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.array([pair_features(r["claim"], r["evidence"]) for r in rows])
    y = np.array([r["label"] for r in rows])
    return X, y


class EntailmentModel:
    """Logistic-regression stance classifier over engineered features."""

    def __init__(self, model: Optional[LogisticRegression] = None) -> None:
        self.model = model

    def fit(self, rows: Sequence[dict]) -> "EntailmentModel":
        """Train on a sequence of {claim, evidence, label} triples."""
        X, y = _build_xy(rows)
        # Modern scikit-learn (>=1.7) uses the multinomial solver by default
        # for multiclass problems, so no multi_class argument is needed.
        clf = LogisticRegression(
            C=4.0,
            max_iter=2000,
            class_weight="balanced",
        )
        clf.fit(X, y)
        self.model = clf
        return self

    @classmethod
    def train_default(cls) -> "EntailmentModel":
        """Train a model on the committed seed dataset."""
        return cls().fit(load_seed())

    def _check(self) -> None:
        if self.model is None:
            raise RuntimeError("EntailmentModel is not trained yet.")

    def predict(self, claim: str, evidence: str) -> str:
        """Return the predicted stance label for one pair."""
        self._check()
        x = pair_features(claim, evidence).reshape(1, -1)
        return str(self.model.predict(x)[0])

    def predict_proba(self, claim: str, evidence: str) -> dict:
        """Return a {label: probability} mapping for one pair."""
        self._check()
        x = pair_features(claim, evidence).reshape(1, -1)
        probs = self.model.predict_proba(x)[0]
        return {label: float(p) for label, p in zip(self.model.classes_, probs)}

    def score(self, rows: Sequence[dict]) -> float:
        """Return accuracy on a held-out sequence of triples."""
        self._check()
        X, y = _build_xy(rows)
        return float(self.model.score(X, y))
