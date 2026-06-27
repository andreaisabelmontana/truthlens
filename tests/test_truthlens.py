"""Tests for the TruthLens retrieve-then-classify fact-checking pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

from truthlens import (
    FactChecker,
    Retriever,
    EntailmentModel,
    split_into_claims,
    pair_features,
    FEATURE_NAMES,
)
from truthlens.entailment import load_seed


# --------------------------------------------------------------------------- #
# Shared fixtures (build the checker once; it trains a model).
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def checker() -> FactChecker:
    return FactChecker()


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    return Retriever()


# --------------------------------------------------------------------------- #
# Splitter
# --------------------------------------------------------------------------- #
def test_split_basic():
    claims = split_into_claims(
        "Water boils at 100 degrees. Paris is the capital of France."
    )
    assert claims == [
        "Water boils at 100 degrees.",
        "Paris is the capital of France.",
    ]


def test_split_keeps_decimals_intact():
    claims = split_into_claims("Gravity is about 9.8 meters per second squared.")
    assert len(claims) == 1
    assert "9.8" in claims[0]


def test_split_empty():
    assert split_into_claims("") == []
    assert split_into_claims("   ") == []


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
def test_retrieval_returns_correct_evidence(retriever: Retriever):
    """A clear claim must retrieve the matching corpus sentence on top."""
    top = retriever.top("Paris is the capital of France")
    assert top is not None
    assert top.id == "geo-4"
    assert "Paris" in top.text
    assert top.score > 0.3


def test_retrieval_topic_match(retriever: Retriever):
    top = retriever.top("At what temperature does water boil at sea level?")
    assert top is not None
    assert top.topic == "chemistry"
    assert "boils" in top.text


def test_retrieval_offtopic_is_low(retriever: Retriever):
    top = retriever.top("My favorite color is blue and I like pizza")
    # Off-topic queries should score below the checker's relevance floor.
    assert top.score < 0.08


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def test_feature_vector_shape():
    vec = pair_features("Water boils at 90 C", "Water boils at 100 C")
    assert vec.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(vec))


def test_feature_number_conflict_signal():
    feats = dict(
        zip(FEATURE_NAMES, pair_features("Water boils at 90 C", "Water boils at 100 C"))
    )
    assert feats["number_conflict"] == 1.0
    assert feats["number_match"] == 0.0


# --------------------------------------------------------------------------- #
# End-to-end verdicts (the core behaviour contract)
# --------------------------------------------------------------------------- #
def test_supported_claim(checker: FactChecker):
    v = checker.check_claim("Paris is the capital of France.")
    assert v.verdict == "SUPPORTED"
    assert v.evidence is not None
    assert "Paris" in v.evidence


def test_contradicted_claim(checker: FactChecker):
    v = checker.check_claim("Water boils at 90 degrees Celsius at sea level.")
    assert v.verdict == "CONTRADICTED"
    assert "100" in v.evidence


def test_contradicted_negation_claim(checker: FactChecker):
    v = checker.check_claim(
        "The Great Wall of China is visible from space with the naked eye."
    )
    assert v.verdict == "CONTRADICTED"


def test_offtopic_claim_is_nei(checker: FactChecker):
    v = checker.check_claim("My favorite color is blue and I love pizza on Fridays.")
    assert v.verdict == "NEI"
    # An off-topic claim has no genuinely relevant evidence to show.
    assert v.evidence is None


def test_check_note_mixed():
    """A note with true + false + off-topic statements gets the right mix."""
    fc = FactChecker()
    note = (
        "Paris is the capital of France. "
        "Water boils at 90 degrees Celsius at sea level. "
        "My favorite color is blue today."
    )
    verdicts = fc.check_note(note)
    labels = [v.verdict for v in verdicts]
    assert labels == ["SUPPORTED", "CONTRADICTED", "NEI"]


# --------------------------------------------------------------------------- #
# Model quality: held-out entailment accuracy above a sane floor
# --------------------------------------------------------------------------- #
def test_heldout_entailment_accuracy_above_floor():
    rows = load_seed()
    X = np.array([pair_features(r["claim"], r["evidence"]) for r in rows])
    y = np.array([r["label"] for r in rows])
    clf = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=skf)
    mean_acc = scores.mean()
    # 3-class problem; chance is ~0.33. Floor is well above chance and below
    # the observed ~0.77 so it stays green across scikit-learn patch versions.
    assert mean_acc > 0.65, f"held-out CV accuracy too low: {mean_acc:.3f}"


def test_training_fit_is_reasonable():
    # The model is regularised (C=4.0, balanced class weights), so it does not
    # memorise the seed set; training accuracy sits in the high-0.7s rather
    # than near 1.0, which is the desired behaviour for a small linear model.
    model = EntailmentModel.train_default()
    rows = load_seed()
    train_acc = model.score(rows)
    assert train_acc > 0.75
