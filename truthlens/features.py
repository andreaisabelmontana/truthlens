"""Feature extraction for the (claim, evidence) entailment classifier.

These are transparent, hand-designed features rather than learned embeddings.
The intuition behind the three stances:

* SUPPORTED  -> high lexical overlap AND no conflicting tokens.
* CONTRADICTED -> high overlap on context BUT a mismatched number, a negation,
  or a swapped key term (antonym-like difference).
* NEI         -> low overlap; the evidence is about something else entirely.

The features below try to capture exactly those signals.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set

import numpy as np

FEATURE_NAMES: List[str] = [
    "token_overlap",        # Jaccard overlap of content tokens
    "claim_coverage",       # fraction of claim tokens found in evidence
    "evidence_coverage",    # fraction of evidence tokens found in claim
    "shared_token_count",   # raw count of shared content tokens (scaled)
    "number_match",         # 1 if claim & evidence share a number
    "number_conflict",      # 1 if both have numbers but none match
    "negation_mismatch",    # 1 if exactly one side is negated
    "antonym_conflict",     # 1 if an antonym pair spans the two sides
    "claim_len",            # length of claim in content tokens (scaled)
]

_STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "by", "for", "with", "and", "or", "as",
    "that", "this", "these", "those", "it", "its", "into", "from", "than",
    "about", "most", "some", "all", "only", "also", "which", "when", "while",
    "approximately", "roughly", "about", "called", "used", "many", "much",
}

_NEGATIONS: Set[str] = {
    "not", "no", "never", "cannot", "n't", "without", "neither", "nor",
    "contrary",
}

# Antonym / mutually-exclusive pairs that flip a claim's truth value.
_ANTONYM_PAIRS = [
    ("largest", "smallest"), ("highest", "lowest"), ("tallest", "shortest"),
    ("longest", "shortest"), ("biggest", "smallest"), ("first", "fourth"),
    ("first", "last"), ("hot", "cold"), ("static", "dynamic"),
    ("statically", "dynamically"), ("linear", "logarithmic"),
    ("above", "below"), ("loses", "keeps"), ("visible", "invisible"),
    ("star", "planet"), ("created", "destroyed"), ("longest", "longest"),
]

_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _content_tokens(text: str) -> Set[str]:
    return {t for t in _tokens(text) if t not in _STOPWORDS and not t.isdigit()
            and not _NUMBER_RE.fullmatch(t)}


def _numbers(text: str) -> Set[str]:
    out: Set[str] = set()
    for n in _NUMBER_RE.findall(text):
        # Normalise "100.0" and "100" to the same key.
        out.add(str(float(n)) if "." in n else str(float(int(n))))
    return out


def _has_negation(text: str) -> bool:
    toks = set(_tokens(text))
    if toks & _NEGATIONS:
        return True
    return "n't" in text.lower() or "not " in text.lower()


def pair_features(claim: str, evidence: str) -> np.ndarray:
    """Return the fixed-length feature vector for one (claim, evidence) pair."""
    c_tokens = _content_tokens(claim)
    e_tokens = _content_tokens(evidence)
    shared = c_tokens & e_tokens
    union = c_tokens | e_tokens

    token_overlap = len(shared) / len(union) if union else 0.0
    claim_coverage = len(shared) / len(c_tokens) if c_tokens else 0.0
    evidence_coverage = len(shared) / len(e_tokens) if e_tokens else 0.0
    shared_count = min(len(shared), 10) / 10.0

    c_nums = _numbers(claim)
    e_nums = _numbers(evidence)
    if c_nums and e_nums:
        number_match = 1.0 if (c_nums & e_nums) else 0.0
        number_conflict = 0.0 if (c_nums & e_nums) else 1.0
    else:
        number_match = 0.0
        number_conflict = 0.0

    c_neg = _has_negation(claim)
    e_neg = _has_negation(evidence)
    negation_mismatch = 1.0 if (c_neg != e_neg) else 0.0

    antonym_conflict = 0.0
    c_all = set(_tokens(claim))
    e_all = set(_tokens(evidence))
    for a, b in _ANTONYM_PAIRS:
        if a == b:
            continue
        if (a in c_all and b in e_all) or (b in c_all and a in e_all):
            antonym_conflict = 1.0
            break

    claim_len = min(len(c_tokens), 12) / 12.0

    return np.array([
        token_overlap,
        claim_coverage,
        evidence_coverage,
        shared_count,
        number_match,
        number_conflict,
        negation_mismatch,
        antonym_conflict,
        claim_len,
    ], dtype=float)


def feature_dict(claim: str, evidence: str) -> Dict[str, float]:
    """Convenience: features as a name -> value mapping (for inspection)."""
    vec = pair_features(claim, evidence)
    return dict(zip(FEATURE_NAMES, vec.tolist()))
