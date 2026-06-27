"""TF-IDF cosine retrieval over the trusted reference corpus.

Given a claim, return the most lexically relevant evidence sentences from the
corpus, each with a cosine-similarity score in [0, 1].
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CORPUS = DATA_DIR / "corpus.json"


@dataclass
class Evidence:
    """A retrieved evidence sentence and its similarity to the query."""

    id: str
    topic: str
    text: str
    source: str
    score: float


def load_corpus(path: Path = DEFAULT_CORPUS) -> List[dict]:
    """Load the corpus JSON and return its list of sentence records."""
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload["sentences"]


class Retriever:
    """Index a reference corpus and retrieve evidence for claims."""

    def __init__(self, sentences: Optional[Sequence[dict]] = None) -> None:
        if sentences is None:
            sentences = load_corpus()
        self.records: List[dict] = list(sentences)
        if not self.records:
            raise ValueError("Retriever needs a non-empty corpus.")
        self._texts = [r["text"] for r in self.records]
        # Word 1-2 grams, sublinear tf, drop English stop words. This rewards
        # shared content terms (the signal that a claim is on-topic).
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(self._texts)

    def retrieve(self, claim: str, k: int = 3) -> List[Evidence]:
        """Return up to ``k`` evidence records most similar to ``claim``."""
        if not claim or not claim.strip():
            return []
        q = self.vectorizer.transform([claim])
        # Rows are L2-normalised by TfidfVectorizer, so dot product == cosine.
        sims = (self.matrix @ q.T).toarray().ravel()
        order = np.argsort(-sims)[:k]
        out: List[Evidence] = []
        for idx in order:
            rec = self.records[idx]
            out.append(
                Evidence(
                    id=rec["id"],
                    topic=rec["topic"],
                    text=rec["text"],
                    source=rec["source"],
                    score=float(sims[idx]),
                )
            )
        return out

    def top(self, claim: str) -> Optional[Evidence]:
        """Return the single best evidence record, or None for an empty query."""
        results = self.retrieve(claim, k=1)
        return results[0] if results else None
