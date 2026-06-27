"""End-to-end fact-checker: split -> retrieve -> classify.

Ties the pipeline together. For each claim in a note it retrieves the best
evidence from the corpus, then asks the trained entailment model for a stance.
If the best evidence is too weakly related (low retrieval similarity), the
claim is reported as NEI without trusting the classifier, because there is no
genuinely relevant evidence to reason over.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .splitter import split_into_claims
from .retriever import Retriever, Evidence
from .entailment import EntailmentModel, LABELS

# If the top retrieved evidence scores below this cosine similarity, we treat
# the claim as off-topic (NEI) regardless of what the classifier says.
RETRIEVAL_FLOOR = 0.08


@dataclass
class ClaimVerdict:
    """The fact-check result for a single claim."""

    claim: str
    verdict: str                 # SUPPORTED / CONTRADICTED / NEI
    confidence: float            # probability assigned to the chosen verdict
    evidence: Optional[str]      # supporting/contradicting evidence sentence
    evidence_source: Optional[str]
    retrieval_score: float       # cosine similarity of the top evidence

    def __str__(self) -> str:
        head = f"[{self.verdict:12s} {self.confidence:5.0%}] {self.claim}"
        if self.evidence:
            head += f"\n    evidence: {self.evidence}"
            head += f"\n    source:   {self.evidence_source} (sim {self.retrieval_score:.2f})"
        return head


class FactChecker:
    """Sentence-level fact-checker over a reference corpus."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        model: Optional[EntailmentModel] = None,
        retrieval_floor: float = RETRIEVAL_FLOOR,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.model = model or EntailmentModel.train_default()
        self.retrieval_floor = retrieval_floor

    def check_claim(self, claim: str) -> ClaimVerdict:
        """Fact-check a single claim string."""
        top = self.retriever.top(claim)

        if top is None or top.score < self.retrieval_floor:
            # No corpus sentence is relevant enough to reason over, so we do
            # not surface a misleading "evidence" line for an off-topic claim.
            return ClaimVerdict(
                claim=claim,
                verdict="NEI",
                confidence=1.0 if top is None else 1.0 - top.score,
                evidence=None,
                evidence_source=None,
                retrieval_score=top.score if top else 0.0,
            )

        probs = self.model.predict_proba(claim, top.text)
        verdict = max(probs, key=probs.get)
        return ClaimVerdict(
            claim=claim,
            verdict=verdict,
            confidence=probs[verdict],
            evidence=top.text,
            evidence_source=top.source,
            retrieval_score=top.score,
        )

    def check_note(self, note: str) -> List[ClaimVerdict]:
        """Split a note into claims and fact-check each one."""
        return [self.check_claim(c) for c in split_into_claims(note)]
