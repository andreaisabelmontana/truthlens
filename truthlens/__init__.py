"""TruthLens: sentence-level fact-checking over a trusted reference corpus.

Pipeline:
    1. Split study notes into individual claims (sentences).
    2. Retrieve the most relevant evidence sentences from the reference
       corpus using TF-IDF cosine similarity.
    3. Classify the stance of (claim, evidence) as SUPPORTED, CONTRADICTED,
       or NEI (not enough info) with a trained logistic-regression model that
       reads lexical + similarity features.

The "AI" here is honest and small: TF-IDF retrieval plus a linear entailment
classifier. There is no large language model involved.
"""

from .splitter import split_into_claims
from .retriever import Retriever
from .features import pair_features, FEATURE_NAMES
from .entailment import EntailmentModel
from .checker import FactChecker, ClaimVerdict, LABELS

__all__ = [
    "split_into_claims",
    "Retriever",
    "pair_features",
    "FEATURE_NAMES",
    "EntailmentModel",
    "FactChecker",
    "ClaimVerdict",
    "LABELS",
]

__version__ = "0.1.0"
