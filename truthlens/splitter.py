"""Split a block of study notes into individual claims (sentences).

A deliberately simple, dependency-free sentence splitter. It breaks on
sentence-final punctuation (. ! ?) while guarding against decimal numbers
(e.g. "9.8") and a few common abbreviations, so claims are not chopped in
half at a dot that does not actually end a sentence.
"""

from __future__ import annotations

import re
from typing import List

# Abbreviations that end in a period but usually do not end a sentence.
_ABBREVIATIONS = {
    "approx", "etc", "e.g", "i.e", "vs", "mr", "mrs", "ms", "dr", "prof",
    "fig", "no", "al", "st", "mt", "ca",
}

# Private-use sentinels swapped in for dots that must not trigger a split.
_DECIMAL_DOT = chr(0xE000)
_ABBR_DOT = chr(0xE001)


def split_into_claims(text: str) -> List[str]:
    """Return a list of trimmed claim strings extracted from ``text``.

    Empty input yields an empty list. Decimal numbers and a small set of
    abbreviations do not trigger a split.
    """
    if not text or not text.strip():
        return []

    work = text
    # Protect decimals like "9.8".
    work = re.sub(r"(?<=\d)\.(?=\d)", _DECIMAL_DOT, work)
    # Protect known abbreviations (case-insensitive): keep the period, just
    # swap it for a sentinel so the splitter ignores it.
    for abbr in _ABBREVIATIONS:
        pattern = re.compile(r"\b" + re.escape(abbr) + r"\.", re.IGNORECASE)
        work = pattern.sub(lambda m: m.group(0)[:-1] + _ABBR_DOT, work)

    # Split on sentence-final punctuation followed by whitespace.
    parts = re.split(r"(?<=[.!?])\s+", work)

    claims: List[str] = []
    for part in parts:
        claim = part.replace(_DECIMAL_DOT, ".").replace(_ABBR_DOT, ".").strip()
        if _looks_like_claim(claim):
            claims.append(claim)
    return claims


def _looks_like_claim(s: str) -> bool:
    """A claim must have at least two word characters and some letters."""
    return len(re.findall(r"\w", s)) >= 2 and any(c.isalpha() for c in s)
