"""TruthLens demo: fact-check a short study note containing true and false claims.

Run:  python demo.py
"""

from __future__ import annotations

from truthlens import FactChecker

NOTE = (
    "The mitochondria is the powerhouse of the cell. "
    "Water boils at 90 degrees Celsius at sea level. "
    "The Great Wall of China is visible from space with the naked eye. "
    "Python is a dynamically typed language. "
    "Paris is the capital of France. "
    "Binary search runs in linear time on a sorted array. "
    "My favorite color is blue today."
)

ICON = {"SUPPORTED": "[+]", "CONTRADICTED": "[x]", "NEI": "[?]"}


def main() -> None:
    print("TruthLens - sentence-level fact-check")
    print("=" * 64)
    print("Note under review:")
    print(f"  {NOTE}\n")
    print("=" * 64)

    checker = FactChecker()
    verdicts = checker.check_note(NOTE)

    counts = {"SUPPORTED": 0, "CONTRADICTED": 0, "NEI": 0}
    for v in verdicts:
        counts[v.verdict] += 1
        print(f"\n{ICON[v.verdict]} {v.verdict}  (confidence {v.confidence:.0%})")
        print(f"    claim:    {v.claim}")
        if v.evidence:
            print(f"    evidence: {v.evidence}")
            print(f"    source:   {v.evidence_source}  (retrieval sim {v.retrieval_score:.2f})")
        else:
            print("    evidence: none relevant in corpus")

    print("\n" + "=" * 64)
    print(
        f"Summary: {counts['SUPPORTED']} supported, "
        f"{counts['CONTRADICTED']} contradicted, "
        f"{counts['NEI']} not-enough-info."
    )


if __name__ == "__main__":
    main()
