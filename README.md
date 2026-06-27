# TruthLens

Sentence-level fact-checking for study notes. TruthLens reads a block of notes,
splits it into individual claims, finds the most relevant sentence in a trusted
reference corpus, and decides whether each claim is **SUPPORTED**,
**CONTRADICTED**, or **NOT-ENOUGH-INFO** — with the evidence and a confidence.

Originally a BCSAI hackathon project; this repo is a from-scratch, runnable
core of the idea in pure Python.

🔗 **Showcase site:** https://andreaisabelmontana.github.io/truthlens/

## Pipeline

```
note ─▶ split into claims ─▶ TF-IDF retrieve evidence ─▶ classify stance ─▶ verdict
```

1. **Split** (`truthlens/splitter.py`) — break notes into sentences/claims,
   guarding decimals (`9.8`) and a few abbreviations so claims stay intact.
2. **Retrieve** (`truthlens/retriever.py`) — a `TfidfVectorizer` (1–2 word
   grams, sublinear TF, English stop words) over the reference corpus; cosine
   similarity returns the best-matching evidence sentence. If the top match is
   below a similarity floor (0.08), the claim is reported NOT-ENOUGH-INFO,
   because there is nothing relevant to reason over.
3. **Classify** (`truthlens/features.py`, `truthlens/entailment.py`) — a
   multinomial **logistic-regression** entailment model reads transparent
   features of the `(claim, evidence)` pair: token overlap / coverage, number
   match vs. number conflict, negation mismatch, and antonym conflict. It
   outputs the stance plus a probability used as the confidence.

## Honest scope

The "AI" here is small and transparent: **TF-IDF retrieval + a trained linear
entailment classifier over hand-designed lexical/similarity features.** There
is no large language model and no neural network. The corpus is small and
fixed, so TruthLens only knows what is in `data/corpus.json` — anything outside
it lands as NOT-ENOUGH-INFO.

## Data (committed)

- `data/corpus.json` — 38 factual reference sentences across biology,
  chemistry, physics, astronomy, geography, computer science, and history, each
  with a topic and a source label.
- `data/nli_seed.json` — 98 labelled `(claim, evidence, label)` triples
  (42 SUPPORTED / 36 CONTRADICTED / 20 NEI) used to train the entailment model.

## Real results

Entailment classifier on the seed set (5-fold stratified cross-validation):

| metric | value |
|---|---|
| mean held-out accuracy (5-fold CV) | **0.77** (±0.06) |
| training-fit accuracy | 0.80 |
| classes | SUPPORTED / CONTRADICTED / NEI (3-way; chance ≈ 0.33) |

Example verdicts from `demo.py` (verbatim):

```
[+] SUPPORTED    (confidence 92%)  Paris is the capital of France.
      evidence: The capital of France is Paris.  (sim 0.89)
[x] CONTRADICTED (confidence 84%)  Water boils at 90 degrees Celsius at sea level.
      evidence: Water boils at 100 degrees Celsius ... at sea level.  (sim 0.62)
[x] CONTRADICTED (confidence 57%)  The Great Wall of China is visible from space with the naked eye.
      evidence: The Great Wall of China is not visible to the unaided human eye ...  (sim 0.57)
[x] CONTRADICTED (confidence 86%)  Binary search runs in linear time on a sorted array.
      evidence: Binary search runs in logarithmic time on a sorted array.  (sim 0.88)
[?] NEI          (confidence 100%) My favorite color is blue today.
      evidence: none relevant in corpus
```

## Run it

```bash
pip install -r requirements.txt
python demo.py          # fact-check a mixed true/false/off-topic note
python -m pytest -q     # 15 tests
```

```python
from truthlens import FactChecker

checker = FactChecker()
for v in checker.check_note("Water boils at 90 C at sea level. Paris is the capital of France."):
    print(v.verdict, round(v.confidence, 2), "|", v.claim)
```

## Tests

`tests/test_truthlens.py` (15 tests) covers: a corpus-matching claim is
SUPPORTED; a contradicting claim is CONTRADICTED (including the negation case);
an off-topic claim is NOT-ENOUGH-INFO; retrieval returns the correct evidence
sentence for a clear claim; and held-out entailment accuracy stays above a
sane floor (0.65).

```
15 passed
```

## Layout

```
truthlens/
  splitter.py     note -> claims
  retriever.py    TF-IDF cosine retrieval over the corpus
  features.py     lexical + similarity features for a (claim, evidence) pair
  entailment.py   logistic-regression stance classifier
  checker.py      end-to-end FactChecker
data/
  corpus.json     trusted reference sentences
  nli_seed.json   labelled entailment triples
demo.py
tests/test_truthlens.py
```

## License

MIT — see [LICENSE](LICENSE).
