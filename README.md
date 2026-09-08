# Neuropod

Turns recent arXiv papers into citation-grounded, audio-ready research scripts.
Pick topics, and it finds papers, reads the PDFs, retrieves the passages that
matter, and writes a 6–9 minute narration grounded in the actual paper.

Built as an ML engineering portfolio project. **The retrieval evaluation is the
point** — the app is what gives it something to be right or wrong about.

```bash
make demo        # seed episodes (no API keys needed)
make dev         # API on :8000
cd frontend && npm run dev
```

Then open http://localhost:3000/login and sign in as `demo@neuropod.local`
(stub auth, any email, no password).

---

## Two ranking problems

They are separate ML problems with separate data, labels, baselines and models.
Confusing them is the most common way to misread this repo.

| | **Task A — which papers?** | **Task B — which passages?** |
|---|---|---|
| Unit | a paper | a ~110-word chunk of one paper |
| Decides | which 3–5 papers become episodes | which 14 chunks reach the model |
| Labels | 300 hand-graded, pooled | 1,935 self-supervised (ICT) |
| Baseline | TF-IDF, nDCG@5 **0.808** | BM25, nDCG@10 **0.359** |
| Model | none — see below | LambdaMART, **+0.160** held out |
| Code | `pipeline/discover/ranker.py` | `pipeline/generate/retriever.py` |

---

## Results

### Retrieval (Task B)

168 papers, 1,935 queries, hash embeddings, retrieval scoped to one paper.
Every comparison is a paired bootstrap on the same queries.

| Config | nDCG@10 | vs. legacy |
|---|---|---|
| `legacy` (raw TF cosine + hand-set section prior) | 0.281 | — |
| `dense` | 0.318 | +0.037 p<0.001 |
| **`shipped`** (BM25 + dense, RRF) | **0.360** | **+0.080 p<0.001** |

### The learned reranker

Grouped by paper, never by query — queries from one paper share a chunk pool,
so a query-level split lets a model memorise chunks it is then scored on.
5-fold GroupKFold, plus 34 papers held out and scored once.

| Model | held-out nDCG@10 | vs. BM25 |
|---|---|---|
| `dense+prior` (what shipped before) | 0.237 | −0.086 p<0.001 |
| BM25 | 0.324 | — |
| Logistic regression | 0.320 | −0.003 **p=0.598** |
| GBDT | 0.423 | +0.099 p<0.001 |
| **LambdaMART** | **0.484** | **+0.160 p<0.001** |

**The ranking objective is worth +0.061.** LambdaMART and the GBDT are the same
model family on identical features; only LambdaMART knows candidates compete
within a query. **A linear model ties BM25** (p=0.598), so the gain is not "any
learned model beats a heuristic" — it needs capacity for feature interactions.

### What nDCG@10 = 0.48 means here

With one relevant chunk per query it reduces to `1/log₂(rank+1)`, so it is a
restatement of where the answer lands. In ranks:

| k | LambdaMART | BM25 |
|---|---|---|
| 1 | 24.1% | 8.4% |
| 5 | 61.8% | 46.1% |
| **14** | **84.0%** | 69.1% |

Median gold rank **4 of 45**. **hit@14 is the number that matters for the
product** — the scriptwriter puts 14 chunks in the prompt, so 84% is the rate at
which the relevant passage actually reaches the model.

### Three things that did not work

Reported because a table with only wins is not an evaluation.

- **The hand-tuned section prior made retrieval worse.** `abstract: 0.18`,
  `results: 0.16`… added onto a cosine. Removing it: **−0.034 nDCG@10, p<0.001**.
- **RRF fusion does not beat BM25 alone** on this query set (−0.003, p=0.480),
  and with real embeddings it does not beat dense alone either (−0.006, p=0.228).
  The hybrid still ships — no evidence it helps is not evidence it hurts, and
  lexical matching covers terms coined after an embedding model's cutoff.
- **Hyperparameter tuning gained nothing.** Nested CV measured +0.017 on the
  inner folds and **+0.0018, p=0.768** on unseen papers. Selection bias measured
  at +0.0056. The tuned config was not adopted.

### Real embeddings changed a conclusion

Everything above uses `HashEmbedder` — SHA256 bag-of-words, no semantics. It is
the offline default and its contribution was measured, not assumed. Re-embedding
with `text-embedding-3-small` (~$0.04):

| Config | hash | OpenAI | Δ |
|---|---|---|---|
| `dense` | 0.318 | **0.418** | +0.100 p<0.001 |
| `bm25` | 0.359 | 0.359 | — |

**The ordering flips.** With hash embeddings BM25 beats dense (p<0.001); with
real ones dense beats BM25 (p<0.001). "BM25 is the strongest single retriever"
was an artefact of the embedder. **CI runs on hash embeddings** — deterministic,
free, offline.

---

## Two leaks I found in my own benchmark

The most useful part of the project.

**1. The gold chunk was identifiable by length.** ICT redacts the query sentence
from its source chunk. The chunker caps chunks at 110 words, so after redacting
only the gold, **88.9% of distractors sat exactly at the cap and 0% of targets
did**.

| Ranker | nDCG@10 |
|---|---|
| **chunk length alone, query ignored** | **0.369** |
| BM25 | 0.224 |
| random | 0.075 |

A query-independent rule beat BM25, and a GBDT reached **0.667** learning
nothing else. The tell was a fitted coefficient of **−2.07** on chunk length —
a model insisting shorter chunks are more relevant is not describing retrieval.
Fixed by redacting a sentence from *every* candidate.

**2. The test for the fix was one-sided.** It ranked shortest-first only, and
passed while longest-first still beat random. A one-sided test for a two-sided
property is not a test.

Both are now guarded by tests. Every number above is post-fix.

---

## Task A: paper recommendation

The question "how do we know these are the best papers?" previously had no
answer. Now it has labels and baselines — and a reason not to train a model yet.

**300 pooled judgments**, 5 topic profiles. Candidates are top-40 by TF-IDF
**plus 20 random**: labelling only what the current system surfaces makes its
misses invisible. The random arm found positives TF-IDF's top-40 never showed.

| Ranker | nDCG@5 | P@5 |
|---|---|---|
| production heuristic | 0.834 | 80% |
| TF-IDF cosine | 0.808 | 80% |
| recency only | 0.200 | 20% |
| random | 0.111 | 8% |

**The production heuristic is `0.45·recency + 0.35·trending + 0.20·affinity`,
and 0.80 of that weight is inert.** `trending` is constant (arXiv citations are
0 at publication) and `recency` underflows (a 3.5-day half-life against a corpus
247–2074 days old). The 0.20 affinity term alone reproduces the full score
exactly.

**No model is trained on this.** TF-IDF and the heuristic are statistically
indistinguishable at n=5 topics (p=0.63), and 5 topics cannot support a learned
ranker. Building the labels and declining to train on them is the result.

### Label quality, measured three ways

- **LLM vs. itself** (75 papers, re-annotated blind): κ **0.932**
- **LLM vs. human**: κ **0.664** on the binary rubric
- **LLM vs. arXiv categories** (expertise-free third signal): κ 0.83 / 0.83 / 0.80
  where the categories have coverage

The graded 0/1/2 rubric was **retired**. Every self-disagreement involved the
middle grade, 7 of 12 of the LLM's "adjacent" calls were the human's 0, and the
category signal showed the middle band was tracking *the arXiv category* rather
than the topic. Binary raised agreement to κ 0.664.

`theory` labels are single-annotator and marked as such — the human annotator
lacked domain expertise there, which is recorded next to the data rather than
in a commit message.

---

## Generation quality

Retrieval was measured from the start; generation was not measured at all.

`pipeline/generate/claims.py` extracts every numeric claim from a script and
checks it against the context that produced it. Deterministic, $0, runs on every
generation.

| Script | numeric precision | flagged |
|---|---|---|
| "71% accuracy, 5.2x speedup on 8 GPUs" | 1.00 | no |
| "94% accuracy, 12.7x speedup, 33.5 BLEU" | 0.00 | **yes** |
| "accuracy was **17**%" (one digit swapped) | 0.00 | — |

It replaced a unigram-overlap check that **flagged 100% of real scripts** (they
scored 0.12–0.17 against a 0.30 threshold) and that scored a fabricated claim
reusing source vocabulary *higher* than a faithful paraphrase.

It also verifies what the prompt always demanded and nothing checked: 800–1200
words, 4–7 paragraphs, no markdown.

---

## Cost controls

Per-user daily limits are not a spend bound — stub login mints an identity for
any email, so the quota resets for free.

| Control | Default |
|---|---|
| Episodes per run | 5 |
| **Global runs/day (all users)** | 60 |
| **Monthly / daily USD ceiling** | $5 / $1 |

Spend is measured, not estimated: every call writes `llm_calls` with exact
tokens from the response's `usage` block. Over budget, generation degrades to a
zero-cost path tagged `demo-budget` rather than erroring — verified at 3 episodes
with 0 provider calls. Set a cap at the provider too; that is the only ceiling a
bug here cannot bypass.

---

## Commands

```bash
make demo         # seed episodes, no API keys required
make test         # 137 tests, offline, ~60s
make eval         # retrieval ablation with confidence intervals
make eval-openai  # same, with real embeddings (needs OPENAI_API_KEY, ~$0.04)
make reranker     # train + evaluate on held-out papers
make recommend    # Task A baselines against the labels
make annotate TOPIC=graph N=20    # human label review, reports kappa
make deps-audit   # fail if any pinned dependency is <7 days old
```

## Stack

Python 3.12, FastAPI, Postgres + pgvector (SQLite for local runs), Next.js.
LightGBM and scikit-learn for ranking, PyMuPDF for extraction, tiktoken for
token accounting. Anthropic / OpenAI / Bedrock for generation with a fallback
chain; ElevenLabs / OpenAI for optional TTS.

Dependencies are pinned exactly, including transitives, with a 7-day minimum
release age enforced by `scripts/audit_deps.py`. `sentence-transformers` is
deliberately absent — CVE-2026-68770 bypasses `trust_remote_code=False` when a
model directory exists on disk, so the cross-encoder loads through
`transformers` with safetensors instead.

## Layout

```text
pipeline/
  discover/     arXiv client, paper ranker (Task A)
  ingest/       PDF extraction, chunking, tokenizer
  generate/     retrieval, BM25, features, reranker, claims, scriptwriter
eval/
  corpus_build  fetch + pin the frozen corpus
  queries       ICT query generation and redaction
  harness       retrieval ablation
  train_reranker / tune          learned reranker, nested CV
  topics / annotate / recommend  Task A labels and baselines
  metrics       nDCG, bootstrap CIs, paired tests
api/            FastAPI routes, spend caps, telemetry
```

## Not done

- **Not deployed.** Local only, deliberately — an un-applied cloud scaffold is
  a liability, not an asset.
- **The cross-encoder frontier is unrun** (`make frontier`). The harness is
  written and its plumbing tested; HuggingFace is unreachable from the dev
  sandbox.
- **The Postgres test leg has never run** — no Docker locally. CI runs the suite
  against `pgvector/pgvector:pg16`; those two tests skip on this machine.
- **Task A has no trained model**, for the reason given above.

## License

MIT
