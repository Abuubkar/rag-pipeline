# 0011. Milestone 3 retrieval defaults, judged with paired confidence intervals

- Status: accepted
- Date: 2026-11-13

## Context
Milestone 3 tried seven retrieval changes (steps 3.1–3.6). Step 3.7 compares each one with the
current default (plain hybrid on the sentence index), question by question, using paired
bootstrap 95% confidence intervals (5,000 resamples, fixed seed). Most runs used all 451
answerable dev questions; rerank used an 87-question sample because it is slow.

## Decision
| Setting | Default | Evidence (recall@6 Δ vs hybrid, 95% CI) |
|---|---|---|
| `retrieval.mode` | `hybrid` (kept) | dense −0.045 [−0.068, −0.022]; keyword −0.124 |
| `chunking.splitter` | `sentence` (kept) | semantic +0.005, recursive +0.001; both CIs span 0 |
| `query_transform` | `none` | rewrite −0.041, multi-query −0.032 (really worse); HyDE +0.004 (no effect), +8 s |
| `metadata_filter` | `none` for now | +0.054 [+0.037, +0.072], free; whether to turn it on is still open (see Consequences) |
| `rerank` | `none` | +0.134 [+0.088, +0.180] on 87 questions, but ≈ 51 s per question locally |
| `compression` | `none` | fact recall −0.037 (keep 0.5), −0.017 (keep 0.7); answer impact not measured |

- A change becomes a default only if its CI excludes 0 in the right direction **and** its
  latency is acceptable for interactive `ask`.
- Comparisons are paired on questions and use the current default as the reference, not the
  Milestone 2 baseline.

## Alternatives considered
- **Comparing averages only** (Milestone 2 style): it can't tell a +0.005 tie from a +0.05 win.
- **Unpaired tests / t-test on the two means:** they ignore the pairing, so their intervals are
  much wider. A t-test also assumes roughly normal differences; recall differences are steps
  of 1/2, 1/3 or 1/4.
- **Multiple-comparison correction (Bonferroni):** stricter, but it would only matter for the
  borderline semantic-chunker MRR result, which is treated as weak evidence anyway.

## Consequences
- The source filter's gain partly reflects how MultiHop-RAG questions are written (they name
  publications). It falls back to no filter when nothing is named, so turning it on is cheap
  and safe. The decision is left to the next change of defaults.
- Rerank is the largest gain. A small local cross-encoder (or a faster LLM) could make it fit
  interactive latency. That is backlog work.
- Compression needs the answer eval before it can be judged; retrieval metrics only show what
  it costs.
- Every regenerated `docs/results/README.md` now shows these intervals, so later milestones
  are judged the same way.
