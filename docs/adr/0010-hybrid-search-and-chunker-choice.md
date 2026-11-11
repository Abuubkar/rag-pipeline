# 0010. Hybrid search (RRF) by default; sentence chunking stays the default chunker

- Status: accepted
- Date: 2026-11-11

## Context
Milestone 3 compares retrieval options on the dev split (451 answerable questions). Step 3.1
added Postgres full-text search and RRF fusion. Step 3.2 added two more chunkers: LlamaIndex
semantic and LangChain recursive. All of them need a default.

## Decision
- **`retrieval.mode: hybrid`.** Dense and keyword search each bring 30 candidates, which are fused
  with RRF (k = 60). Against dense alone: recall@6 0.56 → 0.61, fact recall 0.44 → 0.48.
- **Keyword query uses OR semantics** (the `plainto_tsquery` terms joined with `|`) and is
  ranked with `ts_rank_cd`. With AND semantics, long questions matched no chunks.
- **`chunking.splitter: sentence` stays the default.** Under hybrid search, all three chunkers
  score the same (recall@6 0.61, fact recall 0.47–0.48). Sentence chunking is the cheapest and
  already indexed.
- Every chunker counts length in `cl100k_base` tokens, so `chunk_size` means the same thing for
  all of them.

## Alternatives considered
- **Convex α-fusion** of min-max-normalised scores: it needs a tuned α and score normalisation.
  RRF needs neither. It stays in the backlog.
- **Semantic chunking:** no gain on short single-topic news articles, 5× the indexing time,
  and uneven chunk sizes (an extra re-split step is needed above 512 tokens).
- **Recursive chunking:** same quality; switching would only add a second chunking library to
  the default path.

## Consequences
- Every query now runs two searches. On this corpus that is still well under the embedding time.
- The semantic and recursive indexes remain in Postgres for later experiments. Switching is a
  config change, plus `index activate`.
- The ties will be re-checked with paired confidence intervals (step 3.7) and could change on a
  corpus with long multi-topic documents.
