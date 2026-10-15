# 0005. MultiHop-RAG as the corpus and gold set

- Status: accepted (replaces rag-mini-wikipedia)
- Date: 2026-10-16

## Context
rag-mini-wikipedia's questions have no link to relevant passages, so retrieval metrics would need
hand labelling.

## Decision
Use `yixuantt/MultiHopRAG` (ODC-BY), pinned to revision `71ac0d0b…`:
609 news articles (Sep–Dec 2023; title, source, category, date, url, body) and 2,556 questions
(inference 816, comparison 856, temporal 583, null 301) with gold evidence (article title + fact
sentence). Null questions expect "Insufficient information."

## Consequences
- Gold labels are free. Article titles are unique, so `document_id = hash(title)` maps evidence → documents.
- Full-length articles (avg ~1.7k words), so chunking choices matter.
- Answers are short (entities, yes/no, before/after). That keeps scoring cheap but doesn't test long-form answers.
- Contamination risk (2023 news may be in model training data) → measured by a no-retrieval baseline in M2.
- Attribution required (ODC-BY): credit Tang & Yang, 2024, arXiv:2401.15391.
