"""Dense retrieval regression floor, using MultiHop-RAG's gold evidence sentences as queries.

Each `fact` is a verbatim sentence from a known article, so we measure: does searching for it
bring back its own article? Scored per *article* (chunks of the same article count once).

Measured baseline (M1: dense only, sentence 512/64, contextual header, 40 facts):
    article hit@5 = 0.75, hit@10 = 0.88
Misses are generic sentences shared by near-duplicate articles (sportsbook promos, many FTX-trial
stories). Milestone 3 (keyword + RRF hybrid) should raise these numbers; then raise the floors.
"""

import pytest
from datasets import load_dataset

from rag.packages.documents.multihop_rag import document_id_for_title
from rag.retrieval.search import dense_search

FLOORS = {5: 0.70, 10: 0.85}


def _evidence_facts(container, n: int) -> list[tuple[str, str]]:
    cfg = container.settings.dataset
    questions = load_dataset(cfg.name, "MultiHopRAG", split="train", revision=cfg.revision)
    pairs = [
        (ev["fact"], document_id_for_title(ev["title"]))
        for row in questions.select(range(200))
        for ev in row["evidence_list"]
    ]
    return pairs[:n]


def _ranked_articles(chunks) -> list[str]:
    seen: list[str] = []
    for c in chunks:
        if c.document_id not in seen:
            seen.append(c.document_id)
    return seen


def test_fact_sentences_retrieve_their_own_article(container, active_version):
    if active_version.manifest.dataset["documents"] < 600:
        pytest.skip("needs the full corpus indexed (run ingest without --limit)")

    embedder = container.query_embedder(active_version)
    facts = _evidence_facts(container, n=40)
    hits = {k: 0 for k in FLOORS}
    for fact, doc_id in facts:
        chunks = dense_search(fact, db_url=container.db_url, version_id=active_version.id,
                              embedder=embedder, top_k=40)
        articles = _ranked_articles(chunks)
        for k in FLOORS:
            hits[k] += doc_id in articles[:k]

    rates = {k: hits[k] / len(facts) for k in FLOORS}
    for k, floor in FLOORS.items():
        assert rates[k] >= floor, f"article hit@{k} = {rates[k]:.2f} (floor {floor}); all: {rates}"
