"""Retrieval metrics: did search bring back the articles (and sentences) the answer needs?

All of them look at the ranked list of chunks that search returned:

hit@k          1 if any of the top k chunks comes from a gold article, else 0.
               "Did we find *something* useful?"
recall@k       gold articles found in the top k ÷ all gold articles.
               Multi-hop questions need every article, so this matters more than hit@k.
MRR            1 / rank of the first chunk from a gold article, averaged over questions.
               Rank 1 → 1.0, rank 2 → 0.5, not found → 0. Rewards putting good chunks first.
fact recall    evidence sentences that appear in the top k chunks ÷ all evidence sentences.
               Stricter than recall: the right article isn't enough if the chunk that was
               retrieved doesn't contain the sentence the answer relies on.
"""

import re
import unicodedata

from rag.domain.models import GoldenItem, RetrievalScore, ScoredChunk


def _normalize(text: str) -> str:
    # Same unicode + whitespace rules as ingestion's cleaning, so a fact matches a chunk
    # even when line breaks or spacing differ; lowercase to ignore casing.
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().lower()


def score_retrieval(
    item: GoldenItem, chunks: list[ScoredChunk], ks: list[int], fact_k: int
) -> RetrievalScore:
    gold = set(item.evidence_document_ids)
    ranked_docs = [c.document_id for c in chunks]

    first_gold_rank = next((n for n, d in enumerate(ranked_docs, start=1) if d in gold), None)
    retrieved_text = _normalize(" ".join(c.text for c in chunks[:fact_k]))
    facts_found = sum(_normalize(f) in retrieved_text for f in item.evidence_facts)

    return RetrievalScore(
        item_id=item.id,
        question_type=item.question_type,
        hit={k: bool(gold & set(ranked_docs[:k])) for k in ks},
        recall={k: len(gold & set(ranked_docs[:k])) / len(gold) for k in ks},
        reciprocal_rank=1 / first_gold_rank if first_gold_rank else 0.0,
        fact_recall=facts_found / len(item.evidence_facts),
    )


def summarize(scores: list[RetrievalScore], ks: list[int]) -> dict[str, float]:
    """Average every metric over a group of questions."""
    n = len(scores)
    summary = {"questions": n}
    for k in ks:
        summary[f"hit@{k}"] = sum(s.hit[k] for s in scores) / n
        summary[f"recall@{k}"] = sum(s.recall[k] for s in scores) / n
    summary["mrr"] = sum(s.reciprocal_rank for s in scores) / n
    summary["fact_recall"] = sum(s.fact_recall for s in scores) / n
    return summary
