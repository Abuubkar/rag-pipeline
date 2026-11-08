"""Run a part of the pipeline over the golden set and score every question.

The pipeline passes in the function to evaluate (e.g. a `search(question, k)` callable), so
evaluation never imports retrieval or generation itself: the same runner can score any search.
"""

from collections import defaultdict
from typing import Callable

from rag.domain.models import GoldenItem, RetrievalScore, ScoredChunk
from rag.evaluation.metrics import score_retrieval, summarize
from rag.observability.tracing import annotate, traced

SearchFn = Callable[[str, int], list[ScoredChunk]]


@traced("evaluation.retrieval")
def run_retrieval_eval(
    items: list[GoldenItem], search: SearchFn, ks: list[int], fact_k: int,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[RetrievalScore], dict[str, dict[str, float]]]:
    # Null queries have no gold articles, so there is nothing for retrieval to find.
    items = [i for i in items if i.answerable]
    scores = []
    for n, item in enumerate(items, start=1):
        chunks = search(item.question, max(ks))  # search once, then score at every k
        scores.append(score_retrieval(item, chunks, ks, fact_k))
        if on_progress:
            on_progress(n, len(items))

    by_type: dict[str, list[RetrievalScore]] = defaultdict(list)
    for s in scores:
        by_type[s.question_type].append(s)
    report = {qtype: summarize(group, ks) for qtype, group in sorted(by_type.items())}
    report["all"] = summarize(scores, ks)
    annotate(output=report["all"])
    return scores, report
