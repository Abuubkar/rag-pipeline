"""The golden set: questions with known answers and known evidence articles, split into dev and test.

dev  — run it as often as you like while tuning (chunk size, top_k, prompts…).
test — run it only to report final numbers. Tuning on test makes the numbers look better than
       they really are, because you end up fitting the pipeline to those exact questions.
"""

import hashlib
from collections import Counter, defaultdict

from rag.domain.models import GoldenItem
from rag.observability.tracing import annotate, traced
from rag.packages.documents import multihop_rag


@traced("evaluation.dataset")
def load_golden_set(dataset: str, revision: str, dev_fraction: float) -> list[GoldenItem]:
    items = multihop_rag.load_questions(dataset, revision)

    # Stratified split: each question type gets the same dev share, so dev looks like test.
    # Items are ordered by a hash of their id, not shuffled with a seed, so an item always lands
    # in the same split, even if the dataset order changes or items are added.
    by_type: dict[str, list[GoldenItem]] = defaultdict(list)
    for item in items:
        by_type[item.question_type].append(item)
    for group in by_type.values():
        group.sort(key=lambda i: hashlib.sha256(i.id.encode()).hexdigest())
        n_dev = round(len(group) * dev_fraction)
        for n, item in enumerate(group):
            item.split = "dev" if n < n_dev else "test"

    annotate(output=dict(Counter(f"{i.split}/{i.question_type}" for i in items)))
    return items


def sample(items: list[GoldenItem], limit: int | None) -> list[GoldenItem]:
    """The first `limit` items in hash order: a fixed, roughly proportional mix of question types.
    (The dataset's own order groups types together, so its first N would be lopsided.)"""
    if limit is None:
        return items
    return sorted(items, key=lambda i: hashlib.sha256(i.id.encode()).hexdigest())[:limit]
