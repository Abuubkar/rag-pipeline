"""Is a difference between two eval runs real, or noise? Paired bootstrap confidence intervals.

Two runs answered the SAME questions, so compare them question by question (paired): the
per-question difference cancels out how hard each question is. An unpaired comparison of two
averages would have to treat that difficulty spread as noise, and would need far bigger gaps.

Bootstrap: resample the questions with replacement (same size, some picked twice, some not at
all), average the per-question differences, and repeat thousands of times. The middle 95% of
those averages is the 95% confidence interval of the improvement. If it doesn't contain 0, the
change is unlikely to be luck on these particular questions.

It says nothing about other corpora or question styles, and with many comparisons a few
"significant" ones are expected by chance (1 in 20 at 95%). So treat a CI that barely excludes 0
with care.
"""

import random
from typing import Any

from pydantic import BaseModel

RESAMPLES = 5000
LEVEL = 0.95


class Comparison(BaseModel):
    """One metric, run B minus reference run A, over the questions both answered."""

    metric: str
    questions: int
    delta: float      # mean per-question difference
    low: float        # CI bounds of that mean
    high: float
    better: int       # questions where B scored higher
    worse: int

    @property
    def significant(self) -> bool:
        return self.low > 0 or self.high < 0


def metric_value(scores: dict[str, Any], metric: str) -> float:
    """One question's value of a summary metric name, from its stored RetrievalScore."""
    if metric == "mrr":
        return scores["reciprocal_rank"]
    if "@" in metric:
        name, k = metric.split("@")
        return float(scores[name][k])
    return float(scores[metric])


def paired_bootstrap(diffs: list[float], resamples: int = RESAMPLES, level: float = LEVEL,
                     seed: int = 0) -> tuple[float, float, float]:
    """Mean of the paired differences and its percentile CI. Fixed seed: the same runs always
    give the same interval, so a regenerated report doesn't change by itself."""
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(resamples))
    tail = (1 - level) / 2
    return sum(diffs) / n, means[int(tail * resamples)], means[int((1 - tail) * resamples) - 1]


def compare(reference: dict[str, dict], run: dict[str, dict], metrics: list[str]) -> list[Comparison]:
    """reference and run: item_id → stored scores (eval_results). Only shared questions count."""
    shared = sorted(reference.keys() & run.keys())
    out = []
    for metric in metrics:
        diffs = [metric_value(run[i], metric) - metric_value(reference[i], metric) for i in shared]
        delta, low, high = paired_bootstrap(diffs)
        out.append(Comparison(metric=metric, questions=len(shared), delta=delta, low=low, high=high,
                              better=sum(d > 0 for d in diffs), worse=sum(d < 0 for d in diffs)))
    return out
