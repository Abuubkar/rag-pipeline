"""Eval runs are saved in Postgres (eval_runs + eval_results) with everything needed to reproduce
them: config + its hash, index version, git commit, prompt, model and dataset revision."""

from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Iterator

from rag.core import provenance
from rag.core.config import Settings
from rag.db import eval_registry
from rag.db.session import session_scope
from rag.domain.models import EvalRun


def start(db_url: str, settings: Settings, kind: str, mode: str, split: str, limit: int | None,
          index_version_id: str, extra: dict[str, Any], label: str | None = None) -> EvalRun:
    snapshot = provenance.config_snapshot(settings)
    run = EvalRun(
        id=f"{kind}-{mode}-{datetime.now():%Y%m%d-%H%M%S}",
        kind=kind, mode=mode, split=split, status="running", label=label,
        index_version_id=index_version_id,
        config_hash=provenance.config_hash(snapshot),
        config=snapshot,
        provenance={**provenance.git_state(), "limit": limit,
                    "model": settings.llm.generator.model,
                    "dataset_revision": settings.dataset.revision, **extra},
    )
    with session_scope(db_url) as s:
        eval_registry.start_run(s, run)
    return run


@contextmanager
def recorded(db_url: str, run_id: str) -> Iterator[Callable[[list, dict], None]]:
    """Yields save(scores, report). If anything raises (an error, Ctrl-C) the run is marked failed."""
    def save(scores: list, report: dict) -> None:
        with session_scope(db_url) as s:
            eval_registry.finish_run(s, run_id, scores, report)

    try:
        yield save
    except BaseException:
        with session_scope(db_url) as s:
            eval_registry.fail_run(s, run_id)
        raise


def get(db_url: str, run_id: str) -> EvalRun:
    with session_scope(db_url) as s:
        run = eval_registry.get_run(s, run_id)
    assert run is not None
    return run


def recent(db_url: str, limit: int = 20) -> list[EvalRun]:
    with session_scope(db_url) as s:
        return eval_registry.list_runs(s, limit)


def set_label(db_url: str, run_id: str, label: str | None) -> bool:
    with session_scope(db_url) as s:
        return eval_registry.set_label(s, run_id, label)


def labelled(db_url: str, kind: str) -> list[EvalRun]:
    with session_scope(db_url) as s:
        return eval_registry.labelled_runs(s, kind)
