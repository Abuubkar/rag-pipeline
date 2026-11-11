"""Bookkeeping for evaluation runs: start a run, save its per-question scores, read runs back."""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from rag.db.models import EvalResultRow, EvalRunRow
from rag.domain.models import EvalRun


def start_run(session: Session, run: EvalRun) -> None:
    session.add(EvalRunRow(**run.model_dump(exclude={"created_at", "finished_at", "summary"}), summary=None))


def finish_run(
    session: Session, run_id: str, scores: list[BaseModel], summary: dict[str, dict[str, float]]
) -> None:
    session.add_all(
        EvalResultRow(run_id=run_id, item_id=s.item_id, question_type=s.question_type,  # type: ignore[attr-defined]
                      scores=s.model_dump(exclude={"item_id", "question_type"}))
        for s in scores
    )
    row = session.get_one(EvalRunRow, run_id)
    row.status, row.questions, row.summary = "done", len(scores), summary
    row.finished_at = datetime.now(timezone.utc)


def fail_run(session: Session, run_id: str) -> None:
    row = session.get(EvalRunRow, run_id)
    if row:
        row.status, row.finished_at = "failed", datetime.now(timezone.utc)


def _to_domain(row: EvalRunRow) -> EvalRun:
    return EvalRun.model_validate(row, from_attributes=True)


def list_runs(session: Session, limit: int = 20) -> list[EvalRun]:
    rows = session.scalars(select(EvalRunRow).order_by(EvalRunRow.created_at.desc()).limit(limit))
    return [_to_domain(r) for r in rows]


def get_run(session: Session, run_id: str) -> EvalRun | None:
    row = session.get(EvalRunRow, run_id)
    return _to_domain(row) if row else None


def set_label(session: Session, run_id: str, label: str | None) -> bool:
    row = session.get(EvalRunRow, run_id)
    if row is None:
        return False
    row.label = label
    return True


def labelled_runs(session: Session, kind: str) -> list[EvalRun]:
    """Finished, labelled runs of one kind, oldest first (the first one is the baseline)."""
    rows = session.scalars(
        select(EvalRunRow)
        .where(EvalRunRow.kind == kind, EvalRunRow.label.is_not(None), EvalRunRow.status == "done")
        .order_by(EvalRunRow.created_at)
    )
    return [_to_domain(r) for r in rows]


def get_results(session: Session, run_id: str) -> dict[str, dict[str, Any]]:
    rows = session.scalars(select(EvalResultRow).where(EvalResultRow.run_id == run_id))
    return {r.item_id: {"question_type": r.question_type, **r.scores} for r in rows}
