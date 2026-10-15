"""Bookkeeping for documents and index versions (plain SQL, no vector logic)."""

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from rag.db.models import ActiveIndexRow, DocumentRow, IndexVersionRow
from rag.domain.models import Document, IndexManifest, IndexVersion


def upsert_documents(session: Session, docs: list[Document]) -> None:
    rows = [d.model_dump() for d in docs]
    stmt = insert(DocumentRow).values(rows)
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=["id"],
            set_={c: stmt.excluded[c] for c in rows[0] if c != "id"},
        )
    )


def _to_domain(row: IndexVersionRow) -> IndexVersion:
    return IndexVersion(
        id=row.id,
        manifest=IndexManifest.model_validate(row.manifest),
        status=row.status,
        expected_chunks=row.expected_chunks,
        indexed_chunks=row.indexed_chunks,
        error=row.error,
        created_at=row.created_at,
        finished_at=row.finished_at,
    )


def get_version(session: Session, version_id: str) -> IndexVersion | None:
    row = session.get(IndexVersionRow, version_id)
    return _to_domain(row) if row else None


def list_versions(session: Session) -> list[IndexVersion]:
    rows = session.scalars(select(IndexVersionRow).order_by(IndexVersionRow.created_at))
    return [_to_domain(r) for r in rows]


def start_building(session: Session, version_id: str, manifest: IndexManifest, expected: int) -> None:
    row = session.get(IndexVersionRow, version_id)
    if row is None:
        row = IndexVersionRow(id=version_id, manifest=manifest.model_dump(mode="json"))
        session.add(row)
    row.status = "building"
    row.expected_chunks = expected
    row.error = None
    row.finished_at = None


def record_progress(session: Session, version_id: str, indexed: int) -> None:
    session.get_one(IndexVersionRow, version_id).indexed_chunks = indexed


def finish(session: Session, version_id: str, status: str, error: str | None = None) -> None:
    row = session.get_one(IndexVersionRow, version_id)
    row.status = status
    row.error = error
    row.finished_at = datetime.now(timezone.utc)


def set_active(session: Session, alias: str, version_id: str) -> None:
    stmt = insert(ActiveIndexRow).values(alias=alias, version_id=version_id)
    session.execute(
        stmt.on_conflict_do_update(index_elements=["alias"], set_={"version_id": version_id})
    )


def get_active(session: Session, alias: str) -> IndexVersion | None:
    row = session.get(ActiveIndexRow, alias)
    return get_version(session, row.version_id) if row else None
