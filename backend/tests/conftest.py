from pathlib import Path

import pytest
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from rag.container import Container  # noqa: E402
from rag.db.session import get_engine  # noqa: E402
from rag.domain.errors import NoActiveIndex  # noqa: E402


@pytest.fixture(scope="session")
def container() -> Container:
    c = Container()
    try:
        with get_engine(c.db_url).connect() as conn:
            conn.execute(text("select 1"))
    except Exception:
        pytest.skip("Postgres is not running (docker compose up -d)")
    return c


@pytest.fixture(scope="session")
def active_version(container: Container):
    try:
        return container.active_index()
    except NoActiveIndex:
        pytest.skip("No active index (run: uv run main.py ingest)")
