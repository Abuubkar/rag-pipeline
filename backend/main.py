"""Single entry point.

    uv run main.py db-upgrade                 # create / migrate the Postgres schema
    uv run main.py ingest [--limit 50]        # build (or reuse) an index version
    uv run main.py ask "your question"        # answer with citations
    uv run main.py index list                 # see index versions
    uv run main.py index activate <id>        # point `ask` at a version
"""

import subprocess
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TimeElapsedColumn
from rich.table import Table

BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")  # Langfuse / LangSmith keys must be in os.environ

from rag import pipeline  # noqa: E402
from rag.container import Container  # noqa: E402
from rag.core.logging import setup_logging  # noqa: E402
from rag.db import index_registry  # noqa: E402
from rag.db.session import session_scope  # noqa: E402
from rag.domain.errors import RagError  # noqa: E402
from rag.observability import tracing  # noqa: E402

app = typer.Typer(no_args_is_help=True, add_completion=False)
index_app = typer.Typer(no_args_is_help=True, help="Manage index versions.")
app.add_typer(index_app, name="index")
console = Console()


@app.callback()
def _setup() -> None:
    setup_logging()
    if not tracing.enabled():
        console.print("[dim]Tracing off (set LANGFUSE_* keys in backend/.env to enable).[/dim]")


@app.command("db-upgrade")
def db_upgrade() -> None:
    """Apply database migrations (Alembic)."""
    sys.exit(subprocess.call(["alembic", "upgrade", "head"], cwd=BACKEND_DIR))


@app.command()
def ingest(
    limit: int = typer.Option(None, help="Only ingest the first N articles (quick experiments)."),
    activate: bool = typer.Option(True, help="Point the active alias at this version."),
) -> None:
    """Load → clean → chunk → embed → index (versioned, resumable)."""
    c = Container()
    with Progress("[progress.description]{task.description}", BarColumn(),
                  MofNCompleteColumn(), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("Embedding chunks", total=None)
        report = pipeline.ingest(
            c, limit=limit, activate=activate,
            on_progress=lambda done, total: progress.update(task, completed=done, total=total),
        )
    table = Table(show_header=False, title="Ingestion report")
    for key, value in report.model_dump().items():
        table.add_row(key, str(value))
    console.print(table)


@app.command()
def ask(question: str) -> None:
    """Answer a question from the active index, with citations."""
    answer, trace_url = pipeline.answer(Container(), question)
    console.print(Panel(answer.text, title="Answer", subtitle=f"{answer.model} · {answer.prompt}"))

    table = Table(title=f"Retrieved sources (index {answer.version_id})")
    for col in ("#", "score", "title", "source", "cited"):
        table.add_column(col)
    cited = {c.number for c in answer.citations}
    for n, s in enumerate(answer.sources, start=1):
        table.add_row(str(n), f"{s.score:.3f}", s.title[:70], s.source or "", "✓" if n in cited else "")
    console.print(table)
    if answer.invalid_citations:
        console.print(f"[yellow]Invalid citations (no such source): {answer.invalid_citations}[/yellow]")
    if trace_url:
        console.print(f"Trace: {trace_url}")


@index_app.command("list")
def index_list() -> None:
    """Show all index versions and which one is active."""
    c = Container()
    with session_scope(c.db_url) as s:
        versions = index_registry.list_versions(s)
        active = index_registry.get_active(s, c.settings.index.alias)
    table = Table()
    for col in ("", "version", "status", "chunks", "chunker", "embedder", "docs", "created"):
        table.add_column(col)
    for v in versions:
        m = v.manifest
        table.add_row(
            "★" if active and active.id == v.id else "", v.id, v.status,
            f"{v.indexed_chunks}/{v.expected_chunks}",
            f"{m.chunking['library']}/{m.chunking['splitter']} {m.chunking['chunk_size']}/{m.chunking['chunk_overlap']}",
            m.dense.model, str(m.dataset["documents"]),
            v.created_at.strftime("%Y-%m-%d %H:%M") if v.created_at else "",
        )
    console.print(table)


@index_app.command("activate")
def index_activate(version_id: str) -> None:
    """Point the active alias at a ready version."""
    c = Container()
    with session_scope(c.db_url) as s:
        version = index_registry.get_version(s, version_id)
        if not version or version.status != "ready":
            raise RagError(f"Version '{version_id}' does not exist or is not ready.")
        index_registry.set_active(s, c.settings.index.alias, version_id)
    console.print(f"Active index '{c.settings.index.alias}' → {version_id}")


def main() -> None:
    try:
        app()
    except RagError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
