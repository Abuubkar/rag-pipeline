"""Single entry point.

    uv run main.py db-upgrade                 # create / migrate the Postgres schema
    uv run main.py ingest [--limit 50]        # build (or reuse) an index version
    uv run main.py ask "your question"        # answer with citations
    uv run main.py index list                 # see index versions
    uv run main.py index activate <id>        # point `ask` at a version
    uv run main.py eval dataset               # golden set: questions per type and split
    uv run main.py eval retrieval [--split dev] # hit@k, recall@k, MRR, fact recall
    uv run main.py eval answers [--limit 50]  # accuracy, exact match, refusals, citations
    uv run main.py eval answers --no-retrieval  # baseline: same LLM, no sources
    uv run main.py eval runs                  # saved runs (Postgres), newest first
    uv run main.py eval show <run-id>         # a saved run's report + provenance
"""

import subprocess
from contextlib import contextmanager
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
eval_app = typer.Typer(no_args_is_help=True, help="Evaluate the pipeline on the golden set.")
app.add_typer(eval_app, name="eval")
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


@eval_app.command("dataset")
def eval_dataset() -> None:
    """Show the golden set: how many questions of each type are in dev and test."""
    items = pipeline.golden_set(Container())
    table = Table(title=f"Golden set ({len(items)} questions)")
    for col in ("question type", "dev", "test", "avg evidence articles", "example answer"):
        table.add_column(col)
    for qtype in sorted({i.question_type for i in items}):
        group = [i for i in items if i.question_type == qtype]
        table.add_row(
            qtype,
            str(sum(i.split == "dev" for i in group)),
            str(sum(i.split == "test" for i in group)),
            f"{sum(len(i.evidence_document_ids) for i in group) / len(group):.1f}",
            group[0].answer[:40],
        )
    console.print(table)


@eval_app.command("retrieval")
def eval_retrieval(
    split: str = typer.Option("dev", help="dev (for tuning) or test (final numbers only)."),
    limit: int = typer.Option(None, help="Only the first N questions (quick check)."),
) -> None:
    """Score search: does it bring back the gold articles and evidence sentences?"""
    c = Container()
    with _progress("Searching") as on_progress:
        run = pipeline.evaluate_retrieval(c, split, limit, on_progress=on_progress)
    _print_run(run)


@eval_app.command("answers")
def eval_answers(
    split: str = typer.Option("dev", help="dev (for tuning) or test (final numbers only)."),
    limit: int = typer.Option(None, help="Only the first N questions (each one is an LLM call)."),
    retrieval: bool = typer.Option(True, help="--no-retrieval: baseline that answers without sources."),
) -> None:
    """Score generated answers against the reference answers (runs the full answer graph)."""
    c = Container()
    with _progress("Answering") as on_progress:
        run = pipeline.evaluate_answers(c, split, limit, retrieval, on_progress=on_progress)
    _print_run(run)


@eval_app.command("runs")
def eval_runs(limit: int = typer.Option(20, help="How many recent runs to show.")) -> None:
    """List saved eval runs with their headline number."""
    table = Table(title="Eval runs (newest first)")
    for col in ("run", "status", "split", "n", "headline", "commit"):
        table.add_column(col, no_wrap=col == "run")
    for run in pipeline.eval_runs(Container(), limit):
        headline = ""
        if run.summary:
            key = "recall@" + str(Container().settings.retrieval.top_k) if run.kind == "retrieval" else "accuracy"
            headline = f"{key} {run.summary['all'][key]:.2f}"
        commit = (run.provenance.get("git_commit") or "") + ("*" if run.provenance.get("git_dirty") else "")
        table.add_row(run.id, run.status, run.split, str(run.questions), headline, commit)
    console.print(table)
    console.print("[dim]* = uncommitted changes when the run started · details: main.py eval show <run>[/dim]")


@eval_app.command("show")
def eval_show(run_id: str) -> None:
    """Show a saved run's report and how it was produced."""
    _print_run(pipeline.eval_run(Container(), run_id))


def _print_run(run) -> None:
    p = run.provenance
    what = {"dense": "dense search", "rag": "RAG answers", "closed_book": "no-retrieval baseline"}[run.mode]
    detail = f"fact recall @ top {run.config['retrieval']['top_k']}" if run.kind == "retrieval" else f"{p['model']} · {p['prompt']}"
    _print_report(run.summary or {}, f"{what} on {run.split} · {detail}")
    console.print(f"[dim]run {run.id} · status {run.status} · index {run.index_version_id} · config {run.config_hash} · "
                  f"commit {p.get('git_commit')}{'*' if p.get('git_dirty') else ''}[/dim]")


def _print_report(report: dict[str, dict[str, float]], title: str) -> None:
    """Metrics as rows, question types as columns."""
    if not report:
        return
    table = Table(title=title)
    table.add_column("metric")
    for qtype in report:
        table.add_column(qtype.removesuffix("_query"), justify="right", style="bold" if qtype == "all" else None)
    for metric in report["all"]:
        fmt = (lambda v: str(v)) if metric == "questions" else (lambda v: f"{v:.2f}")
        table.add_row(metric, *(fmt(summary[metric]) for summary in report.values()))
    console.print(table)


@contextmanager
def _progress(label: str):
    with Progress("[progress.description]{task.description}", BarColumn(),
                  MofNCompleteColumn(), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task(label, total=None)
        yield lambda done, total: progress.update(task, completed=done, total=total)


def main() -> None:
    try:
        app()
    except RagError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
