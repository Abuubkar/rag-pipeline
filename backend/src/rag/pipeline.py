"""THE STORY — read this file top to bottom to see the whole RAG system.

Workflow 1  ingest()               documents → clean → chunk → embed → versioned index
Workflow 2  build_answer_graph()   question → search → context → generate → citations
            answer()               runs the graph for one question
Workflow 3  golden_set()           evaluation questions with known answers, split dev / test
            evaluate_retrieval()   score search against the gold articles
            evaluate_answers()     score generated answers against the reference answers
                                   (retrieval=False: no-retrieval baseline)
            eval_runs() / eval_run()   saved runs, to compare setups over time

Folders hold the *capabilities* (ingestion/, retrieval/, generation/ …);
this file decides the *order* they run in.
"""

import time

from rag.container import Container
from rag.db import index_registry
from rag.db.session import session_scope
from rag.domain.models import Answer, EvalRun, GoldenItem, IndexVersion, IngestionReport
from rag.domain.state import AnswerState
from rag.evaluation import dataset, experiments, runs
from rag.generation import citations, context, generate
from rag.generation.prompts import load_prompt
from rag.ingestion import chunking, cleaning, indexing, loader
from rag.ingestion.indexing import ProgressFn
from rag.observability import tracing
from rag.packages.orchestration import graph as graphs
from rag.retrieval import search


# ══ WORKFLOW 1: INGEST ════════════════════════════════════════════════
def ingest(
    c: Container, limit: int | None = None, activate: bool = True, on_progress: ProgressFn | None = None
) -> IngestionReport:
    started = time.perf_counter()
    cfg = c.settings
    with tracing.trace("ingest", input={"limit": limit}, tags=["ingest"]):
        docs = loader.load_documents(cfg.dataset.name, cfg.dataset.revision, limit)   # 1. load
        docs = cleaning.clean_documents(docs)                                          # 2. clean
        manifest = indexing.build_manifest(cfg.dataset, docs, cfg.chunking, c.embedder.info())
        version_id = indexing.version_id_for(manifest)

        with session_scope(c.db_url) as s:
            existing = index_registry.get_version(s, version_id)
            index_registry.upsert_documents(s, docs)

        if existing and existing.status == "ready":                                    # no-op
            newly, skipped, n_chunks = 0, existing.indexed_chunks, existing.indexed_chunks
        else:
            chunks = chunking.chunk_documents(docs, cfg.chunking)                      # 3. chunk
            newly, skipped = indexing.index_chunks(                                    # 4+5. embed + index
                c.db_url, version_id, manifest, chunks, docs, c.embedder,
                cfg.embeddings.batch_size, cfg.chunking.contextual_header, on_progress,
            )
            n_chunks = len(chunks)

        if activate:
            with session_scope(c.db_url) as s:
                index_registry.set_active(s, cfg.index.alias, version_id)
        tracing.annotate(output={"version_id": version_id, "embedded": newly})

    tracing.flush()
    return IngestionReport(
        version_id=version_id, status="ready", documents=len(docs), chunks=n_chunks,
        newly_embedded=newly, skipped_existing=skipped,
        seconds=round(time.perf_counter() - started, 1), activated=activate,
    )


# ══ WORKFLOW 2: ANSWER A QUESTION (LangGraph) ═════════════════════════
def build_answer_graph(c: Container, version: IndexVersion, callbacks: list, with_search: bool = True):
    """with_search=False starts at generation: the caller puts `chunks` in the input state
    (evaluation retrieves everything first, so only one Ollama model is loaded at a time)."""
    prompt = load_prompt(c.settings.prompts.answer)
    chat = c.generator(callbacks)

    nodes = []
    if with_search:
        embedder = c.query_embedder(version)        # refuses a mismatched embedder
        nodes.append(("retrieval.search", search.search_node(
            db_url=c.db_url, version_id=version.id, embedder=embedder,
            top_k=c.settings.retrieval.top_k)))
    nodes += [
        ("generation.context",   context.context_node),
        ("generation.generate",  generate.generate_node(chat=chat, prompt=prompt)),
        ("generation.citations", citations.citations_node),
    ]
    return graphs.build_graph(AnswerState, nodes, graphs.linear_edges(nodes))


def answer(c: Container, question: str) -> tuple[Answer, str | None]:
    """Returns the answer and (if tracing is on) a link to its trace."""
    version = c.active_index()
    with tracing.trace("answer", input={"question": question}, tags=["answer"],
                       metadata={"index_version": version.id}):
        callbacks = tracing.langchain_callbacks()
        graph = build_answer_graph(c, version, callbacks)
        result = _run_answer_graph(graph, question, version, callbacks)
        tracing.annotate(output=result.text)
        trace_url = tracing.current_trace_url()
    tracing.flush()
    return result, trace_url


def _run_answer_graph(graph, question: str, version: IndexVersion, callbacks: list,
                      chunks: list | None = None) -> Answer:
    initial = {"question": question} if chunks is None else {"question": question, "chunks": chunks}
    state = graphs.invoke(graph, initial, callbacks, run_name="answer-graph")
    return Answer(
        question=question,
        text=state["completion"].text,
        citations=state["citations"],
        invalid_citations=state["invalid_citations"],
        sources=state["chunks"],
        version_id=version.id,
        prompt=state["prompt"],
        model=state["completion"].model,
    )


# ══ WORKFLOW 3: EVALUATE ══════════════════════════════════════════════
def golden_set(c: Container, split: str | None = None) -> list[GoldenItem]:
    cfg = c.settings
    items = dataset.load_golden_set(cfg.dataset.name, cfg.dataset.revision, cfg.evaluation.dev_fraction)
    return [i for i in items if split is None or i.split == split]


def evaluate_retrieval(
    c: Container, split: str = "dev", limit: int | None = None, on_progress=None
) -> EvalRun:
    version = c.active_index()
    embedder = c.query_embedder(version)
    items = dataset.sample(golden_set(c, split), limit)

    def search_fn(question: str, k: int):
        return search.dense_search(
            question, db_url=c.db_url, version_id=version.id, embedder=embedder, top_k=k
        )

    ks = sorted({1, 3, c.settings.retrieval.top_k, 10})
    run = runs.start(c.db_url, c.settings, "retrieval", "dense", split, limit, version.id,
                     extra={"ks": ks, "embedder": embedder.info().model_dump()})
    with tracing.trace("evaluate-retrieval", input={"split": split, "limit": limit},
                       tags=["eval"], metadata={"index_version": version.id, "eval_run": run.id}):
        with runs.recorded(c.db_url, run.id) as save:
            save(*experiments.run_retrieval_eval(
                items, search_fn, ks, fact_k=c.settings.retrieval.top_k, on_progress=on_progress
            ))
    tracing.flush()
    return runs.get(c.db_url, run.id)


def evaluate_answers(
    c: Container, split: str = "dev", limit: int | None = None,
    retrieval: bool = True, on_progress=None,
) -> EvalRun:
    """retrieval=False is the no-retrieval baseline: the same LLM answers with no sources.
    If it scores close to RAG, the model already knew the answers and RAG adds little."""
    version = c.active_index()
    items = dataset.sample(golden_set(c, split), limit)
    mode = "rag" if retrieval else "closed_book"
    prompt = c.settings.prompts.answer if retrieval else c.settings.prompts.closed_book
    run = runs.start(c.db_url, c.settings, "answers", mode, split, limit, version.id,
                     extra={"prompt": load_prompt(prompt).label})
    with tracing.trace(f"evaluate-answers-{mode}", input={"split": split, "limit": limit},
                       tags=["eval", mode], metadata={"index_version": version.id, "eval_run": run.id}):
        callbacks = tracing.langchain_callbacks()
        if retrieval:
            # Two phases, so only one Ollama model is in memory at a time:
            # 1. retrieve for every question (embedding model), then unload it;
            # 2. answer every question from those chunks (chat model only).
            embedder = c.query_embedder(version)
            retrieved = {}
            for n, item in enumerate(items, start=1):
                retrieved[item.question] = search.dense_search(
                    item.question, db_url=c.db_url, version_id=version.id,
                    embedder=embedder, top_k=c.settings.retrieval.top_k)
                if on_progress:
                    on_progress(n, len(items))
            embedder.unload()
            graph = build_answer_graph(c, version, callbacks, with_search=False)
            answer_fn = lambda q: _run_answer_graph(graph, q, version, callbacks, retrieved[q])  # noqa: E731
        else:
            answer_fn = _closed_book_answerer(c, version, callbacks)
        with runs.recorded(c.db_url, run.id) as save:
            save(*experiments.run_answer_eval(items, answer_fn, on_progress))
    tracing.flush()
    return runs.get(c.db_url, run.id)


def _closed_book_answerer(c: Container, version: IndexVersion, callbacks: list):
    prompt = load_prompt(c.settings.prompts.closed_book)
    chat = c.generator(callbacks)

    def answer_fn(question: str) -> Answer:
        completion = generate.closed_book(chat=chat, prompt=prompt, question=question)
        return Answer(question=question, text=completion.text, citations=[], invalid_citations=[],
                      sources=[], version_id=version.id, prompt=prompt.label, model=completion.model)

    return answer_fn



def eval_runs(c: Container, limit: int = 20) -> list[EvalRun]:
    return runs.recent(c.db_url, limit)


def eval_run(c: Container, run_id: str) -> EvalRun:
    return runs.get(c.db_url, run_id)
