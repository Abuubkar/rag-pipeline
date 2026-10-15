"""THE STORY — read this file top to bottom to see the whole RAG system.

Workflow 1  ingest()               documents → clean → chunk → embed → versioned index
Workflow 2  build_answer_graph()   question → search → context → generate → citations
            answer()               runs the graph for one question

Folders hold the *capabilities* (ingestion/, retrieval/, generation/ …);
this file decides the *order* they run in.
"""

import time

from rag.container import Container
from rag.db import index_registry
from rag.db.session import session_scope
from rag.domain.models import Answer, IndexVersion, IngestionReport
from rag.domain.state import AnswerState
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
def build_answer_graph(c: Container, version: IndexVersion, callbacks: list):
    prompt = load_prompt(c.settings.prompts.answer)
    embedder = c.query_embedder(version)            # refuses a mismatched embedder
    chat = c.generator(callbacks)

    nodes = [
        ("retrieval.search", search.search_node(
            db_url=c.db_url, version_id=version.id, embedder=embedder,
            top_k=c.settings.retrieval.top_k)),
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
        state = graphs.invoke(graph, {"question": question}, callbacks, run_name="answer-graph")
        result = Answer(
            question=question,
            text=state["completion"].text,
            citations=state["citations"],
            invalid_citations=state["invalid_citations"],
            sources=state["chunks"],
            version_id=version.id,
            prompt=state["prompt"],
            model=state["completion"].model,
        )
        tracing.annotate(output=result.text)
        trace_url = tracing.current_trace_url()
    tracing.flush()
    return result, trace_url
