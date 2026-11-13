"""THE STORY — read this file top to bottom to see the whole RAG system.

Workflow 1  ingest()               documents → clean → chunk → embed → versioned index
Workflow 2  build_answer_graph()   question → search → context → generate → citations
            answer()               runs the graph for one question
Workflow 3  golden_set()           evaluation questions with known answers, split dev / test
            evaluate_retrieval()   score search against the gold articles
            evaluate_answers()     score generated answers against the reference answers
                                   (retrieval=False: no-retrieval baseline)
            eval_runs() / eval_run()   saved runs, to compare setups over time
            write_results()        docs/results: before/after tables + charts from labelled runs

Folders hold the *capabilities* (ingestion/, retrieval/, generation/ …);
this file decides the *order* they run in.
"""

import time

from rag.container import Container
from rag.db import index_registry
from rag.db.session import session_scope
from rag.domain.models import Answer, EvalRun, GoldenItem, IndexVersion, IngestionReport, SearchQueries
from rag.domain.state import AnswerState
from rag.core.config import RESULTS_DIR
from rag.evaluation import dataset, experiments, report, runs
from rag.generation import citations, context, generate
from rag.core.prompts import load_prompt
from rag.ingestion import chunking, cleaning, indexing, loader
from rag.ingestion.indexing import ProgressFn
from rag.observability import tracing
from rag.packages.orchestration import graph as graphs
from rag.retrieval import compress, query_transform, rerank, search


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
            chunks = chunking.chunk_documents(docs, cfg.chunking, c.embedder.embed_documents)  # 3. chunk
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
        if c.settings.retrieval.query_transform != "none":
            nodes.append(("retrieval.transform", query_transform.transform_node(**_transform_kwargs(c, chat))))
        embedder = c.query_embedder(version)        # refuses a mismatched embedder
        nodes.append(("retrieval.search", search.search_node(**_search_kwargs(c, version, embedder))))
        if c.settings.retrieval.rerank != "none":
            nodes.append(("retrieval.rerank", rerank.rerank_node(top_k=c.settings.retrieval.top_k,
                                                                 **_rerank_kwargs(c, chat))))
        if c.settings.retrieval.compression != "none":
            nodes.append(("retrieval.compress", compress.compress_node(**_compress_kwargs(c, embedder))))
    nodes += [
        ("generation.context",   context.context_node),
        ("generation.generate",  generate.generate_node(chat=chat, prompt=prompt)),
        ("generation.citations", citations.citations_node),
    ]
    return graphs.build_graph(AnswerState, nodes, graphs.linear_edges(nodes))


def _transform_kwargs(c: Container, chat) -> dict:
    r = c.settings.retrieval
    prompt = None if r.query_transform == "none" else load_prompt(r.query_transform)
    return dict(mode=r.query_transform, chat=chat, prompt=prompt, n=r.multi_queries)


def _search_kwargs(c: Container, version: IndexVersion, embedder) -> dict:
    """With a reranker, search brings `rerank_candidates` chunks and the reranker keeps top_k."""
    r = c.settings.retrieval
    top_k = r.top_k if r.rerank == "none" else max(r.rerank_candidates, r.top_k)
    return dict(mode=r.mode, db_url=c.db_url, version_id=version.id, embedder=embedder,
                top_k=top_k, candidates=r.candidates, rrf_k=r.rrf_k, filter_mode=r.metadata_filter)


def _retrieval_mode(c: Container) -> str:
    """The run name for the configured retrieval setup, e.g. "hybrid-source" (none parts left out)."""
    r = c.settings.retrieval
    return "-".join(m for m in (r.mode, r.query_transform, r.metadata_filter, r.rerank, r.compression)
                    if m != "none")


def _compress_kwargs(c: Container, embedder) -> dict:
    r = c.settings.retrieval
    return dict(mode=r.compression, embedder=embedder, keep=r.compression_keep)


def _rerank_kwargs(c: Container, chat) -> dict:
    r = c.settings.retrieval
    return dict(chat=chat, prompt=load_prompt("rerank"), passage_chars=r.rerank_passage_chars)


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
    c: Container, split: str = "dev", limit: int | None = None, on_progress=None,
    label: str | None = None,
) -> EvalRun:
    r = c.settings.retrieval
    version = c.active_index()
    items = [i for i in dataset.sample(golden_set(c, split), limit) if i.answerable]
    ks = sorted({1, 3, r.top_k, 10})
    mode = _retrieval_mode(c)
    run = runs.start(c.db_url, c.settings, "retrieval", mode, split, limit, version.id,
                     extra={"ks": ks, "embedder": version.manifest.dense.model_dump(),
                            "query_prompt": _query_prompt_label(c),
                            "rerank_prompt": None if r.rerank == "none" else load_prompt("rerank").label},
                     label=label)
    with tracing.trace("evaluate-retrieval", input={"split": split, "limit": limit},
                       tags=["eval"], metadata={"index_version": version.id, "eval_run": run.id}):
        # Phases, so only one Ollama model is in memory at a time:
        # 1. (query transform only) the chat model writes every question's search queries, unloads;
        # 2. the embedding model searches for every question;
        # 3. (rerank only) the embedding model unloads and the chat model reorders the candidates;
        # 4. (compression only) the embedding model trims the top_k chunks' sentences.
        searched = _transform_all(c, items, on_progress)
        retrieved = _search_all(c, version, items, searched, max(ks), on_progress)
        retrieved = _rerank_all(c, items, retrieved, on_progress)
        retrieved = _compress_all(c, version, items, retrieved, on_progress)

        def search_fn(question: str, k: int):
            return retrieved[question][:k]

        with runs.recorded(c.db_url, run.id) as save:
            scores, report = experiments.run_retrieval_eval(items, search_fn, ks, fact_k=r.top_k,
                                                            on_progress=on_progress)
            if r.query_transform != "none":  # keep what was searched, to read next to each score
                questions = {i.id: i.question for i in items}
                scores = [s.model_copy(update={"searched": searched[questions[s.item_id]]}) for s in scores]
            save(scores, report)
    tracing.flush()
    return runs.get(c.db_url, run.id)


def _transform_all(c: Container, items: list[GoldenItem], on_progress=None) -> dict[str, SearchQueries]:
    """Phase 1 of an eval: the chat model writes every question's search queries, then unloads,
    so it's never in memory next to the embedding model. With no transform, nothing is loaded."""
    if c.settings.retrieval.query_transform == "none":
        return {i.question: SearchQueries(queries=[i.question]) for i in items}
    chat = c.generator(tracing.langchain_callbacks())
    kwargs = _transform_kwargs(c, chat)
    searched = {}
    for n, item in enumerate(items, start=1):
        searched[item.question] = query_transform.transform(item.question, **kwargs)
        if on_progress:
            on_progress(n, len(items))
    chat.unload()
    return searched


def _search_all(c: Container, version: IndexVersion, items: list[GoldenItem],
                searched: dict[str, SearchQueries], at_least: int, on_progress=None) -> dict:
    """Phase 2: search every question (embedding model only), then unload it."""
    embedder = c.query_embedder(version)
    kwargs = _search_kwargs(c, version, embedder)
    kwargs["top_k"] = max(kwargs["top_k"], at_least)
    retrieved = {}
    for n, item in enumerate(items, start=1):
        retrieved[item.question] = search.search(searched[item.question], **kwargs)
        if on_progress:
            on_progress(n, len(items))
    embedder.unload()
    return retrieved


def _rerank_all(c: Container, items: list[GoldenItem], retrieved: dict, on_progress=None) -> dict:
    """Phase 3 (rerank only): the chat model reorders every question's candidates, then unloads."""
    if c.settings.retrieval.rerank == "none":
        return retrieved
    chat = c.generator(tracing.langchain_callbacks())
    kwargs = _rerank_kwargs(c, chat)
    reranked = {}
    for n, item in enumerate(items, start=1):
        reranked[item.question] = rerank.rerank(item.question, retrieved[item.question], **kwargs)
        if on_progress:
            on_progress(n, len(items))
    chat.unload()
    return reranked


def _compress_all(c: Container, version: IndexVersion, items: list[GoldenItem], retrieved: dict,
                  on_progress=None) -> dict:
    """Phase 4 (compression only): the embedding model trims every question's top_k chunks, then
    unloads. Chunks below top_k stay as they are; only recall@10 looks at them, by article."""
    if c.settings.retrieval.compression == "none":
        return retrieved
    top_k = c.settings.retrieval.top_k
    embedder = c.query_embedder(version)
    kwargs = _compress_kwargs(c, embedder)
    compressed = {}
    for n, item in enumerate(items, start=1):
        chunks = retrieved[item.question]
        compressed[item.question] = compress.compress(item.question, chunks[:top_k], **kwargs) + chunks[top_k:]
        if on_progress:
            on_progress(n, len(items))
    embedder.unload()
    return compressed


def _query_prompt_label(c: Container) -> str | None:
    t = c.settings.retrieval.query_transform
    return None if t == "none" else load_prompt(t).label


def evaluate_answers(
    c: Container, split: str = "dev", limit: int | None = None,
    retrieval: bool = True, on_progress=None, label: str | None = None,
) -> EvalRun:
    """retrieval=False is the no-retrieval baseline: the same LLM answers with no sources.
    If it scores close to RAG, the model already knew the answers and RAG adds little."""
    version = c.active_index()
    items = dataset.sample(golden_set(c, split), limit)
    mode = "rag" if retrieval else "closed_book"
    prompt = c.settings.prompts.answer if retrieval else c.settings.prompts.closed_book
    run = runs.start(c.db_url, c.settings, "answers", mode, split, limit, version.id,
                     extra={"prompt": load_prompt(prompt).label,
                            "query_prompt": _query_prompt_label(c) if retrieval else None,
                            "rerank_prompt": load_prompt("rerank").label
                            if retrieval and c.settings.retrieval.rerank != "none" else None}, label=label)
    with tracing.trace(f"evaluate-answers-{mode}", input={"split": split, "limit": limit},
                       tags=["eval", mode], metadata={"index_version": version.id, "eval_run": run.id}):
        callbacks = tracing.langchain_callbacks()
        if retrieval:
            # Phases, so only one Ollama model is in memory at a time: retrieve for every
            # question (query transform → search → rerank, as configured), then answer every
            # question from those chunks with the chat model only.
            searched = _transform_all(c, items, on_progress)
            retrieved = _search_all(c, version, items, searched, 0, on_progress)
            retrieved = _rerank_all(c, items, retrieved, on_progress)
            retrieved = {q: chunks[:c.settings.retrieval.top_k] for q, chunks in retrieved.items()}
            retrieved = _compress_all(c, version, items, retrieved, on_progress)
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


def label_eval_run(c: Container, run_id: str, label: str | None) -> bool:
    return runs.set_label(c.db_url, run_id, label)


def write_results(c: Container):
    retrieval_runs = runs.labelled(c.db_url, "retrieval")
    results = {r.id: runs.results(c.db_url, r.id) for r in retrieval_runs}
    reference = (_retrieval_mode(c), c.active_index().id)  # the configured default setup, active index
    return report.write_report(RESULTS_DIR, retrieval_runs, runs.labelled(c.db_url, "answers"),
                               results, reference)
