"""Before searching: let the LLM change what gets searched (retrieval.query_transform).

none         search the question as asked.
rewrite      search ONE short query instead: key names and terms, no question words. Helps
             keyword search most, since long questions add many weak OR terms.
multi_query  search the question AND n other queries, then fuse every result list with RRF.
             Multi-hop questions need several articles, and one query usually finds only one.
hyde         Hypothetical Document Embeddings: the LLM writes a fake answer passage. Its vector is
             compared document-to-document with the chunks (it's embedded as a document), which
             is often closer than question-to-document. The question is still searched too, so a
             made-up detail in the passage can't take over the results.

Each costs one LLM call per question, so it adds seconds of latency before the search starts.
"""

import re
from typing import Callable

from rag.core.prompts import Prompt
from rag.domain.models import SearchQueries
from rag.domain.state import AnswerState
from rag.observability.tracing import annotate
from rag.packages.llm.ollama import OllamaChat

TRANSFORMS = ("none", "rewrite", "multi_query", "hyde")
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")  # "1. ", "- " if the model numbers its lines anyway


def transform(question: str, *, mode: str, chat: OllamaChat | None, prompt: Prompt | None, n: int) -> SearchQueries:
    if mode == "none":
        return SearchQueries(queries=[question])
    if mode not in TRANSFORMS:
        raise ValueError(f"Unknown query transform: {mode} ({' | '.join(TRANSFORMS)})")
    assert chat and prompt
    text = chat.complete(system=prompt.system.format(n=n), user=prompt.render_user(question=question, n=str(n)),
                         run_name=f"llm.{prompt.label}").text
    lines = [_LIST_MARKER.sub("", line).strip().strip('"') for line in text.splitlines()]
    lines = [line for line in lines if line]
    if mode == "rewrite":
        return SearchQueries(queries=lines[:1] or [question])  # the model said nothing: fall back
    if mode == "multi_query":
        return SearchQueries(queries=[question, *lines[:n]])
    return SearchQueries(queries=[question], passages=[text])  # hyde


def transform_node(**transform_kwargs) -> Callable[[AnswerState], dict]:
    """transform_kwargs: everything `transform` takes except the question."""
    def node(state: AnswerState) -> dict:
        searched = transform(state["question"], **transform_kwargs)
        annotate(output=searched.model_dump())
        return {"searched": searched}

    return node
