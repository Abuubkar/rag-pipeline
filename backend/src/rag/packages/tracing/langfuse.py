"""Langfuse (SDK v5, OpenTelemetry-based).

- `span()` opens a nested observation; spans opened inside it become its children.
- `langchain_handler()` makes LangChain/LangGraph runs (graph nodes, LLM calls with token usage)
  appear inside whatever span is currently open.
Keys come from LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL.
"""

import os
from contextlib import contextmanager
from typing import Any, Iterator

from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler


def is_configured() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


@contextmanager
def span(name: str, as_type: str = "span", input: Any = None, metadata: Any = None) -> Iterator[Any]:
    with get_client().start_as_current_observation(
        name=name, as_type=as_type, input=input, metadata=metadata  # type: ignore[arg-type]
    ) as obs:
        yield obs


def update_current(output: Any = None, metadata: Any = None) -> None:
    get_client().update_current_span(output=output, metadata=metadata)


@contextmanager
def trace_attributes(
    trace_name: str, tags: list[str] | None = None, metadata: dict[str, str] | None = None
) -> Iterator[None]:
    with propagate_attributes(trace_name=trace_name, tags=tags, metadata=metadata):
        yield


def langchain_handler() -> Any:
    return CallbackHandler()


def current_trace_url() -> str | None:
    client = get_client()
    trace_id = client.get_current_trace_id()
    return client.get_trace_url(trace_id=trace_id) if trace_id else None


def flush() -> None:
    get_client().flush()
