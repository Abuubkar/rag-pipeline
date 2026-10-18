"""Vendor-neutral tracing used by capability code.

    @traced("ingestion.chunk")
    def chunk_documents(...): ...

    annotate(output={"chunks": 2711})   # attach data to the current span

Everything is a no-op when no tracing backend is configured, so the pipeline runs fine
without Langfuse keys. Today the backend is Langfuse; LangSmith traces LangChain/LangGraph
by itself when LANGSMITH_TRACING=true (no code needed).
"""

import functools
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

from rag.packages.tracing import langfuse

F = TypeVar("F", bound=Callable[..., Any])


def enabled() -> bool:
    return langfuse.is_configured()


def traced(name: str, as_type: str = "span") -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not enabled():
                return fn(*args, **kwargs)
            with langfuse.span(name, as_type=as_type):
                return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def annotate(output: Any = None, metadata: Any = None) -> None:
    """Attach data to the current span."""
    if enabled():
        langfuse.update_current(output=output, metadata=metadata)


@contextmanager
def trace(name: str, input: Any = None, tags: list[str] | None = None,
          metadata: dict[str, str] | None = None) -> Iterator[None]:
    """Root span for one workflow run (one `ask`, one `ingest`)."""
    if not enabled():
        yield
        return
    with langfuse.trace_attributes(trace_name=name, tags=tags, metadata=metadata):
        with langfuse.span(name, as_type="chain", input=input):
            yield


def langchain_callbacks() -> list[Any]:
    """Handlers to pass into LangChain/LangGraph so their runs nest under the current span."""
    return [langfuse.langchain_handler()] if enabled() else []


def current_trace_url() -> str | None:
    return langfuse.current_trace_url() if enabled() else None


def flush() -> None:
    if enabled():
        langfuse.flush()
