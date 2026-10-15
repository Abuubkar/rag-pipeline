# 0007. Tracing: Langfuse behind a tiny vendor-neutral module

- Status: accepted
- Date: 2026-10-16

## Context
We want to learn Langfuse and LangSmith without spreading tracing calls for both vendors through every capability.

## Decision
- Capability code only uses `rag.observability.tracing`: `@traced("capability.step")`, `annotate(...)`, `trace(...)`.
- The backend is Langfuse (SDK v5, OpenTelemetry-based). LangChain/LangGraph runs nest under the current span
  through Langfuse's LangChain `CallbackHandler`, which the container passes in.
- LangSmith needs no code: `LANGSMITH_TRACING=true` makes LangChain/LangGraph report to it natively (non-OTel mode),
  which avoids both SDKs fighting over the global OpenTelemetry tracer provider.
- With no keys set, everything is a no-op.

## Consequences
Answer traces show `answer → answer-graph → retrieval.search / generation.* → llm.answer@1` with token usage.
