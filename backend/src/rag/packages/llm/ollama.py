"""Chat model served by Ollama, through LangChain's ChatOllama.

`callbacks` are LangChain callback handlers (e.g. Langfuse) handed in by the container,
so this package never imports a tracing library itself.
"""

from typing import Any

import ollama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from rag.domain.models import Completion


class OllamaChat:
    def __init__(
        self,
        model: str,
        base_url: str,
        temperature: float = 0.0,
        seed: int | None = None,
        num_ctx: int | None = None,
        callbacks: list[Any] | None = None,
    ):
        self.model = model
        self._client = ollama.Client(host=base_url)
        self._callbacks = callbacks or []
        self._lc = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            seed=seed,
            num_ctx=num_ctx,
            reasoning=False,  # no hidden "thinking" tokens; answers only
        )

    def complete(self, system: str, user: str, run_name: str = "llm") -> Completion:
        message = self._lc.invoke(
            [SystemMessage(system), HumanMessage(user)],
            config={"callbacks": self._callbacks, "run_name": run_name},
        )
        usage = message.usage_metadata or {}
        return Completion(
            text=str(message.content).strip(),
            model=self.model,
            usage={
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            },
        )

    def unload(self) -> None:
        """Free the model's memory now instead of after Ollama's 5-minute idle timeout."""
        self._client.generate(model=self.model, prompt="", keep_alive=0)
