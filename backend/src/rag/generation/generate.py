"""Ask the LLM to answer from the numbered sources."""

from typing import Callable

from rag.domain.state import AnswerState
from rag.generation.prompts import Prompt
from rag.packages.llm.ollama import OllamaChat


def generate_node(*, chat: OllamaChat, prompt: Prompt) -> Callable[[AnswerState], dict]:
    def node(state: AnswerState) -> dict:
        completion = chat.complete(
            system=prompt.system,
            user=prompt.render_user(context=state["context"], question=state["question"]),
            run_name=f"llm.{prompt.label}",
        )
        return {"completion": completion, "prompt": prompt.label}

    return node
