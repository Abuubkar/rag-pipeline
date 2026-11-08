"""Ask the LLM to answer from the numbered sources."""

from typing import Callable

from rag.domain.models import Completion
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


def closed_book(*, chat: OllamaChat, prompt: Prompt, question: str) -> Completion:
    """Answer with no sources at all: the no-retrieval baseline used in evaluation."""
    return chat.complete(system=prompt.system, user=prompt.render_user(question=question),
                         run_name=f"llm.{prompt.label}")
