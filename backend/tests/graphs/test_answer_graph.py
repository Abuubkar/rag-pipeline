"""The answer graph wires its nodes in order and validates citations.

Uses a scripted chat model so the test checks LangGraph wiring and citation handling,
not the LLM's wording.
"""

from rag.domain.models import Completion, ScoredChunk
from rag.domain.state import AnswerState
from rag.generation import citations, context, generate
from rag.generation.prompts import load_prompt
from rag.packages.orchestration import graph as graphs


class ScriptedChat:
    model = "scripted"

    def __init__(self, reply: str):
        self.reply = reply
        self.seen_user_prompt = ""

    def complete(self, system: str, user: str, run_name: str = "") -> Completion:
        self.seen_user_prompt = user
        return Completion(text=self.reply, model=self.model)


def _chunks() -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk_id=f"c{i}", document_id=f"d{i}", text=f"text {i}", score=0.9,
                    title=f"Title {i}", source="Test")
        for i in (1, 2)
    ]


def _graph(chat: ScriptedChat):
    nodes = [
        ("retrieval.search", lambda state: {"chunks": _chunks()}),
        ("generation.context", context.context_node),
        ("generation.generate", generate.generate_node(chat=chat, prompt=load_prompt("answer"))),
        ("generation.citations", citations.citations_node),
    ]
    return graphs.build_graph(AnswerState, nodes, graphs.linear_edges(nodes))


def test_graph_passes_numbered_sources_to_the_llm():
    chat = ScriptedChat("Answer [1].")
    _graph(chat).invoke({"question": "Q?"})
    assert "[1] Title 1" in chat.seen_user_prompt and "[2] Title 2" in chat.seen_user_prompt
    assert "Question: Q?" in chat.seen_user_prompt


def test_citations_are_resolved_and_invalid_ones_flagged():
    state = _graph(ScriptedChat("Yes [2], and also [7].")).invoke({"question": "Q?"})
    assert [c.document_id for c in state["citations"]] == ["d2"]
    assert state["invalid_citations"] == [7]
    assert state["prompt"] == "answer@1"


def test_grouped_citation_markers_are_understood():
    state = _graph(ScriptedChat("Both grew [1, 2].")).invoke({"question": "Q?"})
    assert [c.number for c in state["citations"]] == [1, 2]
