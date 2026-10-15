"""Building LangGraph graphs from plain (name, function) pairs.

Capability code stays free of LangGraph: it only writes functions `state -> partial update`.
This module turns them into a compiled StateGraph.
"""

from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

Node = tuple[str, Callable[[Any], dict[str, Any]]]
Edge = tuple[str, str]

BEGIN = "__begin__"
FINISH = "__finish__"


def build_graph(state_type: type, nodes: list[Node], edges: list[Edge]) -> Any:
    """`edges` use node names plus BEGIN / FINISH for the entry and exit points."""
    graph = StateGraph(state_type)
    for name, fn in nodes:
        graph.add_node(name, fn)
    alias = {BEGIN: START, FINISH: END}
    for src, dst in edges:
        graph.add_edge(alias.get(src, src), alias.get(dst, dst))
    return graph.compile()


def linear_edges(nodes: list[Node]) -> list[Edge]:
    """BEGIN → n1 → n2 → … → FINISH."""
    names = [BEGIN, *(name for name, _ in nodes), FINISH]
    return list(zip(names, names[1:]))


def invoke(graph: Any, state: dict[str, Any], callbacks: list[Any], run_name: str) -> dict[str, Any]:
    return graph.invoke(state, config={"callbacks": callbacks, "run_name": run_name})
