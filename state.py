"""LangGraph shared state. Nodes return only the keys they update."""
from typing import NotRequired, TypedDict

class ResearchState(TypedDict):
    goal: str
    query: NotRequired[str]
    papers: NotRequired[list[dict]]
    draft: NotRequired[str]
    review: NotRequired[str]
    revisions: NotRequired[int]
    tool_trace: NotRequired[list[dict]]
