"""Planner Agent: turn a user goal into a short search query."""
from model import ask_model

def planner_node(state, model):
    text = ask_model(
        model,
        "[PLANNER] You are a research planner. Output ONE short English arXiv search "
        "query only: no quotation marks, no explanations, no markdown.",
        state["goal"],
    )
    query = text.strip().strip("`\"' ").splitlines()[0][:120] if text else ""
    if not query:
        raise RuntimeError("Planner returned an empty search query; try a clearer goal or another model.")
    print(f"[Planner] {query}")
    return {"query": query}
