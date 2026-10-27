"""Graph orchestration: planner -> researcher -> writer -> critic -> optional rewrite."""
from langgraph.graph import END, START, StateGraph
from agents.critic import critic_node
from agents.planner import planner_node
from agents.researcher import researcher_node
from agents.writer import writer_node
from model import get_model
from state import ResearchState

def route_after_review(state: ResearchState) -> str:
    verdict = state.get("review", "").strip().splitlines()[0] if state.get("review") else ""
    if verdict == "REVISE" and state.get("revisions", 0) < 1:
        return "rewrite"
    return "finish"

def build_workflow(model=None, search_fn=None):
    """Dependency injection enables offline tests without real API calls."""
    if model is None:
        model = get_model()
    graph = StateGraph(ResearchState)
    graph.add_node("planner", lambda s: planner_node(s, model))
    graph.add_node("researcher", lambda s: researcher_node(s, model, search_fn))
    graph.add_node("writer", lambda s: writer_node(s, model))
    graph.add_node("critic", lambda s: critic_node(s, model))
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "critic")
    graph.add_conditional_edges(
        "critic", route_after_review,
        {"rewrite": "writer", "finish": END},
    )
    return graph.compile()

def final_report(state: ResearchState) -> str:
    """Reference URLs come from tool data, not generated LLM prose."""
    content = state.get("draft", "")
    papers = state.get("papers", [])
    if not papers:
        return content
    sources = ["\n\n## 来源列表（检索元数据）"]
    for i, paper in enumerate(papers, 1):
        sources.append(f"[{i}] {paper['title']} ({paper['year']}) — {paper['url']}")
    sources.append("\n> 注意：报告基于论文摘要，未经全文或事实独立验证。")
    return content.rstrip() + "\n" + "\n".join(sources)
