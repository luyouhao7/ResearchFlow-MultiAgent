"""Researcher Agent: attempt tool calling and label deterministic fallback."""
from tools.arxiv_search import search_arxiv

def researcher_node(state, model, search_fn=None):
    if search_fn is None:
        search_fn = search_arxiv.invoke
    calls = []
    results = []
    try:
        tool_model = model.bind_tools([search_arxiv])
        response = tool_model.invoke([
            ("system", "[RESEARCHER] You are a literature researcher. Call the "
             "search_arxiv tool once, using the search query provided. "
             "Use max_results=5. Do not invent papers."),
            ("human", f"Research goal: {state['goal']}\nSearch query: {state['query']}"),
        ])
        calls = list(getattr(response, "tool_calls", []) or [])
    except Exception as exc:
        print(f"[Researcher] Tool selection unavailable: {type(exc).__name__}")
    for call in calls[:2]:
        if call.get("name") != "search_arxiv":
            continue
        args = call.get("args", {})
        if not isinstance(args, dict):
            continue
        query = str(args.get("query", state["query"]))[:180]
        try:
            count = int(args.get("max_results", 5))
        except (TypeError, ValueError):
            count = 5
        count = min(max(count, 1), 5)
        papers = search_fn({"query": query, "max_results": count})
        results.extend(papers)
        print(f"[Researcher] LLM selected arXiv query: {query} ({len(papers)} papers)")
        call["_trace"] = {
            "mode": "llm_tool_call", "tool": "search_arxiv",
            "query": query, "count": len(papers)
        }
    trace = [call["_trace"] for call in calls[:2] if "_trace" in call]
    if not trace:
        results = search_fn({"query": state["query"], "max_results": 5})
        trace = [{
            "mode": "fallback", "tool": "search_arxiv",
            "query": state["query"], "count": len(results)
        }]
        print(f"[Researcher] Fallback search ({len(results)} papers)")
    unique = {}
    for paper in results:
        if paper.get("url"):
            unique.setdefault(paper["url"], paper)
    papers = list(unique.values())[:5]
    print(f"[Researcher] Evidence: {len(papers)} distinct papers")
    return {"papers": papers, "tool_trace": trace}
