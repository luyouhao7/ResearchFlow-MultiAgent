"""Real external tool: retrieve public arXiv metadata and abstracts."""
import arxiv
from langchain_core.tools import tool

@tool
def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """Search arXiv for academic papers. Returns title, abstract, year and source URL."""
    query = query.strip()[:180]
    if not query:
        return []
    count = max(1, min(int(max_results), 8))
    client = arxiv.Client(page_size=count, delay_seconds=3.0, num_retries=1)
    search = arxiv.Search(
        query=query,
        max_results=count,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers = []
    for item in client.results(search):
        papers.append({
            "title": item.title.strip(),
            "year": item.published.year,
            "authors": ", ".join(a.name for a in item.authors[:3]),
            "abstract": " ".join(item.summary.split())[:1100],
            "url": item.entry_id,
        })
    return papers
