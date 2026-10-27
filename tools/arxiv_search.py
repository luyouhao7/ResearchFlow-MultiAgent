"""Search arXiv and fall back to OpenAlex only when arXiv rate-limits.

The fallback uses real external metadata. It never fabricates paper records.
"""
import os

import arxiv
import requests
from langchain_core.tools import tool


def _abstract_from_openalex(inverted_index: dict | None) -> str:
    """OpenAlex encodes abstracts as word -> positions."""
    if not inverted_index:
        return ""
    words = [
        (position, word)
        for word, positions in inverted_index.items()
        for position in positions
    ]
    words.sort(key=lambda item: item[0])
    return " ".join(word for _, word in words)


def _search_openalex(query: str, count: int) -> list[dict]:
    """Get abstract-grounded paper records from an independent provider."""
    params = {"search": query, "per_page": count}
    key = os.getenv("OPENALEX_API_KEY", "").strip()
    if key:
        params["api_key"] = key

    response = requests.get(
        "https://api.openalex.org/works",
        params=params,
        timeout=25,
        headers={"User-Agent": "ResearchFlow-MultiAgent/0.1 (academic research demo)"},
    )
    response.raise_for_status()

    papers = []
    for work in response.json().get("results", []):
        abstract = _abstract_from_openalex(work.get("abstract_inverted_index"))
        title = work.get("title") or work.get("display_name")
        url = work.get("doi") or work.get("id")
        if not (title and url and abstract):
            continue
        authors = work.get("authorships") or []
        papers.append({
            "title": title.strip(),
            "year": work.get("publication_year") or "unknown",
            "authors": ", ".join(
                (a.get("author") or {}).get("display_name", "")
                for a in authors[:3]
            ),
            "abstract": " ".join(abstract.split())[:1100],
            "url": url,
            "provider": "openalex",
        })
        if len(papers) >= count:
            break
    return papers


@tool
def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """Search real academic papers via arXiv; on HTTP 429, try OpenAlex."""
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
    try:
        for item in client.results(search):
            papers.append({
                "title": item.title.strip(),
                "year": item.published.year,
                "authors": ", ".join(a.name for a in item.authors[:3]),
                "abstract": " ".join(item.summary.split())[:1100],
                "url": item.entry_id,
                "provider": "arxiv",
            })
        return papers
    except arxiv.HTTPError as exc:
        if exc.status != 429:
            raise
        print("[Search] arXiv returned HTTP 429. Trying OpenAlex instead...")
        try:
            alternate = _search_openalex(query, count)
        except requests.RequestException as alternate_exc:
            raise RuntimeError(
                "arXiv rate-limited the request (HTTP 429), and OpenAlex "
                "is also unavailable. Do not retry rapidly; try later, "
                "or configure OPENALEX_API_KEY for the fallback provider."
            ) from alternate_exc
        if not alternate:
            raise RuntimeError(
                "arXiv returned HTTP 429, and OpenAlex returned no papers "
                "with usable abstracts. Try a different search query later."
            ) from exc
        print(f"[Search] OpenAlex returned {len(alternate)} real papers.")
        return alternate
