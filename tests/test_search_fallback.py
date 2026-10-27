"""Offline unit test for arXiv 429 -> OpenAlex fallback; no external calls."""
import unittest
from unittest.mock import patch
import arxiv
from tools.arxiv_search import search_arxiv


class RejectingClient:
    def __init__(self, **_kwargs):
        pass

    def results(self, _search):
        raise arxiv.HTTPError("https://export.arxiv.org/api/query", 1, 429)


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"results": [{
            "title": "Synthetic OpenAlex test record",
            "id": "https://openalex.org/W100",
            "publication_year": 2025,
            "authorships": [{"author": {"display_name": "Example Author"}}],
            "abstract_inverted_index": {
                "graph": [0], "neural": [1], "networks": [2],
            },
        }]}


class TestSearchFallback(unittest.TestCase):
    @patch("tools.arxiv_search.requests.get", return_value=FakeResponse())
    @patch("tools.arxiv_search.arxiv.Client", RejectingClient)
    def test_fallback_keeps_provenance(self, mock_get):
        results = search_arxiv.invoke({"query": "graph neural networks", "max_results": 1})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["provider"], "openalex")
        self.assertEqual(results[0]["abstract"], "graph neural networks")
        self.assertTrue(mock_get.called)


if __name__ == "__main__":
    unittest.main()
