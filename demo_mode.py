"""Offline demo uses mock LLM and mock search. Papers are synthetic."""
from dataclasses import dataclass, field

@dataclass
class FakeMessage:
    content: str = ""
    tool_calls: list = field(default_factory=list)

class DemoModel:
    def __init__(self):
        self.drafts = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        system = messages[0][1]
        if "[PLANNER]" in system:
            return FakeMessage(content="graph neural networks drug discovery")
        if "[RESEARCHER]" in system:
            return FakeMessage(tool_calls=[{
                "name": "search_arxiv",
                "args": {"query": "graph neural networks drug discovery", "max_results": 2},
                "id": "demo-call-1",
            }])
        if "[WRITER]" in system:
            self.drafts += 1
            return FakeMessage(content=(
                "# 离线演示报告（非真实论文调查）\n\n"
                "用于测试 Planner→Researcher→Writer→Critic 的状态传递。"
                "样例文献[1][2]均为演示占位数据，不支持真实学术结论。"
            ))
        if "[CRITIC]" in system:
            return FakeMessage(content="PASS\nDemo is clearly marked as synthetic.")
        raise AssertionError("Unknown fake role")

def demo_search(_args):
    return [
        {"title": "[DEMO] Sample paper A — not a real paper", "year": 2026,
         "authors": "Demo", "abstract": "Synthetic test data only.",
         "url": "https://example.org/demo-a"},
        {"title": "[DEMO] Sample paper B — not a real paper", "year": 2026,
         "authors": "Demo", "abstract": "Synthetic test data only.",
         "url": "https://example.org/demo-b"},
    ]
