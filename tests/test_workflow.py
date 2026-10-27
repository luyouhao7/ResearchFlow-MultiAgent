"""Fast offline tests. No API key or live arXiv network required."""
import unittest
from demo_mode import DemoModel, FakeMessage, demo_search
from workflow import build_workflow, final_report, route_after_review

class ReviseOnceModel(DemoModel):
    def __init__(self):
        super().__init__()
        self.review_count = 0

    def invoke(self, messages):
        if "[CRITIC]" in messages[0][1]:
            self.review_count += 1
            return FakeMessage(content=(
                "REVISE\nClarify methodology." if self.review_count == 1 else "PASS\nGood."
            ))
        return super().invoke(messages)

class TestWorkflow(unittest.TestCase):
    def test_complete_demo(self):
        result = build_workflow(model=DemoModel(), search_fn=demo_search).invoke({
            "goal": "Test ResearchFlow", "revisions": 0
        })
        self.assertEqual(len(result["papers"]), 2)
        self.assertIn("demo-a", final_report(result))
        self.assertEqual(result["review"].splitlines()[0], "PASS")
        self.assertEqual(result["tool_trace"][0]["mode"], "llm_tool_call")

    def test_writer_critic_one_revision(self):
        fake = ReviseOnceModel()
        result = build_workflow(model=fake, search_fn=demo_search).invoke({
            "goal": "Research loop test", "revisions": 0
        })
        self.assertEqual(result["revisions"], 1)
        self.assertEqual(fake.drafts, 2)
        self.assertEqual(fake.review_count, 2)

    def test_revision_bounded(self):
        self.assertEqual(route_after_review({"review": "REVISE\nfix", "revisions": 0}), "rewrite")
        self.assertEqual(route_after_review({"review": "REVISE\nfix", "revisions": 1}), "finish")
        self.assertEqual(route_after_review({"review": "PASS", "revisions": 0}), "finish")

if __name__ == "__main__":
    unittest.main()
