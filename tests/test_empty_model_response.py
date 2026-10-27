"""Unit tests for handling a reasoning-model empty visible response."""
import unittest
from model import _visible_text, ask_model


class Message:
    def __init__(self, content, finish_reason="length", output_tokens=1600):
        self.content = content
        self.response_metadata = {"finish_reason": finish_reason}
        self.usage_metadata = {"output_tokens": output_tokens}


class EmptyThenSuccessfulModel:
    def __init__(self):
        self.calls = 0
        self.max_tokens = None

    def bind(self, **kwargs):
        self.max_tokens = kwargs.get("max_tokens")
        return self

    def invoke(self, _messages):
        self.calls += 1
        if self.calls == 1:
            return Message("")
        return Message(
            [{"type": "text", "text": "这是一份测试报告。"}],
            "stop",
            100,
        )


class AlwaysEmptyModel(EmptyThenSuccessfulModel):
    def invoke(self, _messages):
        self.calls += 1
        return Message("")


class TestEmptyOutput(unittest.TestCase):
    def test_retry_once_when_first_response_has_no_text(self):
        model = EmptyThenSuccessfulModel()
        answer = ask_model(model, "system", "task")
        self.assertEqual(answer, "这是一份测试报告。")
        self.assertEqual(model.calls, 2)
        self.assertEqual(model.max_tokens, 12288)

    def test_clear_error_after_second_empty_response(self):
        model = AlwaysEmptyModel()
        with self.assertRaisesRegex(RuntimeError, "no visible answer twice"):
            ask_model(model, "system", "task")
        self.assertEqual(model.calls, 2)

    def test_does_not_expose_reasoning_as_final_answer(self):
        text = _visible_text([{"type": "reasoning", "text": "internal"},
                              {"type": "text", "text": "公开答案"}])
        self.assertEqual(text, "公开答案")


if __name__ == "__main__":
    unittest.main()
