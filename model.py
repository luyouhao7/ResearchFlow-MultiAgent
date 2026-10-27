"""One model factory for all research agents, with visible-output diagnostics."""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).resolve().parent / ".env")

DEFAULT_MAX_TOKENS = 8192
RETRY_MAX_TOKENS = 12288


def get_model():
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key or key == "YOUR_OPENROUTER_KEY":
        raise RuntimeError("Missing OPENROUTER_API_KEY. Copy .env.example to .env and fill it in.")
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"),
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
        temperature=0,
        max_tokens=DEFAULT_MAX_TOKENS,
        timeout=180,
        max_retries=1,
    )


def _visible_text(content) -> str:
    """Extract only the answer text, never internal model reasoning."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"text", "output_text"}:
                value = part.get("text", "")
                if isinstance(value, str):
                    texts.append(value)
        return "\n".join(texts).strip()
    return ""


def _log_empty_response(response, attempt: int) -> None:
    """Log non-sensitive diagnostics, not full prompt or private reasoning."""
    metadata = getattr(response, "response_metadata", {}) or {}
    usage = getattr(response, "usage_metadata", {}) or {}
    print(
        f"[Model] Empty visible response (attempt {attempt}): "
        f"finish_reason={metadata.get('finish_reason', 'unknown')}, "
        f"output_tokens={usage.get('output_tokens', 'unknown')}"
    )


def ask_model(model, system_prompt: str, task: str) -> str:
    """Call the model; retry once if it returns no visible answer text.

    Reasoning-enabled providers can exhaust max_tokens without generating
    visible content. Never treat the provider's private reasoning as an answer.
    """
    messages = [("system", system_prompt), ("human", task)]
    response = model.invoke(messages)
    text = _visible_text(response.content)
    if text:
        return text

    _log_empty_response(response, 1)

    # Give the model enough room for a final answer after reasoning.
    # This is a single retry, never an unlimited loop.
    retry_model = model.bind(max_tokens=RETRY_MAX_TOKENS) if hasattr(model, "bind") else model
    response = retry_model.invoke(messages)
    text = _visible_text(response.content)
    if text:
        return text

    _log_empty_response(response, 2)
    finish_reason = (getattr(response, "response_metadata", {}) or {}).get(
        "finish_reason", "unknown"
    )
    raise RuntimeError(
        "The model returned no visible answer twice "
        f"(last finish_reason={finish_reason}). "
        "Try changing OPENROUTER_MODEL in your local .env to an available "
        "model suitable for long-form Chinese writing."
    )
