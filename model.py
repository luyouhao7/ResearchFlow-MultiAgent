"""One model factory for all research agents."""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).resolve().parent / ".env")

def get_model():
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key or key == "YOUR_OPENROUTER_KEY":
        raise RuntimeError("Missing OPENROUTER_API_KEY. Copy .env.example to .env and fill it in.")
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"),
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
        temperature=0,
        max_tokens=1600,
        timeout=90,
        max_retries=1,
    )

def ask_model(model, system_prompt: str, task: str) -> str:
    """Return text from a chat-model response (including text content blocks)."""
    result = model.invoke([("system", system_prompt), ("human", task)])
    content = result.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    return str(content).strip()
