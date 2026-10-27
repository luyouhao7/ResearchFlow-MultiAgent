# ResearchFlow · Multi-Agent Research System

An educational LangGraph multi-agent research MVP: a Planner chooses arXiv search terms, a Researcher attempts model tool-calling and retrieves real paper **titles/abstracts** (arXiv first, OpenAlex fallback on arXiv HTTP 429), a Writer prepares a Markdown report, and a Critic can request **one** revision.

> **Project status:** Source code implemented. Offline demo/tests are provided, but real OpenRouter + arXiv execution has not yet been verified on the user's Mac. This is a local development app, not a public deployment. No measured hallucination-reduction claim.

## Architecture

```text
Research goal
  → Planner Agent (LLM) → query
  → Researcher Agent (LLM tool-call attempt; labeled fallback) → arXiv abstracts / OpenAlex abstracts if arXiv returns 429
  → Writer Agent → draft
  → Critic Agent → PASS or REVISE
                     REVISE → Writer (at most one revision) → Critic → END
```

The Researcher checks actual `tool_calls` after `bind_tools`. Python executes the allowlisted arXiv tool. When the model/provider does not return a suitable tool call, a deterministic fallback search runs and is recorded as `mode: "fallback"` in `tool_trace`. A fallback is **not** evidence that the LLM independently called a tool.

Critic review is LLM-generated feedback against abstracts, **not** independent fact verification. This version does not read full PDFs. The OpenAlex fallback is deterministic error recovery, not an autonomous choice by the LLM.

## Quick start — macOS

```bash
cd /Users/luyouhao
git clone https://github.com/luyouhao7/ResearchFlow-MultiAgent.git
cd ResearchFlow-MultiAgent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy the working local key from project one (never commit it):

```bash
cp /Users/luyouhao/my_langchain_project/.env .env
```

Alternatively: `cp .env.example .env`, then fill `OPENROUTER_API_KEY` locally.

**Offline demonstration**, no API credentials or arXiv network calls:

```bash
python main.py --demo
python -m unittest discover -s tests -v
```

Demo uses *fabricated test records* explicitly labeled `[DEMO]`; these are not real academic papers.

**Live research**, requires external APIs:

```bash
python main.py --goal "Graph neural networks for drug discovery"
```

Markdown output is saved under `outputs/report_YYYYMMDD_HHMMSS.md`. Source links come from actual arXiv or OpenAlex metadata provided to the workflow; claims within the generated narrative may still be wrong or cite inappropriate papers.

## Optional local web app

Run each service in its own terminal:

```bash
# Terminal 1
source .venv/bin/activate
python -m uvicorn api:app --host 127.0.0.1 --port 8001
```

```bash
# Terminal 2
source .venv/bin/activate
python -m streamlit run app.py --server.port 8502
```

Open http://localhost:8502 ; backend check: http://127.0.0.1:8001/health .

The web app has **no authentication, queue, per-user isolation or production hardening**. Do not expose it publicly.

## Structure and learning priorities

| File | Purpose |
|---|---|
| `state.py` | Shared `ResearchState`, e.g. `papers`, `draft`, `review` |
| `workflow.py` | LangGraph nodes, edges, conditional rewrite |
| `agents/planner.py` | Query planning |
| `agents/researcher.py` | `bind_tools`, `tool_calls`, tool execution and trace |
| `tools/arxiv_search.py` | arxiv.Client metadata/abstract retrieval; OpenAlex on HTTP 429 |
| `agents/writer.py` | Report draft and revision |
| `agents/critic.py` | PASS / REVISE feedback |
| `model.py` | OpenRouter model configuration |
| `main.py` | CLI and Markdown export |
| `api.py`, `app.py` | Local FastAPI and Streamlit |

### Five key concepts

1. **State:** each node returns partial updates such as `{"papers": papers}`, which LangGraph merges into shared state.
2. **Execution:** `add_edge` defines the default sequence. `add_conditional_edges` routes from Critic to Writer or END.
3. **Tool calling:** an LLM **proposes** a call; Python **executes** it.
4. **Guardrails:** at most 5 distinct papers and one Writer revision; these are execution budgets, not proof of factual correctness.
5. **Testing:** offline fake model/search tests verify orchestration, not live search quality.

## Future improvements

- More search tools and an actual corrective **re-search** loop after Critic feedback
- Structured extraction, robust citation validation, task budgets and persistence
- Evaluation dataset and measurements
- Container/deployment security, authentication and background jobs

## Troubleshooting

- `Missing OPENROUTER_API_KEY`: check `.env` in the project root.
- `ModuleNotFoundError`: activate `.venv` and install `requirements.txt`.
- arXiv returns `HTTP 429`: the tool tries OpenAlex automatically. Optional: create a free OpenAlex API key and add `OPENALEX_API_KEY=...` to your local `.env` for more reliable fallback. Keyless casual calls may work but can be limited. Do not retry rapidly.\n- If **both** providers fail, the program raises a clear error; it will never silently invent paper records.\n- Run `python -m unittest discover -s tests -v` to test the fallback without network calls.
- Python 3.14 package incompatibility: use Python 3.12 or 3.13 for the venv.
- Streamlit cannot connect: start the FastAPI service on port 8001.

**Privacy:** `.env` is excluded by `.gitignore`. Never post credentials, private papers, or confidential data to hosted free-model endpoints.
