"""Local-only FastAPI endpoint; use port 8001, separate from project one."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from workflow import build_workflow, final_report

app = FastAPI(title="ResearchFlow API", version="0.1.0")

class ResearchRequest(BaseModel):
    goal: str = Field(min_length=5, max_length=500)

@app.get("/health")
def health():
    return {"status": "ok", "project": "ResearchFlow"}

@app.post("/research")
def research(request: ResearchRequest):
    try:
        result = build_workflow().invoke({"goal": request.goal, "revisions": 0})
        return {
            "report": final_report(result),
            "papers": result.get("papers", []),
            "tool_trace": result.get("tool_trace", []),
            "review": result.get("review", ""),
            "revisions": result.get("revisions", 0),
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Research failed; inspect server logs.") from exc
