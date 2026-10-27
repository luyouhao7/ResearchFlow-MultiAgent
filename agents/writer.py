"""Writer Agent: draft or revise a paper-grounded report."""
import json
from model import ask_model

def writer_node(state, model):
    revisions = state.get("revisions", 0)
    if state.get("review", "").lstrip().startswith("REVISE"):
        revisions += 1
    if not state.get("papers"):
        return {"draft": "未检索到相关论文，无法生成有依据的调研报告。", "revisions": revisions}
    papers_json = json.dumps(state["papers"], ensure_ascii=False, indent=2)
    feedback = state.get("review", "无（首次撰写）")
    previous = state.get("draft", "") if revisions else ""
    draft = ask_model(
        model,
        "[WRITER] You write concise Chinese academic reports. Use ONLY the "
        "paper titles, abstracts and metadata provided. Number papers in the "
        "given order [1], [2], etc. Every factual claim needs an evidence "
        "reference when possible. Organize as 研究概述、代表论文、方法比较、局限性. "
        "Do not claim to have read full PDFs or verified mathematical proofs. "
        "Do NOT invent references or write a bibliography; it will be added "
        "programmatically from real arXiv results.",
        f"Goal: {state['goal']}\nEvidence: {papers_json}\n"
        f"Previous draft: {previous}\nCritic feedback: {feedback}\n"
        "Write the report in Markdown.",
    )
    if not draft:
        raise RuntimeError("Writer generated empty text; try a different model.")
    print(f"[Writer] Draft created (revision {revisions})")
    return {"draft": draft, "revisions": revisions}
