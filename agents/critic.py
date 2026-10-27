"""Critic Agent: bounded evidence-based feedback; not independent fact verification."""
import json
from model import ask_model

def critic_node(state, model):
    if not state.get("papers"):
        return {"review": "PASS\nNo retrieved evidence; report correctly says so."}
    feedback = ask_model(
        model,
        "[CRITIC] You review an academic report only against supplied abstracts. "
        "Check fabricated claims, wrong citation numbers and overgeneralization. "
        "Your FIRST LINE must be exactly PASS or REVISE. "
        "If REVISE, list concrete issues briefly. You have NOT independently "
        "verified the papers or their full text.",
        f"Evidence: {json.dumps(state['papers'], ensure_ascii=False)}\n"
        f"Draft: {state['draft']}",
    )
    verdict = feedback.strip().splitlines()[0].strip().upper() if feedback else ""
    if verdict not in {"PASS", "REVISE"}:
        feedback = "REVISE\nCritic did not return a recognized verdict. Review claims and citation numbers."
    print("[Critic]", feedback.splitlines()[0])
    return {"review": feedback}
