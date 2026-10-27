"""Command-line entry point: python main.py --goal '...'"""
import argparse
from datetime import datetime
from pathlib import Path
from workflow import build_workflow, final_report

def run():
    parser = argparse.ArgumentParser(description="ResearchFlow multi-agent CLI")
    parser.add_argument("--goal", default="Graph neural networks for drug discovery")
    parser.add_argument("--demo", action="store_true", help="Offline fake-data test; does not call OpenRouter or arXiv")
    args = parser.parse_args()
    if args.demo:
        from demo_mode import DemoModel, demo_search
        workflow = build_workflow(model=DemoModel(), search_fn=demo_search)
        print("[DEMO MODE] Using synthetic paper data and a fake LLM")
    else:
        workflow = build_workflow()
    result = workflow.invoke({"goal": args.goal, "revisions": 0})
    text = final_report(result)
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    filename = out_dir / f"report_{datetime.now():%Y%m%d_%H%M%S}.md"
    filename.write_text(text, encoding="utf-8")
    print("\n==== FINAL REPORT ====\n")
    print(text)
    print("\nTool execution trace:", result.get("tool_trace", []))
    print("Report:", filename)

if __name__ == "__main__":
    run()
