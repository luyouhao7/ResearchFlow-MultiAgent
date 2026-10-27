"""Optional local Streamlit UI."""
import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("RESEARCHFLOW_API_URL", "http://127.0.0.1:8001").rstrip("/")
st.set_page_config(page_title="ResearchFlow", page_icon="📑", layout="wide")
st.title("ResearchFlow · Multi-Agent Research")
st.caption("Planner → Researcher → Writer → Critic；摘要级研究，不等于全文事实核查。")

goal = st.text_area("研究目标", "Graph neural networks for drug discovery", height=100)
if st.button("生成研究报告", type="primary"):
    if not goal.strip():
        st.warning("请输入研究目标")
    else:
        with st.spinner("多个 Agent 正在协作...（请保持后端运行）"):
            try:
                response = requests.post(
                    f"{API_URL}/research", json={"goal": goal.strip()}, timeout=300
                )
                response.raise_for_status()
                data = response.json()
            except requests.RequestException as exc:
                st.error(f"无法连接后端或处理失败：{exc}")
            else:
                st.markdown(data["report"])
                st.download_button("保存 Markdown", data["report"], file_name="research_report.md")
                with st.expander("工具执行记录 / Tool trace"):
                    st.json(data.get("tool_trace", []))
                with st.expander("Critic 审查结果"):
                    st.text(data.get("review", ""))
