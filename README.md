# ResearchFlow

基于 LangGraph 的多智能体学术研究工作流系统。系统围绕研究目标组织任务规划、文献检索、报告撰写与内容审查，通过共享状态和条件路由协调各模块执行。

## 系统架构

```text
Research Goal
     |
     v
Planner Agent
     |  Search Query
     v
Researcher Agent
     |  Tool Calling
     v
Literature Search
     |-- arXiv API
     |-- OpenAlex (arXiv HTTP 429 fallback)
     |
     v
Paper Metadata & Abstracts
     |
     v
Writer Agent
     |  Report Draft
     v
Critic Agent
     |-- PASS ----------------------> Final Report
     |
     |-- REVISE --> Writer Agent --> Critic Agent --> Final Report
                    (maximum one revision)
```

工作流由 LangGraph `StateGraph` 编排，节点通过 `ResearchState` 传递检索关键词、文献记录、报告草稿及审查结果。Critic 的判断由条件边控制后续执行路径。

## 核心模块

| 模块 | 职责 |
| --- | --- |
| Planner | 将研究目标转换为学术检索查询 |
| Researcher | 基于 LLM Tool Calling 发起检索，执行工具并记录调用轨迹 |
| Literature Search | 从 arXiv 获取论文元数据与摘要；遇到 HTTP 429 时尝试 OpenAlex |
| Writer | 基于检索证据生成、修订 Markdown 研究报告 |
| Critic | 检查草稿的引用对应关系、证据支持程度及潜在过度推断 |
| Workflow | 管理节点执行顺序、状态更新和有限次数的审查修订 |
| API / UI | 通过 FastAPI 提供研究接口，使用 Streamlit 展示报告 |

Researcher 在模型未返回有效工具调用时使用确定性检索回退，并在执行轨迹中区分模型工具调用与回退执行。文献检索目前以标题、摘要和元数据为依据；Critic 不构成独立事实核验。

## 项目结构

```text
ResearchFlow-MultiAgent/
├── agents/
│   ├── planner.py                 # 检索规划
│   ├── researcher.py              # 工具选择、执行与轨迹记录
│   ├── writer.py                  # 报告生成与修订
│   └── critic.py                  # 报告审查
├── tools/
│   └── arxiv_search.py            # arXiv 检索与 OpenAlex 回退
├── tests/
│   ├── test_workflow.py           # 工作流与条件路由测试
│   ├── test_search_fallback.py    # 检索回退测试
│   └── test_empty_model_response.py # 模型空输出处理测试
├── outputs/                       # Markdown 报告输出目录
├── main.py                        # 命令行入口
├── model.py                       # LLM 配置与响应处理
├── state.py                       # 共享状态定义
├── workflow.py                    # LangGraph 编排
├── api.py                         # FastAPI 接口
├── app.py                         # Streamlit 前端
├── demo_mode.py                   # 离线模拟运行
├── requirements.txt               # Python 依赖
└── .env.example                   # 环境变量模板
```

## 技术组成

| 层级 | 技术 |
| --- | --- |
| 语言与框架 | Python、LangGraph、LangChain |
| 模型访问 | OpenRouter、langchain-openai |
| 外部数据 | arXiv API、OpenAlex API |
| 应用服务 | FastAPI、Uvicorn、Streamlit |
| 测试 | unittest |

项目采用顺序多角色协作及受限审查循环，未实现并行 Agent 调度或持久化长期记忆。
