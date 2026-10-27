# ResearchFlow — 多智能体学术研究工作流

基于 **Python + LangGraph + LLM Tool Calling** 的多智能体研究项目。用户输入一个研究主题，系统依次完成检索规划、论文搜索、报告撰写和证据审查，并根据审查反馈进行**最多一次**报告修订。

这是一个用于学习和展示 Agent 工程的 **MVP（最小可行版本）**，不是已上线的生产系统，也不是自动验证数学证明的工具。

## 一、项目定位与当前进度

与项目一 [MathResearch Agent](https://github.com/luyouhao7/MathResearch-Agent) 的区别：

| 对比维度 | MathResearch Agent（项目一） | ResearchFlow（项目二） |
| --- | --- | --- |
| 用户输入 | 针对已上传论文的问题 | 一个需要调研的研究主题 |
| 主要功能 | PDF 解析、FAISS 检索、原文读取、问答 | 研究规划、联网检索、多角色协作、报告生成 |
| 工作流 | 固定 Search → Read → Answer | Planner → Researcher → Writer → Critic → 条件修订 |
| 技术重点 | RAG、向量检索、来源页码 | Tool Calling、State、条件路由、执行反馈 |

**实现与验证状态（2026-10-27）：**

- ✅ 已编写 Planner、Researcher、Writer、Critic 四个角色节点，以及 LangGraph 工作流、命令行入口、本地 Web 界面和测试代码。
- ✅ 已在本机运行确认 Planner 能生成检索词，Researcher 能完成模型工具选择，并通过 arXiv 获取 5 条真实论文记录。
- ✅ 针对 arXiv HTTP 429 添加了 OpenAlex 备用检索；针对模型返回空文本添加了输出预算调整、一次受限重试和诊断日志。
- ⏳ **Writer 曾因模型返回空文本而失败。最新修复已提交，但尚待本机重新验证。** Critic、完整报告链路和在线 Web 演示也尚未全部通过真实联调。
- ❌ 尚未完成公网部署；没有经过独立事实验证或量化的幻觉降低实验。

> 注意：简历应如实标注“开发中”。不要把代码已实现等同于功能已验收。

## 二、系统架构

```text
                  用户输入研究目标
                         │
                         ▼
                  Planner Agent
                  生成英文检索词
                         │
                         ▼
                 Researcher Agent
               LLM 尝试选择检索工具
                         │
                         ▼
              search_arxiv（Python 执行）
                 │                │
               arXiv         HTTP 429 时
                 │                ▼
                 │            OpenAlex 备用
                 └────────┬───────┘
                          ▼
                   论文标题与摘要
                          │
                          ▼
                     Writer Agent
                   生成 Markdown 草稿
                          │
                          ▼
                     Critic Agent
                    检查证据和引用
                       │      │
                     PASS    REVISE
                       │      │
                       │      └──→ Writer（最多重写一次）
                       ▼
                    最终研究报告
```

### 四个 Agent 各自负责什么？

- **Planner Agent**：将自然语言研究目标转成适合检索的英文关键词。
- **Researcher Agent**：模型通过 `bind_tools()` 获得工具描述，可能产生 `tool_calls`；随后由 **Python** 真正执行检索，并保存结果及执行记录。
- **Writer Agent**：只根据实际检索到的论文元数据与摘要编写中文 Markdown 报告。
- **Critic Agent**：检查草稿中是否存在缺乏摘要证据的结论、引用错配和过度推断，给出 `PASS` 或 `REVISE`。

**两种回退机制不能混淆：**

1. 当 LLM 没有给出有效的工具调用时，Researcher 使用**确定性的 Python 检索回退**；`tool_trace` 中记录 `mode: "fallback"`，不能把它说成 LLM 自主选择了工具。
2. 当 arXiv 返回 **HTTP 429** 时，检索函数尝试 **OpenAlex**；这属于外部 API 的故障恢复，并非由 LLM 自主选择数据库。

当前只读取**论文标题、摘要和元数据**，不会自动下载并通读全文。Critic 的审查是模型判断，不等于事实核查；报告中的观点仍需人工检查。若两个外部来源都失败，系统会明确报错，**不会伪造文献**。

## 三、技术栈

| 类型 | 使用技术 |
| --- | --- |
| 开发语言 | Python |
| 模型与接口 | OpenRouter、`langchain-openai` |
| Agent 编排 | LangGraph、LangChain Tool Calling |
| 论文检索 | arXiv API；OpenAlex 作为限流回退来源 |
| 后端 | FastAPI、Uvicorn |
| 前端 | Streamlit |
| 测试 | `unittest`、模拟模型和模拟检索结果 |

**本项目没有使用 FAISS 构建向量知识库**；那是项目一的技术重点。本项目也暂未实现长期 Memory、多模型协商或真正的并行 Agent 调度。

## 四、本地安装与运行（macOS）

### 1. 下载代码

首次下载：

```bash
cd ~
git clone https://github.com/luyouhao7/ResearchFlow-MultiAgent.git
cd ResearchFlow-MultiAgent
```

如果已经克隆过，只需进入仓库并更新：

```bash
cd ~/ResearchFlow-MultiAgent
git pull origin main
```

### 2. 创建和激活虚拟环境

首次安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

以后重新打开终端只需要：

```bash
cd ~/ResearchFlow-MultiAgent
source .venv/bin/activate
```

### 3. 配置 API Key

```bash
cp .env.example .env
```

打开本地 `.env`，配置：

```dotenv
OPENROUTER_API_KEY=这里填写你自己的密钥
OPENROUTER_MODEL=nvidia/nemotron-3.5-lightning:free
```

也可以在确认两个项目路径正确后，将项目一已配置的 `.env` 复制到项目二。模型名称可以修改为 OpenRouter 上当前可用、支持所需调用方式的模型。

如需要，也可以配置备用文献来源：

```dotenv
OPENALEX_API_KEY=可选的OpenAlex密钥
```

**不要提交、截图分享或公开你的真实密钥。** 本仓库的 `.gitignore` 已排除 `.env` 和 `.venv/`。

### 4. 先运行离线演示与自动化测试

```bash
python main.py --demo
python -m unittest discover -s tests -v
```

离线演示使用**人工构造的测试论文**，且带有 `[DEMO]` 标记。它不调用真实 OpenRouter 或 arXiv，只用于检查节点顺序、State 传递和报告生成流程。离线成功不代表真实外部服务已经通过测试。

### 5. 运行真实研究任务

```bash
python main.py --goal "Graph neural networks for drug discovery"
```

预期执行顺序：

```text
[Planner] ...
[Researcher] ...
[Writer] ...
[Critic] ...
Report: outputs/report_YYYYMMDD_HHMMSS.md
```

程序成功结束后，Markdown 报告保存在 `outputs/` 目录。报告尾部的文献链接由检索到的真实元数据生成；**链接真实不保证报告中每句话都准确**。

## 五、启动本地网页

在两个独立终端中运行以下命令（均需在项目根目录）：

**终端 1：FastAPI 后端**

```bash
source .venv/bin/activate
python -m uvicorn api:app --host 127.0.0.1 --port 8001
```

**终端 2：Streamlit 前端**

```bash
source .venv/bin/activate
python -m streamlit run app.py --server.port 8502
```

浏览器访问：

- **前端页面**：http://localhost:8502
- **后端健康检查**：http://127.0.0.1:8001/health
- **FastAPI 接口文档**：http://127.0.0.1:8001/docs

之所以使用端口 **8001** 和 **8502**，是为了避免与项目一使用的 **8000** 和 **8501** 冲突。

**安全提醒：** 当前 Web 应用没有登录鉴权、任务队列、用户隔离、完整的调用预算管理或生产级安全加固，不应直接暴露在公网。

## 六、代码目录与复习重点

```text
ResearchFlow-MultiAgent/
├── main.py                    # CLI 入口、报告保存
├── model.py                   # OpenRouter 模型配置、空输出诊断与重试
├── state.py                   # Agent 共享状态 ResearchState
├── workflow.py                # LangGraph 构图和条件路由
├── agents/
│   ├── planner.py             # 研究规划 Agent
│   ├── researcher.py          # Tool Calling、论文检索和轨迹记录
│   ├── writer.py              # 草稿与修订
│   └── critic.py              # PASS / REVISE 审查
├── tools/
│   └── arxiv_search.py        # arXiv 检索、HTTP 429 时 OpenAlex 回退
├── api.py                     # FastAPI 后端
├── app.py                     # Streamlit 页面
├── demo_mode.py               # 离线模拟
├── tests/                     # 离线单元测试
├── outputs/                   # 生成的 Markdown 报告（被 Git 忽略）
├── requirements.txt
├── .env.example
└── README.md
```

**时间有限时，优先看四处：**

1. `state.py`：理解 `ResearchState`。例如 `{"papers": papers}` 就是在更新共享状态中的 `papers` 字段。
2. `workflow.py`：理解 `add_node()`、`add_edge()`、`add_conditional_edges()`、`compile()` 和 `invoke()`。
3. `agents/researcher.py`：区别 **LLM 生成工具调用请求** 与 **Python 实际执行工具**，查看 `tool_trace` 如何记录结果。
4. `agents/critic.py`：理解 `PASS / REVISE` 如何控制 Writer 是否再执行一次。

### 五个必须掌握的概念

| 概念 | 在本项目中的具体含义 |
| --- | --- |
| State | 保存 `goal`、`query`、`papers`、`draft`、`review` 等字段的共享状态 |
| Node | 每个节点读取当前 State，并返回需要更新的字段 |
| Edge | 定义节点的常规执行顺序 |
| Conditional Edge | 根据 Critic 审查结果进入 Writer 或结束 |
| Tool Calling | 模型提出调用什么工具及参数，Python 执行，并把外部结果交回工作流 |

## 七、常见错误及处理

| 错误或现象 | 可能原因 | 处理方式 |
| --- | --- | --- |
| `requirements.txt` 找不到 | 终端不在 GitHub 项目根目录，或只创建了空文件夹 | 执行 `pwd`、`ls`，确认仓库已下载 |
| `Missing OPENROUTER_API_KEY` | 缺少 `.env` 或变量名写错 | 检查项目根目录的 `.env` |
| `ModuleNotFoundError` | 依赖未安装或虚拟环境未激活 | 激活 `.venv` 后安装依赖 |
| arXiv `HTTP 429` | arXiv API 限流 | 程序会尝试 OpenAlex，避免频繁手动重试 |
| 两个检索来源均失败 | 网络/API 限制 | 查看报错，稍后重试或配置备用来源密钥 |
| `Writer generated empty text` | 模型没有返回可见回答；可能与输出预算有关 | 拉取最新代码；查看 `[Model]` 的 `finish_reason`，必要时更换模型 |
| Python 3.14 依赖兼容性问题 | 某些依赖尚未适配 | 尝试 Python 3.12 或 3.13 虚拟环境 |
| Streamlit 无法连接后端 | FastAPI 未启动或端口错误 | 检查 8001 端口与 `/health` |

模型空输出处理使用了较大的输出上限，并且**最多额外重试一次**，不会无限调用。它提高了排障能力，但不保证任何免费模型都会正常输出。

## 八、下一阶段待完成

- 真实运行验证 Writer、Critic、最终报告文件。
- 加入多个搜索工具及 Critic 触发的**补充检索**，而不只是重写报告。
- 对引用进行结构化校验，区分检索到的证据与生成的推断。
- 建立评估集，记录任务成功率、引用正确率和成本。
- 增加运行轨迹、持久化、任务预算与失败恢复机制。
- 如需公开部署，再补身份认证、隔离、限流和服务器安全配置。

---

**项目用途：** AI Agent / LLM 应用开发学习、工程作品展示与面试复习。

**当前事实边界：** 已有可阅读和运行的 MVP 代码，但真实完整链路仍在调试；本项目不能替代人工学术审查。
