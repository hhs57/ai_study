# LangChain 和 LangGraph 学习教程

这是一套循序渐进的 LangChain 和 LangGraph 学习教程，每个文件只讲解一个核心知识点，包含详细的中文注释。

## 📚 学习路径

### LangChain 基础（01-05）

| 文件 | 主题 | 知识点 |
|------|------|--------|
| [01-basic-chain.py](01-basic-chain.py) | 基础链概念 | LLM 调用、LLMChain、链的基本概念 |
| [02-prompt-template.py](02-prompt-template.py) | 提示词模板 | PromptTemplate、ChatPromptTemplate、输出解析器 |
| [03-chains-sequentials.py](03-chains-sequentials.py) | 链的组合 | SimpleSequentialChain、并行执行、LCEL |
| [04-memory-conversation.py](04-memory-conversation.py) | 对话记忆 | BufferMemory、WindowMemory、SummaryMemory |
| [05-agents-basic.py](05-agents-basic.py) | 基础代理 | Tools、ReAct Agent、多步推理 |

### LangChain 进阶（11-15）

| 文件 | 主题 | 知识点 |
|------|------|--------|
| [11-document-loading.py](11-document-loading.py) | 文档加载与处理 | Document Loaders、Text Splitters、文档分割 |
| [12-vector-storage-rag.py](12-vector-storage-rag.py) | 向量存储与 RAG | Embeddings、Vector Stores、Retrievers、RAG 系统 |
| [13-output-parsers-advanced.py](13-output-parsers-advanced.py) | 高级输出解析器 | PydanticOutputParser、结构化数据提取 |
| [14-callbacks-streaming.py](14-callbacks-streaming.py) | 回调和流式输出 | Callbacks、Token 计数、流式输出 |
| [15-complete-rag-app.py](15-complete-rag-app.py) | 完整的 RAG 应用 | RAG 最佳实践、生产级实现 |

### LangGraph 进阶（06-10）

| 文件 | 主题 | 知识点 |
|------|------|--------|
| [06-langgraph-intro.py](06-langgraph-intro.py) | LangGraph 入门 | 图的基本概念、节点、边、StateGraph |
| [07-langgraph-state.py](07-langgraph-state.py) | 状态管理 | Annotated、operator.add、检查点 |
| [08-langgraph-conditional.py](08-langgraph-conditional.py) | 条件边 | 条件路由、多分支、决策树 |
| [09-langgraph-loops.py](09-langgraph-loops.py) | 循环 | ReAct 循环、迭代优化、防止无限循环 |
| [10-langgraph-agent.py](10-langgraph-agent.py) | 智能体 | 构建 Agent、工具集成、多步推理 |

### LangGraph 高级（16-20）

| 文件 | 主题 | 知识点 |
|------|------|--------|
| [16-human-in-loop.py](16-human-in-loop.py) | 人机交互 | interrupt()、人工批准、多级审批 |
| [17-state-persistence.py](17-state-persistence.py) | 状态持久化 | 数据库持久化、跨会话管理、版本控制 |
| [18-multi-agent.py](18-multi-agent.py) | 多 Agent 系统 | 协作式 Agent、竞争式 Agent、层级式系统 |
| [19-visualization-debug.py](19-visualization-debug.py) | 可视化与调试 | 图可视化、执行追踪、性能分析 |
| [20-error-handling.py](20-error-handling.py) | 错误处理与容错 | 重试机制、降级处理、断路器模式 |

## 🚀 快速开始

### 1. 安装依赖

```bash
# 基础依赖
pip install langchain langchain-openai langgraph python-dotenv

# 文档处理和向量存储（11-12 课需要）
pip install langchain-community langchain-text-splitters chromadb

# 输出解析（13 课需要）
pip install pydantic

# 可选：其他向量数据库
pip install faiss-cpu  # 或 faiss-gpu
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
OPENAI_API_KEY=your_api_key_here
```

### 3. 运行示例

```bash
# 运行任意示例文件
python 01-basic-chain.py
```

## 📖 学习建议

### 循序渐进
- 按照顺序学习，每个文件都建立在前面的知识基础上
- 先掌握 LangChain 基础，再学习 LangGraph

### 动手实践
- 不要只看代码，要运行每个示例
- 修改参数，观察效果变化
- 尝试组合不同的概念

### 理解核心概念
- **链（Chain）**：预定义的执行流程
- **图（Graph）**：可以表达复杂的有向图结构
- **状态（State）**：在节点间共享的数据
- **节点（Node）**：执行操作的函数
- **边（Edge）**：定义节点之间的连接
- **代理（Agent）**：可以自主决策的智能系统

## 🎯 学习重点

### LangChain 核心概念
1. **提示词模板**：复用和管理提示词
2. **链的组合**：串联多个操作
3. **对话记忆**：维护上下文
4. **工具和代理**：扩展 LLM 能力
5. **文档处理**：加载、分割、向量化
6. **RAG 系统**：检索增强生成
7. **输出解析**：结构化数据提取
8. **回调机制**：追踪和流式输出

### LangGraph 核心概念
1. **状态管理**：如何定义和更新状态
2. **条件边**：动态决定执行流程
3. **循环**：实现迭代和重试
4. **智能体**：构建自主决策系统
5. **人机交互**：中断和人工干预
6. **状态持久化**：检查点和恢复
7. **多 Agent 系统**：协作和竞争
8. **可视化调试**：性能分析和监控
9. **错误处理**：重试和容错

## 💡 关键区别

### LangChain vs LangGraph

| 特性 | LangChain Chains | LangGraph |
|------|------------------|-----------|
| 适用场景 | 简单的线性流程 | 复杂的非线性流程 |
| 流程控制 | 固定的顺序 | 动态的决策 |
| 状态管理 | 有限 | 灵活强大 |
| 循环 | 难以实现 | 原生支持 |
| 适用示例 | 简单的数据处理 | Agent、对话机器人 |

## 🔧 常见问题

### Q: 为什么要分这么多文件？
A: 每个文件只讲一个知识点，更容易理解和消化。你可以一次学一个，不需要一次理解所有内容。

### Q: 需要什么基础？
A: 基本的 Python 知识即可。熟悉函数、类、类型注解会有帮助。

### Q: 如何调试代码？
A: 每个示例都包含详细的打印语句，可以看到执行过程。设置 `verbose=True` 可以看到更多细节。

### Q: 可以跳过某些文件吗？
A: 可以，但不建议。LangGraph 的内容依赖 LangChain 的基础。

### Q: 学习路径建议？
A: 初学者按顺序学习（01-20），有基础者可选择性学习进阶内容（11-20）。

## 📈 完整学习路线

### 第一阶段：LangChain 基础（01-05）
**目标**：掌握 LangChain 的核心概念
- 理解链的概念和 LCEL 语法
- 学会使用提示词模板
- 掌握链的组合方式
- 了解对话记忆机制
- 认识基础 Agent

### 第二阶段：LangGraph 基础（06-10）
**目标**：学会构建复杂的工作流
- 理解图的基本结构
- 掌握状态管理
- 学会使用条件边和循环
- 构建简单的 Agent

### 第三阶段：LangChain 进阶（11-15）
**目标**：构建生产级 RAG 应用
- 文档加载和处理
- 向量存储和检索
- 完整的 RAG 系统实现
- 高级输出解析
- 回调和流式输出

### 第四阶段：LangGraph 高级（16-20）
**目标**：掌握企业级应用开发
- 人机交互设计
- 状态持久化
- 多 Agent 系统
- 可视化和调试
- 错误处理和容错

## 💼 实践项目建议

### 初级项目
1. **简单问答机器人**（使用 01-05）
2. **文档摘要工具**（使用 11-12）
3. **聊天机器人**（使用 04-07）

### 中级项目
4. **RAG 问答系统**（使用 11-15）
5. **客服 Agent**（使用 05-10）
6. **多轮对话系统**（使用 06-09）

### 高级项目
7. **多 Agent 协作系统**（使用 16-18）
8. **内容审核平台**（使用 16, 19-20）
9. **企业知识库系统**（使用 11-20）

## 📊 知识图谱

```
LangChain 生态系统
├── 基础组件
│   ├── Prompts (提示词)
│   ├── LLMs (模型)
│   ├── Chains (链)
│   └── Tools (工具)
│
├── 高级功能
│   ├── Memory (记忆)
│   ├── Agents (代理)
│   ├── Retrievers (检索器)
│   └── Callbacks (回调)
│
└── 应用场景
    ├── RAG (检索增强)
    ├── Chatbots (聊天机器人)
    ├── Summarization (摘要)
    └── Agents (智能体)

LangGraph 核心概念
├── 图结构
│   ├── Nodes (节点)
│   ├── Edges (边)
│   └── State (状态)
│
├── 控制流
│   ├── Sequential (顺序)
│   ├── Conditional (条件)
│   └── Loop (循环)
│
└── 高级特性
    ├── Checkpoints (检查点)
    ├── Persistence (持久化)
    ├── Multi-Agent (多智能体)
    └── Human-in-the-Loop (人机交互)
```

## 📚 进阶资源

- [LangChain 官方文档](https://python.langchain.com/)
- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [OpenAI API 文档](https://platform.openai.com/docs)

## 🤝 贡献

欢迎提出建议和改进！

## 📄 许可

本教程仅供学习使用。

---

**祝学习愉快！如有问题，随时询问。** 🎉
