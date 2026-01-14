# AI 学习平台 (AI Study Platform)

欢迎来到 **AI 学习平台**！这是一个开源的、系统化的 AI 技术学习知识库，旨在帮助开发者从理论到实践全面掌握现代人工智能技术栈。

本项目不仅仅是代码仓库，更是一套交互式的学习课程。

## 🌟 核心学习路径

本项目包含以下核心学习主题：

### 1. 🦜⛓️ [LangChain & LangGraph 应用开发](./langchain/index.html)
从基础概念到企业级 Agent 开发，系统掌握 LLM 应用框架。

- **内容**: Prompt Engineering, Chains, RAG, Agents, Multi-Agent Systems, State Management.
- **形式**: Python 源代码教程 + 详细注释。
- **状态**: ✅ 已上线 (Modules 01-20)
- **目录**: [`langchain/`](./langchain) 和 [`langgraph/`](./langgraph)

### 2. ⚡ [Transformer 架构深度解析](./transformer/index.html)
从零开始理解大模型的基石，通过**交互式可视化**深入学习底层原理。

- **内容**: Self-Attention, Multi-Head Attention, Encoder/Decoder, BERT, GPT, ViT.
- **形式**: 交互式 HTML/JS 可视化 (无需 Python 环境即可体验)。
- **状态**: 🚀 进行中 (Phase 1 & 2 Completed)
- **入口**: [`transformer/index.html`](./transformer/index.html)

### 3. 🕸️ 知识图谱 (Knowledge Graph)
*（筹备中）*
结合 LLM 与结构化数据，构建更精准的 RAG 系统。

---

## 🚀 快速开始

### 环境依赖
本项目主要使用 Python 进行开发 (LangChain 部分) 和 HTML/JS (Transformer 可视化部分)。

```bash
# 克隆仓库
git clone https://github.com/hhs57/ai_study.git
cd ai_study

# 安装 Python 依赖 (用于 LangChain/LangGraph 部分)
pip install -r requirements.txt
```

### 运行可视化课程
Transformer 课程为纯静态 HTML 文件，无需后端服务器。
直接用浏览器打开 `transformer/index.html` 即可开始学习。

## 📂 项目结构

```
ai_study/
├── langchain/          # LangChain 基础与进阶示例
├── langgraph/          # LangGraph 工作流与 Agent 示例
├── knowledge-graph/    # (筹备中) 知识图谱相关代码
├── transformer/        # Transformer 架构可视化互动课程
│   ├── index.html      # 课程主页
│   └── ...             # 各章节 HTML 文件
├── CLAUDE.md           # 开发规范与最佳实践
├── ANTIGRAVITY_RULES.md # 智能体行为准则
└── index.html          # 项目总导航页
```

## 🤝 贡献与反馈
欢迎提交 Issue 或 Pull Request 来改进课程内容。

## 📄 许可证
MIT License
