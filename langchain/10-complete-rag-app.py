"""
LangChain 学习 15：完整的 RAG 应用

知识点：
1. 综合运用前面学到的所有概念
2. 构建一个完整的 RAG 应用
3. 添加引用来源
4. 实现查询重写
5. 评估和优化 RAG 系统
"""

import sys
import io
import os

# 设置标准输出为 UTF-8 编码
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
from config import get_llm


# ============ 示例 1：构建完整 RAG 系统的步骤 ============

def example_1_rag_architecture():
    """示例 1：RAG 系统架构"""
    print("=" * 70)
    print("示例 1：RAG 系统架构")
    print("=" * 70)

    print("""
完整的 RAG 系统包含以下组件：

┌─────────────────────────────────────────────────────────────┐
│                        RAG 架构                              │
└─────────────────────────────────────────────────────────────┘

1. 数据准备阶段（离线）:
   ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐
   │ 加载文档│ -> │  分割    │ -> │  嵌入    │ -> │ 存储   │
   └─────────┘    └──────────┘    └──────────┘    └────────┘
                                                  Vector DB

2. 查询处理阶段（在线）:
   ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐
   │用户问题 │ -> │ 查询重写 │ -> │  向量化  │ -> │ 检索   │
   └─────────┘    └──────────┘    └──────────┘    └────────┘
                                                  Docs

3. 答案生成阶段（在线）:
   ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐
   │ 问题+   │ -> │  提示词  │ -> │   LLM    │ -> │  答案 │
   │ 上下文  │    │  模板    │    │          │    │        │
   └─────────┘    └──────────┘    └──────────┘    └────────┘

关键组件：
- Document Loaders：加载各种格式的文档
- Text Splitters：智能分割文档
- Embeddings：文本向量化
- Vector Store：向量存储和检索
- Retriever：检索相关文档
- Prompt Template：构建 RAG 提示词
- LLM：生成答案
    """)


# ============ 示例 2：简化的 RAG 实现 ============

def example_2_simple_rag_implementation():
    """示例 2：简化的 RAG 实现（不依赖向量数据库）"""
    print("=" * 70)
    print("示例 2：简化的 RAG 实现")
    print("=" * 70)

    # 创建知识库
    knowledge_base = [
        Document(
            page_content="LangChain 是一个开源框架，用于开发由语言模型驱动的应用程序。它提供了链、代理、工具等核心组件。",
            metadata={"source": "langchain_docs.txt", "topic": "langchain"}
        ),
        Document(
            page_content="RAG（检索增强生成）结合了信息检索和文本生成。它从知识库中检索相关信息，然后使用大语言模型生成答案。",
            metadata={"source": "rag_intro.txt", "topic": "rag"}
        ),
        Document(
            page_content="向量数据库专门存储和检索向量。常见的向量数据库包括 Chroma、FAISS、Pinecone、Weaviate 等。",
            metadata={"source": "vector_db.txt", "topic": "database"}
        ),
        Document(
            page_content="Embeddings 将文本转换为数字向量。语义相似的文本会有相似的向量表示。OpenAI 的嵌入模型生成 1536 维向量。",
            metadata={"source": "embeddings.txt", "topic": "embeddings"}
        ),
        Document(
            page_content="LLM（大语言模型）如 GPT-4、Claude 等能够理解和生成自然语言。它们通过在海量文本上训练获得语言能力。",
            metadata={"source": "llm_intro.txt", "topic": "llm"}
        )
    ]

    # 简单的关键词匹配检索器
    def simple_retriever(query: str, docs: List[Document], k: int = 3) -> List[Document]:
        """简单的关键词匹配检索"""
        query_lower = query.lower()
        scored_docs = []

        for doc in docs:
            score = 0
            # 简单的关键词匹配
            for word in query_lower.split():
                if word in doc.page_content.lower():
                    score += 1

            if score > 0:
                scored_docs.append((score, doc))

        # 排序并返回前 k 个
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:k]]

    # RAG 提示词模板
    rag_template = """你是一个有帮助的助手。请基于以下上下文回答问题。

上下文：
{context}

问题：{question}

回答要求：
1. 只使用上下文中的信息
2. 如果上下文中没有相关信息，请明确说明
3. 在回答中引用信息来源
4. 保持回答简洁准确

回答："""

    prompt = ChatPromptTemplate.from_template(rag_template)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def format_docs_with_sources(docs: List[Document]) -> str:
        """格式化文档，包含来源信息"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            formatted.append(
                f"[来源 {i}: {source}]\n{doc.page_content}"
            )
        return "\n\n".join(formatted)

    # 构建 RAG 链
    def rag_chain(query: str) -> str:
        """完整的 RAG 流程"""
        print(f"\n[1] 检索相关文档...")
        retrieved_docs = simple_retriever(query, knowledge_base, k=3)

        print(f"[2] 找到 {len(retrieved_docs)} 个相关文档:")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"    {i}. {doc.metadata.get('source', 'Unknown')} - {doc.page_content[:50]}...")

        print(f"\n[3] 生成答案...")
        context = format_docs_with_sources(retrieved_docs)

        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "question": query
        })

        return answer, retrieved_docs

    # 测试
    questions = [
        "什么是 RAG？",
        "向量数据库有哪些？",
        "LangChain 是什么？"
    ]

    for question in questions:
        print("\n" + "=" * 70)
        print(f"问题: {question}")
        print("=" * 70)

        answer, sources = rag_chain(question)

        print(f"\n[答案]")
        print(answer)
        print()


# ============ 示例 3：添加引用来源 ============

def example_3_rag_with_citations():
    """示例 3：带引用的 RAG 系统"""
    print("=" * 70)
    print("示例 3：带引用的 RAG")
    print("=" * 70)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # 知识库
    docs = [
        Document(page_content="Python 是一种高级编程语言，由 Guido van Rossum 创建。", metadata={"source": "python_intro.txt", "page": 1}),
        Document(page_content="Python 支持多种编程范式，包括面向对象、函数式和过程式编程。", metadata={"source": "python_features.txt", "page": 2}),
        Document(page_content="Python 广泛应用于 Web 开发、数据科学、人工智能等领域。", metadata={"source": "python_apps.txt", "page": 3})
    ]

    # 改进的提示词
    template = """基于以下上下文回答问题，并在每句话后面标注引用来源。

上下文：
{context}

问题：{question}

回答格式示例：
Python 是一种编程语言 [来源 1]。它支持多种编程范式 [来源 2]。

注意：
- 只使用上下文中的信息
- 在相关句子后添加 [来源 X]
- 如果上下文中没有信息，明确说明

回答："""

    prompt = ChatPromptTemplate.from_template(template)

    def format_with_citations(docs: List[Document]) -> str:
        """格式化文档以便引用"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            formatted.append(
                f"[来源 {i}: {source}, 页码 {page}]\n{doc.page_content}"
            )
        return "\n\n".join(formatted)

    # 简化的检索器
    def retrieve(query: str) -> List[Document]:
        query_lower = query.lower()
        results = []
        for doc in docs:
            # 简单的关键词匹配
            if any(word in doc.page_content.lower() for word in query_lower.split()):
                results.append(doc)
        return results[:3]

    # 构建链
    def ask_with_citations(question: str) -> str:
        retrieved = retrieve(question)
        context = format_with_citations(retrieved)

        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "question": question
        })

        return answer

    # 测试
    question = "Python 的特点和用途是什么？"
    print(f"问题: {question}\n")

    answer = ask_with_citations(question)
    print("答案:")
    print(answer)


# ============ 示例 4：查询重写（Query Rewriting）============

def example_4_query_rewriting():
    """示例 4：查询重写优化检索"""
    print("=" * 70)
    print("示例 4：查询重写")
    print("=" * 70)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    print("""
查询重写的作用：

1. 问题规范化：
   用户: "怎么弄？"
   重写: "如何使用 LangChain 构建应用？"

2. 添加上下文：
   用户: "它贵吗？"
   重写: "GPT-4 API 的价格贵吗？"

3. 多角度查询：
   原始: "Python 性能"
   重写: ["Python 执行速度", "Python 性能优化", "Python vs C++ 性能"]

实现示例：
""")

    # 查询重写提示词
    rewrite_template = """你是一个查询优化专家。请重写用户的查询，使其更清晰、更具体。

原始查询：{query}

上下文：这是一个关于 LangChain 和 Python 编程的对话。

请提供：
1. 重写后的查询（更完整、更具体）
2. 2-3 个相关的搜索查询

输出格式：
重写查询: [重写后的查询]
相关查询 1: [相关查询]
相关查询 2: [相关查询]
"""

    rewrite_prompt = ChatPromptTemplate.from_template(rewrite_template)

    # 测试查询重写
    test_queries = [
        "怎么做 RAG？",
        "它快吗？",
        "有什么用？"
    ]

    for query in test_queries:
        print(f"\n原始查询: {query}")
        print("-" * 50)

        rewrite_chain = rewrite_prompt | llm | StrOutputParser()
        rewritten = rewrite_chain.invoke({"query": query})

        print(rewritten)
        print()


# ============ 示例 5：RAG 评估 ============

def example_5_rag_evaluation():
    """示例 5：评估 RAG 系统质量"""
    print("=" * 70)
    print("示例 5：RAG 评估指标")
    print("=" * 70)

    print("""
RAG 系统的评估维度：

1. 检索质量指标：

   a. Precision（精确率）：
      检索到的文档中有多少是相关的？
      Precision = 相关文档数 / 检索到的文档数

   b. Recall（召回率）：
      相关文档中有多少被检索到？
      Recall = 检索到的相关文档数 / 所有关联文档数

   c. MRR（Mean Reciprocal Rank）：
      第一个相关文档的平均排名
      MRR = 1 / 第一个相关文档的排名

   d. NDCG（Normalized Discounted Cumulative Gain）：
      考虑排序质量的指标

2. 生成质量指标：

   a. Faithfulness（忠实度）：
      答案是否基于检索到的上下文？
      - 检查答案中的陈述是否在上下文中
      - 避免幻觉

   b. Answer Relevance（答案相关性）：
      答案是否回答了用户的问题？
      - 答案是否切题
      - 信息是否完整

   c. Citation Quality（引用质量）：
      引用是否准确和充分？
      - 引用的来源是否正确
      - 关键信息是否都有引用

3. 端到端评估：

   a. 用户满意度：
      - 人工评估
      - 用户反馈
      - 点击率

   b. 任务完成率：
      - 是否解决了用户问题
      - 是否需要进一步查询

   c. 响应时间：
      - 检索时间
      - 生成时间
      - 总体延迟

评估数据集示例：
```python
eval_data = [
    {
        "question": "什么是 RAG？",
        "expected_context": ["检索增强生成的文档"],
        "expected_answer_keywords": ["检索", "生成", "结合"],
        "reference_answer": "RAG 是检索增强生成，结合了检索和生成..."
    },
    # ... 更多测试案例
]

# 评估函数
def evaluate_rag(rag_system, eval_data):
    results = {
        "precision": [],
        "recall": [],
        "faithfulness": [],
        "relevance": []
    }

    for case in eval_data:
        # 运行 RAG 系统
        answer = rag_system.query(case["question"])

        # 评估各项指标
        # ...

    return results
```
    """)


# ============ 示例 6：RAG 最佳实践 ============

def example_6_rag_best_practices():
    """示例 6：RAG 系统最佳实践"""
    print("=" * 70)
    print("示例 6：RAG 最佳实践")
    print("=" * 70)

    print("""
RAG 系统的最佳实践：

1. 文档准备：
   ✓ 清理和标准化文档
   ✓ 保留有意义的元数据
   ✓ 移除重复和低质量内容
   ✓ 结构化数据（标题、章节等）

2. 文档分割：
   ✓ chunk_size: 500-1500 字符
   ✓ chunk_overlap: 10-20%
   ✓ 保持语义完整性
   ✓ 使用合适的分隔符

3. 嵌入模型选择：
   ✓ OpenAI text-embedding-3: 通用场景
   ✓ 领域特定模型: 专业领域
   ✓ 多语言模型: 多语言内容
   ✓ 考虑成本和性能

4. 向量数据库选择：
   ✓ Chroma: 开发和原型
   ✓ FAISS: 高性能本地
   ✓ Pinecone: 云托管
   ✓ Weaviate: 功能丰富

5. 检索策略：
   ✓ k 值: 通常 3-5
   ✓ 相似度阈值: 过滤低质量结果
   ✓ 混合检索: 关键词 + 向量
   ✓ 查询扩展: 提高召回率

6. 提示词工程：
   ✓ 明确使用上下文的要求
   ✓ 提供拒绝回答的指示
   ✓ 要求引用来源
   ✓ 格式化输出

7. 性能优化：
   ✓ 缓存常见查询
   ✓ 批量处理
   ✓ 异步检索
   ✓ 索引优化

8. 监控和改进：
   ✓ 记录查询和答案
   ✓ 收集用户反馈
   ✓ A/B 测试
   ✓ 定期重新索引

常见问题及解决方案：

问题 1：检索结果不相关
→ 改进文档质量
→ 调整 chunk_size
→ 使用查询重写
→ 增加检索数量

问题 2：答案不准确
→ 改进提示词
→ 增加 k 值
→ 使用更好的 LLM
→ 添加重排序

问题 3：响应速度慢
→ 减少检索的文档数
→ 使用更快的向量数据库
→ 缓存查询结果
→ 优化索引

问题 4：答案太简短
→ 调整 temperature
→ 改进提示词
→ 提供更多上下文
→ 使用更强的模型

问题 5：产生幻觉
→ 明确要求使用上下文
→ 添加验证步骤
→ 降低 temperature
→ 使用更严格的提示词
    """)


# ============ 示例 7：生产级 RAG 模板 ============

def example_7_production_template():
    """示例 7：生产级 RAG 代码模板"""
    print("=" * 70)
    print("示例 7：生产级 RAG 代码模板")
    print("=" * 70)

    print("""
完整的 RAG 系统代码模板：

```python
import os
from typing import List, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RAGSystem:
    \"\"\"生产级 RAG 系统\"\"\"

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-3.5-turbo",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 4
    ):
        # 初始化组件
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.top_k = top_k
        self.persist_directory = persist_directory

        # 初始化向量存储
        self.vectorstore = None
        self.retriever = None

        # 设置提示词
        self._setup_prompts()

    def _setup_prompts(self):
        \"\"\"设置提示词模板\"\"\"
        self.rag_template = \"\"\"你是一个专业的助手。请基于以下上下文回答问题。

上下文：
{context}

问题：{question}

要求：
1. 只使用上下文中的信息
2. 如果上下文没有相关信息，明确说明
3. 在关键信息后标注引用 [来源 X]
4. 保持回答简洁准确

回答：\"\"\"

        self.prompt = ChatPromptTemplate.from_template(self.rag_template)

    def index_documents(self, documents: List[Document]):
        \"\"\"索引文档\"\"\"
        # 分割文档
        splits = self.text_splitter.split_documents(documents)

        # 创建向量存储
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

        # 创建检索器
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )

        print(f"索引完成: {len(splits)} 个文档块")

    def load_existing_index(self):
        \"\"\"加载已存在的索引\"\"\"
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )

    def format_docs(self, docs: List[Document]) -> str:
        \"\"\"格式化文档\"\"\"
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            content = doc.page_content
            formatted.append(
                f\"[来源 {i}: {source}, 页码 {page}]\\n{content}\"
            )
        return \"\\n\\n\".join(formatted)

    def query(self, question: str) -> dict:
        \"\"\"查询 RAG 系统\"\"\"
        if not self.retriever:
            raise ValueError("请先索引或加载文档")

        # 构建 RAG 链
        rag_chain = (
            {
                "context": self.retriever | self.format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        # 生成答案
        answer = rag_chain.invoke(question)

        # 获取源文档
        source_docs = self.retriever.invoke(question)

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in source_docs
            ]
        }

# 使用示例
if __name__ == "__main__":
    # 创建 RAG 系统
    rag = RAGSystem(
        persist_directory="./my_db",
        top_k=3
    )

    # 方式 1: 索引新文档
    documents = [
        Document(page_content="...", metadata={"source": "doc1.txt"}),
        # ... 更多文档
    ]
    rag.index_documents(documents)

    # 方式 2: 加载已有索引
    # rag.load_existing_index()

    # 查询
    result = rag.query("什么是 RAG？")
    print(f"问题: {result['question']}")
    print(f"答案: {result['answer']}")
    print(f"来源: {len(result['sources'])} 个文档")
```

这个模板包含：
✓ 完整的初始化配置
✓ 文档索引功能
✓ 持久化存储
✓ 灵活的查询接口
✓ 格式化的输出
✓ 来源引用
✓ 错误处理
    """)


# 总结：核心概念
"""
【完整 RAG 应用的核心概念】

1. RAG 架构：
   - 索引阶段：加载 → 分割 → 嵌入 → 存储
   - 检索阶段：查询 → 向量化 → 检索
   - 生成阶段：上下文 + 问题 → LLM → 答案

2. 关键组件：
   - Document Loaders：文档加载
   - Text Splitters：文档分割
   - Embeddings：文本向量化
   - Vector Store：向量存储
   - Retriever：检索器
   - LLM：生成器

3. 高级功能：
   - 引用来源：追溯信息来源
   - 查询重写：优化检索质量
   - 混合检索：关键词 + 向量
   - 重排序：优化检索结果
   - 流式输出：实时反馈

4. 评估指标：
   - 检索质量：Precision, Recall, MRR
   - 生成质量：Faithfulness, Relevance
   - 用户体验：满意度，响应时间

5. 最佳实践：
   - 仔细准备文档
   - 合理设置参数
   - 监控系统性能
   - 持续优化改进
   - 收集用户反馈

6. 生产部署考虑：
   - 成本控制：API 调用，存储
   - 性能优化：缓存，批量处理
   - 可扩展性：水平扩展，负载均衡
   - 安全性：数据加密，访问控制
   - 监控：日志，指标，告警

【LangChain 部分总结】

你已经学习了 LangChain 的核心概念：
✓ 基础链和 LCEL
✓ 提示词模板
✓ 链的组合
✓ 对话记忆
✓ Agents 和工具
✓ 文档加载和处理
✓ 向量存储和 RAG
✓ 高级输出解析器
✓ 回调和流式输出
✓ 完整的 RAG 应用

【下一步学习】

在 16-human-in-loop.py 中，你将学习 LangGraph 的高级特性：
- 人机交互
- 状态持久化
- 多 Agent 系统
- 错误处理
"""

if __name__ == "__main__":
    example_1_rag_architecture()
    example_2_simple_rag_implementation()
    example_3_rag_with_citations()
    example_4_query_rewriting()
    example_5_rag_evaluation()
    example_6_rag_best_practices()
    example_7_production_template()
