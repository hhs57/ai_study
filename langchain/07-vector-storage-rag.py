"""
LangChain 学习 12：向量存储与检索（RAG 基础）

知识点：
1. Embeddings：将文本转换为向量
2. Vector Stores：存储和检索向量
3. Retrievers：从向量存储中检索相关文档
4. 构建完整的 RAG（检索增强生成）系统
"""

import sys
import io

# 设置标准输出为 UTF-8 编码
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import get_llm


# ============ 示例 1：理解 Embeddings ============

def example_1_embeddings_basics():
    """示例 1：将文本转换为向量"""
    print("=" * 70)
    print("示例 1：文本嵌入（Embeddings）")
    print("=" * 70)

    # 注意：需要设置 OPENAI_API_KEY 环境变量
    # embeddings = OpenAIEmbeddings()

    # 为了演示，我们使用模拟的向量
    print("""
什么是 Embeddings？

Embeddings 将文本转换为数字向量，使计算机能够"理解"文本的语义。

示例：
- "猫" → [0.1, -0.3, 0.8, ...]  (1536维向量)
- "狗" → [0.2, -0.2, 0.7, ...]  (1536维向量)
- "汽车" → [-0.5, 0.6, 0.1, ...]  (1536维向量)

相似度：
- "猫" 和 "狗" 的向量距离很近（都是动物）
- "猫" 和 "汽车" 的向量距离很远

实际代码示例：
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# 单个文本嵌入
vector = embeddings.embed_query("Hello, world!")
print(f"向量维度: {len(vector)}")  # 1536

# 批量嵌入
vectors = embeddings.embed_documents([
    "Hello",
    "World",
    "Hello, world!"
])
```
    """)

    # 演示向量相似度（简化版）
    def cosine_similarity(vec1, vec2):
        """计算余弦相似度"""
        import numpy as np
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # 模拟向量（3维简化）
    cat_vector = [0.9, 0.1, 0.2]
    dog_vector = [0.8, 0.2, 0.3]
    car_vector = [0.1, 0.9, 0.8]

    cat_dog_sim = cosine_similarity(cat_vector, dog_vector)
    cat_car_sim = cosine_similarity(cat_vector, car_vector)

    print(f"\n模拟的向量相似度:")
    print(f"  '猫' 和 '狗': {cat_dog_sim:.3f} (相似)")
    print(f"  '猫' 和 '汽车': {cat_car_sim:.3f} (不相似)")


# ============ 示例 2：简单的向量存储（内存中）============

def example_2_simple_vector_store():
    """示例 2：使用简单的向量存储"""
    print("=" * 70)
    print("示例 2：内存向量存储（简化演示）")
    print("=" * 70)

    # 创建文档集合
    documents = [
        Document(page_content="Python 是一种编程语言", metadata={"id": 1}),
        Document(page_content="JavaScript 用于 Web 开发", metadata={"id": 2}),
        Document(page_content="Python 适合数据科学", metadata={"id": 3}),
        Document(page_content="JavaScript 适合前端", metadata={"id": 4}),
        Document(page_content="Java 是企业级语言", metadata={"id": 5})
    ]

    print(f"\n文档集合: {len(documents)} 个文档")
    for doc in documents:
        print(f"  - {doc.page_content}")

    print("""
在实际应用中，你会使用真实的向量数据库：

1. Chroma（轻量级，适合开发）
2. FAISS（Facebook，高性能）
3. Pinecone（云服务）
4. Weaviate（开源，功能丰富）
5. Qdrant（高性能，易用）

使用 Chroma 的示例：
```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 创建向量存储
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=OpenAIEmbeddings(),
    collection_name="my_collection"
)

# 检索相关文档
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}  # 返回最相关的 2 个文档
)

results = retriever.invoke("Python 数据分析")
```
    """)


# ============ 示例 3：构建完整的 RAG 系统 ============

def example_3_simple_rag():
    """示例 3：简化的 RAG 系统"""
    print("=" * 70)
    print("示例 3：检索增强生成（RAG）")
    print("=" * 70)

    print("""
RAG 的工作流程：

1. 索引阶段（离线）：
   文档 → 分割 → 嵌入 → 存储到向量数据库

2. 检索阶段（在线）：
   用户问题 → 嵌入 → 向量检索 → 获取相关文档

3. 生成阶段（在线）：
   问题 + 相关文档 → LLM → 生成答案

完整示例代码：
```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 步骤 1：准备文档
documents = [
    Document(page_content="LangChain 是一个框架..."),
    Document(page_content="RAG 是检索增强生成..."),
    # ...更多文档
]

# 步骤 2：分割文档
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
splits = splitter.split_documents(documents)

# 步骤 3：创建向量存储
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 步骤 4：创建 RAG 链
template = \"\"\"基于以下上下文回答问题：
上下文：
{context}

问题：{question}

答案：\"\"\"

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(model="gpt-3.5-turbo")

def format_docs(docs):
    return "\\n\\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 步骤 5：提问
answer = rag_chain.invoke("什么是 RAG？")
print(answer)
```
    """)


# ============ 示例 4：RAG 详解 ============

def example_4_rag_explained():
    """示例 4：RAG 各组件详解"""
    print("=" * 70)
    print("示例 4：RAG 组件详解")
    print("=" * 70)

    # 模拟的文档检索
    def mock_retrieve(query, documents):
        """模拟向量检索"""
        # 简单的关键词匹配
        query_lower = query.lower()
        scored_docs = []

        for doc in documents:
            score = 0
            # 简单的关键词匹配计分
            for word in query_lower.split():
                if word in doc.page_content.lower():
                    score += 1

            if score > 0:
                scored_docs.append((score, doc))

        # 按分数排序，返回前 3 个
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:3]]

    # 创建文档库
    knowledge_base = [
        Document(page_content="LangChain 是一个开发语言模型应用的框架。它提供了链、代理、工具等组件。"),
        Document(page_content="RAG（检索增强生成）是一种技术，结合了检索和生成。它从知识库中检索相关信息，然后使用 LLM 生成答案。"),
        Document(page_content="向量数据库专门存储和检索向量。它们使用近似最近邻（ANN）算法快速找到相似向量。"),
        Document(page_content="Embeddings 将文本转换为数字向量。语义相似的文本会有相似的向量。"),
        Document(page_content="LLM 是大语言模型的缩写，如 GPT-4、Claude 等。它们可以理解和生成自然语言。")
    ]

    # 模拟 RAG 流程
    question = "什么是 RAG？"

    print(f"\n用户问题: {question}")

    # 步骤 1：检索
    print("\n[步骤 1] 检索相关文档...")
    retrieved_docs = mock_retrieve(question, knowledge_base)

    print(f"找到 {len(retrieved_docs)} 个相关文档:")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"\n  文档 {i}:")
        print(f"    {doc.page_content[:100]}...")

    # 步骤 2：构建提示词
    print("\n[步骤 2] 构建 RAG 提示词...")

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    rag_prompt = f"""基于以下上下文回答问题。如果上下文中没有相关信息，请说"我没有找到相关信息"。

上下文：
{context}

问题：{question}

答案："""

    print(f"\n生成的提示词:")
    print("-" * 70)
    print(rag_prompt[:200] + "...")
    print("-" * 70)

    # 步骤 3：LLM 生成答案
    print("\n[步骤 3] LLM 生成答案...")
    print("""
在实际应用中，这里会调用 LLM：
```python
llm = ChatOpenAI(model="gpt-3.5-turbo")
answer = llm.invoke(rag_prompt)
```

模拟输出：
"RAG（检索增强生成）是一种技术，它从知识库中检索相关信息，然后使用大语言模型生成答案。这种方法结合了检索的准确性和生成的灵活性。"
    """)


# ============ 示例 5：不同的检索策略 ============

def example_5_retrieval_strategies():
    """示例 5：不同的检索策略"""
    print("=" * 70)
    print("示例 5：检索策略")
    print("=" * 70)

    print("""
向量检索支持多种策略：

1. 相似度检索（Similarity）：
   - 返回与查询最相似的 k 个文档
   - 最常用
   ```python
   retriever = vectorstore.as_retriever(
       search_type="similarity",
       search_kwargs={"k": 3}
   )
   ```

2. 相似度分数阈值（Similarity with Score）：
   - 只返回相似度超过阈值的文档
   - 保证质量
   ```python
   retriever = vectorstore.as_retriever(
       search_type="similarity_score_threshold",
       search_kwargs={
           "k": 3,
           "score_threshold": 0.5  # 只保留相似度 > 0.5 的
       }
   )
   ```

3. 最大边际相关性（MMR）：
   - 平衡相关性和多样性
   - 避免返回过于相似的文档
   ```python
   retriever = vectorstore.as_retriever(
       search_type="mmr",
       search_kwargs={"k": 3, "fetch_k": 10}
   )
   ```

选择策略的建议：
- 默认使用 similarity
- 需要多样性时使用 mmr
- 需要质量保证时使用 threshold
    """)


# ============ 示例 6：RAG 最佳实践 ============

def example_6_rag_best_practices():
    """示例 6：RAG 最佳实践"""
    print("=" * 70)
    print("示例 6：RAG 最佳实践")
    print("=" * 70)

    print("""
1. 文档准备：
   ✓ 清理文档，移除无关内容
   ✓ 保留有意义的元数据
   ✓ 标准化格式

2. 文档分割：
   ✓ chunk_size: 500-1500 字符（根据内容调整）
   ✓ chunk_overlap: 10-20% 的 chunk_size
   ✓ 选择合适的分隔符保持语义完整

3. 向量化：
   ✓ 使用高质量的 Embeddings 模型
   ✓ 考虑使用领域特定的模型
   ✓ 缓存向量结果

4. 检索：
   ✓ k 值通常选择 3-5
   ✓ 根据查询类型调整策略
   ✓ 考虑查询重写（Query Rewriting）

5. 提示词工程：
   ✓ 明确指示使用上下文
   ✓ 提供拒绝回答的指示
   ✓ 格式化输出要求

6. 评估：
   ✓ 测量检索准确率
   ✓ 测量答案质量
   ✓ 收集用户反馈

常见问题：
❌ 检索到的文档不相关
   → 改进文档质量
   → 调整 chunk_size
   → 使用查询重写

❌ 答案不准确
   → 增加 k 值
   → 改进提示词
   → 使用更好的 LLM

❌ 响应速度慢
   → 减少检索的文档数
   → 使用更快的向量数据库
   → 缓存常见查询
    """)


# ============ 示例 7：完整的 RAG 代码模板 ============

def example_7_rag_template():
    """示例 7：完整的 RAG 代码模板"""
    print("=" * 70)
    print("示例 7：RAG 代码模板")
    print("=" * 70)

    print("""
完整的 RAG 实现模板：

```python
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. 加载文档
documents = load_documents("path/to/docs")

# 2. 分割文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(documents)

# 3. 创建向量存储
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db"  # 持久化
)

# 4. 创建检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# 5. 创建 RAG 提示词
template = \"\"\"
你是一个有帮助的助手。使用以下上下文片段回答问题。
如果上下文中没有答案，就说"我没有找到相关信息"，不要编造答案。

上下文：
{context}

问题：{question}

答案：
\"\"\"

prompt = ChatPromptTemplate.from_template(template)

# 6. 创建 LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 7. 格式化文档
def format_docs(docs):
    return "\\n\\n---\\n\\n".join(doc.page_content for doc in docs)

# 8. 创建 RAG 链
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 9. 使用 RAG
question = "你的问题"
answer = rag_chain.invoke(question)
print(answer)
```

添加引用来源：
```python
def format_docs_with_sources(docs):
    context = ""
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        context += f"[来源 {i}: {source}]\\n{doc.page_content}\\n\\n"
    return context

rag_chain = (
    {
        "context": retriever | format_docs_with_sources,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
```
    """)


# 总结：核心概念
"""
【向量存储与 RAG 的核心概念】

1. Embeddings（嵌入）：
   - 将文本转换为数字向量
   - 语义相似的文本有相似的向量
   - OpenAI embeddings: 1536 维

2. Vector Stores（向量数据库）：
   - 存储文档的向量表示
   - 支持快速相似度搜索
   - 常见选项：
     * Chroma: 轻量级，易用
     * FAISS: 高性能
     * Pinecone: 云服务
     * Weaviate: 功能丰富

3. Retrievers（检索器）：
   - 从向量数据库中检索相关文档
   - 支持不同的检索策略
   - 可以集成到链中

4. RAG（检索增强生成）：
   流程：
   a. 索引阶段：
      - 加载文档 → 分割 → 嵌入 → 存储

   b. 检索阶段：
      - 查询 → 嵌入 → 向量搜索 → 获取文档

   c. 生成阶段：
      - 查询 + 文档 → LLM → 生成答案

5. RAG 的优势：
   - 减少幻觉（基于真实文档）
   - 知识可更新（不需要重新训练模型）
   - 可解释性（可以引用来源）
   - 领域知识（可以加入专业数据）

6. 关键参数：
   - chunk_size: 每块大小（500-1500）
   - chunk_overlap: 重叠大小（10-20%）
   - k: 检索的文档数（3-5）
   - score_threshold: 相似度阈值

【下一步学习】

在 13-output-parsers-advanced.py 中，你将学习：
- PydanticOutputParser：结构化数据提取
- 其他高级输出解析器
- 如何处理复杂的数据结构
"""

if __name__ == "__main__":
    example_1_embeddings_basics()
    example_2_simple_vector_store()
    example_3_simple_rag()
    example_4_rag_explained()
    example_5_retrieval_strategies()
    example_6_rag_best_practices()
    example_7_rag_template()
