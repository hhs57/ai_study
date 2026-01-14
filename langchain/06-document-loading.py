"""
LangChain 学习 11：文档加载与处理

知识点：
1. Document Loaders：从不同来源加载文档
2. Text Splitters：将长文档分割成小块
3. 不同的分割策略（字符、递归、语义等）
4. 文档元数据管理
"""

import sys
import io

# 设置标准输出为 UTF-8 编码
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownTextSplitter
)
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    JSONLoader,
    CSVLoader
)
from config import get_llm


# ============ 示例 1：基础文档加载 ============

def example_1_text_loader():
    """示例 1：加载文本文件"""
    print("=" * 70)
    print("示例 1：使用 TextLoader 加载文本")
    print("=" * 70)

    # 先创建一个示例文本文件
    sample_file = "sample_text.txt"
    sample_content = """
    LangChain 是一个强大的框架，用于开发由语言模型驱动的应用程序。

    它提供了以下主要功能：
    1. 提示词管理
    2. 链式调用
    3. 数据增强生成
    4. 智能体（Agents）
    5. 记忆管理

    LangChain 的核心理念是可组合性，让开发者能够灵活地组合不同的组件。
    """

    # 写入示例文件
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(sample_content)

    # 使用 TextLoader 加载
    loader = TextLoader(sample_file, encoding='utf-8')
    documents = loader.load()

    print(f"\n加载的文档数量: {len(documents)}")
    print(f"\n文档内容预览:")
    print(f"  页面内容: {documents[0].page_content[:100]}...")
    print(f"  元数据: {documents[0].metadata}\n")

    # 清理
    import os
    os.remove(sample_file)


# ============ 示例 2：文档分割 ============

def example_2_text_splitting():
    """示例 2：分割长文档"""
    print("=" * 70)
    print("示例 2：文本分割 - RecursiveCharacterTextSplitter")
    print("=" * 70)

    # 创建一个长文本示例
    long_text = """
    第一章：Python 简介

    Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年首次发布。
    它的设计哲学强调代码的可读性和简洁的语法。

    第二章：Python 的特点

    1. 简单易学：Python 的语法简洁明了，适合初学者。
    2. 功能强大：拥有丰富的标准库和第三方库。
    3. 跨平台：可以在 Windows、Mac、Linux 上运行。
    4. 面向对象：支持面向对象编程。

    第三章：Python 的应用

    Python 被广泛应用于：
    - Web 开发（Django、Flask）
    - 数据科学（Pandas、NumPy）
    - 人工智能（TensorFlow、PyTorch）
    - 自动化脚本

    第四章：总结

    Python 是一门优秀的编程语言，值得深入学习。
    """ * 5  # 重复几次使文本更长

    # 使用递归字符分割器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,  # 每块的最大字符数
        chunk_overlap=50,  # 块之间的重叠字符数
        length_function=len,  # 计算长度的函数
        separators=["\n\n", "\n", "。", "，", " ", ""]  # 分隔符优先级
    )

    chunks = splitter.split_text(long_text)

    print(f"\n原始文本长度: {len(long_text)} 字符")
    print(f"分割后块数: {len(chunks)}")
    print(f"\n前3块内容:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n--- 块 {i} ({len(chunk)} 字符) ---")
        print(chunk[:100] + "...")

    print()


# ============ 示例 3：分割文档对象 ============

def example_3_split_documents():
    """示例 3：分割 Document 对象"""
    print("=" * 70)
    print("示例 3：分割 Document 对象并保留元数据")
    print("=" * 70)

    # 创建文档列表
    documents = [
        Document(
            page_content="LangChain 是一个框架。它让开发语言模型应用变得简单。",
            metadata={"source": "intro.txt", "chapter": "1"}
        ),
        Document(
            page_content="LangChain 支持多种模型。包括 OpenAI、Anthropic 等。",
            metadata={"source": "models.txt", "chapter": "2"}
        ),
        Document(
            page_content="LangChain 可以连接外部数据。这就是 RAG（检索增强生成）。",
            metadata={"source": "rag.txt", "chapter": "3"}
        )
    ]

    # 创建分割器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=10,
        separators=["。", "，", " ", ""]
    )

    # 分割文档
    split_docs = splitter.split_documents(documents)

    print(f"\n原始文档数: {len(documents)}")
    print(f"分割后文档数: {len(split_docs)}")
    print(f"\n分割结果:")
    for i, doc in enumerate(split_docs, 1):
        print(f"\n文档 {i}:")
        print(f"  内容: {doc.page_content}")
        print(f"  来源: {doc.metadata}")


# ============ 示例 4：不同的分割策略 ============

def example_4_splitting_strategies():
    """示例 4：比较不同的分割策略"""
    print("=" * 70)
    print("示例 4：不同分割策略的比较")
    print("=" * 70)

    text = """
    # Python 学习指南

    ## 第一章：基础语法

    Python 是一种解释型语言。代码逐行执行。

    ## 第二章：数据类型

    Python 有多种数据类型：整数、浮点数、字符串、列表等。

    ## 第三章：控制流

    if 语句用于条件判断。for 循环用于迭代。
    """

    print("\n策略 1: 按字符分割")
    print("-" * 50)
    char_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=100,
        chunk_overlap=0,
        length_function=len
    )
    char_chunks = char_splitter.split_text(text)
    print(f"块数: {len(char_chunks)}")
    for i, chunk in enumerate(char_chunks, 1):
        print(f"块 {i}: {chunk.strip()[:50]}...")

    print("\n\n策略 2: 递归分割")
    print("-" * 50)
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        separators=["\n##", "\n", "。", " ", ""]
    )
    recursive_chunks = recursive_splitter.split_text(text)
    print(f"块数: {len(recursive_chunks)}")
    for i, chunk in enumerate(recursive_chunks, 1):
        print(f"块 {i}: {chunk.strip()[:50]}...")

    print("\n\n策略 3: Markdown 分割")
    print("-" * 50)
    md_splitter = MarkdownTextSplitter(chunk_size=100, chunk_overlap=10)
    md_chunks = md_splitter.split_text(text)
    print(f"块数: {len(md_chunks)}")
    for i, chunk in enumerate(md_chunks, 1):
        print(f"块 {i}: {chunk.strip()[:50]}...")


# ============ 示例 5：加载和分割 JSON 文件 ============

def example_5_json_loader():
    """示例 5：加载 JSON 文件"""
    print("=" * 70)
    print("示例 5：使用 JSONLoader")
    print("=" * 70)

    # 创建示例 JSON 文件
    import json
    sample_data = [
        {
            "id": 1,
            "title": "Python 入门",
            "content": "Python 是一种简单的编程语言",
            "category": "programming"
        },
        {
            "id": 2,
            "title": "机器学习基础",
            "content": "机器学习是人工智能的一个分支",
            "category": "AI"
        },
        {
            "id": 3,
            "title": "Web 开发指南",
            "content": "Web 开发涉及前端和后端技术",
            "category": "web"
        }
    ]

    with open("sample_data.json", 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)

    # 使用 JSONLoader
    loader = JSONLoader(
        file_path="sample_data.json",
        jq_schema=".[]",  # jq 语法：选择数组中的每个元素
        text_content=False  # 返回完整的字典，不只是文本
    )

    documents = loader.load()

    print(f"\n加载的文档数: {len(documents)}")
    print(f"\n示例文档:")
    for i, doc in enumerate(documents[:2], 1):
        print(f"\n文档 {i}:")
        print(f"  内容: {doc.page_content[:100]}")
        print(f"  元数据: {doc.metadata}")

    # 分割文档
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=10
    )
    split_docs = splitter.split_documents(documents)

    print(f"\n分割后文档数: {len(split_docs)}")

    # 清理
    import os
    os.remove("sample_data.json")


# ============ 示例 6：加载整个目录 ============

def example_6_directory_loader():
    """示例 6：使用 DirectoryLoader 加载目录"""
    print("=" * 70)
    print("示例 6：加载整个目录")
    print("=" * 70)

    # 创建示例目录和文件
    import os
    os.makedirs("sample_docs", exist_ok=True)

    samples = {
        "doc1.txt": "这是第一个文档。它包含一些示例内容。",
        "doc2.txt": "这是第二个文档。它也包含示例内容。",
        "doc3.txt": "这是第三个文档。内容略有不同。"
    }

    for filename, content in samples.items():
        with open(f"sample_docs/{filename}", 'w', encoding='utf-8') as f:
            f.write(content)

    # 使用 DirectoryLoader
    loader = DirectoryLoader(
        path="sample_docs",
        glob="**/*.txt",  # 匹配所有 .txt 文件
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'},
        show_progress=True
    )

    documents = loader.load()

    print(f"\n从目录加载的文档数: {len(documents)}")
    for i, doc in enumerate(documents, 1):
        print(f"\n文档 {i}:")
        print(f"  内容: {doc.page_content}")
        print(f"  来源: {doc.metadata['source']}")

    # 清理
    for filename in samples.keys():
        os.remove(f"sample_docs/{filename}")
    os.rmdir("sample_docs")


# ============ 示例 7：自定义文档处理 ============

def example_7_custom_processing():
    """示例 7：自定义文档处理流程"""
    print("=" * 70)
    print("示例 7：自定义文档处理")
    print("=" * 70)

    # 模拟从多个来源创建文档
    raw_data = [
        {
            "text": "人工智能正在改变世界。",
            "source": "news",
            "date": "2024-01-01",
            "author": "张三"
        },
        {
            "text": "机器学习是AI的子领域。",
            "source": "blog",
            "date": "2024-01-02",
            "author": "李四"
        },
        {
            "text": "深度学习使用神经网络。",
            "source": "paper",
            "date": "2024-01-03",
            "author": "王五"
        }
    ]

    # 创建 Document 对象
    documents = [
        Document(
            page_content=item["text"],
            metadata={
                "source": item["source"],
                "date": item["date"],
                "author": item["author"]
            }
        )
        for item in raw_data
    ]

    # 分割文档
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=30,
        chunk_overlap=5,
        separators=["。", "，", " ", ""]
    )

    split_docs = splitter.split_documents(documents)

    print(f"\n原始文档: {len(documents)} 个")
    print(f"分割后: {len(split_docs)} 个")

    # 按来源分组
    by_source = {}
    for doc in split_docs:
        source = doc.metadata["source"]
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(doc)

    print(f"\n按来源分组:")
    for source, docs in by_source.items():
        print(f"\n{source}: {len(docs)} 个片段")
        for doc in docs[:2]:  # 只显示前两个
            print(f"  - {doc.page_content}")


# 总结：核心概念
"""
【文档加载的核心概念】

1. Document 对象：
   - page_content：文档的文本内容
   - metadata：文档的元数据（来源、日期、作者等）
   - 元数据对于检索和追踪很重要

2. Document Loaders：
   - TextLoader：加载文本文件
   - DirectoryLoader：加载整个目录
   - JSONLoader：加载 JSON 文件
   - CSVLoader：加载 CSV 文件
   - PDFLoader：加载 PDF 文件
   - 还有更多：Web、Markdown、Word 等

3. Text Splitters（分割器）：

   为什么需要分割？
   - LLM 有上下文长度限制
   - 小块更容易检索到相关内容
   - 便于并行处理

   分割器类型：
   a. RecursiveCharacterTextSplitter（推荐）：
      - 按优先级尝试多个分隔符
      - 递归地分割文本
      - 保持段落和句子的完整性

   b. CharacterTextSplitter：
      - 按单个分隔符分割
      - 更简单但可能不够智能

   c. MarkdownTextSplitter：
      - 专门用于 Markdown
      - 理解 Markdown 结构

   d. 其他：
      - CodeTextSplitter（代码）
      - LatexTextSplitter（LaTeX）
      - NotionTextSplitter（Notion）

4. 分割参数：
   - chunk_size：每块的最大字符数
   - chunk_overlap：块之间的重叠（保持上下文）
   - length_function：计算长度的函数
   - separators：分隔符列表（按优先级）

5. 最佳实践：
   - chunk_size：通常 500-2000 字符
   - chunk_overlap：通常是 chunk_size 的 10-20%
   - 根据文本类型选择合适的分隔符
   - 保留元数据信息

【下一步学习】

在 12-vector-storage-rag.py 中，你将学习：
- 如何将分割后的文档转换为向量
- 如何使用向量数据库存储和检索
- 如何构建完整的 RAG（检索增强生成）系统
"""

if __name__ == "__main__":
    example_1_text_loader()
    example_2_text_splitting()
    example_3_split_documents()
    example_4_splitting_strategies()
    example_5_json_loader()
    example_6_directory_loader()
    example_7_custom_processing()
