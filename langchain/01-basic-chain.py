"""
LangChain 学习 01：从直接调用 LLM 到第一个链

本文件目标：理解为什么需要"链"，以及如何创建最简单的链

知识点：
1. 如何直接调用 LLM（不使用链）
2. 什么情况下需要使用链
3. 如何创建最简单的链：Prompt + LLM + Output Parser
4. LCEL（LangChain Expression Language）的基础语法
"""

import sys
import io

# 设置标准输出为 UTF-8 编码，解决 Windows 中文显示问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import get_llm


def example_1_direct_llm_call():
    """
    示例 1：直接调用 LLM（不使用链）

    这是最原始的方式，直接把问题传给 LLM。
    适用于：简单、一次性、不需要复用的场景。
    """
    print("=" * 60)
    print("示例 1：直接调用 LLM")
    print("=" * 50)

    llm = get_llm(temperature=0.7)

    # 直接调用，把问题写死在代码里
    response = llm.invoke("什么是 LangChain？用一句话解释。")
    print(f"回答: {response.content}\n")

    # 问题：如果我要问不同的问题，就需要重复写代码
    response2 = llm.invoke("什么是 Python？用一句话解释。")
    print(f"回答: {response2.content}\n")


def example_2_simple_chain():
    """
    示例 2：创建第一个链

    链 = Prompt Template + LLM + Output Parser

    为什么要用链？
    - 可以复用：一次定义，多次调用
    - 可以参数化：用变量动态生成提示词
    - 可以组合：把多个链串联起来

    LCEL 语法：使用管道操作符 | 串联组件
    """
    print("=" * 60)
    print("示例 2：创建第一个简单的链")
    print("=" * 60)

    # 步骤 1：获取 LLM
    llm = get_llm(temperature=0.7)

    # 步骤 2：创建提示词模板（使用变量）
    # {topic} 会在调用时被替换
    prompt = ChatPromptTemplate.from_template(
        "用一句话解释：{topic}"
    )

    # 步骤 3：创建输出解析器
    # StrOutputParser 把 LLM 的输出转换成字符串
    parser = StrOutputParser()

    # 步骤 4：使用管道操作符 | 串联成链
    # prompt | llm | parser 的执行流程：
    # 1. prompt.invoke({"topic": "Python"}) -> 格式化提示词
    # 2. llm.invoke(formatted_prompt) -> 调用 LLM
    # 3. parser.parse(llm_output) -> 解析输出
    chain = prompt | llm | parser

    # 步骤 5：调用链
    print("调用链，传入不同的参数：\n")
    result1 = chain.invoke({"topic": "Python"})
    print(f"Python: {result1}\n")

    result2 = chain.invoke({"topic": "JavaScript"})
    print(f"JavaScript: {result2}\n")

    result3 = chain.invoke({"topic": "LangChain"})
    print(f"LangChain: {result3}\n")


# 总结：核心概念
"""
【问题：直接调用 LLM 的痛点】

1. 提示词写死在代码里，无法复用
2. 想修改提示词时，需要改代码
3. 无法参数化，不能动态生成提示词

【解决方案：链（Chain）】

链 = Prompt Template + LLM + Output Parser

每个组件的作用：
- Prompt Template：管理提示词，支持变量替换
- LLM：执行推理
- Output Parser：解析输出，转换成想要的格式

【LCEL 管道操作符 |】

chain = prompt | llm | parser

| 是管道操作符，表示数据的流向：
- 先执行 prompt，格式化提示词
- 再执行 llm，调用大模型
- 最后执行 parser，解析输出

等价于：
```python
step1 = prompt.invoke({"topic": "Python"})  # 格式化提示词
step2 = llm.invoke(step1)                    # 调用 LLM
step3 = parser.parse(step2)                  # 解析输出
```

【为什么需要链？】

1. 复用性：定义一次，多次调用
2. 可维护性：修改提示词只需改模板
3. 可组合性：可以把多个链串联成更复杂的流程
4. 标准化：统一的接口，易于测试和调试

【下一步学习】

在 02-prompt-template.py 中，你将学习：
- 如何创建更复杂的提示词模板
- 如何使用多个变量
- 如何使用系统消息和用户消息
"""

if __name__ == "__main__":
    # 按顺序运行示例
    example_1_direct_llm_call()
    example_2_simple_chain()
