"""
LangChain 学习 04：对话记忆（Memory）

知识点：
1. 为什么需要 Memory：LLM 本身是无状态的
2. ConversationBufferMemory：保存所有对话历史
3. ConversationBufferWindowMemory：只保存最近的 k 轮对话
4. ConversationSummaryMemory：总结历史对话
5. ConversationKGMemory（知识图谱）：提取实体和关系
6. 使用 LCEL 集成 Memory
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory
)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()


def example_1_no_memory():
    """示例 1：没有 Memory 的情况"""
    print("=" * 60)
    print("示例 1：无状态对话（LLM 默认行为）")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # 第一轮对话
    response1 = llm.invoke([HumanMessage(content="我的名字是张三")])
    print(f"用户: 我的名字是张三")
    print(f"AI: {response1.content}\n")

    # 第二轮对话
    response2 = llm.invoke([HumanMessage(content="我叫什么名字？")])
    print(f"用户: 我叫什么名字？")
    print(f"AI: {response2.content}\n")
    print("注意：AI 不记得之前对话的内容！\n")


def example_2_conversation_buffer_memory():
    """示例 2：ConversationBufferMemory - 保存所有历史"""
    print("=" * 60)
    print("示例 2：ConversationBufferMemory - 缓冲所有对话")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # 创建记忆组件
    memory = ConversationBufferMemory(
        return_messages=True  # 返回消息对象而不是字符串
    )

    # 创建对话链
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )

    # 第一轮对话
    response1 = conversation.predict(input="我的名字是张三，我是一名软件工程师")
    print(f"用户: 我的名字是张三，我是一名软件工程师")
    print(f"AI: {response1}\n")

    # 第二轮对话
    response2 = conversation.predict(input="我叫什么名字？")
    print(f"用户: 我叫什么名字？")
    print(f"AI: {response2}\n")

    # 第三轮对话
    response3 = conversation.predict(input="我的工作是什么？")
    print(f"用户: 我的工作是什么？")
    print(f"AI: {response3}\n")

    # 查看保存的历史
    print("保存的对话历史:")
    print(memory.load_memory_variables({})['history'][:200])
    print("...\n")


def example_3_buffer_window_memory():
    """示例 3：ConversationBufferWindowMemory - 滑动窗口"""
    print("=" * 60)
    print("示例 3：ConversationBufferWindowMemory - 只保留最近对话")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # 只保留最近 2 轮对话（即 2 个用户输入和 2 个 AI 回复）
    memory = ConversationBufferWindowMemory(
        k=2,  # 保留最近 k 轮对话
        return_messages=True
    )

    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )

    # 进行多轮对话
    conversation.predict(input="我喜欢苹果")
    print("第1轮: 用户说喜欢苹果")

    conversation.predict(input="我还喜欢香蕉")
    print("第2轮: 用户说还喜欢香蕉")

    conversation.predict(input="我也喜欢橙子")
    print("第3轮: 用户说也喜欢橙子")

    response = conversation.predict(input="我之前说喜欢什么水果？")
    print(f"第4轮: 用户问喜欢什么水果")
    print(f"AI: {response}\n")
    print("注意：由于 k=2，AI 只记得最近 2 轮对话（香蕉和橙子），不记得第一轮的苹果\n")


def example_4_conversation_summary_memory():
    """示例 4：ConversationSummaryMemory - 总结历史"""
    print("=" * 60)
    print("示例 4：ConversationSummaryMemory - 自动总结历史对话")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # 记忆组件会自动总结之前的对话
    memory = ConversationSummaryMemory(
        llm=llm,
        return_messages=True
    )

    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )

    # 长对话场景
    conversation.predict(input="我在学习 Python 编程")
    print("用户: 我在学习 Python 编程")

    conversation.predict(input="Python 是一种高级编程语言")
    print("AI: 介绍了 Python")

    conversation.predict(input="我还在学习机器学习")
    print("用户: 我还在学习机器学习")

    conversation.predict(input="机器学习需要数学基础")
    print("AI: 提到机器学习需要数学")

    conversation.predict(input="我最近在学什么？")
    response = conversation.predict(input="我最近在学什么？")
    print(f"用户: 我最近在学什么？")
    print(f"AI: {response}\n")

    print("当前记忆总结:")
    print(memory.load_memory_variables({})['history'])
    print("\n")


def example_5_memory_with_lcel():
    """示例 5：在 LCEL 中使用 Memory"""
    print("=" * 60)
    print("示例 5：使用 LCEL 手动管理记忆")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # 创建提示词模板，包含历史消息的占位符
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="你是一个友好的助手。"),
        # 这里的 messages 会包含历史对话
        ("placeholder", "{history}"),
        HumanMessage(content="{input}")
    ])

    # 初始化历史记录
    history = []

    def chat(message, history):
        """带记忆的聊天函数"""
        # 构建完整的消息链
        chain = prompt | llm | StrOutputParser()

        response = chain.invoke({
            "input": message,
            "history": history
        })

        # 更新历史
        history.append(HumanMessage(content=message))
        history.append(AIMessage(content=response))

        return response

    # 模拟对话
    response1 = chat("我的名字是李四", history)
    print(f"用户: 我的名字是李四")
    print(f"AI: {response1}\n")

    response2 = chat("我叫什么名字？", history)
    print(f"用户: 我叫什么名字？")
    print(f"AI: {response2}\n")

    response3 = chat("我的名字有几个字？", history)
    print(f"用户: 我的名字有几个字？")
    print(f"AI: {response3}\n")


def example_6_memory_with_variables():
    """示例 6：在 Memory 中存储额外信息"""
    print("=" * 60)
    print("示例 6：在 Memory 中保存用户信息")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # 创建记忆
    memory = ConversationBufferMemory(
        return_messages=True
    )

    # 可以手动添加信息到记忆中
    memory.save_context(
        {"input": ""},
        {"output": "用户张三是一名软件工程师，喜欢编程和阅读。"}
    )

    conversation = ConversationChain(
        llm=llm,
        memory=memory
    )

    response = conversation.predict(input="我的职业是什么？")
    print(f"用户: 我的职业是什么？")
    print(f"AI: {response}\n")


# 总结：核心概念
"""
【为什么需要 Memory】
- LLM 本身是无状态的，每次调用都是独立的
- 要实现多轮对话，需要手动传递历史消息
- Memory 组件帮助我们自动管理对话历史

【Memory 的类型】
1. ConversationBufferMemory：
   - 保存所有对话历史
   - 优点：信息完整
   - 缺点：Token 消耗大，可能超出上下文限制

2. ConversationBufferWindowMemory：
   - 只保留最近 k 轮对话
   - 优点：控制 Token 消耗
   - 缺点：丢失早期信息

3. ConversationSummaryMemory：
   - 自动总结历史对话
   - 优点：节省 Token，保留关键信息
   - 缺点：可能丢失细节

【Memory 的使用方式】
1. ConversationChain：
   - 最简单，自动集成 Memory
   - 适合快速原型

2. 手动管理（LCEL）：
   - 更灵活，完全控制
   - 适合复杂场景

3. 自定义 Memory：
   - 可以根据需求扩展 Memory 类

【选择 Memory 的建议】
- 短对话（< 10 轮）：ConversationBufferMemory
- 中等对话（10-50 轮）：ConversationBufferWindowMemory
- 长对话（> 50 轮）：ConversationSummaryMemory
- 需要精确信息：ConversationBufferMemory + 手动总结
"""

if __name__ == "__main__":
    example_1_no_memory()
    example_2_conversation_buffer_memory()
    example_3_buffer_window_memory()
    example_4_conversation_summary_memory()
    example_5_memory_with_lcel()
    example_6_memory_with_variables()
