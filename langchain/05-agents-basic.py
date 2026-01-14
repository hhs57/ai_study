"""
LangChain 学习 05：代理（Agents）和工具（Tools）

知识点：
1. 什么是 Agent：可以根据用户输入自主决定使用哪些工具
2. 如何创建自定义 Tool
3. ReAct Agent：推理（Reasoning）+ 行动（Acting）
4. Agent 的核心组件：LLM、Tools、Prompt Template
5. 不同类型的 Agent：Zero-shot, ReAct, Self-ask
"""

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent, create_tool_calling_agent
from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_community.utilities import SerpAPIWrapper
from dotenv import load_dotenv
import math

load_dotenv()


# ============ 自定义工具定义 ============

@tool
def calculator(expression: str) -> str:
    """
    执行数学计算。
    输入应该是一个数学表达式字符串，比如 '2 + 2' 或 '10 * 5'。
    """
    try:
        # 注意：实际应用中需要更安全的方式执行计算
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def get_word_length(word: str) -> str:
    """
    计算一个单词的长度。
    输入应该是一个单词或短语。
    """
    return f"'{word}' 的长度是 {len(word)} 个字符"


@tool
def temperature_converter(celsius: float) -> str:
    """
    将摄氏度转换为华氏度。
    输入是摄氏度温度值。
    """
    fahrenheit = (celsius * 9/5) + 32
    return f"{celsius}°C = {fahrenheit}°F"


def example_1_basic_tools():
    """示例 1：直接使用工具（不使用 Agent）"""
    print("=" * 60)
    print("示例 1：直接使用工具")
    print("=" * 60)

    # 工具可以直接调用
    result1 = calculator.invoke("15 + 27")
    print(f"计算器: {result1}")

    result2 = get_word_length.invoke("Hello World")
    print(f"单词长度: {result2}")

    result3 = temperature_converter.invoke(25)
    print(f"温度转换: {result3}\n")


def example_2_simple_react_agent():
    """示例 2：创建一个简单的 ReAct Agent"""
    print("=" * 60)
    print("示例 2：ReAct Agent - 推理和行动")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # 定义工具列表
    tools = [
        calculator,
        get_word_length,
        temperature_converter
    ]

    # ReAct Agent 的提示词模板
    template = """你是一个有帮助的助手。你可以使用以下工具：

{tools}

使用以下格式：

Question: 用户的问题
Thought: 你应该思考该做什么
Action: 要使用的工具名称，必须是 [{tool_names}] 中的一个
Action Input: 工具的输入
Observation: 工具返回的结果
... (这个 Thought/Action/Action Input/Observation 可以重复多次)
Thought: 我现在知道最终答案了
Final Answer: 对原始问题的最终回答

开始！

Question: {input}
Thought: {agent_scratchpad}"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={
            "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
            "tool_names": ", ".join([tool.name for tool in tools])
        }
    )

    # 创建 Agent
    agent = create_react_agent(
        llm=llm,
        prompt=prompt,
        tools=tools
    )

    # 创建 AgentExecutor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # 显示推理过程
        handle_parsing_errors=True  # 处理解析错误
    )

    # 测试 Agent
    print("\n问题 1: 25 加 35 等于多少？")
    result1 = agent_executor.invoke({"input": "25 加 35 等于多少？"})
    print(f"最终答案: {result1['output']}\n")

    print("问题 2: 'Artificial Intelligence' 这个词有多少个字符？")
    result2 = agent_executor.invoke({"input": "'Artificial Intelligence' 这个词有多少个字符？"})
    print(f"最终答案: {result2['output']}\n")


def example_3_multi_step_reasoning():
    """示例 3：多步推理 Agent"""
    print("=" * 60)
    print("示例 3：多步推理 - 需要使用多个工具")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    tools = [
        calculator,
        get_word_length,
        temperature_converter
    ]

    # 使用更简化的提示词
    prompt = PromptTemplate(
        template="""回答以下问题，可以使用的工具：
{tools}

工具名称: {tool_names}

Question: {input}
Thought: {agent_scratchpad}""",
        input_variables=["input", "agent_scratchpad"],
        partial_variables={
            "tools": "\n".join([f"- {tool.name}: {tool.description}" for tool in tools]),
            "tool_names": ", ".join([tool.name for tool in tools])
        }
    )

    agent = create_react_agent(llm=llm, prompt=prompt, tools=tools)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5  # 限制最大迭代次数
    )

    # 复杂问题：需要多个步骤
    print("问题: 如果 25 摄氏度转换成华氏度，然后再加上 100，结果是多少？")
    result = agent_executor.invoke({
        "input": "如果 25 摄氏度转换成华氏度，然后再加上 100，结果是多少？"
    })
    print(f"\n最终答案: {result['output']}\n")


def example_4_custom_tool_with_logic():
    """示例 4：带复杂逻辑的自定义工具"""
    print("=" * 60)
    print("示例 4：自定义复杂工具")
    print("=" * 60)

    @tool
    def analyze_text(text: str) -> str:
        """
        分析文本的统计信息，包括字符数、单词数、句子数。
        输入应该是要分析的文本。
        """
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = text.count('.') + text.count('!') + text.count('?')

        return f"""文本分析结果：
- 字符数: {char_count}
- 单词数: {word_count}
- 句子数: {sentence_count}"""

    @tool
    def text_case_converter(text: str, operation: str) -> str:
        """
        转换文本的大小写。
        参数：
        - text: 要转换的文本
        - operation: 操作类型（'upper', 'lower', 'title'）
        """
        if operation == "upper":
            return text.upper()
        elif operation == "lower":
            return text.lower()
        elif operation == "title":
            return text.title()
        else:
            return f"未知操作: {operation}"

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    tools = [analyze_text, text_case_converter]

    prompt = PromptTemplate(
        template="""使用以下工具回答问题：
{tools}

Question: {input}
Thought: {agent_scratchpad}""",
        input_variables=["input", "agent_scratchpad"],
        partial_variables={
            "tools": "\n".join([f"- {tool.name}: {tool.description}" for tool in tools]),
            "tool_names": ", ".join([tool.name for tool in tools])
        }
    )

    agent = create_react_agent(llm=llm, prompt=prompt, tools=tools)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    print("问题: 分析这段文本并转换成大写：'hello world. this is a test.'")
    result = agent_executor.invoke({
        "input": "分析这段文本 'hello world. this is a test.' 的统计信息，然后把它转换成大写"
    })
    print(f"\n最终答案: {result['output']}\n")


def example_5_agent_with_memory():
    """示例 5：带记忆的 Agent"""
    print("=" * 60)
    print("示例 5：带记忆的 Agent（需要在 LangChain 中配置）")
    print("=" * 60)
    print("注意：完整的记忆 Agent 需要更复杂的配置\n"
          "这里展示基本概念：Agent 可以记住之前的对话历史\n")


# 总结：核心概念
"""
【Agent 的核心思想】
Agent = LLM + 工具（Tools）+ 推理（Reasoning）

不是简单地调用工具，而是：
1. 理解用户的请求
2. 决定是否需要使用工具
3. 选择合适的工具
4. 观察工具的输出
5. 继续推理或给出最终答案

【工具（Tools）】
工具是 Agent 可以使用的外部功能：
- API 调用（搜索、天气、新闻等）
- 数据库查询
- 文件操作
- 数学计算
- 任何自定义功能

定义工具的方式：
1. 使用 @tool 装饰器（推荐）
2. 使用 Tool 类手动创建
3. 实现 BaseTool 接口（最灵活）

【ReAct 模式】
ReAct = Reasoning + Acting

循环过程：
1. Thought（思考）：分析当前情况
2. Action（行动）：选择并执行工具
3. Observation（观察）：查看工具结果
4. 重复直到得到答案

【Agent 的类型】
1. ReAct Agent：最常用，推理+行动
2. Zero-shot Agent：不需要示例，直接推理
3. Few-shot Agent：需要示例来学习
4. Self-ask Agent：自问自答，分解问题
5. OpenAI Functions Agent：使用 OpenAI 的函数调用

【使用场景】
- 需要多步推理的问题
- 需要访问外部数据的任务
- 需要执行操作的场景
- 复杂的决策过程

【注意事项】
1. 工具的描述要清晰，LLM 依赖描述来选择工具
2. 限制 max_iterations 避免无限循环
3. 设置 handle_parsing_errors 处理解析错误
4. 敏感操作要谨慎（如数据库写入、删除等）
"""

if __name__ == "__main__":
    example_1_basic_tools()
    example_2_simple_react_agent()
    example_3_multi_step_reasoning()
    example_4_custom_tool_with_logic()
    example_5_agent_with_memory()
