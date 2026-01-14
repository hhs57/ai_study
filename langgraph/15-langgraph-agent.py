"""
LangGraph 学习 05：构建智能体（Agent）

知识点：
1. 什么是 Agent：使用 LLM 决定行动的自主系统
2. Agent 的核心循环：思考 -> 行动 -> 观察
3. 如何定义工具（Tools）
4. 如何构建 ReAct Agent
5. 如何让 Agent 使用工具解决问题
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, Sequence
import operator
from dotenv import load_dotenv

load_dotenv()


# ============ 定义工具 ============

@tool
def calculator(expression: str) -> str:
    """
    执行数学计算。
    输入应该是一个数学表达式字符串，比如 '2 + 2' 或 '10 * 5'。
    """
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def search_database(query: str) -> str:
    """
    搜索模拟数据库中的信息。
    输入是搜索关键词。
    """
    # 模拟数据库
    database = {
        "python": "Python 是一种高级编程语言",
        "javascript": "JavaScript 是一种脚本语言",
        "java": "Java 是一种面向对象的编程语言",
        "langchain": "LangChain 是一个 LLM 应用框架",
        "langgraph": "LangGraph 是用于构建有状态应用的框架"
    }

    query_lower = query.lower()
    results = [v for k, v in database.items() if query_lower in k.lower()]

    if results:
        return "搜索结果:\n" + "\n".join(f"- {r}" for r in results)
    else:
        return f"未找到关于 '{query}' 的信息"


@tool
def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息。
    输入是城市名称。
    """
    # 模拟天气数据
    weather_data = {
        "北京": "晴天，温度 25°C",
        "上海": "多云，温度 28°C",
        "广州": "雨天，温度 30°C",
        "深圳": "晴天，温度 32°C"
    }

    return weather_data.get(city, f"未找到 {city} 的天气信息")


@tool
def get_time() -> str:
    """
    获取当前时间。
    不需要任何输入参数。
    """
    from datetime import datetime
    now = datetime.now()
    return f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}"


# ============ 示例 1：最简单的 Agent ============

def example_1_simple_agent():
    """示例 1：使用单个工具的 Agent"""
    print("=" * 70)
    print("示例 1：简单 Agent - 计算助手")
    print("=" * 70)

    # 定义状态
    class AgentState(TypedDict):
        messages: Annotated[Sequence[str], operator.add]
        tool_calls: list[dict]
        step_count: int

    # 初始化 LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Agent 节点：决定是否使用工具
    def agent_node(state: AgentState) -> AgentState:
        """Agent 主节点"""
        print(f"\n  [Agent] 第 {state['step_count'] + 1} 步")
        print("  [Agent] 分析用户请求...")

        last_message = state["messages"][-1] if state["messages"] else ""

        # 简单的规则：如果包含"计算"，使用计算器
        if "计算" in last_message or any(op in last_message for op in ["+", "-", "*", "/"]):
            print("  [Agent] 识别为计算请求")
            # 提取表达式（简化处理）
            expression = last_message.split("计算")[-1].strip()
            result = calculator.invoke(expression)
            return {
                "messages": [f"工具结果: {result}"],
                "step_count": state["step_count"] + 1
            }
        else:
            print("  [Agent] 直接回答")
            response = f"我不理解你的请求: {last_message}"
            return {
                "messages": [response],
                "step_count": state["step_count"] + 1
            }

    # 构建图
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)

    compiled_graph = graph.compile()

    # 执行
    print("\n用户请求: 计算 25 + 17")
    result = compiled_graph.invoke({
        "messages": ["计算 25 + 17"],
        "tool_calls": [],
        "step_count": 0
    })

    print(f"\n最终结果: {result['messages'][-1]}\n")


# ============ 示例 2：多工具 Agent ============

def example_2_multi_tool_agent():
    """示例 2：可以使用多个工具的 Agent"""
    print("=" * 70)
    print("示例 2：多工具 Agent - 智能助手")
    print("=" * 70)

    class MultiToolAgentState(TypedDict):
        messages: Annotated[Sequence[str], operator.add]
        should_continue: bool

    tools = [calculator, search_database, get_weather, get_time]

    def agent_node(state: MultiToolAgentState) -> MultiToolAgentState:
        """Agent 节点：分析并决定使用哪个工具"""
        last_message = state["messages"][-1]
        print(f"\n  [Agent] 分析: {last_message}")

        # 简单的意图识别
        result = None
        should_continue = False

        # 检查是否需要使用工具
        if any(op in last_message for op in ["+", "-", "*", "/"]) or "计算" in last_message:
            print("  [Agent] 使用计算器")
            # 提取表达式
            expr = last_message.split("计算")[-1].strip() if "计算" in last_message else last_message
            result = calculator.invoke(expr)

        elif "天气" in last_message:
            print("  [Agent] 使用天气查询")
            # 提取城市名（简化）
            city = "北京"  # 默认
            for c in ["北京", "上海", "广州", "深圳"]:
                if c in last_message:
                    city = c
                    break
            result = get_weather.invoke(city)

        elif "时间" in last_message or "几点" in last_message:
            print("  [Agent] 使用时间查询")
            result = get_time.invoke()

        elif any(kw in last_message for kw in ["python", "javascript", "java", "langchain", "langgraph"]):
            print("  [Agent] 使用数据库搜索")
            result = search_database.invoke(last_message)

        if result:
            return {
                "messages": [f"工具返回: {result}"],
                "should_continue": False
            }
        else:
            return {
                "messages": [f"抱歉，我无法处理这个请求: {last_message}"],
                "should_continue": False
            }

    # 构建图
    graph = StateGraph(MultiToolAgentState)
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)

    compiled_graph = graph.compile()

    # 测试不同请求
    test_requests = [
        "计算 15 * 8",
        "北京的天气怎么样",
        "现在几点了",
        "什么是 Python？"
    ]

    for request in test_requests:
        print(f"\n{'='*60}")
        print(f"用户请求: {request}")
        result = compiled_graph.invoke({
            "messages": [request],
            "should_continue": True
        })
        print(f"\nAgent 回复: {result['messages'][-1]}")
    print()


# ============ 示例 3：带循环的 Agent ============

def example_3_react_agent():
    """示例 3：ReAct 模式的 Agent（推理 + 行动）"""
    print("=" * 70)
    print("示例 3：ReAct Agent - 多步推理")
    print("=" * 70)

    class ReActAgentState(TypedDict):
        question: str
        thoughts: list[str]
        actions: list[str]
        observations: list[str]
        answer: str
        step_count: int
        max_steps: int

    tools = {
        "calculator": calculator,
        "search": search_database,
        "weather": get_weather
    }

    def think_node(state: ReActAgentState) -> ReActAgentState:
        """思考节点：分析问题并规划行动"""
        step = state["step_count"] + 1
        print(f"\n  [思考] 第 {step} 步")

        question = state["question"]
        thoughts = state["thoughts"]

        # 简化的推理逻辑
        if step == 1:
            thought = f"问题: {question}。我需要理解用户想要什么。"
        elif step == 2:
            thought = "我需要使用工具来获取信息。"
        elif step == 3:
            thought = "基于工具结果，我可以给出答案了。"
        else:
            thought = "我已经有足够的信息回答问题。"

        print(f"  [思考] {thought}")

        return {
            "thoughts": thoughts + [thought],
            "step_count": step
        }

    def act_node(state: ReActAgentState) -> ReActAgentState:
        """行动节点：执行工具"""
        print("\n  [行动] 执行工具...")

        question = state["question"]
        actions = state["actions"]
        observations = state["observations"]

        # 决定使用哪个工具
        tool_name = None
        tool_input = question

        if "计算" in question or any(op in question for op in ["+", "-", "*", "/"]):
            tool_name = "calculator"
            tool_input = question.split("计算")[-1].strip() if "计算" in question else question
        elif "天气" in question:
            tool_name = "weather"
            for city in ["北京", "上海", "广州", "深圳"]:
                if city in question:
                    tool_input = city
                    break
        else:
            tool_name = "search"

        # 执行工具
        tool = tools[tool_name]
        result = tool.invoke(tool_input)

        action = f"使用工具 {tool_name}，输入: {tool_input}"
        observation = f"工具输出: {result}"

        print(f"  [行动] {action}")
        print(f"  [观察] {observation}")

        return {
            "actions": actions + [action],
            "observations": observations + [observation]
        }

    def should_continue(state: ReActAgentState) -> str:
        """决定是继续还是结束"""
        if state["step_count"] < state["max_steps"]:
            return "continue"
        else:
            return "finish"

    def answer_node(state: ReActAgentState) -> ReActAgentState:
        """答案节点：生成最终答案"""
        print("\n  [答案] 生成最终答案...")

        answer = f"基于 {state['step_count']} 步思考，我的答案是：\n"
        answer += "\n".join(f"- {obs}" for obs in state["observations"])

        return {"answer": answer}

    # 构建图
    graph = StateGraph(ReActAgentState)

    graph.add_node("think", think_node)
    graph.add_node("act", act_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("think")

    # 思考 -> 行动
    graph.add_edge("think", "act")

    # 行动后决定下一步
    graph.add_conditional_edges(
        "act",
        should_continue,
        {
            "continue": "think",  # 继续循环
            "finish": "answer"  # 生成答案
        }
    )

    graph.add_edge("answer", END)

    compiled_graph = graph.compile()

    # 执行
    print("\n问题: 计算 25 + 17 并告诉我结果")
    result = compiled_graph.invoke({
        "question": "计算 25 + 17",
        "thoughts": [],
        "actions": [],
        "observations": [],
        "answer": "",
        "step_count": 0,
        "max_steps": 3
    })

    print(f"\n{result['answer']}\n")


# ============ 示例 4：使用 LangGraph 的预构建 Agent ============

def example_4_prebuilt_agent():
    """示例 4：使用 LangGraph 的预构建工具"""
    print("=" * 70)
    print("示例 4：LangGraph 预构建 Agent")
    print("=" * 70)

    print("""
    LangGraph 提供了预构建的 Agent 组件，包括：
    - ToolNode：自动执行工具调用
    - create_react_agent：快速创建 ReAct Agent

    这可以大大简化 Agent 的构建过程。

    实际使用示例：
    ```python
    from langgraph.prebuilt import create_react_agent

    # 定义工具
    tools = [calculator, search_database]

    # 创建 Agent
    agent = create_react_agent(llm, tools)

    # 执行
    response = agent.invoke({"messages": [("user", "计算 2+2")]})
    ```

    注意：实际使用需要配置 OpenAI 函数调用
    """)


# 总结：核心概念
"""
【Agent 的核心概念】

1. Agent 的定义：
   - 使用 LLM 作为"大脑"的自主系统
   - 可以自主决定使用哪些工具
   - 通过推理和行动循环解决问题

2. ReAct 模式：
   ReAct = Reasoning（推理）+ Acting（行动）

   循环过程：
   ```
   思考(Thought) -> 决定行动
      |
   行动(Action) -> 执行工具
      |
   观察(Observation) -> 获取结果
      |
   回到思考，直到得到答案
   ```

3. Agent 的关键组件：

   a. 状态（State）：
      - 问题/任务
      - 思考过程
      - 行动历史
      - 观察结果
      - 最终答案

   b. 节点（Nodes）：
      - Agent 节点：LLM 推理
      - Tool 节点：执行工具
      - 条件判断：是否继续

   c. 工具（Tools）：
      - 搜索
      - 计算
      - API 调用
      - 数据库查询
      - 任何自定义功能

   d. 循环（Loop）：
      - 通过条件边实现
      - 直到得到满意答案
      - 或达到最大迭代次数

4. Agent vs 链（Chain）：

   链（Chain）：
   - 预定义的固定流程
   - 按顺序执行
   - 适合简单任务

   Agent：
   - 动态决策流程
   - 自主选择工具
   - 适合复杂任务
   - 可以处理不确定性

5. 构建 Agent 的步骤：

   步骤 1：定义工具
   ```python
   @tool
   def my_tool(input: str) -> str:
       return result
   ```

   步骤 2：定义状态
   ```python
   class AgentState(TypedDict):
       messages: Annotated[Sequence, operator.add]
       # 其他必要字段...
   ```

   步骤 3：创建节点
   ```python
   def agent_node(state):
       # LLM 推理
       # 决定使用什么工具
       pass

   def tool_node(state):
       # 执行工具
       pass
   ```

   步骤 4：构建图
   ```python
   graph = StateGraph(AgentState)
   graph.add_node("agent", agent_node)
   graph.add_node("tools", tool_node)
   # 添加边...
   ```

   步骤 5：添加循环
   ```python
   graph.add_conditional_edges(
       "agent",
       should_continue,
       {"continue": "agent", "end": END}
   )
   ```

6. 最佳实践：

   a. 工具设计：
      - 描述要清晰
      - 输入输出明确
      - 错误处理完善

   b. 状态管理：
      - 追踪思考过程
      - 记录工具使用
      - 保存中间结果

   c. 循环控制：
      - 设置最大迭代次数
      - 提供明确的退出条件
      - 避免无限循环

   d. 性能优化：
      - 减少不必要的 LLM 调用
      - 并行执行独立工具
      - 缓存重复查询

7. 应用场景：
   - 聊天机器人（带工具）
   - 研究助手（搜索 + 分析）
   - 数据分析（查询 + 计算）
   - 任务自动化（多步骤操作）
   - 编程助手（代码执行 + 调试）
"""

if __name__ == "__main__":
    example_1_simple_agent()
    example_2_multi_tool_agent()
    example_3_react_agent()
    example_4_prebuilt_agent()
