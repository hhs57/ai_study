"""
LangGraph 学习 03：条件边（Conditional Edges）

知识点：
1. 什么是条件边：根据状态动态决定下一个节点
2. add_conditional_edges：添加条件边
3. 条件函数：返回下一个节点的名称
4. 构建分支流程和决策树
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence, Literal
import operator
from dotenv import load_dotenv

load_dotenv()


# ============ 示例 1：简单的条件分支 ============

def example_1_simple_conditional():
    """示例 1：根据数值大小决定分支"""
    print("=" * 70)
    print("示例 1：简单条件边 - 根据分数决定处理方式")
    print("=" * 70)

    class ScoreState(TypedDict):
        score: int
        result: str

    def grade_node(state: ScoreState) -> ScoreState:
        """评分节点"""
        score = state["score"]
        print(f"  [评分节点] 分数: {score}")
        return state

    def route_condition(state: ScoreState) -> Literal["pass_node", "fail_node"]:
        """
        条件函数：根据分数决定下一个节点
        返回值必须是下一个节点的名称
        """
        score = state["score"]
        if score >= 60:
            return "pass_node"
        else:
            return "fail_node"

    def pass_node(state: ScoreState) -> ScoreState:
        """通过节点"""
        print("  [通过节点] 成绩及格！")
        return {"result": "恭喜，你及格了！"}

    def fail_node(state: ScoreState) -> ScoreState:
        """失败节点"""
        print("  [失败节点] 成绩不及格")
        return {"result": "很遗憾，你需要补考。"}

    # 构建图
    graph = StateGraph(ScoreState)

    # 添加节点
    graph.add_node("grade", grade_node)
    graph.add_node("pass_node", pass_node)
    graph.add_node("fail_node", fail_node)

    # 设置入口
    graph.set_entry_point("grade")

    # 添加条件边
    # 从 grade 节点出发，根据 route_condition 的结果选择下一个节点
    graph.add_conditional_edges(
        "grade",  # 源节点
        route_condition,  # 条件函数
        {
            "pass_node": "pass_node",  # 返回 "pass_node" 时去往 pass_node
            "fail_node": "fail_node"   # 返回 "fail_node" 时去往 fail_node
        }
    )

    # 两个分支都结束
    graph.add_edge("pass_node", END)
    graph.add_edge("fail_node", END)

    compiled_graph = graph.compile()

    # 测试不同分数
    print("\n测试 1: 分数 75")
    result1 = compiled_graph.invoke({"score": 75, "result": ""})
    print(f"结果: {result1['result']}\n")

    print("测试 2: 分数 45")
    result2 = compiled_graph.invoke({"score": 45, "result": ""})
    print(f"结果: {result2['result']}\n")


# ============ 示例 2：多分支条件 ============

def example_2_multiple_branches():
    """示例 2：三个或更多分支"""
    print("=" * 70)
    print("示例 2：多分支条件 - 根据天气决定活动")
    print("=" * 70)

    class WeatherState(TypedDict):
        weather: str  # sunny, rainy, cloudy, snowy
        activity: str

    def check_weather(state: WeatherState) -> WeatherState:
        """检查天气"""
        print(f"  [天气节点] 当前天气: {state['weather']}")
        return state

    def route_by_weather(state: WeatherState) -> Literal["sunny", "rainy", "cloudy", "snowy"]:
        """根据天气决定活动"""
        weather = state["weather"]
        return weather

    def sunny_activity(state: WeatherState) -> WeatherState:
        """晴天活动"""
        print("  [晴天活动] 去公园野餐")
        return {"activity": "去公园野餐"}

    def rainy_activity(state: WeatherState) -> WeatherState:
        """雨天活动"""
        print("  [雨天活动] 在家看电影")
        return {"activity": "在家看电影"}

    def cloudy_activity(state: WeatherState) -> WeatherState:
        """多云活动"""
        print("  [多云活动] 散步和购物")
        return {"activity": "散步和购物"}

    def snowy_activity(state: WeatherState) -> WeatherState:
        """雪天活动"""
        print("  [雪天活动] 堆雪人！")
        return {"activity": "堆雪人"}

    # 构建图
    graph = StateGraph(WeatherState)

    graph.add_node("check_weather", check_weather)
    graph.add_node("sunny", sunny_activity)
    graph.add_node("rainy", rainy_activity)
    graph.add_node("cloudy", cloudy_activity)
    graph.add_node("snowy", snowy_activity)

    graph.set_entry_point("check_weather")

    # 添加多分支条件边
    graph.add_conditional_edges(
        "check_weather",
        route_by_weather,
        {
            "sunny": "sunny",
            "rainy": "rainy",
            "cloudy": "cloudy",
            "snowy": "snowy"
        }
    )

    graph.add_edge("sunny", END)
    graph.add_edge("rainy", END)
    graph.add_edge("cloudy", END)
    graph.add_edge("snowy", END)

    compiled_graph = graph.compile()

    # 测试不同天气
    for weather in ["sunny", "rainy", "cloudy", "snowy"]:
        print(f"\n测试天气: {weather}")
        result = compiled_graph.invoke({"weather": weather, "activity": ""})
        print(f"推荐活动: {result['activity']}")
    print()


# ============ 示例 3：基于 LLM 输出的条件路由 ============

def example_3_llm_conditional():
    """示例 3：使用 LLM 决定路由"""
    print("=" * 70)
    print("示例 3：LLM 条件路由 - 分析用户意图")
    print("=" * 70)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    class IntentState(TypedDict):
        messages: Annotated[Sequence[str], operator.add]
        intent: str
        response: str

    def classify_intent(state: IntentState) -> IntentState:
        """分类用户意图"""
        user_input = state["messages"][-1]
        print(f"  [意图分类] 分析用户输入: {user_input}")

        # 简单的意图分类逻辑
        user_input_lower = user_input.lower()
        if any(word in user_input_lower for word in ["天气", "温度", "下雨", "晴天"]):
            intent = "weather"
        elif any(word in user_input_lower for word in ["时间", "几点", "日期"]):
            intent = "time"
        elif any(word in user_input_lower for word in ["计算", "加", "减", "乘", "除"]):
            intent = "calculator"
        else:
            intent = "general"

        print(f"  [意图分类] 识别意图: {intent}")
        return {"intent": intent}

    def route_by_intent(state: IntentState) -> Literal["weather_handler", "time_handler", "calculator_handler", "general_handler"]:
        """根据意图路由"""
        return state["intent"]

    def weather_handler(state: IntentState) -> IntentState:
        """天气处理"""
        print("  [天气处理] 处理天气查询")
        return {"response": "今天晴天，温度 25°C"}

    def time_handler(state: IntentState) -> IntentState:
        """时间处理"""
        print("  [时间处理] 处理时间查询")
        return {"response": "现在是 2024年1月1日 12:00"}

    def calculator_handler(state: IntentState) -> IntentState:
        """计算器处理"""
        print("  [计算器处理] 处理计算请求")
        return {"response": "计算功能开发中..."}

    def general_handler(state: IntentState) -> IntentState:
        """通用处理"""
        print("  [通用处理] 处理一般请求")
        return {"response": "你好！有什么可以帮助你的吗？"}

    # 构建图
    graph = StateGraph(IntentState)

    graph.add_node("classify", classify_intent)
    graph.add_node("weather_handler", weather_handler)
    graph.add_node("time_handler", time_handler)
    graph.add_node("calculator_handler", calculator_handler)
    graph.add_node("general_handler", general_handler)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "weather": "weather_handler",
            "time": "time_handler",
            "calculator": "calculator_handler",
            "general": "general_handler"
        }
    )

    graph.add_edge("weather_handler", END)
    graph.add_edge("time_handler", END)
    graph.add_edge("calculator_handler", END)
    graph.add_edge("general_handler", END)

    compiled_graph = graph.compile()

    # 测试不同意图
    test_inputs = [
        "今天天气怎么样？",
        "现在几点了？",
        "帮我计算 2+2",
        "你好"
    ]

    for user_input in test_inputs:
        print(f"\n用户输入: {user_input}")
        result = compiled_graph.invoke({
            "messages": [user_input],
            "intent": "",
            "response": ""
        })
        print(f"系统回复: {result['response']}")
    print()


# ============ 示例 4：复杂的决策树 ============

def example_4_decision_tree():
    """示例 4：多级条件判断（决策树）"""
    print("=" * 70)
    print("示例 4：多级条件 - 客户服务路由")
    print("=" * 70)

    class ServiceState(TypedDict):
        customer_type: str  # vip, regular
        issue_type: str  # technical, billing, general
        priority: str
        handler: str

    def classify_customer(state: ServiceState) -> ServiceState:
        """第一步：分类客户类型"""
        c_type = state["customer_type"]
        print(f"  [客户分类] 客户类型: {c_type}")
        return state

    def classify_issue(state: ServiceState) -> ServiceState:
        """第二步：分类问题类型"""
        i_type = state["issue_type"]
        print(f"  [问题分类] 问题类型: {i_type}")

        # 根据客户类型和问题类型决定优先级
        if state["customer_type"] == "vip" and state["issue_type"] == "technical":
            priority = "urgent"
        elif state["customer_type"] == "vip":
            priority = "high"
        else:
            priority = "normal"

        print(f"  [问题分类] 优先级: {priority}")
        return {"priority": priority}

    def route_by_priority(state: ServiceState) -> Literal["vip_team", "regular_team"]:
        """根据优先级路由"""
        if state["priority"] in ["urgent", "high"]:
            return "vip_team"
        else:
            return "regular_team"

    def vip_team_handler(state: ServiceState) -> ServiceState:
        """VIP 团队处理"""
        print("  [VIP 团队] 专人处理中...")
        return {"handler": "VIP 专家团队"}

    def regular_team_handler(state: ServiceState) -> ServiceState:
        """普通团队处理"""
        print("  [普通团队] 标准流程处理...")
        return {"handler": "客户服务团队"}

    # 构建图
    graph = StateGraph(ServiceState)

    graph.add_node("classify_customer", classify_customer)
    graph.add_node("classify_issue", classify_issue)
    graph.add_node("vip_team", vip_team_handler)
    graph.add_node("regular_team", regular_team_handler)

    graph.set_entry_point("classify_customer")
    graph.add_edge("classify_customer", "classify_issue")

    graph.add_conditional_edges(
        "classify_issue",
        route_by_priority,
        {
            "vip_team": "vip_team",
            "regular_team": "regular_team"
        }
    )

    graph.add_edge("vip_team", END)
    graph.add_edge("regular_team", END)

    compiled_graph = graph.compile()

    # 测试不同场景
    scenarios = [
        {"customer_type": "vip", "issue_type": "technical", "priority": "", "handler": ""},
        {"customer_type": "vip", "issue_type": "billing", "priority": "", "handler": ""},
        {"customer_type": "regular", "issue_type": "general", "priority": "", "handler": ""},
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n场景 {i}:")
        print(f"  客户类型: {scenario['customer_type']}")
        print(f"  问题类型: {scenario['issue_type']}")
        result = compiled_graph.invoke(scenario)
        print(f"  处理团队: {result['handler']}")
    print()


# 总结：核心概念
"""
【条件边的核心概念】

1. 条件边（Conditional Edges）：
   - 根据状态动态决定下一个节点
   - 实现分支流程和决策逻辑
   - 比固定边更灵活

2. 条件函数（Condition Function）：
   - 接收当前状态作为输入
   - 返回下一个节点的名称（字符串）
   - 使用 Literal 类型注解可以指定可能的返回值

3. 语法：
   ```python
   graph.add_conditional_edges(
       source_node,  # 源节点
       condition_function,  # 条件函数
       {
           "return_value_1": "target_node_1",  # 返回值 -> 目标节点
           "return_value_2": "target_node_2",
           ...
       }
   )
   ```

4. 条件函数的要求：
   - 必须返回节点名称（字符串）
   - 可以使用 Literal[...] 类型注解
   - 逻辑要清晰，确保所有返回值都有对应的目标节点

5. 典型应用场景：
   - 意图识别和路由
   - 质量检查和分支处理
   - 错误处理和重试逻辑
   - 个性化推荐系统

【条件边的使用模式】

1. 简单二分支：
   - if-else 逻辑
   - 成功/失败路径

2. 多分支：
   - switch-case 逻辑
   - 根据类型/类别路由

3. 级联条件：
   - 多级决策树
   - 逐步细化的分类

4. 基于模型输出：
   - LLM 分类结果
   - 情感分析
   - 实体识别

【最佳实践】

1. 条件函数要简单明确：
   - 避免复杂逻辑
   - 便于调试和维护

2. 使用类型注解：
   - Literal 类型可以帮助检查错误
   - 明确可能的返回值

3. 命名清晰：
   - 条件函数名要表达意图
   - 节点名要有意义

4. 处理所有情况：
   - 确保每个可能的返回值都有对应节点
   - 考虑添加默认分支
"""

if __name__ == "__main__":
    example_1_simple_conditional()
    example_2_multiple_branches()
    example_3_llm_conditional()
    example_4_decision_tree()
