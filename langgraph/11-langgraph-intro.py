"""
LangGraph 学习 01：LangGraph 基础概念

知识点：
1. 什么是 LangGraph：用于构建有状态、多参与者应用程序的框架
2. 图（Graph）的基本组成：节点（Nodes）和边（Edges）
3. StateGraph：管理状态的图
4. 如何构建一个简单的图：定义节点、添加边、编译和调用
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
import operator
from dotenv import load_dotenv

load_dotenv()


# ============ 定义状态结构 ============

class SimpleState(TypedDict):
    """
    定义图的状态结构
    TypedDict 让我们可以明确指定状态中包含哪些字段
    """
    messages: Sequence[str]  # 消息列表
    counter: int  # 计数器


# ============ 示例 1：最简单的图 ============

def example_1_simple_graph():
    """示例 1：创建一个包含两个节点的简单图"""
    print("=" * 70)
    print("示例 1：最简单的图 - 两个节点顺序执行")
    print("=" * 70)

    # 定义节点函数
    def node_a(state: SimpleState) -> SimpleState:
        """节点 A：处理第一步"""
        print("  [节点 A] 正在执行...")
        messages = state["messages"]
        messages = list(messages) + ["节点 A 已处理"]
        return {"messages": messages, "counter": state["counter"] + 1}

    def node_b(state: SimpleState) -> SimpleState:
        """节点 B：处理第二步"""
        print("  [节点 B] 正在执行...")
        messages = state["messages"]
        messages = list(messages) + ["节点 B 已处理"]
        return {"messages": messages, "counter": state["counter"] + 1}

    # 创建图
    graph = StateGraph(SimpleState)

    # 添加节点
    graph.add_node("node_a", node_a)
    graph.add_node("node_b", node_b)

    # 添加边：定义节点之间的连接关系
    graph.set_entry_point("node_a")  # 设置入口点
    graph.add_edge("node_a", "node_b")  # node_a 执行完后执行 node_b
    graph.add_edge("node_b", END)  # node_b 执行完后结束

    # 编译图
    compiled_graph = graph.compile()

    # 执行图
    print("\n执行图:")
    initial_state = {
        "messages": [],
        "counter": 0
    }

    result = compiled_graph.invoke(initial_state)

    print(f"\n最终状态:")
    print(f"  消息: {result['messages']}")
    print(f"  计数器: {result['counter']}\n")


# ============ 示例 2：带 LLM 的图 ============

def example_2_graph_with_llm():
    """示例 2：在图中使用 LLM"""
    print("=" * 70)
    print("示例 2：包含 LLM 的图")
    print("=" * 70)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    class LLMState(TypedDict):
        input: str
        output: str

    def generate_node(state: LLMState) -> LLMState:
        """生成节点：使用 LLM 生成回复"""
        print("  [生成节点] 正在调用 LLM...")
        response = llm.invoke(state["input"])
        return {"output": response.content}

    def process_node(state: LLMState) -> LLMState:
        """处理节点：后处理 LLM 的输出"""
        print("  [处理节点] 正在处理输出...")
        processed = state["output"].upper()
        return {"output": f"处理后的输出: {processed}"}

    # 构建图
    graph = StateGraph(LLMState)
    graph.add_node("generate", generate_node)
    graph.add_node("process", process_node)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "process")
    graph.add_edge("process", END)

    compiled_graph = graph.compile()

    # 执行
    print("\n执行图:")
    result = compiled_graph.invoke({
        "input": "用一句话介绍 Python",
        "output": ""
    })

    print(f"\n最终输出:\n{result['output']}\n")


# ============ 示例 3：分支图 ============

def example_3_branching_graph():
    """示例 3：并行执行的分支"""
    print("=" * 70)
    print("示例 3：分支图 - 并行执行多个节点")
    print("=" * 70)

    class BranchState(TypedDict):
        input: str
        branch_a_result: str
        branch_b_result: str
        branch_c_result: str

    def start_node(state: BranchState) -> BranchState:
        """开始节点"""
        print("  [开始节点] 分发任务到三个分支...")
        return state

    def branch_a(state: BranchState) -> BranchState:
        """分支 A：转换为大写"""
        print("  [分支 A] 转换为大写")
        return {"branch_a_result": state["input"].upper()}

    def branch_b(state: BranchState) -> BranchState:
        """分支 B：转换为小写"""
        print("  [分支 B] 转换为小写")
        return {"branch_b_result": state["input"].lower()}

    def branch_c(state: BranchState) -> BranchState:
        """分支 C：反转文本"""
        print("  [分支 C] 反转文本")
        return {"branch_c_result": state["input"][::-1]}

    def end_node(state: BranchState) -> BranchState:
        """结束节点：汇总结果"""
        print("  [结束节点] 汇总所有分支的结果")
        return state

    # 构建图
    graph = StateGraph(BranchState)

    # 添加所有节点
    graph.add_node("start", start_node)
    graph.add_node("branch_a", branch_a)
    graph.add_node("branch_b", branch_b)
    graph.add_node("branch_c", branch_c)
    graph.add_node("end", end_node)

    # 设置入口
    graph.set_entry_point("start")

    # 从 start 分发到三个分支（并行执行）
    graph.add_edge("start", "branch_a")
    graph.add_edge("start", "branch_b")
    graph.add_edge("start", "branch_c")

    # 三个分支都执行完后，执行 end 节点
    # 注意：这里简化处理，实际需要使用更复杂的同步机制
    graph.add_edge("branch_a", "end")
    graph.add_edge("branch_b", "end")
    graph.add_edge("branch_c", "end")
    graph.add_edge("end", END)

    compiled_graph = graph.compile()

    # 执行
    print("\n执行图:")
    result = compiled_graph.invoke({
        "input": "Hello World",
        "branch_a_result": "",
        "branch_b_result": "",
        "branch_c_result": ""
    })

    print(f"\n最终结果:")
    print(f"  分支 A (大写): {result['branch_a_result']}")
    print(f"  分支 B (小写): {result['branch_b_result']}")
    print(f"  分支 C (反转): {result['branch_c_result']}\n")


# ============ 示例 4：可视化图结构 ============

def example_4_visualize_graph():
    """示例 4：获取图的结构信息"""
    print("=" * 70)
    print("示例 4：理解图的结构")
    print("=" * 70)

    class SimpleState(TypedDict):
        value: int

    def node_1(state: SimpleState) -> SimpleState:
        return {"value": state["value"] + 1}

    def node_2(state: SimpleState) -> SimpleState:
        return {"value": state["value"] * 2}

    # 构建图
    graph = StateGraph(SimpleState)
    graph.add_node("node_1", node_1)
    graph.add_node("node_2", node_2)
    graph.set_entry_point("node_1")
    graph.add_edge("node_1", "node_2")
    graph.add_edge("node_2", END)

    compiled_graph = graph.compile()

    print("\n图的结构信息:")
    print("  - 节点: node_1 -> node_2 -> END")
    print("  - 入口点: node_1")

    result = compiled_graph.invoke({"value": 5})
    print(f"\n执行结果:")
    print(f"  初始值: 5")
    print(f"  node_1 (5+1=6)")
    print(f"  node_2 (6*2=12)")
    print(f"  最终值: {result['value']}\n")


# 总结：核心概念
"""
【LangGraph 的核心概念】

1. 图（Graph）：
   - 由节点（Nodes）和边（Edges）组成的有向图
   - 用于表示复杂的工作流和状态转换

2. 节点（Nodes）：
   - 执行具体操作的函数
   - 接收当前状态作为输入，返回状态更新
   - 可以是任何 Python 函数

3. 边（Edges）：
   - 定义节点之间的连接关系
   - 控制执行流程
   - 类型：
     * 普通边：从一个节点到另一个节点
     * 条件边：根据状态决定下一步
     * 到 END：结束图的执行

4. 状态（State）：
   - 在整个图中共享的数据
   - 使用 TypedDict 定义结构
   - 每个节点可以读取和更新状态

5. 图的构建步骤：
   a. 定义状态结构（TypedDict）
   b. 创建 StateGraph 实例
   c. 添加节点（add_node）
   d. 添加边（add_edge）
   e. 设置入口点（set_entry_point）
   f. 编译图（compile）
   g. 调用图（invoke）

【LangGraph vs LangChain Chains】

LangChain Chains：
- 线性执行，适合简单的顺序流程
- 难以处理复杂的分支和循环

LangGraph：
- 可以表示复杂的流程（分支、循环、并行）
- 状态管理更灵活
- 适合构建智能体、聊天机器人等复杂应用

【何时使用 LangGraph】
1. 需要循环执行（如 Agent 的推理循环）
2. 需要条件分支（如根据结果决定下一步）
3. 需要持久化状态（如长期运行的对话）
4. 需要人工干预（如需要用户确认的流程）
"""

if __name__ == "__main__":
    example_1_simple_graph()
    example_2_graph_with_llm()
    example_3_branching_graph()
    example_4_visualize_graph()
