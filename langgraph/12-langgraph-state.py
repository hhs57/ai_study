"""
LangGraph 学习 02：状态管理（State Management）

知识点：
1. Annotated：用于定义状态的更新方式（追加、覆盖、合并等）
2. operator.add：追加消息到列表
3. Reducer：控制状态如何更新
4. 在节点中读取和更新状态
5. 状态的持久化和检查点（Checkpointing）
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Sequence
import operator
from dotenv import load_dotenv

load_dotenv()


# ============ 示例 1：基本状态管理 ============

def example_1_basic_state():
    """示例 1：最基本的状态管理"""
    print("=" * 70)
    print("示例 1：基本状态 - 字符串覆盖")
    print("=" * 70)

    class BasicState(TypedDict):
        """状态定义：简单的键值对"""
        user_input: str  # 用户输入
        response: str    # AI 响应
        step_count: int  # 步骤计数

    def process_node(state: BasicState) -> BasicState:
        """处理节点：读取状态并更新"""
        print(f"  [处理节点] 处理输入: {state['user_input']}")

        # 读取状态
        current_input = state["user_input"]
        step = state["step_count"]

        # 更新状态（返回要更新的字段）
        return {
            "response": f"已处理你的输入: {current_input}",
            "step_count": step + 1
        }

    def enhance_node(state: BasicState) -> BasicState:
        """增强节点：进一步处理"""
        print("  [增强节点] 增强响应...")

        enhanced = state["response"] + " (已增强)"
        return {
            "response": enhanced,
            "step_count": state["step_count"] + 1
        }

    # 构建图
    graph = StateGraph(BasicState)
    graph.add_node("process", process_node)
    graph.add_node("enhance", enhance_node)
    graph.set_entry_point("process")
    graph.add_edge("process", "enhance")
    graph.add_edge("enhance", END)

    compiled_graph = graph.compile()

    # 执行
    print("\n执行图:")
    result = compiled_graph.invoke({
        "user_input": "你好，世界！",
        "response": "",
        "step_count": 0
    })

    print(f"\n最终状态:")
    print(f"  输入: {result['user_input']}")
    print(f"  响应: {result['response']}")
    print(f"  步骤数: {result['step_count']}\n")


# ============ 示例 2：消息列表状态（使用 Annotated） ============

def example_2_message_list_state():
    """示例 2：使用 Annotated 管理消息列表"""
    print("=" * 70)
    print("示例 2：消息列表状态 - 使用 operator.add 追加消息")
    print("=" * 70)

    class MessageState(TypedDict):
        """
        使用 Annotated 定义如何更新消息列表
        operator.add 表示追加（而不是替换）
        """
        messages: Annotated[Sequence[BaseMessage], operator.add]

    def chatbot_node(state: MessageState) -> MessageState:
        """聊天机器人节点：生成回复"""
        print("  [聊天机器人] 生成回复...")

        # 获取最后一条用户消息
        last_message = state["messages"][-1]
        print(f"    用户说: {last_message.content}")

        # 生成回复
        response = f"我收到了你的消息：{last_message.content}"
        return {"messages": [AIMessage(content=response)]}

    def human_node(state: MessageState) -> MessageState:
        """人类节点：模拟人类输入"""
        print("  [人类节点] 添加人类消息...")

        # 在实际应用中，这里会等待真实用户输入
        # 这里我们模拟添加一条消息
        return {"messages": [HumanMessage(content="这很好！")]}

    # 构建图
    graph = StateGraph(MessageState)
    graph.add_node("chatbot", chatbot_node)
    graph.add_node("human", human_node)
    graph.set_entry_point("chatbot")
    graph.add_edge("chatbot", "human")
    graph.add_edge("human", END)

    compiled_graph = graph.compile()

    # 执行
    print("\n执行图:")
    result = compiled_graph.invoke({
        "messages": [HumanMessage(content="你好")]
    })

    print(f"\n最终消息列表:")
    for i, msg in enumerate(result["messages"]):
        msg_type = "用户" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {i+1}. [{msg_type}] {msg.content}\n")


# ============ 示例 3：使用 LLM 和消息状态 ============

def example_3_llm_with_messages():
    """示例 3：LLM 与消息状态的结合"""
    print("=" * 70)
    print("示例 3：使用 LLM 处理消息状态")
    print("=" * 70)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    class LLMState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]

    def call_model(state: LLMState) -> LLMState:
        """调用 LLM 生成回复"""
        print("  [模型节点] 调用 LLM...")

        # 直接将消息列表传递给 LLM
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    # 构建图
    graph = StateGraph(LLMState)
    graph.add_node("model", call_model)
    graph.set_entry_point("model")
    graph.add_edge("model", END)

    compiled_graph = graph.compile()

    # 执行多轮对话
    print("\n执行多轮对话:")
    messages = [
        SystemMessage(content="你是一个友好的助手。"),
        HumanMessage(content="我叫张三")
    ]

    # 第一轮
    result1 = compiled_graph.invoke({"messages": messages})
    print(f"\nAI 回复 1: {result1['messages'][-1].content}")

    # 第二轮
    new_messages = result1["messages"] + [HumanMessage(content="我刚才说我叫什么？")]
    result2 = compiled_graph.invoke({"messages": new_messages})
    print(f"AI 回复 2: {result2['messages'][-1].content}\n")


# ============ 示例 4：复杂的状态结构 ============

def example_4_complex_state():
    """示例 4：包含多种类型的状态"""
    print("=" * 70)
    print("示例 4：复杂状态 - 多种数据类型")
    print("=" * 70)

    class ComplexState(TypedDict):
        """复杂状态：包含多种数据类型"""
        messages: Annotated[Sequence[BaseMessage], operator.add]  # 消息列表（追加）
        current_topic: str  # 当前话题（覆盖）
        confidence: float  # 置信度（覆盖）
        tags: list[str]  # 标签列表（覆盖）
        metadata: dict  # 元数据（合并）

    def analyze_node(state: ComplexState) -> ComplexState:
        """分析节点"""
        print("  [分析节点] 分析输入...")

        last_message = state["messages"][-1].content

        return {
            "current_topic": f"关于: {last_message[:20]}",
            "confidence": 0.85,
            "tags": ["分析", "初稿"],
            "metadata": {"analyzer": "node_1", "timestamp": "2024-01-01"}
        }

    def refine_node(state: ComplexState) -> ComplexState:
        """优化节点"""
        print("  [优化节点] 优化结果...")

        # 更新多个字段
        return {
            "messages": [AIMessage(content="优化完成")],
            "confidence": state["confidence"] + 0.1,  # 提高置信度
            "tags": state["tags"] + ["已优化"],  # 添加标签
            "metadata": {"refiner": "node_2", "quality": "high"}
        }

    # 构建图
    graph = StateGraph(ComplexState)
    graph.add_node("analyze", analyze_node)
    graph.add_node("refine", refine_node)
    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "refine")
    graph.add_edge("refine", END)

    compiled_graph = graph.compile()

    # 执行
    print("\n执行图:")
    result = compiled_graph.invoke({
        "messages": [HumanMessage(content="分析人工智能的发展趋势")],
        "current_topic": "",
        "confidence": 0.0,
        "tags": [],
        "metadata": {}
    })

    print(f"\n最终状态:")
    print(f"  话题: {result['current_topic']}")
    print(f"  置信度: {result['confidence']}")
    print(f"  标签: {result['tags']}")
    print(f"  元数据: {result['metadata']}")
    print(f"  消息数: {len(result['messages'])}\n")


# ============ 示例 5：状态检查点（Checkpointing） ============

def example_5_checkpointing():
    """示例 5：使用检查点保存和恢复状态"""
    print("=" * 70)
    print("示例 5：检查点 - 保存和恢复状态")
    print("=" * 70)

    class CheckpointState(TypedDict):
        counter: int
        history: list[str]

    def step_node(state: CheckpointState) -> CheckpointState:
        """步骤节点"""
        step = state["counter"] + 1
        print(f"  [步骤节点] 执行第 {step} 步")

        return {
            "counter": step,
            "history": state["history"] + [f"步骤 {step}"]
        }

    # 创建图
    graph = StateGraph(CheckpointState)
    graph.add_node("step", step_node)
    graph.set_entry_point("step")
    graph.add_edge("step", END)

    # 创建内存检查点保存器
    memory = MemorySaver()
    compiled_graph = graph.compile(checkpointer=memory)

    # 使用 thread_id 标识对话会话
    config = {"configurable": {"thread_id": "session-1"}}

    print("\n执行第一次（保存到检查点）:")
    result1 = compiled_graph.invoke(
        {"counter": 0, "history": []},
        config=config
    )
    print(f"结果: counter={result1['counter']}, history={result1['history']}")

    print("\n执行第二次（继续之前的会话）:")
    result2 = compiled_graph.invoke(
        result1,  # 使用之前的结果作为起点
        config=config
    )
    print(f"结果: counter={result2['counter']}, history={result2['history']}")

    print("\n执行第三次:")
    result3 = compiled_graph.invoke(result2, config=config)
    print(f"结果: counter={result3['counter']}, history={result3['history']}")

    print("\n注意：检查点允许我们暂停和恢复执行状态！\n")


# 总结：核心概念
"""
【状态管理的关键概念】

1. Annotated（注解）：
   - 用于定义状态的更新策略
   - 语法：Annotated[类型, 更新方式]

2. 更新策略（Reducers）：
   a. operator.add（追加）：
      - 用于列表，追加新元素而不是替换
      - 适用于消息历史、日志等

   b. 默认（覆盖）：
      - 直接替换旧值
      - 适用于计数器、当前值等

   c. 自定义 Reducer：
      - 可以定义复杂的合并逻辑
      - 如字典的深度合并

3. 状态更新规则：
   - 节点函数返回需要更新的字段
   - 未返回的字段保持不变
   - 使用 Annotated 的字段按指定方式更新

4. 检查点（Checkpointing）：
   - 保存图的执行状态
   - 可以暂停和恢复执行
   - 支持长期运行的对话
   - 使用 thread_id 标识会话

【状态管理的最佳实践】

1. 合理设计状态结构：
   - 只包含必要的数据
   - 区分需要追加和覆盖的字段

2. 使用 TypedDict：
   - 明确状态的结构
   - 提供类型提示

3. 消息列表使用 operator.add：
   - 自动追加新消息
   - 保持完整的历史记录

4. 使用检查点：
   - 长期对话需要保存状态
   - 支持多用户并发访问

【状态 vs 临时变量】

状态（State）：
- 在节点间共享
- 持久化到检查点
- 用于控制流程

临时变量：
- 仅在节点内部使用
- 不保存在状态中
- 适合中间计算
"""

if __name__ == "__main__":
    example_1_basic_state()
    example_2_message_list_state()
    example_3_llm_with_messages()
    example_4_complex_state()
    example_5_checkpointing()
