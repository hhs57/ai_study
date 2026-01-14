"""
LangGraph 学习 04：循环（Loops）

知识点：
1. 如何在图中创建循环
2. 使用条件边实现循环逻辑
3. 设置最大循环次数防止无限循环
4. 构建 Agent 的推理循环（ReAct 模式）
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence, Literal
import operator
from dotenv import load_dotenv

load_dotenv()


# ============ 示例 1：基础计数循环 ============

def example_1_counter_loop():
    """示例 1：简单的计数循环"""
    print("=" * 70)
    print("示例 1：计数循环 - 累加到目标值")
    print("=" * 70)

    class CounterState(TypedDict):
        count: int
        target: int
        history: list[str]

    def increment_node(state: CounterState) -> CounterState:
        """递增节点"""
        new_count = state["count"] + 1
        print(f"  [递增节点] 计数: {state['count']} -> {new_count}")

        return {
            "count": new_count,
            "history": state["history"] + [f"计数增加到 {new_count}"]
        }

    def should_continue(state: CounterState) -> Literal["continue", "end"]:
        """
        条件函数：决定是继续循环还是结束
        这是循环的关键！
        """
        if state["count"] < state["target"]:
            print(f"  [判断] {state['count']} < {state['target']}, 继续循环")
            return "continue"
        else:
            print(f"  [判断] {state['count']} >= {state['target']}, 结束循环")
            return "end"

    # 构建图
    graph = StateGraph(CounterState)

    graph.add_node("increment", increment_node)
    graph.set_entry_point("increment")

    # 添加条件边：根据 should_continue 的结果决定下一步
    # 如果返回 "continue"，则回到 "increment" 节点（形成循环）
    # 如果返回 "end"，则结束图的执行
    graph.add_conditional_edges(
        "increment",
        should_continue,
        {
            "continue": "increment",  # 循环：回到 increment 节点
            "end": END  # 结束
        }
    )

    compiled_graph = graph.compile()

    # 执行
    print("\n执行循环（目标：5）:")
    result = compiled_graph.invoke({
        "count": 0,
        "target": 5,
        "history": []
    })

    print(f"\n最终结果:")
    print(f"  计数: {result['count']}")
    print(f"  历史记录: {result['history']}\n")


# ============ 示例 2：带最大迭代限制的循环 ============

def example_2_max_iterations():
    """示例 2：防止无限循环"""
    print("=" * 70)
    print("示例 2：带安全限制的循环")
    print("=" * 70)

    class SafeLoopState(TypedDict):
        attempts: int
        max_attempts: int
        success: bool
        message: str

    def try_operation(state: SafeLoopState) -> SafeLoopState:
        """尝试执行操作"""
        attempt = state["attempts"] + 1
        print(f"  [尝试节点] 第 {attempt} 次尝试")

        # 模拟：第 3 次尝试会成功
        success = (attempt >= 3)

        return {
            "attempts": attempt,
            "success": success,
            "message": f"第 {attempt} 次尝试" + (" 成功！" if success else " 失败")
        }

    def should_retry(state: SafeLoopState) -> Literal["retry", "success", "give_up"]:
        """
        决定是否重试
        这是一个三分支的条件：重试、成功、放弃
        """
        if state["success"]:
            return "success"
        elif state["attempts"] < state["max_attempts"]:
            return "retry"
        else:
            return "give_up"

    def success_node(state: SafeLoopState) -> SafeLoopState:
        """成功节点"""
        print("  [成功节点] 操作成功完成！")
        return {"message": state["message"] + " - 最终成功"}

    def give_up_node(state: SafeLoopState) -> SafeLoopState:
        """放弃节点"""
        print(f"  [放弃节点] 达到最大尝试次数 {state['max_attempts']}，放弃")
        return {"message": state["message"] + " - 达到最大限制"}

    # 构建图
    graph = StateGraph(SafeLoopState)

    graph.add_node("try_operation", try_operation)
    graph.add_node("success", success_node)
    graph.add_node("give_up", give_up_node)

    graph.set_entry_point("try_operation")

    # 条件边：三种可能性
    graph.add_conditional_edges(
        "try_operation",
        should_retry,
        {
            "retry": "try_operation",  # 循环：重试
            "success": "success",  # 成功分支
            "give_up": "give_up"  # 放弃分支
        }
    )

    graph.add_edge("success", END)
    graph.add_edge("give_up", END)

    compiled_graph = graph.compile()

    # 执行
    print("\n场景 1: 会在第 3 次成功")
    result1 = compiled_graph.invoke({
        "attempts": 0,
        "max_attempts": 5,
        "success": False,
        "message": ""
    })
    print(f"结果: {result1['message']}\n")

    print("场景 2: 达到最大尝试次数（设置为 2 次）")
    result2 = compiled_graph.invoke({
        "attempts": 0,
        "max_attempts": 2,
        "success": False,
        "message": ""
    })
    print(f"结果: {result2['message']}\n")


# ============ 示例 3：ReAct 风格的推理循环 ============

def example_3_react_loop():
    """示例 3：模拟 Agent 的 ReAct 循环"""
    print("=" * 70)
    print("示例 3：ReAct 循环 - 推理和行动")
    print("=" * 70)

    class ReActState(TypedDict):
        question: str
        thoughts: list[str]
        actions: list[str]
        observations: list[str]
        answer: str
        step_count: int

    def think_node(state: ReActState) -> ReActState:
        """思考节点"""
        step = state["step_count"] + 1
        print(f"  [思考节点] 第 {step} 步")

        thought = f"思考步骤 {step}: 我需要解决 '{state['question']}'"
        print(f"    {thought}")

        return {
            "thoughts": state["thoughts"] + [thought],
            "step_count": step
        }

    def should_continue(state: ReActState) -> Literal["think", "answer"]:
        """
        决定是继续思考还是给出答案
        这里简单地在第 3 步后给出答案
        """
        if state["step_count"] < 3:
            return "think"
        else:
            return "answer"

    def answer_node(state: ReActState) -> ReActState:
        """答案节点"""
        answer = f"基于 {state['step_count']} 步思考，答案如下："
        print(f"  [答案节点] 生成最终答案")
        return {"answer": answer}

    # 构建图
    graph = StateGraph(ReActState)

    graph.add_node("think", think_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("think")

    # 核心：通过条件边实现循环
    # 如果应该继续思考，则回到 "think" 节点
    # 如果应该给出答案，则去往 "answer" 节点
    graph.add_conditional_edges(
        "think",
        should_continue,
        {
            "think": "think",  # 循环：继续思考
            "answer": "answer"  # 结束：给出答案
        }
    )

    graph.add_edge("answer", END)

    compiled_graph = graph.compile()

    # 执行
    print("\n执行 ReAct 循环:")
    result = compiled_graph.invoke({
        "question": "什么是 LangGraph？",
        "thoughts": [],
        "actions": [],
        "observations": [],
        "answer": "",
        "step_count": 0
    })

    print(f"\n最终结果:")
    print(f"  思考步骤: {len(result['thoughts'])}")
    print(f"  答案: {result['answer']}\n")


# ============ 示例 4：累积式循环 ============

def example_4_accumulator_loop():
    """示例 4：在循环中累积结果"""
    print("=" * 70)
    print("示例 4：累积循环 - 收集多个结果")
    print("=" * 70)

    class AccumulatorState(TypedDict):
        items: list[str]
        current_index: int
        total_items: int
        result: str

    def process_item_node(state: AccumulatorState) -> AccumulatorState:
        """处理单个项目"""
        index = state["current_index"]
        print(f"  [处理节点] 处理第 {index + 1} 项")

        # 模拟处理：生成一个字符串
        processed = f"处理后的项目 {index + 1}"
        items = state["items"] + [processed]

        return {
            "items": items,
            "current_index": index + 1
        }

    def has_more_items(state: AccumulatorState) -> Literal["continue", "finish"]:
        """检查是否还有更多项目要处理"""
        if state["current_index"] < state["total_items"]:
            return "continue"
        else:
            return "finish"

    def finish_node(state: AccumulatorState) -> AccumulatorState:
        """完成节点：汇总所有结果"""
        print("  [完成节点] 汇总所有处理结果")

        result = f"处理完成！共处理 {len(state['items'])} 项：\n"
        result += "\n".join(f"  - {item}" for item in state['items'])

        return {"result": result}

    # 构建图
    graph = StateGraph(AccumulatorState)

    graph.add_node("process", process_item_node)
    graph.add_node("finish", finish_node)

    graph.set_entry_point("process")

    # 循环条件
    graph.add_conditional_edges(
        "process",
        has_more_items,
        {
            "continue": "process",  # 继续处理下一项
            "finish": "finish"  # 完成所有处理
        }
    )

    graph.add_edge("finish", END)

    compiled_graph = graph.compile()

    # 执行
    print("\n处理 5 个项目:")
    result = compiled_graph.invoke({
        "items": [],
        "current_index": 0,
        "total_items": 5,
        "result": ""
    })

    print(f"\n{result['result']}\n")


# ============ 示例 5：使用 LLM 的循环 ============

def example_5_llm_loop():
    """示例 5：LLM 在循环中逐步改进答案"""
    print("=" * 70)
    print("示例 5：LLM 改进循环 - 迭代优化")
    print("=" * 70)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    class ImprovementState(TypedDict):
        original_question: str
        current_answer: str
        iteration: int
        max_iterations: int
        is_satisfied: bool
        history: list[str]

    def improve_answer_node(state: ImprovementState) -> ImprovementState:
        """改进答案节点"""
        iteration = state["iteration"] + 1
        print(f"  [改进节点] 第 {iteration} 轮改进")

        # 在实际应用中，这里会调用 LLM 改进答案
        # 这里简化处理
        if iteration == 1:
            improved = "初步答案：这是一个简单的问题"
        elif iteration == 2:
            improved = "改进答案：这个问题涉及多个方面"
        else:
            improved = "最终答案：经过深入思考，答案如下..."

        print(f"    改进后的答案: {improved}")

        return {
            "current_answer": improved,
            "iteration": iteration,
            "history": state["history"] + [f"第 {iteration} 轮: {improved}"],
            "is_satisfied": iteration >= state["max_iterations"]
        }

    def should_continue_improving(state: ImprovementState) -> Literal["improve", "finish"]:
        """决定是否继续改进"""
        if state["is_satisfied"]:
            return "finish"
        else:
            return "improve"

    # 构建图
    graph = StateGraph(ImprovementState)

    graph.add_node("improve", improve_answer_node)
    graph.set_entry_point("improve")

    graph.add_conditional_edges(
        "improve",
        should_continue_improving,
        {
            "improve": "improve",  # 循环：继续改进
            "finish": END  # 结束
        }
    )

    compiled_graph = graph.compile()

    # 执行
    print("\n迭代改进（最多 3 轮）:")
    result = compiled_graph.invoke({
        "original_question": "什么是 AI？",
        "current_answer": "",
        "iteration": 0,
        "max_iterations": 3,
        "is_satisfied": False,
        "history": []
    })

    print(f"\n最终答案:")
    print(f"  {result['current_answer']}")
    print(f"\n迭代历史: {len(result['history'])} 轮")
    for i, entry in enumerate(result['history'], 1):
        print(f"  {i}. {entry}")
    print()


# 总结：核心概念
"""
【循环的核心概念】

1. 循环的实现方式：
   - 使用条件边（conditional_edges）实现
   - 条件函数返回源节点名称 = 形成循环
   - 条件函数返回其他节点名称 = 退出循环

2. 循环的三要素：
   a. 初始化：设置起始状态
   b. 循环体：执行操作的节点
   c. 循环条件：决定是继续还是退出

3. 循环的模式：

   模式 1：计数循环
   ```
   state.count < target -> 继续循环
   state.count >= target -> 退出循环
   ```

   模式 2：条件循环
   ```
   not success -> 继续重试
   success -> 退出循环
   ```

   模式 3：累积循环
   ```
   has_more -> 继续处理
   no_more -> 完成处理
   ```

4. 防止无限循环：
   - 设置最大迭代次数（max_iterations）
   - 在条件函数中检查计数器
   - 提供强制退出的路径
   - 使用 recursion_limit 参数

5. 循环的应用场景：
   - Agent 的推理循环（ReAct）
   - 重试失败的请求
   - 迭代优化结果
   - 批量处理项目
   - 渐进式解决问题

【循环的最佳实践】

1. 明确的退出条件：
   - 总是提供明确的退出路径
   - 避免依赖复杂的逻辑

2. 限制迭代次数：
   - 设置合理的最大次数
   - 在状态中跟踪迭代次数

3. 状态管理：
   - 在循环中维护必要的状态
   - 使用 history 记录迭代过程

4. 调试技巧：
   - 在每个循环步骤打印日志
   - 检查状态的变化
   - 可视化循环流程

5. 性能考虑：
   - LLM 调用很昂贵，控制循环次数
   - 考虑提前退出条件
   - 批量处理时限制并发

【LangGraph 中的循环 vs 传统循环】

传统循环（for/while）：
- 一次性执行完成
- 执行过程不可见
- 难以暂停和恢复

LangGraph 循环：
- 每步都是独立的状态转换
- 可以保存中间状态（检查点）
- 支持长时间运行的任务
- 可以人工干预
"""

if __name__ == "__main__":
    example_1_counter_loop()
    example_2_max_iterations()
    example_3_react_loop()
    example_4_accumulator_loop()
    example_5_llm_loop()
