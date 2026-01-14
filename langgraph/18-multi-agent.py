"""
LangGraph 学习 08：多 Agent 系统

知识点：
1. 多 Agent 架构模式
2. Agent 之间的通信
3. 协作式 Agent
4. 竞争式 Agent
5. 层级式 Agent 系统
"""

import sys
import io
from typing import TypedDict, Annotated, Sequence, Literal
import operator

# 设置标准输出为 UTF-8 编码
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from langgraph.graph import StateGraph, END


# ============ 示例 1：简单的协作 Agent ============

def example_1_collaborative_agents():
    """示例 1：协作式 Agent 系统"""
    print("=" * 70)
    print("示例 1：协作式 Agent - 多个 Agent 合作完成任务")
    print("=" * 70)

    class CollaborativeState(TypedDict):
        task: str
        researcher_result: str
        writer_result: str
        reviewer_result: str
        final_output: str
        step: str

    def researcher_agent(state: CollaborativeState) -> CollaborativeState:
        """研究员 Agent：收集信息"""
        print("  [研究员] 正在研究主题...")
        research = f"关于'{state['task']}'的研究：\n- 关键点1\n- 关键点2\n- 关键点3"
        print("  [研究员] 研究完成")
        return {"researcher_result": research, "step": "research_done"}

    def writer_agent(state: CollaborativeState) -> CollaborativeState:
        """作家 Agent：撰写内容"""
        print("  [作家] 正在撰写内容...")
        article = f"基于研究：\n{state['researcher_result']}\n\n这是一篇关于{state['task']}的文章。"
        print("  [作家] 撰写完成")
        return {"writer_result": article, "step": "writing_done"}

    def reviewer_agent(state: CollaborativeState) -> CollaborativeState:
        """审核员 Agent：审核内容"""
        print("  [审核员] 正在审核内容...")
        review = f"审核意见：\n文章质量良好，内容完整。\n\n{state['writer_result']}"
        print("  [审核员] 审核通过")
        return {"reviewer_result": review, "final_output": review, "step": "done"}

    # 构建协作流程
    graph = StateGraph(CollaborativeState)

    graph.add_node("researcher", researcher_agent)
    graph.add_node("writer", writer_agent)
    graph.add_node("reviewer", reviewer_agent)

    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "reviewer")
    graph.add_edge("reviewer", END)

    compiled_graph = graph.compile()

    # 执行
    print("\n执行协作流程:")
    print("-" * 70)

    result = compiled_graph.invoke({
        "task": "人工智能的发展",
        "researcher_result": "",
        "writer_result": "",
        "reviewer_result": "",
        "final_output": "",
        "step": "start"
    })

    print(f"\n最终输出:\n{result['final_output'][:200]}...")


# ============ 示例 2：竞争式 Agent ============

def example_2_competitive_agents():
    """示例 2：竞争式 Agent - 多个 Agent 提供方案"""
    print("=" * 70)
    print("示例 2：竞争式 Agent - 比较不同方案")
    print("=" * 70)

    class CompetitiveState(TypedDict):
        problem: str
        solution_a: str
        solution_b: str
        solution_c: str
        best_solution: str
        scores: dict

    def agent_a(state: CompetitiveState) -> CompetitiveState:
        """Agent A：保守方案"""
        print("  [Agent A] 提出保守方案...")
        solution = "保守方案：使用成熟技术，风险低，收益稳定。"
        score = 7.5
        return {
            "solution_a": solution,
            "scores": {"agent_a": score, "agent_b": 0, "agent_c": 0}
        }

    def agent_b(state: CompetitiveState) -> CompetitiveState:
        """Agent B：激进方案"""
        print("  [Agent B] 提出激进方案...")
        solution = "激进方案：使用最新技术，风险高，潜在收益大。"
        score = 8.5
        return {"solution_b": solution, "scores": {**state["scores"], "agent_b": score}}

    def agent_c(state: CompetitiveState) -> CompetitiveState:
        """Agent C：平衡方案"""
        print("  [Agent C] 提出平衡方案...")
        solution = "平衡方案：部分创新，保持稳定，中等风险收益。"
        score = 8.0
        return {"solution_c": solution, "scores": {**state["scores"], "agent_c": score}}

    def evaluator(state: CompetitiveState) -> CompetitiveState:
        """评估者：选择最佳方案"""
        print("  [评估者] 比较所有方案...")

        scores = state["scores"]
        solutions = {
            "agent_a": state["solution_a"],
            "agent_b": state["solution_b"],
            "agent_c": state["solution_c"]
        }

        # 选择得分最高的方案
        best_agent = max(scores, key=scores.get)
        best = solutions[best_agent]

        print(f"  [评估者] 选择 {best_agent.upper()} 的方案（得分: {scores[best_agent]}）")

        return {"best_solution": best}

    # 构建竞争流程
    graph = StateGraph(CompetitiveState)

    graph.add_node("agent_a", agent_a)
    graph.add_node("agent_b", agent_b)
    graph.add_node("agent_c", agent_c)
    graph.add_node("evaluator", evaluator)

    graph.set_entry_point("agent_a")
    graph.add_edge("agent_a", "agent_b")
    graph.add_edge("agent_b", "agent_c")
    graph.add_edge("agent_c", "evaluator")
    graph.add_edge("evaluator", END)

    compiled_graph = graph.compile()

    # 执行
    print("\n执行竞争流程:")
    print("-" * 70)

    result = compiled_graph.invoke({
        "problem": "如何提高系统性能？",
        "solution_a": "",
        "solution_b": "",
        "solution_c": "",
        "best_solution": "",
        "scores": {}
    })

    print(f"\n最佳方案:\n{result['best_solution']}")


# ============ 示例 3：层级式 Agent 系统 ============

def example_3_hierarchical_agents():
    """示例 3：层级式 Agent - 管理 Agent 协调工作 Agent"""
    print("=" * 70)
    print("示例 3：层级式 Agent 系统")
    print("=" * 70)

    class HierarchicalState(TypedDict):
        task: str
        subtasks: list[str]
        worker_results: list[str]
        manager_decision: str
        status: str

    def manager_agent(state: HierarchicalState) -> HierarchicalState:
        """管理 Agent：分配任务"""
        print("  [管理 Agent] 分析任务并分配工作...")

        task = state["task"]
        subtasks = [
            f"子任务 1: 分析{task}的需求",
            f"子任务 2: 设计{task}的方案",
            f"子任务 3: 评估{task}的可行性"
        ]

        print(f"  [管理 Agent] 分配了 {len(subtasks)} 个子任务")

        return {"subtasks": subtasks, "status": "assigned"}

    def worker_agents(state: HierarchicalState) -> HierarchicalState:
        """工作 Agent：执行子任务"""
        print("  [工作 Agent] 执行子任务...")

        results = []
        for i, subtask in enumerate(state["subtasks"], 1):
            result = f"完成 {subtask}: 结果 {i}"
            results.append(result)
            print(f"    - {result}")

        return {"worker_results": results, "status": "completed"}

    def manager_review(state: HierarchicalState) -> HierarchicalState:
        """管理 Agent：审查结果"""
        print("  [管理 Agent] 审查工作结果...")

        decision = f"所有子任务已完成：\n" + "\n".join(state["worker_results"])
        decision += "\n\n管理 Agent 决定：任务成功完成！"

        print("  [管理 Agent] 批准并总结")

        return {"manager_decision": decision, "status": "approved"}

    # 构建层级流程
    graph = StateGraph(HierarchicalState)

    graph.add_node("manager_assign", manager_agent)
    graph.add_node("workers", worker_agents)
    graph.add_node("manager_review", manager_review)

    graph.set_entry_point("manager_assign")
    graph.add_edge("manager_assign", "workers")
    graph.add_edge("workers", "manager_review")
    graph.add_edge("manager_review", END)

    compiled_graph = graph.compile()

    # 执行
    print("\n执行层级流程:")
    print("-" * 70)

    result = compiled_graph.invoke({
        "task": "开发新功能",
        "subtasks": [],
        "worker_results": [],
        "manager_decision": "",
        "status": "start"
    })

    print(f"\n最终决定:\n{result['manager_decision']}")


# ============ 示例 4：Agent 通信模式 ============

def example_4_agent_communication():
    """示例 4：Agent 之间的通信"""
    print("=" * 70)
    print("示例 4：Agent 通信模式")
    print("=" * 70)

    print("""
多 Agent 系统的通信模式：

1. 共享状态（Shared State）：
   - 所有 Agent 共享同一个状态对象
   - 通过状态传递信息
   - 简单但可能产生冲突

   示例：
   ┌─────────┐    ┌─────────┐    ┌─────────┐
   │ Agent A │ ->│ State   │<-  │ Agent B │
   └─────────┘    └─────────┘    └─────────┘
         ↑                              ↑
         └──────────┬───────────────────┘
                    读写共享状态

2. 消息传递（Message Passing）：
   - Agent 之间直接发送消息
   - 异步通信
   - 更灵活

   示例：
   ┌─────────┐              ┌─────────┐
   │ Agent A │ ──消息───>  │ Agent B │
   └─────────┘              └─────────┘
        ↑                        │
        │                        │
        └────────反馈消息────────┘

3. 事件驱动（Event-Driven）：
   - 发布/订阅模式
   - Agent 订阅感兴趣的事件
   - 松耦合

   示例：
   ┌─────────┐              ┌─────────┐
   │ Agent A │ ──事件───>  │ Event   │───>│ Agent B │
   └─────────┘              │  Bus    │    └─────────┘
                           └─────────┘
                                │
                                ├───>│ Agent C │
                                │    └─────────┘
                                │
                                └───>│ Agent D │
                                     └─────────┘

4. 黑板系统（Blackboard）：
   - 共享的工作空间
   - Agent 读取和写入黑板
   - 适合复杂问题解决

   示例：
            ┌─────────────────┐
            │   Blackboard    │
            │  (共享工作空间)  │
            └─────────────────┘
                 ↑      ↑      ↑
                ┌┴──────┴──────┴┐
                │               │
           ┌────┴───┐     ┌────┴───┐
           │Agent A │     │Agent B │
           └────────┘     └────────┘

通信模式选择：
┌──────────────┬──────────┬──────────┬──────────┐
│ 模式         │ 复杂度   │ 耦合度   │ 适用场景 │
├──────────────┼──────────┼──────────┼──────────┤
│ 共享状态     │ 低       │ 高       │ 简单协作 │
│ 消息传递     │ 中       │ 中       │ 点对点   │
│ 事件驱动     │ 高       │ 低       │ 大规模   │
│ 黑板系统     │ 高       │ 低       │ 复杂问题 │
└──────────────┴──────────┴──────────┴──────────┘
    """)


# ============ 示例 5：实际应用 - 客服系统 ============

def example_5_customer_service_system():
    """示例 5：多 Agent 客服系统"""
    print("=" * 70)
    print("示例 5：多 Agent 客服系统")
    print("=" * 70)

    class CustomerServiceState(TypedDict):
        customer_query: str
        category: str
        technical_analysis: str
        billing_info: str
        general_response: str
        final_response: str

    def classifier_agent(state: CustomerServiceState) -> CustomerServiceState:
        """分类 Agent：识别问题类型"""
        query = state["customer_query"].lower()

        if any(word in query for word in ["技术", "bug", "错误", "无法"]):
            category = "technical"
        elif any(word in query for word in ["钱", "费用", "账单", "退款"]):
            category = "billing"
        else:
            category = "general"

        print(f"  [分类 Agent] 问题类型: {category}")
        return {"category": category}

    def technical_agent(state: CustomerServiceState) -> CustomerServiceState:
        """技术支持 Agent"""
        print("  [技术 Agent] 分析技术问题...")
        analysis = "技术问题诊断：\n- 检查系统日志\n- 重启服务\n- 联系高级支持"
        return {"technical_analysis": analysis}

    def billing_agent(state: CustomerServiceState) -> CustomerServiceState:
        """账务 Agent"""
        print("  [账务 Agent] 查询账务信息...")
        info = "账务信息：\n- 当前账单：$100\n- 下次付款日期：2024-02-01"
        return {"billing_info": info}

    def general_agent(state: CustomerServiceState) -> CustomerServiceState:
        """通用客服 Agent"""
        print("  [通用 Agent] 准备通用回复...")
        response = "感谢您的咨询，我们会尽快处理。"
        return {"general_response": response}

    def coordinator_agent(state: CustomerServiceState) -> CustomerServiceState:
        """协调 Agent：整合所有回复"""
        print("  [协调 Agent] 整合回复...")

        category = state["category"]
        if category == "technical":
            final = state["technical_analysis"]
        elif category == "billing":
            final = state["billing_info"]
        else:
            final = state["general_response"]

        return {"final_response": final}

    # 构建多 Agent 系统
    graph = StateGraph(CustomerServiceState)

    graph.add_node("classifier", classifier_agent)
    graph.add_node("technical", technical_agent)
    graph.add_node("billing", billing_agent)
    graph.add_node("general", general_agent)
    graph.add_node("coordinator", coordinator_agent)

    graph.set_entry_point("classifier")

    # 根据分类路由到不同的 Agent
    graph.add_conditional_edges(
        "classifier",
        lambda state: state["category"],
        {
            "technical": "technical",
            "billing": "billing",
            "general": "general"
        }
    )

    graph.add_edge("technical", "coordinator")
    graph.add_edge("billing", "coordinator")
    graph.add_edge("general", "coordinator")
    graph.add_edge("coordinator", END)

    compiled_graph = graph.compile()

    # 测试不同类型的查询
    queries = [
        "我的系统无法登录",
        "我想查看最近的账单",
        "你们的工作时间是什么？"
    ]

    for query in queries:
        print(f"\n客户查询: {query}")
        print("-" * 70)

        result = compiled_graph.invoke({
            "customer_query": query,
            "category": "",
            "technical_analysis": "",
            "billing_info": "",
            "general_response": "",
            "final_response": ""
        })

        print(f"\n回复:\n{result['final_response']}\n")


# ============ 示例 6：多 Agent 最佳实践 ============

def example_6_best_practices():
    """示例 6：多 Agent 系统最佳实践"""
    print("=" * 70)
    print("示例 6：最佳实践")
    print("=" * 70)

    print("""
多 Agent 系统的设计原则：

1. 单一职责原则：
   ✓ 每个 Agent 专注于一个任务
   ✓ 避免功能重叠
   ✓ 明确的责任边界

2. 通信简洁：
   ✓ 定义清晰的消息格式
   ✓ 避免过度通信
   ✓ 批量传递信息

3. 容错设计：
   ✓ Agent 失败时的降级策略
   ✓ 超时处理
   ✓ 重试机制

4. 可观测性：
   ✓ 记录 Agent 的行为
   ✓ 追踪消息流
   ✓ 性能监控

5. 可扩展性：
   ✓ 易于添加新 Agent
   ✓ 支持动态配置
   ✓ 模块化设计

常见模式：

模式 1：专家团队（Expert Team）
┌─────────────┐
│   Manager   │（协调者）
└──────┬──────┘
       │
   ┌───┴────┬────┬────┐
   ▼        ▼    ▼    ▼
┌─────┐ ┌─────┐┌─────┐┌─────┐
│Expert││Expert││Expert││Expert│
│  A   ││  B   ││  C   ││  D   │
└─────┘ └─────┘└─────┘└─────┘

适用：需要专业知识的任务

模式 2：流水线（Pipeline）
┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
│Stage1│->│Stage2│->│Stage3│->│Stage4│
└──────┘  └──────┘  └──────┘  └──────┘

适用：顺序处理的任务

模式 3：竞争选择（Competition）
┌──────────┐
│ Problem  │
└─────┬────┘
      │
  ┌───┴───┬────┬────┐
  ▼       ▼    ▼    ▼
┌─────┐┌─────┐┌─────┐┌─────┐
│ Sol ││ Sol ││ Sol ││ Sol │
│  A  ││  B  ││  C  ││  D  │
└──┬──┘└──┬──┘└──┬──┘└──┬──┘
   │      │     │      │
   └──────┴─────┴──────┘
          │
          ▼
    ┌──────────┐
    │ Selector │
    └──────────┘

适用：需要多个方案比较

模式 4：层次管理（Hierarchy）
        ┌──────────┐
        │ Manager  │
        └─────┬────┘
              │
       ┌──────┴──────┐
       │             │
    ┌──┴──┐       ┌──┴──┐
    │Team A│       │Team B│
    └──┬──┘       └──┬──┘
       │             │
    ┌──┴────┐    ┌──┴────┐
    │Worker │    │Worker │
    │Agent 1│    │Agent 2│
    └───────┘    └───────┘

适用：大型复杂系统

实现建议：

1. 从简单开始：
   - 先实现 2-3 个 Agent
   - 逐步增加复杂度
   - 充分测试通信

2. 使用标准接口：
```python
class Agent(ABC):
    @abstractmethod
    def process(self, state: State) -> State:
        pass

    @abstractmethod
    def can_handle(self, state: State) -> bool:
        pass
```

3. 添加监控：
```python
def monitored_agent(agent_func):
    def wrapper(state):
        start_time = time.time()
        try:
            result = agent_func(state)
            duration = time.time() - start_time
            log_success(agent_func.__name__, duration)
            return result
        except Exception as e:
            log_error(agent_func.__name__, e)
            raise
    return wrapper
```

4. 测试策略：
   - 单元测试每个 Agent
   - 集成测试 Agent 交互
   - 压力测试整个系统
   - 模拟 Agent 失败场景
    """)


# 总结：核心概念
"""
【多 Agent 系统的核心概念】

1. 多 Agent 架构：
   - 协作式：Agent 合作完成任务
   - 竞争式：Agent 提供竞争方案
   - 层级式：管理 Agent 协调工作 Agent
   - 自组织式：Agent 自主协作

2. 通信模式：
   - 共享状态：简单直接
   - 消息传递：异步灵活
   - 事件驱动：松耦合
   - 黑板系统：复杂问题

3. 协作模式：
   - 流水线：顺序处理
   - 并行：独立执行
   - 分治：任务分解
   - 投票：集体决策

4. 设计原则：
   - 单一职责
   - 简洁通信
   - 容错设计
   - 可观测性
   - 可扩展性

5. 应用场景：
   - 客服系统：分类路由
   - 研发协作：多专家协作
   - 决策系统：多方案比较
   - 内容创作：分工创作

6. 最佳实践：
   - 从简单开始
   - 定义清晰接口
   - 添加监控日志
   - 充分测试
   - 文档完善

【下一步学习】

在 19-visualization-debug.py 中，你将学习：
- 图的可视化
- 调试工具
- 性能分析
- 日志和追踪
"""

if __name__ == "__main__":
    example_1_collaborative_agents()
    example_2_competitive_agents()
    example_3_hierarchical_agents()
    example_4_agent_communication()
    example_5_customer_service_system()
    example_6_best_practices()
