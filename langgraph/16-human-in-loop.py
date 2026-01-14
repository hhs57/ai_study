"""
LangGraph 学习 06：人机交互（Human-in-the-Loop）

知识点：
1. 中断图执行等待人工输入
2. 人工批准和审核
3. 动态修改状态
4. 恢复执行
5. 实际应用场景
"""

import sys
import io

# 设置标准输出为 UTF-8 编码
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Sequence, Literal
import operator


# ============ 示例 1：基础人机交互 ============

def example_1_basic_human_loop():
    """示例 1：基础的人机交互流程"""
    print("=" * 70)
    print("示例 1：基础人机交互 - 等待人工输入")
    print("=" * 70)

    class HumanLoopState(TypedDict):
        messages: Sequence[str]
        step: str
        pending_approval: bool

    def step1_generate(state: HumanLoopState) -> HumanLoopState:
        """步骤 1：生成内容"""
        print("  [步骤 1] 生成内容...")
        content = "这是一份需要审核的文档内容。"
        return {
            "messages": state["messages"] + [content],
            "step": "review",
            "pending_approval": True
        }

    def step2_review(state: HumanLoopState) -> HumanLoopState:
        """步骤 2：审核节点（需要人工介入）"""
        print("  [步骤 2] 等待人工审核...")
        print("  ⚠️  此时图执行暂停，等待人工批准")

        # 在实际应用中，这里会等待真实的用户输入
        # 这里我们模拟人工批准
        approved = True  # 假设人工批准了

        if approved:
            return {
                "messages": state["messages"] + ["人工审核: 已批准"],
                "step": "publish",
                "pending_approval": False
            }
        else:
            return {
                "messages": state["messages"] + ["人工审核: 已拒绝"],
                "step": "revise",
                "pending_approval": False
            }

    def step3_publish(state: HumanLoopState) -> HumanLoopState:
        """步骤 3：发布"""
        print("  [步骤 3] 发布内容...")
        return {"messages": state["messages"] + ["已发布！"], "step": "done"}

    def step3_revise(state: HumanLoopState) -> HumanLoopState:
        """步骤 3：修改"""
        print("  [步骤 3] 修改内容...")
        return {"messages": state["messages"] + ["已退回修改"], "step": "done"}

    # 构建图
    graph = StateGraph(HumanLoopState)

    graph.add_node("generate", step1_generate)
    graph.add_node("review", step2_review)
    graph.add_node("publish", step3_publish)
    graph.add_node("revise", step3_revise)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "review")

    # 根据审核结果决定下一步
    graph.add_conditional_edges(
        "review",
        lambda state: "publish" if state["step"] == "publish" else "revise",
        {
            "publish": "publish",
            "revise": "revise"
        }
    )

    graph.add_edge("publish", END)
    graph.add_edge("revise", END)

    compiled_graph = graph.compile()

    # 执行
    print("\n执行流程（包含人工审核步骤）:")
    result = compiled_graph.invoke({
        "messages": [],
        "step": "start",
        "pending_approval": False
    })

    print(f"\n最终状态:")
    print(f"  步骤: {result['step']}")
    print(f"  消息: {result['messages']}")


# ============ 示例 2：使用 interrupt() 函数 ============

def example_2_interrupt_function():
    """示例 2：使用 interrupt() 暂停执行"""
    print("=" * 70)
    print("示例 2：使用 interrupt() 函数")
    print("=" * 70)

    print("""
LangGraph 提供了 interrupt() 函数来暂停图执行。

实际使用示例：
```python
from langgraph.graph import StateGraph
from langgraph.types import interrupt

def approval_step(state):
    # 需要人工批准
    content = state["draft_content"]

    # 暂停执行，等待人工输入
    feedback = interrupt({
        "type": "approval",
        "content": content,
        "message": "请审核以下内容并提供反馈"
    })

    # 继续执行
    if feedback.get("approved"):
        return {"status": "approved", "final_content": content}
    else:
        return {
            "status": "rejected",
            "revisions": feedback.get("comments", "")
        }

# 使用方法
# 1. 执行到 interrupt 时暂停
config = {"configurable": {"thread_id": "session-1"}}
result = graph.invoke(initial_state, config)

# 2. 获取人工输入
human_input = get_human_input()

# 3. 恢复执行，传入人工输入
result = graph.invoke(
    None,  # 不需要新状态
    config,
    interrupt_value=human_input  # 传入人工输入
)
```

应用场景：
✓ 内容审核：文章、评论、邮件
✓ 决策批准：金融交易、采购审批
✓ 数据验证：关键数据录入
✓ 安全确认：删除、修改操作
    """)


# ============ 示例 3：多步骤人工审批流程 ============

def example_3_multi_step_approval():
    """示例 3：多级审批流程"""
    print("=" * 70)
    print("示例 3：多级审批流程")
    print("=" * 70)

    class ApprovalState(TypedDict):
        request_id: str
        amount: float
        department: str
        manager_approved: bool
        finance_approved: bool
        final_status: str
        history: list[str]

    def submit_request(state: ApprovalState) -> ApprovalState:
        """提交请求"""
        print("  [提交] 发起审批请求")
        req_id = f"REQ-{state['amount']}-{state['department']}"
        return {
            "request_id": req_id,
            "history": [f"提交请求: {req_id}"]
        }

    def manager_approval(state: ApprovalState) -> ApprovalState:
        """经理审批"""
        print(f"  [经理审批] 审批请求 {state['request_id']}")
        print("  ⚠️  等待经理批准...")

        # 模拟：金额 < 5000 自动批准
        approved = state['amount'] < 5000

        status = "经理批准" if approved else "经理拒绝"
        print(f"  [经理审批] 结果: {status}")

        return {
            "manager_approved": approved,
            "history": state["history"] + [status]
        }

    def finance_approval(state: ApprovalState) -> ApprovalState:
        """财务审批"""
        print(f"  [财务审批] 审批请求 {state['request_id']}")
        print("  ⚠️  等待财务批准...")

        # 模拟：金额 < 10000 自动批准
        approved = state['amount'] < 10000

        status = "财务批准" if approved else "财务拒绝"
        print(f"  [财务审批] 结果: {status}")

        return {
            "finance_approved": approved,
            "history": state["history"] + [status]
        }

    def final_decision(state: ApprovalState) -> ApprovalState:
        """最终决定"""
        if state["manager_approved"] and state["finance_approved"]:
            status = "已批准"
        else:
            status = "已拒绝"

        print(f"  [最终决定] {status}")
        return {
            "final_status": status,
            "history": state["history"] + [f"最终状态: {status}"]
        }

    # 构建图
    graph = StateGraph(ApprovalState)

    graph.add_node("submit", submit_request)
    graph.add_node("manager", manager_approval)
    graph.add_node("finance", finance_approval)
    graph.add_node("final", final_decision)

    graph.set_entry_point("submit")
    graph.add_edge("submit", "manager")

    # 经理批准后，根据金额决定是否需要财务审批
    graph.add_conditional_edges(
        "manager",
        lambda state: "finance" if state['amount'] >= 3000 and state['manager_approved'] else "final",
        {
            "finance": "finance",
            "final": "final"
        }
    )

    graph.add_edge("finance", "final")
    graph.add_edge("final", END)

    compiled_graph = graph.compile()

    # 测试不同金额的请求
    test_cases = [
        {"amount": 2000, "department": "研发部"},
        {"amount": 5000, "department": "市场部"},
        {"amount": 15000, "department": "销售部"}
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n测试案例 {i}: {case['amount']} 元 - {case['department']}")
        print("-" * 70)

        result = compiled_graph.invoke({
            "amount": case['amount'],
            "department": case['department'],
            "manager_approved": False,
            "finance_approved": False,
            "final_status": "",
            "history": []
        })

        print(f"\n结果:")
        print(f"  请求 ID: {result['request_id']}")
        print(f"  最终状态: {result['final_status']}")
        print(f"  审批历史:")
        for step in result['history']:
            print(f"    - {step}")


# ============ 示例 4：交互式调试 ============

def example_4_interactive_debugging():
    """示例 4：使用人机交互进行调试"""
    print("=" * 70)
    print("示例 4：交互式调试")
    print("=" * 70)

    print("""
人机交互在调试中的应用：

1. 检查中间状态：
```python
def debug_step(state):
    # 显示当前状态
    current_state = state

    # 暂停，让开发者检查
    action = interrupt({
        "type": "debug",
        "current_state": current_state,
        "message": "检查状态。继续？修改？"
    })

    if action == "continue":
        return state
    elif action == "modify":
        # 允许修改状态
        return action.get("new_state", state)
```

2. 测试不同分支：
```python
def test_branch(state):
    # 在关键分支点暂停
    branch_choice = interrupt({
        "type": "branch_test",
        "available_branches": ["branch_a", "branch_b", "branch_c"],
        "current_state": state
    })

    return {"branch": branch_choice}
```

3. 性能分析：
```python
def performance_checkpoint(state):
    # 在关键点暂停，记录性能
    import time
    checkpoint_time = time.time()

    action = interrupt({
        "type": "performance",
        "checkpoint": checkpoint_time,
        "message": "性能检查点"
    })

    return state
```

调试工作流：
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐
│ 执行步骤│ -> │ 暂停检查  │ -> │ 分析/修改 │ -> │ 继续执行 │
└─────────┘    └──────────┘    └──────────┘    └─────────┘
                      ↑                              |
                      └──────────────────────────────┘
                         多次循环调试

好处：
✓ 逐步检查每个节点的输出
✓ 在中间修改状态测试不同场景
✓ 不需要重新运行整个流程
✓ 实时理解系统行为
    """)


# ============ 示例 5：实际应用 - 内容审核系统 ============

def example_5_content_moderation_system():
    """示例 5：内容审核系统"""
    print("=" * 70)
    print("示例 5：内容审核系统")
    print("=" * 70)

    class ContentModerationState(TypedDict):
        content: str
        category: str
        ai_score: float
        human_review: str
        status: str
        history: list[str]

    def ai_classify(state: ContentModerationState) -> ContentModerationState:
        """AI 分类"""
        print("  [AI 分类] 分析内容...")

        content = state["content"]

        # 简单的规则分类
        if any(word in content.lower() for word in ["优惠", "折扣", "购买"]):
            category = "广告"
            score = 0.8
        elif any(word in content.lower() for word in ["骂", "坏", "差"]):
            category = "负面"
            score = 0.9
        else:
            category = "正常"
            score = 0.3

        print(f"  [AI 分类] 类别: {category}, 风险分数: {score}")

        return {
            "category": category,
            "ai_score": score,
            "history": state["history"] + [f"AI分类: {category} (分数: {score})"]
        }

    def decide_review(state: ContentModerationState) -> str:
        """决定是否需要人工审核"""
        # 高风险或不确定的内容需要人工审核
        if state["ai_score"] > 0.7:
            return "human_review"
        elif state["ai_score"] > 0.4:
            return "optional_review"
        else:
            return "auto_approve"

    def human_review_node(state: ContentModerationState) -> ContentModerationState:
        """人工审核"""
        print(f"  [人工审核] 内容: {state['content'][:50]}...")
        print("  ⚠️  需要人工审核员审核")

        # 模拟人工审核
        # 实际应用中，这里会等待真实的审核员输入
        review_result = "approve"  # 或 "reject" 或 "modify"

        status_map = {
            "approve": "已批准",
            "reject": "已拒绝",
            "modify": "需要修改"
        }

        print(f"  [人工审核] 结果: {status_map[review_result]}")

        return {
            "human_review": review_result,
            "status": status_map[review_result],
            "history": state["history"] + [f"人工审核: {status_map[review_result]}"]
        }

    def auto_approve_node(state: ContentModerationState) -> ContentModerationState:
        """自动批准"""
        print("  [自动批准] 内容安全，自动通过")
        return {
            "status": "自动批准",
            "history": state["history"] + ["自动批准"]
        }

    # 构建图
    graph = StateGraph(ContentModerationState)

    graph.add_node("classify", ai_classify)
    graph.add_node("human_review", human_review_node)
    graph.add_node("auto_approve", auto_approve_node)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify",
        decide_review,
        {
            "human_review": "human_review",
            "optional_review": "auto_approve",  # 可选审核，这里选择自动批准
            "auto_approve": "auto_approve"
        }
    )

    graph.add_edge("human_review", END)
    graph.add_edge("auto_approve", END)

    compiled_graph = graph.compile()

    # 测试不同类型的内容
    test_contents = [
        "这是一个很好的产品，非常推荐！",
        "限时优惠，点击购买，打折促销！",
        "这东西太差了，根本不好用！"
    ]

    for i, content in enumerate(test_contents, 1):
        print(f"\n测试内容 {i}: {content}")
        print("-" * 70)

        result = compiled_graph.invoke({
            "content": content,
            "category": "",
            "ai_score": 0.0,
            "human_review": "",
            "status": "",
            "history": []
        })

        print(f"\n结果:")
        print(f"  类别: {result['category']}")
        print(f"  AI 分数: {result['ai_score']}")
        print(f"  最终状态: {result['status']}")


# ============ 示例 6：人机交互最佳实践 ============

def example_6_best_practices():
    """示例 6：人机交互最佳实践"""
    print("=" * 70)
    print("示例 6：最佳实践")
    print("=" * 70)

    print("""
人机交互的最佳实践：

1. 设计清晰的交互点：
   ✓ 在关键决策点暂停
   ✓ 提供充分的上下文信息
   ✓ 明确说明需要什么输入
   ✓ 给出推荐的选项

2. 提供好的用户体验：
   ✓ 显示当前进度
   ✓ 提供历史记录
   ✓ 支持撤销和重做
   ✓ 超时处理机制

3. 状态管理：
   ✓ 保存所有中间状态
   ✓ 支持恢复和回滚
   ✓ 记录人工修改
   ✓ 审计追踪

4. 错误处理：
   ✓ 验证人工输入
   ✓ 提供清晰的错误信息
   ✓ 允许重试
   ✓ 优雅降级

5. 安全考虑：
   ✓ 验证用户权限
   ✓ 记录所有人工操作
   ✓ 敏感数据加密
   ✓ 防止未授权访问

6. 性能优化：
   ✓ 避免不必要的暂停
   ✓ 批量处理审核
   ✓ 异步处理
   ✓ 缓存常用决策

应用场景分类：

场景 1：内容审核
- 社交媒体帖子
- 评论和反馈
- 用户生成内容
- 自动化 vs 人工审核

场景 2：工作流审批
- 采购审批
- 费用报销
- 合同审批
- 多级审批流程

场景 3：数据标注
- 训练数据标注
- 异常数据确认
- 分类标签验证
- 人工校正

场景 4：决策支持
- 金融交易审核
- 风险评估确认
- 投资决策
- 策略调整

场景 5：调试和开发
- 流程调试
- 参数调优
- 错误诊断
- 性能分析

实现建议：

1. 使用检查点：
```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = StateGraph(MyState)
# ... 构建图
compiled = graph.compile(checkpointer=memory)

# 支持暂停和恢复
config = {"configurable": {"thread_id": "session-1"}}
result = compiled.invoke(state, config)
# ... 人工干预
result = compiled.invoke(None, config)  # 恢复执行
```

2. 记录完整历史：
```python
class AuditState(TypedDict):
    history: list[dict]  # 记录所有操作
    # ... 其他字段

def audit_step(state):
    # 记录每个步骤
    return {
        "history": state["history"] + [{
            "timestamp": datetime.now(),
            "action": "step_name",
            "user": "system or human",
            "changes": {...}
        }]
    }
```

3. 提供回滚机制：
```python
def checkpoint_state(state):
    # 保存检查点
    return {
        "checkpoints": state.get("checkpoints", []) + [state.copy()],
        "last_checkpoint": len(state.get("checkpoints", []))
    }

def rollback_to_checkpoint(state, checkpoint_id):
    # 回滚到指定检查点
    checkpoints = state.get("checkpoints", [])
    if 0 <= checkpoint_id < len(checkpoints):
        return checkpoints[checkpoint_id]
    return state
```
    """)


# 总结：核心概念
"""
【人机交互的核心概念】

1. interrupt() 函数：
   - 暂停图执行
   - 等待人工输入
   - 可以传入任何数据
   - 恢复时继续执行

2. 应用场景：
   - 内容审核和批准
   - 多级工作流审批
   - 数据标注和验证
   - 交互式调试
   - 决策确认

3. 检查点（Checkpoint）：
   - 保存执行状态
   - 支持暂停和恢复
   - 记录完整历史
   - 审计追踪

4. 状态管理：
   - 保存中间状态
   - 支持修改和回滚
   - 记录人工操作
   - 维护历史记录

5. 最佳实践：
   - 清晰的交互点设计
   - 充分的上下文信息
   - 良好的用户体验
   - 健壮的错误处理
   - 安全考虑

6. 实现模式：
   - 审批模式：多级审批
   - 标注模式：数据标注
   - 调试模式：逐步调试
   - 决策模式：关键决策

【下一步学习】

在 17-state-persistence.py 中，你将学习：
- 数据库持久化
- 跨会话状态管理
- 状态版本控制
- 分布式状态管理
"""

if __name__ == "__main__":
    example_1_basic_human_loop()
    example_2_interrupt_function()
    example_3_multi_step_approval()
    example_4_interactive_debugging()
    example_5_content_moderation_system()
    example_6_best_practices()
