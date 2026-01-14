"""
LangGraph 学习 10：错误处理与容错

知识点：
1. 错误处理策略
2. 重试机制
3. 降级处理
4. 容错设计
5. 生产环境的最佳实践
"""

import sys
import io
import random
import time
from typing import TypedDict, Optional
from enum import Enum

# 设置标准输出为 UTF-8 编码
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from langgraph.graph import StateGraph, END


# ============ 示例 1：基础错误处理 ============

def example_1_basic_error_handling():
    """示例 1：节点的错误处理"""
    print("=" * 70)
    print("示例 1：基础错误处理")
    print("=" * 70)

    class SafeState(TypedDict):
        input_data: str
        output_data: str
        error_message: Optional[str]
        status: str

    def safe_operation(state: SafeState) -> SafeState:
        """带错误处理的操作"""
        try:
            print("  [操作] 执行操作...")
            data = state["input_data"]

            # 模拟可能出现的错误
            if not data:
                raise ValueError("输入数据为空")

            if len(data) > 100:
                raise ValueError("输入数据过长")

            # 正常处理
            result = f"处理成功: {data}"
            print(f"  [操作] {result}")

            return {
                "output_data": result,
                "status": "success"
            }

        except ValueError as e:
            error_msg = f"验证错误: {str(e)}"
            print(f"  [操作] ⚠️ {error_msg}")

            return {
                "error_message": error_msg,
                "status": "error"
            }

        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            print(f"  [操作] ❌ {error_msg}")

            return {
                "error_message": error_msg,
                "status": "error"
            }

    # 构建图
    graph = StateGraph(SafeState)
    graph.add_node("operation", safe_operation)
    graph.set_entry_point("operation")
    graph.add_edge("operation", END)

    compiled_graph = graph.compile()

    # 测试不同场景
    test_cases = [
        {"input_data": "正常数据"},
        {"input_data": ""},
        {"input_data": "x" * 150}
    ]

    for i, test_input in enumerate(test_cases, 1):
        print(f"\n测试案例 {i}:")
        print("-" * 70)

        result = compiled_graph.invoke({
            "input_data": test_input["input_data"],
            "output_data": "",
            "error_message": None,
            "status": "pending"
        })

        print(f"\n结果:")
        print(f"  状态: {result['status']}")
        print(f"  输出: {result['output_data']}")
        print(f"  错误: {result['error_message']}")


# ============ 示例 2：重试机制 ============

def example_2_retry_mechanism():
    """示例 2：实现重试逻辑"""
    print("=" * 70)
    print("示例 2：重试机制")
    print("=" * 70)

    class RetryState(TypedDict):
        attempt: int
        max_attempts: int
        success: bool
        result: str
        error_history: list[str]

    def unreliable_operation(state: RetryState) -> RetryState:
        """不可靠的操作（可能失败）"""
        attempt = state["attempt"] + 1
        print(f"  [操作] 尝试 {attempt}/{state['max_attempts']}")

        # 模拟随机失败（70% 失败率）
        success = random.random() > 0.7

        if success:
            print(f"  [操作] ✓ 成功！")
            return {
                "attempt": attempt,
                "success": True,
                "result": "操作成功完成"
            }
        else:
            error = f"尝试 {attempt} 失败"
            print(f"  [操作] ✗ {error}")
            return {
                "attempt": attempt,
                "success": False,
                "error_history": state["error_history"] + [error]
            }

    def should_retry(state: RetryState) -> str:
        """决定是否重试"""
        if state["success"]:
            return "success"
        elif state["attempt"] < state["max_attempts"]:
            return "retry"
        else:
            return "give_up"

    def success_node(state: RetryState) -> RetryState:
        """成功节点"""
        print("  [成功] 操作成功完成")
        return state

    def give_up_node(state: RetryState) -> RetryState:
        """放弃节点"""
        print("  [放弃] 达到最大重试次数，放弃")
        return {
            "result": f"操作失败，已重试 {state['attempt']} 次"
        }

    # 构建图
    graph = StateGraph(RetryState)
    graph.add_node("operation", unreliable_operation)
    graph.add_node("success", success_node)
    graph.add_node("give_up", give_up_node)

    graph.set_entry_point("operation")

    graph.add_conditional_edges(
        "operation",
        should_retry,
        {
            "retry": "operation",
            "success": "success",
            "give_up": "give_up"
        }
    )

    graph.add_edge("success", END)
    graph.add_edge("give_up", END)

    compiled_graph = graph.compile()

    # 执行多次测试
    print("\n执行重试测试（运行 3 次）:")
    print("=" * 70)

    for i in range(3):
        print(f"\n第 {i+1} 次运行:")

        result = compiled_graph.invoke({
            "attempt": 0,
            "max_attempts": 5,
            "success": False,
            "result": "",
            "error_history": []
        })

        print(f"\n最终结果: {result['result']}")
        print(f"总尝试次数: {result['attempt']}")


# ============ 示例 3：降级处理 ============

def example_3_graceful_degradation():
    """示例 3：优雅降级"""
    print("=" * 70)
    print("示例 3：优雅降级")
    print("=" * 70)

    class DegradationState(TypedDict):
        query: str
        premium_result: str
        standard_result: str
        cached_result: str
        final_result: str
        level: str

    def premium_api(state: DegradationState) -> DegradationState:
        """高级 API（可能失败）"""
        print("  [高级API] 尝试调用高级服务...")

        # 模拟 50% 失败率
        if random.random() > 0.5:
            result = "高级服务结果：详细、准确的信息"
            print("  [高级API] ✓ 成功")
            return {"premium_result": result, "level": "premium"}
        else:
            print("  [高级API] ✗ 服务不可用")
            return {"level": "fallback_standard"}

    def standard_api(state: DegradationState) -> DegradationState:
        """标准 API（备用方案）"""
        print("  [标准API] 使用标准服务...")

        # 模拟 30% 失败率
        if random.random() > 0.3:
            result = "标准服务结果：基本信息"
            print("  [标准API] ✓ 成功")
            return {"standard_result": result, "level": "standard"}
        else:
            print("  [标准API] ✗ 服务不可用")
            return {"level": "fallback_cache"}

    def cache_api(state: DegradationState) -> DegradationState:
        """缓存 API（最后方案）"""
        print("  [缓存] 使用缓存数据...")
        result = "缓存结果：可能过时的信息"
        print("  [缓存] ✓ 从缓存恢复")
        return {"cached_result": result, "level": "cache"}

    def aggregator(state: DegradationState) -> DegradationState:
        """聚合器：选择最佳可用结果"""
        level = state["level"]

        if level == "premium":
            result = state["premium_result"]
        elif level == "standard":
            result = state["standard_result"]
        else:
            result = state["cached_result"]

        print(f"  [聚合] 使用 {level} 级别的结果")

        return {"final_result": result}

    # 构建降级流程
    graph = StateGraph(DegradationState)
    graph.add_node("premium", premium_api)
    graph.add_node("standard", standard_api)
    graph.add_node("cache", cache_api)
    graph.add_node("aggregate", aggregator)

    graph.set_entry_point("premium")

    graph.add_conditional_edges(
        "premium",
        lambda state: state["level"],
        {
            "premium": "aggregate",
            "fallback_standard": "standard"
        }
    )

    graph.add_conditional_edges(
        "standard",
        lambda state: state["level"],
        {
            "standard": "aggregate",
            "fallback_cache": "cache"
        }
    )

    graph.add_edge("cache", "aggregate")
    graph.add_edge("aggregate", END)

    compiled_graph = graph.compile()

    # 测试降级
    print("\n测试降级机制（运行 3 次）:")
    print("=" * 70)

    for i in range(3):
        print(f"\n第 {i+1} 次运行:")

        result = compiled_graph.invoke({
            "query": "什么是人工智能？",
            "premium_result": "",
            "standard_result": "",
            "cached_result": "",
            "final_result": "",
            "level": ""
        })

        print(f"\n结果级别: {result['level']}")
        print(f"最终结果: {result['final_result']}\n")


# ============ 示例 4：容错设计模式 ============

def example_4_fault_tolerance_patterns():
    """示例 4：容错设计模式"""
    print("=" * 70)
    print("示例 4：容错设计模式")
    print("=" * 70)

    print("""
常见的容错设计模式：

1. 断路器模式（Circuit Breaker）：
   防止级联失败

   状态：Closed -> Open -> Half-Open

   ```
   Closed（正常）:
   - 请求正常通过
   - 失败率超过阈值 -> Open

   Open（断路）:
   - 快速失败
   - 不调用实际服务
   - 超时后 -> Half-Open

   Half-Open（试探）:
   - 允许少量请求通过
   - 成功 -> Closed
   - 失败 -> Open
   ```

2. 超时模式（Timeout）：
   防止无限等待

   ```python
   def with_timeout(func, timeout_seconds=5):
       start = time.time()
       while time.time() - start < timeout_seconds:
           try:
               return func()
           except TimeoutError:
               continue
       raise TimeoutError(f"操作超时（{timeout_seconds}秒）")
   ```

3. 舱壁隔离（Bulkhead）：
   资源隔离

   ```python
   # 不同服务使用独立的资源池
   service_a_pool = ResourcePool(max_connections=10)
   service_b_pool = ResourcePool(max_connections=5)

   # 服务 A 的失败不影响服务 B
   ```

4. 重试模式（Retry）：
   指数退避

   ```python
   def retry_with_backoff(func, max_retries=3):
       for attempt in range(max_retries):
           try:
               return func()
           except Exception as e:
               if attempt < max_retries - 1:
                   wait_time = 2 ** attempt  # 1s, 2s, 4s
                   time.sleep(wait_time)
               else:
                   raise e
   ```

5. 备用模式（Fallback）：
   多个备用方案

   ```
   Primary -> Secondary -> Tertiary -> Default
   ```

6. 隔离模式（Isolation）：
   故障隔离

   ```python
   # 使用独立的进程/线程/容器
   isolated_service = run_in_isolation(
       service_func,
       timeout=10,
       memory_limit="1GB"
   )
   ```

容错模式选择：
┌──────────────┬────────┬──────┬────────┐
│ 模式         │ 复杂度 │成本  │ 适用场景│
├──────────────┼────────┼──────┼────────┤
│ 断路器       │ 中     │ 低   │ 外部API│
│ 超时         │ 低     │ 低   │ 任何操作│
│ 舱壁隔离     │ 高     │ 高   │ 关键系统│
│ 重试         │ 低     │ 低   │ 临时故障│
│ 备用         │ 中     │ 中   │ 多个服务│
│ 隔离         │ 高     │ 高   │ 核心功能│
└──────────────┴────────┴──────┴────────┘
    """)


# ============ 示例 5：综合容错系统 ============

def example_5_comprehensive_fault_tolerance():
    """示例 5：综合容错系统"""
    print("=" * 70)
    print("示例 5：综合容错系统")
    print("=" * 70)

    class RobustState(TypedDict):
        task: str
        attempt: int
        result: str
        fallback_used: bool
        errors: list[str]
        status: str

    def robust_operation(state: RobustState) -> RobustState:
        """带完整容错的操作"""
        attempt = state["attempt"] + 1

        try:
            print(f"  [操作] 尝试 {attempt}")

            # 模拟不同类型的错误
            failure_type = random.choice([
                None,  # 成功
                "timeout",  # 超时
                "connection",  # 连接错误
                "rate_limit",  # 限流
                "server_error"  # 服务器错误
            ])

            if failure_type is None:
                print(f"  [操作] ✓ 成功")
                return {
                    "attempt": attempt,
                    "result": "操作成功",
                    "status": "success"
                }

            elif failure_type == "timeout":
                raise TimeoutError("请求超时")

            elif failure_type == "connection":
                raise ConnectionError("无法连接到服务")

            elif failure_type == "rate_limit":
                raise Exception("API 限流")

            else:
                raise Exception(f"服务器错误: {failure_type}")

        except TimeoutError as e:
            error_msg = f"超时错误: {str(e)}"
            print(f"  [操作] ⚠️ {error_msg}")
            return {
                "attempt": attempt,
                "errors": state["errors"] + [error_msg],
                "status": "timeout"
            }

        except ConnectionError as e:
            error_msg = f"连接错误: {str(e)}"
            print(f"  [操作] ⚠️ {error_msg}")
            return {
                "attempt": attempt,
                "errors": state["errors"] + [error_msg],
                "status": "connection_error"
            }

        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            print(f"  [操作] ❌ {error_msg}")
            return {
                "attempt": attempt,
                "errors": state["errors"] + [error_msg],
                "status": "error"
            }

    def handle_error(state: RobustState) -> str:
        """错误处理路由"""
        if state["status"] == "success":
            return "done"
        elif state["attempt"] < 3:
            return "retry"
        elif state["status"] in ["timeout", "connection_error"]:
            return "use_fallback"
        else:
            return "fail"

    def fallback_operation(state: RobustState) -> RobustState:
        """备用方案"""
        print("  [备用] 使用备用服务")
        return {
            "result": "备用服务结果",
            "fallback_used": True,
            "status": "fallback_success"
        }

    def fail_node(state: RobustState) -> RobustState:
        """失败节点"""
        print("  [失败] 所有方案失败")
        return {
            "result": "操作失败",
            "status": "failed"
        }

    def done_node(state: RobustState) -> RobustState:
        """完成节点"""
        print(f"  [完成] {state['result']}")
        return state

    # 构建容错流程
    graph = StateGraph(RobustState)
    graph.add_node("operation", robust_operation)
    graph.add_node("fallback", fallback_operation)
    graph.add_node("fail", fail_node)
    graph.add_node("done", done_node)

    graph.set_entry_point("operation")

    graph.add_conditional_edges(
        "operation",
        handle_error,
        {
            "retry": "operation",
            "use_fallback": "fallback",
            "fail": "fail",
            "done": "done"
        }
    )

    graph.add_edge("fallback", "done")
    graph.add_edge("fail", END)
    graph.add_edge("done", END)

    compiled_graph = graph.compile()

    # 测试
    print("\n测试综合容错系统:")
    print("=" * 70)

    result = compiled_graph.invoke({
        "task": "测试任务",
        "attempt": 0,
        "result": "",
        "fallback_used": False,
        "errors": [],
        "status": "pending"
    })

    print(f"\n最终状态: {result['status']}")
    print(f"结果: {result['result']}")
    print(f"尝试次数: {result['attempt']}")
    print(f"使用备用: {result['fallback_used']}")
    print(f"错误历史: {result['errors']}")


# ============ 示例 6：生产环境最佳实践 ============

def example_6_production_best_practices():
    """示例 6：生产环境最佳实践"""
    print("=" * 70)
    print("示例 6：生产环境最佳实践")
    print("=" * 70)

    print("""
生产环境的错误处理清单：

1. 错误分类：
   ✓ 可重试错误（临时故障）
   ✓ 不可重试错误（永久故障）
   ✓ 业务错误（预期内的错误）
   ✓ 系统错误（意外故障）

2. 重试策略：
   ✓ 指数退避（1s, 2s, 4s, 8s...）
   ✓ 最大重试次数（通常 3-5 次）
   ✓ 只重试幂等操作
   ✓ 记录重试日志

3. 超时设置：
   ✓ 连接超时（3-5 秒）
   ✓ 读取超时（10-30 秒）
   ✓ 总超时（60 秒）
   ✓ 根据操作类型调整

4. 降级策略：
   ✓ 功能降级（关闭非关键功能）
   ✓ 服务降级（使用备用服务）
   ✓ 数据降级（使用缓存数据）
   ✓ 体验降级（简化交互）

5. 监控告警：
   ✓ 错误率监控
   ✓ 延迟监控
   ✓ 可用性监控
   ✓ 异常检测

6. 恢复策略：
   ✓ 自动恢复
   ✓ 人工介入
   ✓ 灾难恢复
   ✓ 数据备份

7. 测试验证：
   ✓ 混沌工程
   ✓ 故障注入
   ✓ 压力测试
   ✓ 恢复测试

实际代码模板：

```python
class RobustNode:
    def __init__(self, max_retries=3, timeout=10):
        self.max_retries = max_retries
        self.timeout = timeout
        self.circuit_breaker = CircuitBreaker()

    def __call__(self, state):
        # 断路器检查
        if self.circuit_breaker.is_open():
            return self.fallback(state)

        # 重试逻辑
        for attempt in range(self.max_retries):
            try:
                # 超时控制
                result = self.with_timeout(
                    lambda: self.execute(state),
                    self.timeout
                )

                # 成功，重置断路器
                self.circuit_breaker.reset()
                return result

            except RetryableError as e:
                if attempt < self.max_retries - 1:
                    # 指数退避
                    time.sleep(2 ** attempt)
                    continue
                else:
                    # 重试失败，打开断路器
                    self.circuit_breaker.open()
                    return self.fallback(state)

            except NonRetryableError as e:
                # 不可重试，直接失败
                return self.handle_error(e, state)
```

错误处理最佳实践：

DO（推荐做法）:
✓ 区分错误类型
✓ 实现适当的重试
✓ 设置超时限制
✓ 提供降级方案
✓ 记录详细日志
✓ 监控错误指标
✓ 定期测试容错
✓ 文档化错误处理

DON'T（避免）:
✗ 吞掉异常
✗ 无限重试
✗ 无超时限制
✗ 无降级方案
✗ 无错误日志
✗ 无监控告警
✗ 无测试验证
✗ 无文档说明

总结：
- 预防优于治疗
- 设计时考虑故障
- 测试时模拟故障
- 运行时监控故障
- 恢复时快速响应
    """)


# 总结：核心概念
"""
【错误处理与容错的核心概念】

1. 错误处理：
   - try-except 捕获异常
   - 区分错误类型
   - 记录错误信息
   - 返回错误状态

2. 重试机制：
   - 最大重试次数
   - 指数退避
   - 只重试幂等操作
   - 避免级联失败

3. 降级处理：
   - 功能降级
   - 服务降级
   - 数据降级
   - 多级备用方案

4. 容错模式：
   - 断路器模式
   - 超时模式
   - 舱壁隔离
   - 备用模式
   - 隔离模式

5. 生产实践：
   - 错误分类
   - 监控告警
   - 恢复策略
   - 测试验证
   - 文档记录

6. 设计原则：
   - 快速失败
   - 优雅降级
   - 故障隔离
   - 可观测性
   - 自动恢复

【完整课程总结】

恭喜你完成了所有 20 个课程！

LangChain 部分（01-15）:
✓ 基础链和 LCEL
✓ 提示词模板
✓ 链的组合
✓ 对话记忆
✓ Agents 和工具
✓ 文档加载和处理
✓ 向量存储和 RAG
✓ 高级输出解析器
✓ 回调和流式输出
✓ 完整的 RAG 应用

LangGraph 部分（06-10, 16-20）:
✓ LangGraph 基础
✓ 状态管理
✓ 条件边
✓ 循环
✓ 智能体
✓ 人机交互
✓ 状态持久化
✓ 多 Agent 系统
✓ 可视化与调试
✓ 错误处理与容错

你现在已经掌握了：
- LangChain 和 LangGraph 的核心概念
- 如何构建 RAG 应用
- 如何设计复杂的工作流
- 如何处理错误和容错
- 生产环境的最佳实践

下一步建议：
1. 实践项目：构建一个完整的 RAG 应用
2. 深入学习：探索高级特性和优化
3. 关注社区：了解最新发展和最佳实践
4. 贡献开源：分享你的经验和代码

祝你在 LangChain 和 LangGraph 的学习和应用中取得成功！🎉
"""

if __name__ == "__main__":
    example_1_basic_error_handling()
    example_2_retry_mechanism()
    example_3_graceful_degradation()
    example_4_fault_tolerance_patterns()
    example_5_comprehensive_fault_tolerance()
    example_6_production_best_practices()
