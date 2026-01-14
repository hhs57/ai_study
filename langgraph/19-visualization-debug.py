"""
LangGraph 学习 09：可视化与调试

知识点：
1. 图结构可视化
2. 执行过程追踪
3. 调试工具
4. 性能分析
5. 日志和监控
"""

import sys
import io
from typing import TypedDict
import time

# 设置标准输出为 UTF-8 编码
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from langgraph.graph import StateGraph, END


# ============ 示例 1：图结构可视化 ============

def example_1_visualize_graph():
    """示例 1：可视化图结构"""
    print("=" * 70)
    print("示例 1：图结构可视化")
    print("=" * 70)

    class SimpleState(TypedDict):
        value: int
        history: list[str]

    def node_a(state: SimpleState) -> SimpleState:
        return {"value": state["value"] + 1, "history": state["history"] + ["A"]}

    def node_b(state: SimpleState) -> SimpleState:
        return {"value": state["value"] * 2, "history": state["history"] + ["B"]}

    def node_c(state: SimpleState) -> SimpleState:
        return {"value": state["value"] + 10, "history": state["history"] + ["C"]}

    # 构建图
    graph = StateGraph(SimpleState)
    graph.add_node("A", node_a)
    graph.add_node("B", node_b)
    graph.add_node("C", node_c)

    graph.set_entry_point("A")
    graph.add_conditional_edges(
        "A",
        lambda state: "B" if state["value"] < 5 else "C",
        {"B": "B", "C": "C"}
    )
    graph.add_edge("B", END)
    graph.add_edge("C", END)

    compiled_graph = graph.compile()

    print("""
图结构可视化：

方式 1：打印图信息
```python
print(compiled_graph.get_graph().print_ascii())
```

输出示例：
┌─────────┐
│   A     │
└────┬────┘
     │
     ├─────┐
     │     │
┌────▼─┐ ┌▼──────┐
│  B   │ │   C   │
└──────┘ └───────┘

方式 2：使用 Mermaid
```python
from IPython.display import Markdown, display

mermaid_code = compiled_graph.get_graph().draw_mermaid()
display(Markdown(mermaid_code))
```

方式 3：导出为图片
```python
from langgraph.visualization import visualize_graph

visualize_graph(compiled_graph, output_path="graph.png")
```

当前图的节点和边：
- 节点：A, B, C
- 入口：A
- 条件边：A -> (B 或 C，取决于值)
- 终点：B, C -> END
    """)


# ============ 示例 2：执行追踪 ============

def example_2_execution_tracing():
    """示例 2：追踪执行过程"""
    print("=" * 70)
    print("示例 2：执行追踪")
    print("=" * 70)

    class TracedState(TypedDict):
        step: int
        value: str
        trace: list[dict]

    def step_1(state: TracedState) -> TracedState:
        trace_entry = {
            "step": 1,
            "node": "step_1",
            "timestamp": time.time(),
            "input": state.copy()
        }
        print("  [追踪] 进入 step_1")
        result = {"step": 1, "value": "步骤1完成"}
        trace_entry["output"] = result
        return {"step": 1, "value": result["value"], "trace": state["trace"] + [trace_entry]}

    def step_2(state: TracedState) -> TracedState:
        trace_entry = {
            "step": 2,
            "node": "step_2",
            "timestamp": time.time(),
            "input": state.copy()
        }
        print("  [追踪] 进入 step_2")
        result = {"step": 2, "value": "步骤2完成"}
        trace_entry["output"] = result
        return {"step": 2, "value": result["value"], "trace": state["trace"] + [trace_entry]}

    def step_3(state: TracedState) -> TracedState:
        trace_entry = {
            "step": 3,
            "node": "step_3",
            "timestamp": time.time(),
            "input": state.copy()
        }
        print("  [追踪] 进入 step_3")
        result = {"step": 3, "value": "步骤3完成"}
        trace_entry["output"] = result
        return {"step": 3, "value": result["value"], "trace": state["trace"] + [trace_entry]}

    # 构建图
    graph = StateGraph(TracedState)
    graph.add_node("step1", step_1)
    graph.add_node("step2", step_2)
    graph.add_node("step3", step_3)

    graph.set_entry_point("step1")
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", "step3")
    graph.add_edge("step3", END)

    compiled_graph = graph.compile()

    # 执行并追踪
    print("\n执行流程（带追踪）:")
    print("-" * 70)

    result = compiled_graph.invoke({
        "step": 0,
        "value": "",
        "trace": []
    })

    print("\n执行追踪日志:")
    print("-" * 70)
    for i, entry in enumerate(result["trace"], 1):
        print(f"\n追踪记录 {i}:")
        print(f"  节点: {entry['node']}")
        print(f"  输入: {entry['input']}")
        print(f"  输出: {entry['output']}")
        print(f"  时间戳: {entry['timestamp']}")


# ============ 示例 3：性能分析 ============

def example_3_performance_analysis():
    """示例 3：性能分析工具"""
    print("=" * 70)
    print("示例 3：性能分析")
    print("=" * 70)

    class PerformanceState(TypedDict):
        data: str
        metrics: dict

    def slow_node(state: PerformanceState) -> PerformanceState:
        """模拟慢节点"""
        start = time.time()
        print("  [性能] 执行慢节点...")
        time.sleep(0.1)  # 模拟耗时操作
        duration = time.time() - start

        return {
            "data": "处理完成",
            "metrics": {
                **state.get("metrics", {}),
                "slow_node_duration": duration
            }
        }

    def fast_node(state: PerformanceState) -> PerformanceState:
        """模拟快节点"""
        start = time.time()
        print("  [性能] 执行快节点...")
        time.sleep(0.01)  # 模拟快速操作
        duration = time.time() - start

        return {
            "metrics": {
                **state.get("metrics", {}),
                "fast_node_duration": duration
            }
        }

    # 构建图
    graph = StateGraph(PerformanceState)
    graph.add_node("slow", slow_node)
    graph.add_node("fast", fast_node)

    graph.set_entry_point("slow")
    graph.add_edge("slow", "fast")
    graph.add_edge("fast", END)

    compiled_graph = graph.compile()

    # 执行并收集性能指标
    print("\n执行性能测试:")
    print("-" * 70)

    start_time = time.time()
    result = compiled_graph.invoke({
        "data": "",
        "metrics": {}
    })
    total_time = time.time() - start_time

    print(f"\n性能分析结果:")
    print(f"  总执行时间: {total_time:.3f} 秒")
    print(f"  慢节点耗时: {result['metrics']['slow_node_duration']:.3f} 秒")
    print(f"  快节点耗时: {result['metrics']['fast_node_duration']:.3f} 秒")
    print(f"  节点开销: {total_time - result['metrics']['slow_node_duration'] - result['metrics']['fast_node_duration']:.3f} 秒")


# ============ 示例 4：调试工具 ============

def example_4_debugging_tools():
    """示例 4：调试工具和技巧"""
    print("=" * 70)
    print("示例 4：调试工具")
    print("=" * 70)

    print("""
LangGraph 调试工具箱：

1. 使用 verbose 模式：
```python
# 编译时启用详细日志
compiled = graph.compile(debug=True)

# 执行时查看详细信息
result = compiled.invoke(state)
```

2. 状态快照：
```python
def snapshot_node(state):
    # 在关键点保存状态快照
    snapshot = state.copy()
    print(f"快照: {snapshot}")
    # 可以保存到文件或数据库
    return state
```

3. 条件断点：
```python
def debug_node(state):
    # 只在特定条件下暂停
    if state.get("debug_mode"):
        import pdb
        pdb.set_trace()  # 设置断点
    return process(state)
```

4. 日志记录：
```python
import logging

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("langgraph")

def logged_node(state):
    logger.debug(f"输入状态: {state}")
    result = process(state)
    logger.debug(f"输出状态: {result}")
    return result
```

5. 状态验证：
```python
def validate_state(state):
    # 验证状态完整性
    required_fields = ["field1", "field2", "field3"]
    missing = [f for f in required_fields if f not in state]

    if missing:
        raise ValueError(f"缺少必需字段: {missing}")

    # 验证数据类型
    if not isinstance(state["field1"], str):
        raise TypeError("field1 必须是字符串")

    return state
```

6. 性能分析器：
```python
import cProfile

def profile_graph_execution(graph, state):
    # 分析性能瓶颈
    profiler = cProfile.Profile()
    profiler.enable()

    result = graph.invoke(state)

    profiler.disable()
    profiler.print_stats(sort='cumulative')

    return result
```

7. 可视化执行流：
```python
def visualize_execution(compiled_graph, state):
    # 显示执行路径
    print("执行路径:")

    # 使用 stream 获取每步输出
    for output in compiled_graph.stream(state):
        for node_name, node_output in output.items():
            print(f"  -> {node_name}")
            print(f"     输出: {node_output}")
```

调试技巧：

✓ 从简单开始：先用最小化的图测试
✓ 分步验证：逐个节点测试
✓ 打印中间状态：在每个节点打印状态
✓ 使用测试数据：用已知的输入测试
✓ 隔离问题：移除可疑的节点
✓ 版本控制：保存工作的版本
✓ 文档化：记录已知问题和解决方案

常见问题诊断：

问题：图执行卡住
→ 检查循环条件
→ 查看是否有死循环
→ 验证条件边逻辑

问题：状态丢失
→ 检查状态更新
→ 验证返回值
→ 确认检查点配置

问题：性能差
→ 分析每个节点耗时
→ 查找瓶颈
→ 优化慢节点

问题：输出不符合预期
→ 打印每个节点的输入输出
→ 验证状态传递
→ 检查条件边逻辑
    """)


# ============ 示例 5：监控和日志 ============

def example_5_monitoring():
    """示例 5：监控系统"""
    print("=" * 70)
    print("示例 5：监控和日志")
    print("=" * 70)

    print("""
生产环境监控系统：

1. 指标收集：
```python
class MetricsCollector:
    def __init__(self):
        self.metrics = {
            "invocations": 0,
            "errors": 0,
            "latencies": [],
            "node_executions": {}
        }

    def record_invocation(self):
        self.metrics["invocations"] += 1

    def record_error(self):
        self.metrics["errors"] += 1

    def record_latency(self, duration):
        self.metrics["latencies"].append(duration)

    def record_node_execution(self, node_name, duration):
        if node_name not in self.metrics["node_executions"]:
            self.metrics["node_executions"][node_name] = []
        self.metrics["node_executions"][node_name].append(duration)

    def get_summary(self):
        return {
            "total_invocations": self.metrics["invocations"],
            "error_rate": self.metrics["errors"] / max(self.metrics["invocations"], 1),
            "avg_latency": sum(self.metrics["latencies"]) / len(self.metrics["latencies"]) if self.metrics["latencies"] else 0,
            "node_stats": {
                node: {
                    "calls": len(times),
                    "avg_duration": sum(times) / len(times) if times else 0
                }
                for node, times in self.metrics["node_executions"].items()
            }
        }
```

2. 日志系统：
```python
import logging
from logging.handlers import RotatingFileHandler

# 配置日志
logger = logging.getLogger("langgraph")
logger.setLevel(logging.INFO)

# 文件处理器（自动轮转）
file_handler = RotatingFileHandler(
    "langgraph.log",
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(file_handler)

# 使用日志
def logged_node(state):
    logger.info(f"节点开始执行: {__name__}")
    try:
        result = process(state)
        logger.info("节点执行成功")
        return result
    except Exception as e:
        logger.error(f"节点执行失败: {e}", exc_info=True)
        raise
```

3. 告警系统：
```python
class AlertManager:
    def __init__(self):
        self.thresholds = {
            "error_rate": 0.05,  # 5% 错误率
            "latency_p95": 5.0,  # 95分位延迟 5秒
            "node_failure_rate": 0.1  # 10% 节点失败率
        }

    def check_metrics(self, metrics):
        alerts = []

        if metrics["error_rate"] > self.thresholds["error_rate"]:
            alerts.append(f"错误率过高: {metrics['error_rate']:.2%}")

        if metrics["avg_latency"] > self.thresholds["latency_p95"]:
            alerts.append(f"平均延迟过高: {metrics['avg_latency']:.2f}s")

        return alerts

    def send_alert(self, alert):
        # 发送告警（邮件、Slack、PagerDuty等）
        print(f"⚠️ 告警: {alert}")
        # 实际实现可以集成告警服务
```

4. 实时监控面板：
```python
# 使用 Prometheus + Grafana
from prometheus_client import Counter, Histogram, start_http_server

# 定义指标
invocation_counter = Counter('graph_invocations', 'Total graph invocations')
error_counter = Counter('graph_errors', 'Total errors')
latency_histogram = Histogram('graph_latency_seconds', 'Graph execution latency')

# 在代码中使用
def monitored_node(state):
    invocation_counter.inc()
    with latency_histogram.time():
        try:
            result = process(state)
            return result
        except Exception as e:
            error_counter.inc()
            raise

# 启动监控服务器
start_http_server(8000)  # http://localhost:8000/metrics
```

监控最佳实践：

✓ 收集关键指标：调用次数、错误率、延迟
✓ 设置合理阈值：基于基线数据
✓ 及时告警：发现问题立即通知
✓ 定期审查：每周分析监控数据
✓ 持续优化：根据监控数据改进

监控指标示例：
┌──────────────────┬─────────────┬──────────┐
│ 指标             │ 目标值      │ 告警阈值 │
├──────────────────┼─────────────┼──────────┤
│ 错误率           │ < 1%        │ > 5%     │
│ 平均延迟         │ < 2s        │ > 5s     │
│ P95 延迟         │ < 5s        │ > 10s    │
│ 节点失败率       │ < 0.1%      │ > 1%     │
│ 内存使用         │ < 1GB       │ > 2GB    │
│ CPU 使用率       │ < 70%       │ > 90%    │
└──────────────────┴─────────────┴──────────┘
    """)


# ============ 示例 6：调试实战 ============

def example_6_debugging_practice():
    """示例 6：调试实战案例"""
    print("=" * 70)
    print("示例 6：调试实战")
    print("=" * 70)

    class DebugState(TypedDict):
        input: int
        current: int
        history: list[str]
        errors: list[str]

    def process_node(state: DebugState) -> DebugState:
        """处理节点（可能有 bug）"""
        try:
            print(f"  [调试] 处理输入: {state['input']}")

            # 模拟一个 bug
            if state['input'] == 0:
                raise ValueError("输入不能为 0")

            result = state['input'] * 2
            print(f"  [调试] 计算结果: {result}")

            return {
                "current": result,
                "history": state["history"] + [f"处理: {state['input']} -> {result}"]
            }

        except Exception as e:
            error_msg = f"处理节点错误: {str(e)}"
            print(f"  [调试] ⚠️ {error_msg}")
            return {
                "errors": state["errors"] + [error_msg]
            }

    # 构建图
    graph = StateGraph(DebugState)
    graph.add_node("process", process_node)
    graph.set_entry_point("process")
    graph.add_edge("process", END)

    compiled_graph = graph.compile()

    # 测试不同场景
    test_cases = [
        {"input": 5},
        {"input": 0},
        {"input": -3}
    ]

    print("\n调试测试:")
    print("=" * 70)

    for i, test_input in enumerate(test_cases, 1):
        print(f"\n测试案例 {i}: input = {test_input['input']}")
        print("-" * 70)

        result = compiled_graph.invoke({
            "input": test_input['input'],
            "current": 0,
            "history": [],
            "errors": []
        })

        print(f"\n结果:")
        print(f"  当前值: {result['current']}")
        print(f"  历史: {result['history']}")
        print(f"  错误: {result['errors']}")


# 总结：核心概念
"""
【可视化与调试的核心概念】

1. 图可视化：
   - print_ascii()：文本表示
   - Mermaid：图形化表示
   - 导出图片：保存可视化
   - 节点和边的清晰展示

2. 执行追踪：
   - 记录每个节点的执行
   - 时间戳记录
   - 输入输出追踪
   - 完整的执行历史

3. 性能分析：
   - 测量节点耗时
   - 识别性能瓶颈
   - 优化慢节点
   - 监控资源使用

4. 调试工具：
   - verbose 模式
   - 状态快照
   - 断点调试
   - 日志记录
   - 状态验证

5. 监控系统：
   - 指标收集
   - 日志系统
   - 告警机制
   - 实时监控

6. 最佳实践：
   - 从简单开始
   - 分步验证
   - 充分测试
   - 记录问题
   - 持续改进

【下一步学习】

在 20-error-handling.py 中，你将学习：
- 错误处理策略
- 重试机制
- 降级处理
- 容错设计
"""

if __name__ == "__main__":
    example_1_visualize_graph()
    example_2_execution_tracing()
    example_3_performance_analysis()
    example_4_debugging_tools()
    example_5_monitoring()
    example_6_debugging_practice()
