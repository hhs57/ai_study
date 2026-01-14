"""
LangGraph 学习 07：状态持久化

知识点：
1. 数据库持久化
2. 跨会话状态管理
3. 状态版本控制
4. 分布式状态管理
5. 实际应用场景
"""

import sys
import io
import json
from datetime import datetime
from typing import TypedDict, Annotated, Sequence
import operator

# 设置标准输出为 UTF-8 编码
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# ============ 示例 1：基础检查点 ============

def example_1_basic_checkpoint():
    """示例 1：使用 MemorySaver 保存状态"""
    print("=" * 70)
    print("示例 1：基础检查点 - MemorySaver")
    print("=" * 70)

    class ChatState(TypedDict):
        messages: Sequence[str]
        step_count: int

    def chat_node(state: ChatState) -> ChatState:
        """聊天节点"""
        step = state["step_count"] + 1
        message = f"第 {step} 步：你好！"

        print(f"  [聊天节点] 生成消息: {message}")

        return {
            "messages": state["messages"] + [message],
            "step_count": step
        }

    # 创建图
    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.set_entry_point("chat")
    graph.add_edge("chat", END)

    # 创建检查点保存器
    memory = MemorySaver()
    compiled_graph = graph.compile(checkpointer=memory)

    # 配置：thread_id 用于标识会话
    config = {"configurable": {"thread_id": "user-session-123"}}

    print("\n执行第 1 次:")
    result1 = compiled_graph.invoke(
        {"messages": [], "step_count": 0},
        config
    )
    print(f"  步骤数: {result1['step_count']}")
    print(f"  消息数: {len(result1['messages'])}")

    print("\n执行第 2 次（继续同一个会话）:")
    result2 = compiled_graph.invoke(
        result1,  # 使用之前的结果
        config
    )
    print(f"  步骤数: {result2['step_count']}")
    print(f"  消息数: {len(result2['messages'])}")

    print("\n执行第 3 次:")
    result3 = compiled_graph.invoke(result2, config)
    print(f"  步骤数: {result3['step_count']}")
    print(f"  消息数: {len(result3['messages'])}")

    print("\n✓ 检查点自动保存每次执行的状态")


# ============ 示例 2：多会话管理 ============

def example_2_multiple_sessions():
    """示例 2：管理多个独立会话"""
    print("=" * 70)
    print("示例 2：多会话管理")
    print("=" * 70)

    class UserSessionState(TypedDict):
        user_id: str
        messages: list[str]
        last_active: str

    def process_message(state: UserSessionState) -> UserSessionState:
        """处理消息"""
        message = f"[用户 {state['user_id']}] 的消息"
        timestamp = datetime.now().strftime("%H:%M:%S")

        print(f"  [{timestamp}] 处理用户 {state['user_id']} 的消息")

        return {
            "messages": state["messages"] + [message],
            "last_active": timestamp
        }

    # 创建图
    graph = StateGraph(UserSessionState)
    graph.add_node("process", process_message)
    graph.set_entry_point("process")
    graph.add_edge("process", END)

    memory = MemorySaver()
    compiled_graph = graph.compile(checkpointer=memory)

    # 模拟多个用户会话
    users = [
        {"user_id": "alice", "initial_state": {"user_id": "alice", "messages": [], "last_active": ""}},
        {"user_id": "bob", "initial_state": {"user_id": "bob", "messages": [], "last_active": ""}},
        {"user_id": "charlie", "initial_state": {"user_id": "charlie", "messages": [], "last_active": ""}}
    ]

    print("\n模拟多个用户会话:")

    # 第一轮：每个用户发送一条消息
    for user in users:
        config = {"configurable": {"thread_id": user["user_id"]}}
        result = compiled_graph.invoke(user["initial_state"], config)
        print(f"  → {result['user_id']}: {len(result['messages'])} 条消息")

    # 第二轮：某些用户继续发送消息
    print("\n第二轮会话:")
    for user_id in ["alice", "bob"]:
        config = {"configurable": {"thread_id": user_id}}
        # 获取之前的状态并继续
        result = compiled_graph.invoke(None, config)
        print(f"  → {result['user_id']}: {len(result['messages'])} 条消息")

    print("\n✓ 每个用户的会话独立管理")


# ============ 示例 3：自定义检查点后端 ============

def example_3_custom_checkpoint():
    """示例 3：自定义检查点实现"""
    print("=" * 70)
    print("示例 3：自定义检查点后端")
    print("=" * 70)

    print("""
LangGraph 支持多种检查点后端：

1. MemorySaver（内存）：
   - 最简单
   - 仅用于开发
   - 重启后丢失

2. SQLiteSaver（本地文件）：
   - 持久化到文件
   - 适合单机应用
   ```python
   from langgraph.checkpoint.sqlite import SqliteSaver

   # 连接到 SQLite 数据库
   saver = SqliteSaver.from_conn_string("checkpoints.db")
   compiled = graph.compile(checkpointer=saver)
   ```

3. PostgresSaver（PostgreSQL）：
   - 生产环境
   - 支持分布式
   ```python
   from langgraph.checkpoint.postgres import PostgresSaver

   # 连接到 PostgreSQL
   saver = PostgresSaver.from_conn_string(
       "postgresql://user:password@localhost/db"
   )
   compiled = graph.compile(checkpointer=saver)
   ```

4. RedisSaver（Redis）：
   - 高性能
   - 适合缓存
   ```python
   from langgraph.checkpoint.redis import RedisSaver

   # 连接到 Redis
   saver = RedisSaver.from_conn_info(
       host="localhost",
       port=6379,
       db=0
   )
   compiled = graph.compile(checkpointer=saver)
   ```

选择建议：
┌──────────────┬────────────┬────────────┬──────────┐
│   后端       │ 性能      │ 持久化    │ 适用场景 │
├──────────────┼────────────┼────────────┼──────────┤
│ MemorySaver  │ ⭐⭐⭐⭐⭐ │ ❌         │ 开发测试 │
│ SQLiteSaver  │ ⭐⭐⭐     │ ✅         │ 单机应用 │
│ PostgresSaver│ ⭐⭐⭐⭐   │ ✅         │ 生产环境 │
│ RedisSaver   │ ⭐⭐⭐⭐⭐ │ ✅         │ 高性能   │
└──────────────┴────────────┴────────────┴──────────┘
    """)


# ============ 示例 4：状态版本控制 ============

def example_4_state_versioning():
    """示例 4：状态版本控制"""
    print("=" * 70)
    print("示例 4：状态版本控制")
    print("=" * 70)

    class VersionedState(TypedDict):
        version: int
        data: str
        history: list[dict]

    def update_state(state: VersionedState) -> VersionedState:
        """更新状态"""
        new_version = state["version"] + 1
        new_data = f"数据版本 {new_version}"

        # 记录历史
        history_entry = {
            "version": new_version,
            "timestamp": datetime.now().isoformat(),
            "data": new_data
        }

        print(f"  [更新] 创建版本 {new_version}")

        return {
            "version": new_version,
            "data": new_data,
            "history": state["history"] + [history_entry]
        }

    # 创建图
    graph = StateGraph(VersionedState)
    graph.add_node("update", update_state)
    graph.set_entry_point("update")
    graph.add_edge("update", END)

    memory = MemorySaver()
    compiled_graph = graph.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "versioned-session"}}

    print("\n创建多个版本:")
    print("-" * 70)

    # 创建多个版本
    for i in range(5):
        if i == 0:
            result = compiled_graph.invoke({
                "version": 0,
                "data": "",
                "history": []
            }, config)
        else:
            result = compiled_graph.invoke(result, config)

        print(f"  版本 {result['version']}: {result['data']}")

    # 查看历史
    print(f"\n版本历史:")
    print("-" * 70)
    for entry in result['history']:
        print(f"  版本 {entry['version']}: {entry['timestamp']}")

    print("\n✓ 完整的版本历史记录")


# ============ 示例 5：时间旅行调试 ============

def example_5_time_travel_debugging():
    """示例 5：时间旅行调试"""
    print("=" * 70)
    print("示例 5：时间旅行调试")
    print("=" * 70)

    print("""
检查点支持"时间旅行"调试：

1. 查看历史状态：
```python
# 获取特定会话的所有检查点
config = {"configurable": {"thread_id": "session-123"}}

# 查看检查点历史
checkpoints = compiled_graph.get_state_history(config)

for checkpoint in checkpoints:
    print(f"版本: {checkpoint.config['configurable']['checkpoint_id']}")
    print(f"状态: {checkpoint.values}")
    print(f"时间: {checkpoint.timestamp}")
```

2. 回溯到之前的版本：
```python
# 回到之前的检查点
old_state = compiled_graph.get_state(config, checkpoint_id="old_checkpoint")

# 从旧状态重新开始
new_config = {"configurable": {"thread_id": "new-session"}}
result = compiled_graph.invoke(old_state.values, new_config)
```

3. 比较不同版本：
```python
# 获取多个检查点
checkpoint_1 = compiled_graph.get_state(config, checkpoint_id="v1")
checkpoint_2 = compiled_graph.get_state(config, checkpoint_id="v2")

# 比较状态差异
diff = compare_states(checkpoint_1.values, checkpoint_2.values)
```

4. 分支探索：
```python
# 从某个检查点创建分支
branch_config = {"configurable": {"thread_id": "branch-1"}}

# 从检查点 A 开始，尝试不同的路径
result_branch_1 = compiled_graph.invoke(checkpoint_a.values, branch_config)
```

应用场景：
✓ 调试复杂流程
✓ 测试不同决策
✓ 回滚错误操作
✓ 分析系统行为
✓ A/B 测试

实际工作流：
┌──────────┐    ┌──────────┐    ┌──────────┐
│ 执行流程 │ -> │ 保存检查点│ -> │ 发现问题  │
└──────────┘    └──────────┘    └──────────┘
                                    |
                                    v
                            ┌──────────┐
                            │ 回溯调试  │
                            └──────────┘
                                    |
                                    v
                            ┌──────────┐    ┌──────────┐
                            │ 修复问题  │ -> │ 重新执行  │
                            └──────────┘    └──────────┘
    """)


# ============ 示例 6：分布式状态管理 ============

def example_6_distributed_state():
    """示例 6：分布式状态管理"""
    print("=" * 70)
    print("示例 6：分布式状态管理")
    print("=" * 70)

    print("""
分布式系统的状态管理挑战：

1. 数据一致性：
   - 多个节点访问同一状态
   - 需要事务支持
   - 避免竞争条件

2. 高可用性：
   - 状态存储的冗余
   - 故障转移
   - 数据备份

3. 性能优化：
   - 减少网络延迟
   - 缓存策略
   - 批量操作

架构模式：

┌────────────────────────────────────────────────┐
│           分布式状态管理架构                    │
└────────────────────────────────────────────────┘

应用层：
┌─────────┐  ┌─────────┐  ┌─────────┐
│ 实例 A  │  │ 实例 B  │  │ 实例 C  │
└────┬────┘  └────┬────┘  └────┬────┘
     │            │            │
     └────────────┼────────────┘
                  │
         ┌────────┴────────┐
         │  负载均衡器      │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │  状态存储层      │
         ├─────────────────┤
         │ • PostgreSQL    │
         │ • Redis         │
         │ • 分布式缓存    │
         └─────────────────┘

实现建议：

1. 使用 PostgreSQL 作为主要存储：
```python
from langgraph.checkpoint.postgres import PostgresSaver

# 使用连接池
saver = PostgresSaver.from_conn_string(
    "postgresql://user:pass@host/db",
    pool_size=20,
    max_overflow=10
)
```

2. 使用 Redis 作为缓存层：
```python
from langgraph.checkpoint.redis import RedisSaver

# Redis 缓存
redis_saver = RedisSaver.from_conn_info(
    host="redis-cluster",
    port=6379,
    password="secure-password"
)
```

3. 实现降级策略：
```python
class HybridCheckpointSaver:
    def __init__(self, primary, fallback):
        self.primary = primary  # 主存储
        self.fallback = fallback  # 降级存储

    def put(self, config, checkpoint):
        try:
            return self.primary.put(config, checkpoint)
        except Exception as e:
            print(f"主存储失败，使用降级: {e}")
            return self.fallback.put(config, checkpoint)
```

4. 监控和告警：
```python
# 监控检查点性能
metrics = {
    "checkpoint_save_time": [],
    "checkpoint_load_time": [],
    "failure_rate": []
}

# 设置告警
if metrics["failure_rate"] > 0.05:  # 5% 失败率
    alert_team("状态存储故障率过高")
```

最佳实践：

✓ 使用可靠的存储（PostgreSQL）
✓ 实现连接池
✓ 设置超时和重试
✓ 监控性能指标
✓ 定期备份
✓ 灾难恢复计划
✓ 文档记录架构
✓ 压力测试
    """)


# ============ 示例 7：生产环境最佳实践 ============

def example_7_production_best_practices():
    """示例 7：生产环境最佳实践"""
    print("=" * 70)
    print("示例 7：生产环境最佳实践")
    print("=" * 70)

    print("""
生产环境的检查点配置：

1. 数据库设置（PostgreSQL）：
```sql
-- 创建专用数据库
CREATE DATABASE langgraph_checkpoints;

-- 优化配置
ALTER DATABASE langgraph_checkpoints SET
    shared_buffers = '256MB',
    effective_cache_size = '1GB',
    maintenance_work_mem = '64MB';

-- 创建索引
CREATE INDEX idx_thread_id ON checkpoints(thread_id);
CREATE INDEX idx_timestamp ON checkpoints(created_at);
```

2. 连接池配置：
```python
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg2.pool

# 使用连接池
saver = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost/db",
    pool_class=psycopg2.pool.ThreadedConnectionPool,
    minconn=5,
    maxconn=20
)
```

3. 监控指标：
```python
class CheckpointMetrics:
    def __init__(self):
        self.save_count = 0
        self.load_count = 0
        self.errors = 0
        self.latencies = []

    def record_save(self, duration):
        self.save_count += 1
        self.latencies.append(("save", duration))

    def record_load(self, duration):
        self.load_count += 1
        self.latencies.append(("load", duration))

    def get_stats(self):
        avg_latency = sum(l[1] for l in self.latencies) / len(self.latencies)
        return {
            "total_saves": self.save_count,
            "total_loads": self.load_count,
            "errors": self.errors,
            "avg_latency": avg_latency
        }
```

4. 备份策略：
```bash
#!/bin/bash
# 每日备份脚本
DATE=$(date +%Y%m%d)
pg_dump -U user langgraph_checkpoints > "backup_$DATE.sql"

# 保留最近 7 天的备份
find /path/to/backups -name "backup_*.sql" -mtime +7 -delete
```

5. 清理策略：
```python
# 定期清理旧检查点
from datetime import datetime, timedelta

def cleanup_old_checkpoints(saver, days=30):
    """清理超过指定天数的检查点"""
    cutoff = datetime.now() - timedelta(days=days)

    # 获取所有会话
    sessions = list_all_sessions(saver)

    for session_id in sessions:
        # 获取检查点历史
        history = saver.get_state_history(session_id)

        # 删除旧检查点
        for checkpoint in history:
            if checkpoint.created_at < cutoff:
                saver.delete_checkpoint(session_id, checkpoint.checkpoint_id)
```

6. 错误处理：
```python
class RobustCheckpointSaver:
    def __init__(self, primary_saver, max_retries=3):
        self.primary_saver = primary_saver
        self.max_retries = max_retries

    def put(self, config, checkpoint):
        for attempt in range(self.max_retries):
            try:
                return self.primary_saver.put(config, checkpoint)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                    continue
                else:
                    # 最后一次尝试失败
                    log_error(e)
                    raise
```

7. 性能优化：
```python
# 批量写入
class BatchCheckpointSaver:
    def __init__(self, batch_size=100, flush_interval=5):
        self.batch = []
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.last_flush = time.time()

    def put(self, config, checkpoint):
        self.batch.append((config, checkpoint))

        # 批量写入条件
        if (len(self.batch) >= self.batch_size or
            time.time() - self.last_flush > self.flush_interval):
            self.flush()

    def flush(self):
        if self.batch:
            # 批量写入数据库
            self.saver.put_many(self.batch)
            self.batch = []
            self.last_flush = time.time()
```

检查清单：

开发环境：
✓ 使用 MemorySaver
✓ 快速迭代
✓ 频繁测试

测试环境：
✓ 使用 SQLite
✓ 模拟生产
✓ 性能测试

生产环境：
✓ 使用 PostgreSQL
✓ 连接池
✓ 监控告警
✓ 定期备份
✓ 清理策略
✓ 错误处理
✓ 文档完善
    """)


# 总结：核心概念
"""
【状态持久化的核心概念】

1. 检查点（Checkpoint）：
   - 保存执行状态
   - 支持暂停和恢复
   - 记录完整历史
   - 版本控制

2. 检查点后端：
   - MemorySaver：内存存储
   - SQLiteSaver：文件存储
   - PostgresSaver：生产数据库
   - RedisSaver：高性能缓存

3. 会话管理：
   - thread_id 标识会话
   - 独立的会话状态
   - 跨请求保持状态
   - 多用户支持

4. 版本控制：
   - 记录每次变更
   - 支持回溯
   - 时间旅行调试
   - 分支探索

5. 分布式管理：
   - 数据一致性
   - 高可用性
   - 性能优化
   - 故障转移

6. 最佳实践：
   - 选择合适的后端
   - 实现连接池
   - 设置监控
   - 定期备份
   - 清理旧数据
   - 错误处理

【下一步学习】

在 18-multi-agent.py 中，你将学习：
- 多 Agent 系统架构
- Agent 之间的通信
- 协作式 Agent
- 竞争式 Agent
"""

if __name__ == "__main__":
    example_1_basic_checkpoint()
    example_2_multiple_sessions()
    example_3_custom_checkpoint()
    example_4_state_versioning()
    example_5_time_travel_debugging()
    example_6_distributed_state()
    example_7_production_best_practices()
