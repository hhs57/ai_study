"""
LangChain 学习 14：回调机制与流式输出

知识点：
1. 回调系统（Callbacks）的概念
2. 使用标准回调处理器
3. 自定义回调处理器
4. 流式输出（Streaming）
5. Token 级别的实时输出
"""

import sys
import io
from typing import Any, Dict, List

# 设置标准输出为 UTF-8 编码
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from langchain.callbacks import StdOutCallbackHandler
from config import get_llm


# ============ 自定义回调处理器 ============

class TokenCounterHandler(BaseCallbackHandler):
    """Token 计数回调处理器"""

    def __init__(self):
        self.token_count = 0
        self.tokens = []

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """当 LLM 开始时"""
        print(f"\n[TokenCounter] LLM 开始处理...")
        print(f"[TokenCounter] 提示词: {prompts[0][:50]}...")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """当生成新 token 时"""
        self.token_count += 1
        self.tokens.append(token)
        # 实时打印 token（可选）
        print(token, end='', flush=True)

    def on_llm_end(self, response: Any, **kwargs) -> None:
        """当 LLM 结束时"""
        print(f"\n\n[TokenCounter] 总 token 数: {self.token_count}")
        print(f"[TokenCounter] 完整文本: {''.join(self.tokens)}")


class TimingHandler(BaseCallbackHandler):
    """计时回调处理器"""

    def __init__(self):
        import time
        self.start_time = None
        self.end_time = None

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """开始计时"""
        import time
        self.start_time = time.time()
        print(f"\n[Timer] 开始时间: {time.strftime('%H:%M:%S')}")

    def on_llm_end(self, response: Any, **kwargs) -> None:
        """结束计时"""
        import time
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"\n[Timer] 结束时间: {time.strftime('%H:%M:%S')}")
        print(f"[Timer] 总耗时: {elapsed:.2f} 秒")


class DetailedLogger(BaseCallbackHandler):
    """详细日志记录器"""

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """记录开始"""
        print("\n" + "=" * 70)
        print("[Logger] === LLM 调用开始 ===")
        print(f"[Logger] 模型: {serialized.get('name', 'unknown')}")
        print(f"[Logger] 提示词数量: {len(prompts)}")
        for i, prompt in enumerate(prompts, 1):
            print(f"[Logger] 提示词 {i}: {prompt[:100]}...")

    def on_llm_end(self, response: Any, **kwargs) -> None:
        """记录结束"""
        print("\n[Logger] === LLM 调用结束 ===")
        if hasattr(response, 'llm_output'):
            print(f"[Logger] Token 用量: {response.llm_output.get('token_usage', 'N/A')}")
        print("=" * 70)


# ============ 示例 1：基础回调使用 ============

def example_1_basic_callbacks():
    """示例 1：使用标准回调处理器"""
    print("=" * 70)
    print("示例 1：StdOutCallbackHandler - 标准输出回调")
    print("=" * 70)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # 使用标准输出回调
    from langchain.callbacks import StdOutCallbackHandler

    handler = StdOutCallbackHandler()

    response = llm.invoke(
        "用一句话解释什么是 Python",
        config={"callbacks": [handler]}
    )

    print(f"\n最终回答: {response.content}")


# ============ 示例 2：Token 计数回调 ============

def example_2_token_counter():
    """示例 2：使用自定义 Token 计数器"""
    print("=" * 70)
    print("示例 2：Token 计数回调")
    print("=" * 70)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    token_counter = TokenCounterHandler()

    print("\n生成文本（实时显示）:")
    print("-" * 70)

    response = llm.invoke(
        "写一个简短的故事：一只会说话的猫",
        config={"callbacks": [token_counter]}
    )

    print(f"\n\n最终结果:")
    print(f"  {response.content}")


# ============ 示例 3：多个回调组合 ============

def example_3_multiple_callbacks():
    """示例 3：组合多个回调处理器"""
    print("=" * 70)
    print("示例 3：组合多个回调")
    print("=" * 70)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # 组合多个回调
    token_counter = TokenCounterHandler()
    timer = TimingHandler()
    logger = DetailedLogger()

    print("\n执行 LLM 调用（带多个回调）:")

    response = llm.invoke(
        "解释什么是机器学习",
        config={"callbacks": [token_counter, timer, logger]}
    )


# ============ 示例 4：链中的回调 ============

def example_4_chain_callbacks():
    """示例 4：在链中使用回调"""
    print("=" * 70)
    print("示例 4：链中的回调")
    print("=" * 70)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    prompt = ChatPromptTemplate.from_template(
        "写一首关于{topic}的短诗。"
    )

    chain = prompt | llm | StrOutputParser()

    # 创建回调
    token_counter = TokenCounterHandler()

    print("\n执行链（带回调）:")
    print("-" * 70)

    result = chain.invoke(
        {"topic": "春天"},
        config={"callbacks": [token_counter]}
    )

    print(f"\n\n最终结果:")
    print(f"  {result}")


# ============ 示例 5：流式输出 ============

def example_5_streaming():
    """示例 5：流式输出"""
    print("=" * 70)
    print("示例 5：流式输出")
    print("=" * 70)

    # 方法 1：使用 invoke 的 stream_mode
    print("\n方法 1：使用 stream() 方法")
    print("-" * 70)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    print("\n流式输出:")

    # 逐 token 流式输出
    for chunk in llm.stream("讲一个关于AI的笑话"):
        print(chunk.content, end="", flush=True)

    print("\n")

    # 方法 2：在链中流式输出
    print("\n" + "=" * 70)
    print("方法 2：链中的流式输出")
    print("-" * 70)

    prompt = ChatPromptTemplate.from_template(
        "用简单的语言解释：{concept}"
    )

    chain = prompt | llm | StrOutputParser()

    print("\n流式生成答案:")

    for chunk in chain.stream({"concept": "量子计算"}):
        print(chunk, end="", flush=True)

    print("\n")


# ============ 示例 6：异步流式输出 ============

def example_6_async_streaming():
    """示例 6：异步流式输出（概念）"""
    print("=" * 70)
    print("示例 6：异步流式输出")
    print("=" * 70)

    print("""
异步流式输出可以提高性能，特别是在处理多个请求时。

基本示例：
```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

async def stream_example():
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # 异步流式输出
    async for chunk in llm.astream("讲一个故事"):
        print(chunk.content, end="", flush=True)

    print()

    # 在链中异步流式输出
    prompt = ChatPromptTemplate.from_template("解释：{topic}")
    chain = prompt | llm

    async for chunk in chain.astream({"topic": "AI"}):
        if hasattr(chunk, 'content'):
            print(chunk.content, end="", flush=True)

# 运行异步函数
asyncio.run(stream_example())
```

使用异步的好处：
✓ 并发处理多个请求
✓ 不阻塞主线程
✓ 更好的资源利用率
✓ 适合 Web 应用

注意事项：
- 需要 async/await 语法
- 所有组件都支持异步
- 在异步上下文中运行
    """)


# ============ 示例 7：高级回调应用 ============

def example_7_advanced_callbacks():
    """示例 7：高级回调应用场景"""
    print("=" * 70)
    print("示例 7：高级回调应用")
    print("=" * 70)

    print("""
1. 成本追踪回调：

```python
class CostTrackerHandler(BaseCallbackHandler):
    \"\"\"追踪 API 调用成本\"\"\"

    PRICING = {
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},  # 每 1K tokens
        "gpt-4": {"input": 0.03, "output": 0.06}
    }

    def __init__(self):
        self.total_cost = 0.0

    def on_llm_end(self, response, **kwargs):
        model = kwargs.get('invocation_params', {}).get('model', 'gpt-3.5-turbo')
        usage = response.llm_output.get('token_usage', {})

        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)

        pricing = self.PRICING.get(model, self.PRICING["gpt-3.5-turbo"])
        cost = (input_tokens / 1000 * pricing['input'] +
                output_tokens / 1000 * pricing['output'])

        self.total_cost += cost
        print(f"\\n[Cost] 本次调用: ${cost:.6f}")
        print(f"[Cost] 累计成本: ${self.total_cost:.6f}")
```

2. 缓存回调：

```python
class CacheHandler(BaseCallbackHandler):
    \"\"\"缓存 LLM 响应\"\"\"

    def __init__(self):
        self.cache = {}

    def on_llm_start(self, serialized, prompts, **kwargs):
        # 检查缓存
        prompt_key = prompts[0]
        if prompt_key in self.cache:
            print("[Cache] 命中缓存！")
            return self.cache[prompt_key]

    def on_llm_end(self, response, **kwargs):
        # 保存到缓存
        # 实际实现需要获取原始提示词
        pass
```

3. 安全检查回调：

```python
class SecurityCheckHandler(BaseCallbackHandler):
    \"\"\"检查输出内容的安全性\"\"\"

    def __init__(self, forbidden_words=None):
        self.forbidden_words = forbidden_words or [
            "密码", "secret", "token"
        ]

    def on_llm_new_token(self, token, **kwargs):
        # 检查每个 token
        for word in self.forbidden_words:
            if word in token.lower():
                print(f"\\n[Security] 警告: 检测到敏感词: {word}")
                # 可以选择中断或替换
```

4. 进度条回调：

```python
from tqdm import tqdm

class ProgressBarHandler(BaseCallbackHandler):
    \"\"\"显示进度条\"\"\"

    def __init__(self, total_tokens=100):
        self.pbar = tqdm(total=total_tokens, desc="生成中")

    def on_llm_new_token(self, token, **kwargs):
        self.pbar.update(1)

    def on_llm_end(self, response, **kwargs):
        self.pbar.close()
```

5. 日志记录回调：

```python
import logging

class LoggingHandler(BaseCallbackHandler):
    \"\"\"记录到文件\"\"\"

    def __init__(self, filename="langchain.log"):
        logging.basicConfig(
            filename=filename,
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )

    def on_llm_start(self, serialized, prompts, **kwargs):
        logging.info(f"LLM 调用开始: {prompts[0][:50]}")

    def on_llm_end(self, response, **kwargs):
        logging.info(f"LLM 调用结束")
```
    """)


# ============ 示例 8：回调最佳实践 ============

def example_8_best_practices():
    """示例 8：回调最佳实践"""
    print("=" * 70)
    print("示例 8：最佳实践")
    print("=" * 70)

    print("""
回调系统最佳实践：

1. 回调处理器设计：
   ✓ 单一职责：每个处理器只做一件事
   ✓ 可组合：可以组合多个处理器
   ✓ 低开销：避免影响性能
   ✓ 线程安全：考虑并发访问

2. 何时使用回调：
   ✓ 调试和日志记录
   ✓ 性能监控
   ✓ 成本追踪
   ✓ 安全检查
   ✓ 用户界面更新

3. 何时使用流式输出：
   ✓ 需要实时反馈
   ✓ 长文本生成
   ✓ 聊天应用
   ✓ 需要提前显示部分结果

4. 性能考虑：
   ✓ 流式输出会增加 CPU 使用
   ✓ 回调会略微延迟响应
   ✓ 避免在回调中执行耗时操作
   ✓ 考虑异步处理

5. 调试技巧：
   ✓ 使用 StdOutCallbackHandler 查看详细输出
   ✓ 记录每个阶段的输入输出
   ✓ 测量各部分的耗时
   ✓ 检查 token 使用量

6. 常见使用模式：

   模式 1：调试模式
   ```python
   DEBUG = True
   callbacks = [StdOutCallbackHandler()] if DEBUG else []
   response = llm.invoke(prompt, config={"callbacks": callbacks})
   ```

   模式 2：生产环境监控
   ```python
   callbacks = [
       CostTrackerHandler(),
       TimingHandler(),
       LoggingHandler("production.log")
   ]
   ```

   模式 3：开发环境
   ```python
   callbacks = [
       StdOutCallbackHandler(),
       TokenCounterHandler(),
       DetailedLogger()
   ]
   ```

7. 错误处理：
   ✓ 在回调中捕获异常
   ✓ 避免回调失败影响主流程
   ✓ 记录错误但不中断执行

示例：安全的回调处理器
```python
class SafeCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        try:
            # 你的逻辑
            pass
        except Exception as e:
            # 记录错误但不抛出
            print(f"Callback error: {e}")
```

8. 回调 vs 其他方法：

   使用回调：
   - 需要实时处理
   - 需要细粒度控制
   - 需要插入到执行流程中

   不使用回调：
   - 简单的后处理
   - 批量处理
   - 不需要实时性
    """)


# 总结：核心概念
"""
【回调与流式输出的核心概念】

1. 回调系统（Callbacks）：
   - 在执行的不同阶段触发
   - 可以监控和干预执行流程
   - 支持多个回调组合
   - 不影响主流程逻辑

2. 回调事件：
   - on_llm_start：LLM 开始调用
   - on_llm_new_token：生成新 token
   - on_llm_end：LLM 调用结束
   - on_chain_start/end：链开始/结束
   - on_tool_start/end：工具开始/结束
   - on_chat_model_start：聊天模型开始

3. 流式输出：
   - stream()：逐 token 输出
   - astream()：异步流式输出
   - 实时反馈
   - 更好的用户体验

4. 自定义回调：
   - 继承 BaseCallbackHandler
   - 实现需要的事件方法
   - 可以访问和修改数据
   - 保持简单和高效

5. 应用场景：
   - 调试和开发
   - 性能监控
   - 成本追踪
   - 日志记录
   - 安全检查
   - 用户界面更新

6. 最佳实践：
   - 单一职责原则
   - 组合多个回调
   - 避免性能影响
   - 正确的错误处理
   - 根据环境选择回调

【下一步学习】

在 15-complete-rag-app.py 中，你将学习：
- 如何构建一个完整的 RAG 应用
- 综合运用前面学到的所有知识
- 实际项目的最佳实践
"""

if __name__ == "__main__":
    example_1_basic_callbacks()
    example_2_token_counter()
    example_3_multiple_callbacks()
    example_4_chain_callbacks()
    example_5_streaming()
    example_6_async_streaming()
    example_7_advanced_callbacks()
    example_8_best_practices()
