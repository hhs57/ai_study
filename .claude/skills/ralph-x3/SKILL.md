---
name: ralph-x3
description: "Quick Ralph Loop runner - automatically runs Ralph Loop 3 times with your task. Perfect for iterative improvement. Usage: /ralph-x3 'your task description'"
---

# Ralph-x3 - Quick 3-Iteration Ralph Loop

简化版 Ralph Loop：自动运行 3 次迭代来改进你的代码或任务。

## 使用方法

```
/ralph-x3 "你的任务描述"
```

## 示例

```
/ralph-x3 "优化这个函数的性能"
/ralph-x3 "添加错误处理和日志"
/ralph-x3 "重构这段代码使其更易读"
/ralph-x3 "修复这个 bug"
```

## 工作原理

这个 skill 会：
1. 使用你的任务描述启动 Ralph Loop
2. 自动设置 `--max-iterations 3`
3. 自动设置 `--completion-promise "DONE"`
4. 运行最多 3 次迭代
5. 每次迭代都会看到前一次的工作结果
6. 如果任务完成，输出 `<promise>DONE</promise>` 即可提前结束
7. 持续改进直到完成或达到 3 次

## 适用场景

✅ **适合使用：**
- 代码优化和重构
- 添加错误处理
- 改进代码质量
- 修复已知 bug
- 添加新功能

❌ **不适合使用：**
- 需要人类设计的创意任务
- 一次性操作
- 目标不明确的任务

## 完成条件

当任务完成时，输出：

```
<promise>DONE</promise>
```

循环会立即停止，无需等待 3 次迭代完成。

## 高级用法

如果你想自定义完成条件或迭代次数，可以手动使用完整命令：

```
/ralph-loop "你的任务" --max-iterations 5 --completion-promise "自定义"
```

## 停止循环

如果需要提前强制停止：

```
/cancel-ralph
```

## 等效命令

```
/ralph-x3 "你的任务"
```

等同于：

```
/ralph-loop "你的任务" --max-iterations 3 --completion-promise "DONE"
```
