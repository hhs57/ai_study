"""
模块 35: Self-Attention 详解
========================

本模块通过 Python 代码演示 Self-Attention 的计算过程。

注意: 这是配套的代码演示,核心学习内容请查看 35-self-attention.html

知识点:
1. Query、Key、Value 的概念
2. Attention Score 的计算
3. Scaled Dot-Product Attention
4. 可视化 Attention Map

运行示例:
python 35-self-attention.py
"""

import numpy as np
import sys
import io

# 设置标准输出为 UTF-8 编码
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def softmax(x):
    """Softmax 激活函数"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention

    Args:
        Q: Query 矩阵, shape (..., seq_len_q, d_k)
        K: Key 矩阵, shape (..., seq_len_k, d_k)
        V: Value 矩阵, shape (..., seq_len_k, d_v)
        mask: 可选的掩码矩阵

    Returns:
        output: 注意力输出, shape (..., seq_len_q, d_v)
        attention_weights: 注意力权重, shape (..., seq_len_q, seq_len_k)
    """
    # 1. 计算 Q 和 K 的点积
    scores = np.matmul(Q, K.swapaxes(-2, -1))  # (..., seq_len_q, seq_len_k)

    # 2. 缩放 (除以 √d_k)
    d_k = Q.shape[-1]
    scaled_scores = scores / np.sqrt(d_k)

    # 3. 如果有 mask,应用 mask
    if mask is not None:
        scaled_scores = scaled_scores + (mask * -1e9)

    # 4. Softmax 归一化
    attention_weights = softmax(scaled_scores)

    # 5. 与 V 相乘
    output = np.matmul(attention_weights, V)

    return output, attention_weights


def example_1_qkv_concept():
    """
    示例 1: Query、Key、Value 的概念

    类比: 数据库查询
    - Query (查询): 我要找什么?
    - Key (键): 索引,用于匹配查询
    - Value (值): 实际的内容
    """
    print("=" * 60)
    print("示例 1: Query、Key、Value 的概念")
    print("=" * 60)

    # 简单示例: 3个单词,每个词用4维向量表示
    words = ["我", "爱", "编程"]

    # 模拟词嵌入 (实际中这些是通过训练得到的)
    X = np.array([
        [0.1, 0.2, 0.3, 0.4],  # 我
        [0.5, 0.6, 0.7, 0.8],  # 爱
        [0.9, 1.0, 1.1, 1.2],  # 编程
    ])

    print("\n输入词嵌入 X:")
    for word, vec in zip(words, X):
        print(f"  {word}: {vec}")

    # 通过线性变换得到 Q、K、V
    # 实际中这些权重矩阵是学习得到的
    W_q = np.random.randn(4, 4) * 0.1
    W_k = np.random.randn(4, 4) * 0.1
    W_v = np.random.randn(4, 4) * 0.1

    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)

    print("\nQuery (Q) - 我要找什么?:")
    for word, vec in zip(words, Q):
        print(f"  {word}: {np.round(vec, 3)}")

    print("\nKey (K) - 索引:")
    for word, vec in zip(words, K):
        print(f"  {word}: {np.round(vec, 3)}")

    print("\nValue (V) - 内容:")
    for word, vec in zip(words, V):
        print(f"  {word}: {np.round(vec, 3)}")

    print("\n💡 类比:")
    print("  就像在数据库中查询:")
    print("  Query: '名字=张三 AND 年龄>25'")
    print("  Key: {id, name, age, ...}")
    print("  Value: {张三, 26, 工程师, ...}")


def example_2_attention_calculation():
    """
    示例 2: 逐步计算 Attention
    """
    print("\n" + "=" * 60)
    print("示例 2: 逐步计算 Attention")
    print("=" * 60)

    # 简化示例: 2个词,每个词用3维向量
    Q = np.array([
        [1.0, 0.5, 0.2],  # 词1的Query
        [0.3, 0.8, 0.6],  # 词2的Query
    ])

    K = np.array([
        [0.9, 0.4, 0.3],  # 词1的Key
        [0.5, 0.7, 0.1],  # 词2的Key
    ])

    V = np.array([
        [1.0, 2.0],        # 词1的Value
        [3.0, 4.0],        # 词2的Value
    ])

    print("\n步骤1: 计算 Q·K^T (原始分数)")
    scores = np.dot(Q, K.T)
    print(scores)
    print("  解释: 每个词与其他词的相关性分数")

    print("\n步骤2: 缩放 (除以 √d_k)")
    d_k = Q.shape[-1]
    scaled_scores = scores / np.sqrt(d_k)
    print(f"  d_k = {d_k}, √d_k = {np.sqrt(d_k):.3f}")
    print(scaled_scores)
    print("  解释: 缩放防止梯度消失")

    print("\n步骤3: Softmax 归一化")
    attention_weights = softmax(scaled_scores)
    print(attention_weights)
    print("  解释: 转换成概率分布,每行和为1")

    print("\n步骤4: 与 V 相乘得到输出")
    output = np.dot(attention_weights, V)
    print(output)
    print("  解释: 加权求和,权重是 attention_weights")

    print("\n完整流程总结:")
    print("  Output = Softmax(Q·K^T / √d_k) · V")


def example_3_attention_map():
    """
    示例 3: 可视化 Attention Map
    """
    print("\n" + "=" * 60)
    print("示例 3: Attention Map 可视化")
    print("=" * 60)

    # 示例句子: "The cat sat on the mat"
    sentence = ["The", "cat", "sat", "on", "the", "mat"]

    # 模拟 Q、K、V (每个词用8维向量)
    d_model = 8
    np.random.seed(42)
    X = np.random.randn(len(sentence), d_model) * 0.1

    W_q = np.random.randn(d_model, d_model) * 0.1
    W_k = np.random.randn(d_model, d_model) * 0.1
    W_v = np.random.randn(d_model, d_model) * 0.1

    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)

    # 计算 attention
    output, attention_weights = scaled_dot_product_attention(Q, K, V)

    print("\n句子:", " ".join(sentence))
    print("\nAttention Map (热力图数值):")
    print("     ", end="")
    for word in sentence:
        print(f"{word:>6}", end="")
    print()

    for i, word in enumerate(sentence):
        print(f"{word:>4}", end=" ")
        for j in range(len(sentence)):
            print(f"{attention_weights[i, j]:>6.3f}", end="")
        print()

    print("\n💡 观察要点:")
    print("  - 对角线数值较大: 每个词与自己最相关")
    print("  - 某些词对之间有较强的关联(例如 'cat' 和 'sat')")
    print("  - 这就是 Attention 机制的'注意力分配'")


def example_4_multi_head_intuition():
    """
    示例 4: Multi-Head 的直观理解
    """
    print("\n" + "=" * 60)
    print("示例 4: 为什么要 Multi-Head?")
    print("=" * 60)

    print("\n直觉解释:")
    print("  单头 Attention 就像只用一种方式看问题")
    print("  多头 Attention 就像从多个角度理解语义")

    print("\n类比: 理解一句话 '苹果公司发布了新产品'")
    print("  Head 1 可能关注: '苹果' <-> '公司' (实体关系)")
    print("  Head 2 可能关注: '发布' <-> '产品' (动作关系)")
    print("  Head 3 可能关注: '新' <-> '产品' (属性关系)")

    print("\n单头的局限:")
    print("  - 只能捕捉一种类型的关联")
    print("  - 表达能力受限")

    print("\n多头的优势:")
    print("  - 不同的头关注不同的语义子空间")
    print("  - 并行计算,提高模型容量")
    print("  - 综合多角度的信息")

    print("\n数学原理:")
    print("  MultiHead(Q,K,V) = Concat(head_1, ..., head_h) · W^O")
    print("  其中 head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)")


# 总结
"""
【核心概念总结】

1. Self-Attention 的本质:
   - 让序列中的每个词都能"看到"其他所有词
   - 通过注意力权重动态地决定关注哪些信息

2. Q、K、V 的含义:
   - Query (查询): 当前位置想找什么信息
   - Key (键): 其他位置提供什么样的索引
   - Value (值): 其他位置的实际内容

3. 计算步骤:
   a) 计算 Q·K^T: 得到原始相关性分数
   b) 缩放: 除以 √d_k,防止梯度消失
   c) Softmax: 转换成概率分布
   d) 与 V 相乘: 加权求和得到输出

4. 为什么要缩放?
   - 当维度很大时,点积结果会很大
   - Softmax 会进入饱和区,梯度很小
   - 缩放可以缓解这个问题

5. Attention Map 的意义:
   - 展示模型如何分配注意力
   - 每行表示一个词对所有词的关注权重
   - 可视化有助于理解模型的决策过程

【下一步学习】

在 36-multi-head-attention.html 中,你将学习:
- 为什么需要多头注意力
- 多头是如何并行计算的
- 位置编码的作用
- BERT 和 GPT 如何使用 Multi-Head Attention
"""


if __name__ == "__main__":
    # 运行所有示例
    example_1_qkv_concept()
    example_2_attention_calculation()
    example_3_attention_map()
    example_4_multi_head_intuition()

    print("\n" + "=" * 60)
    print("✅ 所有示例运行完成!")
    print("=" * 60)
    print("\n📌 提示: 请打开 35-self-attention.html 查看交互式可视化")
