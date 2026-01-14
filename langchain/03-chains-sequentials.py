"""
LangChain 学习 03：链的组合（Sequential Chains）

知识点：
1. SimpleSequentialChain：最简单的顺序链，每个步骤的输出是下一步的输入
2. SequentialChain：更复杂的顺序链，可以传递多个输入输出
3. 使用 LCEL（LangChain Expression Language）的管道操作符 |
4. RunnablePassthrough 和 RunnableParallel 用于数据传递
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from dotenv import load_dotenv

load_dotenv()


def example_1_simple_sequential_chain():
    """示例 1：最简单的顺序链"""
    print("=" * 60)
    print("示例 1：SimpleSequentialChain - 一步步处理")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # 第一步：生成故事大纲
    prompt1 = ChatPromptTemplate.from_template(
        "为一个关于{topic}的故事生成一个简短的大纲。"
    )
    chain1 = prompt1 | llm | StrOutputParser()

    # 第二步：根据大纲写故事
    prompt2 = ChatPromptTemplate.from_template(
        "根据以下大纲，写一个完整的故事：\n\n{outline}"
    )
    chain2 = prompt2 | llm | StrOutputParser()

    # 组合成顺序链（使用 LCEL 的管道操作符）
    # chain1 的输出会自动成为 chain2 的输入
    combined_chain = chain1 | chain2

    # 执行
    result = combined_chain.invoke({"topic": "一只会说话的猫"})
    print(f"最终故事:\n{result}\n")


def example_2_sequential_chain_multiple_inputs_outputs():
    """示例 2：多输入多输出的顺序链"""
    print("=" * 60)
    print("示例 2：SequentialChain - 多个输入输出")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # 链 1：生成产品名称
    prompt1 = ChatPromptTemplate.from_template(
        "为以下产品起5个有创意的名字：\n产品描述：{product_desc}\n目标用户：{target_audience}"
    )
    chain1 = LLMChain(
        llm=llm,
        prompt=prompt1,
        output_key="product_names"  # 指定输出的 key
    )

    # 链 2：生成产品口号
    prompt2 = ChatPromptTemplate.from_template(
        "为产品生成3个吸引人的口号：\n{product_names}"
    )
    chain2 = LLMChain(
        llm=llm,
        prompt=prompt2,
        output_key="slogans"
    )

    # 链 3：生成广告文案
    prompt3 = ChatPromptTemplate.from_template(
        "基于以下信息，撰写一段广告文案：\n产品：{product_names}\n口号：{slogans}"
    )
    chain3 = LLMChain(
        llm=llm,
        prompt=prompt3,
        output_key="ad_copy"
    )

    # 使用 SequentialChain 组合
    overall_chain = SequentialChain(
        chains=[chain1, chain2, chain3],
        input_variables=["product_desc", "target_audience"],  # 初始输入
        output_variables=["product_names", "slogans", "ad_copy"]  # 最终输出
    )

    # 执行
    result = overall_chain.invoke({
        "product_desc": "一款智能咖啡机，可以根据用户的心情自动调节口味",
        "target_audience": "上班族"
    })

    print(f"产品名称:\n{result['product_names']}\n")
    print(f"口号:\n{result['slogans']}\n")
    print(f"广告文案:\n{result['ad_copy']}\n")


def example_3_lcel_with_runnable_passthrough():
    """示例 3：使用 LCEL 和 RunnablePassthrough"""
    print("=" * 60)
    print("示例 3：LCEL + RunnablePassthrough - 数据传递")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # 场景：我们需要在多个步骤中使用同一个输入
    prompt1 = ChatPromptTemplate.from_template(
        "将以下文本翻译成中文：{text}"
    )
    chain1 = prompt1 | llm | StrOutputParser()

    prompt2 = ChatPromptTemplate.from_template(
        "总结以下翻译后的文本：{translation}"
    )
    chain2 = prompt2 | llm | StrOutputParser()

    # 使用 RunnablePassthrough 传递原始输入
    # 它会把输入原封不动地传递下去
    def prepare_input(data):
        return {
            "translation": chain1.invoke(data),  # chain1 的输出
            "text": data["text"]  # 原始输入
        }

    chain = {
        "translation": chain1,  # 先翻译
        "text": RunnablePassthrough()  # 保留原始文本
    } | chain2

    result = chain.invoke({"text": "Hello, how are you today?"})
    print(f"结果:\n{result}\n")


def example_4_parallel_execution():
    """示例 4：并行执行多个链"""
    print("=" * 60)
    print("示例 4：RunnableParallel - 并行执行")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # 链 1：生成笑话
    joke_prompt = ChatPromptTemplate.from_template(
        "讲一个关于{topic}的笑话。"
    )
    joke_chain = joke_prompt | llm | StrOutputParser()

    # 链 2：生成故事
    story_prompt = ChatPromptTemplate.from_template(
        "讲一个关于{topic}的短故事。"
    )
    story_chain = story_prompt | llm | StrOutputParser()

    # 链 3：生成诗歌
    poem_prompt = ChatPromptTemplate.from_template(
        "写一首关于{topic}的诗歌。"
    )
    poem_chain = poem_prompt | llm | StrOutputParser()

    # 使用 RunnableParallel 并行执行
    # 这三个链会同时执行，而不是按顺序
    parallel_chain = RunnableParallel(
        joke=joke_chain,
        story=story_chain,
        poem=poem_chain
    )

    result = parallel_chain.invoke({"topic": "春天"})

    print(f"笑话:\n{result['joke']}\n")
    print(f"故事:\n{result['story']}\n")
    print(f"诗歌:\n{result['poem']}\n")


def example_5_complex_chain():
    """示例 5：复杂的链组合"""
    print("=" * 60)
    print("示例 5：复杂链组合 - 实际应用场景")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # 场景：创建一个内容生成器
    # 步骤 1：并行生成标题和摘要
    title_prompt = ChatPromptTemplate.from_template(
        "为一篇关于{topic}的文章生成3个吸引人的标题。"
    )
    summary_prompt = ChatPromptTemplate.from_template(
        "为一篇关于{topic}的文章生成一段简短的摘要（50字以内）。"
    )

    # 步骤 2：根据标题和摘要生成大纲
    outline_prompt = ChatPromptTemplate.from_template(
        "根据以下标题和摘要，生成文章大纲：\n\n标题：{title}\n摘要：{summary}"
    )

    # 步骤 3：生成完整文章
    article_prompt = ChatPromptTemplate.from_template(
        "根据以下大纲，写一篇完整的文章：\n\n{outline}"
    )

    # 组合链
    title_chain = title_prompt | llm | StrOutputParser()
    summary_chain = summary_prompt | llm | StrOutputParser()

    # 并行生成标题和摘要
    parallel = RunnableParallel(
        title=title_chain,
        summary=summary_chain
    )

    # 顺序执行：根据标题和摘要生成大纲，再根据大纲生成文章
    outline_chain = outline_prompt | llm | StrOutputParser()
    article_chain = article_prompt | llm | StrOutputParser()

    # 完整的工作流
    full_chain = parallel | outline_chain | article_chain

    result = full_chain.invoke({"topic": "人工智能的未来"})
    print(f"最终文章:\n{result}\n")


# 总结：核心概念
"""
【顺序链的类型】
1. SimpleSequentialChain：
   - 最简单，每步的输出是下一步的输入
   - 只能有一个输入和一个输出
   - 适合线性、简单的流程

2. SequentialChain：
   - 可以有多个输入和多个输出
   - 每个链需要指定 input_key 和 output_key
   - 适合复杂的多步骤流程

3. LCEL (LangChain Expression Language)：
   - 使用 | 操作符串联链
   - 更灵活、更现代的语法
   - 推荐使用这种方式

【数据传递工具】
1. RunnablePassthrough：
   - 将输入原封不动地传递给下一步
   - 常用于需要保留原始输入的场景

2. RunnableParallel：
   - 并行执行多个链
   - 所有链使用相同的输入
   - 输出是一个字典，包含所有链的结果

【链的组合模式】
- 串行：chain1 | chain2 | chain3
- 并行：RunnableParallel(chain1=..., chain2=...)
- 混合：先并行，再串行
"""

if __name__ == "__main__":
    example_1_simple_sequential_chain()
    example_2_sequential_chain_multiple_inputs_outputs()
    example_3_lcel_with_runnable_passthrough()
    example_4_parallel_execution()
    example_5_complex_chain()
