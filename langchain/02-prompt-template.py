"""
LangChain 学习 02：提示词模板（Prompt Template）

知识点：
1. PromptTemplate 的基础用法
2. ChatPromptTemplate 的使用（系统消息 + 用户消息）
3. 模板的部分填充（Partial Prompt Templates）
4. 从文件加载提示词模板
5. 输出解析器（Output Parser）的基础概念
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()


def example_1_basic_prompt_template():
    """示例 1：最基础的 PromptTemplate"""
    print("=" * 60)
    print("示例 1：基础 PromptTemplate")
    print("=" * 60)

    # 创建一个简单的字符串模板
    prompt = PromptTemplate(
        input_variables=["product", "audience"],  # 定义变量
        template="请为 {product} 写一段针对 {audience} 的广告语。"
    )

    # 格式化提示词
    formatted_prompt = prompt.format(
        product="智能手表",
        audience="健身爱好者"
    )

    print(f"格式化后的提示词:\n{formatted_prompt}\n")

    # 也可以直接调用 prompt.invoke()
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"product": "智能手表", "audience": "健身爱好者"})
    print(f"LLM 回答: {result}\n")


def example_2_chat_prompt_template():
    """示例 2：ChatPromptTemplate（聊天模型专用）"""
    print("=" * 60)
    print("示例 2：ChatPromptTemplate - 系统消息 + 用户消息")
    print("=" * 60)

    # 方法一：使用 from_messages 创建聊天模板
    prompt = ChatPromptTemplate.from_messages([
        # 系统消息：设定 AI 的角色和行为
        SystemMessage(content="你是一个专业的{role}，总是{style}地回答问题。"),
        # 用户消息
        HumanMessage(content="{question}")
    ])

    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # 创建链
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "role": "Python 编程专家",
        "style": "简洁明了",
        "question": "什么是装饰器？"
    })

    print(f"回答:\n{result}\n")


def example_3_chat_prompt_template_builder():
    """示例 3：使用 Builder 模式创建聊天模板"""
    print("=" * 60)
    print("示例 3：使用消息模板类")
    print("=" * 60)

    # 创建系统消息模板
    system_template = SystemMessagePromptTemplate.from_template(
        "你是一个{domain}专家，你的名字叫{name}。"
    )

    # 创建用户消息模板
    human_template = HumanMessagePromptTemplate.from_template(
        "{user_input}"
    )

    # 组合成聊天提示词模板
    chat_prompt = ChatPromptTemplate.from_messages([
        system_template,
        human_template
    ])

    llm = ChatOpenAI(model="gpt-3.5-turbo")
    chain = chat_prompt | llm | StrOutputParser()

    result = chain.invoke({
        "domain": "天文学",
        "name": "星巴克",
        "user_input": "什么是黑洞？"
    })

    print(f"回答:\n{result}\n")


def example_4_partial_prompt():
    """示例 4：部分填充提示词模板"""
    print("=" * 60)
    print("示例 4：部分填充（Partial Prompt）")
    print("=" * 60)

    # 创建一个包含多个变量的模板
    prompt = PromptTemplate(
        template="在{company}，{role}的主要职责是{responsibility}。",
        input_variables=["company", "role", "responsibility"]
    )

    # 预填充部分变量（比如公司信息是固定的）
    partial_prompt = prompt.partial(
        company="谷歌",
        role="软件工程师"
    )

    # 使用时只需要填写剩余变量
    formatted = partial_prompt.format(responsibility="开发搜索引擎算法")
    print(f"部分填充结果:\n{formatted}\n")

    # 也可以在定义时就部分填充
    prompt_with_defaults = PromptTemplate(
        template="你好{name}，欢迎使用{product}！",
        input_variables=["name"],
        partial_variables={"product": "超级App"}  # 默认值
    )

    print(f"使用默认值: {prompt_with_defaults.format(name='张三')}\n")


def example_5_validation_template():
    """示例 5：带验证的提示词模板"""
    print("=" * 60)
    print("示例 5：模板验证")
    print("=" * 60)

    # 创建模板
    prompt = PromptTemplate(
        template="请分析{company}的{aspect}",
        input_variables=["company", "aspect"]
    )

    # 验证输入变量
    print(f"模板需要的变量: {prompt.input_variables}")
    print(f"模板内容:\n{prompt.template}\n")

    # 检查变量是否完整
    try:
        # 缺少 aspect 变量会报错
        formatted = prompt.format(company="特斯拉")
    except Exception as e:
        print(f"预期错误: {e}\n")


def example_6_output_parsers():
    """示例 6：输出解析器"""
    print("=" * 60)
    print("示例 6：使用输出解析器")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # 1. 字符串输出解析器（最常用）
    prompt = ChatPromptTemplate.from_template(
        "用一句话解释{concept}"
    )
    chain1 = prompt | llm | StrOutputParser()
    result1 = chain1.invoke({"concept": "量子计算"})
    print(f"字符串输出: {result1}\n")

    # 2. JSON 输出解析器
    prompt2 = ChatPromptTemplate.from_template(
        "请用JSON格式回答，包含 'definition' 和 'example' 两个字段。"
        "主题：{topic}。只返回JSON，不要其他文字。"
    )
    chain2 = prompt2 | llm | JsonOutputParser()
    result2 = chain2.invoke({"topic": "递归"})
    print(f"JSON 输出: {result2}")
    print(f"访问字段: {result2['example']}\n")


# 总结：核心概念
"""
【提示词模板的核心作用】
1. 复用性：一次定义模板，多次使用
2. 参数化：通过变量动态生成提示词
3. 结构化：对于聊天模型，区分系统消息和用户消息
4. 可维护性：集中管理提示词，方便修改

【ChatPromptTemplate vs PromptTemplate】
- PromptTemplate：用于简单的文本模型，生成纯文本提示词
- ChatPromptTemplate：用于聊天模型，生成结构化的消息列表（系统/用户/助手消息）

【输出解析器（Output Parser）】
- StrOutputParser：将 LLM 输出转换为字符串（最常用）
- JsonOutputParser：将 LLM 输出解析为 JSON 对象
- 其他：ListOutputParser, CommaSeparatedListOutputParser 等

【管道操作符 |】
prompt | llm | parser 表示将三个组件串联成一个链
"""

if __name__ == "__main__":
    example_1_basic_prompt_template()
    example_2_chat_prompt_template()
    example_3_chat_prompt_template_builder()
    example_4_partial_prompt()
    example_5_validation_template()
    example_6_output_parsers()
