"""
LangChain 学习 13：高级输出解析器

知识点：
1. PydanticOutputParser：结构化数据提取
2. CommaSeparatedListOutputParser：列表输出
3. DatetimeOutputParser：日期时间解析
4. 自定义输出解析器
5. 处理解析错误和重试
"""

import sys
import io
from datetime import datetime

# 设置标准输出为 UTF-8 编码
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    CommaSeparatedListOutputParser
)
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field
from typing import List, Optional
from config import get_llm


# ============ Pydantic 模型定义 ============

class MovieReview(BaseModel):
    """电影评论的数据模型"""
    title: str = Field(description="电影标题")
    rating: float = Field(description="评分，0-10分", ge=0, le=10)
    summary: str = Field(description="简短摘要")
    pros: List[str] = Field(description="优点列表")
    cons: List[str] = Field(description="缺点列表")
    recommended: bool = Field(description="是否推荐")


class Person(BaseModel):
    """个人信息模型"""
    name: str = Field(description="姓名")
    age: int = Field(description="年龄", ge=0, le=150)
    occupation: str = Field(description="职业")
    skills: List[str] = Field(description="技能列表")
    email: Optional[str] = Field(description="电子邮箱（可选）", default=None)


class Event(BaseModel):
    """事件模型"""
    name: str = Field(description="事件名称")
    date: str = Field(description="日期，格式：YYYY-MM-DD")
    location: str = Field(description="地点")
    attendees: int = Field(description="参加人数")


# ============ 示例 1：PydanticOutputParser 基础 ============

def example_1_pydantic_parser():
    """示例 1：使用 PydanticOutputParser 提取结构化数据"""
    print("=" * 70)
    print("示例 1：PydanticOutputParser - 结构化数据提取")
    print("=" * 70)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # 创建解析器
    parser = PydanticOutputParser(pydantic_object=MovieReview)

    # 获取格式说明
    format_instructions = parser.get_format_instructions()

    print("格式说明:")
    print(format_instructions[:300] + "...\n")

    # 创建提示词
    prompt = PromptTemplate(
        template="""分析以下电影评论，提取结构化信息。

评论内容：
{review}

{format_instructions}

确保：
- 评分在 0-10 之间
- 优点和缺点用列表表示
- 推荐用 true/false 表示
""",
        input_variables=["review"],
        partial_variables={"format_instructions": format_instructions}
    )

    # 创建链
    chain = prompt | llm | parser

    # 测试
    review = """
    我昨天看了《盗梦空间》，这部电影真是太精彩了！诺兰的导演功力深厚，
    剧情烧脑但逻辑严密。视觉效果震撼，配乐也很棒。演员表演出色。

    我给这部电影 9 分。缺点可能是剧情比较复杂，需要认真看才能理解。
    但这正是它的优点。强烈推荐给喜欢科幻和悬疑的观众。
    """

    print("分析评论...")
    try:
        result = chain.invoke({"review": review})

        print("\n提取的结构化数据:")
        print(f"  标题: {result.title}")
        print(f"  评分: {result.rating}/10")
        print(f"  摘要: {result.summary}")
        print(f"  优点: {', '.join(result.pros)}")
        print(f"  缺点: {', '.join(result.cons)}")
        print(f"  推荐: {'是' if result.recommended else '否'}")

    except Exception as e:
        print(f"解析错误: {e}")


# ============ 示例 2：提取多个对象 ============

def example_2_multiple_objects():
    """示例 2：从文本中提取多个对象"""
    print("=" * 70)
    print("示例 2：提取多个对象")
    print("=" * 70)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # 定义包含多个对象的模型
    class PersonList(BaseModel):
        """人员列表"""
        people: List[Person] = Field(description="人员信息列表")

    parser = PydanticOutputParser(pydantic_object=PersonList)

    prompt = PromptTemplate(
        template="""从以下文本中提取所有人员的信息。

文本：
{text}

{format_instructions}
""",
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | llm | parser

    text = """
    张三是一名软件工程师，今年28岁。他精通 Python、JavaScript 和 Go 语言。
    他的邮箱是 zhangsan@example.com。

    李四是数据科学家，32岁。擅长机器学习和数据分析，熟练使用 Pandas 和 TensorFlow。

    王五是产品经理，35岁。负责产品规划和用户研究。
    """

    print("提取人员信息...")
    try:
        result = chain.invoke({"text": text})

        print(f"\n找到 {len(result.people)} 个人:")
        for i, person in enumerate(result.people, 1):
            print(f"\n人员 {i}:")
            print(f"  姓名: {person.name}")
            print(f"  年龄: {person.age}")
            print(f"  职业: {person.occupation}")
            print(f"  技能: {', '.join(person.skills)}")
            if person.email:
                print(f"  邮箱: {person.email}")

    except Exception as e:
        print(f"解析错误: {e}")


# ============ 示例 3：CommaSeparatedListOutputParser ============

def example_3_list_parser():
    """示例 3：逗号分隔列表解析器"""
    print("=" * 70)
    print("示例 3：CommaSeparatedListOutputParser")
    print("=" * 70)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # 创建列表解析器
    parser = CommaSeparatedListOutputParser()

    # 获取格式说明
    format_instructions = parser.get_format_instructions()

    print("格式说明:")
    print(format_instructions)
    print()

    # 创建提示词
    prompt = PromptTemplate(
        template="""列出5个{topic}。

{format_instructions}
""",
        input_variables=["topic"],
        partial_variables={"format_instructions": format_instructions}
    )

    # 创建链
    chain = prompt | llm | parser

    # 测试
    topics = ["Python Web 框架", "机器学习算法", "编程最佳实践"]

    for topic in topics:
        print(f"主题: {topic}")
        result = chain.invoke({"topic": topic})

        print(f"结果 ({len(result)} 项):")
        for i, item in enumerate(result, 1):
            print(f"  {i}. {item}")
        print()


# ============ 示例 4：处理解析错误 ============

def example_4_error_handling():
    """示例 4：处理解析错误和重试"""
    print("=" * 70)
    print("示例 4：错误处理和重试")
    print("=" * 70)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    parser = PydanticOutputParser(pydantic_object=Event)

    prompt = PromptTemplate(
        template="""提取事件信息。

文本：
{text}

{format_instructions}
""",
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # 创建带有错误处理的链
    def parse_with_retry(text, max_retries=3):
        """带重试的解析函数"""
        chain = prompt | llm

        for attempt in range(max_retries):
            try:
                print(f"尝试 {attempt + 1}/{max_retries}...")
                output = chain.invoke({"text": text})
                result = parser.parse(output)
                print("✓ 解析成功!")
                return result

            except Exception as e:
                print(f"✗ 解析失败: {str(e)[:100]}")
                if attempt < max_retries - 1:
                    print("  重试中...")
                    # 可以在这里添加纠正提示
                else:
                    print("  达到最大重试次数")
                    return None

    # 测试案例
    test_cases = [
        "团队会议定在下周三下午3点，在会议室A。",
        "产品发布会将于2024年3月15日在北京举行，预计有500人参加。",
        "invalid input that will fail parsing"
    ]

    for i, text in enumerate(test_cases, 1):
        print(f"\n测试 {i}:")
        print(f"输入: {text}")

        result = parse_with_retry(text)

        if result:
            print(f"事件: {result.name}")
            print(f"日期: {result.date}")
            print(f"地点: {result.location}")
            print(f"人数: {result.attendees}")


# ============ 示例 5：自定义输出解析器 ============

def example_5_custom_parser():
    """示例 5：自定义输出解析器"""
    print("=" * 70)
    print("示例 5：自定义输出解析器")
    print("=" * 70)

    from langchain_core.output_parsers import BaseOutputParser

    class CustomListOutputParser(BaseOutputParser):
        """自定义列表解析器，支持多种分隔符"""

        def parse(self, text: str) -> list[str]:
            """解析文本为列表"""
            # 尝试不同的分隔符
            separators = ["\n", ";", ","]

            for sep in separators:
                if sep in text:
                    items = [item.strip() for item in text.split(sep)]
                    # 过滤空项
                    items = [item for item in items if item]
                    if items:
                        return items

            # 如果没有找到分隔符，返回整个文本
            return [text.strip()]

        def get_format_instructions(self) -> str:
            return """请用换行符分隔每个项目，每行一个。"""

    # 使用自定义解析器
    parser = CustomListOutputParser()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    prompt = PromptTemplate(
        template="""列出{topic}的前5个。

{format_instructions}
""",
        input_variables=["topic"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | llm | parser

    print("使用自定义解析器:")
    result = chain.invoke({"topic": "JavaScript 框架"})

    print(f"结果 ({len(result)} 项):")
    for i, item in enumerate(result, 1):
        print(f"  {i}. {item}")


# ============ 示例 6：组合多个解析器 ============

def example_6_combining_parsers():
    """示例 6：组合多个解析器"""
    print("=" * 70)
    print("示例 6：组合多个解析器")
    print("=" * 70)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # 首先使用 Pydantic 解析器提取结构化数据
    class AnalysisResult(BaseModel):
        """分析结果"""
        summary: str = Field(description="总结")
        key_points: List[str] = Field(description="关键点列表")
        sentiment: str = Field(description="情感：positive/neutral/negative")

    pydantic_parser = PydanticOutputParser(pydantic_object=AnalysisResult)

    # 然后对关键点进行进一步处理
    def process_key_points(result: AnalysisResult) -> dict:
        """进一步处理解析结果"""
        return {
            "summary": result.summary,
            "sentiment": result.sentiment,
            "key_points_count": len(result.key_points),
            "key_points": result.key_points,
            "has_positive_sentiment": result.sentiment == "positive"
        }

    # 创建链
    prompt = PromptTemplate(
        template="""分析以下文本的情感和关键点。

文本：
{text}

{format_instructions}
""",
        input_variables=["text"],
        partial_variables={"format_instructions": pydantic_parser.get_format_instructions()}
    )

    chain = prompt | llm | pydantic_parser | process_key_points

    # 测试
    text = """
    今天使用了新的开发工具，体验非常棒！界面简洁，功能强大，
    大大提高了我的工作效率。虽然有些功能还需要学习，
    但总体来说这是一个很棒的产品。强烈推荐给其他开发者。
    """

    print("分析文本:")
    print(f"  {text}\n")

    result = chain.invoke({"text": text})

    print("分析结果:")
    print(f"  总结: {result['summary']}")
    print(f"  情感: {result['sentiment']}")
    print(f"  关键点数量: {result['key_points_count']}")
    print(f"  关键点:")
    for i, point in enumerate(result['key_points'], 1):
        print(f"    {i}. {point}")
    print(f"  积极情感: {'是' if result['has_positive_sentiment'] else '否'}")


# ============ 示例 7：输出解析器最佳实践 ============

def example_7_best_practices():
    """示例 7：输出解析器最佳实践"""
    print("=" * 70)
    print("示例 7：最佳实践")
    print("=" * 70)

    print("""
使用输出解析器的最佳实践：

1. 选择合适的解析器：
   ✓ PydanticOutputParser - 复杂结构化数据（推荐）
   ✓ JsonOutputParser - 简单 JSON 数据
   ✓ CommaSeparatedListOutputParser - 简单列表
   ✓ StrOutputParser - 纯文本输出
   ✓ 自定义解析器 - 特殊需求

2. Pydantic 模型设计：
   ✓ 使用 Field 添加详细描述
   ✓ 设置合理的约束（ge, le, regex 等）
   ✓ 使用 Optional 标记可选字段
   ✓ 提供清晰的字段描述

3. 错误处理：
   ✓ 总是使用 try-except
   ✓ 实现重试机制
   ✓ 提供清晰的错误信息
   ✓ 记录解析失败的情况

4. 提示词优化：
   ✓ 明确输出格式要求
   ✓ 提供示例
   ✓ 强调约束条件
   ✓ 使用格式说明（format_instructions）

5. 性能优化：
   ✓ 缓存解析器实例
   ✓ 重用 Pydantic 模型
   ✓ 避免不必要的解析

示例：健壮的解析函数
```python
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel

class MyData(BaseModel):
    field1: str
    field2: int

def safe_parse(llm_output: str, max_retries: int = 3) -> Optional[MyData]:
    \"\"\"安全的解析函数\"\"\"
    parser = PydanticOutputParser(pydantic_object=MyData)

    for attempt in range(max_retries):
        try:
            return parser.parse(llm_output)
        except OutputParserException as e:
            if attempt == max_retries - 1:
                logger.error(f"解析失败: {e}")
                return None
            # 可以在这里添加修正逻辑

    return None
```

6. 调试技巧：
   ✓ 打印原始 LLM 输出
   ✓ 检查格式说明是否正确
   ✓ 验证 Pydantic 模型定义
   ✓ 使用更低的 temperature

7. 常见问题：

   问题：JSON 格式不正确
   解决：在提示词中强调严格的 JSON 格式

   问题：字段类型错误
   解决：在 Field 中明确类型和范围

   问题：缺少必需字段
   解决：在提示词中列出所有必需字段

   问题：解析速度慢
   解决：考虑使用更简单的解析器或缓存
    """)


# 总结：核心概念
"""
【高级输出解析器的核心概念】

1. PydanticOutputParser：
   - 最强大的解析器
   - 使用 Pydantic 模型定义结构
   - 自动生成格式说明
   - 支持类型验证和约束

2. 其他常用解析器：
   - JsonOutputParser：简单 JSON
   - CommaSeparatedListOutputParser：列表
   - StrOutputParser：纯文本
   - DatetimeOutputParser：日期时间

3. 自定义解析器：
   - 继承 BaseOutputParser
   - 实现 parse() 方法
   - 实现 get_format_instructions()

4. 错误处理：
   - 使用 try-except
   - 实现重试机制
   - 提供有意义的错误信息
   - 记录失败案例

5. 最佳实践：
   - 详细的 Field 描述
   - 合理的约束条件
   - 清晰的格式说明
   - 健壮的错误处理

6. 解析器选择指南：
   - 复杂结构 → PydanticOutputParser
   - 简单列表 → CommaSeparatedListOutputParser
   - JSON 对象 → JsonOutputParser
   - 纯文本 → StrOutputParser
   - 特殊需求 → 自定义解析器

【下一步学习】

在 14-callbacks-streaming.py 中，你将学习：
- 如何使用回调机制追踪执行
- 如何实现流式输出
- 如何自定义回调处理器
"""

if __name__ == "__main__":
    example_1_pydantic_parser()
    example_2_multiple_objects()
    example_3_list_parser()
    example_4_error_handling()
    example_5_custom_parser()
    example_6_combining_parsers()
    example_7_best_practices()
