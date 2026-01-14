"""
配置文件 - 统一管理 LLM 配置
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# Groq API 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "moonshotai/kimi-k2-instruct-0905")


def get_llm(temperature: float = 0.7, model: str = None):
    """
    获取配置好的 LLM 实例

    Args:
        temperature: 温度参数，控制随机性
        model: 模型名称，默认使用配置文件中的模型

    Returns:
        ChatOpenAI 实例
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model or MODEL_NAME,
        temperature=temperature,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
    )


# 测试配置
if __name__ == "__main__":
    print(f"API Key: {OPENAI_API_KEY[:20]}...")
    print(f"API Base: {OPENAI_API_BASE}")
    print(f"Model: {MODEL_NAME}")

    # 测试连接
    print("\n测试 LLM 连接...")
    try:
        llm = get_llm()
        response = llm.invoke("你好，请用一句话介绍你自己。")
        print(f"响应: {response.content}")
        print("\n配置成功！")
    except Exception as e:
        print(f"错误: {e}")
