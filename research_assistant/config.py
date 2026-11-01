"""模型配置：所有节点都通过这里拿 LLM，换模型只改这一个文件（或 .env）。

对应 langchain-academy：每个 notebook 里的 `ChatOpenAI(model="gpt-4o")`。
这里把它集中起来，并默认用国内可用的 DeepSeek（兼容 OpenAI 接口）。
"""

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Context:
    """运行时上下文（LangGraph v1 的 context_schema）。

    它和 State 的区别：
    - State 是图在运行中不断被节点读写的数据，会存进 checkpoint；
    - Context 是调用方在 invoke/stream 时传进来的"只读配置"，比如当前是哪个用户。
    节点里通过 `runtime.context.user_id` 读取。
    """

    user_id: str = "default-user"


@lru_cache
def get_llm():
    """返回一个聊天模型。设置 MOCK_LLM=1 时返回离线假模型，不需要任何 API key。"""
    if os.getenv("MOCK_LLM") == "1":
        from research_assistant.mock_llm import MockLLM

        return MockLLM()

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        api_key=os.getenv("LLM_API_KEY"),
        temperature=0,
    )


def structured_llm(schema):
    """让模型按 Pydantic schema 输出结构化结果。

    注意 method="function_calling"：langchain-openai 默认用 OpenAI 的 json_schema 模式，
    DeepSeek / Qwen 不支持，会报错；用工具调用的方式实现结构化输出则都兼容。
    """
    llm = get_llm()
    if os.getenv("MOCK_LLM") == "1":
        return llm.with_structured_output(schema)
    return llm.with_structured_output(schema, method="function_calling")
