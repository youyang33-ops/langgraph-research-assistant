"""离线假模型：不调用任何 API，根据提示词返回固定格式的内容。

只用于 `python main.py --mock`，让你在没有 API key 时也能观察图的完整执行流程。
学习 LangGraph 时【不需要读这个文件】。
"""

import re

from langchain_core.messages import AIMessage, BaseMessage

from research_assistant.state import Analyst, Perspectives, SearchQuery


def _system_text(messages: list[BaseMessage]) -> str:
    return messages[0].content if messages else ""


def _find(pattern: str, text: str, default: str = "") -> str:
    match = re.search(pattern, text)
    return match.group(1).strip() if match else default


class _Structured:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, messages):
        text = _system_text(messages)
        if self.schema is Perspectives:
            topic = _find(r"研究课题：(.+)", text, "未知课题")
            count = int(_find(r"只输出 (\d+) 位", text, "2"))
            feedback = _find(r"人工审核意见.*?：\n(.+)", text, "（无）")
            angles = ["技术架构", "产业应用", "风险与治理", "人才与就业"]
            analysts = [
                Analyst(
                    name=f"分析师{i + 1}",
                    affiliation="模拟研究院",
                    role=f"{angles[i % len(angles)]}研究员",
                    description=f"从{angles[i % len(angles)]}角度研究「{topic}」"
                    + ("" if feedback == "（无）" else f"；已按意见调整：{feedback}"),
                )
                for i in range(count)
            ]
            return Perspectives(analysts=analysts)
        if self.schema is SearchQuery:
            conversation = messages[-1].content
            last_line = [line for line in conversation.splitlines() if line.strip()][-1]
            return SearchQuery(search_query=last_line[-40:])
        raise ValueError(f"MockLLM 不支持的 schema：{self.schema}")


class MockLLM:
    def with_structured_output(self, schema, **kwargs):
        return _Structured(schema)

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        text = _system_text(messages)
        if "你是一名分析师" in text:
            role = _find(r"角色：(.+)", text)
            turn = sum(1 for m in messages if getattr(m, "name", None) == "analyst") + 1
            return AIMessage(f"我是{role}。第 {turn} 个问题：LangGraph 的状态和 checkpoint 是怎么工作的？")
        if "你是一名专家" in text:
            sources = re.findall(r'source="([^"]+)"', text)
            return AIMessage(f"根据资料，LangGraph 用共享 State 和 checkpoint 管理执行过程 [1]。来源：{sources[:2]}")
        if "技术写作者" in text:
            focus = _find(r"分析师关注点：\n(.+)", text)
            return AIMessage(f"## 关于{focus[:20]}的发现\n\n### 摘要\n模拟的小节内容 [1]。\n\n### 来源\n[1] knowledge/langgraph.md")
        if "整合成报告正文" in text:
            count = text.count("## 关于")
            return AIMessage(f"## 洞察\n\n综合 {count} 个小节的模拟正文 [1]。\n\n## 来源\n[1] knowledge/langgraph.md")
        if "撰写引言" in text:
            return AIMessage("## 引言\n\n这是模拟生成的引言。")
        if "撰写结论" in text:
            return AIMessage("## 结论\n\n这是模拟生成的结论。")
        return AIMessage("（MockLLM 默认回复）")
