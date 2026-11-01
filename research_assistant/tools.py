"""两个检索源：联网搜索 + 本地知识库。

对应 langchain-academy：module-4/research-assistant 里的 search_web（Tavily）和
search_wikipedia。国内访问 Wikipedia 不稳定，这里换成了本地知识库（knowledge/ 目录），
你可以往里面放自己的 .md / .txt 资料。
"""

import os
import re
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"


def _format_docs(docs: list[tuple[str, str]]) -> str:
    return "\n\n---\n\n".join(
        f'<Document source="{source}">\n{content}\n</Document>' for source, content in docs
    )


def web_search(query: str, max_results: int = 3) -> str:
    """联网搜索。配置了 TAVILY_API_KEY 就用 Tavily，否则返回一条提示（不影响流程跑通）。"""
    if not os.getenv("TAVILY_API_KEY"):
        return _format_docs([("web-search-disabled", f"（未配置 TAVILY_API_KEY，跳过联网搜索：{query}）")])

    from langchain_tavily import TavilySearch

    results = TavilySearch(max_results=max_results).invoke({"query": query})["results"]
    return _format_docs([(r["url"], r["content"]) for r in results])


def _tokens(text: str) -> set[str]:
    """极简分词：英文按单词，中文按相邻两个字（bigram）。足够做演示用的关键词匹配。"""
    text = text.lower()
    words = set(re.findall(r"[a-z][a-z0-9_]+", text))
    chinese = "".join(re.findall(r"[一-鿿]", text))
    words |= {chinese[i : i + 2] for i in range(len(chinese) - 1)}
    return words


def local_search(query: str, max_results: int = 2) -> str:
    """在 knowledge/ 目录里按关键词重合度检索段落。真实项目里这里会换成向量数据库。"""
    query_tokens = _tokens(query)
    scored = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")) + sorted(KNOWLEDGE_DIR.glob("*.txt")):
        for i, para in enumerate(p for p in path.read_text(encoding="utf-8").split("\n\n") if p.strip()):
            score = len(query_tokens & _tokens(para))
            if score:
                scored.append((score, f"{path.name}#段落{i + 1}", para.strip()))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [(source, content) for _, source, content in scored[:max_results]]
    return _format_docs(top or [("knowledge", "（本地知识库中没有相关内容）")])
