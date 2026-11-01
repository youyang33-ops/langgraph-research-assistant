"""State 定义：整个项目最先要读懂的文件。

LangGraph 的核心思想：图里所有节点共享一份 State。
节点 = 函数：读 State → 返回"要更新的字段" → LangGraph 把更新合并进 State。
合并规则由 reducer 决定：
- 没写 reducer 的字段：新值直接覆盖旧值；
- `Annotated[list, operator.add]`：新值追加到旧列表后面（并行节点同时写也不会冲突）。

对应 langchain-academy：module-2/state-schema、state-reducers、multiple-schemas。
"""

import operator
from typing import Annotated

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# 1. 结构化输出用的 Pydantic 模型（让 LLM 直接返回这些对象）
# ---------------------------------------------------------------------------

class Analyst(BaseModel):
    name: str = Field(description="分析师的名字")
    affiliation: str = Field(description="所属机构")
    role: str = Field(description="在本课题中扮演的角色")
    description: str = Field(description="关注重点、关心的问题和动机")

    @property
    def persona(self) -> str:
        return (
            f"姓名：{self.name}\n机构：{self.affiliation}\n"
            f"角色：{self.role}\n关注点：{self.description}"
        )


class Perspectives(BaseModel):
    analysts: list[Analyst] = Field(description="分析师列表")


class SearchQuery(BaseModel):
    search_query: str = Field(description="用于检索的查询语句")


# ---------------------------------------------------------------------------
# 2. 访谈子图的 State（一个分析师 ↔ 一个专家 的多轮对话）
# ---------------------------------------------------------------------------

class InterviewState(MessagesState):
    """继承 MessagesState，自带 `messages` 字段，其 reducer 是 add_messages（追加消息）。"""

    analyst: Analyst
    max_num_turns: int
    # 两个检索节点会"并行"写 context，所以必须用 operator.add 合并，否则会报
    # InvalidUpdateError: Can receive only one value per step.
    context: Annotated[list[str], operator.add]
    interview: str
    sections: list[str]


class InterviewOutput(TypedDict):
    """子图的输出 schema：只把 sections 交还给父图，其余中间字段都不外泄。"""

    sections: list[str]


# ---------------------------------------------------------------------------
# 3. 主图的 State
# ---------------------------------------------------------------------------

class ResearchInput(TypedDict):
    """主图的输入 schema：调用方只需要提供这几个字段。"""

    topic: str
    max_analysts: int
    max_num_turns: int


class ResearchState(TypedDict):
    topic: str
    max_analysts: int
    max_num_turns: int
    past_topics: list[str]           # 从长期记忆（Store）里读出来的历史课题
    human_analyst_feedback: str      # 人工审核时填写的意见
    analysts: list[Analyst]
    # 多个访谈子图并行执行，每个都返回 {"sections": [一段]}，用 operator.add 汇总
    sections: Annotated[list[str], operator.add]
    introduction: str
    content: str
    conclusion: str
    final_report: str


class ResearchOutput(TypedDict):
    final_report: str
