"""访谈子图：一位分析师采访一位"专家"，最后写出报告的一个小节。

    START → ask_question ─┬→ search_web   ─┬→ answer_question ─┬→ ask_question（继续追问，形成循环）
                          └→ search_local ─┘                   └→ save_interview → write_section → END

本文件演示的概念：
- 并行（fan-out / fan-in）：ask_question 之后两条边同时触发 search_web 和 search_local，
  两者都结束后 answer_question 才执行（module-4/parallelization）；
- 循环 + 条件边：route_messages 决定继续追问还是结束（module-1/router、agent）；
- 子图：编译后的子图可以直接当作父图的一个节点（module-4/sub-graph）；
- 输出 schema：子图只把 sections 交还给父图（module-2/multiple-schemas）。
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from research_assistant import prompts
from research_assistant.config import get_llm, structured_llm
from research_assistant.state import InterviewOutput, InterviewState, SearchQuery
from research_assistant.tools import local_search, web_search

END_SIGNAL = "非常感谢你的帮助"


# ----------------------------------------------------------------------------- 节点

def ask_question(state: InterviewState):
    """分析师提问。返回的 messages 会被 add_messages 追加到历史里，而不是覆盖。"""
    system = SystemMessage(prompts.QUESTION_INSTRUCTIONS.format(persona=state["analyst"].persona))
    question = get_llm().invoke([system] + state["messages"])
    # 统一用 HumanMessage 表示分析师（对模型来说它是"提问方"），name 用来区分说话人
    return {"messages": [HumanMessage(question.content, name="analyst")]}


def _make_query(state: InterviewState) -> str:
    conversation = get_buffer_string(state["messages"])
    result = structured_llm(SearchQuery).invoke(
        [SystemMessage(prompts.SEARCH_INSTRUCTIONS), HumanMessage(conversation)]
    )
    return result.search_query


def search_web(state: InterviewState):
    """检索节点 1。返回 {"context": [一段]}，与 search_local 的结果通过 operator.add 合并。"""
    return {"context": [web_search(_make_query(state))]}


def search_local(state: InterviewState):
    """检索节点 2，和 search_web 在同一个 super-step 里并行执行。"""
    return {"context": [local_search(_make_query(state))]}


def answer_question(state: InterviewState):
    """专家基于检索到的资料回答。"""
    system = SystemMessage(
        prompts.ANSWER_INSTRUCTIONS.format(
            persona=state["analyst"].persona, context="\n\n".join(state["context"])
        )
    )
    answer = get_llm().invoke([system] + state["messages"])
    return {"messages": [AIMessage(answer.content, name="expert")]}


def save_interview(state: InterviewState):
    return {"interview": get_buffer_string(state["messages"])}


def write_section(state: InterviewState):
    system = SystemMessage(prompts.SECTION_WRITER_INSTRUCTIONS.format(focus=state["analyst"].description))
    sources = "\n\n".join(state["context"])
    section = get_llm().invoke([system, HumanMessage(f"访谈记录：\n{state['interview']}\n\n资料：\n{sources}")])
    # 这个字段名和父图 ResearchState.sections 相同，子图结束后会被写回父图
    return {"sections": [section.content]}


# ----------------------------------------------------------------------------- 路由

def route_messages(state: InterviewState):
    """条件边：返回下一个节点的名字。"""
    expert_turns = sum(1 for m in state["messages"] if isinstance(m, AIMessage) and m.name == "expert")
    if expert_turns >= state.get("max_num_turns", 2):
        return "save_interview"
    # 倒数第二条是分析师的问题；如果分析师说了结束语，就停止追问
    last_question = state["messages"][-2]
    if END_SIGNAL in last_question.content:
        return "save_interview"
    return "ask_question"


# ----------------------------------------------------------------------------- 组装

def build_interview_graph():
    builder = StateGraph(InterviewState, output_schema=InterviewOutput)

    # 检索节点调用了外部服务，加上重试策略：失败后自动重试，最多 3 次（Thinking in LangGraph 的"临时错误"）
    retry = RetryPolicy(max_attempts=3)
    builder.add_node("ask_question", ask_question, retry_policy=retry)
    builder.add_node("search_web", search_web, retry_policy=retry)
    builder.add_node("search_local", search_local, retry_policy=retry)
    builder.add_node("answer_question", answer_question, retry_policy=retry)
    builder.add_node("save_interview", save_interview)
    builder.add_node("write_section", write_section, retry_policy=retry)

    builder.add_edge(START, "ask_question")
    # 一个节点连出两条普通边 = 并行
    builder.add_edge("ask_question", "search_web")
    builder.add_edge("ask_question", "search_local")
    # 两条边汇入同一个节点 = 等两者都完成再继续
    builder.add_edge(["search_web", "search_local"], "answer_question")
    builder.add_conditional_edges("answer_question", route_messages, ["ask_question", "save_interview"])
    builder.add_edge("save_interview", "write_section")
    builder.add_edge("write_section", END)

    return builder.compile()
