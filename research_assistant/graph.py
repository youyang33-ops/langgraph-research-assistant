"""主图：研究助手。

    START → load_memory → create_analysts → human_feedback ──(人工提了意见)──→ create_analysts
                                                 │
                                                 └─(批准)─ Send × N ─→ conduct_interview（子图，N 个并行）
                                                                              │
                              ┌───────────────────┬───────────────────────────┤
                              ↓                   ↓                           ↓
                        write_report      write_introduction          write_conclusion   （3 个并行）
                              └───────────────────┴─────────────┬─────────────┘
                                                                ↓
                                              finalize_report → save_memory → END

本文件演示的概念：
- interrupt() + Command(resume=...)：人在环，暂停等待人工审核（module-3，新版写法）；
- Command(goto=...)：在节点内部决定下一步去哪；
- Send：动态 map-reduce，分析师有几个就并行启动几个子图（module-4/map-reduce）；
- Runtime：v1 的运行时对象，节点通过它访问 store（长期记忆）和 context（module-5）。
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, RetryPolicy, Send, interrupt

from research_assistant import prompts
from research_assistant.config import Context, get_llm, structured_llm
from research_assistant.interview import build_interview_graph
from research_assistant.state import Perspectives, ResearchInput, ResearchOutput, ResearchState


def _memory_namespace(runtime: Runtime[Context]) -> tuple[str, ...]:
    # Store 用"命名空间 + key"组织数据，类似文件夹 + 文件名；按用户隔离
    return ("users", runtime.context.user_id, "research_history")


# ----------------------------------------------------------------------------- 节点

def load_memory(state: ResearchState, runtime: Runtime[Context]):
    """从长期记忆读取该用户研究过的课题。

    长期记忆（Store）和短期记忆（checkpointer）的区别：
    - checkpointer 按 thread_id 保存一次对话/一次运行的完整状态；
    - store 跨 thread 共享，换一个 thread_id 也能读到。
    """
    if runtime.store is None:
        return {"past_topics": []}
    items = runtime.store.search(_memory_namespace(runtime), limit=10)
    return {"past_topics": [item.value["topic"] for item in items]}


def create_analysts(state: ResearchState):
    prompt = prompts.ANALYST_INSTRUCTIONS.format(
        topic=state["topic"],
        max_analysts=state["max_analysts"],
        past_topics="\n".join(state.get("past_topics") or []) or "（无）",
        human_analyst_feedback=state.get("human_analyst_feedback") or "（无）",
    )
    result = structured_llm(Perspectives).invoke(
        [SystemMessage(prompt), HumanMessage("请生成分析师团队。")]
    )
    return {"analysts": result.analysts[: state["max_analysts"]]}


def human_feedback(state: ResearchState) -> Command[Literal["create_analysts", "conduct_interview"]]:
    """人在环：把分析师名单交给人审核。

    执行到 interrupt() 时，图会暂停，当前状态存入 checkpoint，调用方拿到 interrupt 的值。
    调用方用 graph.stream(Command(resume="意见"), config) 恢复后，
    这个节点会【从头重新执行】，此时 interrupt() 不再暂停，而是直接返回 resume 的值。
    所以 interrupt() 之前不要放有副作用的代码（比如写数据库），否则会执行两次。
    """
    feedback = interrupt(
        {
            "question": "请审核分析师团队。直接回车表示批准；输入修改意见则重新生成。",
            "analysts": [a.model_dump() for a in state["analysts"]],
        }
    )

    if feedback and feedback.strip():
        # 有意见：带着意见回到 create_analysts
        return Command(goto="create_analysts", update={"human_analyst_feedback": feedback.strip()})

    # 批准：用 Send 为每个分析师启动一个子图实例。
    # Send(节点名, 该实例的输入)——每个实例拿到的是各自独立的输入，而不是共享的 State。
    return Command(
        goto=[
            Send(
                "conduct_interview",
                {
                    "analyst": analyst,
                    "max_num_turns": state["max_num_turns"],
                    "messages": [HumanMessage(f"你好，我们来聊聊这个话题：{state['topic']}")],
                },
            )
            for analyst in state["analysts"]
        ]
    )


def _sections_text(state: ResearchState) -> str:
    return "\n\n".join(state["sections"])


def write_report(state: ResearchState):
    prompt = prompts.REPORT_WRITER_INSTRUCTIONS.format(topic=state["topic"], context=_sections_text(state))
    report = get_llm().invoke([SystemMessage(prompt), HumanMessage("请根据这些小节撰写报告正文。")])
    return {"content": report.content}


def write_introduction(state: ResearchState):
    prompt = prompts.INTRO_CONCLUSION_INSTRUCTIONS.format(
        topic=state["topic"], context=_sections_text(state), part="引言", title="引言"
    )
    intro = get_llm().invoke([SystemMessage(prompt), HumanMessage("请撰写引言。")])
    return {"introduction": intro.content}


def write_conclusion(state: ResearchState):
    prompt = prompts.INTRO_CONCLUSION_INSTRUCTIONS.format(
        topic=state["topic"], context=_sections_text(state), part="结论", title="结论"
    )
    conclusion = get_llm().invoke([SystemMessage(prompt), HumanMessage("请撰写结论。")])
    return {"conclusion": conclusion.content}


def finalize_report(state: ResearchState):
    """fan-in：三个写作节点都完成后才会执行到这里，不需要调用 LLM。"""
    report = "\n\n---\n\n".join([state["introduction"], state["content"], state["conclusion"]])
    return {"final_report": f"# {state['topic']}\n\n{report}"}


def save_memory(state: ResearchState, runtime: Runtime[Context]):
    """把本次课题写进长期记忆，下次 load_memory 就能读到。"""
    if runtime.store is not None:
        runtime.store.put(
            _memory_namespace(runtime),
            key=state["topic"],
            value={"topic": state["topic"], "analysts": [a.name for a in state["analysts"]]},
        )
    return {}


# ----------------------------------------------------------------------------- 组装

def build_graph():
    """返回未编译的 builder。编译时再决定用哪个 checkpointer / store。"""
    builder = StateGraph(
        ResearchState,
        context_schema=Context,
        input_schema=ResearchInput,
        output_schema=ResearchOutput,
    )
    retry = RetryPolicy(max_attempts=3)

    builder.add_node("load_memory", load_memory)
    builder.add_node("create_analysts", create_analysts, retry_policy=retry)
    builder.add_node("human_feedback", human_feedback)
    # 编译好的子图直接作为节点
    builder.add_node("conduct_interview", build_interview_graph())
    builder.add_node("write_report", write_report, retry_policy=retry)
    builder.add_node("write_introduction", write_introduction, retry_policy=retry)
    builder.add_node("write_conclusion", write_conclusion, retry_policy=retry)
    builder.add_node("finalize_report", finalize_report)
    builder.add_node("save_memory", save_memory)

    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "create_analysts")
    builder.add_edge("create_analysts", "human_feedback")
    # human_feedback 之后去哪由它返回的 Command 决定，所以这里不用画边
    # 所有访谈结束后，三个写作节点并行执行
    builder.add_edge("conduct_interview", "write_report")
    builder.add_edge("conduct_interview", "write_introduction")
    builder.add_edge("conduct_interview", "write_conclusion")
    builder.add_edge(["write_report", "write_introduction", "write_conclusion"], "finalize_report")
    builder.add_edge("finalize_report", "save_memory")
    builder.add_edge("save_memory", END)

    return builder


# 给 `langgraph dev`（LangSmith Studio）用的入口。
# 在 Studio / 部署环境里，checkpointer 和 store 由平台提供，所以这里不传。
graph = build_graph().compile()
