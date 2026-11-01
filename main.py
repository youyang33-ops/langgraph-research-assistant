"""命令行入口：运行研究助手，并把图的每一步执行过程打印出来。

用法：
    python main.py "AI Agent 在国内的落地现状"            # 真实模型（需配置 .env）
    python main.py "AI Agent 在国内的落地现状" --mock     # 离线模式，不需要 API key
    python main.py --graph                                # 打印图结构（Mermaid）
    python main.py --history <thread_id>                  # 查看某次运行的 checkpoint 历史

本文件演示的概念：
- checkpointer（SqliteSaver）+ thread_id：短期记忆 / 断点续跑（module-1/agent-memory）；
- store（SqliteStore）：跨 thread 的长期记忆（module-5）；
- stream(stream_mode="updates", subgraphs=True)：逐步观察每个节点的输出（module-3/streaming）；
- 处理 interrupt，并用 Command(resume=...) 恢复（module-3，新版写法）；
- get_state_history：查看 checkpoint 历史，也就是 time travel 的基础（module-3/time-travel）。
"""

import argparse
import os
import sqlite3
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"


def parse_args():
    parser = argparse.ArgumentParser(description="LangGraph 研究助手")
    parser.add_argument("topic", nargs="?", help="研究课题")
    parser.add_argument("--analysts", type=int, default=2, help="分析师数量（= 并行子图数量）")
    parser.add_argument("--turns", type=int, default=2, help="每场访谈的问答轮数")
    parser.add_argument("--thread", help="thread_id；传入已有的 id 可以接着上次中断的地方继续")
    parser.add_argument("--user", default="default-user", help="用户 id，用于区分长期记忆")
    parser.add_argument("--mock", action="store_true", help="使用离线假模型")
    parser.add_argument("--auto-approve", action="store_true", help="自动批准分析师名单，不等待输入")
    parser.add_argument("--graph", action="store_true", help="打印图结构")
    parser.add_argument("--history", metavar="THREAD_ID", help="打印某个 thread 的 checkpoint 历史")
    return parser.parse_args()


def short(value, limit=80) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def print_task(namespace: tuple, task: dict):
    """把一个执行完的节点打印成一行。

    namespace 为空表示主图节点；否则形如 ("conduct_interview:<task_id>",)，表示来自子图。
    Send 并行启动的多个子图实例 task_id 不同，这里取前 4 位来区分它们。
    """
    if namespace:
        name, task_id = namespace[0].split(":")
        where = f"  └ [{name}#{task_id[:4]}] "
    else:
        where = ""
    writes = task.get("result") or {}
    changed = ", ".join(f"{k}={short(v, 50)}" for k, v in writes.items()) or "（无更新）"
    print(f"▶ {where}{task['name']}: {changed}")


def run(graph, stream_input, config, context, auto_approve: bool):
    """反复执行 stream，直到图不再被 interrupt 暂停。

    stream_mode 的常见取值：
    - "values"：每步结束后输出完整 State；
    - "updates"：每步输出各节点返回的更新（但子图设置了 output_schema 时，子图内部字段会被过滤）；
    - "tasks"：每个节点开始/结束时各输出一次，结束事件里带着节点的完整返回值和 interrupt 信息；
    - "messages"：逐 token 输出 LLM 的生成内容（做打字机效果用）。
    这里用 "tasks"，并开启 subgraphs=True，这样子图内部的每一步也能看到。
    """
    from langgraph.types import Command

    while True:
        interrupted = None
        for namespace, task in graph.stream(
            stream_input, config, context=context, stream_mode="tasks", subgraphs=True
        ):
            if "result" not in task:
                continue  # 节点开始执行的事件，跳过，只打印执行完的
            if task.get("interrupts"):
                interrupted = task["interrupts"][0]["value"]
                continue
            print_task(namespace, task)

        if interrupted is None:
            return  # 图正常结束

        # ---- 图在 human_feedback 节点暂停了，此时状态已经存进 checkpoint ----
        print("\n⏸  图已暂停：" + interrupted["question"])
        for a in interrupted["analysts"]:
            print(f"   - {a['name']}（{a['role']}，{a['affiliation']}）：{a['description']}")
        feedback = "" if auto_approve else input("你的意见 > ")
        print()
        # 用 Command(resume=...) 恢复：interrupt() 会返回这里传入的值
        stream_input = Command(resume=feedback)


def main():
    args = parse_args()
    if args.mock:
        os.environ["MOCK_LLM"] = "1"

    # 先处理好环境变量，再导入图（config.get_llm 会读取 MOCK_LLM）
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.store.sqlite import SqliteStore

    from research_assistant.config import Context
    from research_assistant.graph import build_graph

    if args.graph:
        print(build_graph().compile().get_graph(xray=True).draw_mermaid())
        return

    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # checkpointer：每个 super-step 结束都会把 State 存一份，按 thread_id 区分。
    # State 里有自定义的 Pydantic 类型 Analyst，需要显式允许它被反序列化（安全限制）。
    serde = JsonPlusSerializer(allowed_msgpack_modules=[("research_assistant.state", "Analyst")])
    checkpointer = SqliteSaver(
        sqlite3.connect(DATA_DIR / "checkpoints.db", check_same_thread=False), serde=serde
    )
    # store：长期记忆，和 thread 无关
    store_conn = sqlite3.connect(DATA_DIR / "memory.db", check_same_thread=False, isolation_level=None)
    store = SqliteStore(store_conn)
    store.setup()

    graph = build_graph().compile(checkpointer=checkpointer, store=store)

    if args.history:
        config = {"configurable": {"thread_id": args.history}}
        for snapshot in graph.get_state_history(config):
            step = snapshot.metadata.get("step")
            print(f"step={step:<3} 下一步={snapshot.next}  checkpoint_id={snapshot.config['configurable']['checkpoint_id']}")
        return

    if not args.topic and not args.thread:
        raise SystemExit("请提供研究课题，例如：python main.py \"AI Agent 的发展\" --mock")

    thread_id = args.thread or uuid.uuid4().hex[:8]
    config = {"configurable": {"thread_id": thread_id}}
    context = Context(user_id=args.user)
    print(f"thread_id = {thread_id}（之后可用 --history {thread_id} 查看执行历史）\n")

    state = graph.get_state(config)
    if state.next:
        # 这个 thread 上次停在了某个 interrupt，传 None 表示"从断点接着跑"
        print(f"检测到未完成的运行，从 {state.next} 继续。\n")
        stream_input = None
        if state.interrupts:
            from langgraph.types import Command

            stream_input = Command(resume="" if args.auto_approve else input("你的意见 > "))
    else:
        stream_input = {"topic": args.topic, "max_analysts": args.analysts, "max_num_turns": args.turns}

    try:
        run(graph, stream_input, config, context, args.auto_approve)
    except (KeyboardInterrupt, EOFError):
        # 状态已经存在 checkpoint 里了，退出程序也不会丢
        raise SystemExit(f"\n已退出。之后运行 python main.py --thread {thread_id} 可从断点继续。")

    report = graph.get_state(config).values.get("final_report")
    if report:
        path = OUTPUT_DIR / f"report_{thread_id}.md"
        path.write_text(report, encoding="utf-8")
        print(f"\n✅ 报告已生成：{path.relative_to(ROOT)}\n")
        print(report)


if __name__ == "__main__":
    main()
