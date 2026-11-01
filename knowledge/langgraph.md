# LangGraph 核心概念（示例知识库，可替换成你自己的资料）

LangGraph 是一个用于构建有状态、多步骤 Agent 的框架。它把应用建模成一张图：节点（node）是普通的 Python 函数，边（edge）决定节点之间的执行顺序，所有节点共享同一份状态（State）。

State 的每个字段都可以指定 reducer。没有 reducer 的字段会被新值覆盖；使用 operator.add 或 add_messages 的字段会把新值追加到旧值之后。当多个节点在同一步并行写入同一个字段时，必须为该字段指定 reducer，否则会抛出 InvalidUpdateError。

LangGraph 的运行时基于 Pregel 模型，按 super-step 执行：每一步找出被触发的节点，并行执行它们，然后统一把它们的写入应用到状态上。每个 super-step 结束后，如果配置了 checkpointer，状态会被保存为一个 checkpoint。

checkpointer 按 thread_id 保存状态，这是短期记忆、断点续跑、人在环和 time travel 的基础。常用实现有 InMemorySaver、SqliteSaver 和 PostgresSaver。

Store 提供跨 thread 的长期记忆。它用命名空间和 key 组织数据，例如 ("users", user_id, "memories")，适合保存用户画像、偏好和历史记录。

interrupt() 可以在节点内部暂停图的执行，把一个值交给调用方。调用方通过 Command(resume=value) 恢复执行，此时节点会从头重新运行，interrupt() 直接返回 value。因此 interrupt 之前的代码应当是幂等的。

Send 用于动态并行（map-reduce）：一个节点可以返回多个 Send(节点名, 输入)，LangGraph 会为每个 Send 启动一个独立的节点实例，并行执行后再通过 reducer 汇总结果。

子图（subgraph）是把一个编译好的图当作另一个图的节点使用。父图和子图共享同名的状态字段，子图可以通过 output_schema 限制返回给父图的字段。
