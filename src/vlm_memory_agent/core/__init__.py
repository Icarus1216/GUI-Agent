"""功能: 标记 core 子包，集中放置 agent/environment 共享的数据类型。
上游依赖: 被 Python 包导入机制和类型模块使用。
下游依赖: agent、env、llm parser、memory、tests 通过 core.types 共享结构化协议。
"""
