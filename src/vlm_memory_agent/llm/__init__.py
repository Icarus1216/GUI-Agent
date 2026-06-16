"""功能: 标记 llm 子包，承载 VLM 后端接口、解析器和具体模型客户端。
上游依赖: 被 Python 包导入机制和后端模块使用。
下游依赖: agent、CLI、healthcheck、tests 从该子包选择 rule/OpenAI/Qwen 本地后端。
"""
