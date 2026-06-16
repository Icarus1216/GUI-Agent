"""功能: 标记 envs 子包，承载 mock 环境、OSWorld adapter 和 OSWorld 构造逻辑。
上游依赖: 被 Python 包导入机制和环境模块使用。
下游依赖: CLI、BenchmarkRunner、tests 通过该子包获得可交互环境实现。
"""
