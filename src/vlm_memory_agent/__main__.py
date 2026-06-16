"""功能: 支持 `python -m vlm_memory_agent` 调用主命令行入口。
上游依赖: 依赖 vlm_memory_agent.cli.main 解析参数并执行 agent。
下游依赖: Shell 脚本、README 示例和用户命令通过该模块启动单任务运行。
"""

from vlm_memory_agent.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
