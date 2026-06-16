"""功能: 提供 OSWorld task JSON 文件或目录的批量评测命令行入口。
上游依赖: 依赖 cli.build_vlm、OSWorldEnvConfig、BenchmarkRunner 和 Qwen3.6 默认模型路径。
下游依赖: scripts/eval_osworld_qwen36.sh 调用本模块产出 report 和 per-task trajectory。
"""

from __future__ import annotations

import argparse

from vlm_memory_agent.cli import build_vlm
from vlm_memory_agent.envs.osworld_runner import OSWorldEnvConfig
from vlm_memory_agent.eval import BenchmarkRunner, write_report
from vlm_memory_agent.llm.qwen36_local import DEFAULT_QWEN36_MODEL_PATH


def main(argv: list[str] | None = None) -> int:
    """OSWorld batch eval 的命令行入口。

    该入口只负责解析参数、构造 BenchmarkRunner 和基础 OSWorldEnvConfig。
    具体 task 目录遍历、环境创建、trajectory 写入由 eval.py 负责。
    """

    parser = argparse.ArgumentParser(description="Run memory-augmented agent on OSWorld task JSON files.")
    parser.add_argument("--task-config", required=True, help="OSWorld task JSON file or directory.")
    parser.add_argument("--task-id", default=None, help="Task id when --task-config is a directory and only one task is desired.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report", default="runs/osworld_eval_report.json")
    parser.add_argument("--memory-path", default="runs/osworld_eval_memory.json")
    parser.add_argument("--trajectory-dir", default="runs/osworld_eval_trajectories")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--retrieve-k", type=int, default=4)
    parser.add_argument("--vlm-backend", choices=["openai", "qwen36-local"], default="openai")
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model-path", default=DEFAULT_QWEN36_MODEL_PATH)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--osworld-vm-path", default=None)
    parser.add_argument("--osworld-provider", default="vmware")
    parser.add_argument("--osworld-region", default=None)
    parser.add_argument("--osworld-snapshot-name", default="init_state")
    parser.add_argument("--osworld-action-space", default="pyautogui")
    parser.add_argument("--osworld-observation-type", default="screenshot_a11y_tree")
    parser.add_argument("--osworld-cache-dir", default=None)
    parser.add_argument("--screen-width", type=int, default=1920)
    parser.add_argument("--screen-height", type=int, default=1080)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--os-type", default="Ubuntu")
    parser.add_argument("--no-a11y-tree", action="store_true")
    parser.add_argument("--require-terminal", action="store_true")
    parser.add_argument("--enable-osworld-proxy", action="store_true")
    parser.add_argument("--client-password", default="")
    parser.add_argument("--no-evaluate-on-done", action="store_true")
    args = parser.parse_args(argv)

    runner = BenchmarkRunner(build_vlm(args), memory_path=args.memory_path, max_steps=args.max_steps, trajectory_dir=args.trajectory_dir)
    runner.agent.config.retrieve_k = args.retrieve_k
    # base_config 只保存所有任务共享的 provider/VM/屏幕配置；每个任务的
    # task_config 和 screenshot_dir 会在 run_osworld_directory 中派生。
    base_config = OSWorldEnvConfig(
        task_config=args.task_config,
        task_id=args.task_id,
        vm_path=args.osworld_vm_path,
        provider_name=args.osworld_provider,
        region=args.osworld_region,
        snapshot_name=args.osworld_snapshot_name,
        action_space=args.osworld_action_space,
        observation_type=args.osworld_observation_type,
        screen_width=args.screen_width,
        screen_height=args.screen_height,
        headless=args.headless,
        os_type=args.os_type,
        require_a11y_tree=not args.no_a11y_tree,
        require_terminal=args.require_terminal,
        enable_proxy=args.enable_osworld_proxy,
        client_password=args.client_password,
        cache_dir=args.osworld_cache_dir,
        evaluate_on_done=not args.no_evaluate_on_done,
    )
    results = runner.run_osworld_directory(base_config, args.task_config, limit=args.limit)
    write_report(results, args.report)
    success = sum(result.success for result in results)
    print(f"tasks={len(results)} success={success} report={args.report}")
    return 0 if all(result.success for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
