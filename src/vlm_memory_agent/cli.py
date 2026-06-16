"""功能: 提供单任务 agent 运行的命令行入口，负责组装环境、模型后端、记忆库和输出文件。
上游依赖: 依赖 agent、mock/OSWorld 环境、OpenAI/Qwen/rule VLM 后端和 HierarchicalMemoryStore。
下游依赖: `python -m vlm_memory_agent`、OSWorld 运行脚本、smoke 脚本和测试调用本模块。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from vlm_memory_agent.agent import AgentConfig, VLMGuiAgent
from vlm_memory_agent.envs.local_browser import LocalBrowserSalesEnv
from vlm_memory_agent.envs.mock_osworld import MockOSWorldEnv
from vlm_memory_agent.envs.osworld_runner import OSWorldEnvConfig, build_osworld_adapter
from vlm_memory_agent.llm.openai_compatible import OpenAICompatibleVLMClient
from vlm_memory_agent.llm.qwen36_local import DEFAULT_QWEN36_MODEL_PATH, Qwen36LocalVLMClient
from vlm_memory_agent.llm.rule_based import RuleBasedVLMClient
from vlm_memory_agent.memory.controller import MemoryController, MemoryControllerConfig
from vlm_memory_agent.memory.reflection_clients import DEFAULT_MEMORY_REFLECTION_MODEL_PATH, Qwen35MemoryReflectionClient
from vlm_memory_agent.memory.store import HierarchicalMemoryStore


MOCK_DEFAULT_TASK_ID = "search_report"
LOCAL_BROWSER_DEFAULT_TASK_ID = "sales_approval"


def build_vlm(args: argparse.Namespace):
    """按 CLI 参数构造模型后端。

    `openai` 覆盖 GPT、vLLM、SGLang 等 OpenAI-compatible endpoint；
    `qwen36-local` 用本地 Transformers 直载权重；`rule` 用于确定性测试。
    """

    if args.vlm_backend == "openai":
        return OpenAICompatibleVLMClient(model=args.model, base_url=args.base_url, api_key=args.api_key)
    if args.vlm_backend == "qwen36-local":
        return Qwen36LocalVLMClient(
            model_path=args.model_path,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            max_new_tokens=args.max_new_tokens,
        )
    return RuleBasedVLMClient()


def build_env(args: argparse.Namespace):
    """按 CLI 参数构造交互环境。

    mock/local_browser 不需要外部依赖；真实 OSWorld 则通过
    OSWorldEnvConfig 汇总 provider、VM、snapshot、observation/action
    schema 等参数，再交给 osworld_runner 构造 adapter。
    """

    if args.env == "mock_osworld":
        return MockOSWorldEnv()
    if args.env == "local_browser":
        return LocalBrowserSalesEnv(screenshot_dir=args.screenshot_dir)
    if not args.osworld_task_config:
        raise ValueError("--osworld-task-config is required when --env osworld.")
    return build_osworld_adapter(
        OSWorldEnvConfig(
            task_config=args.osworld_task_config,
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
            screenshot_dir=args.screenshot_dir,
            adapter_action_mode=args.adapter_action_mode,
            evaluate_on_done=not args.no_evaluate_on_done,
        )
    )


def build_memory_controller(memory: HierarchicalMemoryStore, args: argparse.Namespace) -> MemoryController:
    """按 CLI 参数构造 memory controller。

    默认只启用规则化 consolidation、合并和主动遗忘，不加载任何小模型；
    当显式选择 `qwen35-local` 时，才把本地 Qwen3.5-4B 作为反思增强客户端。
    """

    reflection_client = None
    if args.memory_reflection_backend == "qwen35-local":
        reflection_client = Qwen35MemoryReflectionClient(
            model_path=args.memory_reflection_model_path,
            device_map=args.memory_reflection_device_map,
            torch_dtype=args.memory_reflection_torch_dtype,
            max_new_tokens=args.memory_reflection_max_new_tokens,
        )
    return MemoryController(
        memory,
        reflection_client=reflection_client,
        config=MemoryControllerConfig(
            max_nodes=args.memory_max_nodes,
            enable_reflection_enhancement=not args.disable_memory_reflection_enhancement,
            enable_active_forgetting=not args.disable_active_forgetting,
        ),
    )


def run(args: argparse.Namespace) -> int:
    """CLI 主执行流程。

    这里完成依赖注入：memory store、VLM backend、agent config、environment。
    运行结束后无论成功失败都会写 trajectory；进程退出码用 success 映射，
    便于 shell 脚本和 CI 判断任务是否完成。
    """

    memory = HierarchicalMemoryStore(args.memory_path)
    memory_controller = build_memory_controller(memory, args)
    agent = VLMGuiAgent(
        vlm=build_vlm(args),
        memory=memory,
        memory_controller=memory_controller,
        config=AgentConfig(max_steps=args.max_steps, retrieve_k=args.retrieve_k, update_memory=not args.no_memory_update),
    )
    env = build_env(args)
    task_id = args.task_id
    if task_id is None and args.env == "mock_osworld":
        task_id = MOCK_DEFAULT_TASK_ID
    if task_id is None and args.env == "local_browser":
        task_id = LOCAL_BROWSER_DEFAULT_TASK_ID
    if task_id is None:
        task_id = getattr(env, "task_id", None) or None
    try:
        trajectory = agent.run_episode(env, task_id=task_id)
    finally:
        env.close()
    payload = trajectory.to_dict()
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if trajectory.success else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a memory-augmented VLM GUI agent.")
    parser.add_argument("--env", choices=["mock_osworld", "local_browser", "osworld"], default="mock_osworld")
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--memory-path", default="runs/memory.json")
    parser.add_argument("--output", default="runs/trajectory.json")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--retrieve-k", type=int, default=4)
    parser.add_argument("--no-memory-update", action="store_true")
    parser.add_argument(
        "--memory-reflection-backend",
        choices=["none", "qwen35-local"],
        default=os.environ.get("MEMORY_REFLECTION_BACKEND", "none"),
    )
    parser.add_argument(
        "--memory-reflection-model-path",
        default=os.environ.get("MEMORY_REFLECTION_MODEL_PATH", DEFAULT_MEMORY_REFLECTION_MODEL_PATH),
    )
    parser.add_argument("--memory-reflection-device-map", default=os.environ.get("MEMORY_REFLECTION_DEVICE_MAP", "auto"))
    parser.add_argument("--memory-reflection-torch-dtype", default=os.environ.get("MEMORY_REFLECTION_TORCH_DTYPE", "auto"))
    parser.add_argument("--memory-reflection-max-new-tokens", type=int, default=256)
    parser.add_argument("--memory-max-nodes", type=int, default=2000)
    parser.add_argument("--disable-memory-reflection-enhancement", action="store_true")
    parser.add_argument("--disable-active-forgetting", action="store_true")
    parser.add_argument("--vlm-backend", choices=["rule", "openai", "qwen36-local"], default="rule")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model-path", default=DEFAULT_QWEN36_MODEL_PATH)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--osworld-task-config", default=None)
    parser.add_argument("--osworld-vm-path", default=None)
    parser.add_argument("--osworld-provider", default=None)
    parser.add_argument("--osworld-region", default=None)
    parser.add_argument("--osworld-snapshot-name", default="init_state")
    parser.add_argument("--osworld-action-space", default="pyautogui")
    parser.add_argument("--osworld-observation-type", default="screenshot_a11y_tree")
    parser.add_argument("--osworld-cache-dir", default=None)
    parser.add_argument("--adapter-action-mode", choices=["pyautogui", "dict"], default="pyautogui")
    parser.add_argument("--screen-width", type=int, default=1920)
    parser.add_argument("--screen-height", type=int, default=1080)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--os-type", default="Ubuntu")
    parser.add_argument("--no-a11y-tree", action="store_true")
    parser.add_argument("--require-terminal", action="store_true")
    parser.add_argument("--enable-osworld-proxy", action="store_true")
    parser.add_argument("--client-password", default="")
    parser.add_argument("--no-evaluate-on-done", action="store_true")
    parser.add_argument("--screenshot-dir", default="runs/osworld_screens")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (FileNotFoundError, IsADirectoryError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
