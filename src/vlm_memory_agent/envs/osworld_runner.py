"""功能: 读取 OSWorld task config，构造 desktop_env.DesktopEnv，并包装为 OSWorldAdapter。
上游依赖: 依赖 runtime_paths 注入外部 OSWorld 依赖、task JSON、provider/VM 参数和 desktop_env 包。
下游依赖: CLI、BenchmarkRunner、preflight、OSWorld 批量评测和测试复用任务解析与环境构造工具。
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vlm_memory_agent.envs.osworld_adapter import OSWorldAdapter
from vlm_memory_agent.runtime_paths import apply_runtime_dependency_paths


@dataclass(slots=True)
class OSWorldEnvConfig:
    """构造真实 OSWorld DesktopEnv 所需的配置集合。

    这些字段基本对应 OSWorld/desktop_env 的构造参数，但通过 dataclass
    集中管理后，CLI、batch eval、preflight 和测试可以共用同一份配置模型。
    `adapter_action_mode` 是本项目额外的兼容开关，用来选择 pyautogui 字符串
    或 dict wrapper。
    """

    task_config: str | Path
    task_id: str | None = None
    vm_path: str | None = None
    provider_name: str | None = None
    region: str | None = None
    snapshot_name: str = "init_state"
    action_space: str = "pyautogui"
    observation_type: str = "screenshot_a11y_tree"
    screen_width: int = 1920
    screen_height: int = 1080
    headless: bool = False
    os_type: str = "Ubuntu"
    require_a11y_tree: bool = True
    require_terminal: bool = False
    enable_proxy: bool = False
    client_password: str = ""
    cache_dir: str | None = None
    screenshot_dir: str | Path = "runs/osworld_screens"
    adapter_action_mode: str = "pyautogui"
    evaluate_on_done: bool = True


def load_task_config(path: str | Path, task_id: str | None = None) -> dict[str, Any]:
    """读取单个 OSWorld task JSON，或从目录中按 task_id 定位任务。

    OSWorld 官方任务通常按应用/任务目录组织。为了让脚本既能跑单个 JSON，
    又能跑目录批量评测，这里把目录解析也放在同一个入口中。
    """

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"OSWorld task config not found: {config_path}")
    if config_path.is_dir():
        config_path = _resolve_task_config_from_dir(config_path, task_id)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"OSWorld task config must be a JSON object: {config_path}")
    return config


def build_desktop_env(config: OSWorldEnvConfig) -> Any:
    """构造外部 `desktop_env.desktop_env.DesktopEnv` 实例。

    该函数只负责把本项目配置映射到 OSWorld 构造参数，并在初始化失败时
    给出 provider 相关 hint。真正的 action/observation 归一化交给
    `OSWorldAdapter`。
    """

    apply_runtime_dependency_paths()
    try:
        from desktop_env.desktop_env import DesktopEnv
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OSWorld is not importable. Install the OSWorld repository/dependencies so "
            "`from desktop_env.desktop_env import DesktopEnv` works in this Python environment."
        ) from exc

    kwargs: dict[str, Any] = {
        "provider_name": config.provider_name,
        "region": config.region,
        "path_to_vm": config.vm_path,
        "snapshot_name": config.snapshot_name,
        "action_space": config.action_space,
        "observation_type": config.observation_type,
        "screen_size": (config.screen_width, config.screen_height),
        "headless": config.headless,
        "os_type": config.os_type,
        "require_a11y_tree": config.require_a11y_tree,
        "require_terminal": config.require_terminal,
        "enable_proxy": config.enable_proxy,
        "client_password": config.client_password,
        "cache_dir": config.cache_dir,
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    try:
        return _construct_with_supported_kwargs(DesktopEnv, kwargs)
    except Exception as exc:
        vm_hint = _desktop_env_failure_hint(config)
        raise RuntimeError(f"OSWorld DesktopEnv initialization failed: {type(exc).__name__}: {exc}. {vm_hint}") from exc


def build_osworld_adapter(config: OSWorldEnvConfig) -> OSWorldAdapter:
    """从配置一步构造可被 VLMGuiAgent 使用的 OSWorldAdapter。"""

    task_config = load_task_config(config.task_config, task_id=config.task_id)
    env = build_desktop_env(config)
    task_id = config.task_id or str(task_config.get("id") or "")
    return OSWorldAdapter(
        env,
        task_config=task_config,
        action_mode=config.adapter_action_mode,
        screenshot_dir=config.screenshot_dir,
        evaluate_on_done=config.evaluate_on_done,
        task_id=task_id,
    )


def _construct_with_supported_kwargs(cls: Any, kwargs: dict[str, Any]) -> Any:
    """按 DesktopEnv 当前版本支持的参数过滤 kwargs。

    desktop_env 在不同 OSWorld 版本中构造参数会变化；过滤后再调用可以让
    本项目对 provider/cache/headless 等字段保持向前和向后兼容。
    """

    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError):
        return cls(**kwargs)

    params = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return cls(**kwargs)

    filtered = {key: value for key, value in kwargs.items() if key in params}
    return cls(**filtered)


def _desktop_env_failure_hint(config: OSWorldEnvConfig) -> str:
    provider = (config.provider_name or "vmware").lower()
    if config.vm_path:
        return f"Check that --osworld-vm-path exists and is usable for provider {provider}: {config.vm_path}"
    if provider in {"vmware", "virtualbox"}:
        return f"Provider {provider} requires --osworld-vm-path to an existing local OSWorld VM."
    if provider == "docker":
        return "Provider Docker may download the qcow2 image; check Docker availability, image download access, and /dev/kvm permissions."
    return f"Provider {provider} resolves machines through provider configuration; check credentials, region, quota, and network access."


def _resolve_task_config_from_dir(task_dir: Path, task_id: str | None) -> Path:
    """在任务目录中解析指定 task_id 对应的 JSON 文件。

    先按文件 stem 匹配，再打开 JSON 按 id/task_id/uuid 匹配。这样兼容
    官方任务、重命名任务文件和用户自定义任务集合。
    """

    candidates = sorted(task_dir.rglob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No JSON task configs found under: {task_dir}")
    if task_id is None:
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError("--task-id is required when --osworld-task-config points to a directory.")

    for candidate in candidates:
        if candidate.stem == task_id:
            return candidate
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if str(payload.get("id") or payload.get("task_id") or payload.get("uuid") or "") == task_id:
            return candidate
    raise FileNotFoundError(f"No OSWorld task config with id/stem `{task_id}` found under: {task_dir}")


def iter_task_config_paths(task_dir: str | Path) -> list[Path]:
    """返回目录下可作为 OSWorld task 的 JSON 文件列表。

    只保留包含 `instruction` 的 JSON，避免把 README 元数据、全局配置或
    非任务 JSON 混入 batch eval。
    """

    root = Path(task_dir)
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"OSWorld task config path not found: {root}")
    paths = []
    for candidate in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "instruction" in payload:
            paths.append(candidate)
    if not paths:
        raise FileNotFoundError(f"No OSWorld task JSON files found under: {root}")
    return paths
