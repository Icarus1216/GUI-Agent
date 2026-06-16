"""功能: 根据环境变量把隔离安装的 Qwen/OSWorld 依赖目录注入 sys.path 或 PYTHONPATH。
上游依赖: 依赖 VLM_MEMORY_AGENT_QWEN_DEPS、VLM_MEMORY_AGENT_OSWORLD_DEPS 和现有 PYTHONPATH。
下游依赖: Qwen 本地加载、preflight、OSWorld runner、vLLM serving wrapper 和 worker 脚本共享该路径逻辑。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def add_paths_from_env(env_name: str, prepend: bool = False) -> list[str]:
    """把某个环境变量中的路径列表加入 sys.path。

    `prepend=True` 用于 Qwen/Transformers 依赖，确保 /tmp target 里的新版本
    优先于系统包；OSWorld 依赖一般 append，避免覆盖模型侧关键依赖。
    """

    raw = os.environ.get(env_name, "")
    added: list[str] = []
    for item in raw.split(os.pathsep):
        if not item:
            continue
        path = str(Path(item).expanduser())
        if path in sys.path or not Path(path).exists():
            continue
        if prepend:
            sys.path.insert(0, path)
        else:
            sys.path.append(path)
        added.append(path)
    return added


def apply_runtime_dependency_paths() -> None:
    """按项目约定注入 Qwen 和 OSWorld 的隔离依赖目录。"""

    add_paths_from_env("VLM_MEMORY_AGENT_QWEN_DEPS", prepend=True)
    add_paths_from_env("VLM_MEMORY_AGENT_OSWORLD_DEPS", prepend=False)


def build_pythonpath_with_runtime_deps(include_osworld: bool = False) -> str:
    """构造传给子进程的 PYTHONPATH。

    vLLM/uvicorn 等服务通常在子进程中启动，单纯修改当前进程 sys.path 不够；
    需要把相同依赖目录拼进环境变量传下去。
    """

    paths: list[str] = []
    for env_name in ("VLM_MEMORY_AGENT_QWEN_DEPS", "VLM_MEMORY_AGENT_OSWORLD_DEPS" if include_osworld else ""):
        if not env_name:
            continue
        raw = os.environ.get(env_name, "")
        for item in raw.split(os.pathsep):
            if item and Path(item).expanduser().exists():
                paths.append(str(Path(item).expanduser()))
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        paths.append(existing)
    return os.pathsep.join(dict.fromkeys(paths))
