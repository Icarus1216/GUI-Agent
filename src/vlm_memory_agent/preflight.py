"""功能: 检查 OSWorld、Qwen3.6、本地 CUDA 和 OpenAI-compatible endpoint 的基础运行依赖。
上游依赖: 依赖 runtime_paths、OSWorld task config 工具、Qwen3.6 compat 和标准库导入/网络探测。
下游依赖: OSWorld 运行脚本、readiness 聚合器、README 示例和测试用它提前暴露缺失依赖。
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from vlm_memory_agent.llm.qwen36_compat import apply_qwen36_transformers_compat
from vlm_memory_agent.llm.qwen36_local import DEFAULT_QWEN36_MODEL_PATH
from vlm_memory_agent.runtime_paths import apply_runtime_dependency_paths
from vlm_memory_agent.envs.osworld_runner import iter_task_config_paths, load_task_config


@dataclass(slots=True)
class CheckResult:
    """单项 readiness 检查结果。

    `required=False` 表示缺失不会让总检查失败，例如 host pyperclip 或可选
    serving backend。脚本根据 required 字段决定最终退出码。
    """

    name: str
    ok: bool
    detail: str
    required: bool = True


def module_available(name: str) -> bool:
    """只检查模块 spec 是否存在，不执行 import 副作用。"""

    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def module_importable(name: str) -> tuple[bool, str]:
    """实际 import 模块，并返回版本或异常详情。"""

    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "")
        return True, str(version or "import ok")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def check_osworld() -> list[CheckResult]:
    """检查 OSWorld Python 依赖是否可见。

    pyautogui 在 host 上 import 失败不一定代表 OSWorld 不能跑，因为实际
    GUI action 常在 VM/controller 内执行；这里把信息写进 detail，避免误判。
    """

    apply_runtime_dependency_paths()
    desktop_ok, desktop_detail = module_importable("desktop_env.desktop_env")
    pyautogui_installed = module_available("pyautogui")
    display_hint = "not found"
    if pyautogui_installed:
        display_hint = "installed; DISPLAY is not required unless pyautogui is imported/executed on this host."
        pyautogui_import_ok, pyautogui_detail = module_importable("pyautogui")
        if pyautogui_import_ok:
            display_hint = pyautogui_detail
        else:
            display_hint = f"{pyautogui_detail}; OSWorld usually executes pyautogui inside the VM."
    return [
        CheckResult(
            "desktop_env.desktop_env",
            desktop_ok,
            desktop_detail,
        ),
        CheckResult("pyautogui", pyautogui_installed, display_hint),
        CheckResult(
            "pyperclip",
            module_available("pyperclip"),
            "Optional on host; paste command imports pyperclip inside the VM and falls back to typewrite.",
            required=False,
        ),
    ]


def check_osworld_runtime(
    task_config: str | None = None,
    task_id: str | None = None,
    vm_path: str | None = None,
    provider: str | None = None,
) -> list[CheckResult]:
    """检查 task config、VM 路径和 provider 运行时依赖。

    这一步不启动 VM，只做低成本静态/连接前检查，用于在真实 OSWorld reset
    之前暴露最常见的配置错误。
    """

    results: list[CheckResult] = []
    if task_config:
        try:
            if Path(task_config).is_dir() and not task_id:
                task_paths = iter_task_config_paths(task_config)
                results.append(CheckResult("osworld_task_config", bool(task_paths), f"{len(task_paths)} task JSON file(s) found"))
            else:
                config = load_task_config(task_config, task_id=task_id)
                required = {"id", "instruction", "evaluator"}
                missing = sorted(required - set(config))
                results.append(
                    CheckResult(
                        "osworld_task_config",
                        not missing,
                        f"loaded task id={config.get('id', task_id)}" if not missing else f"missing keys: {missing}",
                    )
                )
        except Exception as exc:
            results.append(CheckResult("osworld_task_config", False, f"{type(exc).__name__}: {exc}"))

    provider = provider or "vmware"
    provider = provider.lower().strip()
    local_vm_provider = provider in {"vmware", "virtualbox"}

    if vm_path:
        path = Path(vm_path)
        must_exist = local_vm_provider or provider == "docker"
        results.append(CheckResult("osworld_vm_path", path.exists() if must_exist else True, str(path), required=must_exist))
    else:
        results.append(
            CheckResult(
                "osworld_vm_path",
                not local_vm_provider,
                (
                    "not provided; required for local vmware/virtualbox runs"
                    if local_vm_provider
                    else "not provided; desktop-env/provider will resolve or download the backing VM/image"
                ),
                required=local_vm_provider,
            )
        )

    if provider == "vmware":
        results.append(CheckResult("vmware_vmrun", shutil.which("vmrun") is not None, "vmrun binary for VMware provider"))
    elif provider == "virtualbox":
        results.append(
            CheckResult("virtualbox_vboxmanage", shutil.which("VBoxManage") is not None, "VBoxManage binary for VirtualBox provider")
        )
    elif provider == "docker":
        docker_importable, docker_detail = module_importable("docker")
        results.append(CheckResult("docker_python", docker_importable, docker_detail))
        results.append(CheckResult("docker_cli", shutil.which("docker") is not None, "docker binary for Docker provider"))
        results.append(_check_docker_daemon() if docker_importable else CheckResult("docker_daemon", False, docker_detail))
    return results


def _check_docker_daemon() -> CheckResult:
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return CheckResult("docker_daemon", True, "Docker daemon reachable through Docker SDK.")
    except Exception as exc:
        return CheckResult("docker_daemon", False, f"{type(exc).__name__}: {exc}")


def check_qwen36(model_path: str | Path = DEFAULT_QWEN36_MODEL_PATH, require_cuda: bool = False) -> list[CheckResult]:
    """检查本地 Qwen3.6 权重和本地推理依赖。

    如果使用远程 endpoint，`require_cuda=False` 时 CUDA 只作为可选参考；
    如果要本地 Transformers 直载 35B 权重，应开启 require_cuda。
    """

    apply_runtime_dependency_paths()
    path = Path(model_path)
    torch_ok, torch_detail = module_importable("torch")
    results = [
        CheckResult("torch", torch_ok, torch_detail),
        CheckResult("transformers", *module_importable("transformers")),
        CheckResult("model_path", path.exists(), str(path)),
        CheckResult("model_config", (path / "config.json").exists(), str(path / "config.json")),
        CheckResult(
            "model_weights",
            any(path.glob("*.safetensors")) or any(path.glob("*.bin")),
            "At least one model weight shard exists.",
        ),
    ]
    if (path / "config.json").exists():
        try:
            config = json.loads((path / "config.json").read_text(encoding="utf-8"))
            results.append(CheckResult("model_type", True, str(config.get("model_type", "unknown"))))
        except json.JSONDecodeError:
            results.append(CheckResult("model_type", False, "config.json is not valid JSON."))
    if module_available("transformers"):
        report = apply_qwen36_transformers_compat()
        results.append(
            CheckResult(
                "qwen36_transformers_compat",
                report.config_aliases,
                report.detail or "Qwen3.6 qwen3_5/qwen3_5_moe aliases registered.",
            )
        )
    if torch_ok:
        try:
            import torch

            cuda_count = torch.cuda.device_count()
            cuda_detail = (
                f"{cuda_count} CUDA device(s) visible"
                if torch.cuda.is_available()
                else "No CUDA device visible; use a GPU worker or OpenAI-compatible serving endpoint for Qwen3.6-35B."
            )
            results.append(CheckResult("cuda_for_qwen36_local", torch.cuda.is_available(), cuda_detail, required=require_cuda))
        except Exception as exc:
            results.append(CheckResult("cuda_for_qwen36_local", False, f"{type(exc).__name__}: {exc}", required=require_cuda))
    results.append(CheckResult("vllm", module_available("vllm"), "Optional OpenAI-compatible serving backend.", required=False))
    results.append(
        CheckResult("sglang", module_available("sglang"), "Optional OpenAI-compatible serving backend.", required=False)
    )
    return results


def check_openai_compatible_server(
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    timeout: int = 5,
) -> list[CheckResult]:
    """检查 OpenAI-compatible `/models` 是否可访问且包含目标模型。"""

    if not base_url:
        return [
            CheckResult(
                "openai_compatible_server",
                False,
                "not configured; set QWEN_BASE_URL/--base-url when serving Qwen3.6 outside this process",
                required=False,
            )
        ]

    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return [CheckResult("openai_compatible_server", False, f"HTTP {exc.code} from {url}")]
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return [CheckResult("openai_compatible_server", False, f"{type(exc).__name__}: {exc}")]

    model_ids = []
    for item in payload.get("data", []) if isinstance(payload, dict) else []:
        if isinstance(item, dict) and item.get("id"):
            model_ids.append(str(item["id"]))
    if model and model_ids and model not in model_ids:
        return [
            CheckResult(
                "openai_compatible_server",
                False,
                f"reachable at {url}, but model `{model}` not in served models: {model_ids}",
            )
        ]
    detail = f"reachable at {url}"
    if model_ids:
        detail += f"; models={model_ids}"
    return [CheckResult("openai_compatible_server", True, detail)]


def run_checks(checks: list[str], model_path: str | Path, require_cuda: bool = False) -> list[CheckResult]:
    """根据用户选择执行一组基础检查。"""

    results: list[CheckResult] = []
    if "osworld" in checks:
        results.extend(check_osworld())
    if "qwen36" in checks:
        results.extend(check_qwen36(model_path, require_cuda=require_cuda))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check optional runtime dependencies.")
    parser.add_argument("--check", choices=["osworld", "qwen36"], action="append", default=[])
    parser.add_argument("--model-path", default=DEFAULT_QWEN36_MODEL_PATH)
    parser.add_argument("--vm-path", default=os.environ.get("OSWORLD_VM_PATH"))
    parser.add_argument("--task-config", default=os.environ.get("OSWORLD_TASK_CONFIG"))
    parser.add_argument("--task-id", default=os.environ.get("TASK_ID"))
    parser.add_argument("--provider", default=os.environ.get("OSWORLD_PROVIDER", "vmware"))
    parser.add_argument("--base-url", default=os.environ.get("QWEN_BASE_URL") or os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "qwen3.6-35b-a3b"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--require-cuda", action="store_true", help="Fail if no CUDA device is visible for local Qwen3.6 inference.")
    args = parser.parse_args(argv)
    checks = args.check or ["osworld", "qwen36"]
    results = run_checks(checks, args.model_path, require_cuda=args.require_cuda)
    if "osworld" in checks:
        results.extend(
            check_osworld_runtime(
                task_config=args.task_config,
                task_id=args.task_id,
                vm_path=args.vm_path,
                provider=args.provider,
            )
        )
    if args.base_url:
        results.extend(check_openai_compatible_server(args.base_url, args.model, args.api_key))
    for result in results:
        status = "OK" if result.ok else "MISSING" if result.required else "OPTIONAL"
        print(f"{status:7} {result.name}: {result.detail}")
    return 0 if all(result.ok for result in results if result.required) else 1


if __name__ == "__main__":
    raise SystemExit(main())
