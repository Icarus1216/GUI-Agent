"""功能: 汇总 OSWorld、Qwen3.6 本地/endpoint 与运行参数的就绪度检查。
上游依赖: 依赖 preflight 检查、Qwen endpoint healthcheck、模型路径和环境变量配置。
下游依赖: scripts/check_osworld_ready.sh、README readiness gate 和测试读取 JSON/文本检查结果。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from vlm_memory_agent.llm.qwen36_local import DEFAULT_QWEN36_MODEL_PATH
from vlm_memory_agent.preflight import CheckResult, check_osworld, check_osworld_runtime, check_qwen36
from vlm_memory_agent.qwen36_healthcheck import check_endpoint


def collect_readiness(
    task_config: str | None,
    task_id: str | None = None,
    provider: str = "vmware",
    vm_path: str | None = None,
    model_path: str | Path = DEFAULT_QWEN36_MODEL_PATH,
    base_url: str | None = None,
    model: str = "qwen3.6-35b-a3b",
    api_key: str | None = None,
    endpoint_timeout: int = 30,
    skip_endpoint_healthcheck: bool = False,
    serving_mode: str = "auto",
) -> dict:
    """收集一次 OSWorld+Qwen 运行前的所有 readiness 信号。

    `serving_mode=auto` 时，如果配置了 base_url 就按 endpoint 模式检查，
    否则按本地 Qwen 模式检查 CUDA/权重。返回值同时给人类文本输出和 JSON
    自动化脚本使用。
    """

    checks: list[CheckResult] = []
    mode = _resolve_serving_mode(serving_mode, base_url)
    checks.extend(check_osworld())
    checks.extend(check_osworld_runtime(task_config=task_config, task_id=task_id, vm_path=vm_path, provider=provider))

    if mode == "local":
        checks.extend(check_qwen36(model_path=model_path, require_cuda=True))
    else:
        checks.append(
            CheckResult(
                "qwen36_local_model_files",
                Path(model_path).exists(),
                f"{model_path}; optional on OSWorld runner when using endpoint serving",
                required=False,
            )
        )

    if mode == "endpoint" and not base_url:
        checks.append(CheckResult("qwen36_endpoint_protocol", False, "serving_mode=endpoint requires --base-url/QWEN_BASE_URL"))
    elif mode == "endpoint" and base_url and not skip_endpoint_healthcheck:
        ok, detail = check_endpoint(base_url, model, api_key=api_key, timeout=endpoint_timeout)
        checks.append(CheckResult("qwen36_endpoint_protocol", ok, detail))
    elif mode == "endpoint" and base_url:
        checks.append(CheckResult("qwen36_endpoint_protocol", True, "skipped by request", required=False))
    else:
        checks.append(
            CheckResult(
                "qwen36_endpoint_protocol",
                False,
                "not configured; local Qwen3.6 inference requires CUDA on this host",
                required=False,
            )
        )

    blockers = [check for check in checks if check.required and not check.ok]
    return {
        "ready": not blockers,
        "provider": provider,
        "task_config": task_config,
        "task_id": task_id,
        "model_path": str(model_path),
        "base_url": base_url,
        "serving_mode": mode,
        "checks": [asdict(check) for check in checks],
        "blockers": [asdict(check) for check in blockers],
    }


def _resolve_serving_mode(serving_mode: str, base_url: str | None) -> str:
    """把 auto/local/endpoint 解析成明确模式。"""

    mode = serving_mode.strip().lower()
    if mode == "auto":
        return "endpoint" if base_url else "local"
    if mode not in {"local", "endpoint"}:
        raise ValueError("serving_mode must be one of: auto, local, endpoint")
    return mode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report whether this environment is ready for a real OSWorld+Qwen3.6 run.")
    parser.add_argument("--task-config", default=os.environ.get("OSWORLD_TASK_CONFIG"))
    parser.add_argument("--task-id", default=os.environ.get("TASK_ID"))
    parser.add_argument("--provider", default=os.environ.get("OSWORLD_PROVIDER", "vmware"))
    parser.add_argument("--vm-path", default=os.environ.get("OSWORLD_VM_PATH"))
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", DEFAULT_QWEN36_MODEL_PATH))
    parser.add_argument("--base-url", default=os.environ.get("QWEN_BASE_URL") or os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "qwen3.6-35b-a3b"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--endpoint-timeout", type=int, default=int(os.environ.get("QWEN_HEALTHCHECK_TIMEOUT", "30")))
    parser.add_argument("--skip-endpoint-healthcheck", action="store_true")
    parser.add_argument("--serving-mode", choices=["auto", "local", "endpoint"], default=os.environ.get("QWEN_SERVING_MODE", "auto"))
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    payload = collect_readiness(
        task_config=args.task_config,
        task_id=args.task_id,
        provider=args.provider,
        vm_path=args.vm_path,
        model_path=args.model_path,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        endpoint_timeout=args.endpoint_timeout,
        skip_endpoint_healthcheck=args.skip_endpoint_healthcheck,
        serving_mode=args.serving_mode,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"ready={str(payload['ready']).lower()} provider={payload['provider']} task_config={payload['task_config']}")
        for item in payload["checks"]:
            status = "OK" if item["ok"] else "MISSING" if item["required"] else "OPTIONAL"
            print(f"{status:8} {item['name']}: {item['detail']}")
    return 0 if payload["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
