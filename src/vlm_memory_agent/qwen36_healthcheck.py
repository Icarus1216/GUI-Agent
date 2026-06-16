"""功能: 校验 Qwen3.6 OpenAI-compatible endpoint 的 `/models` 和 chat JSON action 协议。
上游依赖: 依赖 urllib 请求 endpoint，并复用 llm.base 的 VLM JSON 解析器验证返回动作。
下游依赖: readiness 检查、OSWorld 运行脚本、worker endpoint 检查脚本和测试使用该协议门禁。
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import urllib.error
import urllib.request

from vlm_memory_agent.llm.base import parse_vlm_response_text


def check_endpoint(
    base_url: str,
    model: str,
    api_key: str | None = None,
    timeout: int = 30,
    extra_headers: dict[str, str] | None = None,
) -> tuple[bool, str]:
    """完整校验 Qwen endpoint。

    先检查 `/models`，确保服务和模型名可见；再发一个最小
    `/chat/completions` 请求，要求模型返回 `wait` JSON action。第二步很
    关键：只通 `/models` 不能证明 agent 需要的动作协议可用。
    """

    base_url = base_url.rstrip("/")
    headers = _headers(api_key, extra_headers)
    context = _ssl_context_from_env()
    models_ok, models_detail = _check_models(base_url, model, headers, timeout, context)
    if not models_ok:
        return False, models_detail
    chat_ok, chat_detail = _check_chat(base_url, model, headers, timeout, context)
    if not chat_ok:
        return False, chat_detail
    return True, f"{models_detail}; {chat_detail}"


def _check_models(
    base_url: str,
    model: str,
    headers: dict[str, str],
    timeout: int,
    context: ssl.SSLContext | None = None,
) -> tuple[bool, str]:
    """检查 OpenAI-compatible `/models` 并确认目标模型名。"""

    req = urllib.request.Request(f"{base_url}/models", headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return False, f"/models failed: {type(exc).__name__}: {exc}"
    model_ids = [str(item.get("id")) for item in payload.get("data", []) if isinstance(item, dict) and item.get("id")]
    if model_ids and model not in model_ids:
        return False, f"/models reachable but `{model}` not served; models={model_ids}"
    return True, f"/models ok; models={model_ids or 'unknown'}"


def _check_chat(
    base_url: str,
    model: str,
    headers: dict[str, str],
    timeout: int,
    context: ssl.SSLContext | None = None,
) -> tuple[bool, str]:
    """检查 chat completion 是否能按 JSON action contract 返回。"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only valid JSON for the next GUI action."},
            {
                "role": "user",
                "content": (
                    'Healthcheck. Return exactly {"thought":"healthcheck","action_type":"wait"} '
                    "and no markdown."
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": 64,
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "\n".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
        response = parse_vlm_response_text(str(content))
    except (KeyError, IndexError, TypeError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        return False, f"/chat/completions failed protocol check: {type(exc).__name__}: {exc}"
    if response.action_type != "wait":
        return False, f"/chat/completions returned unexpected action_type={response.action_type!r}"
    return True, "/chat/completions ok; JSON action protocol ok"


def _headers(api_key: str | None, extra_headers: dict[str, str] | None = None) -> dict[str, str]:
    """合并环境 header 和可选 Bearer token。"""

    headers = _extra_headers_from_env() if extra_headers is None else dict(extra_headers)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _extra_headers_from_env() -> dict[str, str]:
    """读取 Qwen/OpenAI endpoint 的额外请求头配置。"""

    headers: dict[str, str] = {}
    raw = os.environ.get("OPENAI_EXTRA_HEADERS") or os.environ.get("QWEN_EXTRA_HEADERS")
    if raw:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("OPENAI_EXTRA_HEADERS/QWEN_EXTRA_HEADERS must be a JSON object")
        headers.update({str(key): str(value) for key, value in parsed.items()})
    destination_service = os.environ.get("DESTINATION_SERVICE") or os.environ.get("QWEN_DESTINATION_SERVICE")
    if destination_service and "Destination-Service" not in headers:
        headers["Destination-Service"] = destination_service
    return headers


def _ssl_context_from_env() -> ssl.SSLContext | None:
    """按环境变量决定是否跳过 TLS 证书校验。"""

    verify = os.environ.get("OPENAI_VERIFY_SSL") or os.environ.get("QWEN_VERIFY_SSL")
    if verify and verify.lower() in {"0", "false", "no"}:
        return ssl._create_unverified_context()
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a Qwen3.6 OpenAI-compatible endpoint before OSWorld runs.")
    parser.add_argument("--base-url", default=os.environ.get("QWEN_BASE_URL") or os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "qwen3.6-35b-a3b"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args(argv)
    if not args.base_url:
        print("MISSING base_url: set QWEN_BASE_URL or pass --base-url")
        return 2
    ok, detail = check_endpoint(args.base_url, args.model, args.api_key, timeout=args.timeout)
    print(("OK      " if ok else "MISSING ") + f"qwen36_endpoint: {detail}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
