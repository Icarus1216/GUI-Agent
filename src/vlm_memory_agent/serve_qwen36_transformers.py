"""功能: 用 Transformers 直载 Qwen3.6 并暴露最小 OpenAI-compatible HTTP API。
上游依赖: 依赖 qwen36_local、FastAPI/Uvicorn、Pillow/Transformers/Torch 和本地模型权重。
下游依赖: vLLM 不支持该 checkpoint 时，devbox/OSWorld agent 可通过 /v1/chat/completions 调用本服务。
"""

from __future__ import annotations

import argparse
import base64
import json
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from vlm_memory_agent.llm.qwen36_local import DEFAULT_QWEN36_MODEL_PATH, Qwen36LocalVLMClient
from vlm_memory_agent.runtime_paths import apply_runtime_dependency_paths


def main(argv: list[str] | None = None) -> int:
    """启动 Transformers 版最小 OpenAI-compatible 服务。

    这是 vLLM 不可用时的 fallback：协议足够让 agent 调用 `/v1/models` 和
    `/v1/chat/completions`，但吞吐和并发能力不如 vLLM。
    """

    parser = argparse.ArgumentParser(description="Serve Qwen3.6 with Transformers through a small OpenAI-compatible API.")
    parser.add_argument("--model-path", default=DEFAULT_QWEN36_MODEL_PATH)
    parser.add_argument("--served-model-name", default="qwen3.6-35b-a3b")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args(argv)

    apply_runtime_dependency_paths()
    try:
        import uvicorn
        from fastapi import FastAPI, HTTPException
    except ModuleNotFoundError as exc:
        raise RuntimeError("Transformers endpoint requires fastapi and uvicorn in the runtime dependency path.") from exc

    app = FastAPI()
    client = Qwen36LocalVLMClient(
        model_path=args.model_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
    )

    @app.get("/v1/models")
    @app.get("/models")
    def models() -> dict[str, Any]:
        """返回 OpenAI-compatible model list。"""

        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": args.served_model_name,
                    "object": "model",
                    "created": now,
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    def chat_completions(payload: dict[str, Any]) -> dict[str, Any]:
        """处理 chat completion，并把 VLMResponse 包回 OpenAI 响应格式。"""

        prompt, image_path = _prompt_and_image(payload.get("messages", []))
        if not prompt:
            raise HTTPException(status_code=400, detail="No text content found in messages.")
        try:
            response = client.decide(prompt, image_path=image_path)
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Qwen3.6 inference failed: {type(exc).__name__}: {exc}") from exc
        content = json.dumps(
            {
                "thought": response.thought,
                "action_type": response.action_type,
                "target": response.target,
                "text": response.text,
                **response.metadata,
            },
            ensure_ascii=True,
            separators=(",", ":"),
        )
        return {
            "id": f"chatcmpl-local-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model") or args.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _prompt_and_image(messages: Any) -> tuple[str, str | None]:
    """从 OpenAI messages 中抽取文本 prompt 和首张 data-URL 图片。"""

    texts: list[str] = []
    image_path: str | None = None
    if not isinstance(messages, list):
        return "", None
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and item.get("text") is not None:
                    texts.append(str(item["text"]))
                elif item.get("type") == "image_url" and image_path is None:
                    image_path = _write_image_url(item.get("image_url"))
    return "\n\n".join(texts), image_path


def _write_image_url(image_url: Any) -> str | None:
    """把 base64 data URL 图片写入临时 PNG，供本地 Qwen processor 读取。"""

    url = image_url.get("url") if isinstance(image_url, dict) else image_url
    if not isinstance(url, str):
        return None
    marker = "base64,"
    if marker not in url:
        return None
    encoded = url.split(marker, 1)[1]
    data = base64.b64decode(encoded)
    tmp = tempfile.NamedTemporaryFile(prefix="qwen36-openai-image-", suffix=".png", delete=False)
    try:
        tmp.write(data)
        return str(Path(tmp.name))
    finally:
        tmp.close()


if __name__ == "__main__":
    raise SystemExit(main())
