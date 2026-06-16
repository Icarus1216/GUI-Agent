"""功能: 实现 OpenAI-compatible chat/completions VLM 客户端，支持文本和可选截图输入。
上游依赖: 依赖 llm.base 解析协议、OPENAI_* 环境变量、urllib 网络请求和本地截图文件。
下游依赖: CLI、OSWorld 脚本、BenchmarkRunner 和测试用它连接 GPT/Qwen/vLLM/SGLang endpoint。
"""

from __future__ import annotations

import base64
import json
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path

from vlm_memory_agent.llm.base import VLMClient, VLMResponse, parse_vlm_response_text


class OpenAICompatibleVLMClient(VLMClient):
    """Minimal OpenAI-compatible chat completion client.

    The model must return JSON:
    {"thought": "...", "action_type": "click|type|hotkey|wait|done|fail", "target": "...", "text": "..."}

    这个客户端刻意只实现 chat/completions 所需的最小协议，避免引入 OpenAI
    SDK 版本差异。它可以连接官方 OpenAI API，也可以连接 vLLM/SGLang 暴露的
    OpenAI-compatible endpoint。
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.extra_headers = _extra_headers_from_env()
        self.ssl_context = _ssl_context_from_env()
        self.timeout = timeout

    def decide(self, prompt: str, image_path: str | None = None) -> VLMResponse:
        """发送一次 chat completion 请求并解析 GUI action。

        如果提供 screenshot_path，就把图片转成 data URL 放进 user content；
        如果底层模型不是多模态，也可以忽略该字段，仅基于 prompt 中的
        screen text/a11y tree 决策。
        """

        content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        if image_path:
            encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                }
            )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a GUI VLM agent. Return only valid JSON for the next action.",
                },
                {"role": "user", "content": content},
            ],
            "temperature": 0,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers("application/json"),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=self.ssl_context) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"VLM API request failed: {exc}") from exc

        text = self._message_text(data["choices"][0]["message"]["content"])
        return parse_vlm_response_text(text)

    def _message_text(self, content: object) -> str:
        """兼容 OpenAI/vLLM 返回的 string 或 content part list。"""

        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    value = item.get("text")
                    if value is not None:
                        parts.append(str(value))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(content)

    def _headers(self, content_type: str | None = None) -> dict[str, str]:
        """组装请求头。

        `extra_headers` 用于公司网关、workspace-proxy 或 Destination-Service
        等非 OpenAI 标准鉴权路径；api_key 则按 Bearer token 处理。
        """

        headers = dict(self.extra_headers)
        if content_type:
            headers["Content-Type"] = content_type
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


def _extra_headers_from_env() -> dict[str, str]:
    """从环境变量读取额外 HTTP headers。

    支持 JSON 格式的 `OPENAI_EXTRA_HEADERS/QWEN_EXTRA_HEADERS`，也支持历史上
    用于内部服务路由的 `DESTINATION_SERVICE/QWEN_DESTINATION_SERVICE`。
    """

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
    """根据环境变量关闭证书校验。

    workspace-proxy 或临时内网 endpoint 调试时可能需要 `QWEN_VERIFY_SSL=0`；
    默认仍使用系统证书校验。
    """

    verify = os.environ.get("OPENAI_VERIFY_SSL") or os.environ.get("QWEN_VERIFY_SSL")
    if verify and verify.lower() in {"0", "false", "no"}:
        return ssl._create_unverified_context()
    return None
