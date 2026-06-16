"""功能: 提供简单 HTTP JSON API，用 rule backend 在 mock 环境中运行 episode 并查看 memory。
上游依赖: 依赖 VLMGuiAgent、MockOSWorldEnv、RuleBasedVLMClient 和 HierarchicalMemoryStore。
下游依赖: 外部集成测试或调试服务可调用 /health、/memory、/run_episode。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from vlm_memory_agent.agent import AgentConfig, VLMGuiAgent
from vlm_memory_agent.envs.mock_osworld import MockOSWorldEnv
from vlm_memory_agent.llm.rule_based import RuleBasedVLMClient
from vlm_memory_agent.memory.store import HierarchicalMemoryStore


class AgentAPI:
    """极简 HTTP API 背后的业务对象。

    当前 API 固定使用 rule backend + MockOSWorldEnv，目的是提供可预测的
    集成测试入口，而不是承载真实 OSWorld 或 Qwen 推理服务。
    """

    def __init__(self, memory_path: str = "runs/api_memory.json"):
        self.memory = HierarchicalMemoryStore(memory_path)

    def run_episode(self, task_id: str = "search_report") -> dict:
        """运行一个 mock episode，并返回 trajectory dict。"""

        agent = VLMGuiAgent(
            vlm=RuleBasedVLMClient(),
            memory=self.memory,
            config=AgentConfig(max_steps=10),
        )
        trajectory = agent.run_episode(MockOSWorldEnv(), task_id=task_id)
        return trajectory.to_dict()

    def memory_snapshot(self) -> dict:
        """导出当前 memory 图，供调试查看。"""

        return {
            "nodes": [asdict(node) for node in self.memory.nodes.values()],
            "edges": {src: sorted(dst) for src, dst in self.memory.edges.items()},
        }


def make_handler(api: AgentAPI):
    """闭包生成 BaseHTTPRequestHandler，让 handler 能访问同一个 AgentAPI。

    使用标准库 HTTP server 是为了让这个调试 API 零额外依赖；真实服务化
    集成可以在不改 AgentAPI 的情况下替换成 FastAPI/Flask。
    """

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict) -> None:
            """统一 JSON 响应写法。"""

            data = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/health":
                self._send(200, {"ok": True})
            elif path == "/memory":
                self._send(200, api.memory_snapshot())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if path == "/run_episode":
                self._send(200, api.run_episode(task_id=body.get("task_id", "search_report")))
            else:
                self._send(404, {"error": "not found"})

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8765, memory_path: str = "runs/api_memory.json") -> None:
    """启动调试 API 服务。"""

    Path(memory_path).parent.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((host, port), make_handler(AgentAPI(memory_path)))
    print(f"Serving VLM memory agent API on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    serve()
