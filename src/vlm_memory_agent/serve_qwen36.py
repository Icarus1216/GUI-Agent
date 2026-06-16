"""功能: 封装 vLLM OpenAI API server 命令，用本地 Qwen3.6 权重启动服务进程。
上游依赖: 依赖 vLLM 可导入、Qwen3.6 模型路径和 runtime_paths 生成的 PYTHONPATH。
下游依赖: launch/start worker 脚本和测试通过本模块构造 serving 子进程命令。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from vlm_memory_agent.llm.qwen36_compat import apply_qwen36_transformers_compat
from vlm_memory_agent.llm.qwen36_local import DEFAULT_QWEN36_MODEL_PATH
from vlm_memory_agent.runtime_paths import build_pythonpath_with_runtime_deps


def main(argv: list[str] | None = None) -> int:
    """启动 vLLM OpenAI API server 的包装入口。

    这里不用直接 `python -m vllm.entrypoints.openai.api_server`，而是先在
    子进程 preload 中调用 Qwen3.6 兼容补丁，再 runpy 进入 vLLM。这样可以
    在不修改 vLLM 源码的情况下注册本地 checkpoint 需要的 config/model
    alias。
    """

    parser = argparse.ArgumentParser(description="Serve Qwen3.6 through vLLM's OpenAI-compatible API.")
    parser.add_argument("--model-path", default=DEFAULT_QWEN36_MODEL_PATH)
    parser.add_argument("--served-model-name", default="qwen3.6-35b-a3b")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=int(os.environ.get("TENSOR_PARALLEL_SIZE", "1")))
    parser.add_argument("--gpu-memory-utilization", default=os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"))
    parser.add_argument("--max-model-len", default=os.environ.get("MAX_MODEL_LEN"))
    parser.add_argument("--extra-arg", action="append", default=[])
    args = parser.parse_args(argv)

    api_args = [
        "--model",
        args.model_path,
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--trust-remote-code",
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    # vLLM 参数保持显式列表，便于 worker 启动日志中完整打印和复现。
    if args.max_model_len:
        api_args.extend(["--max-model-len", str(args.max_model_len)])
    api_args.extend(args.extra_arg)

    preload = (
        "import runpy, sys; "
        "from vlm_memory_agent.llm.qwen36_compat import apply_qwen36_transformers_compat; "
        "apply_qwen36_transformers_compat(); "
        "sys.argv = ['vllm.entrypoints.openai.api_server'] + sys.argv[1:]; "
        "runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')"
    )
    cmd = [sys.executable, "-c", preload, *api_args]
    print(" ".join(cmd), flush=True)
    env = os.environ.copy()
    runtime_pythonpath = build_pythonpath_with_runtime_deps()
    if runtime_pythonpath:
        # 子进程必须看到 /tmp target 依赖，否则 vLLM/transformers 补丁导入会
        # 回落到系统旧版本。
        env["PYTHONPATH"] = runtime_pythonpath
    apply_qwen36_transformers_compat()
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
