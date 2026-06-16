#!/usr/bin/env bash
# 功能: 在 worker 后台启动 GPU keepalive 计算循环，默认保活 GPU1-GPU7，避开 GPU0 上的 Qwen vLLM 服务。
# 上游依赖: 依赖 gpu_keepalive_worker.py、worker 上可用的 torch/CUDA runtime 和可选 GPU_KEEPALIVE_* 环境变量。
# 下游依赖: tmux 后台会话直接调用该脚本，用于维持 GPU 活跃度和写入 /tmp/gpu_keepalive.log。

set -euo pipefail

ROOT="${ROOT:-/mnt/hdfs/byte_ai_sales/user/zhangjuntian/vlm-memory-agent}"
QWEN_DEPS_DIR="${QWEN_DEPS_DIR:-/tmp/qwen36_vllm019_deps}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python}"
LOG_PATH="${LOG_PATH:-/tmp/gpu_keepalive.log}"

export PYTHONPATH="${QWEN_DEPS_DIR}:${ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6,7}"
export GPU_KEEPALIVE_SIZE="${GPU_KEEPALIVE_SIZE:-8192}"
export GPU_KEEPALIVE_SLEEP="${GPU_KEEPALIVE_SLEEP:-0}"
export GPU_KEEPALIVE_DTYPE="${GPU_KEEPALIVE_DTYPE:-bfloat16}"

exec "${PYTHON_BIN}" "${ROOT}/scripts/gpu_keepalive_worker.py" 2>&1 | tee "${LOG_PATH}"
