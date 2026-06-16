#!/usr/bin/env bash
set -euo pipefail

# 功能: 在 worker 本机检查默认 127.0.0.1:QWEN_PORT 的 Qwen3.6 endpoint 健康状态。
# 上游依赖: 依赖 worker 上的项目路径、Qwen 依赖目录、QWEN_PORT/MODEL_NAME 和 qwen36_healthcheck 模块。
# 下游依赖: worker 端 endpoint 启动后用它确认本机服务已经可用。

ROOT="${ROOT:-/mnt/hdfs/byte_ai_sales/user/zhangjuntian/vlm-memory-agent}"
QWEN_PORT="${QWEN_PORT:-9001}"
MODEL_NAME="${MODEL_NAME:-qwen3.6-35b-a3b}"
QWEN_DEPS_DIR="${QWEN_DEPS_DIR:-/tmp/vlm_memory_agent_qwen_deps}"
TIMEOUT="${TIMEOUT:-120}"

cd "${ROOT}"
export VLM_MEMORY_AGENT_QWEN_DEPS="${QWEN_DEPS_DIR}"
export PATH="${QWEN_DEPS_DIR}/bin:${PATH}"
export PYTHONPATH="${QWEN_DEPS_DIR}:${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m vlm_memory_agent.qwen36_healthcheck \
  --base-url "http://127.0.0.1:${QWEN_PORT}/v1" \
  --model "${MODEL_NAME}" \
  --timeout "${TIMEOUT}"
