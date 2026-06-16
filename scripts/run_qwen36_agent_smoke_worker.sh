#!/usr/bin/env bash
set -euo pipefail

# 功能: 在 worker 上用本地 OpenAI-compatible Qwen endpoint 跑 mock_osworld agent smoke。
# 上游依赖: 依赖已启动的 127.0.0.1:QWEN_PORT/v1 endpoint、Qwen 依赖目录和主 CLI。
# 下游依赖: endpoint serving 路线部署后用它验证完整 agent loop 是否能消费该服务。

ROOT="${ROOT:-/mnt/hdfs/byte_ai_sales/user/zhangjuntian/vlm-memory-agent}"
QWEN_PORT="${QWEN_PORT:-9001}"
MODEL_NAME="${MODEL_NAME:-qwen3.6-35b-a3b}"
QWEN_DEPS_DIR="${QWEN_DEPS_DIR:-/tmp/vlm_memory_agent_qwen_deps}"
OUTPUT_PATH="${OUTPUT_PATH:-/tmp/qwen36_agent_smoke_trajectory.json}"
MEMORY_PATH="${MEMORY_PATH:-/tmp/qwen36_agent_smoke_memory.json}"
MAX_STEPS="${MAX_STEPS:-6}"

cd "${ROOT}"
export VLM_MEMORY_AGENT_QWEN_DEPS="${QWEN_DEPS_DIR}"
export PYTHONPATH="${QWEN_DEPS_DIR}:${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m vlm_memory_agent \
  --env mock_osworld \
  --vlm-backend openai \
  --base-url "http://127.0.0.1:${QWEN_PORT}/v1" \
  --model "${MODEL_NAME}" \
  --output "${OUTPUT_PATH}" \
  --memory-path "${MEMORY_PATH}" \
  --max-steps "${MAX_STEPS}"
