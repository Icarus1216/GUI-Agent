#!/usr/bin/env bash
set -euo pipefail

# 功能: 在 worker 上直接用本地 Transformers Qwen3.6 后端跑 mock_osworld agent smoke。
# 上游依赖: 依赖本地模型权重、install_qwen36_local_worker_deps.sh 产物、CUDA_VISIBLE_DEVICES 和主 CLI。
# 下游依赖: 当前 4 卡 worker 的推荐验证入口，输出 /tmp/qwen36_agent_smoke_trajectory.json。

ROOT="${ROOT:-/mnt/hdfs/byte_ai_sales/user/zhangjuntian/vlm-memory-agent}"
QWEN_DEPS_DIR="${QWEN_DEPS_DIR:-/tmp/vlm_memory_agent_qwen_deps_tf59_nodeps}"
MODEL_PATH="${MODEL_PATH:-/mnt/hdfs/byte_ai_sales/user/zhangjuntian/model_cache/models/Qwen3.6-35B-A3B}"
OUTPUT_PATH="${OUTPUT_PATH:-/tmp/qwen36_agent_smoke_trajectory.json}"
MEMORY_PATH="${MEMORY_PATH:-/tmp/qwen36_agent_smoke_memory.json}"
MAX_STEPS="${MAX_STEPS:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

cd "${ROOT}"
export VLM_MEMORY_AGENT_QWEN_DEPS="${QWEN_DEPS_DIR}"
export PYTHONPATH="${QWEN_DEPS_DIR}:${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES

python -m vlm_memory_agent \
  --env mock_osworld \
  --vlm-backend qwen36-local \
  --model-path "${MODEL_PATH}" \
  --device-map auto \
  --torch-dtype bfloat16 \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --max-steps "${MAX_STEPS}" \
  --no-memory-update \
  --output "${OUTPUT_PATH}" \
  --memory-path "${MEMORY_PATH}"
