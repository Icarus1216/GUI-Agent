#!/usr/bin/env bash
set -euo pipefail

# 功能: 在 GPU worker 上用本地 Qwen3.6 Transformers 后端运行本地浏览器式 GUI 长程任务。
# 上游依赖: 依赖本地 Qwen3.6 权重、install_qwen36_local_worker_deps.sh 产物、CUDA worker 和主 CLI。
# 下游依赖: 用于验证真实本地 Qwen3.6 后端是否能完成无需 Docker/VM 的 GUI agent demo。

ROOT="${ROOT:-/mnt/hdfs/byte_ai_sales/user/zhangjuntian/vlm-memory-agent}"
QWEN_DEPS_DIR="${QWEN_DEPS_DIR:-/tmp/vlm_memory_agent_qwen_deps_tf59_nodeps}"
MODEL_PATH="${MODEL_PATH:-/mnt/hdfs/byte_ai_sales/user/zhangjuntian/model_cache/models/Qwen3.6-35B-A3B}"
OUTPUT_PATH="${OUTPUT_PATH:-/tmp/qwen36_local_browser_gui_trajectory.json}"
MEMORY_PATH="${MEMORY_PATH:-/tmp/qwen36_local_browser_gui_memory.json}"
SCREENSHOT_DIR="${SCREENSHOT_DIR:-/tmp/qwen36_local_browser_gui_screens}"
MAX_STEPS="${MAX_STEPS:-18}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

cd "${ROOT}"
export VLM_MEMORY_AGENT_QWEN_DEPS="${QWEN_DEPS_DIR}"
export PYTHONPATH="${QWEN_DEPS_DIR}:${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES

python -m vlm_memory_agent \
  --env local_browser \
  --vlm-backend qwen36-local \
  --model-path "${MODEL_PATH}" \
  --device-map auto \
  --torch-dtype bfloat16 \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --max-steps "${MAX_STEPS}" \
  --output "${OUTPUT_PATH}" \
  --memory-path "${MEMORY_PATH}" \
  --screenshot-dir "${SCREENSHOT_DIR}"
