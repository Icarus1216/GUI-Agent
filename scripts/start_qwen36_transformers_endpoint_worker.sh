#!/usr/bin/env bash
set -euo pipefail

# 功能: 在 worker 上用 Transformers 直载 Qwen3.6，并启动 OpenAI-compatible endpoint。
# 上游依赖: 依赖本地 Qwen3.6 权重、/tmp Qwen runtime 依赖、GPU 和 serve_qwen36_transformers 模块。
# 下游依赖: devbox/OSWorld agent 通过 /v1/models 和 /v1/chat/completions 调用该服务。

ROOT="${ROOT:-/mnt/hdfs/byte_ai_sales/user/zhangjuntian/vlm-memory-agent}"
MODEL_PATH="${MODEL_PATH:-/mnt/hdfs/byte_ai_sales/user/zhangjuntian/model_cache/models/Qwen3.6-35B-A3B}"
MODEL_NAME="${MODEL_NAME:-qwen3.6-35b-a3b}"
HOST="${QWEN_HOST:-${HOST:-0.0.0.0}}"
QWEN_PORT="${QWEN_PORT:-9001}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
QWEN_DEPS_DIR="${QWEN_DEPS_DIR:-/tmp/vlm_memory_agent_qwen_deps}"
LOG_PATH="${LOG_PATH:-/tmp/qwen36_transformers_endpoint.log}"
PID_PATH="${PID_PATH:-/tmp/qwen36_transformers_endpoint.pid}"
SESSION_NAME="${SESSION_NAME:-qwen36_transformers_endpoint}"
RESTART="${RESTART:-0}"

cd "${ROOT}"
mkdir -p "${QWEN_DEPS_DIR}" "$(dirname "${LOG_PATH}")"

export VLM_MEMORY_AGENT_QWEN_DEPS="${QWEN_DEPS_DIR}"
export QWEN_DEPS_DIR
export PYTHONPATH="${QWEN_DEPS_DIR}:${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if command -v tmux >/dev/null 2>&1 && tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  if [[ "${RESTART}" != "1" ]]; then
    echo "Qwen3.6 Transformers endpoint already running in tmux session ${SESSION_NAME}; log=${LOG_PATH}"
    exit 0
  fi
  tmux kill-session -t "${SESSION_NAME}" || true
  sleep 5
fi

if [[ -f "${PID_PATH}" ]] && kill -0 "$(cat "${PID_PATH}")" 2>/dev/null; then
  if [[ "${RESTART}" != "1" ]]; then
    echo "Qwen3.6 Transformers endpoint already running: pid=$(cat "${PID_PATH}") log=${LOG_PATH}"
    exit 0
  fi
  kill "$(cat "${PID_PATH}")" || true
  sleep 5
fi

SERVE_CMD=(
  python -m vlm_memory_agent.serve_qwen36_transformers
  --model-path "${MODEL_PATH}"
  --served-model-name "${MODEL_NAME}"
  --host "${HOST}"
  --port "${QWEN_PORT}"
  --device-map "${DEVICE_MAP}"
  --torch-dtype "${TORCH_DTYPE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
)

if command -v tmux >/dev/null 2>&1; then
  tmux new-session -d -s "${SESSION_NAME}" \
    "cd '${ROOT}' && export VLM_MEMORY_AGENT_QWEN_DEPS='${QWEN_DEPS_DIR}' QWEN_DEPS_DIR='${QWEN_DEPS_DIR}' PYTHONPATH='${PYTHONPATH}' && exec ${SERVE_CMD[*]} >'${LOG_PATH}' 2>&1"
  tmux display-message -p -t "${SESSION_NAME}" "#{pane_pid}" > "${PID_PATH}"
  echo "Started Qwen3.6 Transformers endpoint in tmux session ${SESSION_NAME} pane_pid=$(cat "${PID_PATH}") base_url=http://127.0.0.1:${QWEN_PORT}/v1 log=${LOG_PATH}"
else
  setsid "${SERVE_CMD[@]}" >"${LOG_PATH}" 2>&1 < /dev/null &
  echo "$!" > "${PID_PATH}"
  echo "Started Qwen3.6 Transformers endpoint pid=$(cat "${PID_PATH}") base_url=http://127.0.0.1:${QWEN_PORT}/v1 log=${LOG_PATH}"
fi
