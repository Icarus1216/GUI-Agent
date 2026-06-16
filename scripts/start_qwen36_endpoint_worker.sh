#!/usr/bin/env bash
set -euo pipefail

# 功能: 在已登录 worker 上安装/准备依赖并用 tmux 或后台进程启动 Qwen3.6 vLLM endpoint。
# 上游依赖: 依赖本地模型路径、vLLM/Transformers 依赖目录、GPU/TP 参数、tmux 可选和 serve_qwen36 模块。
# 下游依赖: endpoint serving 路线使用该脚本创建本机 /v1 服务，后续由 healthcheck 和 agent smoke 调用。

ROOT="${ROOT:-/mnt/hdfs/byte_ai_sales/user/zhangjuntian/vlm-memory-agent}"
MODEL_PATH="${MODEL_PATH:-/mnt/hdfs/byte_ai_sales/user/zhangjuntian/model_cache/models/Qwen3.6-35B-A3B}"
MODEL_NAME="${MODEL_NAME:-qwen3.6-35b-a3b}"
HOST="${QWEN_HOST:-${HOST:-::}}"
QWEN_PORT="${QWEN_PORT:-9001}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
QWEN_DEPS_DIR="${QWEN_DEPS_DIR:-/tmp/vlm_memory_agent_qwen_deps}"
LOG_PATH="${LOG_PATH:-/tmp/qwen36_vllm.log}"
PID_PATH="${PID_PATH:-/tmp/qwen36_vllm.pid}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
VLLM_VERSION="${VLLM_VERSION:-0.9.2}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers>=4.57.0,<5}"
RESTART="${RESTART:-0}"

cd "${ROOT}"
mkdir -p "${QWEN_DEPS_DIR}" "$(dirname "${LOG_PATH}")"

export VLM_MEMORY_AGENT_QWEN_DEPS="${QWEN_DEPS_DIR}"
export QWEN_DEPS_DIR
export PATH="${QWEN_DEPS_DIR}/bin:${PATH}"
export PYTHONPATH="${QWEN_DEPS_DIR}:${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  # 安装到 /tmp target 而不是系统 site-packages，避免污染 worker 基础镜像；
  # PATH/PYTHONPATH 在下方传给 tmux 子进程。
  python -m pip install --target "${QWEN_DEPS_DIR}" --upgrade --no-compile \
    "${TRANSFORMERS_SPEC}" \
    qwen-vl-utils \
    "vllm==${VLLM_VERSION}"
fi

python - <<'PY'
import os
from pathlib import Path

deps = Path(os.environ["QWEN_DEPS_DIR"])
ovis = deps / "vllm" / "transformers_utils" / "configs" / "ovis.py"
if ovis.exists():
    text = ovis.read_text(encoding="utf-8")
    old = 'AutoConfig.register("aimv2", AIMv2Config)'
    new = 'AutoConfig.register("aimv2", AIMv2Config, exist_ok=True)'
    if old in text and new not in text:
        ovis.write_text(text.replace(old, new), encoding="utf-8")
        print(f"Patched {ovis} for Transformers aimv2 duplicate registration.")

routing = deps / "prometheus_fastapi_instrumentator" / "routing.py"
if routing.exists():
    text = routing.read_text(encoding="utf-8")
    old = "        if match == Match.FULL:\n            route_name = route.path\n"
    new = "        if match == Match.FULL:\n            if not hasattr(route, \"path\"):\n                continue\n            route_name = route.path\n"
    if old in text and new not in text:
        routing.write_text(text.replace(old, new), encoding="utf-8")
        print(f"Patched {routing} for Starlette _IncludedRouter compatibility.")
PY

SESSION_NAME="${SESSION_NAME:-qwen36_vllm}"

if command -v tmux >/dev/null 2>&1 && tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  # 默认复用已有服务；只有 RESTART=1 时才杀掉 tmux，避免误中断长加载模型。
  if [[ "${RESTART}" != "1" ]]; then
    echo "Qwen3.6 endpoint already running in tmux session ${SESSION_NAME}; log=${LOG_PATH}"
    exit 0
  fi
  tmux kill-session -t "${SESSION_NAME}" || true
  sleep 5
fi

if [[ -f "${PID_PATH}" ]] && kill -0 "$(cat "${PID_PATH}")" 2>/dev/null; then
  if [[ "${RESTART}" != "1" ]]; then
    echo "Qwen3.6 endpoint already running: pid=$(cat "${PID_PATH}") log=${LOG_PATH}"
    exit 0
  fi
  kill "$(cat "${PID_PATH}")" || true
  sleep 5
fi

SERVE_CMD=(
  python -m vlm_memory_agent.serve_qwen36
  --model-path "${MODEL_PATH}"
  --served-model-name "${MODEL_NAME}"
  --host "${HOST}"
  --port "${QWEN_PORT}"
  --tensor-parallel-size "${TP_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
)

if command -v tmux >/dev/null 2>&1; then
  # worker 上优先用 tmux 保持服务；mlx worker login 断开后模型进程仍继续运行。
  tmux new-session -d -s "${SESSION_NAME}" \
    "cd '${ROOT}' && export VLM_MEMORY_AGENT_QWEN_DEPS='${QWEN_DEPS_DIR}' QWEN_DEPS_DIR='${QWEN_DEPS_DIR}' PATH='${PATH}' PYTHONPATH='${PYTHONPATH}' VLLM_USE_V1='${VLLM_USE_V1:-}' && exec ${SERVE_CMD[*]} >'${LOG_PATH}' 2>&1"
  tmux display-message -p -t "${SESSION_NAME}" "#{pane_pid}" > "${PID_PATH}"
  echo "Started Qwen3.6 endpoint in tmux session ${SESSION_NAME} pane_pid=$(cat "${PID_PATH}") base_url=http://127.0.0.1:${QWEN_PORT}/v1 log=${LOG_PATH}"
else
  setsid "${SERVE_CMD[@]}" >"${LOG_PATH}" 2>&1 < /dev/null &
  echo "$!" > "${PID_PATH}"
  echo "Started Qwen3.6 endpoint pid=$(cat "${PID_PATH}") base_url=http://127.0.0.1:${QWEN_PORT}/v1 log=${LOG_PATH}"
fi
