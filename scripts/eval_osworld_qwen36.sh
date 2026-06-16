#!/usr/bin/env bash
set -euo pipefail

# 功能: 批量运行 OSWorld task JSON，并用 Qwen3.6 endpoint 或本地 Qwen3.6 后端生成评测报告。
# 上游依赖: 依赖 env 文件、OSWORLD_TASK_CONFIG/VM/provider 配置、Qwen 模型或 endpoint、preflight 和 osworld_eval。
# 下游依赖: 实验评测入口，输出 aggregate report、memory 文件和 per-task trajectory 目录。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT}/configs/osworld_qwen36.env.example}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # 批量评测沿用单任务 env 文件，确保模型 endpoint、VM/provider 和输出路径一致。
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

MODEL_PATH="${MODEL_PATH:-/mnt/hdfs/byte_ai_sales/user/zhangjuntian/model_cache/models/Qwen3.6-35B-A3B}"
MODEL_NAME="${MODEL_NAME:-qwen3.6-35b-a3b}"
OSWORLD_PROVIDER="${OSWORLD_PROVIDER:-vmware}"
OSWORLD_REGION="${OSWORLD_REGION:-}"
OSWORLD_SNAPSHOT_NAME="${OSWORLD_SNAPSHOT_NAME:-init_state}"
OSWORLD_OS_TYPE="${OSWORLD_OS_TYPE:-Ubuntu}"
OSWORLD_CLIENT_PASSWORD="${OSWORLD_CLIENT_PASSWORD:-}"
OSWORLD_VM_PATH="${OSWORLD_VM_PATH:-}"
MEMORY_PATH="${MEMORY_PATH:-runs/osworld_qwen36_eval_memory.json}"
REPORT_PATH="${REPORT_PATH:-runs/osworld_qwen36_eval_report.json}"
TRAJECTORY_DIR="${TRAJECTORY_DIR:-runs/osworld_qwen36_eval_trajectories}"
MAX_STEPS="${MAX_STEPS:-15}"
RETRIEVE_K="${RETRIEVE_K:-4}"
LIMIT="${LIMIT:-}"
TASK_ID="${TASK_ID:-}"
QWEN_HEALTHCHECK_TIMEOUT="${QWEN_HEALTHCHECK_TIMEOUT:-30}"
SKIP_QWEN_HEALTHCHECK="${SKIP_QWEN_HEALTHCHECK:-0}"

if [[ -z "${OSWORLD_TASK_CONFIG:-}" ]]; then
  echo "error: set OSWORLD_TASK_CONFIG to an OSWorld task JSON file or directory" >&2
  exit 2
fi

case "${OSWORLD_PROVIDER}" in
  vmware|virtualbox)
    # 本地 VM provider 需要显式 VM 路径；目录评测会反复 reset 同一 VM。
    if [[ -z "${OSWORLD_VM_PATH}" ]]; then
      echo "error: set OSWORLD_VM_PATH to a local OSWorld VM path for provider ${OSWORLD_PROVIDER}" >&2
      exit 2
    fi
    if [[ ! -e "${OSWORLD_VM_PATH}" ]]; then
      echo "error: OSWORLD_VM_PATH does not exist: ${OSWORLD_VM_PATH}" >&2
      exit 2
    fi
    ;;
  docker)
    if [[ -n "${OSWORLD_VM_PATH}" && ! -e "${OSWORLD_VM_PATH}" ]]; then
      echo "error: OSWORLD_VM_PATH does not exist: ${OSWORLD_VM_PATH}" >&2
      exit 2
    fi
    ;;
esac

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export VLM_MEMORY_AGENT_QWEN_DEPS="${VLM_MEMORY_AGENT_QWEN_DEPS:-/tmp/vlm_memory_agent_qwen_deps}"
export VLM_MEMORY_AGENT_OSWORLD_DEPS="${VLM_MEMORY_AGENT_OSWORLD_DEPS:-/tmp/vlm_memory_agent_runtime_deps}"

cd "${ROOT}"

PREFLIGHT_ARGS=(
  --check osworld
  --check qwen36
  --model-path "${MODEL_PATH}"
  --task-config "${OSWORLD_TASK_CONFIG}"
  --provider "${OSWORLD_PROVIDER}"
  --model "${MODEL_NAME}"
)
if [[ -n "${OSWORLD_VM_PATH}" ]]; then
  PREFLIGHT_ARGS+=(--vm-path "${OSWORLD_VM_PATH}")
fi
if [[ -n "${TASK_ID}" ]]; then
  PREFLIGHT_ARGS+=(--task-id "${TASK_ID}")
fi
if [[ -n "${QWEN_BASE_URL:-}" ]]; then
  # endpoint 模式适合 devbox 跑 OSWorld、Merlin worker 跑模型；当前机器不需要本地 GPU。
  PREFLIGHT_ARGS+=(--base-url "${QWEN_BASE_URL}")
else
  PREFLIGHT_ARGS+=(--require-cuda)
fi
python -m vlm_memory_agent.preflight "${PREFLIGHT_ARGS[@]}"

if [[ -n "${QWEN_BASE_URL:-}" && "${SKIP_QWEN_HEALTHCHECK}" != "1" ]]; then
  # 目录评测前只做一次 endpoint 协议检查；单任务失败细节写到各自 trajectory。
  python -m vlm_memory_agent.qwen36_healthcheck \
    --base-url "${QWEN_BASE_URL}" \
    --model "${MODEL_NAME}" \
    --timeout "${QWEN_HEALTHCHECK_TIMEOUT}"
fi

ARGS=(
  --task-config "${OSWORLD_TASK_CONFIG}"
  --osworld-provider "${OSWORLD_PROVIDER}"
  --osworld-region "${OSWORLD_REGION}"
  --osworld-snapshot-name "${OSWORLD_SNAPSHOT_NAME}"
  --os-type "${OSWORLD_OS_TYPE}"
  --client-password "${OSWORLD_CLIENT_PASSWORD}"
  --memory-path "${MEMORY_PATH}"
  --report "${REPORT_PATH}"
  --trajectory-dir "${TRAJECTORY_DIR}"
  --max-steps "${MAX_STEPS}"
  --retrieve-k "${RETRIEVE_K}"
)

if [[ -n "${OSWORLD_VM_PATH}" ]]; then
  ARGS+=(--osworld-vm-path "${OSWORLD_VM_PATH}")
fi

if [[ -n "${TASK_ID}" ]]; then
  ARGS+=(--task-id "${TASK_ID}")
fi
if [[ -n "${LIMIT}" ]]; then
  # LIMIT 用于先抽样冒烟，避免一次性启动大量 OSWorld 任务浪费 sandbox 时间。
  ARGS+=(--limit "${LIMIT}")
fi

if [[ -n "${QWEN_BASE_URL:-}" ]]; then
  python -m vlm_memory_agent.osworld_eval \
    "${ARGS[@]}" \
    --vlm-backend openai \
    --base-url "${QWEN_BASE_URL}" \
    --model "${MODEL_NAME}"
else
  python -m vlm_memory_agent.osworld_eval \
    "${ARGS[@]}" \
    --vlm-backend qwen36-local \
    --model-path "${MODEL_PATH}"
fi
