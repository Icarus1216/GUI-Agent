#!/usr/bin/env bash
set -euo pipefail

# 功能: 运行单个 OSWorld 任务，自动选择 Qwen endpoint 后端或本地 qwen36-local 后端。
# 上游依赖: 依赖 env 文件、OSWorld task/VM/provider 参数、Qwen 模型或 QWEN_BASE_URL、preflight 和主 CLI。
# 下游依赖: 单任务真实 OSWorld 调试入口，输出 trajectory 和 memory 文件。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT}/configs/osworld_qwen36.env.example}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # env 文件既给本脚本使用，也会通过 export 传给 Python healthcheck/client。
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
MEMORY_PATH="${MEMORY_PATH:-runs/osworld_qwen36_memory.json}"
OUTPUT_PATH="${OUTPUT_PATH:-runs/osworld_qwen36_trajectory.json}"
MAX_STEPS="${MAX_STEPS:-15}"
RETRIEVE_K="${RETRIEVE_K:-4}"
TASK_ID="${TASK_ID:-}"
QWEN_HEALTHCHECK_TIMEOUT="${QWEN_HEALTHCHECK_TIMEOUT:-30}"
SKIP_QWEN_HEALTHCHECK="${SKIP_QWEN_HEALTHCHECK:-0}"

if [[ -z "${OSWORLD_TASK_CONFIG:-}" ]]; then
  echo "error: set OSWORLD_TASK_CONFIG to an OSWorld task JSON file or directory" >&2
  exit 2
fi

case "${OSWORLD_PROVIDER}" in
  vmware|virtualbox)
    # 本地 VM provider 需要显式 VM 路径；否则 desktop-env 初始化到一半才会失败。
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
  # endpoint 模式用于 devbox 跑 OSWorld、Merlin worker 跑模型的部署形态；
  # preflight 只检查 endpoint 协议，不要求 devbox 拥有本地 Qwen 权重/GPU。
  PREFLIGHT_ARGS+=(--base-url "${QWEN_BASE_URL}")
else
  # 没有 endpoint 时才要求当前机器能本地直载 Qwen3.6。
  PREFLIGHT_ARGS+=(--require-cuda)
fi
python -m vlm_memory_agent.preflight "${PREFLIGHT_ARGS[@]}"

if [[ -n "${QWEN_BASE_URL:-}" && "${SKIP_QWEN_HEALTHCHECK}" != "1" ]]; then
  # 先做严格 chat JSON action 协议检查，再启动 VM 任务，避免 OSWorld 已经
  # reset 后才发现模型 endpoint 不可用。
  python -m vlm_memory_agent.qwen36_healthcheck \
    --base-url "${QWEN_BASE_URL}" \
    --model "${MODEL_NAME}" \
    --timeout "${QWEN_HEALTHCHECK_TIMEOUT}"
fi

COMMON_ARGS=(
  --env osworld
  --osworld-task-config "${OSWORLD_TASK_CONFIG}"
  --osworld-provider "${OSWORLD_PROVIDER}"
  --osworld-region "${OSWORLD_REGION}"
  --osworld-snapshot-name "${OSWORLD_SNAPSHOT_NAME}"
  --os-type "${OSWORLD_OS_TYPE}"
  --client-password "${OSWORLD_CLIENT_PASSWORD}"
  --memory-path "${MEMORY_PATH}"
  --output "${OUTPUT_PATH}"
  --max-steps "${MAX_STEPS}"
  --retrieve-k "${RETRIEVE_K}"
)

if [[ -n "${OSWORLD_VM_PATH}" ]]; then
  COMMON_ARGS+=(--osworld-vm-path "${OSWORLD_VM_PATH}")
fi

if [[ -n "${TASK_ID}" ]]; then
  COMMON_ARGS+=(--task-id "${TASK_ID}")
fi

if [[ -n "${QWEN_BASE_URL:-}" ]]; then
  # OpenAI backend 覆盖 GPT、vLLM、SGLang 和 workspace-proxy 暴露的 Qwen。
  python -m vlm_memory_agent \
    "${COMMON_ARGS[@]}" \
    --vlm-backend openai \
    --base-url "${QWEN_BASE_URL}" \
    --model "${MODEL_NAME}"
else
  python -m vlm_memory_agent \
    "${COMMON_ARGS[@]}" \
    --vlm-backend qwen36-local \
    --model-path "${MODEL_PATH}"
fi
