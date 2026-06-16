#!/usr/bin/env bash
set -euo pipefail

# 功能: 从 env 文件加载 OSWorld/Qwen 配置并运行综合 readiness gate。
# 上游依赖: 依赖 configs/osworld_qwen36.env*、PYTHONPATH、Qwen/OSWorld 依赖目录和 vlm_memory_agent.osworld_ready。
# 下游依赖: README、部署前检查和 OSWorld 运行流程用它在启动 VM/模型前发现阻塞项。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT}/configs/osworld_qwen36.env.example}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export VLM_MEMORY_AGENT_QWEN_DEPS="${VLM_MEMORY_AGENT_QWEN_DEPS:-/tmp/vlm_memory_agent_qwen_deps}"
export VLM_MEMORY_AGENT_OSWORLD_DEPS="${VLM_MEMORY_AGENT_OSWORLD_DEPS:-/tmp/vlm_memory_agent_runtime_deps}"

cd "${ROOT}"

ARGS=(
  --task-config "${OSWORLD_TASK_CONFIG:-}"
  --provider "${OSWORLD_PROVIDER:-vmware}"
  --model-path "${MODEL_PATH:-/mnt/hdfs/byte_ai_sales/user/zhangjuntian/model_cache/models/Qwen3.6-35B-A3B}"
  --model "${MODEL_NAME:-qwen3.6-35b-a3b}"
  --endpoint-timeout "${QWEN_HEALTHCHECK_TIMEOUT:-30}"
  --serving-mode "${QWEN_SERVING_MODE:-auto}"
)

if [[ -n "${TASK_ID:-}" ]]; then
  ARGS+=(--task-id "${TASK_ID}")
fi
if [[ -n "${OSWORLD_VM_PATH:-}" ]]; then
  ARGS+=(--vm-path "${OSWORLD_VM_PATH}")
fi
if [[ -n "${QWEN_BASE_URL:-}" ]]; then
  ARGS+=(--base-url "${QWEN_BASE_URL}")
fi
if [[ "${SKIP_QWEN_HEALTHCHECK:-0}" == "1" ]]; then
  ARGS+=(--skip-endpoint-healthcheck)
fi
if [[ "${JSON:-0}" == "1" ]]; then
  ARGS+=(--json)
fi

python -m vlm_memory_agent.osworld_ready "${ARGS[@]}"
