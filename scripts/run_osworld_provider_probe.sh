#!/usr/bin/env bash
set -euo pipefail

# 功能: 对当前主机执行 OSWorld provider 端到端探测：host runtime、preflight、最小 DesktopEnv episode。
# 上游依赖: 依赖 probe_osworld_host.sh、desktop-env 运行依赖目录、最小 OSWorld task 和 vlm_memory_agent CLI。
# 下游依赖: 机器补齐 Docker/VMware/VirtualBox/provider 后，用该脚本一键验证真实 OSWorld 是否能启动。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROVIDER="${OSWORLD_PROVIDER:-docker}"
TASK_CONFIG="${OSWORLD_TASK_CONFIG:-${ROOT}/examples/osworld_minimal_infeasible_task.json}"
OUTPUT_PATH="${OUTPUT_PATH:-/tmp/osworld_provider_probe_trajectory.json}"
MEMORY_PATH="${MEMORY_PATH:-/tmp/osworld_provider_probe_memory.json}"

export VLM_MEMORY_AGENT_OSWORLD_DEPS="${VLM_MEMORY_AGENT_OSWORLD_DEPS:-/tmp/vlm_memory_agent_runtime_deps}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT}"

scripts/probe_osworld_host.sh

PREFLIGHT_ARGS=(
  --check osworld
  --provider "${PROVIDER}"
  --task-config "${TASK_CONFIG}"
)

if [[ -n "${OSWORLD_VM_PATH:-}" ]]; then
  PREFLIGHT_ARGS+=(--vm-path "${OSWORLD_VM_PATH}")
fi

python -m vlm_memory_agent.preflight "${PREFLIGHT_ARGS[@]}"

RUN_ARGS=(
  --env osworld
  --osworld-task-config "${TASK_CONFIG}"
  --osworld-provider "${PROVIDER}"
  --vlm-backend rule
  --max-steps 1
  --no-memory-update
  --output "${OUTPUT_PATH}"
  --memory-path "${MEMORY_PATH}"
)

if [[ -n "${OSWORLD_VM_PATH:-}" ]]; then
  RUN_ARGS+=(--osworld-vm-path "${OSWORLD_VM_PATH}")
fi

python -m vlm_memory_agent "${RUN_ARGS[@]}"
