#!/usr/bin/env bash
set -euo pipefail

# 功能: 一键进入持久化的 VLM Memory Agent 运行环境，必要时从 HDFS 归档恢复 /tmp 依赖目录。
# 上游依赖: 依赖 runs/tmp_artifacts_20260612.tar、runtime_paths.py 使用的环境变量和当前 Python/Torch 基础环境。
# 下游依赖: 开发者交互 shell、smoke 脚本、Qwen3.6 本地后端和 OSWorld 预检可复用该入口。

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUNTIME_ARCHIVE="${RUNTIME_ARCHIVE:-${ROOT}/runs/tmp_artifacts_20260612.tar}"
QWEN_DEPS_DIR="${QWEN_DEPS_DIR:-/tmp/vlm_memory_agent_qwen_deps}"
OSWORLD_DEPS_DIR="${OSWORLD_DEPS_DIR:-/tmp/vlm_memory_agent_runtime_deps}"
DESKTOP_ENV_PKG_DIR="${DESKTOP_ENV_PKG_DIR:-/tmp/vlm_memory_agent_desktop_env_pkg}"
AUTO_RESTORE="${AUTO_RESTORE:-1}"
FORCE_RESTORE="${FORCE_RESTORE:-0}"

restore_entry() {
  local path="$1"
  local entry="$2"
  if [[ "${FORCE_RESTORE}" == "1" && -e "${path}" ]]; then
    echo "Removing existing ${path} before restore ..."
    rm -rf "${path}"
  fi
  if [[ -e "${path}" ]]; then
    return
  fi
  if [[ ! -f "${RUNTIME_ARCHIVE}" ]]; then
    cat >&2 <<EOF
error: runtime archive not found: ${RUNTIME_ARCHIVE}

Expected a persisted archive created from /tmp artifacts. Set RUNTIME_ARCHIVE
or recreate it before entering this runtime environment.
EOF
    exit 2
  fi
  echo "Restoring ${entry} from ${RUNTIME_ARCHIVE} ..."
  tar -C /tmp -xf "${RUNTIME_ARCHIVE}" "${entry}"
}

if [[ "${AUTO_RESTORE}" != "0" ]]; then
  restore_entry "${QWEN_DEPS_DIR}" "$(basename "${QWEN_DEPS_DIR}")"
  restore_entry "${OSWORLD_DEPS_DIR}" "$(basename "${OSWORLD_DEPS_DIR}")"
  restore_entry "${DESKTOP_ENV_PKG_DIR}" "$(basename "${DESKTOP_ENV_PKG_DIR}")"
fi

export VLM_MEMORY_AGENT_QWEN_DEPS="${QWEN_DEPS_DIR}"
export VLM_MEMORY_AGENT_OSWORLD_DEPS="${OSWORLD_DEPS_DIR}:${DESKTOP_ENV_PKG_DIR}"
export PYTHONPATH="${QWEN_DEPS_DIR}:${OSWORLD_DEPS_DIR}:${DESKTOP_ENV_PKG_DIR}:${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export VLM_MEMORY_AGENT_RUNTIME_ACTIVE=1

cat <<EOF
VLM Memory Agent runtime environment is active.

ROOT=${ROOT}
VLM_MEMORY_AGENT_QWEN_DEPS=${VLM_MEMORY_AGENT_QWEN_DEPS}
VLM_MEMORY_AGENT_OSWORLD_DEPS=${VLM_MEMORY_AGENT_OSWORLD_DEPS}
PYTHONPATH=${PYTHONPATH}

Quick checks:
  python -m vlm_memory_agent.preflight --check qwen36
  bash scripts/run_local_browser_gui_smoke.sh
EOF

if [[ "$#" -gt 0 ]]; then
  exec "$@"
fi

cd "${ROOT}"
exec "${SHELL:-/bin/bash}" -i
