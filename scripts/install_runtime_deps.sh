#!/usr/bin/env bash
set -euo pipefail

# 功能: 将 Qwen 和 OSWorld 可选运行依赖分别安装到隔离 target 目录。
# 上游依赖: 依赖 pip、QWEN_DEPS_DIR、OSWORLD_DEPS_DIR 和网络/包源可用性。
# 下游依赖: preflight、OSWorld runner、Qwen serving wrapper 通过 VLM_MEMORY_AGENT_*_DEPS 使用这些依赖。

QWEN_DEPS_DIR="${QWEN_DEPS_DIR:-/tmp/vlm_memory_agent_qwen_deps}"
OSWORLD_DEPS_DIR="${OSWORLD_DEPS_DIR:-/tmp/vlm_memory_agent_runtime_deps}"

python -m pip install --target "${QWEN_DEPS_DIR}" --no-compile \
  'transformers>=4.57.0,<5' \
  qwen-vl-utils

python -m pip install --target "${OSWORLD_DEPS_DIR}" --no-compile \
  desktop-env==1.0.2 \
  pyperclip

cat <<EOF
Runtime deps installed.

Use:
  export VLM_MEMORY_AGENT_QWEN_DEPS="${QWEN_DEPS_DIR}"
  export VLM_MEMORY_AGENT_OSWORLD_DEPS="${OSWORLD_DEPS_DIR}"
  export PYTHONPATH=src
EOF
