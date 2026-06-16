#!/usr/bin/env bash
set -euo pipefail

# 功能: 在当前机器上运行无需 Docker/VM 的本地浏览器式 GUI 长程任务 smoke。
# 上游依赖: 依赖主 CLI、LocalBrowserSalesEnv、RuleBasedVLMClient 和 Pillow 截图渲染。
# 下游依赖: 当前机器无法运行 OSWorld 时，用该脚本验证 GUI agent loop、截图、动作执行和轨迹记录。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_PATH="${OUTPUT_PATH:-/tmp/local_browser_gui_trajectory.json}"
MEMORY_PATH="${MEMORY_PATH:-/tmp/local_browser_gui_memory.json}"
SCREENSHOT_DIR="${SCREENSHOT_DIR:-/tmp/local_browser_gui_screens}"
MAX_STEPS="${MAX_STEPS:-18}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m vlm_memory_agent \
  --env local_browser \
  --vlm-backend rule \
  --max-steps "${MAX_STEPS}" \
  --output "${OUTPUT_PATH}" \
  --memory-path "${MEMORY_PATH}" \
  --screenshot-dir "${SCREENSHOT_DIR}"
