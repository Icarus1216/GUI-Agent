#!/usr/bin/env bash
set -euo pipefail

# 功能: 检查已配置的 Qwen3.6 OpenAI-compatible endpoint 是否满足模型列表和 JSON action 协议。
# 上游依赖: 依赖 env 文件中的 QWEN_BASE_URL/MODEL_NAME、Qwen 依赖目录和 vlm_memory_agent.qwen36_healthcheck。
# 下游依赖: OSWorld 运行脚本和人工部署检查用它验证 endpoint 可被 agent 消费。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT}/configs/osworld_qwen36.env.example}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

MODEL_NAME="${MODEL_NAME:-qwen3.6-35b-a3b}"
TIMEOUT="${TIMEOUT:-30}"

if [[ -z "${QWEN_BASE_URL:-}" ]]; then
  echo "error: set QWEN_BASE_URL to the OpenAI-compatible Qwen3.6 endpoint" >&2
  exit 2
fi

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export VLM_MEMORY_AGENT_QWEN_DEPS="${VLM_MEMORY_AGENT_QWEN_DEPS:-/tmp/vlm_memory_agent_qwen_deps}"

cd "${ROOT}"
python -m vlm_memory_agent.qwen36_healthcheck \
  --base-url "${QWEN_BASE_URL}" \
  --model "${MODEL_NAME}" \
  --timeout "${TIMEOUT}"
