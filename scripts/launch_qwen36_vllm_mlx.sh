#!/usr/bin/env bash
set -euo pipefail

# 功能: 通过 MLX 调度新 GPU worker，并在远端尝试启动 Qwen3.6 vLLM OpenAI-compatible 服务。
# 上游依赖: 依赖 mlx worker launch、模型路径、资源规格环境变量、runtime deps 安装脚本和 serve_qwen36 模块。
# 下游依赖: 需要 endpoint serving 的实验可用它申请 worker 并得到 /v1 服务地址。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/mnt/hdfs/byte_ai_sales/user/zhangjuntian/model_cache/models/Qwen3.6-35B-A3B}"
MODEL_NAME="${MODEL_NAME:-qwen3.6-35b-a3b}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
GPU_TYPE="${GPU_TYPE:-NVIDIA-H20}"
GPU_COUNT="${GPU_COUNT:-2}"
CPU_COUNT="${CPU_COUNT:-32}"
MEMORY_GB="${MEMORY_GB:-256}"
TP_SIZE="${TP_SIZE:-${GPU_COUNT}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
INSTALL_VLLM="${INSTALL_VLLM:-0}"
MLX_RESOURCE_TYPE="${MLX_RESOURCE_TYPE:-}"
MLX_CLUSTER="${MLX_CLUSTER:-}"
MLX_QUEUE_NAME="${MLX_QUEUE_NAME:-}"
MLX_NAMESPACE="${MLX_NAMESPACE:-}"
MLX_USER_GROUP="${MLX_USER_GROUP:-}"
MLX_IMAGE="${MLX_IMAGE:-}"
MLX_LOGDIR="${MLX_LOGDIR:-}"

REMOTE_CMD=$(cat <<EOF
set -euo pipefail
cd '${ROOT}'
bash scripts/install_runtime_deps.sh
export VLM_MEMORY_AGENT_QWEN_DEPS="\${VLM_MEMORY_AGENT_QWEN_DEPS:-/tmp/vlm_memory_agent_qwen_deps}"
export PYTHONPATH=src
if [[ '${INSTALL_VLLM}' == '1' ]]; then
  python -m pip install 'vllm>=0.8.5'
fi
python -m vlm_memory_agent.serve_qwen36 \
  --model-path '${MODEL_PATH}' \
  --served-model-name '${MODEL_NAME}' \
  --host '${HOST}' \
  --port '${PORT}' \
  --tensor-parallel-size '${TP_SIZE}' \
  --max-model-len '${MAX_MODEL_LEN}'
EOF
)

MLX_ARGS=(
  --gpu "${GPU_COUNT}" \
  --type "${GPU_TYPE}" \
  --cpu "${CPU_COUNT}" \
  --memory "${MEMORY_GB}" \
  --hostnetwork true \
  --workdir "${ROOT}"
)

if [[ -n "${MLX_RESOURCE_TYPE}" ]]; then
  MLX_ARGS+=(--resourcetype "${MLX_RESOURCE_TYPE}")
fi
if [[ -n "${MLX_CLUSTER}" ]]; then
  MLX_ARGS+=(--cluster "${MLX_CLUSTER}")
fi
if [[ -n "${MLX_QUEUE_NAME}" ]]; then
  MLX_ARGS+=(--queuename "${MLX_QUEUE_NAME}")
fi
if [[ -n "${MLX_NAMESPACE}" ]]; then
  MLX_ARGS+=(--namespace "${MLX_NAMESPACE}")
fi
if [[ -n "${MLX_USER_GROUP}" ]]; then
  MLX_ARGS+=(--usergroup "${MLX_USER_GROUP}")
fi
if [[ -n "${MLX_IMAGE}" ]]; then
  MLX_ARGS+=(--image "${MLX_IMAGE}")
fi
if [[ -n "${MLX_LOGDIR}" ]]; then
  MLX_ARGS+=(--logdir "${MLX_LOGDIR}")
fi

mlx worker launch "${MLX_ARGS[@]}" -- bash -lc "${REMOTE_CMD}"
