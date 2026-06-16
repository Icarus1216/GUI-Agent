#!/usr/bin/env bash
set -euo pipefail

# 功能: 在 worker 的 /tmp target 目录安装 Qwen3.6 本地 Transformers 推理所需依赖，并刻意不安装 torch。
# 上游依赖: 依赖 worker 全局 CUDA-compatible torch、pip、TRANSFORMERS_VERSION/TOKENIZERS_VERSION/QWEN_DEPS_DIR。
# 下游依赖: run_qwen36_local_agent_smoke_worker.sh 和 qwen36-local 后端通过该依赖目录加载新 Transformers。

QWEN_DEPS_DIR="${QWEN_DEPS_DIR:-/tmp/vlm_memory_agent_qwen_deps_tf59_nodeps}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.9.0}"
TOKENIZERS_VERSION="${TOKENIZERS_VERSION:-0.22.2}"
mkdir -p "${QWEN_DEPS_DIR}"

# Keep torch out of this target directory. The current worker image provides a
# CUDA 12.6-compatible torch build, while recent serving stacks pull CUDA 13
# wheels that do not match the worker driver.
python -m pip install --target "${QWEN_DEPS_DIR}" --upgrade --no-compile --no-deps \
  "transformers==${TRANSFORMERS_VERSION}" \
  "tokenizers==${TOKENIZERS_VERSION}" \
  safetensors \
  huggingface-hub \
  qwen-vl-utils \
  accelerate \
  pillow \
  av

python -m pip install --target "${QWEN_DEPS_DIR}" --upgrade --no-compile --no-deps \
  annotated-doc \
  anyio \
  certifi \
  click \
  filelock \
  fsspec \
  h11 \
  hf-xet \
  httpcore \
  httpx \
  idna \
  numpy \
  packaging \
  psutil \
  pyyaml \
  regex \
  requests \
  rich \
  shellingham \
  tqdm \
  typer \
  typing-extensions \
  urllib3

cat <<EOF
Local Qwen3.6 worker deps installed.

Use:
  export VLM_MEMORY_AGENT_QWEN_DEPS="${QWEN_DEPS_DIR}"
  export PYTHONPATH="${QWEN_DEPS_DIR}:src"
EOF
