# Merlin Worker + Devbox OSWorld Deployment

This document records the verified deployment path for running Qwen3.6
inference on a Merlin MLX GPU worker while running the GUI agent loop and
OSWorld sandbox on a separate devbox.

## Verified Current State

As of 2026-06-15, the working topology is:

```text
devbox OSWorld sandbox + agent loop
  -> workspace-proxy public HTTPS URL
  -> Merlin MLX worker 960661
  -> vLLM OpenAI-compatible endpoint on worker port 9001
  -> local Qwen3.6-35B-A3B checkpoint
```

Current worker-side deployment:

```text
worker id:    960661
GPU:          8 x H100-SXM-80GB
model path:   /mnt/hdfs/byte_ai_sales/user/zhangjuntian/model_cache/models/Qwen3.6-35B-A3B
served model: qwen3.6-35b-a3b
worker port:  9001
vLLM tmux:    qwen36_vllm
keepalive:    gpu_keepalive
vLLM log:     /tmp/qwen36_vllm_tp1.log
```

Current devbox-facing endpoint:

```text
https://workspace-proxy-candy-maliva-tce.tiktok-row.org/s/sX8lC0VZ/v1
```

This URL was created by running `mlx export --public` from inside the worker.
Do not use the older `/s/pUWKnUHC` URL. That URL was exported from the Merlin
master environment, where no service was listening on local port 9001, so
devbox and Mac calls correctly returned `502 Failed to reach upstream service`.

## Why This Architecture

The local Mac or devbox can host the OSWorld sandbox, VM/Docker runtime, and
agent loop, but cannot run the Qwen3.6-35B model efficiently. The Merlin worker
has enough GPU memory for inference but is not suitable for running OSWorld
sandbox isolation in this setup. The practical split is therefore:

```text
devbox:
  - OSWorld checkout and dependencies
  - sandbox provider: VMware, VirtualBox, Docker, or other supported provider
  - GUI task reset/step/evaluate loop
  - vlm-memory-agent runner
  - trajectory and memory files

Merlin worker:
  - Qwen3.6 checkpoint
  - vLLM OpenAI-compatible model server
  - workspace-proxy export of worker port 9001
```

This avoids direct worker IPv6 routing from devbox. Devbox reaches the model
through workspace-proxy, while the model endpoint remains a normal OpenAI-style
`/v1/models` and `/v1/chat/completions` API.

## Key Requirements

Merlin worker requirements:

- MLX worker with enough GPU memory for Qwen3.6-35B-A3B.
- Model checkpoint available at the same HDFS path from the worker.
- vLLM/Transformers dependencies installed into a worker-local target directory.
- Service must listen on IPv6 host `::` or another address accepted by MLX export.
- `mlx export --public` must be executed inside the worker, not from the Merlin
  master shell.

Devbox requirements:

- Network access to `workspace-proxy-candy-maliva-tce.tiktok-row.org`.
- OSWorld and its `desktop_env` Python package available.
- A supported OSWorld isolation backend, such as VMware, VirtualBox, Docker, or
  a company-supported remote sandbox provider.
- Access to task JSON files and a valid VM/image/snapshot configuration.
- `vlm-memory-agent` checkout or synchronized project files.

Security requirement:

- The current endpoint is a public workspace-proxy link. Anyone with the URL can
  call the model. Use only in trusted network contexts and rotate by re-exporting
  when no longer needed.

## Starting Qwen3.6 on the Worker

Start from the Merlin master shell and execute the startup script inside the
worker:

```bash
mlx worker login 960661 -- env \
  ROOT=/mnt/hdfs/byte_ai_sales/user/zhangjuntian/vlm-memory-agent \
  QWEN_DEPS_DIR=/tmp/qwen36_vllm019_deps \
  VLLM_VERSION=0.19.1 \
  TRANSFORMERS_SPEC=transformers==5.9.0 \
  TP_SIZE=1 \
  QWEN_HOST='::' \
  QWEN_PORT=9001 \
  INSTALL_DEPS=1 \
  RESTART=1 \
  GPU_MEMORY_UTILIZATION=0.95 \
  MAX_MODEL_LEN=4096 \
  LOG_PATH=/tmp/qwen36_vllm_tp1.log \
  bash /mnt/hdfs/byte_ai_sales/user/zhangjuntian/vlm-memory-agent/scripts/start_qwen36_endpoint_worker.sh
```

Notes:

- `TP_SIZE=1` is the verified stable setting on the 8 x H100 worker. It loads
  the model on one H100 and avoids the tensor-parallel/NCCL startup failures
  seen in earlier attempts.
- The 35B checkpoint took about 25 minutes to load from HDFS on first startup.
  After loading, vLLM performs profiling, FlashInfer warmup, and CUDA graph
  capture before the HTTP server is ready.
- The service is ready only after the log contains `Application startup complete`.

Check logs:

```bash
mlx worker login 960661 -- tail -n 120 /tmp/qwen36_vllm_tp1.log
```

Check that the worker port is listening:

```bash
mlx worker login 960661 -- ss -ltnp
```

Run the worker-local endpoint healthcheck:

```bash
mlx worker login 960661 -- env \
  ROOT=/mnt/hdfs/byte_ai_sales/user/zhangjuntian/vlm-memory-agent \
  QWEN_DEPS_DIR=/tmp/qwen36_vllm019_deps \
  QWEN_PORT=9001 \
  TIMEOUT=300 \
  bash /mnt/hdfs/byte_ai_sales/user/zhangjuntian/vlm-memory-agent/scripts/check_qwen36_endpoint_worker.sh
```

Expected result:

```text
OK      qwen36_endpoint: /models ok; models=['qwen3.6-35b-a3b']; /chat/completions ok; JSON action protocol ok
```

## Exporting the Worker Service

The export command must run inside the worker:

```bash
mlx worker login 960661 -- mlx export --port 9001 --public
```

The successful export for the current worker produced:

```text
URL:          https://workspace-proxy-candy-maliva-tce.tiktok-row.org/s/sX8lC0VZ
Magic string: pwk960661p11770t1781514685eo52pwt
```

Use the `/v1` suffix for OpenAI-compatible clients:

```text
https://workspace-proxy-candy-maliva-tce.tiktok-row.org/s/sX8lC0VZ/v1
```

If devbox gets `502 Failed to reach upstream service`, verify these separately:

```bash
# This must fail or be irrelevant on the Merlin master if the server only lives
# inside the worker.
curl -sS -m 30 http://127.0.0.1:9001/v1/models

# This must succeed inside the worker.
mlx worker login 960661 -- curl -sS -m 30 http://127.0.0.1:9001/v1/models
```

If worker-local curl succeeds but the public URL returns 502, re-run `mlx export`
inside the worker and use the new `/s/.../v1` URL.

## Devbox Usage

Set the model endpoint on devbox:

```bash
export QWEN_BASE_URL='https://workspace-proxy-candy-maliva-tce.tiktok-row.org/s/sX8lC0VZ/v1'
export OPENAI_BASE_URL="$QWEN_BASE_URL"
export MODEL_NAME='qwen3.6-35b-a3b'
export OPENAI_API_KEY='dummy'
export QWEN_VERIFY_SSL=0
export OPENAI_VERIFY_SSL=0
unset OPENAI_EXTRA_HEADERS
unset QWEN_EXTRA_HEADERS
```

Verify `/models`:

```bash
curl --noproxy '*' -k -L -m 60 "$QWEN_BASE_URL/models"
```

Verify chat completion:

```bash
curl --noproxy '*' -k -L -m 180 "$QWEN_BASE_URL/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"请用一句中文回答：2+2等于多少？"}],"max_tokens":128,"temperature":0}'
```

The devbox test succeeded with HTTP `chat.completion` and model
`qwen3.6-35b-a3b`, returning that `2+2 = 4`.

## Connecting to This Project's OSWorld Runner

On devbox, create a concrete env file from the example:

```bash
cd /path/to/vlm-memory-agent
cp configs/osworld_qwen36.env.example configs/osworld_qwen36.env
```

Set at least:

```bash
MODEL_NAME=qwen3.6-35b-a3b
QWEN_BASE_URL=https://workspace-proxy-candy-maliva-tce.tiktok-row.org/s/sX8lC0VZ/v1
QWEN_SERVING_MODE=endpoint
QWEN_HEALTHCHECK_TIMEOUT=180

OSWORLD_TASK_CONFIG=/path/to/osworld/evaluation_examples/examples/chrome/task.json
OSWORLD_PROVIDER=vmware
OSWORLD_VM_PATH=/path/to/Ubuntu-x86.vmx
OSWORLD_SNAPSHOT_NAME=init_state
OSWORLD_OS_TYPE=Ubuntu

MEMORY_PATH=runs/osworld_qwen36_memory.json
OUTPUT_PATH=runs/osworld_qwen36_trajectory.json
MAX_STEPS=15
RETRIEVE_K=4
```

Run readiness checks:

```bash
ENV_FILE=configs/osworld_qwen36.env bash scripts/check_qwen36_endpoint.sh
ENV_FILE=configs/osworld_qwen36.env bash scripts/check_osworld_ready.sh
```

Run one OSWorld task:

```bash
ENV_FILE=configs/osworld_qwen36.env bash scripts/run_osworld_qwen36.sh
```

Run a batch evaluation:

```bash
ENV_FILE=configs/osworld_qwen36.env LIMIT=10 bash scripts/eval_osworld_qwen36.sh
```

Outputs:

```text
MEMORY_PATH       hierarchical memory JSON used by later episodes
OUTPUT_PATH       single-task trajectory JSON
REPORT_PATH       batch aggregate report
TRAJECTORY_DIR    per-task trajectories for OSWorld eval
```

## Agent and Prompt Considerations

The current Qwen3.6 endpoint may emit thinking-style text before the final
answer. For GUI agent use, always constrain both prompt and parser:

- Prompt: require exactly one JSON action object and no markdown.
- Parser: extract the last valid action JSON object from the response and ignore
  earlier thinking text.
- Healthcheck: keep using `qwen36_healthcheck`, because it validates both the
  endpoint and the JSON action protocol expected by the agent.

The project already routes served Qwen through `OpenAICompatibleVLMClient` when
`QWEN_BASE_URL` is set. The action loop is:

```text
OSWorld screenshot
  -> screen parser
  -> memory retrieval
  -> Qwen3.6 chat completion
  -> JSON action parser
  -> OSWorldAdapter
  -> desktop_env step/evaluate
  -> trajectory + memory update
```

## Keepalive

The current worker also runs a GPU keepalive process:

```text
tmux session: gpu_keepalive
script:       scripts/start_gpu_keepalive_worker.sh
log:          /tmp/gpu_keepalive.log
```

It defaults to `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7`, leaving GPU0 for the vLLM
server. This keeps the unused H100s active without competing with the verified
TP=1 inference service.

Start it manually:

```bash
mlx worker login 960661 -- tmux new-session -d -s gpu_keepalive \
  bash /mnt/hdfs/byte_ai_sales/user/zhangjuntian/vlm-memory-agent/scripts/start_gpu_keepalive_worker.sh
```

Stop it:

```bash
mlx worker login 960661 -- tmux kill-session -t gpu_keepalive
```

Check sessions and GPU usage:

```bash
mlx worker login 960661 -- tmux ls
mlx worker login 960661 -- nvidia-smi
```

## Troubleshooting Matrix

`401 UNAUTHORIZED` from workspace-proxy:

- The URL is private or the token expired.
- Use `mlx export --public` for long-running devbox access, or use a proper
  long-lived company-approved auth path.

`502 Failed to reach upstream service`:

- The workspace-proxy URL exists, but its upstream port is not reachable.
- Most common cause here: export was run from the Merlin master instead of from
  inside the worker.
- Fix: verify worker-local `127.0.0.1:9001/v1/models`, then run
  `mlx worker login 960661 -- mlx export --port 9001 --public`.

Merlin master cannot curl the public workspace-proxy URL:

- The master shell may be blocked by ROW Operations Gateway network segregation.
- This does not invalidate devbox access. Use devbox or Mac to validate the
  public URL.

Direct devbox to worker IPv6 times out:

- Expected in this environment. Use workspace-proxy export instead of direct
  worker IPv6.

Model responds with thinking text:

- Tighten prompt and rely on JSON action extraction.
- Keep `temperature=0` for healthchecks and deterministic action tests.

Worker service is alive but slow on first request:

- First startup loads 26 safetensors shards from HDFS and performs CUDA graph
  capture. Wait for `Application startup complete` before exporting or testing.

## Redeploy Checklist

1. Confirm or launch an MLX worker with sufficient GPUs.
2. Start Qwen3.6 vLLM inside the worker with `QWEN_HOST='::'` and port `9001`.
3. Wait for `Application startup complete`.
4. Run `scripts/check_qwen36_endpoint_worker.sh` inside the worker.
5. Run `mlx export --port 9001 --public` inside the worker.
6. Put the returned `/s/.../v1` URL into devbox `QWEN_BASE_URL`.
7. Test devbox `/models` and `/chat/completions`.
8. Run `check_osworld_ready.sh`.
9. Run one OSWorld task before batch evaluation.
