# VLM Memory Agent

Research scaffold for a memory-augmented VLM GUI agent on OSWorld-like tasks.

The codebase is intentionally small and modular so method work can replace each
piece independently:

```text
interactive env -> observation -> screen parser -> memory retrieval
                -> VLM policy -> action parser -> env.step
                -> trajectory logger -> memory consolidation
```

## What Is Included

- `VLMGuiAgent`: a ReAct-style GUI agent loop with perception, memory retrieval,
  action generation, execution, trajectory logging, and memory update.
- `HierarchicalMemoryStore`: a dynamic experience graph with trajectory leaves,
  state-action pattern nodes, strategy nodes, and evidence edges.
- `MockOSWorldEnv`: a deterministic OSWorld-style GUI environment for local
  smoke tests.
- `LocalBrowserSalesEnv`: a fully local browser-like sales workflow that
  renders real PNG screenshots and exercises a longer GUI action sequence
  without Docker, VM, KVM, or a system browser.
- `OSWorldAdapter`: a thin adapter for a real OSWorld installation or any
  env object exposing `reset()` and `step()`.
- `OSWorldEnvConfig` / CLI runner: constructs OSWorld `DesktopEnv` when the
  external OSWorld package and VM are available.
- `OpenAICompatibleVLMClient`: API backend for GPT-4o, Qwen-VL served by
  vLLM/SGLang, or other OpenAI-compatible VLM servers.
- `Qwen36LocalVLMClient`: optional local Transformers backend for the cached
  Qwen3.6 checkpoint.
- `RuleBasedVLMClient`: deterministic local backend for tests.
- Lightweight JSON API server for integration with external runners.

## Why This Shape

Current GUI/computer-use VLM agents usually separate:

- perception / screen parsing,
- high-level planning,
- action grounding,
- environment execution,
- trajectory logging,
- optional memory or experience reuse.

This scaffold follows that mainstream shape while keeping memory as a first-class
module. It is designed for comparing:

- no memory,
- full trajectory replay,
- flat summary memory,
- vector/RAG memory,
- hierarchical experience memory.

## Quick Start

Run the local mock benchmark:

```bash
cd /mnt/hdfs/byte_ai_sales/user/zhangjuntian/vlm-memory-agent
PYTHONPATH=src python -m vlm_memory_agent --vlm-backend rule
```

Expected result: the mock task succeeds and writes:

```text
runs/trajectory.json
runs/memory.json
```

Run tests:

```bash
PYTHONPATH=src python -m pytest tests
```

Run a local long-horizon GUI task on machines that cannot host OSWorld:

```bash
cd /mnt/hdfs/byte_ai_sales/user/zhangjuntian/vlm-memory-agent
bash scripts/run_local_browser_gui_smoke.sh
```

Expected result: the sales approval workflow succeeds, the trajectory is written
to `/tmp/local_browser_gui_trajectory.json`, memory is written to
`/tmp/local_browser_gui_memory.json`, and screenshots are written under
`/tmp/local_browser_gui_screens/`.

This task is not a substitute for OSWorld isolation or benchmark scoring. It is
intended as the current-machine-compatible research loop for testing perception
prompts, action parsing, trajectory logging, memory updates, and multi-step GUI
policies before moving the same agent interface into a real sandbox.

Enter the persisted runtime environment:

```bash
cd /mnt/hdfs/byte_ai_sales/user/zhangjuntian/vlm-memory-agent
scripts/enter_persistent_runtime_env.sh
```

The script restores `/tmp` dependency directories from
`runs/tmp_artifacts_20260612.tar` when they are missing, exports
`VLM_MEMORY_AGENT_QWEN_DEPS`, `VLM_MEMORY_AGENT_OSWORLD_DEPS`, and `PYTHONPATH`,
then opens an interactive shell. Use `FORCE_RESTORE=1` to rebuild those `/tmp`
directories from the persisted archive.

For the verified split deployment where OSWorld and the agent loop run on a
devbox while Qwen3.6 inference runs on a Merlin MLX worker, see
`docs/merlin_devbox_qwen36_osworld_deployment.md`.

## Use An OpenAI-Compatible VLM

```bash
OPENAI_API_KEY=... \
PYTHONPATH=src python -m vlm_memory_agent \
  --vlm-backend openai \
  --model gpt-4o \
  --base-url https://api.openai.com/v1
```

For local Qwen/InternVL served through vLLM or SGLang, set `--base-url` to the
local OpenAI-compatible endpoint and `--model` to the served model name.

You can also try direct local Transformers inference:

```bash
PYTHONPATH=src python -m vlm_memory_agent \
  --vlm-backend qwen36-local \
  --model-path /mnt/hdfs/byte_ai_sales/user/zhangjuntian/model_cache/models/Qwen3.6-35B-A3B
```

For the cached Qwen3.6-35B-A3B checkpoint on the current MLX GPU worker, use
the direct local Transformers path. The worker image provides a CUDA
12.6-compatible global Torch build; the worker dependency installer therefore
adds only the newer Qwen-aware Transformers stack into `/tmp` and keeps Torch
out of that target directory:

```bash
mlx worker login 955911 -- bash \
  /mnt/hdfs/byte_ai_sales/user/zhangjuntian/vlm-memory-agent/scripts/install_qwen36_local_worker_deps.sh

mlx worker login 955911 -- bash \
  /mnt/hdfs/byte_ai_sales/user/zhangjuntian/vlm-memory-agent/scripts/run_qwen36_local_agent_smoke_worker.sh
```

The smoke script runs `--vlm-backend qwen36-local` on all four visible GPUs and
writes `/tmp/qwen36_agent_smoke_trajectory.json`. A successful run should finish
the mock file-search task with `"success": true`.

On this worker, recent vLLM/SGLang wheels pull Torch/CUDA wheels that require a
newer driver than the installed CUDA 12.6-compatible driver. Use direct local
Transformers inference unless the worker image/driver is upgraded or a serving
backend is built against the installed Torch/CUDA stack.

The model must return JSON:

```json
{
  "thought": "The report result is visible, so open it.",
  "action_type": "click",
  "target": "result_0",
  "text": null
}
```

## API Server

```bash
cd /mnt/hdfs/byte_ai_sales/user/zhangjuntian/vlm-memory-agent
PYTHONPATH=src python -m vlm_memory_agent.api.server
```

Endpoints:

```text
GET  /health
GET  /memory
POST /run_episode {"task_id": "search_report"}
```

## Real OSWorld Integration

Install optional runtime dependencies into separate target directories. Keeping
Qwen dependencies first avoids `desktop-env`'s older Transformers pin from
shadowing Qwen3.6 support:

```bash
cd /mnt/hdfs/byte_ai_sales/user/zhangjuntian/vlm-memory-agent
bash scripts/install_runtime_deps.sh
export VLM_MEMORY_AGENT_QWEN_DEPS=/tmp/vlm_memory_agent_qwen_deps
export VLM_MEMORY_AGENT_OSWORLD_DEPS=/tmp/vlm_memory_agent_runtime_deps
export PYTHONPATH=src
```

First check optional runtime dependencies:

```bash
PYTHONPATH=src python -m vlm_memory_agent.preflight --check osworld --check qwen36
```

For a single readiness gate that combines OSWorld task/provider checks, Qwen3.6
model checks, and the endpoint protocol check, use:

```bash
ENV_FILE=configs/osworld_qwen36.env bash scripts/check_osworld_ready.sh
JSON=1 ENV_FILE=configs/osworld_qwen36.env bash scripts/check_osworld_ready.sh
```

The JSON form reports `ready`, every check, and the blocking required checks.
`QWEN_SERVING_MODE=auto` selects `endpoint` when `QWEN_BASE_URL` is set and
`local` otherwise. Use `QWEN_SERVING_MODE=endpoint` when the OSWorld runner and
Qwen server are on different machines; in that mode local Qwen model files are
reported only as optional reference information, while the endpoint protocol
check is the required model gate.

When using a served Qwen3.6 endpoint, pass the same endpoint/model to preflight
so it checks the OpenAI-compatible `/v1/models` route before a VM is started:

```bash
PYTHONPATH=src python -m vlm_memory_agent.preflight \
  --check osworld \
  --check qwen36 \
  --task-config /path/to/osworld/task_config.json \
  --vm-path /path/to/Ubuntu-x86.vmx \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen3.6-35b-a3b
```

If `QWEN_BASE_URL` is not set, the run scripts use direct local Transformers
inference and require a visible CUDA device. On CPU/master nodes, serve the
checkpoint from a GPU worker and set `QWEN_BASE_URL`; otherwise preflight will
stop before the VM is launched.

After the server is up, run a stricter endpoint check before launching OSWorld:

```bash
ENV_FILE=configs/osworld_qwen36.env bash scripts/check_qwen36_endpoint.sh
```

This verifies both `/v1/models` and a minimal `/v1/chat/completions` request
that must return the JSON action protocol used by the agent.

The OSWorld run/eval scripts perform this endpoint healthcheck automatically
when `QWEN_BASE_URL` is set. Set `SKIP_QWEN_HEALTHCHECK=1` only when debugging
the environment separately.

This repository does not vendor OSWorld or a VM image. A real OSWorld run needs
an installed OSWorld checkout whose Python environment can import
`desktop_env.desktop_env.DesktopEnv`, a valid VM path/provider setup, and a
task JSON config from OSWorld.

For `vmware` and `virtualbox`, provide `--osworld-vm-path` to an existing local
VM file. For `docker`, `desktop-env` can resolve or download the qcow2 image
when `--osworld-vm-path` is omitted, but that requires Docker and network access
to the OSWorld image source. Cloud providers resolve the backing machine through
their provider configuration rather than a local VM file.

Once those exist, run one task through the CLI:

```bash
PYTHONPATH=src python -m vlm_memory_agent \
  --env osworld \
  --osworld-task-config /path/to/osworld/task_config.json \
  --osworld-vm-path /path/to/vm \
  --osworld-provider vmware \
  --os-type Ubuntu \
  --vlm-backend openai \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen3.6-35b-a3b \
  --memory-path runs/osworld_memory.json \
  --output runs/osworld_trajectory.json
```

The same command is wrapped by:

```bash
cp configs/osworld_qwen36.env.example configs/osworld_qwen36.env
# edit OSWORLD_TASK_CONFIG, OSWORLD_VM_PATH, TASK_ID if selecting from a directory,
# and optionally QWEN_BASE_URL
ENV_FILE=configs/osworld_qwen36.env bash scripts/run_osworld_qwen36.sh
```

For a directory of OSWorld task JSON files, use the batch evaluator:

```bash
ENV_FILE=configs/osworld_qwen36.env LIMIT=10 bash scripts/eval_osworld_qwen36.sh
```

It writes an aggregate report with `success_rate`, `avg_steps`, `avg_reward`,
and per-task records. It also writes per-task trajectories under
`TRAJECTORY_DIR` so failed GUI rollouts can be inspected step by step.

If the GPU worker image and driver support the requested vLLM/SGLang wheel
stack, the Qwen3.6 checkpoint can also be served from a GPU worker:

```bash
MODEL_PATH=/mnt/hdfs/byte_ai_sales/user/zhangjuntian/model_cache/models/Qwen3.6-35B-A3B \
MODEL_NAME=qwen3.6-35b-a3b \
GPU_TYPE=NVIDIA-H20 \
GPU_COUNT=2 \
INSTALL_VLLM=1 \
bash scripts/launch_qwen36_vllm_mlx.sh
```

If the requested GPU type is only available through Arnold rather than the
default workspace resource pool, pass the MLX scheduling fields explicitly:

```bash
MLX_RESOURCE_TYPE=arnold \
MLX_CLUSTER=cloudnative-maliva \
MLX_QUEUE_NAME=... \
MLX_USER_GROUP=... \
GPU_TYPE=H100-SXM-80GB \
GPU_COUNT=1 \
TP_SIZE=1 \
INSTALL_VLLM=1 \
bash scripts/launch_qwen36_vllm_mlx.sh
```

Then set `QWEN_BASE_URL` in `configs/osworld_qwen36.env` to the server's
OpenAI-compatible `/v1` endpoint and run `scripts/run_osworld_qwen36.sh`.

The adapter translates the agent's semantic actions into the common OSWorld
`pyautogui` action string format:

```text
click(target with bbox) -> pyautogui.click(x, y)
click("(x, y)")         -> pyautogui.click(x, y)
double_click(x, y)      -> pyautogui.doubleClick(x, y)
right_click(x, y)       -> pyautogui.rightClick(x, y)
type(text)              -> pyautogui.typewrite(text)
paste(text)             -> pyperclip.copy(text), then pyautogui.hotkey("ctrl", "v")
hotkey(ctrl+l)          -> pyautogui.hotkey("ctrl", "l")
press(enter)            -> pyautogui.press("enter")
scroll(down)            -> pyautogui.scroll(-5)
wait                    -> WAIT
done                    -> DONE, then env.evaluate() for reward
fail                    -> FAIL, then env.evaluate() for infeasible tasks
```

The default `adapter-action-mode=pyautogui` sends these strings directly, which
matches `desktop_env`'s standard action path. `adapter-action-mode=dict` wraps
non-terminal actions as `{"action_type": "pyautogui", "command": ...}` for
OSWorld revisions or external wrappers that expect command dictionaries; terminal
actions are still sent as `WAIT`, `DONE`, or `FAIL`.

`paste` imports `pyperclip` inside the OSWorld VM's Python controller. If that
import fails, the generated command falls back to `pyautogui.typewrite`; host
`pyperclip` availability is therefore reported as an optional readiness signal,
not a hard launch blocker.

If you need to construct OSWorld yourself, wrap it directly:

```python
from vlm_memory_agent.agent import VLMGuiAgent
from vlm_memory_agent.envs.osworld_adapter import OSWorldAdapter
from vlm_memory_agent.llm.openai_compatible import OpenAICompatibleVLMClient
from vlm_memory_agent.memory.store import HierarchicalMemoryStore

# osworld_env = ...  # created by your OSWorld checkout
env = OSWorldAdapter(osworld_env)
agent = VLMGuiAgent(
    vlm=OpenAICompatibleVLMClient(model="gpt-4o"),
    memory=HierarchicalMemoryStore("runs/osworld_memory.json"),
)
trajectory = agent.run_episode(env, task_id="your_osworld_task_id")
```

If your OSWorld revision returns a different observation/action schema, edit
`src/vlm_memory_agent/envs/osworld_adapter.py`; the agent and memory code should
not need changes.

## Where To Implement Research Ideas

- New memory method:
  `src/vlm_memory_agent/memory/store.py`
- New VLM backend:
  `src/vlm_memory_agent/llm/`
- Screen parser / OmniParser / OCR integration:
  `src/vlm_memory_agent/tools/screen_parser.py`
- Action parser / grounding constraints:
  `src/vlm_memory_agent/tools/action_parser.py`
- OSWorld or AndroidWorld adapter:
  `src/vlm_memory_agent/envs/`
- Agent policy loop:
  `src/vlm_memory_agent/agent.py`
- Benchmark runner:
  `src/vlm_memory_agent/eval.py`

## Trajectory And Memory Format

Each trajectory records:

- task,
- observation summary,
- action,
- model thought,
- feedback,
- reward,
- terminal status,
- structured observation/action details,
- `metadata.osworld_action`, the exact raw command sent to OSWorld when using
  the real adapter.

Memory consolidation creates:

- L0 trajectory nodes,
- L0 image-evidence nodes for key screenshots,
- L1 failure-reflection nodes for failed histories,
- L1 state-action pattern nodes,
- L2 strategy nodes,
- abstraction/evidence edges.

`image-evidence` nodes store the screenshot path, trajectory node id, step
index, before/after phase, action, status, reward, and feedback in node
metadata. They are linked from the trajectory and pattern nodes, so a retrieved
memory can be traced back to the exact visual state that supported it without
embedding raw image bytes into the memory JSON.

`failure-reflection` nodes are created from failed trajectories or failed steps.
They preserve the bad action, triggering screen state, environment feedback,
avoid condition, recovery hint, and linked image evidence ids. This makes
negative experience first-class memory: future episodes can retrieve not only
what worked, but also what failed and the visual evidence behind that reflection.

This is enough to run controlled studies on cross-episode experience transfer:

```text
past episodes -> memory graph -> future episode decision -> feedback -> update
```
