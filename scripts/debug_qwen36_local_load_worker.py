"""功能: 在 GPU worker 上最小化验证 Qwen3.6 本地 Transformers 后端能否完成权重加载。
上游依赖: 依赖 Qwen36LocalVLMClient、默认/传入模型路径、worker CUDA/Torch/Transformers 环境。
下游依赖: 部署排障时直接运行本脚本，确认问题发生在模型加载还是 agent 推理链路。
"""

from __future__ import annotations

import argparse
import traceback

from vlm_memory_agent.llm.qwen36_local import DEFAULT_QWEN36_MODEL_PATH, Qwen36LocalVLMClient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_QWEN36_MODEL_PATH)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    args = parser.parse_args()

    client = Qwen36LocalVLMClient(
        model_path=args.model_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        max_new_tokens=32,
    )
    try:
        client._load()
    except Exception:
        traceback.print_exc()
        return 1
    print("Qwen3.6 local model load OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
