"""功能: 对比 Qwen3.6 checkpoint 权重键和当前 Transformers 模型类 state_dict 键是否匹配。
上游依赖: 依赖 runtime dependency path、qwen36_compat、accelerate empty init、Transformers AutoModel 类和 safetensors index。
下游依赖: 兼容性排障时用它判断应使用哪个 Transformers 版本/模型类加载本地权重。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm_memory_agent.llm.qwen36_compat import apply_qwen36_transformers_compat
from vlm_memory_agent.runtime_paths import apply_runtime_dependency_paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="/mnt/hdfs/byte_ai_sales/user/zhangjuntian/model_cache/models/Qwen3.6-35B-A3B",
    )
    parser.add_argument("--sample", type=int, default=20)
    args = parser.parse_args()

    apply_runtime_dependency_paths()
    apply_qwen36_transformers_compat()

    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModel

    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        AutoModelForImageTextToText = None
    try:
        from transformers import AutoModelForVision2Seq
    except ImportError:
        AutoModelForVision2Seq = None

    model_path = Path(args.model_path)
    index_path = model_path / "model.safetensors.index.json"
    checkpoint_keys = set(json.loads(index_path.read_text())["weight_map"].keys())

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(f"config_class={type(config).__name__} model_type={getattr(config, 'model_type', None)}")

    model_classes = [
        cls
        for cls in (AutoModelForImageTextToText, AutoModelForVision2Seq, AutoModel)
        if cls is not None
    ]
    for model_cls in model_classes:
        with init_empty_weights():
            model = model_cls.from_config(config, trust_remote_code=True)
        model_keys = set(model.state_dict().keys())
        missing = sorted(model_keys - checkpoint_keys)
        unexpected = sorted(checkpoint_keys - model_keys)
        print(f"\n{model_cls.__name__}")
        print(f"model_keys={len(model_keys)} checkpoint_keys={len(checkpoint_keys)}")
        print(f"missing={len(missing)} unexpected={len(unexpected)}")
        print(f"missing_sample={missing[: args.sample]}")
        print(f"unexpected_sample={unexpected[: args.sample]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
