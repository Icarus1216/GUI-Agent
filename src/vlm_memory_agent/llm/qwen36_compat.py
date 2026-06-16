"""功能: 为不同 Transformers 版本补齐 Qwen3.6/Qwen3.5-VL-MoE checkpoint 兼容注册。
上游依赖: 依赖 torch/transformers 的可选导入和 AutoConfig/AutoModel 注册机制。
下游依赖: Qwen36LocalVLMClient、preflight、debug/inspect 脚本在加载模型前应用该兼容层。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass


@dataclass(slots=True)
class Qwen36CompatReport:
    """兼容补丁执行结果，供 preflight 打印细节。"""

    torch_pytree: bool = False
    processor_mixin: bool = False
    config_aliases: bool = False
    detail: str = ""


def apply_qwen36_transformers_compat() -> Qwen36CompatReport:
    """Register Qwen3.6 checkpoint aliases for Transformers builds that expose Qwen3-VL.

    本地权重 config 里的 model_type 可能是 `qwen3_5_moe`，但部分
    Transformers 版本只内置 qwen3_vl/qwen3_vl_moe。该补丁在加载模型前
    注册别名类，避免 AutoConfig/AutoModel 找不到 checkpoint 类型。
    """

    report = Qwen36CompatReport()
    _patch_torch_pytree(report)
    _patch_processor_mixin(report)
    _register_config_aliases(report)
    return report


def _patch_torch_pytree(report: Qwen36CompatReport) -> None:
    """补齐旧 torch 缺失的 pytree.register_constant。"""

    try:
        import torch.utils._pytree as pytree
    except Exception as exc:
        report.detail += f"torch pytree unavailable: {exc}; "
        return
    if not hasattr(pytree, "register_constant"):
        pytree.register_constant = lambda cls: cls
    report.torch_pytree = True


def _patch_processor_mixin(report: Qwen36CompatReport) -> None:
    """让旧/新 Transformers 都能从顶层访问 ProcessorMixin。"""

    try:
        import transformers
        from transformers.processing_utils import ProcessorMixin
    except Exception as exc:
        report.detail += f"ProcessorMixin unavailable: {exc}; "
        return
    if not hasattr(transformers, "ProcessorMixin"):
        transformers.ProcessorMixin = ProcessorMixin
    report.processor_mixin = True


def _register_config_aliases(report: Qwen36CompatReport) -> None:
    """注册 Qwen3.6 config/model alias。

    如果当前 Transformers 已经原生支持 `qwen3_5_moe`，直接跳过；否则
    基于 qwen3_vl/qwen3_vl_moe 类派生一层 config_class/model_type。
    """

    try:
        from transformers import AutoConfig
    except Exception as exc:
        report.detail += f"AutoConfig unavailable: {exc}; "
        return

    try:
        AutoConfig.for_model("qwen3_5_moe")
    except Exception:
        pass
    else:
        report.config_aliases = True
        report.detail += "Transformers already supports qwen3_5_moe; skipped alias registration. "
        return

    try:
        from transformers import AutoModel, AutoModelForVision2Seq

        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            AutoModelForImageTextToText = None

        from transformers.models.qwen3_vl import Qwen3VLConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLModel
        from transformers.models.qwen3_vl_moe import Qwen3VLMoeConfig
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
            Qwen3VLMoeForConditionalGeneration,
            Qwen3VLMoeModel,
        )
    except Exception as exc:
        report.detail += f"Qwen3-VL classes unavailable: {exc}; "
        return

    class Qwen36VLConfig(Qwen3VLConfig):
        model_type = "qwen3_5"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            _ensure_text_rope_scaling_dict(self)

    class Qwen36VLMoeConfig(Qwen3VLMoeConfig):
        model_type = "qwen3_5_moe"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            _ensure_text_rope_scaling_dict(self)

    class Qwen36VLModel(Qwen3VLModel):
        config_class = Qwen36VLConfig

    class Qwen36VLMoeModel(Qwen3VLMoeModel):
        config_class = Qwen36VLMoeConfig

    class Qwen36VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
        config_class = Qwen36VLConfig

    class Qwen36VLMoeForConditionalGeneration(Qwen3VLMoeForConditionalGeneration):
        config_class = Qwen36VLMoeConfig

    registrations = [
        (AutoConfig.register, ("qwen3_5", Qwen36VLConfig)),
        (AutoConfig.register, ("qwen3_5_moe", Qwen36VLMoeConfig)),
        (AutoModel.register, (Qwen36VLConfig, Qwen36VLModel)),
        (AutoModel.register, (Qwen36VLMoeConfig, Qwen36VLMoeModel)),
        (AutoModelForVision2Seq.register, (Qwen36VLConfig, Qwen36VLForConditionalGeneration)),
        (AutoModelForVision2Seq.register, (Qwen36VLMoeConfig, Qwen36VLMoeForConditionalGeneration)),
    ]
    if AutoModelForImageTextToText is not None:
        registrations.extend(
            [
                (AutoModelForImageTextToText.register, (Qwen36VLConfig, Qwen36VLForConditionalGeneration)),
                (AutoModelForImageTextToText.register, (Qwen36VLMoeConfig, Qwen36VLMoeForConditionalGeneration)),
            ]
        )

    for register, args in registrations:
        try:
            register(*args, exist_ok=True)
        except TypeError:
            try:
                register(*args)
            except ValueError:
                pass
        except ValueError:
            pass
        except Exception as exc:
            warnings.warn(f"Qwen3.6 alias registration failed for {args[0]}: {exc}", stacklevel=2)
    report.config_aliases = True


def _ensure_text_rope_scaling_dict(config: object) -> None:
    """把 rope_parameters 补成 rope_scaling，兼容不同 checkpoint schema。"""

    text_config = getattr(config, "text_config", None)
    if text_config is not None and getattr(text_config, "rope_scaling", None) is None:
        rope_parameters = getattr(text_config, "rope_parameters", None)
        if isinstance(rope_parameters, dict) and "rope_type" in rope_parameters:
            text_config.rope_scaling = dict(rope_parameters)
        else:
            text_config.rope_scaling = {"rope_type": "default"}
