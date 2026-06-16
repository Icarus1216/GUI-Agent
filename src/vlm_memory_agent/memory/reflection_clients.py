"""功能: 提供 memory controller 可选的小模型/小 VLM 反思客户端实现。
上游依赖: 依赖 memory.controller 协议、runtime_paths、Transformers、torch 和本地 Qwen3.5-4B 权重。
下游依赖: CLI 可按需启用本地 Qwen3.5-4B 来增强 failure-reflection 记忆。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vlm_memory_agent.llm.qwen36_compat import apply_qwen36_transformers_compat
from vlm_memory_agent.runtime_paths import apply_runtime_dependency_paths


DEFAULT_MEMORY_REFLECTION_MODEL_PATH = "/mnt/hdfs/byte_ai_sales/user/zhangjuntian/model_cache/models/Qwen3.5-4B"


class Qwen35MemoryReflectionClient:
    """基于本地 Qwen3.5-4B 的 memory reflection 客户端。

    该客户端只在第一次 `reflect()` 时懒加载模型，避免普通 agent 运行、导入
    CLI 或单元测试时占用 GPU/显存。它优先按 VLM processor 组织输入，并把
    failure-reflection 关联的关键截图作为图像证据传给模型；如果当前
    Transformers 版本不支持多模态输入，则退化为纯文本反思。
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MEMORY_REFLECTION_MODEL_PATH,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 256,
    ) -> None:
        self.model_path = str(model_path)
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self._processor: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def reflect(self, prompt: str, image_paths: list[str] | None = None) -> str:
        """生成失败轨迹的结构化反思 JSON 文本。

        Controller 负责解析和保守合并 JSON；这里的职责是约束小模型输出，
        并只返回新生成 token，避免 prompt 中的 JSON schema 干扰解析。
        """

        self._load()
        assert self._model is not None
        assert self._torch is not None

        evidence_paths = [path for path in image_paths or [] if Path(path).exists()]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a memory controller for a GUI agent. Return only one valid JSON object. "
                    "Do not output markdown, code fences, XML tags, or chain-of-thought."
                ),
            },
            {
                "role": "user",
                "content": self._user_content(prompt, evidence_paths),
            },
        ]
        inputs = self._prepare_inputs(messages)
        inputs = {key: value.to(self._model.device) if hasattr(value, "to") else value for key, value in inputs.items()}

        with self._torch.inference_mode():
            output_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated_ids = output_ids[:, inputs["input_ids"].shape[-1] :]
        decoder = self._processor or self._tokenizer
        if hasattr(decoder, "batch_decode"):
            return decoder.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return self._tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    def _load(self) -> None:
        """懒加载 Qwen3.5-4B processor/tokenizer 和 model。"""

        if self._model is not None:
            return
        apply_runtime_dependency_paths()
        try:
            import torch
            import transformers
            from transformers import AutoProcessor, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Qwen3.5 memory reflection requires `transformers` and `torch`. "
                "Set VLM_MEMORY_AGENT_QWEN_DEPS if those dependencies were installed into an isolated target."
            ) from exc

        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Memory reflection model path not found: {model_path}")

        self._torch = torch
        model_kwargs = {"device_map": self.device_map, "trust_remote_code": True}
        if self.torch_dtype == "auto":
            model_kwargs["torch_dtype"] = "auto"
        else:
            model_kwargs["torch_dtype"] = getattr(torch, self.torch_dtype)

        apply_qwen36_transformers_compat()
        try:
            self._processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            self._processor = None
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        model_classes = [
            getattr(transformers, "AutoModelForImageTextToText", None),
            getattr(transformers, "AutoModelForVision2Seq", None),
            getattr(transformers, "AutoModelForCausalLM", None),
            getattr(transformers, "AutoModel", None),
        ]
        last_error: Exception | None = None
        for model_cls in [model_cls for model_cls in model_classes if model_cls is not None]:
            try:
                self._model = model_cls.from_pretrained(model_path, **model_kwargs)
                break
            except Exception as exc:
                last_error = exc
        if self._model is None:
            raise RuntimeError(f"Could not load Qwen3.5 memory reflection model: {last_error}") from last_error
        self._model.eval()

    def _user_content(self, prompt: str, image_paths: list[str]) -> str | list[dict[str, Any]]:
        if not image_paths:
            return prompt
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_path in image_paths[:4]:
            content.append({"type": "image", "image": image_path})
        return content

    def _prepare_inputs(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """兼容 VLM processor、纯文本 tokenizer 和不同 chat template 版本。"""

        if self._processor is not None:
            vision_kwargs = self._process_vision_info(messages)
            try:
                return self._processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    enable_thinking=False,
                    **vision_kwargs,
                )
            except TypeError:
                try:
                    text = self._processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                images = vision_kwargs.get("images") or self._load_images(messages)
                try:
                    return self._processor(text=[text], images=images or None, return_tensors="pt")
                except TypeError:
                    return self._processor(text=[text], return_tensors="pt")

        assert self._tokenizer is not None
        text_messages = [
            {"role": message["role"], "content": self._flatten_content(message["content"])}
            for message in messages
        ]
        try:
            return self._tokenizer.apply_chat_template(
                text_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self._tokenizer.apply_chat_template(text_messages, add_generation_prompt=True, tokenize=False)
            return self._tokenizer(text, return_tensors="pt")

    def _process_vision_info(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            from qwen_vl_utils import process_vision_info
        except ModuleNotFoundError:
            return {}
        image_inputs, video_inputs = process_vision_info(messages)
        kwargs: dict[str, Any] = {}
        if image_inputs:
            kwargs["images"] = image_inputs
        if video_inputs:
            kwargs["videos"] = video_inputs
        return kwargs

    def _load_images(self, messages: list[dict[str, Any]]) -> list[Any]:
        images: list[Any] = []
        for message in messages:
            content = message.get("content", "")
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    try:
                        from PIL import Image
                    except ModuleNotFoundError:
                        return []
                    images.append(Image.open(item["image"]).convert("RGB"))
        return images

    def _flatten_content(self, content: str | list[dict[str, Any]]) -> str:
        if isinstance(content, str):
            return content
        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif item.get("type") == "image":
                parts.append(f"[image evidence: {item.get('image')}]")
        return "\n".join(part for part in parts if part)
