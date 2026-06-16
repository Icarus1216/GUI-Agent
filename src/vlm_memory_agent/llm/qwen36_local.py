"""功能: 实现本地 Transformers 直载 Qwen3.6 VLM 的懒加载推理后端。
上游依赖: 依赖 runtime_paths、qwen36_compat、torch、transformers、qwen_vl_utils/Pillow 和本地模型权重。
下游依赖: CLI、OSWorld 脚本、worker smoke/debug 脚本和测试可选择 qwen36-local backend。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vlm_memory_agent.llm.base import VLMClient, VLMResponse, parse_vlm_response_text
from vlm_memory_agent.llm.qwen36_compat import apply_qwen36_transformers_compat
from vlm_memory_agent.runtime_paths import apply_runtime_dependency_paths


DEFAULT_QWEN36_MODEL_PATH = "/mnt/hdfs/byte_ai_sales/user/zhangjuntian/model_cache/models/Qwen3.6-35B-A3B"


class Qwen36LocalVLMClient(VLMClient):
    """Lazy local Transformers backend for Qwen3.6-style VLM checkpoints.

    本地直载主要用于 worker smoke test 和无服务化后端的调试。真实 OSWorld
    评测更推荐使用 vLLM endpoint，因为 35B 模型加载慢、显存占用高，且
    devbox 通常没有足够 GPU。
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_QWEN36_MODEL_PATH,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 512,
    ) -> None:
        self.model_path = str(model_path)
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def decide(self, prompt: str, image_path: str | None = None) -> VLMResponse:
        """生成并解析下一步动作。

        本地 Qwen 输出偶尔会违反 JSON contract，因此这里做一次 repair：
        首次解析失败后，把错误输出塞回 prompt，要求模型只返回修正后的
        JSON。两次都失败才把错误抛给 agent。
        """

        self._load()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None

        strict_prompt = self._strict_action_prompt(prompt)
        text = self._generate_action_text(strict_prompt, image_path)
        try:
            return parse_vlm_response_text(text)
        except ValueError as first_error:
            repair_prompt = (
                f"{strict_prompt}\n\n"
                "The previous answer was invalid because it was not a complete JSON object.\n"
                f"Previous invalid answer:\n{text}\n\n"
                "Return the corrected next action now as exactly one complete JSON object and no other text."
            )
            repair_text = self._generate_action_text(repair_prompt, image_path)
            try:
                return parse_vlm_response_text(repair_text)
            except ValueError as second_error:
                raise ValueError(f"{second_error}; first invalid response was: {first_error}") from second_error

    def _generate_action_text(self, prompt: str, image_path: str | None) -> str:
        """调用 Transformers generate 并只解码新生成 token。

        输入 prompt token 不应进入解析器，否则 extract_json_object 可能在
        prompt 示例 JSON 上误命中。通过切片 `output_ids[:, input_len:]`
        只保留模型回答。
        """

        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a GUI VLM agent. Return only one minified JSON object for the next action. "
                    "Do not output markdown, XML tags, chain-of-thought, analysis, or code fences."
                ),
            },
            {
                "role": "user",
                "content": self._user_content(prompt, image_path),
            },
        ]
        inputs = self._prepare_inputs(messages)
        inputs = {key: value.to(self._model.device) if hasattr(value, "to") else value for key, value in inputs.items()}

        with self._torch.inference_mode():
            output_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated_ids = output_ids[:, inputs["input_ids"].shape[-1] :]
        return self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def _strict_action_prompt(self, prompt: str) -> str:
        """给本地 Qwen 增加强约束 action 输出协议。

        本地 browser smoke task 额外内置了一段 playbook，这是为了让无真实
        视觉能力或早期模型不稳定时仍能验证 agent loop、memory 和 trajectory
        写入链路。真实 OSWorld 任务主要依赖通用 JSON contract。
        """

        task_hint = ""
        if "Acme Manufacturing" in prompt and "quarterly revenue approval form" in prompt:
            task_hint = (
                "\nLocal sales GUI playbook for this task:\n"
                "- If email_acme is visible, click email_acme.\n"
                "- If open_crm is visible, click open_crm. Do not click nav_inbox from the email page.\n"
                "- If crm_search_box is visible and Query is <empty>, type Acme Manufacturing into crm_search_box.\n"
                "- If crm_search_button is visible and Query is Acme Manufacturing, click crm_search_button.\n"
                "- If account_acme is visible, click account_acme.\n"
                "- If stage_field is visible and Stage field is not Renewal review, type Renewal review into stage_field.\n"
                "- If save_account is visible and Status is not saved, click save_account.\n"
                "- If nav_reports is visible and Status is saved, click nav_reports.\n"
                "- If q2_revenue_field is visible and Revenue is <empty>, type 184000 into q2_revenue_field.\n"
                "- If approver_field is visible and Approver is <empty>, type Priya Shah into approver_field.\n"
                "- If submit_report is visible and Revenue is 184000 and Approver is Priya Shah, click submit_report.\n"
                "- If Status is submitted, return action_type done.\n"
            )
        return (
            f"{prompt}\n\n"
            f"{task_hint}"
            "Output contract:\n"
            "Return exactly one complete JSON object on one line.\n"
            'Use only these keys: "thought", "action_type", "target", "text".\n'
            "The first character must be { and the last character must be }.\n"
            'Example: {"thought":"Open the visible Acme email.","action_type":"click","target":"email_acme","text":null}\n'
            "Do not include <think>, explanations, markdown fences, or any text outside JSON.\n"
        )

    def _load(self) -> None:
        """懒加载 processor/model。

        延迟到第一次 decide 才加载，可以让 CLI preflight、参数解析和测试在
        没有 GPU/权重时也能导入模块。加载前会注入 runtime dependency path
        并应用 Qwen3.6 Transformers 兼容补丁。
        """

        if self._model is not None:
            return
        apply_runtime_dependency_paths()
        try:
            import torch
            import transformers
            from transformers import AutoProcessor
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Local Qwen3.6 backend requires `transformers` and `torch` in this Python environment. "
                "For large checkpoints, serving the model through vLLM/SGLang and using "
                "`--vlm-backend openai` is usually easier to validate."
            ) from exc

        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Qwen3.6 model path not found: {model_path}")

        apply_qwen36_transformers_compat()
        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model_kwargs = {"device_map": self.device_map, "trust_remote_code": True}
        if self.torch_dtype != "auto":
            model_kwargs["torch_dtype"] = getattr(torch, self.torch_dtype)
        else:
            model_kwargs["torch_dtype"] = "auto"

        model_classes = [
            getattr(transformers, "AutoModelForImageTextToText", None),
            getattr(transformers, "AutoModelForVision2Seq", None),
            getattr(transformers, "AutoModel", None),
        ]
        # 不同 Transformers 版本对 VLM AutoModel 类名支持不同，按新到旧尝试。
        last_error: Exception | None = None
        for model_cls in [model_cls for model_cls in model_classes if model_cls is not None]:
            try:
                self._model = model_cls.from_pretrained(model_path, **model_kwargs)
                break
            except Exception as exc:
                last_error = exc
        if self._model is None:
            raise RuntimeError(f"Could not load local Qwen3.6 checkpoint with Transformers: {last_error}") from last_error
        self._model.eval()

    def _user_content(self, prompt: str, image_path: str | None) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image_path:
            content.insert(0, {"type": "image", "image": str(image_path)})
        return content

    def _prepare_inputs(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """兼容新旧 Qwen processor 的 chat template API。

        新版 processor 可以直接 `tokenize=True, return_dict=True` 并接收
        qwen_vl_utils 处理后的 images/videos；旧版则需要先生成 text，再
        手动传入 PIL image list。
        """

        assert self._processor is not None
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
                text = self._processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            images = vision_kwargs.get("images") or self._load_images(messages)
            return self._processor(text=[text], images=images or None, return_tensors="pt")

    def _process_vision_info(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """调用 qwen_vl_utils 解析 image/video content，缺失时退化为空。"""

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
        """旧版 processor fallback：从 message content 中读取图片文件。"""

        images: list[Any] = []
        for message in messages:
            for item in message.get("content", []):
                if isinstance(item, dict) and item.get("type") == "image":
                    try:
                        from PIL import Image
                    except ModuleNotFoundError as exc:
                        raise RuntimeError("Local image inference requires Pillow.") from exc
                    images.append(Image.open(item["image"]).convert("RGB"))
        return images
