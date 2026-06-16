"""功能: 定义 VLMClient 抽象接口、VLMResponse 结构和模型 JSON action 输出解析/归一化逻辑。
上游依赖: 依赖标准库 JSON/regex/dataclass，接收模型原始文本或 dict。
下游依赖: OpenAI-compatible 后端、Qwen 本地后端、healthcheck、ActionParser 和测试复用统一动作协议。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import re
from dataclasses import dataclass, field
from typing import Any


ACTION_ALIASES = {
    "click_coordinate": "click",
    "click_coordinates": "click",
    "click_point": "click",
    "left_click": "click",
    "mouse_click": "click",
    "tap": "click",
    "dblclick": "double_click",
    "doubleclick": "double_click",
    "double_click_coordinate": "double_click",
    "rightclick": "right_click",
    "right_click_coordinate": "right_click",
    "input": "type",
    "type_text": "type",
    "enter_text": "type",
    "text": "type",
    "paste_text": "paste",
    "clipboard": "paste",
    "set_text": "paste",
    "key": "press",
    "press_key": "press",
    "keyboard_press": "press",
    "key_press": "hotkey",
    "keypress": "hotkey",
    "shortcut": "hotkey",
    "mouse_scroll": "scroll",
    "wheel": "scroll",
    "sleep": "wait",
    "finish": "done",
    "complete": "done",
    "success": "done",
    "stop": "done",
    "impossible": "fail",
    "error": "fail",
}


@dataclass(slots=True)
class VLMResponse:
    """模型输出经解析后的中间结构。

    这里还不是最终 `AgentAction`，因为 action_type 可能需要继续做白名单
    校验，metadata 里也可能保留坐标、bbox、滚动量等后端特有字段。
    """

    thought: str
    action_type: str
    target: str | None = None
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class VLMClient(ABC):
    """所有模型后端的统一接口。

    agent 层只依赖 `decide(prompt, image_path)`，因此同一套 GUI loop 可以
    在 rule backend、本地 Transformers、vLLM/SGLang、OpenAI API 间切换。
    """

    @abstractmethod
    def decide(self, prompt: str, image_path: str | None = None) -> VLMResponse:
        raise NotImplementedError


def parse_vlm_response_text(text: str) -> VLMResponse:
    """从模型原始文本中提取并归一化动作 JSON。

    Qwen 类模型有时会先输出 thinking 或 markdown code fence。这里先尝试
    整体 JSON，再尝试 fenced JSON，最后从文本中扫描第一个可解码对象。
    """

    obj = extract_json_object(text)
    if not isinstance(obj, dict):
        raise ValueError(f"Model JSON must be an object: {text}")
    return vlm_response_from_dict(obj)


def vlm_response_from_dict(obj: dict[str, Any]) -> VLMResponse:
    """把多种模型 action schema 归一成 VLMResponse。

    支持字段别名和嵌套参数，例如 `action/type/action_type`、`value/text`、
    `metadata/parameters/args`。未被标准字段消费的内容会进入 metadata，
    供 OSWorld adapter 解析坐标、bbox 或滚动参数。
    """

    nested = _nested_action_payload(obj)
    merged = {**obj, **nested}
    metadata = {
        key: value
        for key, value in merged.items()
        if key
        not in {
            "thought",
            "reason",
            "action_type",
            "action",
            "type",
            "target",
            "element_id",
            "text",
            "value",
            "metadata",
            "parameters",
            "params",
            "args",
            "arguments",
        }
    }
    _copy_coordinate_metadata(merged, metadata)
    _copy_bbox_metadata(merged, metadata)

    target = merged.get("target", merged.get("element_id"))
    action_type = merged.get("action_type", merged.get("action", merged.get("type", "")))
    return VLMResponse(
        thought=str(merged.get("thought", merged.get("reason", ""))),
        action_type=normalize_action_type(str(action_type)),
        target=None if target is None else str(target),
        text=_optional_str(merged.get("text", merged.get("value"))),
        metadata=metadata,
    )


def _nested_action_payload(obj: dict[str, Any]) -> dict[str, Any]:
    """展开模型常见的嵌套参数字段。"""

    merged: dict[str, Any] = {}
    for key in ("metadata", "parameters", "params", "args", "arguments"):
        value = obj.get(key)
        if isinstance(value, dict):
            merged.update(value)
    return merged


def extract_json_object(text: str) -> Any:
    """从非严格模型输出中找到一个 JSON object。

    该函数故意只返回第一个可被 JSONDecoder 解码的对象，避免把后续自然
    语言解释拼进动作协议。parser 层会继续校验对象结构和 action_type。
    """

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return extract_json_object(fenced.group(1))

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start() :])
            return obj
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Model did not return JSON: {text}")


def normalize_action_type(action_type: str) -> str:
    """把模型动作别名归一成内部 action_type。"""

    key = action_type.strip().lower().replace("-", "_").replace(" ", "_")
    return ACTION_ALIASES.get(key, key)


def _copy_coordinate_metadata(obj: dict[str, Any], metadata: dict[str, Any]) -> None:
    """从多种坐标字段中提取 x/y 到 metadata。

    模型可能返回 `x/y`、`coordinate`、`point` 或自然字符串形式坐标；
    adapter 只需要统一读取 metadata["x"], metadata["y"]。
    """

    if "x" in obj and "y" in obj:
        metadata["x"] = obj["x"]
        metadata["y"] = obj["y"]
        return

    for key in ("coordinate", "coordinates", "point", "position", "click_point", "target_point"):
        coords = _parse_point(obj.get(key))
        if coords is not None:
            metadata.setdefault(key, obj.get(key))
            metadata["x"], metadata["y"] = coords
            return


def _copy_bbox_metadata(obj: dict[str, Any], metadata: dict[str, Any]) -> None:
    """从 bbox 推导点击中心点，兼容检测器/OmniParser 类输出。"""

    bbox = obj.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return
    metadata["bbox"] = list(bbox)
    if "x" not in metadata or "y" not in metadata:
        x1, y1, x2, y2 = (float(value) for value in bbox[:4])
        metadata["x"] = int((x1 + x2) / 2)
        metadata["y"] = int((y1 + y2) / 2)


def _parse_point(value: Any) -> tuple[int, int] | None:
    if isinstance(value, dict) and "x" in value and "y" in value:
        return int(float(value["x"])), int(float(value["y"]))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(float(value[0])), int(float(value[1]))
    if isinstance(value, str):
        numbers = re.findall(r"-?\d+(?:\.\d+)?", value)
        if len(numbers) >= 2:
            return int(float(numbers[0])), int(float(numbers[1]))
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
