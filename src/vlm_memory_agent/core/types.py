"""功能: 定义 Observation、AgentAction、StepResult、Trajectory 等 agent 运行时核心数据结构。
上游依赖: 仅依赖标准库 dataclass/Enum/typing，负责把内部状态转换为紧凑文本和 JSON-safe dict。
下游依赖: agent、环境适配器、action parser、memory、API、CLI 和测试统一使用这些类型传递状态。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


ActionType = Literal["click", "double_click", "right_click", "type", "paste", "hotkey", "press", "scroll", "wait", "done", "fail"]


class StepStatus(str, Enum):
    """环境 step 的标准终止状态。

    统一用 RUNNING/SUCCESS/FAILED 避免上游环境把 done/reward/info 的组合
    直接泄漏到 agent 主循环。OSWorld、mock env 和本地 browser env 都会
    在各自 adapter 中归一化成这三个状态。
    """

    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass(slots=True)
class UIElement:
    """屏幕上一个可操作或可引用的 UI 元素。

    `element_id` 是 agent 动作的稳定引用；`bbox` 是真实 GUI 环境里把
    语义目标落到 pyautogui 坐标的关键字段。没有 bbox 时，OSWorld 侧
    仍可让模型直接返回坐标。
    """

    element_id: str
    label: str
    role: str = "unknown"
    bbox: tuple[int, int, int, int] | None = None
    text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "label": self.label,
            "role": self.role,
            "bbox": list(self.bbox) if self.bbox else None,
            "text": self.text,
        }


@dataclass(slots=True)
class Observation:
    """单步环境观测。

    这个结构故意同时支持三类输入：截图路径、可访问性/屏幕文本、结构化
    UI 元素。不同 benchmark 给出的 observation schema 差异很大，统一
    到该类型后，screen parser、memory 和 VLM prompt 不需要关心底层来源。
    """

    step: int
    task: str
    screenshot_path: str | None = None
    screen_text: str = ""
    ui_elements: list[UIElement] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def compact(self) -> str:
        """返回适合写入 trajectory/memory 的短文本描述。

        `screen_text` 可能来自 a11y tree 或 OCR，长度会很大；这里统一截断，
        防止历史轨迹和记忆文件膨胀，也避免后续检索时长文本压过任务关键词。
        """

        elements = ", ".join(f"{e.element_id}:{e.role}:{e.label}" for e in self.ui_elements)
        screen_text = _trim_text(self.screen_text, 4000)
        return f"task={self.task}\nscreen={screen_text}\nelements=[{elements}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "task": self.task,
            "screenshot_path": self.screenshot_path,
            "screen_text": _trim_text(self.screen_text, 4000),
            "ui_elements": [element.to_dict() for element in self.ui_elements],
            "metadata": _json_safe(self.metadata),
        }


@dataclass(slots=True)
class AgentAction:
    """agent 内部动作协议。

    `action_type/target/text` 是跨环境的最小公共字段；坐标、bbox、滚动量等
    环境相关信息放在 `metadata`，由 adapter 决定如何翻译成 pyautogui、
    dict action 或其它 sandbox API。
    """

    action_type: ActionType
    target: str | None = None
    text: str | None = None
    thought: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def compact(self) -> str:
        """生成紧凑动作串，便于写 memory 和 debug trajectory。"""

        args = []
        if self.target is not None:
            args.append(f"target={self.target}")
        if self.text is not None:
            args.append(f"text={self.text!r}")
        for key in ("x", "y", "clicks", "dy"):
            if key in self.metadata:
                args.append(f"{key}={self.metadata[key]}")
        return f"{self.action_type}({', '.join(args)})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target": self.target,
            "text": self.text,
            "thought": self.thought,
            "metadata": _json_safe(self.metadata),
            "compact": self.compact(),
        }


@dataclass(slots=True)
class StepResult:
    """一次 action 执行后的完整记录。

    `observation` 是执行前状态，`next_observation` 是执行后状态。两者都
    保留是为了后续 memory consolidation 能从状态转移中总结经验，而不仅
    是记录模型输出了什么动作。
    """

    observation: Observation
    action: AgentAction
    status: StepStatus
    reward: float = 0.0
    feedback: str = ""
    next_observation: Observation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Trajectory:
    """单个任务 episode 的轨迹。

    轨迹是项目里最重要的实验产物之一：它既用于离线分析失败步骤，也会被
    `HierarchicalMemoryStore` 固化成长期记忆。`to_dict` 因此同时输出
    compact 字符串和 detail JSON，兼顾可读性与可追溯性。
    """

    task_id: str
    task: str
    steps: list[StepResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return bool(self.steps and self.steps[-1].status == StepStatus.SUCCESS)

    def append(self, result: StepResult) -> None:
        self.steps.append(result)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task": self.task,
            "success": self.success,
            "steps": [
                {
                    "observation": step.observation.compact(),
                    "observation_detail": step.observation.to_dict(),
                    "action": step.action.compact(),
                    "action_detail": step.action.to_dict(),
                    "thought": step.action.thought,
                    "status": step.status.value,
                    "reward": step.reward,
                    "feedback": step.feedback,
                    "metadata": _json_safe(step.metadata),
                    "next_observation": step.next_observation.compact() if step.next_observation else None,
                    "next_observation_detail": step.next_observation.to_dict() if step.next_observation else None,
                }
                for step in self.steps
            ],
        }


def _trim_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _json_safe(value: Any) -> Any:
    """把第三方环境返回的任意 metadata 转为 JSON-safe 结构。

    OSWorld/info 里可能混入 numpy、PIL、异常对象或 provider 私有类型。
    这里不丢字段，而是在无法 JSON 序列化时退化为 `repr`，保证 trajectory
    永远能落盘。
    """

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
