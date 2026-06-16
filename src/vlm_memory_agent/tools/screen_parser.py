"""功能: 将 Observation 中已有的屏幕文本、截图路径和 UI 元素格式化为 VLM prompt 感知文本。
上游依赖: 依赖 core.types.Observation；未来可替换接入 OCR、OmniParser 或 a11y tree 解析。
下游依赖: VLMGuiAgent.act 通过该工具构造当前屏幕上下文，测试验证截断行为。
"""

from __future__ import annotations

from vlm_memory_agent.core.types import Observation


class ScreenParserTool:
    """Perception layer placeholder.

    In production this can call OmniParser, OCR, UIA, pyautogui screenshot
    parsing, or OSWorld's accessibility tree. The default implementation
    formats already-provided UI elements.
    """

    def __init__(self, max_screen_text_chars: int = 12000):
        self.max_screen_text_chars = max_screen_text_chars

    def parse(self, observation: Observation) -> str:
        """把 observation 格式化为模型可读的感知文本。

        当前实现不做 OCR/目标检测，只忠实呈现环境已经给出的 screen_text、
        screenshot_path 和 UI element 列表。这样 mock、本地 browser、OSWorld
        a11y tree 都能共用同一个 prompt 入口。
        """

        lines = [f"Screen text: {self._truncate(observation.screen_text)}"]
        if observation.screenshot_path:
            lines.append(f"Screenshot: {observation.screenshot_path}")
        if observation.ui_elements:
            lines.append("Visible UI elements:")
            for element in observation.ui_elements:
                bbox = f" bbox={element.bbox}" if element.bbox else ""
                lines.append(f"- id={element.element_id} role={element.role} label={element.label}{bbox}")
        return "\n".join(lines)

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_screen_text_chars:
            return text
        return text[: self.max_screen_text_chars] + "\n...[truncated]"
