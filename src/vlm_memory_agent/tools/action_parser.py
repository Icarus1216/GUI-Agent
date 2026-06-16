"""功能: 将 VLMResponse 转为内部 AgentAction，并对 action_type 做白名单校验。
上游依赖: 依赖 core.types.AgentAction 和 llm.base.VLMResponse。
下游依赖: VLMGuiAgent.act 和测试通过该解析器拦截非法模型动作。
"""

from __future__ import annotations

from vlm_memory_agent.core.types import AgentAction
from vlm_memory_agent.llm.base import VLMResponse


VALID_ACTIONS = {"click", "double_click", "right_click", "type", "paste", "hotkey", "press", "scroll", "wait", "done", "fail"}


class ActionParser:
    def parse(self, response: VLMResponse) -> AgentAction:
        """校验模型动作并转换为 AgentAction。

        这里不尝试自动修复未知 action_type，因为错误动作如果进入 OSWorld
        可能造成不可逆 UI 状态变化。更安全的策略是返回 fail，让 trajectory
        明确记录“模型输出了非法动作”。
        """

        action_type = response.action_type
        if action_type not in VALID_ACTIONS:
            return AgentAction("fail", text=f"Invalid action_type: {action_type}", thought=response.thought)
        return AgentAction(
            action_type=action_type,  # type: ignore[arg-type]
            target=response.target,
            text=response.text,
            thought=response.thought,
            metadata=response.metadata,
        )
