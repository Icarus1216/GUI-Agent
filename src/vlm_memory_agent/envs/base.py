"""功能: 定义所有 GUI 交互环境需要实现的最小 reset/step/close 抽象接口。
上游依赖: 依赖 core.types 中的 Observation、AgentAction 和 StepResult 协议。
下游依赖: MockOSWorldEnv、OSWorldAdapter 和 VLMGuiAgent 通过该接口解耦环境实现。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from vlm_memory_agent.core.types import AgentAction, Observation, StepResult


class InteractiveEnv(ABC):
    """Minimal OSWorld-like interactive environment interface.

    这是 agent 和环境之间唯一必须遵守的契约。具体环境可以是真实 OSWorld、
    本地 mock、浏览器模拟任务或远程 sandbox wrapper，只要能 reset 得到
    Observation，并在 step 后返回 StepResult。
    """

    @abstractmethod
    def reset(self, task_id: str | None = None) -> Observation:
        """启动或重置一个任务 episode，返回初始 observation。"""

        raise NotImplementedError

    @abstractmethod
    def step(self, action: AgentAction) -> StepResult:
        """执行一个内部 AgentAction，并返回状态转移结果。"""

        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """释放环境资源，例如 VM/session/browser 连接。"""

        raise NotImplementedError
