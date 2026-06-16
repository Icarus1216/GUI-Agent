"""功能: 实现确定性的 mock OSWorld 文件搜索任务，用于本地 smoke test 和单元测试。
上游依赖: 依赖 core.types 和 InteractiveEnv，内置 search_report 任务状态机。
下游依赖: CLI 默认运行、API server、BenchmarkRunner、RuleBasedVLMClient 测试和 Qwen worker smoke 使用它验证 agent loop。
"""

from __future__ import annotations

from dataclasses import dataclass

from vlm_memory_agent.core.types import AgentAction, Observation, StepResult, StepStatus, UIElement
from vlm_memory_agent.envs.base import InteractiveEnv


@dataclass(slots=True)
class MockTask:
    """mock 文件搜索任务的静态配置。"""

    task_id: str
    instruction: str
    query: str
    result_label: str


class MockOSWorldEnv(InteractiveEnv):
    """Tiny deterministic GUI benchmark for local tests.

    The task mimics a search UI:
    click search box -> type query -> click result -> done.
    It is intentionally small but exposes the same loop shape as OSWorld:
    observation, action, feedback, terminal success/failure.
    """

    def __init__(self) -> None:
        self.tasks = {
            "search_report": MockTask(
                task_id="search_report",
                instruction="Find the quarterly sales report and finish when it is open.",
                query="quarterly sales report",
                result_label="Quarterly Sales Report.pdf",
            )
        }
        self.task: MockTask | None = None
        self.step_count = 0
        self.typed_query = ""
        self.page = "home"

    def reset(self, task_id: str | None = None) -> Observation:
        """重置确定性状态机到首页。"""

        selected = task_id or "search_report"
        self.task = self.tasks[selected]
        self.step_count = 0
        self.typed_query = ""
        self.page = "home"
        return self._observe()

    def step(self, action: AgentAction) -> StepResult:
        """根据动作推进文件搜索状态机。

        这个环境故意不模拟真实鼠标焦点和浏览器细节，只验证 agent loop 的
        基本能力：读 observation、生成动作、接收反馈、到达成功终态。
        """

        if self.task is None:
            raise RuntimeError("Call reset() before step().")

        observation = self._observe()
        self.step_count += 1
        status = StepStatus.RUNNING
        reward = 0.0
        feedback = ""

        if action.action_type == "click" and action.target == "search_box":
            feedback = "Search box focused."
        elif action.action_type == "type" and self.page == "home":
            self.typed_query = action.text or ""
            feedback = f"Typed query: {self.typed_query}"
        elif action.action_type == "click" and action.target == "search_button":
            self.page = "results"
            feedback = "Search results loaded."
        elif action.action_type == "click" and action.target == "result_0" and self.page == "results":
            self.page = "document"
            feedback = "Opened Quarterly Sales Report.pdf."
        elif (action.action_type == "done" or (action.action_type == "click" and action.target == "done_button")) and self.page == "document":
            status = StepStatus.SUCCESS
            reward = 1.0
            feedback = "Task completed."
        elif action.action_type == "fail":
            status = StepStatus.FAILED
            feedback = action.text or "Agent failed."
        else:
            feedback = f"No effect for action {action.compact()}."

        if self.step_count >= 8 and status == StepStatus.RUNNING:
            status = StepStatus.FAILED
            feedback = "Step limit reached."

        next_observation = self._observe()
        return StepResult(
            observation=observation,
            action=action,
            status=status,
            reward=reward,
            feedback=feedback,
            next_observation=next_observation,
        )

    def close(self) -> None:
        return None

    def _observe(self) -> Observation:
        """把当前 mock 页面状态转换成 Observation。"""

        assert self.task is not None
        if self.page == "home":
            elements = [
                UIElement("search_box", self.typed_query or "Search files", "textbox"),
                UIElement("search_button", "Search", "button"),
            ]
            text = f"File Finder. Search files. Query: {self.typed_query or '<empty>'}"
        elif self.page == "results":
            elements = [
                UIElement("result_0", self.task.result_label, "listitem"),
                UIElement("search_box", self.typed_query, "textbox"),
            ]
            text = f"Results for {self.typed_query}: {self.task.result_label}"
        else:
            elements = [UIElement("done_button", "Done", "button")]
            text = "Quarterly Sales Report.pdf is open."
        return Observation(
            step=self.step_count,
            task=self.task.instruction,
            screen_text=text,
            ui_elements=elements,
            metadata={"page": self.page, "typed_query": self.typed_query},
        )
