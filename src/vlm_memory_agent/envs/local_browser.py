"""功能: 实现无需 Docker/VM/系统浏览器的本地浏览器式长程 GUI 任务环境。
上游依赖: 依赖 core.types、InteractiveEnv 和 Pillow 生成可视化截图。
下游依赖: CLI、rule backend、测试和本机 smoke 脚本用它验证真实截图驱动的 GUI agent loop。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from vlm_memory_agent.core.types import AgentAction, Observation, StepResult, StepStatus, UIElement
from vlm_memory_agent.envs.base import InteractiveEnv


@dataclass(slots=True)
class SalesWorkflowTask:
    """本地销售审批 GUI 任务的固定业务数据。"""

    task_id: str
    instruction: str
    account: str
    stage: str
    q2_revenue: str
    approver: str


class LocalBrowserSalesEnv(InteractiveEnv):
    """Browser-like CRM workflow rendered locally as screenshots.

    The environment is intentionally deterministic and fully local. It provides
    a long-horizon GUI task without requiring OSWorld's VM sandbox.
    """

    def __init__(self, screenshot_dir: str | Path = "runs/local_browser_screens") -> None:
        self.screenshot_dir = Path(screenshot_dir)
        self.tasks = {
            "sales_approval": SalesWorkflowTask(
                task_id="sales_approval",
                instruction=(
                    "Read the Acme renewal email, update the CRM account stage, "
                    "fill the quarterly revenue approval form, and submit it."
                ),
                account="Acme Manufacturing",
                stage="Renewal review",
                q2_revenue="184000",
                approver="Priya Shah",
            )
        }
        self.task: SalesWorkflowTask | None = None
        self.step_count = 0
        self.page = "inbox"
        self.crm_query = ""
        self.crm_stage = "Prospecting"
        self.crm_saved = False
        self.report_revenue = ""
        self.report_approver = ""
        self.report_submitted = False

    def reset(self, task_id: str | None = None) -> Observation:
        """重置本地浏览器任务状态并清理旧截图。

        每一步都会重新渲染 PNG，因此 reset 时删除上次运行的 step 截图，
        避免调试时把旧 episode 的图片误认为当前状态。
        """

        selected = task_id or "sales_approval"
        self.task = self.tasks[selected]
        self.step_count = 0
        self.page = "inbox"
        self.crm_query = ""
        self.crm_stage = "Prospecting"
        self.crm_saved = False
        self.report_revenue = ""
        self.report_approver = ""
        self.report_submitted = False
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        for old_screenshot in self.screenshot_dir.glob("step_*.png"):
            if old_screenshot.is_file():
                old_screenshot.unlink()
        return self._observe()

    def step(self, action: AgentAction) -> StepResult:
        """执行一个语义动作并返回本地 GUI 状态转移。

        该环境模拟的是业务工作流，而不是像素级浏览器：click/type 只对
        当前页面可见的 element_id 生效。这样能稳定测试长程规划、记忆和
        trajectory 写入，而无需 VM/Docker。
        """

        if self.task is None:
            raise RuntimeError("Call reset() before step().")

        observation = self._observe()
        self.step_count += 1
        status = StepStatus.RUNNING
        reward = 0.0
        feedback = self._apply_action(action)

        if self.report_submitted:
            status = StepStatus.SUCCESS
            reward = 1.0
            feedback = "Sales approval workflow completed."
        elif action.action_type == "fail":
            status = StepStatus.FAILED
            feedback = action.text or "Agent failed."
        elif self.step_count >= 18:
            status = StepStatus.FAILED
            feedback = "Step limit reached before submitting the approval."

        next_observation = self._observe()
        return StepResult(
            observation=observation,
            action=action,
            status=status,
            reward=reward,
            feedback=feedback,
            next_observation=next_observation,
            metadata={"page": self.page},
        )

    def close(self) -> None:
        return None

    def _apply_action(self, action: AgentAction) -> str:
        """按 action_type 分发到 click/type 等页面逻辑。"""

        if action.action_type == "wait":
            return "Waited."
        if action.action_type == "fail":
            return action.text or "Agent failed."
        if action.action_type == "done":
            if self.report_submitted:
                return "Already submitted."
            return "Cannot finish before the approval is submitted."
        if action.action_type == "click":
            return self._click(action.target or "")
        if action.action_type in {"type", "paste"}:
            return self._type(action.target or "", action.text or "")
        if action.action_type == "press":
            return "No focused keyboard shortcut is needed in this task."
        return f"No effect for action {action.compact()}."

    def _click(self, target: str) -> str:
        """处理按钮/列表项点击带来的页面跳转和提交。"""

        if target == "email_acme" and self.page == "inbox":
            self.page = "email_acme"
            return "Opened Acme renewal email."
        if target == "open_crm" and self.page == "email_acme":
            self.page = "crm_search"
            return "Opened CRM search."
        if target == "crm_search_button" and self.page == "crm_search":
            if self.crm_query.strip().lower() == self.task.account.lower():
                self.page = "crm_results"
                return "CRM search results loaded."
            return "No CRM result matched the query."
        if target == "account_acme" and self.page == "crm_results":
            self.page = "crm_account"
            return "Opened Acme CRM account."
        if target == "save_account" and self.page == "crm_account":
            if self.crm_stage.strip().lower() == self.task.stage.lower():
                self.crm_saved = True
                return "Saved CRM stage."
            return "CRM stage is not correct yet."
        if target == "nav_reports" and self.page in {"crm_account", "email_acme", "crm_results", "crm_search"}:
            self.page = "approval_form"
            return "Opened quarterly approval form."
        if target == "submit_report" and self.page == "approval_form":
            if (
                self.crm_saved
                and self.report_revenue.strip() == self.task.q2_revenue
                and self.report_approver.strip().lower() == self.task.approver.lower()
            ):
                self.report_submitted = True
                return "Submitted approval."
            return "Approval form is incomplete or CRM stage was not saved."
        if target == "nav_inbox":
            self.page = "inbox"
            return "Returned to inbox."
        return f"No visible effect for click target={target} on page={self.page}."

    def _type(self, target: str, text: str) -> str:
        """处理文本输入字段写入。"""

        if target == "crm_search_box" and self.page == "crm_search":
            self.crm_query = text
            return f"Typed CRM query: {text}"
        if target == "stage_field" and self.page == "crm_account":
            self.crm_stage = text
            return f"Updated stage field: {text}"
        if target == "q2_revenue_field" and self.page == "approval_form":
            self.report_revenue = text
            return f"Typed Q2 revenue: {text}"
        if target == "approver_field" and self.page == "approval_form":
            self.report_approver = text
            return f"Typed approver: {text}"
        return f"Cannot type into target={target} on page={self.page}."

    def _observe(self) -> Observation:
        """根据当前业务状态生成 Observation 和对应截图。"""

        assert self.task is not None
        elements = self._elements()
        screen_text = self._screen_text()
        screenshot_path = self._render(elements, screen_text)
        return Observation(
            step=self.step_count,
            task=self.task.instruction,
            screenshot_path=str(screenshot_path),
            screen_text=screen_text,
            ui_elements=elements,
            metadata={
                "page": self.page,
                "crm_query": self.crm_query,
                "crm_stage": self.crm_stage,
                "crm_saved": self.crm_saved,
                "report_revenue": self.report_revenue,
                "report_approver": self.report_approver,
                "report_submitted": self.report_submitted,
            },
        )

    def _elements(self) -> list[UIElement]:
        """返回当前页面可见的结构化 UI 元素。"""

        nav = [UIElement("nav_inbox", "Inbox", "button", (24, 92, 136, 132))]
        if self.page == "inbox":
            return nav + [
                UIElement("email_acme", "Acme renewal request", "listitem", (260, 150, 1040, 224)),
                UIElement("email_globex", "Globex contract note", "listitem", (260, 238, 1040, 312)),
            ]
        if self.page == "email_acme":
            return nav + [
                UIElement("open_crm", "Open CRM", "button", (260, 560, 420, 612)),
                UIElement("nav_reports", "Approval form", "button", (438, 560, 636, 612)),
            ]
        if self.page == "crm_search":
            return nav + [
                UIElement("crm_search_box", self.crm_query or "Search accounts", "textbox", (260, 180, 780, 232)),
                UIElement("crm_search_button", "Search", "button", (800, 180, 940, 232)),
                UIElement("nav_reports", "Approval form", "button", (980, 180, 1180, 232)),
            ]
        if self.page == "crm_results":
            return nav + [
                UIElement("account_acme", "Acme Manufacturing", "listitem", (260, 180, 1040, 250)),
                UIElement("nav_reports", "Approval form", "button", (260, 560, 460, 612)),
            ]
        if self.page == "crm_account":
            return nav + [
                UIElement("stage_field", self.crm_stage, "textbox", (430, 260, 820, 312)),
                UIElement("save_account", "Save account", "button", (840, 260, 1010, 312)),
                UIElement("nav_reports", "Approval form", "button", (260, 560, 460, 612)),
            ]
        return nav + [
            UIElement("q2_revenue_field", self.report_revenue or "Q2 revenue", "textbox", (430, 238, 820, 290)),
            UIElement("approver_field", self.report_approver or "Approver", "textbox", (430, 322, 820, 374)),
            UIElement("submit_report", "Submit approval", "button", (430, 440, 640, 496)),
        ]

    def _screen_text(self) -> str:
        """返回给模型和 memory 使用的页面文本摘要。"""

        task = self.task
        if self.page == "inbox":
            return "Inbox: Acme renewal request; Globex contract note."
        if self.page == "email_acme":
            return (
                f"Email from Sales Ops. Account: {task.account}. Required CRM stage: {task.stage}. "
                f"Q2 revenue: {task.q2_revenue}. Approver: {task.approver}."
            )
        if self.page == "crm_search":
            return f"CRM account search. Query: {self.crm_query or '<empty>'}."
        if self.page == "crm_results":
            return f"CRM results for {self.crm_query}: {task.account}."
        if self.page == "crm_account":
            saved = "saved" if self.crm_saved else "not saved"
            return f"CRM account: {task.account}. Stage field: {self.crm_stage}. Status: {saved}."
        submitted = "submitted" if self.report_submitted else "not submitted"
        return (
            f"Quarterly approval form. Revenue: {self.report_revenue or '<empty>'}. "
            f"Approver: {self.report_approver or '<empty>'}. Status: {submitted}."
        )

    def _render(self, elements: list[UIElement], screen_text: str) -> Path:
        """把当前页面状态画成真实 PNG 截图。

        这让本地 smoke test 也能走与 OSWorld 类似的 image_path 代码路径，
        同时保留结构化 UI 元素便于 rule backend 和 prompt 调试。
        """

        path = self.screenshot_dir / f"step_{self.step_count:03d}_{self.page}.png"
        image = Image.new("RGB", (1280, 720), "#f5f7fb")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        draw.rectangle((0, 0, 1280, 64), fill="#1f2937")
        draw.text((24, 22), "Local Sales Workspace", fill="white", font=font)
        draw.rectangle((0, 64, 200, 720), fill="#e5e7eb")
        draw.text((24, 74), "Navigation", fill="#111827", font=font)
        draw.rectangle((230, 96, 1210, 650), fill="white", outline="#cbd5e1", width=2)
        draw.text((260, 118), f"Task: {self.task.instruction}", fill="#111827", font=font)
        draw.text((260, 138), screen_text, fill="#374151", font=font)
        for element in elements:
            if element.bbox is None:
                continue
            fill = "#dbeafe" if element.role == "button" else "#f8fafc"
            if element.role == "listitem":
                fill = "#ecfdf5"
            if element.role == "textbox":
                fill = "#ffffff"
            draw.rectangle(element.bbox, fill=fill, outline="#2563eb", width=2)
            draw.text((element.bbox[0] + 12, element.bbox[1] + 16), f"{element.label} [{element.element_id}]", fill="#111827", font=font)
        image.save(path)
        return path
