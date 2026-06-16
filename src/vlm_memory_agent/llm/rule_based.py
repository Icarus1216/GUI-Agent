"""功能: 实现确定性规则 VLM 后端，用于 mock 文件搜索任务的无模型 smoke test。
上游依赖: 依赖 llm.base 的 VLMClient/VLMResponse，并读取 agent prompt 中的可见 UI 文本。
下游依赖: CLI 默认后端、API server、BenchmarkRunner mock 评测和单元测试使用它稳定验证主循环。
"""

from __future__ import annotations

from vlm_memory_agent.llm.base import VLMClient, VLMResponse


class RuleBasedVLMClient(VLMClient):
    """Deterministic local policy for smoke tests.

    该后端不是通用 GUI agent，只覆盖 mock 文件搜索和本地 sales workflow。
    它的价值是让测试在没有模型、GPU 或网络时仍能验证 agent loop、memory
    更新、trajectory 写入和环境状态机。
    """

    def decide(self, prompt: str, image_path: str | None = None) -> VLMResponse:
        """根据 prompt 中的可见 UI 文本返回固定策略动作。"""

        lower = prompt.lower()
        if "local sales workspace" in lower or "acme renewal email" in lower or "sales approval workflow" in lower:
            return self._decide_local_sales(self._current_observation_lower(prompt), lower)
        if "quarterly sales report.pdf is open" in lower:
            return VLMResponse("The target document is visible, so finish.", "done")
        if "id=result_0 role=listitem label=quarterly sales report.pdf" in lower:
            return VLMResponse("The correct result is visible; open it.", "click", target="result_0")
        if "id=search_box role=textbox label=quarterly sales report" in lower and "id=search_button" in lower:
            return VLMResponse("The query is entered; submit the search.", "click", target="search_button")
        if "id=search_box role=textbox" in lower and "label=quarterly sales report" not in lower:
            return VLMResponse("The search field is visible; type the query.", "type", text="quarterly sales report")
        return VLMResponse("Type the task-specific search query.", "type", text="quarterly sales report")

    def _current_observation_lower(self, prompt: str) -> str:
        """从完整 agent prompt 中截取当前 observation 部分。

        规则策略只应该看当前屏幕，而不是被 memory 里的历史文本误导。
        """

        marker = "Current observation:\n"
        start = prompt.find(marker)
        if start < 0:
            return prompt.lower()
        start += len(marker)
        end = prompt.find("\n\nRelevant long-term experience memory:", start)
        if end < 0:
            end = len(prompt)
        return prompt[start:end].lower()

    def _decide_local_sales(self, obs_lower: str, lower: str) -> VLMResponse:
        """本地 sales workflow 的手写策略。"""

        if "status: submitted" in obs_lower:
            return VLMResponse("The approval form is submitted, so the task is complete.", "done")
        if "id=submit_report" in obs_lower and "revenue: 184000" in obs_lower and "approver: priya shah" in obs_lower:
            return VLMResponse("CRM is saved and the approval fields are filled, so submit the form.", "click", target="submit_report")
        if "id=approver_field" in obs_lower and "approver: <empty>" in obs_lower:
            return VLMResponse("The form needs the approver from the email.", "type", target="approver_field", text="Priya Shah")
        if "id=q2_revenue_field" in obs_lower and "revenue: <empty>" in obs_lower:
            return VLMResponse("The form needs the Q2 revenue from the email.", "type", target="q2_revenue_field", text="184000")
        if "id=nav_reports" in obs_lower and "status: saved" in obs_lower:
            return VLMResponse("The CRM stage is saved; move to the approval form.", "click", target="nav_reports")
        if "id=save_account" in obs_lower and "stage field: renewal review" in obs_lower and "status: not saved" in obs_lower:
            return VLMResponse("The account stage is correct and should be saved.", "click", target="save_account")
        if "id=stage_field" in obs_lower and "stage field: renewal review" not in obs_lower:
            return VLMResponse("The CRM stage must match the instruction in the email.", "type", target="stage_field", text="Renewal review")
        if "id=account_acme" in obs_lower:
            return VLMResponse("Open the matching Acme CRM account.", "click", target="account_acme")
        if "id=crm_search_button" in obs_lower and "query: acme manufacturing" in obs_lower:
            return VLMResponse("The account query is entered; submit the CRM search.", "click", target="crm_search_button")
        if "id=crm_search_box" in obs_lower:
            return VLMResponse("Search CRM for the account named in the email.", "type", target="crm_search_box", text="Acme Manufacturing")
        if "id=open_crm" in obs_lower:
            return VLMResponse("The email contains the needed account data; open CRM.", "click", target="open_crm")
        if "id=email_acme" in obs_lower:
            return VLMResponse("Start by reading the Acme renewal email.", "click", target="email_acme")
        return VLMResponse("Navigate to the Acme workflow item.", "click", target="email_acme")
