"""功能: 提供独立 memory controller，统一管理记忆更新、反思增强、合并抽象和主动遗忘。
上游依赖: 依赖 core Trajectory、memory.store 和可选小 VLM 反思客户端。
下游依赖: VLMGuiAgent 通过该 controller 访问 memory，而不是直接把更新策略写死在 store 中。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from vlm_memory_agent.core.types import Trajectory
from vlm_memory_agent.memory.store import HierarchicalMemoryStore, MemoryNode, jaccard, tokenize


class MemoryReflectionClient(Protocol):
    """可插拔的小 VLM/LLM 反思接口。

    实现方只需要接收 prompt 并返回文本。controller 会从返回文本里解析 JSON，
    用于增强 failure-reflection 节点。这样可以接本地小 VLM、远程
    OpenAI-compatible 小模型，或纯规则 mock。
    """

    def reflect(self, prompt: str) -> str:
        raise NotImplementedError


@dataclass(slots=True)
class MemoryControllerConfig:
    """Memory controller 的策略参数。

    `max_nodes` 控制主动遗忘阈值；`reflection_merge_threshold` 控制相似失败
    反思是否合并；`enable_reflection_enhancement` 控制是否调用可选小 VLM。
    """

    max_nodes: int = 2000
    reflection_merge_threshold: float = 0.45
    enable_reflection_enhancement: bool = True
    enable_active_forgetting: bool = True


class MemoryController:
    """记忆系统的控制层。

    Store 负责保存图结构和基础 consolidation；Controller 负责更高层策略：
    - 调用小 VLM 或规则 prompt 增强失败反思；
    - 合并相似 failure-reflection，避免错误经验碎片化；
    - 在节点过多时主动遗忘低价值、可重建的 evidence。
    """

    def __init__(
        self,
        store: HierarchicalMemoryStore,
        reflection_client: MemoryReflectionClient | None = None,
        config: MemoryControllerConfig | None = None,
    ) -> None:
        self.store = store
        self.reflection_client = reflection_client
        self.config = config or MemoryControllerConfig()

    def retrieve(self, query: str, k: int = 4) -> list[MemoryNode]:
        """代理检索接口，保留 agent 侧调用形态。"""

        return self.store.retrieve(query, k=k)

    def prompt_context(self, nodes: list[MemoryNode]) -> str:
        """代理 prompt context 格式化接口。"""

        return self.store.prompt_context(nodes)

    def update_from_trajectory(self, trajectory: Trajectory) -> list[MemoryNode]:
        """从 trajectory 更新 memory，并执行 controller 级策略。

        返回 store 新建/更新的节点列表；后处理可能会进一步修改节点内容、
        合并 reflection 或删除低价值 evidence。
        """

        nodes = self.store.update_from_trajectory(trajectory)
        if self.config.enable_reflection_enhancement:
            self._enhance_failure_reflections(nodes)
        self._merge_failure_reflections()
        if self.config.enable_active_forgetting:
            self._apply_forgetting_policy()
        if self.store.path:
            self.store.save(self.store.path)
        return nodes

    def _enhance_failure_reflections(self, nodes: list[MemoryNode]) -> None:
        """用可选小 VLM 增强失败反思节点。

        小 VLM 只允许输出受限 JSON 字段，controller 会做保守更新；解析失败
        时保留规则反思结果，不影响主流程。
        """

        if self.reflection_client is None:
            return
        for node in nodes:
            if node.kind != "failure-reflection":
                continue
            prompt = self._build_reflection_prompt(node)
            try:
                payload = _extract_json_object(self.reflection_client.reflect(prompt))
            except (ValueError, TypeError, json.JSONDecodeError):
                node.metadata["reflection_enhancement_error"] = "invalid reflection JSON"
                continue
            self._apply_reflection_payload(node, payload)

    def _build_reflection_prompt(self, node: MemoryNode) -> str:
        """构造给小 VLM 的失败反思 prompt。"""

        return "\n".join(
            [
                "You are a memory controller for a GUI agent.",
                "Reflect on the failed GUI trajectory step and return only JSON.",
                "Required JSON keys: summary, avoid_when, recovery_hint, failure_mode, confidence.",
                f"Task id: {node.metadata.get('task_id')}",
                f"Bad action: {node.metadata.get('bad_action')}",
                f"Screen before action: {node.metadata.get('before_screen_text')}",
                f"Screen after action: {node.metadata.get('after_screen_text')}",
                f"Environment feedback: {node.metadata.get('feedback')}",
                f"Image evidence ids: {node.metadata.get('image_evidence_ids')}",
                "Do not invent UI elements that are not present in the screen text.",
            ]
        )

    def _apply_reflection_payload(self, node: MemoryNode, payload: dict[str, object]) -> None:
        """把小 VLM 反思 JSON 合并回 failure-reflection 节点。"""

        summary = _optional_str(payload.get("summary"))
        avoid_when = _optional_str(payload.get("avoid_when"))
        recovery_hint = _optional_str(payload.get("recovery_hint"))
        failure_mode = _optional_str(payload.get("failure_mode"))
        confidence = payload.get("confidence")
        if summary:
            node.summary = summary
        if avoid_when:
            node.metadata["avoid_when"] = avoid_when
        if recovery_hint:
            node.metadata["recovery_hint"] = recovery_hint
            node.action_hints = [recovery_hint]
        if failure_mode:
            node.failure_modes = [failure_mode]
        if isinstance(confidence, (int, float)):
            node.confidence = min(0.95, max(0.05, float(confidence)))
        node.metadata["reflection_enhanced"] = True

    def _merge_failure_reflections(self) -> None:
        """合并相似失败反思，形成更稳定的错误经验。

        合并键不是简单 task_id，而是 action type + avoid_when 文本相似度。
        这样相同 UI 阶段的重复错误会累计 evidence，而不同阶段的失败仍保留
        独立反思。
        """

        reflections = [node for node in self.store.nodes.values() if node.kind == "failure-reflection"]
        for node in list(reflections):
            if node.node_id not in self.store.nodes:
                continue
            target = self._find_merge_target(node)
            if target is None:
                continue
            self._merge_reflection_into(target, node)
            self._drop_nodes({node.node_id})

    def _find_merge_target(self, node: MemoryNode) -> MemoryNode | None:
        node_action = str(node.metadata.get("bad_action_type") or "")
        node_tokens = tokenize(str(node.metadata.get("avoid_when") or node.summary))
        candidates = [
            item
            for item in self.store.nodes.values()
            if item.node_id != node.node_id
            and item.kind == "failure-reflection"
            and str(item.metadata.get("bad_action_type") or "") == node_action
        ]
        best: tuple[float, MemoryNode] | None = None
        for candidate in candidates:
            score = jaccard(node_tokens, tokenize(str(candidate.metadata.get("avoid_when") or candidate.summary)))
            if score >= self.config.reflection_merge_threshold and (best is None or score > best[0]):
                best = (score, candidate)
        return best[1] if best else None

    def _merge_reflection_into(self, target: MemoryNode, source: MemoryNode) -> None:
        target.evidence_ids = _dedupe([*target.evidence_ids, *source.evidence_ids])
        target.failure_modes = _dedupe([*target.failure_modes, *source.failure_modes])[-6:]
        target.action_hints = _dedupe([*target.action_hints, *source.action_hints])[-4:]
        target.tags = _dedupe([*target.tags, *source.tags])
        target.confidence = min(0.95, max(target.confidence, source.confidence) + 0.02)
        merged_count = int(target.metadata.get("merged_count", 1)) + int(source.metadata.get("merged_count", 1))
        target.metadata["merged_count"] = merged_count
        target.metadata["merged_reflection_ids"] = _dedupe(
            [*target.metadata.get("merged_reflection_ids", []), source.node_id]
        )
        target.metadata["image_evidence_ids"] = _dedupe(
            [
                *target.metadata.get("image_evidence_ids", []),
                *source.metadata.get("image_evidence_ids", []),
            ]
        )

    def _apply_forgetting_policy(self) -> None:
        """主动遗忘低价值节点。

        当前策略非常保守：优先删除未被反思节点引用、未被检索使用的
        image-evidence；必要时再删除低置信、未使用、失败的 trajectory。
        strategy/pattern/reflection 默认不删，避免丢掉抽象经验和错误反思。
        """

        overflow = len(self.store.nodes) - self.config.max_nodes
        if overflow <= 0:
            return
        protected_images = {
            image_id
            for node in self.store.nodes.values()
            if node.kind == "failure-reflection"
            for image_id in node.metadata.get("image_evidence_ids", [])
        }
        candidates = sorted(
            [
                node
                for node in self.store.nodes.values()
                if node.kind == "image-evidence" and node.use_count == 0 and node.node_id not in protected_images
            ],
            key=lambda node: (node.confidence, node.metadata.get("step_index", 0)),
        )
        to_drop = {node.node_id for node in candidates[:overflow]}
        overflow -= len(to_drop)
        if overflow > 0:
            trajectory_candidates = sorted(
                [
                    node
                    for node in self.store.nodes.values()
                    if node.kind == "trajectory" and node.use_count == 0 and node.success_count == 0 and node.confidence < 0.5
                ],
                key=lambda node: node.confidence,
            )
            to_drop.update(node.node_id for node in trajectory_candidates[:overflow])
        if to_drop:
            self._drop_nodes(to_drop)

    def _drop_nodes(self, node_ids: set[str]) -> None:
        """从节点表、边和 evidence_ids 中删除节点引用。"""

        for node_id in node_ids:
            self.store.nodes.pop(node_id, None)
        for src in list(self.store.edges):
            if src in node_ids:
                del self.store.edges[src]
                continue
            self.store.edges[src] = {dst for dst in self.store.edges[src] if dst not in node_ids}
        for node in self.store.nodes.values():
            node.evidence_ids = [evidence_id for evidence_id in node.evidence_ids if evidence_id not in node_ids]
            image_ids = node.metadata.get("image_evidence_ids")
            if isinstance(image_ids, list):
                node.metadata["image_evidence_ids"] = [image_id for image_id in image_ids if image_id not in node_ids]


def _extract_json_object(text: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    text = text.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
                break
            except json.JSONDecodeError:
                continue
        else:
            raise
    if not isinstance(value, dict):
        raise ValueError("reflection output must be a JSON object")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out
