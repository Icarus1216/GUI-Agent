"""功能: 提供独立 memory controller，统一管理记忆更新、反思增强、合并抽象和主动遗忘。
上游依赖: 依赖 core Trajectory、memory.store 和可选小 VLM 反思客户端。
下游依赖: VLMGuiAgent 通过该 controller 访问 memory，而不是直接把更新策略写死在 store 中。
"""

from __future__ import annotations

import json
import inspect
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

    def reflect(self, prompt: str, image_paths: list[str] | None = None) -> str:
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
        self._consolidate_skills_from_reflections()
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
            image_paths = self._reflection_image_paths(node)
            try:
                payload = _extract_json_object(self._call_reflection_client(prompt, image_paths))
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
                "Required JSON keys: summary, error_type, root_cause, avoid_when, diagnostic_check, recovery_hint, failure_mode, confidence.",
                "error_type must be one of: observation, decision, execution.",
                f"Initial error type: {node.metadata.get('error_type')}",
                f"Initial root cause: {node.metadata.get('root_cause')}",
                f"Task id: {node.metadata.get('task_id')}",
                f"Bad action: {node.metadata.get('bad_action')}",
                f"Screen before action: {node.metadata.get('before_screen_text')}",
                f"Screen after action: {node.metadata.get('after_screen_text')}",
                f"Environment feedback: {node.metadata.get('feedback')}",
                f"Image evidence ids: {node.metadata.get('image_evidence_ids')}",
                f"Image evidence paths: {self._reflection_image_paths(node)}",
                "Do not invent UI elements that are not present in the screen text.",
            ]
        )

    def _reflection_image_paths(self, node: MemoryNode) -> list[str]:
        """从 failure-reflection 的 evidence id 解析可传给小 VLM 的截图路径。"""

        paths = []
        for image_id in node.metadata.get("image_evidence_ids", []):
            image_node = self.store.nodes.get(str(image_id))
            if image_node is None:
                continue
            image_path = image_node.metadata.get("image_path")
            if image_path:
                paths.append(str(image_path))
        return paths

    def _call_reflection_client(self, prompt: str, image_paths: list[str]) -> str:
        """兼容新旧 reflection client 签名。

        真实小 VLM 客户端可以接收 image_paths；早期测试和规则 mock 只接收
        prompt，因此这里保留 fallback。
        """

        assert self.reflection_client is not None
        signature = inspect.signature(self.reflection_client.reflect)
        if "image_paths" in signature.parameters:
            return self.reflection_client.reflect(prompt, image_paths=image_paths)
        return self.reflection_client.reflect(prompt)

    def _apply_reflection_payload(self, node: MemoryNode, payload: dict[str, object]) -> None:
        """把小 VLM 反思 JSON 合并回 failure-reflection 节点。"""

        summary = _optional_str(payload.get("summary"))
        error_type = _optional_str(payload.get("error_type"))
        root_cause = _optional_str(payload.get("root_cause"))
        avoid_when = _optional_str(payload.get("avoid_when"))
        diagnostic_check = _optional_str(payload.get("diagnostic_check"))
        recovery_hint = _optional_str(payload.get("recovery_hint"))
        failure_mode = _optional_str(payload.get("failure_mode"))
        confidence = payload.get("confidence")
        if summary:
            node.summary = summary
        if error_type in {"observation", "decision", "execution"}:
            node.metadata["error_type"] = error_type
            node.metadata["learned_skill_id"] = _skill_id_for_error_type(error_type, str(node.metadata.get("bad_action_type") or ""))
        if root_cause:
            node.metadata["root_cause"] = root_cause
        if avoid_when:
            node.metadata["avoid_when"] = avoid_when
        if diagnostic_check:
            node.metadata["diagnostic_check"] = diagnostic_check
        if recovery_hint:
            node.metadata["recovery_hint"] = recovery_hint
            node.metadata["recovery_policy"] = recovery_hint
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
        node_error_type = str(node.metadata.get("error_type") or "")
        node_tokens = tokenize(str(node.metadata.get("avoid_when") or node.summary))
        candidates = [
            item
            for item in self.store.nodes.values()
            if item.node_id != node.node_id
            and item.kind == "failure-reflection"
            and str(item.metadata.get("bad_action_type") or "") == node_action
            and str(item.metadata.get("error_type") or "") == node_error_type
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
        target.metadata["root_causes"] = _dedupe(
            [
                *target.metadata.get("root_causes", []),
                str(target.metadata.get("root_cause") or ""),
                str(source.metadata.get("root_cause") or ""),
            ]
        )

    def _consolidate_skills_from_reflections(self) -> None:
        """把错误反思蒸馏成可复用 GUI skill 节点。

        Skill 是长期 prompt 最应该优先检索的经验：它不记录完整历史，而是
        保存适用条件、诊断检查和恢复步骤，并链接回支持它的失败证据。
        """

        reflections = [node for node in self.store.nodes.values() if node.kind == "failure-reflection"]
        for reflection in reflections:
            skill_id = str(reflection.metadata.get("learned_skill_id") or "")
            if not skill_id:
                continue
            skill = self.store.nodes.get(skill_id)
            if skill is None:
                skill = self._new_skill_from_reflection(skill_id, reflection)
                self.store.nodes[skill.node_id] = skill
            else:
                source_ids = self._reflection_source_ids(reflection)
                if all(source_id in skill.metadata.get("source_reflection_ids", []) for source_id in source_ids):
                    self.store.edges.setdefault(skill.node_id, set()).add(reflection.node_id)
                    continue
                self._merge_reflection_into_skill(skill, reflection)
            self.store.edges.setdefault(skill.node_id, set()).add(reflection.node_id)
            reflection.metadata["learned_skill_id"] = skill.node_id

    def _new_skill_from_reflection(self, skill_id: str, reflection: MemoryNode) -> MemoryNode:
        error_type = str(reflection.metadata.get("error_type") or "decision")
        skill_name = _skill_name(skill_id)
        recovery = str(reflection.metadata.get("recovery_policy") or reflection.metadata.get("recovery_hint") or "")
        diagnostic = str(reflection.metadata.get("diagnostic_check") or "")
        avoid_when = str(reflection.metadata.get("avoid_when") or "")
        return MemoryNode(
            node_id=skill_id,
            level=2,
            kind="skill",
            summary=f"{skill_name}: handle {error_type} GUI errors by diagnosing state and applying the recovery policy.",
            evidence_ids=[*self._reflection_source_ids(reflection), *reflection.evidence_ids],
            preconditions=[avoid_when] if avoid_when else [],
            action_hints=[recovery] if recovery else [],
            expected_effects=["Future GUI decisions avoid the reflected failure mode."],
            failure_modes=list(reflection.failure_modes),
            confidence=min(0.9, max(0.55, reflection.confidence + 0.05)),
            success_count=0,
            tags=_dedupe(["skill", error_type, *reflection.tags]),
            metadata={
                "skill_name": skill_name,
                "error_type": error_type,
                "applies_when": avoid_when,
                "procedure": recovery,
                "check_items": [diagnostic] if diagnostic else [],
                "recovery_steps": [recovery] if recovery else [],
                "source_reflection_ids": self._reflection_source_ids(reflection),
                "source_count": len(self._reflection_source_ids(reflection)),
            },
        )

    def _merge_reflection_into_skill(self, skill: MemoryNode, reflection: MemoryNode) -> None:
        recovery = str(reflection.metadata.get("recovery_policy") or reflection.metadata.get("recovery_hint") or "")
        diagnostic = str(reflection.metadata.get("diagnostic_check") or "")
        avoid_when = str(reflection.metadata.get("avoid_when") or "")
        reflection_source_ids = self._reflection_source_ids(reflection)
        skill.evidence_ids = _dedupe([*skill.evidence_ids, *reflection_source_ids, *reflection.evidence_ids])
        if avoid_when:
            skill.preconditions = _dedupe([*skill.preconditions, avoid_when])[-6:]
        if recovery:
            skill.action_hints = _dedupe([*skill.action_hints, recovery])[-6:]
        skill.failure_modes = _dedupe([*skill.failure_modes, *reflection.failure_modes])[-8:]
        skill.tags = _dedupe([*skill.tags, *reflection.tags])
        skill.confidence = min(0.95, max(skill.confidence, reflection.confidence) + 0.03)
        source_ids = _dedupe([*skill.metadata.get("source_reflection_ids", []), *reflection_source_ids])
        skill.metadata["source_reflection_ids"] = source_ids
        skill.metadata["source_count"] = len(source_ids)
        if avoid_when:
            skill.metadata["applies_when"] = "; ".join(_dedupe([str(skill.metadata.get("applies_when") or ""), avoid_when])[-3:])
        if recovery:
            skill.metadata["procedure"] = recovery
            skill.metadata["recovery_steps"] = _dedupe([*skill.metadata.get("recovery_steps", []), recovery])[-5:]
        if diagnostic:
            skill.metadata["check_items"] = _dedupe([*skill.metadata.get("check_items", []), diagnostic])[-5:]

    def _reflection_source_ids(self, reflection: MemoryNode) -> list[str]:
        return _dedupe([reflection.node_id, *reflection.metadata.get("merged_reflection_ids", [])])

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


def _skill_id_for_error_type(error_type: str, action_type: str) -> str:
    if error_type == "observation":
        return "skill:verify_visible_state_before_terminal_action"
    if error_type == "execution":
        if action_type in {"type", "paste"}:
            return "skill:ground_text_entry_before_typing"
        return "skill:ground_action_target_and_verify_effect"
    if action_type == "done":
        return "skill:check_prerequisites_before_done"
    if action_type == "fail":
        return "skill:recover_before_failing"
    return "skill:prerequisite_aware_next_action"


def _skill_name(skill_id: str) -> str:
    return skill_id.removeprefix("skill:").replace("_", " ")
