"""功能: 实现层次化经验记忆图、多模态图像证据节点的保存、加载、检索和从 trajectory 自动固化经验。
上游依赖: 依赖 core Trajectory/StepStatus、JSON 文件路径和简单 token/Jaccard 相似度。
下游依赖: VLMGuiAgent、API server、BenchmarkRunner、CLI 和测试用它提供长期记忆上下文。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vlm_memory_agent.core.types import StepStatus, Trajectory


def tokenize(text: str) -> set[str]:
    """把任务、屏幕文本和经验摘要转成轻量检索 token。

    当前实现使用可解释的词集合而不是 embedding，目的是让早期实验中
    memory 命中逻辑完全透明。后续可以在保持 `retrieve()` 接口不变的
    情况下替换成向量索引或混合检索。
    """

    return {tok for tok in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(tok) > 1}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _json_safe_value(value: Any) -> Any:
    """把环境 metadata 中的第三方对象转成 JSON-safe 值。"""

    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


@dataclass(slots=True)
class MemoryNode:
    """层次记忆图里的一个节点。

    level=0 表示具体 trajectory，level=1 表示从相似任务中归纳出的
    state-action pattern，level=2 表示更抽象的通用策略。`kind=image-evidence`
    的节点也是 level=0，但它指向某一步关键截图，作为 trajectory/pattern
    的可追溯多模态证据。字段按 prompt 需要组织：preconditions/action_hints/
    effects/failure_modes 会直接进入 VLM 上下文；metadata 保存截图路径、
    step index、before/after phase、reward 等结构化证据。`kind=failure-reflection`
    的节点把错误历史轨迹反思成 avoid/recover 经验，并链接到对应截图证据。
    """

    node_id: str
    level: int
    kind: str
    summary: str
    evidence_ids: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    action_hints: list[str] = field(default_factory=list)
    expected_effects: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    confidence: float = 0.5
    success_count: int = 0
    use_count: int = 0
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def text_for_retrieval(self) -> str:
        """拼接用于检索的文本视图。

        检索不使用 node_id 和 evidence_ids，因为这些字段对当前屏幕匹配
        没有语义帮助；更多权重来自 summary、前置条件、动作提示和失败模式。
        """

        parts = [
            self.kind,
            self.summary,
            " ".join(self.preconditions),
            " ".join(self.action_hints),
            " ".join(self.expected_effects),
            " ".join(self.failure_modes),
            " ".join(self.tags),
            " ".join(
                str(value)
                for key, value in self.metadata.items()
                if key in {"image_path", "phase", "action_type", "status", "bad_action", "avoid_when", "recovery_hint"}
            ),
        ]
        return " ".join(parts)


class HierarchicalMemoryStore:
    """Small hierarchical multimodal experience memory graph.

    L0 nodes are concrete trajectories. L1 nodes are reusable state-action
    patterns. L2 nodes are higher-level strategies. In addition, L0
    `image-evidence` nodes preserve key screenshots as traceable multimodal
    evidence linked from trajectories and patterns. `failure-reflection` nodes
    turn failed histories into explicit avoid/recover experience. The consolidation is
    deliberately simple and inspectable so methods can replace it later.
    """

    def __init__(self, path: str | Path | None = None, max_image_evidence_per_trajectory: int = 6):
        self.path = Path(path) if path else None
        self.max_image_evidence_per_trajectory = max_image_evidence_per_trajectory
        self.nodes: dict[str, MemoryNode] = {}
        self.edges: dict[str, set[str]] = {}
        if self.path and self.path.exists():
            self.load(self.path)

    def retrieve(self, query: str, k: int = 4) -> list[MemoryNode]:
        """返回与当前任务/屏幕最相关的记忆节点。

        打分 = Jaccard(query_tokens, node_tokens) * sqrt(confidence)。这样
        可以让相似文本仍是主因，同时让多次成功固化出的高置信 pattern
        更容易进入 prompt。命中节点会增加 use_count，便于后续分析记忆
        是否真的被使用。
        """

        q_tokens = tokenize(query)
        scored: list[tuple[float, MemoryNode]] = []
        for node in self.nodes.values():
            score = jaccard(q_tokens, tokenize(node.text_for_retrieval()))
            score *= math.sqrt(max(node.confidence, 0.01))
            if score > 0:
                scored.append((score, node))
        scored.sort(key=lambda item: (item[0], item[1].confidence), reverse=True)
        for _, node in scored[:k]:
            node.use_count += 1
        return [node for _, node in scored[:k]]

    def update_from_trajectory(self, trajectory: Trajectory) -> list[MemoryNode]:
        """从完整 trajectory 固化三层记忆。

        每次 episode 都先保存一个 L0 叶子节点，再尝试合并到已有 L1
        pattern，最后挂到固定的 L2 GUI 策略节点。这个流程比较保守：
        不会删除旧证据，也不会让失败轨迹覆盖成功经验，只会通过 confidence
        和 failure_modes 体现风险。
        """

        leaf = self._trajectory_leaf(trajectory)
        image_nodes = self._image_evidence_nodes(trajectory, leaf.node_id)
        leaf.evidence_ids.extend(node.node_id for node in image_nodes)
        self.nodes[leaf.node_id] = leaf
        for image_node in image_nodes:
            self.nodes[image_node.node_id] = image_node
            self._link(leaf.node_id, image_node.node_id)
        reflection = self._failure_reflection_node(trajectory, leaf, image_nodes)
        if reflection is not None:
            leaf.evidence_ids.append(reflection.node_id)
            self.nodes[reflection.node_id] = reflection
            self._link(leaf.node_id, reflection.node_id)
            for evidence_id in reflection.evidence_ids:
                if evidence_id.startswith("image:"):
                    self._link(reflection.node_id, evidence_id)
        pattern = self._consolidate_pattern(leaf)
        strategy = self._consolidate_strategy(pattern)
        self._link(pattern.node_id, leaf.node_id)
        for image_node in image_nodes:
            self._link(pattern.node_id, image_node.node_id)
        if reflection is not None:
            self._link(pattern.node_id, reflection.node_id)
        self._link(strategy.node_id, pattern.node_id)
        if self.path:
            self.save(self.path)
        return [leaf, *image_nodes, *([reflection] if reflection is not None else []), pattern, strategy]

    def prompt_context(self, nodes: list[MemoryNode]) -> str:
        """把检索到的节点格式化为 prompt 上下文。

        这里显式列出 preconditions/action_hints/effects/failures，而不是
        只给 summary，目的是让 VLM 能判断当前 observation 是否真的满足
        经验适用条件，并看到过去失败的触发方式。
        """

        if not nodes:
            return "No relevant long-term memory."
        blocks = []
        for node in nodes:
            if node.kind == "image-evidence":
                blocks.append(self._image_prompt_block(node))
                continue
            if node.kind == "failure-reflection":
                blocks.append(self._failure_prompt_block(node))
                continue
            blocks.append(
                "\n".join(
                    [
                        f"[{node.kind}:{node.node_id} confidence={node.confidence:.2f}]",
                        f"Summary: {node.summary}",
                        f"Preconditions: {', '.join(node.preconditions) or 'n/a'}",
                        f"Action hints: {', '.join(node.action_hints) or 'n/a'}",
                        f"Expected effects: {', '.join(node.expected_effects) or 'n/a'}",
                        f"Failure modes: {', '.join(node.failure_modes) or 'n/a'}",
                    ]
                )
            )
        return "\n\n".join(blocks)

    def save(self, path: str | Path) -> None:
        """把记忆图持久化为 JSON。

        edges 内部用 set 去重，落盘时转成排序 list，保证文件稳定可 diff。
        """

        payload = {
            "nodes": [asdict(node) for node in self.nodes.values()],
            "edges": {src: sorted(dst) for src, dst in self.edges.items()},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.nodes = {item["node_id"]: MemoryNode(**self._memory_node_payload(item)) for item in payload.get("nodes", [])}
        self.edges = {src: set(dst) for src, dst in payload.get("edges", {}).items()}

    def _trajectory_leaf(self, trajectory: Trajectory) -> MemoryNode:
        """把原始 trajectory 压缩成 L0 evidence node。

        summary/action_hints 是给检索和 prompt 使用的抽象视图；metadata 中的
        `interleaved_steps` 保留逐步交错模态原始证据：动作前 GUI 状态、
        action、动作后 GUI 状态，以及环境/evaluator 对这一步的 verify。
        """

        status = "success" if trajectory.success else "failure"
        action_trace = " -> ".join(step.action.compact() for step in trajectory.steps)
        feedback = " | ".join(step.feedback for step in trajectory.steps[-3:])
        node = MemoryNode(
            node_id=f"traj:{trajectory.task_id}:{len(self.nodes)}",
            level=0,
            kind="trajectory",
            summary=f"{status} trajectory for: {trajectory.task}. Actions: {action_trace}. Feedback: {feedback}",
            evidence_ids=[trajectory.task_id],
            preconditions=[self._trim(trajectory.steps[0].observation.screen_text, 2000)] if trajectory.steps else [],
            action_hints=[step.action.compact() for step in trajectory.steps],
            expected_effects=[step.feedback for step in trajectory.steps if step.status == StepStatus.SUCCESS],
            failure_modes=[step.feedback for step in trajectory.steps if step.status == StepStatus.FAILED],
            confidence=0.7 if trajectory.success else 0.45,
            success_count=1 if trajectory.success else 0,
            tags=list(tokenize(trajectory.task))[:8],
            metadata={
                "trajectory_schema": "interleaved_multimodal_v1",
                "task_id": trajectory.task_id,
                "task": trajectory.task,
                "success": trajectory.success,
                "step_count": len(trajectory.steps),
                "interleaved_steps": [
                    self._interleaved_step_payload(index, step) for index, step in enumerate(trajectory.steps)
                ],
            },
        )
        return node

    def _interleaved_step_payload(self, index: int, step: Any) -> dict[str, Any]:
        """保留单步交错模态 trajectory 原始证据。

        `verification.verdict` 区分 progress/correct/wrong：RUNNING step
        代表动作未被判错但仍需继续，SUCCESS/FAILED 才分别写成明确正确或
        错误，具体原因保存在 feedback/metadata 中。
        """

        verdict = "correct" if step.status == StepStatus.SUCCESS else "wrong" if step.status == StepStatus.FAILED else "progress"
        return {
            "step_index": index,
            "before": self._observation_evidence(step.observation),
            "action": step.action.to_dict(),
            "after": self._observation_evidence(step.next_observation) if step.next_observation else None,
            "verification": {
                "status": step.status.value,
                "verdict": verdict,
                "correct": True if step.status == StepStatus.SUCCESS else False if step.status == StepStatus.FAILED else None,
                "failed": step.status == StepStatus.FAILED,
                "reward": step.reward,
                "feedback": step.feedback,
                "metadata": _json_safe_value(step.metadata),
            },
        }

    def _observation_evidence(self, observation: Any) -> dict[str, Any]:
        """把 observation 转成适合嵌入 trajectory node 的 GUI 状态证据。"""

        return observation.to_dict()

    def _image_evidence_nodes(self, trajectory: Trajectory, leaf_id: str) -> list[MemoryNode]:
        """从 trajectory 中抽取关键截图并生成 image-evidence 节点。

        选择策略保持保守和可解释：
        - 首帧 before screenshot，记录任务起始状态；
        - 每个非 wait 动作后的 after screenshot，记录动作效果；
        - 任何成功/失败终态 after screenshot，记录 evaluator 结论对应画面。

        通过 screenshot_path 去重，并限制每条 trajectory 的图像证据数量，
        避免长程任务把 memory JSON 膨胀成逐帧日志。
        """

        candidates: list[tuple[int, str, str, Any]] = []
        for index, step in enumerate(trajectory.steps):
            if index == 0 and step.observation.screenshot_path:
                candidates.append((index, "before", step.observation.screenshot_path, step))
            if step.next_observation and step.next_observation.screenshot_path:
                is_terminal = step.status != StepStatus.RUNNING
                is_action_effect = step.action.action_type != "wait"
                if is_action_effect or is_terminal:
                    candidates.append((index, "after", step.next_observation.screenshot_path, step))

        nodes: list[MemoryNode] = []
        seen_paths: set[str] = set()
        for index, phase, screenshot_path, step in candidates:
            if screenshot_path in seen_paths:
                continue
            seen_paths.add(screenshot_path)
            if len(nodes) >= self.max_image_evidence_per_trajectory:
                break
            status = step.status.value
            action = step.action.compact()
            screen_text = step.next_observation.screen_text if phase == "after" and step.next_observation else step.observation.screen_text
            node_id = f"image:{trajectory.task_id}:{len(self.nodes)}:{len(nodes)}"
            nodes.append(
                MemoryNode(
                    node_id=node_id,
                    level=0,
                    kind="image-evidence",
                    summary=(
                        f"Image evidence for trajectory {leaf_id}, step {index} {phase}. "
                        f"Action: {action}. Status: {status}. Screenshot: {screenshot_path}"
                    ),
                    evidence_ids=[leaf_id],
                    preconditions=[self._trim(screen_text, 1200)] if screen_text else [],
                    action_hints=[action],
                    expected_effects=[step.feedback] if step.status == StepStatus.SUCCESS and step.feedback else [],
                    failure_modes=[step.feedback] if step.status == StepStatus.FAILED and step.feedback else [],
                    confidence=0.72 if step.status == StepStatus.SUCCESS else 0.55,
                    success_count=1 if step.status == StepStatus.SUCCESS else 0,
                    tags=self._dedupe([*list(tokenize(trajectory.task))[:8], "image", "screenshot", phase, step.action.action_type, status]),
                    metadata={
                        "image_path": screenshot_path,
                        "trajectory_node_id": leaf_id,
                        "task_id": trajectory.task_id,
                        "step_index": index,
                        "phase": phase,
                        "action": action,
                        "action_type": step.action.action_type,
                        "status": status,
                        "reward": step.reward,
                        "feedback": step.feedback,
                    },
                )
            )
        return nodes

    def _failure_reflection_node(
        self,
        trajectory: Trajectory,
        leaf: MemoryNode,
        image_nodes: list[MemoryNode],
    ) -> MemoryNode | None:
        """把失败历史轨迹反思成可复用的 avoid/recover 经验。

        失败 memory 的关键不是“这条轨迹失败了”，而是保留：
        - 失败发生前的屏幕状态；
        - 具体错误/无效动作；
        - 环境反馈；
        - 下一次遇到相似状态时应避免什么、尝试什么恢复动作；
        - 能追溯到的截图证据。
        """

        failure_steps = [index for index, step in enumerate(trajectory.steps) if step.status == StepStatus.FAILED]
        if not failure_steps and trajectory.success:
            return None
        focus_index = failure_steps[-1] if failure_steps else max(0, len(trajectory.steps) - 1)
        if not trajectory.steps:
            return None
        step = trajectory.steps[focus_index]
        before_text = self._trim(step.observation.screen_text, 1400)
        after_text = self._trim(step.next_observation.screen_text, 1400) if step.next_observation else ""
        bad_action = step.action.compact()
        feedback = step.feedback or "No explicit environment feedback."
        avoid_when = self._reflection_avoid_when(step)
        recovery_hint = self._reflection_recovery_hint(step, trajectory)
        related_images = self._related_image_ids(image_nodes, focus_index)
        evidence_ids = self._dedupe([leaf.node_id, *related_images])
        tags = self._dedupe(
            [
                *list(tokenize(trajectory.task))[:8],
                "failure",
                "reflection",
                step.action.action_type,
                step.status.value,
                *list(tokenize(feedback))[:6],
            ]
        )
        return MemoryNode(
            node_id=f"reflection:{trajectory.task_id}:{len(self.nodes)}",
            level=1,
            kind="failure-reflection",
            summary=(
                f"Failure reflection for task `{trajectory.task}`. In a state like `{avoid_when}`, "
                f"the action `{bad_action}` led to failure/poor progress: {feedback}"
            ),
            evidence_ids=evidence_ids,
            preconditions=[before_text] if before_text else [],
            action_hints=[recovery_hint],
            expected_effects=[],
            failure_modes=[f"Avoid `{bad_action}` when {avoid_when}. Feedback: {feedback}"],
            confidence=0.68,
            success_count=0,
            tags=tags,
            metadata={
                "trajectory_node_id": leaf.node_id,
                "task_id": trajectory.task_id,
                "failed_step_index": focus_index,
                "bad_action": bad_action,
                "bad_action_type": step.action.action_type,
                "avoid_when": avoid_when,
                "recovery_hint": recovery_hint,
                "feedback": feedback,
                "before_screen_text": before_text,
                "after_screen_text": after_text,
                "image_evidence_ids": related_images,
                "terminal_failure": step.status == StepStatus.FAILED,
            },
        )

    def _consolidate_pattern(self, leaf: MemoryNode) -> MemoryNode:
        """把 L0 trajectory 合并到相似的 L1 pattern，或创建新 pattern。

        合并依据是任务 token tag 的 Jaccard 相似度。这个阈值 intentionally
        simple：它更适合当前小规模研究脚手架，能让每次合并都容易人工检查。
        """

        tags = set(leaf.tags)
        existing = [
            node
            for node in self.nodes.values()
            if node.kind == "state-action-pattern" and node.level == 1 and jaccard(tags, set(node.tags)) >= 0.35
        ]
        if existing:
            node = existing[0]
            node.evidence_ids.extend(eid for eid in leaf.evidence_ids if eid not in node.evidence_ids)
            node.action_hints = self._dedupe(node.action_hints + leaf.action_hints)[-8:]
            node.failure_modes = self._dedupe(node.failure_modes + leaf.failure_modes)[-6:]
            node.expected_effects = self._dedupe(node.expected_effects + leaf.expected_effects)[-6:]
            node.success_count += leaf.success_count
            node.confidence = min(0.95, node.confidence + (0.04 if leaf.success_count else -0.03))
            return node
        node_id = f"pattern:{len([n for n in self.nodes.values() if n.level == 1])}"
        node = MemoryNode(
            node_id=node_id,
            level=1,
            kind="state-action-pattern",
            summary=f"Reusable pattern for tasks like: {leaf.summary[:180]}",
            evidence_ids=list(leaf.evidence_ids),
            preconditions=list(leaf.preconditions),
            action_hints=list(leaf.action_hints[-6:]),
            expected_effects=list(leaf.expected_effects),
            failure_modes=list(leaf.failure_modes),
            confidence=leaf.confidence,
            success_count=leaf.success_count,
            tags=leaf.tags,
        )
        self.nodes[node.node_id] = node
        return node

    def _consolidate_strategy(self, pattern: MemoryNode) -> MemoryNode:
        """维护一个跨任务的 L2 GUI 策略节点。

        当前策略强调“执行后验证状态、可见成功后再 done”，这是 GUI agent
        评测中最常见的失败来源之一。它把所有 pattern 的 evidence 聚合到
        同一策略下，便于之后替换为更复杂的策略聚类。
        """

        key = "strategy:verify-after-action"
        if key not in self.nodes:
            self.nodes[key] = MemoryNode(
                node_id=key,
                level=2,
                kind="strategy",
                summary="For GUI tasks, act on visible UI elements, verify that the screen state changed, and call done only after the target artifact is open or the goal state is visible.",
                preconditions=["A GUI observation with task instruction and visible elements."],
                action_hints=["click relevant element", "type requested value", "verify result", "done when goal is visible"],
                expected_effects=["screen state matches task goal"],
                failure_modes=["assuming success without visual/state evidence"],
                confidence=0.8,
                tags=["gui", "verify", "state", "done"],
            )
        strategy = self.nodes[key]
        strategy.evidence_ids.extend(eid for eid in pattern.evidence_ids if eid not in strategy.evidence_ids)
        strategy.success_count += pattern.success_count
        return strategy

    def _link(self, src: str, dst: str) -> None:
        self.edges.setdefault(src, set()).add(dst)

    def _image_prompt_block(self, node: MemoryNode) -> str:
        """把图像证据节点格式化为 prompt 可读的可追溯引用。"""

        image_path = node.metadata.get("image_path", "unknown")
        return "\n".join(
            [
                f"[{node.kind}:{node.node_id} confidence={node.confidence:.2f}]",
                f"Summary: {node.summary}",
                f"Image path: {image_path}",
                f"Step: {node.metadata.get('step_index', 'n/a')} phase={node.metadata.get('phase', 'n/a')} status={node.metadata.get('status', 'n/a')}",
                f"Action: {node.metadata.get('action', ', '.join(node.action_hints) or 'n/a')}",
                f"Screen evidence: {', '.join(node.preconditions) or 'n/a'}",
                f"Feedback: {node.metadata.get('feedback') or 'n/a'}",
            ]
        )

    def _failure_prompt_block(self, node: MemoryNode) -> str:
        """把失败反思节点格式化为 prompt 中的显式避错经验。"""

        return "\n".join(
            [
                f"[{node.kind}:{node.node_id} confidence={node.confidence:.2f}]",
                f"Summary: {node.summary}",
                f"Avoid when: {node.metadata.get('avoid_when') or 'n/a'}",
                f"Bad action: {node.metadata.get('bad_action') or 'n/a'}",
                f"Failure mode: {', '.join(node.failure_modes) or 'n/a'}",
                f"Recovery hint: {node.metadata.get('recovery_hint') or ', '.join(node.action_hints) or 'n/a'}",
                f"Evidence ids: {', '.join(node.evidence_ids) or 'n/a'}",
            ]
        )

    def _reflection_avoid_when(self, step: Any) -> str:
        """生成失败状态的短描述，用于 prompt 中判断适用条件。"""

        elements = ", ".join(element.element_id for element in step.observation.ui_elements[:8])
        screen = self._trim(step.observation.screen_text, 240).replace("\n", " ")
        if elements:
            return f"visible elements include [{elements}] and screen says `{screen}`"
        return f"screen says `{screen}`"

    def _reflection_recovery_hint(self, step: Any, trajectory: Trajectory) -> str:
        """根据失败动作和轨迹上下文生成保守恢复建议。

        当前不调用 LLM 做反思，避免引入额外不确定性；先用规则生成可解释的
        recovery hint，后续可以替换成离线反思模型。
        """

        if step.action.action_type == "done":
            return "Do not call done until the goal state is visibly achieved or evaluator evidence is present."
        if step.action.action_type == "fail":
            return "Before failing, inspect visible UI, wait once if the screen may still update, or try a reversible navigation/search action."
        if step.action.action_type in {"click", "double_click", "right_click"}:
            return "Verify the target is visible and clickable; if there is no state change, use a more specific element or coordinate from the screenshot."
        if step.action.action_type in {"type", "paste"}:
            return "Verify the intended text field is focused and visible before entering text; clear or select the field if stale text remains."
        if step.action.action_type in {"hotkey", "press"}:
            return "Use keyboard actions only when focus is known; otherwise click the target control first."
        if step.action.action_type == "scroll":
            return "Scroll only when the needed element is off-screen; re-check the visible state after scrolling."
        previous_actions = " -> ".join(item.action.compact() for item in trajectory.steps[max(0, len(trajectory.steps) - 3) :])
        return f"Backtrack from the recent action sequence and choose a different visible UI target. Recent actions: {previous_actions}"

    def _related_image_ids(self, image_nodes: list[MemoryNode], focus_index: int) -> list[str]:
        """选择最接近失败 step 的图像证据节点 id。"""

        scored = []
        for node in image_nodes:
            step_index = node.metadata.get("step_index")
            if isinstance(step_index, int):
                scored.append((abs(step_index - focus_index), node.node_id))
        return [node_id for _, node_id in sorted(scored)[:2]]

    def _memory_node_payload(self, item: dict[str, Any]) -> dict[str, Any]:
        """加载旧版 memory JSON 时补齐新增字段。"""

        payload = dict(item)
        payload.setdefault("metadata", {})
        return payload

    def _dedupe(self, values: list[str]) -> list[str]:
        seen = set()
        out = []
        for value in values:
            if value and value not in seen:
                seen.add(value)
                out.append(value)
        return out

    def _trim(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...[truncated]"
