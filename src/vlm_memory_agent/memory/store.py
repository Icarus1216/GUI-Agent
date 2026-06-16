"""功能: 实现层次化经验记忆图的保存、加载、检索和从 trajectory 自动固化经验。
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


@dataclass(slots=True)
class MemoryNode:
    """层次记忆图里的一个节点。

    level=0 表示具体 trajectory，level=1 表示从相似任务中归纳出的
    state-action pattern，level=2 表示更抽象的通用策略。字段按 prompt
    需要组织：preconditions/action_hints/effects/failure_modes 会直接
    进入 VLM 上下文。
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
        ]
        return " ".join(parts)


class HierarchicalMemoryStore:
    """Small hierarchical experience memory graph.

    L0 nodes are concrete trajectories. L1 nodes are reusable state-action
    patterns. L2 nodes are higher-level strategies. The consolidation is
    deliberately simple and inspectable so methods can replace it later.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
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
        self.nodes[leaf.node_id] = leaf
        pattern = self._consolidate_pattern(leaf)
        strategy = self._consolidate_strategy(pattern)
        self._link(pattern.node_id, leaf.node_id)
        self._link(strategy.node_id, pattern.node_id)
        if self.path:
            self.save(self.path)
        return [leaf, pattern, strategy]

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
        self.nodes = {item["node_id"]: MemoryNode(**item) for item in payload.get("nodes", [])}
        self.edges = {src: set(dst) for src, dst in payload.get("edges", {}).items()}

    def _trajectory_leaf(self, trajectory: Trajectory) -> MemoryNode:
        """把原始 trajectory 压缩成 L0 evidence node。"""

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
        )
        return node

    def _consolidate_pattern(self, leaf: MemoryNode) -> MemoryNode:
        """把 L0 trajectory 合并到相似的 L1 pattern，或创建新 pattern。

        合并依据是任务 token tag 的 Jaccard 相似度。这个阈值 intentionally
        simple：它更适合当前小规模研究脚手架，能让每次合并都容易人工检查。
        """

        tags = set(leaf.tags)
        existing = [
            node
            for node in self.nodes.values()
            if node.level == 1 and jaccard(tags, set(node.tags)) >= 0.35
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
