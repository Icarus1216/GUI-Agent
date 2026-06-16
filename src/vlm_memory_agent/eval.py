"""功能: 实现批量评测封装，运行 mock 或 OSWorld 任务并汇总成功率、步数、奖励和轨迹文件。
上游依赖: 依赖 VLMGuiAgent、环境构造器、VLMClient、记忆库和 OSWorld task config 解析工具。
下游依赖: osworld_eval CLI、批量评测脚本和测试使用 BenchmarkRunner/write_report 生成评测报告。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from vlm_memory_agent.agent import AgentConfig, VLMGuiAgent
from vlm_memory_agent.envs.mock_osworld import MockOSWorldEnv
from vlm_memory_agent.envs.osworld_runner import OSWorldEnvConfig, build_osworld_adapter, iter_task_config_paths, load_task_config
from vlm_memory_agent.llm.base import VLMClient
from vlm_memory_agent.memory.store import HierarchicalMemoryStore


@dataclass(slots=True)
class EvalResult:
    """单个 task 的评测摘要。

    只保留聚合报告需要的字段；完整逐步信息写在 trajectory_path 指向的文件。
    """

    task_id: str
    success: bool
    steps: int
    reward: float
    trajectory_path: str | None = None


class BenchmarkRunner:
    """批量评测执行器。

    Runner 复用同一个 agent 和 memory store 跑多个任务，因此 memory 会跨
    task 累积。这符合“带记忆 agent”的研究设定；如果要做 no-memory baseline，
    应为每个 task 使用独立 memory_path 或关闭 memory update。
    """

    def __init__(self, vlm: VLMClient, memory_path: str | Path, max_steps: int = 15, trajectory_dir: str | Path | None = None):
        self.memory = HierarchicalMemoryStore(memory_path)
        self.agent = VLMGuiAgent(vlm=vlm, memory=self.memory, config=AgentConfig(max_steps=max_steps))
        self.trajectory_dir = Path(trajectory_dir) if trajectory_dir else None

    def run_mock_osworld(self, task_ids: list[str]) -> list[EvalResult]:
        """在确定性 mock 环境上跑一组任务 id。"""

        results = []
        for task_id in task_ids:
            env = MockOSWorldEnv()
            trajectory = self.agent.run_episode(env, task_id=task_id)
            results.append(
                EvalResult(
                    task_id=task_id,
                    success=trajectory.success,
                    steps=len(trajectory.steps),
                    reward=sum(step.reward for step in trajectory.steps),
                )
            )
        return results

    def run_osworld(self, env_config: OSWorldEnvConfig) -> EvalResult:
        """构造一个真实 OSWorld env，运行一次 episode，并写出轨迹。"""

        env = build_osworld_adapter(env_config)
        task_id = env_config.task_id or getattr(env, "task_id", None) or Path(env_config.task_config).stem
        try:
            trajectory = self.agent.run_episode(env, task_id=task_id)
        finally:
            env.close()
        trajectory_path = self._write_trajectory(trajectory.to_dict(), task_id)
        return EvalResult(
            task_id=task_id,
            success=trajectory.success,
            steps=len(trajectory.steps),
            reward=sum(step.reward for step in trajectory.steps),
            trajectory_path=str(trajectory_path) if trajectory_path else None,
        )

    def _write_trajectory(self, payload: dict, task_id: str) -> Path | None:
        if self.trajectory_dir is None:
            return None
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        path = self.trajectory_dir / f"{_safe_file_stem(task_id)}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def run_osworld_directory(
        self,
        base_config: OSWorldEnvConfig,
        task_config_path: str | Path,
        limit: int | None = None,
    ) -> list[EvalResult]:
        """遍历 OSWorld task JSON 目录并逐个运行。

        每个 task 都创建独立 adapter/env，但共享同一个 agent memory。
        screenshot_dir 会按 task_id 分目录，避免不同任务的截图相互覆盖。
        """

        results: list[EvalResult] = []
        if base_config.task_id:
            task_items = [(Path(task_config_path), load_task_config(task_config_path, task_id=base_config.task_id), base_config.task_id)]
        else:
            task_items = []
            for path in iter_task_config_paths(task_config_path):
                task_config = load_task_config(path)
                task_items.append((path, task_config, str(task_config.get("id") or path.stem)))
        for path, task_config, task_id in task_items[:limit]:
            config = OSWorldEnvConfig(
                task_config=path,
                task_id=task_id,
                vm_path=base_config.vm_path,
                provider_name=base_config.provider_name,
                region=base_config.region,
                snapshot_name=base_config.snapshot_name,
                action_space=base_config.action_space,
                observation_type=base_config.observation_type,
                screen_width=base_config.screen_width,
                screen_height=base_config.screen_height,
                headless=base_config.headless,
                os_type=base_config.os_type,
                require_a11y_tree=base_config.require_a11y_tree,
                require_terminal=base_config.require_terminal,
                enable_proxy=base_config.enable_proxy,
                client_password=base_config.client_password,
                cache_dir=base_config.cache_dir,
                screenshot_dir=Path(base_config.screenshot_dir) / task_id,
                adapter_action_mode=base_config.adapter_action_mode,
                evaluate_on_done=base_config.evaluate_on_done,
            )
            results.append(self.run_osworld(config))
        return results


def write_report(results: list[EvalResult], path: str | Path) -> None:
    """写 aggregate JSON report。"""

    total = len(results)
    success = sum(1 for result in results if result.success)
    payload = {
        "num_tasks": total,
        "success_rate": success / total if total else 0.0,
        "avg_steps": sum(result.steps for result in results) / total if total else 0.0,
        "avg_reward": sum(result.reward for result in results) / total if total else 0.0,
        "tasks": [asdict(result) for result in results],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _safe_file_stem(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._") or "task"
