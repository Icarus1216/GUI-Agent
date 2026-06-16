"""功能: 实现记忆增强 VLM GUI agent 的感知、检索、决策、执行和 memory controller 更新主循环。
上游依赖: 依赖 core 类型、InteractiveEnv、VLMClient、HierarchicalMemoryStore/MemoryController、ScreenParserTool 和 ActionParser。
下游依赖: CLI、API server、BenchmarkRunner、测试和真实 OSWorld 运行都通过 VLMGuiAgent 编排任务。
"""

from __future__ import annotations

from dataclasses import dataclass

from vlm_memory_agent.core.types import AgentAction, Observation, StepStatus, Trajectory
from vlm_memory_agent.envs.base import InteractiveEnv
from vlm_memory_agent.llm.base import VLMClient
from vlm_memory_agent.memory.controller import MemoryController
from vlm_memory_agent.memory.store import HierarchicalMemoryStore
from vlm_memory_agent.tools.action_parser import ActionParser
from vlm_memory_agent.tools.screen_parser import ScreenParserTool


@dataclass(slots=True)
class AgentConfig:
    """Agent loop 的运行控制参数。

    `max_steps` 控制单个 episode 的最长交互步数，避免模型反复等待或
    无效点击导致任务无限运行；`retrieve_k` 控制进入 prompt 的记忆条数，
    太大会稀释当前 observation，太小则可能错过可复用经验。
    """

    max_steps: int = 15
    retrieve_k: int = 4
    update_memory: bool = True


class VLMGuiAgent:
    """Memory-augmented VLM GUI agent.

    The architecture follows the common GUI-agent loop:
    perceive screen -> retrieve memory -> plan next action with VLM ->
    parse action -> execute in environment -> log trajectory -> consolidate memory.
    """

    def __init__(
        self,
        vlm: VLMClient,
        memory: HierarchicalMemoryStore,
        memory_controller: MemoryController | None = None,
        screen_parser: ScreenParserTool | None = None,
        action_parser: ActionParser | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self.vlm = vlm
        self.memory = memory
        self.memory_controller = memory_controller or MemoryController(memory)
        self.screen_parser = screen_parser or ScreenParserTool()
        self.action_parser = action_parser or ActionParser()
        self.config = config or AgentConfig()

    def run_episode(self, env: InteractiveEnv, task_id: str | None = None) -> Trajectory:
        """运行完整 episode，并把每一步结果串成 trajectory。

        这里不假设底层环境一定是 OSWorld：只要实现 `reset/step/close`
        的 `InteractiveEnv` 都可以接入。主循环的终止条件只有两个：
        环境返回非 RUNNING 状态，或达到 `max_steps` 后主动发出 fail。
        这样可以让 mock、本地 browser 任务和真实 sandbox 共用同一套
        agent/memory 逻辑。
        """

        observation = env.reset(task_id=task_id)
        trajectory = Trajectory(task_id=task_id or "default", task=observation.task)

        for _ in range(self.config.max_steps):
            # 单步决策严格拆成 act + env.step：act 只看 observation 和 memory，
            # step 只负责环境副作用。这能让 trajectory 记录和回放更清晰。
            action = self.act(observation)
            result = env.step(action)
            trajectory.append(result)
            if result.status != StepStatus.RUNNING:
                break
            observation = result.next_observation or observation
        else:
            # Python for/else 只在循环没有 break 时进入；这里表示模型没有在
            # max_steps 内完成任务，需要给环境一个显式失败动作并记录原因。
            result = env.step(AgentAction("fail", text="Agent max_steps reached.", thought="Stop after max_steps."))
            trajectory.append(result)

        if self.config.update_memory:
            self.memory_controller.update_from_trajectory(trajectory)
        return trajectory

    def act(self, observation: Observation) -> AgentAction:
        """从当前 observation 生成下一步内部动作。

        关键步骤是：把 observation 转成稳定的感知文本，基于任务+感知文本
        检索长期记忆，把检索结果嵌入 prompt，再调用 VLM。VLM 的输出必须
        经过 ActionParser 白名单校验；解析失败时返回 fail，而不是抛异常
        中断整个 runner，便于 trajectory 中保留失败上下文。
        """

        perception = self.screen_parser.parse(observation)
        memory_nodes = self.memory_controller.retrieve(f"{observation.task}\n{perception}", k=self.config.retrieve_k)
        prompt = self._build_prompt(observation, perception, self.memory_controller.prompt_context(memory_nodes))
        try:
            response = self.vlm.decide(prompt, image_path=observation.screenshot_path)
            return self.action_parser.parse(response)
        except ValueError as exc:
            return AgentAction("fail", text=f"Invalid VLM response: {exc}", thought="Stop because the VLM response could not be parsed.")

    def _build_prompt(self, observation: Observation, perception: str, memory_context: str) -> str:
        """构造模型 prompt。

        Prompt 保持显式 JSON action contract，而不是让后端客户端隐式约束。
        原因是同一套 agent 可能连接 GPT、Qwen/vLLM、SGLang 或本地
        Transformers，不同模型的系统 prompt 遵循程度不同；把动作 schema
        和 GUI 操作规则放在主 prompt 中，可以让 parser 的兜底更可靠。
        """

        return f"""You are a memory-augmented VLM GUI agent.

Task:
{observation.task}

Current observation:
{perception}

Relevant long-term experience memory:
{memory_context}

Choose exactly one next action. Return JSON with:
- thought: brief reason grounded in observation and memory
- action_type: click | double_click | right_click | type | paste | hotkey | press | scroll | wait | done | fail
- target: UI element id if needed; for OSWorld screenshots without element ids, use "x,y" click coordinates
- x, y: optional integer screen coordinates for coordinate clicks
- text: text to type or failure reason if needed

Rules:
- Click only visible element ids or visible screen coordinates.
- Type only after a text field is visible or focused.
- Use paste instead of type for long, exact, or non-ASCII text.
- Use done only when the goal state is visibly achieved.
- Prefer memory when it matches the current task, but reject stale memory if observation contradicts it.
"""
