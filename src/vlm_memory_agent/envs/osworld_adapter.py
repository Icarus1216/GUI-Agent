"""功能: 把真实 OSWorld-like 环境的 observation/action schema 归一化为项目内部协议。
上游依赖: 依赖 core.types、InteractiveEnv、Pillow 可选截图保存能力和外部 env.reset/env.step/evaluate。
下游依赖: OSWorld runner、CLI、BenchmarkRunner、真实 OSWorld 脚本和测试通过该适配层执行 GUI 动作。
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Any

from vlm_memory_agent.core.types import AgentAction, Observation, StepResult, StepStatus, UIElement
from vlm_memory_agent.envs.base import InteractiveEnv


class OSWorldAdapter(InteractiveEnv):
    """Thin adapter for an installed OSWorld environment.

    OSWorld revisions expose slightly different Python APIs. This adapter keeps
    the agent code stable by accepting a constructed OSWorld-like env object
    with reset/step methods and normalizing observations/actions.
    """

    def __init__(
        self,
        env: Any,
        task_config: dict[str, Any] | None = None,
        action_mode: str = "pyautogui",
        screenshot_dir: str | Path | None = None,
        evaluate_on_done: bool = True,
        task_id: str | None = None,
    ):
        self.env = env
        self.task_config = task_config
        self.action_mode = action_mode
        self.screenshot_dir = Path(screenshot_dir) if screenshot_dir else None
        self.evaluate_on_done = evaluate_on_done
        self.task_id = task_id or str((task_config or {}).get("id") or "")
        self._last_obs: Observation | None = None
        self._step = 0

    def reset(self, task_id: str | None = None) -> Observation:
        """重置 OSWorld-like 环境并归一化初始 observation。

        OSWorld 不同版本的 `reset` 参数名并不完全一致，真实 task config
        也可能通过 env 构造期或 reset 期传入，所以 `_reset_raw` 会做签名
        兼容，随后统一调用 `_normalize_observation`。
        """

        self._step = 0
        raw = self._reset_raw(task_id)
        self._last_obs = self._normalize_observation(raw, task_id=task_id)
        return self._last_obs

    def step(self, action: AgentAction) -> StepResult:
        """把内部 AgentAction 翻译成 OSWorld action 并执行一步。

        这里的关键是保留两套信息：对外返回统一的 `StepResult`，同时把真实
        发送给 OSWorld 的 action 和原始 info 放进 metadata，方便排查动作
        翻译、坐标落点和评测 reward 问题。
        """

        observation = self._last_obs or Observation(step=self._step, task="")

        raw_action = self._to_osworld_action(action)
        raw_result = self.env.step(raw_action)
        obs, reward, done, info = self._unpack_step(raw_result)
        info = info or {}
        if done and self.evaluate_on_done:
            # 某些 OSWorld 版本的 step reward 不等于最终 evaluator reward。
            # done 后主动 evaluate 一次，能让 trajectory 里的 reward 更贴近
            # benchmark 评分；失败则只记录 evaluate_error，不影响 step 记录。
            eval_reward = self._evaluate_if_available(info)
            if eval_reward is not None:
                reward = eval_reward
                info.setdefault("evaluate_reward", eval_reward)
        self._step += 1
        next_obs = self._normalize_observation(obs)
        status = StepStatus.SUCCESS if done and reward > 0 else StepStatus.FAILED if done else StepStatus.RUNNING
        result = StepResult(
            observation=observation,
            action=action,
            status=status,
            reward=float(reward or 0.0),
            feedback=self._feedback_from_info(info),
            next_observation=next_obs,
            metadata={"osworld_action": self._safe_metadata(raw_action), "osworld_info": self._safe_metadata(info)},
        )
        self._last_obs = next_obs
        return result

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def _reset_raw(self, task_id: str | None) -> Any:
        if self.task_config is not None:
            return self._call_with_supported_kwargs(self.env.reset, {"task_config": self.task_config})
        if task_id is not None:
            return self._call_with_supported_kwargs(self.env.reset, {"task_id": task_id})
        return self.env.reset()

    def _normalize_observation(self, raw: Any, task_id: str | None = None) -> Observation:
        """把 OSWorld/外部 sandbox 的 observation schema 转成内部 Observation。

        支持的输入形态包括：
        - 已经是 Observation 的对象，直接透传；
        - dict，抽取 screenshot/image、screen text、ui_elements、instruction；
        - 其它对象，退化成纯文本 raw observation。
        """

        if isinstance(raw, Observation):
            return raw
        if isinstance(raw, dict):
            screenshot_path = self._materialize_screenshot(self._first_present(raw, "screenshot", "image"))
            screen_text = self._compose_screen_text(raw)
            elements = [
                UIElement(
                    element_id=str(item.get("id", idx)),
                    label=str(item.get("label") or item.get("text") or ""),
                    role=str(item.get("role", "unknown")),
                    bbox=tuple(item["bbox"]) if item.get("bbox") else None,
                )
                for idx, item in enumerate(raw.get("ui_elements", []) or [])
            ]
            return Observation(
                step=int(raw.get("step", self._step)),
                task=str(raw.get("instruction") or raw.get("task") or (self.task_config or {}).get("instruction") or task_id or ""),
                screenshot_path=str(self._first_present(raw, "screenshot_path") or screenshot_path or "") or None,
                screen_text=str(screen_text),
                ui_elements=elements,
                metadata={key: self._safe_metadata(value) for key, value in raw.items() if key not in {"screenshot", "image"}},
            )
        return Observation(step=0, task=str(task_id or ""), screen_text=str(raw), metadata={"raw": raw})

    def _to_osworld_action(self, action: AgentAction) -> str | dict[str, Any]:
        """选择 OSWorld action 输出格式。

        主流 desktop_env 使用 pyautogui 字符串；少数 wrapper 期望 dict。
        这里用 `action_mode` 做一层薄转换，避免 agent 层关心具体 OSWorld
        版本的 action schema。
        """

        if self.action_mode == "pyautogui":
            return self._to_pyautogui_action(action)

        if self.action_mode == "dict":
            raw_action = self._to_pyautogui_action(action)
            if raw_action in {"WAIT", "DONE", "FAIL"}:
                return raw_action
            return {"action_type": "pyautogui", "command": raw_action}

        raise ValueError(f"Unsupported OSWorld adapter action_mode: {self.action_mode}")

    def _to_pyautogui_action(self, action: AgentAction) -> str:
        """把内部动作翻译成 desktop_env 可执行的 pyautogui 命令字符串。

        终止动作使用 OSWorld 约定的 `WAIT/DONE/FAIL`，非终止动作生成
        Python 代码字符串。无法安全落地的动作退化为 WAIT，避免把未知目标
        翻译成错误点击。
        """

        if action.action_type in {"click", "double_click", "right_click"}:
            coords = self._resolve_click_coordinates(action)
            if coords is None:
                return "WAIT"
            x, y = coords
            if action.action_type == "double_click":
                return f"pyautogui.doubleClick({x}, {y})"
            if action.action_type == "right_click":
                return f"pyautogui.rightClick({x}, {y})"
            return f"pyautogui.click({x}, {y})"
        if action.action_type == "type":
            return f"pyautogui.typewrite({json.dumps(action.text or '')}, interval=0.01)"
        if action.action_type == "paste":
            return self._paste_command(action.text or "")
        if action.action_type == "press":
            key = (action.text or action.target or "").strip().lower()
            if not key:
                return "WAIT"
            return f"pyautogui.press({json.dumps(key)})"
        if action.action_type == "hotkey":
            keys = [part.strip().lower() for part in (action.text or action.target or "").replace("+", ",").split(",")]
            keys = [key for key in keys if key]
            if not keys:
                return "time.sleep(1)"
            return f"pyautogui.hotkey({', '.join(json.dumps(key) for key in keys)})"
        if action.action_type == "scroll":
            clicks = self._resolve_scroll_clicks(action)
            if clicks == 0:
                return "WAIT"
            return f"pyautogui.scroll({clicks})"
        if action.action_type == "wait":
            return "WAIT"
        if action.action_type == "done":
            return "DONE"
        if action.action_type == "fail":
            return "FAIL"
        return "WAIT"

    def _evaluate_if_available(self, info: dict[str, Any]) -> float | None:
        evaluate = getattr(self.env, "evaluate", None)
        if not callable(evaluate):
            return None
        try:
            return float(evaluate())
        except Exception as exc:
            info["evaluate_error"] = f"{type(exc).__name__}: {exc}"
            return None

    def _feedback_from_info(self, info: dict[str, Any]) -> str:
        if info.get("feedback"):
            return str(info["feedback"])
        if info.get("evaluate_error"):
            return f"evaluate_error: {info['evaluate_error']}"
        if info.get("done"):
            return "OSWorld reported DONE."
        if info.get("fail"):
            return "OSWorld reported FAIL."
        return ""

    def _resolve_click_coordinates(self, action: AgentAction) -> tuple[int, int] | None:
        """解析点击坐标，优先级为 metadata 坐标、target 坐标串、UI bbox。

        这样同时支持三类模型输出：
        - `{"metadata": {"x": 1, "y": 2}}`
        - `{"target": "100,200"}`
        - `{"target": "button_id"}` 且 observation 中有对应 bbox。
        """

        if "x" in action.metadata and "y" in action.metadata:
            return int(action.metadata["x"]), int(action.metadata["y"])
        if action.target:
            numbers = re.findall(r"-?\d+(?:\.\d+)?", action.target)
            if len(numbers) >= 2:
                return int(float(numbers[0])), int(float(numbers[1]))
        if not self._last_obs or not action.target:
            return None
        for element in self._last_obs.ui_elements:
            if element.element_id == action.target and element.bbox:
                x1, y1, x2, y2 = element.bbox
                return int((x1 + x2) / 2), int((y1 + y2) / 2)
        return None

    def _resolve_scroll_clicks(self, action: AgentAction) -> int:
        for key in ("clicks", "dy", "amount", "value"):
            if key in action.metadata:
                return int(float(action.metadata[key]))
        text = (action.text or action.target or "").strip().lower()
        if not text:
            return -5
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        if numbers:
            return int(float(numbers[0]))
        if text in {"down", "scroll_down", "page_down"}:
            return -5
        if text in {"up", "scroll_up", "page_up"}:
            return 5
        return -5

    def _paste_command(self, text: str) -> str:
        """生成带 fallback 的粘贴命令。

        OSWorld 的 pyautogui 在 VM/控制器里执行，host 是否安装 pyperclip
        不重要。优先 pyperclip+hotkey 能处理长文本和非 ASCII；失败时退化为
        typewrite，保证命令仍可执行。
        """

        script = (
            "try:\n"
            "    import pyperclip\n"
            f"    pyperclip.copy({json.dumps(text)})\n"
            '    pyautogui.hotkey("ctrl", "v")\n'
            "except Exception:\n"
            f"    pyautogui.typewrite({json.dumps(text)}, interval=0.01)"
        )
        return f"exec({json.dumps(script)})"

    def _compose_screen_text(self, raw: dict[str, Any]) -> str:
        primary = self._first_present(raw, "text", "screen_text", "accessibility_tree", "a11y_tree", "a11y")
        terminal = raw.get("terminal")
        parts = []
        if primary:
            parts.append(str(primary))
        if terminal:
            parts.append("Terminal output:\n" + str(terminal))
        return "\n\n".join(parts)

    def _materialize_screenshot(self, value: Any) -> str | None:
        """把 observation 中的截图对象落盘为文件路径。

        上游可能给路径、bytes、PIL Image 或 numpy array。agent/VLM 客户端
        更适合接收稳定文件路径，因此 adapter 在配置了 screenshot_dir 时
        尽量把非路径截图保存为 `step_XXXX.png`。
        """

        if value is None:
            return None
        if isinstance(value, str):
            return value
        if self.screenshot_dir is None:
            return None
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        path = self.screenshot_dir / f"step_{self._step:04d}.png"
        if isinstance(value, bytes):
            path.write_bytes(value)
            return str(path)
        save = getattr(value, "save", None)
        if callable(save):
            save(path)
            return str(path)
        try:
            from PIL import Image

            Image.fromarray(value).save(path)
            return str(path)
        except Exception:
            return None

    def _safe_metadata(self, value: Any) -> Any:
        try:
            json.dumps(value)
            return value
        except TypeError:
            return repr(value)

    def _first_present(self, payload: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in payload and payload[key] is not None:
                return payload[key]
        return None

    def _call_with_supported_kwargs(self, fn: Any, kwargs: dict[str, Any]) -> Any:
        """按函数签名过滤 kwargs，兼容不同 OSWorld reset API。

        如果函数支持 `**kwargs` 就原样传入；否则只传签名里存在的参数。
        对没有签名信息的 C/动态对象，保守地尝试完整 kwargs。
        """

        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            return fn(**kwargs)
        params = signature.parameters
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
            return fn(**kwargs)
        filtered = {key: value for key, value in kwargs.items() if key in params}
        if filtered:
            return fn(**filtered)
        if not params:
            return fn()
        return fn(next(iter(kwargs.values())))

    def _unpack_step(self, raw_result: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        """兼容 gymnasium/gym/dict 三种常见 step 返回协议。"""

        if isinstance(raw_result, tuple):
            if len(raw_result) == 5:
                obs, reward, terminated, truncated, info = raw_result
                return obs, reward, bool(terminated or truncated), info or {}
            if len(raw_result) == 4:
                obs, reward, done, info = raw_result
                return obs, reward, bool(done), info or {}
        if isinstance(raw_result, dict):
            return (
                raw_result.get("observation", raw_result),
                float(raw_result.get("reward", 0.0)),
                bool(raw_result.get("done", False)),
                raw_result.get("info", {}),
            )
        return raw_result, 0.0, False, {}
