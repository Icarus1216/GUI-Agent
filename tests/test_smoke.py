"""功能: 覆盖 agent 主循环、OSWorld adapter、VLM response 解析、preflight 和 Qwen serving wrapper 的 smoke 测试。
上游依赖: 依赖 src/vlm_memory_agent 各核心模块、临时目录、mock patch 和纯本地 mock 环境。
下游依赖: 开发收尾和回归验证通过 `PYTHONPATH=src python -m unittest tests.test_smoke` 执行本测试文件。
"""

import unittest
import contextlib
import io
import os
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest import mock

from vlm_memory_agent.cli import main as cli_main
from vlm_memory_agent.core.types import AgentAction, Observation, StepStatus, UIElement
from vlm_memory_agent.agent import VLMGuiAgent
from vlm_memory_agent.envs.osworld_adapter import OSWorldAdapter
from vlm_memory_agent.envs.osworld_runner import OSWorldEnvConfig, _construct_with_supported_kwargs, _desktop_env_failure_hint, iter_task_config_paths, load_task_config
from vlm_memory_agent.envs.local_browser import LocalBrowserSalesEnv
from vlm_memory_agent.envs.mock_osworld import MockOSWorldEnv
from vlm_memory_agent.eval import _safe_file_stem, write_report, EvalResult
from vlm_memory_agent.llm.base import VLMClient, VLMResponse, parse_vlm_response_text
from vlm_memory_agent.llm.openai_compatible import OpenAICompatibleVLMClient
from vlm_memory_agent.llm.rule_based import RuleBasedVLMClient
from vlm_memory_agent.memory.store import HierarchicalMemoryStore
from vlm_memory_agent.preflight import check_osworld, check_osworld_runtime
from vlm_memory_agent.osworld_ready import collect_readiness
from vlm_memory_agent.qwen36_healthcheck import check_endpoint
from vlm_memory_agent.runtime_paths import build_pythonpath_with_runtime_deps
from vlm_memory_agent.serve_qwen36 import main as serve_qwen36_main
from vlm_memory_agent.tools.action_parser import ActionParser
from vlm_memory_agent.tools.screen_parser import ScreenParserTool


class SmokeTest(unittest.TestCase):
    """覆盖核心协议的低成本回归测试。

    这些测试不启动真实 VM 或大模型，目标是保证本地状态机、adapter 翻译、
    VLM JSON 解析、preflight 聚合和 serving 命令拼装这些基础契约不退化。
    """

    def test_mock_episode_succeeds_and_updates_memory(self):
        memory = HierarchicalMemoryStore()
        agent = VLMGuiAgent(RuleBasedVLMClient(), memory)
        trajectory = agent.run_episode(MockOSWorldEnv(), task_id="search_report")
        self.assertTrue(trajectory.success)
        self.assertTrue(any(node.kind == "trajectory" for node in memory.nodes.values()))
        self.assertTrue(any(node.kind == "strategy" for node in memory.nodes.values()))

    def test_local_browser_sales_episode_succeeds_with_screenshots(self):
        with TemporaryDirectory() as tmp:
            env = LocalBrowserSalesEnv(screenshot_dir=Path(tmp) / "screens")
            trajectory = VLMGuiAgent(RuleBasedVLMClient(), HierarchicalMemoryStore(), config=None).run_episode(
                env, task_id="sales_approval"
            )
            self.assertTrue(trajectory.success)
            payload = trajectory.to_dict()
            self.assertGreaterEqual(len(payload["steps"]), 8)
            screenshot_path = payload["steps"][0]["observation_detail"]["screenshot_path"]
            self.assertTrue(Path(screenshot_path).exists())
            self.assertIn("crm_saved", payload["steps"][-1]["next_observation_detail"]["metadata"])

    def test_agent_turns_invalid_vlm_json_into_fail_action(self):
        class BadJsonVLM(VLMClient):
            def decide(self, prompt, image_path=None):
                raise ValueError("Model did not return JSON")

        agent = VLMGuiAgent(BadJsonVLM(), HierarchicalMemoryStore())
        action = agent.act(Observation(step=0, task="demo"))
        self.assertEqual(action.action_type, "fail")
        self.assertIn("Invalid VLM response", action.text or "")

    def test_osworld_adapter_translates_click_to_pyautogui(self):
        adapter = OSWorldAdapter(env=object())
        adapter._last_obs = Observation(
            step=0,
            task="demo",
            ui_elements=[UIElement("ok_button", "OK", "button", bbox=(10, 20, 30, 60))],
        )
        action = adapter._to_osworld_action(AgentAction("click", target="ok_button"))
        self.assertEqual(action, "pyautogui.click(20, 40)")

    def test_osworld_adapter_translates_coordinate_click_target(self):
        adapter = OSWorldAdapter(env=object())
        action = adapter._to_osworld_action(AgentAction("click", target="(101.7, 202.2)"))
        self.assertEqual(action, "pyautogui.click(101, 202)")

    def test_osworld_adapter_translates_coordinate_click_metadata(self):
        adapter = OSWorldAdapter(env=object())
        action = adapter._to_osworld_action(AgentAction("click", metadata={"x": 101, "y": 202}))
        self.assertEqual(action, "pyautogui.click(101, 202)")

    def test_vlm_response_parser_handles_code_fence_alias_and_coordinates(self):
        response = parse_vlm_response_text(
            '```json\n{"thought": "press visible button", "action_type": "left_click", "coordinate": [101.9, 202.1]}\n```'
        )
        action = ActionParser().parse(response)
        self.assertEqual(action.action_type, "click")
        self.assertEqual(action.metadata["x"], 101)
        self.assertEqual(action.metadata["y"], 202)

    def test_vlm_response_parser_uses_bbox_center(self):
        response = parse_vlm_response_text('{"thought": "open", "action": "click_point", "bbox": [10, 20, 30, 60]}')
        self.assertEqual(response.action_type, "click")
        self.assertEqual(response.metadata["x"], 20)
        self.assertEqual(response.metadata["y"], 40)

    def test_vlm_response_parser_handles_nested_fenced_json(self):
        response = parse_vlm_response_text(
            '```json\n{"thought": "scroll", "action_type": "mouse_scroll", "metadata": {"source": "model"}, "dy": -3}\n```'
        )
        self.assertEqual(response.action_type, "scroll")
        self.assertEqual(response.metadata["dy"], -3)

    def test_vlm_response_parser_merges_nested_action_parameters(self):
        response = parse_vlm_response_text(
            '{"thought": "click", "action_type": "click", "parameters": {"x": 12, "y": 34, "target": "ignored"}}'
        )
        self.assertEqual(response.action_type, "click")
        self.assertEqual(response.metadata["x"], 12)
        self.assertEqual(response.metadata["y"], 34)

    def test_openai_compatible_flattens_list_message_content(self):
        client = OpenAICompatibleVLMClient(model="demo")
        text = client._message_text([{"type": "text", "text": '{"action_type":"wait"}'}])
        self.assertEqual(text, '{"action_type":"wait"}')

    def test_action_parser_rejects_invalid_action_type(self):
        action = ActionParser().parse(VLMResponse("bad", "drag"))
        self.assertEqual(action.action_type, "fail")

    def test_osworld_adapter_normalizes_common_dict_observation(self):
        adapter = OSWorldAdapter(env=object())
        obs = adapter._normalize_observation(
            {
                "instruction": "Open the file.",
                "accessibility_tree": "button Open",
                "ui_elements": [{"id": "open", "label": "Open", "role": "button", "bbox": [1, 2, 3, 4]}],
            }
        )
        self.assertEqual(obs.task, "Open the file.")
        self.assertEqual(obs.screen_text, "button Open")
        self.assertEqual(obs.ui_elements[0].bbox, (1, 2, 3, 4))

    def test_osworld_adapter_includes_terminal_output_in_observation(self):
        adapter = OSWorldAdapter(env=object())
        obs = adapter._normalize_observation({"instruction": "demo", "accessibility_tree": "root", "terminal": "ls output"})
        self.assertIn("root", obs.screen_text)
        self.assertIn("Terminal output:", obs.screen_text)
        self.assertIn("ls output", obs.screen_text)

    def test_osworld_adapter_does_not_truth_test_screenshot_arrays(self):
        class ArrayLike:
            def __bool__(self):
                raise ValueError("ambiguous truth value")

        adapter = OSWorldAdapter(env=object())
        obs = adapter._normalize_observation(
            {
                "instruction": "demo",
                "screenshot": ArrayLike(),
                "accessibility_tree": "root",
            }
        )
        self.assertEqual(obs.task, "demo")
        self.assertEqual(obs.screen_text, "root")

    def test_screen_parser_truncates_large_accessibility_tree(self):
        parser = ScreenParserTool(max_screen_text_chars=10)
        text = parser.parse(Observation(step=0, task="demo", screen_text="x" * 20))
        self.assertIn("...[truncated]", text)

    def test_observation_compact_truncates_large_screen_text(self):
        compact = Observation(step=0, task="demo", screen_text="x" * 5000).compact()
        self.assertLess(len(compact), 4200)
        self.assertIn("...[truncated]", compact)

    def test_trajectory_to_dict_includes_structured_debug_details(self):
        trajectory = VLMGuiAgent(RuleBasedVLMClient(), HierarchicalMemoryStore()).run_episode(
            MockOSWorldEnv(), task_id="search_report"
        )
        payload = trajectory.to_dict()
        first_step = payload["steps"][0]
        self.assertIn("observation_detail", first_step)
        self.assertIn("action_detail", first_step)
        self.assertIn("metadata", first_step)
        self.assertIn("next_observation_detail", first_step)
        self.assertEqual(first_step["action_detail"]["action_type"], "type")
        self.assertIn("ui_elements", first_step["observation_detail"])

    def test_osworld_adapter_done_uses_evaluate_reward(self):
        class FakeEnv:
            def __init__(self):
                self.actions = []

            def step(self, action):
                self.actions.append(action)
                return {"instruction": "demo", "text": "done"}, 0, True, {"done": True}

            def evaluate(self):
                return 1

        env = FakeEnv()
        adapter = OSWorldAdapter(env=env)
        adapter._last_obs = Observation(step=0, task="demo")
        result = adapter.step(AgentAction("done"))
        self.assertEqual(env.actions, ["DONE"])
        self.assertEqual(result.status, StepStatus.SUCCESS)
        self.assertEqual(result.reward, 1.0)

    def test_osworld_adapter_surfaces_evaluate_error_feedback(self):
        class FakeEnv:
            def step(self, action):
                return {"instruction": "demo", "text": "done"}, 0, True, {"done": True}

            def evaluate(self):
                raise RuntimeError("metric failed")

        adapter = OSWorldAdapter(env=FakeEnv())
        adapter._last_obs = Observation(step=0, task="demo")
        result = adapter.step(AgentAction("done"))
        self.assertEqual(result.status, StepStatus.FAILED)
        self.assertIn("evaluate_error", result.feedback)
        self.assertIn("metric failed", result.feedback)

    def test_osworld_adapter_translates_special_actions(self):
        adapter = OSWorldAdapter(env=object())
        self.assertEqual(adapter._to_osworld_action(AgentAction("wait")), "WAIT")
        self.assertEqual(adapter._to_osworld_action(AgentAction("done")), "DONE")
        self.assertEqual(adapter._to_osworld_action(AgentAction("fail")), "FAIL")

    def test_osworld_adapter_translates_extended_pyautogui_actions(self):
        adapter = OSWorldAdapter(env=object())
        self.assertEqual(
            adapter._to_osworld_action(AgentAction("double_click", metadata={"x": 10, "y": 20})),
            "pyautogui.doubleClick(10, 20)",
        )
        self.assertEqual(
            adapter._to_osworld_action(AgentAction("right_click", target="10,20")),
            "pyautogui.rightClick(10, 20)",
        )
        self.assertEqual(adapter._to_osworld_action(AgentAction("press", text="Enter")), 'pyautogui.press("enter")')
        self.assertEqual(adapter._to_osworld_action(AgentAction("scroll", text="down")), "pyautogui.scroll(-5)")
        paste = adapter._to_osworld_action(AgentAction("paste", text="hello 世界"))
        compile("import pyautogui; import time; pyautogui.FAILSAFE = False; " + paste, "<paste>", "exec")
        self.assertTrue(paste.startswith("exec("))
        self.assertIn("pyperclip.copy", paste)
        self.assertIn('pyautogui.hotkey(\\"ctrl\\", \\"v\\")', paste)
        self.assertIn("except Exception", paste)
        self.assertIn("pyautogui.typewrite", paste)

    def test_osworld_adapter_records_raw_action_metadata(self):
        class FakeEnv:
            def step(self, action):
                return {"instruction": "demo", "text": "next"}, 0, False, {"ok": True}

        adapter = OSWorldAdapter(env=FakeEnv())
        adapter._last_obs = Observation(step=0, task="demo")
        result = adapter.step(AgentAction("click", metadata={"x": 10, "y": 20}))
        self.assertEqual(result.metadata["osworld_action"], "pyautogui.click(10, 20)")
        self.assertEqual(result.metadata["osworld_info"], {"ok": True})

    def test_osworld_adapter_dict_mode_uses_desktop_env_command_protocol(self):
        adapter = OSWorldAdapter(env=object(), action_mode="dict")
        action = adapter._to_osworld_action(AgentAction("click", metadata={"x": 10, "y": 20}))
        self.assertEqual(action, {"action_type": "pyautogui", "command": "pyautogui.click(10, 20)"})
        self.assertEqual(adapter._to_osworld_action(AgentAction("done")), "DONE")

    def test_osworld_adapter_rejects_unknown_action_mode(self):
        adapter = OSWorldAdapter(env=object(), action_mode="unknown")
        with self.assertRaises(ValueError):
            adapter._to_osworld_action(AgentAction("wait"))

    def test_osworld_adapter_reset_does_not_force_task_id_into_noarg_reset(self):
        class NoArgResetEnv:
            def reset(self):
                return {"instruction": "demo", "text": "ok"}

        obs = OSWorldAdapter(env=NoArgResetEnv()).reset(task_id="task")
        self.assertEqual(obs.screen_text, "ok")

    def test_osworld_task_config_directory_resolves_by_task_id(self):
        with TemporaryDirectory() as tmp:
            task_dir = Path(tmp)
            (task_dir / "abc.json").write_text('{"id": "target_task", "instruction": "demo"}', encoding="utf-8")
            config = load_task_config(task_dir, task_id="target_task")
        self.assertEqual(config["instruction"], "demo")

    def test_iter_task_config_paths_filters_to_instruction_json(self):
        with TemporaryDirectory() as tmp:
            task_dir = Path(tmp)
            valid = task_dir / "task.json"
            valid.write_text('{"id": "task", "instruction": "demo", "evaluator": {"func": "infeasible"}}', encoding="utf-8")
            (task_dir / "metadata.json").write_text('{"name": "not a task"}', encoding="utf-8")
            self.assertEqual(iter_task_config_paths(task_dir), [valid])

    def test_eval_report_includes_trajectory_path(self):
        with TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            write_report([EvalResult("task/one", True, 2, 1.0, trajectory_path="traj.json")], report)
            payload = report.read_text(encoding="utf-8")
        self.assertIn('"trajectory_path": "traj.json"', payload)
        self.assertEqual(_safe_file_stem("task/one"), "task_one")

    def test_osworld_env_config_kwargs_match_desktop_env_signature(self):
        captured = {}

        class FakeDesktopEnv:
            def __init__(self, provider_name=None, region=None, path_to_vm=None, snapshot_name="init_state", require_terminal=False, enable_proxy=False, client_password=""):
                captured.update(locals())

        config = OSWorldEnvConfig(
            task_config="task.json",
            provider_name="vmware",
            region="us-test",
            vm_path="/tmp/vm.vmx",
            snapshot_name="clean",
            require_terminal=True,
            enable_proxy=True,
            client_password="pw",
        )
        _construct_with_supported_kwargs(
            FakeDesktopEnv,
            {
                "provider_name": config.provider_name,
                "region": config.region,
                "path_to_vm": config.vm_path,
                "snapshot_name": config.snapshot_name,
                "require_terminal": config.require_terminal,
                "enable_proxy": config.enable_proxy,
                "client_password": config.client_password,
                "unknown_future_arg": "ignored",
            },
        )
        self.assertEqual(captured["region"], "us-test")
        self.assertEqual(captured["snapshot_name"], "clean")
        self.assertTrue(captured["require_terminal"])
        self.assertTrue(captured["enable_proxy"])
        self.assertEqual(captured["client_password"], "pw")

    def test_desktop_env_failure_hint_is_provider_aware(self):
        docker_hint = _desktop_env_failure_hint(OSWorldEnvConfig(task_config="task.json", provider_name="docker"))
        vmware_hint = _desktop_env_failure_hint(OSWorldEnvConfig(task_config="task.json", provider_name="vmware"))
        self.assertIn("Docker", docker_hint)
        self.assertIn("--osworld-vm-path", vmware_hint)

    def test_preflight_vm_path_requirement_is_provider_aware(self):
        vmware = {result.name: result for result in check_osworld_runtime(vm_path=None, provider="vmware")}
        docker = {result.name: result for result in check_osworld_runtime(vm_path=None, provider="docker")}
        self.assertFalse(vmware["osworld_vm_path"].ok)
        self.assertTrue(vmware["osworld_vm_path"].required)
        self.assertTrue(docker["osworld_vm_path"].ok)
        self.assertFalse(docker["osworld_vm_path"].required)
        self.assertIn("docker_daemon", docker)

    def test_preflight_pyperclip_is_not_a_required_host_check(self):
        checks = {result.name: result for result in check_osworld()}
        self.assertIn("pyperclip", checks)
        self.assertFalse(checks["pyperclip"].required)

    def test_runtime_pythonpath_includes_qwen_deps(self):
        with TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"VLM_MEMORY_AGENT_QWEN_DEPS": tmp, "PYTHONPATH": "src"}, clear=False):
                value = build_pythonpath_with_runtime_deps()
        self.assertTrue(value.startswith(tmp))
        self.assertIn("src", value)

    def test_serve_qwen36_passes_runtime_pythonpath_to_subprocess(self):
        with TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"VLM_MEMORY_AGENT_QWEN_DEPS": tmp, "PYTHONPATH": "src"}, clear=False):
                with mock.patch("subprocess.call", return_value=0) as call:
                    code = serve_qwen36_main(["--model-path", "/model", "--port", "9999"])
        self.assertEqual(code, 0)
        env = call.call_args.kwargs["env"]
        self.assertTrue(env["PYTHONPATH"].startswith(tmp))

    def test_qwen36_healthcheck_validates_models_and_chat_protocol(self):
        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return self.payload.encode("utf-8")

        responses = [
            FakeResponse('{"data": [{"id": "qwen3.6-35b-a3b"}]}'),
            FakeResponse('{"choices": [{"message": {"content": "{\\"thought\\":\\"healthcheck\\",\\"action_type\\":\\"wait\\"}"}}]}'),
        ]
        with mock.patch("urllib.request.urlopen", side_effect=responses):
            ok, detail = check_endpoint("http://127.0.0.1:8000/v1", "qwen3.6-35b-a3b")
        self.assertTrue(ok)
        self.assertIn("JSON action protocol ok", detail)

    def test_osworld_ready_reports_required_blockers(self):
        with TemporaryDirectory() as tmp:
            task = Path(tmp) / "task.json"
            task.write_text('{"id": "task", "instruction": "demo", "evaluator": {"func": "infeasible"}}', encoding="utf-8")
            payload = collect_readiness(
                task_config=str(task),
                provider="vmware",
                vm_path=None,
                base_url=None,
                model_path="/missing/model",
            )
        self.assertFalse(payload["ready"])
        blocker_names = {item["name"] for item in payload["blockers"]}
        self.assertIn("osworld_vm_path", blocker_names)
        self.assertIn("model_path", blocker_names)

    def test_osworld_ready_includes_endpoint_protocol_check(self):
        with TemporaryDirectory() as tmp:
            task = Path(tmp) / "task.json"
            task.write_text('{"id": "task", "instruction": "demo", "evaluator": {"func": "infeasible"}}', encoding="utf-8")
            model_dir = Path(tmp) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text('{"model_type": "demo"}', encoding="utf-8")
            (model_dir / "weights.safetensors").write_text("", encoding="utf-8")
            with mock.patch("vlm_memory_agent.osworld_ready.check_endpoint", return_value=(True, "endpoint ok")):
                payload = collect_readiness(
                    task_config=str(task),
                    provider="docker",
                    model_path=model_dir,
                    base_url="http://127.0.0.1:8000/v1",
                )
        checks = {item["name"]: item for item in payload["checks"]}
        self.assertIn("qwen36_endpoint_protocol", checks)
        self.assertTrue(checks["qwen36_endpoint_protocol"]["ok"])
        cuda = checks.get("cuda_for_qwen36_local")
        if cuda is not None:
            self.assertFalse(cuda["required"])

    def test_osworld_ready_endpoint_mode_does_not_require_local_model_files(self):
        with TemporaryDirectory() as tmp:
            task = Path(tmp) / "task.json"
            task.write_text('{"id": "task", "instruction": "demo", "evaluator": {"func": "infeasible"}}', encoding="utf-8")
            with mock.patch("vlm_memory_agent.osworld_ready.check_endpoint", return_value=(True, "endpoint ok")):
                payload = collect_readiness(
                    task_config=str(task),
                    provider="docker",
                    model_path="/missing/model",
                    base_url="http://127.0.0.1:8000/v1",
                    serving_mode="endpoint",
                )
        checks = {item["name"]: item for item in payload["checks"]}
        self.assertEqual(payload["serving_mode"], "endpoint")
        self.assertFalse(checks["qwen36_local_model_files"]["ok"])
        self.assertFalse(checks["qwen36_local_model_files"]["required"])
        blocker_names = {item["name"] for item in payload["blockers"]}
        self.assertNotIn("model_path", blocker_names)

    def test_osworld_ready_local_mode_requires_cuda_and_model_files(self):
        with TemporaryDirectory() as tmp:
            task = Path(tmp) / "task.json"
            task.write_text('{"id": "task", "instruction": "demo", "evaluator": {"func": "infeasible"}}', encoding="utf-8")
            payload = collect_readiness(
                task_config=str(task),
                provider="docker",
                model_path="/missing/model",
                base_url=None,
                serving_mode="local",
            )
        blocker_names = {item["name"] for item in payload["blockers"]}
        self.assertIn("model_path", blocker_names)

    def test_cli_mock_default_task_id_still_succeeds(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "trajectory.json"
            memory = Path(tmp) / "memory.json"
            with contextlib.redirect_stdout(io.StringIO()):
                code = cli_main(["--vlm-backend", "rule", "--output", str(output), "--memory-path", str(memory)])
        self.assertEqual(code, 0)

    def test_cli_local_browser_succeeds(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "trajectory.json"
            memory = Path(tmp) / "memory.json"
            screens = Path(tmp) / "screens"
            with contextlib.redirect_stdout(io.StringIO()):
                code = cli_main(
                    [
                        "--env",
                        "local_browser",
                        "--vlm-backend",
                        "rule",
                        "--max-steps",
                        "18",
                        "--output",
                        str(output),
                        "--memory-path",
                        str(memory),
                        "--screenshot-dir",
                        str(screens),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
