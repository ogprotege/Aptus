from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aptus.runtime_env import (
    RuntimeInterpreter,
    probe_runtime_interpreter,
    resolve_runtime_interpreter,
    runtime_inventory,
    validate_runtime_configuration,
)


class RuntimeEnvironmentTests(unittest.TestCase):
    @unittest.skipIf(
        os.name == "nt", "POSIX virtual environments use symlinked interpreters."
    )
    def test_probe_preserves_virtual_environment_interpreter_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            environment = Path(temporary) / "runtime-env"
            (environment / "bin").mkdir(parents=True)
            base_interpreter = Path(temporary) / "base-python"
            base_interpreter.write_text("", encoding="utf-8")
            base_interpreter.chmod(0o700)
            interpreter = environment / "bin" / "python"
            interpreter.symlink_to(base_interpreter)

            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps({"python_version": "3.12.9", "runtimes": {}}),
                stderr="",
            )
            with patch(
                "aptus.runtime_env.subprocess.run", return_value=completed
            ) as runner:
                result = probe_runtime_interpreter(
                    interpreter, source="test", timeout=1
                )

        self.assertEqual(result.path, str(interpreter))
        self.assertEqual(runner.call_args.args[0][0], str(interpreter))

    def test_probe_normalizes_the_supported_runtime_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            interpreter = Path(temporary) / "python"
            interpreter.write_text("", encoding="utf-8")
            interpreter.chmod(0o700)
            payload = {
                "python_version": "3.12.9",
                "runtimes": {
                    "mlx-lm": {"available": True, "versions": {"mlx-lm": "0.31.3"}},
                    "untrusted-runtime": {"available": True},
                },
            }
            completed = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=json.dumps(payload), stderr=""
            )
            with patch("aptus.runtime_env.subprocess.run", return_value=completed):
                result = probe_runtime_interpreter(
                    interpreter, source="test", timeout=1
                )

        self.assertEqual(result.python_version, "3.12.9")
        self.assertTrue(result.runtimes["mlx-lm"]["available"])
        self.assertNotIn("untrusted-runtime", result.runtimes)

    def test_explicit_runtime_path_must_pass_the_executable_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "python"
            path.write_text("", encoding="utf-8")
            path.chmod(0o700)
            probe = RuntimeInterpreter(
                path=str(Path(os.path.abspath(path))),
                source="environment:APTUS_MLX_PYTHON:mlx-lm",
                python_version="3.12.9",
                runtimes={"mlx-lm": {"available": False, "reason": "missing"}},
            )
            with self.assertRaisesRegex(RuntimeError, "not compatible"):
                resolve_runtime_interpreter(
                    "mlx-lm",
                    interpreters=(probe,),
                    environment={"APTUS_MLX_PYTHON": str(path)},
                )

    def test_unconfigured_runtime_requires_a_measured_available_probe(self) -> None:
        unavailable = RuntimeInterpreter(
            path="/python-a",
            source="test",
            python_version="3.12.9",
            runtimes={"mlx-lm": {"available": False}},
        )
        available = RuntimeInterpreter(
            path="/python-b",
            source="test",
            python_version="3.12.9",
            runtimes={
                "mlx-lm": {
                    "available": True,
                    "versions": {"mlx": "0.31.2", "mlx-lm": "0.31.3"},
                }
            },
        )
        self.assertEqual(
            resolve_runtime_interpreter(
                "mlx-lm", interpreters=(unavailable, available), environment={}
            ),
            available,
        )

    def test_unconfigured_runtime_rejects_importable_but_unpinned_mlx(self) -> None:
        incompatible = RuntimeInterpreter(
            path="/python",
            source="test",
            python_version="3.12.9",
            runtimes={
                "mlx-lm": {
                    "available": True,
                    "versions": {"mlx": "0.31.1", "mlx-lm": "0.31.3"},
                }
            },
        )
        with self.assertRaisesRegex(RuntimeError, "No available interpreter"):
            resolve_runtime_interpreter(
                "mlx-lm", interpreters=(incompatible,), environment={}
            )

    def test_inventory_does_not_conflate_inference_services_with_training(self) -> None:
        probe = RuntimeInterpreter(
            path="/python",
            source="test",
            python_version="3.12.9",
            runtimes={
                "mlx-lm": {"available": True},
                "pytorch-mps": {"available": False},
            },
        )
        result = runtime_inventory(interpreters=(probe,))
        self.assertEqual(result["available"]["mlx-lm"], ["/python"])
        self.assertEqual(result["compatible"]["mlx-lm"], [])
        self.assertFalse(result["interpreters"][0]["runtimes"]["mlx-lm"]["compatible"])
        self.assertNotIn("lm-studio", result["available"])
        self.assertNotIn("omlx", result["available"])

    def test_configuration_requires_the_requested_runtime_to_be_available(self) -> None:
        unavailable = RuntimeInterpreter(
            path="/selected/python",
            source="configured:APTUS_MLX_PYTHON",
            python_version="3.12.9",
            runtimes={"mlx-lm": {"available": False, "reason": "missing mlx_lm"}},
        )
        with patch(
            "aptus.runtime_env.probe_runtime_interpreter",
            return_value=unavailable,
        ):
            with self.assertRaisesRegex(RuntimeError, "missing mlx_lm"):
                validate_runtime_configuration("mlx-lm", Path("/selected/python"))

    def test_configuration_accepts_only_the_pinned_mlx_versions(self) -> None:
        available = RuntimeInterpreter(
            path="/selected/python",
            source="configured:APTUS_MLX_PYTHON",
            python_version="3.12.9",
            runtimes={
                "mlx-lm": {
                    "available": True,
                    "versions": {"mlx": "0.31.2", "mlx-lm": "0.31.3"},
                }
            },
        )
        with patch(
            "aptus.runtime_env.probe_runtime_interpreter",
            return_value=available,
        ):
            result = validate_runtime_configuration("mlx-lm", Path("/selected/python"))

        self.assertEqual(result, available)

    def test_configuration_rejects_unpinned_mlx_versions(self) -> None:
        for versions in (
            {"mlx": "0.31.1", "mlx-lm": "0.31.3"},
            {"mlx": "0.31.2", "mlx-lm": "0.31.4"},
            {"mlx": "0.31.2"},
        ):
            with self.subTest(versions=versions):
                available = RuntimeInterpreter(
                    path="/selected/python",
                    source="configured:APTUS_MLX_PYTHON",
                    python_version="3.12.9",
                    runtimes={"mlx-lm": {"available": True, "versions": versions}},
                )
                with patch(
                    "aptus.runtime_env.probe_runtime_interpreter",
                    return_value=available,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "expected.*0\\.31\\.2.*0\\.31\\.3",
                    ):
                        validate_runtime_configuration(
                            "mlx-lm", Path("/selected/python")
                        )

    def test_configuration_does_not_apply_the_mlx_pin_to_cuda(self) -> None:
        available = RuntimeInterpreter(
            path="/selected/python",
            source="configured:APTUS_CUDA_PYTHON",
            python_version="3.12.9",
            runtimes={
                "transformers-peft-cuda": {
                    "available": True,
                    "versions": {"torch": "independently-managed"},
                }
            },
        )
        with patch(
            "aptus.runtime_env.probe_runtime_interpreter",
            return_value=available,
        ):
            result = validate_runtime_configuration(
                "transformers-peft-cuda", Path("/selected/python")
            )

        self.assertEqual(result, available)


if __name__ == "__main__":
    unittest.main()
