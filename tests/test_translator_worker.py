from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


class TranslatorWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        python_path = str(self.repo_root / "src")
        if env.get("PYTHONPATH"):
            python_path = python_path + os.pathsep + env["PYTHONPATH"]
        env["PYTHONPATH"] = python_path
        self.process = subprocess.Popen(
            [sys.executable, "-m", "npu_xmx.translator_worker"],
            cwd=self.repo_root,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def tearDown(self) -> None:
        if self.process.poll() is None:
            try:
                self._request("shutdown", {})
            except Exception:
                self.process.terminate()
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.stdout is not None:
            self.process.stdout.close()
        if self.process.stderr is not None:
            self.process.stderr.close()
        self.process.wait(timeout=5)

    def test_shader_round_trip_on_cpu(self) -> None:
        health = self._request("health", {})
        self.assertTrue(health["ok"])

        compiled = self._request(
            "shader_compile",
            {
                "width": 4,
                "height": 4,
                "device": "CPU",
                "profile": "intel_npu_shader2_v1",
                "hint": "THROUGHPUT",
                "turbo": True,
            },
        )
        self.assertTrue(compiled["ok"])
        session_id = compiled["result"]["session_id"]

        rendered = self._request(
            "shader_run",
            {
                "session_id": session_id,
                "pos_x": 4.0,
                "pos_y": 72.0,
                "pos_z": -8.0,
                "yaw_degrees": 18.0,
                "pitch_degrees": 6.0,
                "time_seconds": 0.8,
                "sun_height": 0.8,
                "rain_strength": 0.1,
                "thunder_strength": 0.0,
                "block_light": 0.5,
                "sky_light": 0.9,
                "submerged_factor": 0.0,
                "quality_budget": 0.7,
                "optimization_pressure": 0.2,
            },
        )
        self.assertTrue(rendered["ok"])
        self.assertEqual(rendered["result"]["profile"], "intel_npu_shader2_v1")
        self.assertEqual(len(rendered["result"]["pixels_abgr"]), 16)

        released = self._request("shader_release", {"session_id": session_id})
        self.assertTrue(released["ok"])
        self.assertTrue(released["result"]["released"])

    def _request(self, op: str, payload: dict[str, object]) -> dict[str, object]:
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.process.stdin.write(json.dumps({"op": op, "payload": payload}) + "\n")
        self.process.stdin.flush()
        raw = self.process.stdout.readline()
        if not raw:
            stderr = ""
            if self.process.stderr is not None:
                stderr = self.process.stderr.read()
            raise RuntimeError(f"translator worker closed unexpectedly: {stderr}")
        return json.loads(raw)


if __name__ == "__main__":
    unittest.main()
