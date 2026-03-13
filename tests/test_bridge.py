from __future__ import annotations

import json
import socket
import struct
import threading
import time
import unittest
from urllib import request

import numpy as np

from npu_xmx.bridge import (
    BridgeState,
    GameBridgeServer,
    SocketBridgeServer,
    _ScenePolicyHints,
    _apply_realtime_budget_policy,
    benchmark_shader_profile,
)
from npu_xmx.engine import OpenVINOTensorCore


class BridgeStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.state = BridgeState(engine=OpenVINOTensorCore(preferred_devices=("CPU",)))

    def test_matmul_payload_returns_output(self) -> None:
        payload = {
            "lhs": [[1.0, 2.0], [3.0, 4.0]],
            "rhs": [[5.0, 6.0], [7.0, 8.0]],
            "device": "CPU",
        }
        response = self.state.matmul(payload)
        output = np.asarray(response["output"], dtype=np.float16)
        expected = np.asarray(payload["lhs"], dtype=np.float16) @ np.asarray(payload["rhs"], dtype=np.float16)
        np.testing.assert_allclose(output, expected, rtol=1e-2, atol=1e-2)

    def test_compile_run_and_release_linear_session(self) -> None:
        compile_response = self.state.compile_linear(
            {
                "weight": [[1.0, 0.0], [0.0, 1.0]],
                "batch_size": 2,
                "device": "CPU",
            }
        )
        session_id = compile_response["session_id"]

        run_response = self.state.run_linear(
            {
                "session_id": session_id,
                "inputs": [[10.0, 20.0], [30.0, 40.0]],
            }
        )
        np.testing.assert_allclose(
            np.asarray(run_response["output"], dtype=np.float16),
            np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float16),
            rtol=1e-2,
            atol=1e-2,
        )

        release_response = self.state.release_linear(session_id)
        self.assertTrue(release_response["released"])

        with self.assertRaises(KeyError):
            self.state.run_linear({"session_id": session_id, "inputs": [[1.0, 2.0]]})

    def test_compile_run_and_release_shader_session(self) -> None:
        compile_response = self.state.compile_shader(
            {
                "width": 4,
                "height": 4,
                "device": "CPU",
            }
        )
        session_id = compile_response["session_id"]
        self.assertEqual(compile_response["profile"], "intel_npu_gi_v3")
        self.assertIn("backend", compile_response)
        self.assertEqual(compile_response["input_features"], 80)

        run_response = self.state.run_shader(
            {
                "session_id": session_id,
                "pos_x": 12.0,
                "pos_y": 80.0,
                "pos_z": -6.0,
                "yaw_degrees": 45.0,
                "pitch_degrees": 12.0,
                "time_seconds": 1.25,
            }
        )
        self.assertEqual(run_response["width"], 4)
        self.assertEqual(run_response["height"], 4)
        self.assertEqual(len(run_response["pixels_abgr"]), 16)
        self.assertEqual(run_response["profile"], "intel_npu_gi_v3")
        self.assertIn("backend", run_response)
        self.assertGreaterEqual(run_response["mean_alpha"], 0.0)
        self.assertLessEqual(run_response["mean_alpha"], 1.0)
        self.assertGreaterEqual(run_response["mean_luma"], 0.0)
        self.assertLessEqual(run_response["mean_luma"], 1.0)

        release_response = self.state.release_shader(session_id)
        self.assertTrue(release_response["released"])

        with self.assertRaises(KeyError):
            self.state.run_shader({"session_id": session_id})

    def test_benchmark_shader_profile_on_cpu(self) -> None:
        results = benchmark_shader_profile(
            profile="intel_npu_gi_v3",
            width=4,
            height=4,
            devices=("CPU",),
            iterations=2,
            warmup=1,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].device, "CPU")
        self.assertEqual(results[0].profile, "intel_npu_gi_v3")
        self.assertGreater(results[0].average_ms, 0.0)
        self.assertGreater(results[0].updates_per_second, 0.0)

    def test_shader_run_accepts_explicit_scene_hints(self) -> None:
        compile_response = self.state.compile_shader(
            {
                "width": 4,
                "height": 4,
                "device": "CPU",
                "profile": "intel_npu_gi_v3",
            }
        )
        session_id = compile_response["session_id"]

        run_response = self.state.run_shader(
            {
                "session_id": session_id,
                "pos_x": 1.0,
                "pos_y": 70.0,
                "pos_z": -3.0,
                "yaw_degrees": 10.0,
                "pitch_degrees": 8.0,
                "time_seconds": 0.5,
                "sun_height": 0.9,
                "rain_strength": 0.4,
                "thunder_strength": 0.2,
                "block_light": 0.6,
                "sky_light": 0.3,
                "submerged_factor": 1.0,
                "quality_budget": 0.7,
                "optimization_pressure": 0.25,
            }
        )

        self.assertEqual(run_response["profile"], "intel_npu_gi_v3")
        self.assertEqual(len(run_response["pixels_abgr"]), 16)
        self.assertGreaterEqual(run_response["mean_alpha"], 0.0)
        self.assertLessEqual(run_response["mean_alpha"], 1.0)

    def test_compile_run_shader2_profile_session(self) -> None:
        compile_response = self.state.compile_shader(
            {
                "width": 4,
                "height": 4,
                "device": "CPU",
                "profile": "intel_npu_shader2_v1",
            }
        )
        session_id = compile_response["session_id"]
        self.assertEqual(compile_response["profile"], "intel_npu_shader2_v1")
        self.assertIn("backend", compile_response)
        self.assertEqual(compile_response["input_features"], 80)

        run_response = self.state.run_shader(
            {
                "session_id": session_id,
                "pos_x": 4.0,
                "pos_y": 73.0,
                "pos_z": -8.0,
                "yaw_degrees": 20.0,
                "pitch_degrees": 6.0,
                "time_seconds": 0.8,
                "quality_budget": 0.62,
                "optimization_pressure": 0.34,
            }
        )
        self.assertEqual(run_response["profile"], "intel_npu_shader2_v1")
        self.assertEqual(len(run_response["pixels_abgr"]), 16)
        self.assertGreaterEqual(run_response["mean_alpha"], 0.0)
        self.assertLessEqual(run_response["mean_alpha"], 1.0)

    def test_shader2_budget_policy_preserves_more_npu_signal(self) -> None:
        output = np.asarray([[0.82, 0.18, 0.90, 0.76]], dtype=np.float16)
        scene_hints = _ScenePolicyHints(
            quality_budget=0.45,
            optimization_pressure=0.75,
        )

        gi_v3 = _apply_realtime_budget_policy(output, scene_hints, profile="intel_npu_gi_v3")
        shader2 = _apply_realtime_budget_policy(output, scene_hints, profile="intel_npu_shader2_v1")

        self.assertGreater(abs(float(shader2[0, 0]) - 0.5), abs(float(gi_v3[0, 0]) - 0.5))
        self.assertGreater(abs(float(shader2[0, 1]) - 0.5), abs(float(gi_v3[0, 1]) - 0.5))
        self.assertGreater(float(shader2[0, 2]), float(gi_v3[0, 2]))
        self.assertGreater(float(shader2[0, 3]), float(gi_v3[0, 3]))


class BridgeHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        state = BridgeState(engine=OpenVINOTensorCore(preferred_devices=("CPU",)))
        cls.server = GameBridgeServer(("127.0.0.1", 0), state=state)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.1)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_health_endpoint(self) -> None:
        with request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["service"], "npu-xmx-bridge")
        self.assertIn("native_npu_backend_available", payload)

    def test_matmul_endpoint(self) -> None:
        body = json.dumps(
            {
                "lhs": [[1.0, 2.0]],
                "rhs": [[3.0], [4.0]],
                "device": "CPU",
            }
        ).encode("utf-8")
        req = request.Request(
            f"http://127.0.0.1:{self.port}/matmul",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        self.assertEqual(payload["shape"], [1, 1])
        self.assertAlmostEqual(float(payload["output"][0][0]), 11.0, places=2)

    def test_shader_endpoint(self) -> None:
        compile_body = json.dumps(
            {
                "width": 4,
                "height": 4,
                "device": "CPU",
                "profile": "intel_npu_gi_v3",
            }
        ).encode("utf-8")
        compile_req = request.Request(
            f"http://127.0.0.1:{self.port}/shader/compile",
            data=compile_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(compile_req, timeout=5) as response:
            compile_payload = json.loads(response.read().decode("utf-8"))

        run_body = json.dumps(
            {
                "session_id": compile_payload["session_id"],
                "pos_x": 3.0,
                "pos_y": 72.0,
                "pos_z": -10.0,
                "yaw_degrees": 30.0,
                "pitch_degrees": 5.0,
                "time_seconds": 0.75,
                "sun_height": 0.8,
                "rain_strength": 0.2,
                "thunder_strength": 0.1,
                "block_light": 0.5,
                "sky_light": 0.35,
                "submerged_factor": 0.0,
                "quality_budget": 0.78,
                "optimization_pressure": 0.18,
            }
        ).encode("utf-8")
        run_req = request.Request(
            f"http://127.0.0.1:{self.port}/shader/run",
            data=run_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(run_req, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(payload["width"], 4)
        self.assertEqual(payload["height"], 4)
        self.assertEqual(len(payload["pixels_abgr"]), 16)
        self.assertEqual(payload["profile"], "intel_npu_gi_v3")
        self.assertIn("backend", payload)

    def test_shader2_profile_endpoint(self) -> None:
        compile_body = json.dumps(
            {
                "width": 4,
                "height": 4,
                "device": "CPU",
                "profile": "intel_npu_shader2_v1",
            }
        ).encode("utf-8")
        compile_req = request.Request(
            f"http://127.0.0.1:{self.port}/shader/compile",
            data=compile_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(compile_req, timeout=5) as response:
            compile_payload = json.loads(response.read().decode("utf-8"))

        run_body = json.dumps(
            {
                "session_id": compile_payload["session_id"],
                "pos_x": 2.0,
                "pos_y": 71.0,
                "pos_z": -7.0,
                "yaw_degrees": 24.0,
                "pitch_degrees": 4.0,
                "time_seconds": 0.6,
                "quality_budget": 0.58,
                "optimization_pressure": 0.32,
            }
        ).encode("utf-8")
        run_req = request.Request(
            f"http://127.0.0.1:{self.port}/shader/run",
            data=run_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(run_req, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(payload["width"], 4)
        self.assertEqual(payload["height"], 4)
        self.assertEqual(len(payload["pixels_abgr"]), 16)
        self.assertEqual(payload["profile"], "intel_npu_shader2_v1")
        self.assertIn("backend", payload)


class BridgeSocketTests(unittest.TestCase):
    BINARY_MAGIC = b"NPXB"
    BINARY_HEADER = struct.Struct("<4sBBHI")
    BINARY_KIND_SHADER_RUN_REQUEST_V1 = 1
    BINARY_KIND_SHADER_RUN_REQUEST_V2 = 3
    BINARY_KIND_SHADER_RUN_REQUEST_V3 = 4
    BINARY_SHADER_RUN_REQUEST_V1 = struct.Struct("<16s6f")
    BINARY_SHADER_RUN_REQUEST_V2 = struct.Struct("<16s12f")
    BINARY_SHADER_RUN_REQUEST_V3 = struct.Struct("<16s14f")
    BINARY_SHADER_RUN_RESPONSE = struct.Struct("<IIffHHI")
    BINARY_ERROR = struct.Struct("<H")

    @classmethod
    def setUpClass(cls) -> None:
        state = BridgeState(engine=OpenVINOTensorCore(preferred_devices=("CPU",)))
        cls.server = SocketBridgeServer(("127.0.0.1", 0), state=state)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.1)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def _send_request(self, stream, payload: dict[str, object]) -> dict[str, object]:
        stream.write(json.dumps(payload).encode("utf-8") + b"\n")
        stream.flush()
        return json.loads(stream.readline().decode("utf-8"))

    def _send_binary_shader_run_v1(
        self,
        stream,
        *,
        session_id: str,
        pos_x: float,
        pos_y: float,
        pos_z: float,
        yaw_degrees: float,
        pitch_degrees: float,
        time_seconds: float,
    ) -> dict[str, object]:
        payload = self.BINARY_SHADER_RUN_REQUEST_V1.pack(
            bytes.fromhex(session_id),
            pos_x,
            pos_y,
            pos_z,
            yaw_degrees,
            pitch_degrees,
            time_seconds,
        )
        stream.write(self.BINARY_HEADER.pack(self.BINARY_MAGIC, 1, self.BINARY_KIND_SHADER_RUN_REQUEST_V1, 0, len(payload)))
        stream.write(payload)
        stream.flush()

        header = stream.read(self.BINARY_HEADER.size)
        magic, version, kind, _reserved, payload_length = self.BINARY_HEADER.unpack(header)
        self.assertEqual(magic, self.BINARY_MAGIC)
        self.assertEqual(version, 1)
        body = stream.read(payload_length)
        if kind == 255:
            (message_length,) = self.BINARY_ERROR.unpack(body[: self.BINARY_ERROR.size])
            message = body[self.BINARY_ERROR.size : self.BINARY_ERROR.size + message_length].decode("utf-8")
            raise AssertionError(f"Binary bridge returned error: {message}")

        width, height, mean_alpha, mean_luma, profile_length, backend_length, pixel_count = self.BINARY_SHADER_RUN_RESPONSE.unpack(
            body[: self.BINARY_SHADER_RUN_RESPONSE.size]
        )
        cursor = self.BINARY_SHADER_RUN_RESPONSE.size
        profile = body[cursor : cursor + profile_length].decode("utf-8")
        cursor += profile_length
        backend = body[cursor : cursor + backend_length].decode("utf-8")
        cursor += backend_length
        pixels = np.frombuffer(body[cursor : cursor + (pixel_count * 4)], dtype="<u4").view(np.int32)
        return {
            "width": width,
            "height": height,
            "mean_alpha": mean_alpha,
            "mean_luma": mean_luma,
            "profile": profile,
            "backend": backend,
            "pixels_abgr": pixels.astype(np.int64).tolist(),
        }

    def _send_binary_shader_run_v2(
        self,
        stream,
        *,
        session_id: str,
        pos_x: float,
        pos_y: float,
        pos_z: float,
        yaw_degrees: float,
        pitch_degrees: float,
        time_seconds: float,
        sun_height: float,
        rain_strength: float,
        thunder_strength: float,
        block_light: float,
        sky_light: float,
        submerged_factor: float,
    ) -> dict[str, object]:
        payload = self.BINARY_SHADER_RUN_REQUEST_V2.pack(
            bytes.fromhex(session_id),
            pos_x,
            pos_y,
            pos_z,
            yaw_degrees,
            pitch_degrees,
            time_seconds,
            sun_height,
            rain_strength,
            thunder_strength,
            block_light,
            sky_light,
            submerged_factor,
        )
        stream.write(self.BINARY_HEADER.pack(self.BINARY_MAGIC, 1, self.BINARY_KIND_SHADER_RUN_REQUEST_V2, 0, len(payload)))
        stream.write(payload)
        stream.flush()

        header = stream.read(self.BINARY_HEADER.size)
        magic, version, kind, _reserved, payload_length = self.BINARY_HEADER.unpack(header)
        self.assertEqual(magic, self.BINARY_MAGIC)
        self.assertEqual(version, 1)
        body = stream.read(payload_length)
        if kind == 255:
            (message_length,) = self.BINARY_ERROR.unpack(body[: self.BINARY_ERROR.size])
            message = body[self.BINARY_ERROR.size : self.BINARY_ERROR.size + message_length].decode("utf-8")
            raise AssertionError(f"Binary bridge returned error: {message}")

        width, height, mean_alpha, mean_luma, profile_length, backend_length, pixel_count = self.BINARY_SHADER_RUN_RESPONSE.unpack(
            body[: self.BINARY_SHADER_RUN_RESPONSE.size]
        )
        cursor = self.BINARY_SHADER_RUN_RESPONSE.size
        profile = body[cursor : cursor + profile_length].decode("utf-8")
        cursor += profile_length
        backend = body[cursor : cursor + backend_length].decode("utf-8")
        cursor += backend_length
        pixels = np.frombuffer(body[cursor : cursor + (pixel_count * 4)], dtype="<u4").view(np.int32)
        return {
            "width": width,
            "height": height,
            "mean_alpha": mean_alpha,
            "mean_luma": mean_luma,
            "profile": profile,
            "backend": backend,
            "pixels_abgr": pixels.astype(np.int64).tolist(),
        }

    def _send_binary_shader_run_v3(
        self,
        stream,
        *,
        session_id: str,
        pos_x: float,
        pos_y: float,
        pos_z: float,
        yaw_degrees: float,
        pitch_degrees: float,
        time_seconds: float,
        sun_height: float,
        rain_strength: float,
        thunder_strength: float,
        block_light: float,
        sky_light: float,
        submerged_factor: float,
        quality_budget: float,
        optimization_pressure: float,
    ) -> dict[str, object]:
        payload = self.BINARY_SHADER_RUN_REQUEST_V3.pack(
            bytes.fromhex(session_id),
            pos_x,
            pos_y,
            pos_z,
            yaw_degrees,
            pitch_degrees,
            time_seconds,
            sun_height,
            rain_strength,
            thunder_strength,
            block_light,
            sky_light,
            submerged_factor,
            quality_budget,
            optimization_pressure,
        )
        stream.write(self.BINARY_HEADER.pack(self.BINARY_MAGIC, 1, self.BINARY_KIND_SHADER_RUN_REQUEST_V3, 0, len(payload)))
        stream.write(payload)
        stream.flush()

        header = stream.read(self.BINARY_HEADER.size)
        magic, version, kind, _reserved, payload_length = self.BINARY_HEADER.unpack(header)
        self.assertEqual(magic, self.BINARY_MAGIC)
        self.assertEqual(version, 1)
        body = stream.read(payload_length)
        if kind == 255:
            (message_length,) = self.BINARY_ERROR.unpack(body[: self.BINARY_ERROR.size])
            message = body[self.BINARY_ERROR.size : self.BINARY_ERROR.size + message_length].decode("utf-8")
            raise AssertionError(f"Binary bridge returned error: {message}")

        width, height, mean_alpha, mean_luma, profile_length, backend_length, pixel_count = self.BINARY_SHADER_RUN_RESPONSE.unpack(
            body[: self.BINARY_SHADER_RUN_RESPONSE.size]
        )
        cursor = self.BINARY_SHADER_RUN_RESPONSE.size
        profile = body[cursor : cursor + profile_length].decode("utf-8")
        cursor += profile_length
        backend = body[cursor : cursor + backend_length].decode("utf-8")
        cursor += backend_length
        pixels = np.frombuffer(body[cursor : cursor + (pixel_count * 4)], dtype="<u4").view(np.int32)
        return {
            "width": width,
            "height": height,
            "mean_alpha": mean_alpha,
            "mean_luma": mean_luma,
            "profile": profile,
            "backend": backend,
            "pixels_abgr": pixels.astype(np.int64).tolist(),
        }

    def test_socket_matmul_request(self) -> None:
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as client:
            with client.makefile("rwb") as stream:
                payload = self._send_request(
                    stream,
                    {
                        "op": "matmul",
                        "payload": {
                            "lhs": [[1.0, 2.0]],
                            "rhs": [[3.0], [4.0]],
                            "device": "CPU",
                        },
                    },
                )

        self.assertTrue(payload["ok"])
        result = payload["result"]
        self.assertEqual(result["shape"], [1, 1])
        self.assertAlmostEqual(float(result["output"][0][0]), 11.0, places=2)

    def test_socket_linear_session_round_trip(self) -> None:
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as client:
            with client.makefile("rwb") as stream:
                compile_payload = self._send_request(
                    stream,
                    {
                        "op": "linear_compile",
                        "payload": {
                            "weight": [[1.0, 0.0], [0.0, 1.0]],
                            "batch_size": 2,
                            "device": "CPU",
                        },
                    },
                )
                self.assertTrue(compile_payload["ok"])
                session_id = compile_payload["result"]["session_id"]

                run_payload = self._send_request(
                    stream,
                    {
                        "op": "linear_run",
                        "payload": {
                            "session_id": session_id,
                            "inputs": [[10.0, 20.0], [30.0, 40.0]],
                        },
                    },
                )
                self.assertTrue(run_payload["ok"])
                np.testing.assert_allclose(
                    np.asarray(run_payload["result"]["output"], dtype=np.float16),
                    np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float16),
                    rtol=1e-2,
                    atol=1e-2,
                )

                release_payload = self._send_request(
                    stream,
                    {
                        "op": "linear_release",
                        "payload": {"session_id": session_id},
                    },
                )
                self.assertTrue(release_payload["ok"])
                self.assertTrue(release_payload["result"]["released"])

    def test_socket_shader_round_trip(self) -> None:
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as client:
            with client.makefile("rwb") as stream:
                compile_payload = self._send_request(
                    stream,
                    {
                        "op": "shader_compile",
                        "payload": {
                            "width": 4,
                            "height": 4,
                            "device": "CPU",
                        },
                    },
                )
                self.assertTrue(compile_payload["ok"])
                session_id = compile_payload["result"]["session_id"]

                run_payload = self._send_request(
                    stream,
                    {
                        "op": "shader_run",
                        "payload": {
                            "session_id": session_id,
                            "pos_x": 3.0,
                            "pos_y": 70.0,
                            "pos_z": -2.0,
                            "yaw_degrees": 15.0,
                            "pitch_degrees": 7.0,
                            "time_seconds": 0.5,
                        },
                    },
                )
                self.assertTrue(run_payload["ok"])
                self.assertEqual(run_payload["result"]["width"], 4)
                self.assertEqual(run_payload["result"]["height"], 4)
                self.assertEqual(len(run_payload["result"]["pixels_abgr"]), 16)
                self.assertEqual(run_payload["result"]["profile"], "intel_npu_gi_v3")
                self.assertIn("backend", run_payload["result"])

    def test_socket_binary_shader_v1_round_trip(self) -> None:
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as client:
            with client.makefile("rwb") as stream:
                compile_payload = self._send_request(
                    stream,
                    {
                        "op": "shader_compile",
                        "payload": {
                            "width": 4,
                            "height": 4,
                            "device": "CPU",
                        },
                    },
                )
                self.assertTrue(compile_payload["ok"])
                session_id = compile_payload["result"]["session_id"]

                run_payload = self._send_binary_shader_run_v1(
                    stream,
                    session_id=session_id,
                    pos_x=3.0,
                    pos_y=70.0,
                    pos_z=-2.0,
                    yaw_degrees=15.0,
                    pitch_degrees=7.0,
                    time_seconds=0.5,
                )
                self.assertEqual(run_payload["width"], 4)
                self.assertEqual(run_payload["height"], 4)
                self.assertEqual(len(run_payload["pixels_abgr"]), 16)
                self.assertEqual(run_payload["profile"], "intel_npu_gi_v3")
                self.assertIn("backend", run_payload)

                neutral_payload = self._send_request(
                    stream,
                    {
                        "op": "shader_run",
                        "payload": {
                            "session_id": session_id,
                            "pos_x": 3.0,
                            "pos_y": 70.0,
                            "pos_z": -2.0,
                            "yaw_degrees": 15.0,
                            "pitch_degrees": 7.0,
                            "time_seconds": 0.5,
                            "sun_height": 0.5,
                            "rain_strength": 0.0,
                            "thunder_strength": 0.0,
                            "block_light": 0.0,
                            "sky_light": 1.0,
                            "submerged_factor": 0.0,
                            "quality_budget": 1.0,
                            "optimization_pressure": 0.0,
                        },
                    },
                )
                self.assertTrue(neutral_payload["ok"])
                self.assertEqual(run_payload["pixels_abgr"], neutral_payload["result"]["pixels_abgr"])

    def test_socket_binary_shader_v2_round_trip(self) -> None:
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as client:
            with client.makefile("rwb") as stream:
                compile_payload = self._send_request(
                    stream,
                    {
                        "op": "shader_compile",
                        "payload": {
                            "width": 4,
                            "height": 4,
                            "device": "CPU",
                            "profile": "intel_npu_gi_v3",
                        },
                    },
                )
                self.assertTrue(compile_payload["ok"])
                session_id = compile_payload["result"]["session_id"]

                run_payload = self._send_binary_shader_run_v2(
                    stream,
                    session_id=session_id,
                    pos_x=3.0,
                    pos_y=70.0,
                    pos_z=-2.0,
                    yaw_degrees=15.0,
                    pitch_degrees=7.0,
                    time_seconds=0.5,
                    sun_height=0.95,
                    rain_strength=0.35,
                    thunder_strength=0.15,
                    block_light=0.55,
                    sky_light=0.25,
                    submerged_factor=1.0,
                )
                self.assertEqual(run_payload["width"], 4)
                self.assertEqual(run_payload["height"], 4)
                self.assertEqual(len(run_payload["pixels_abgr"]), 16)
                self.assertEqual(run_payload["profile"], "intel_npu_gi_v3")
                self.assertIn("backend", run_payload)

    def test_socket_binary_shader_v3_round_trip(self) -> None:
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as client:
            with client.makefile("rwb") as stream:
                compile_payload = self._send_request(
                    stream,
                    {
                        "op": "shader_compile",
                        "payload": {
                            "width": 4,
                            "height": 4,
                            "device": "CPU",
                            "profile": "intel_npu_gi_v3",
                        },
                    },
                )
                self.assertTrue(compile_payload["ok"])
                session_id = compile_payload["result"]["session_id"]

                run_payload = self._send_binary_shader_run_v3(
                    stream,
                    session_id=session_id,
                    pos_x=3.0,
                    pos_y=70.0,
                    pos_z=-2.0,
                    yaw_degrees=15.0,
                    pitch_degrees=7.0,
                    time_seconds=0.5,
                    sun_height=0.95,
                    rain_strength=0.35,
                    thunder_strength=0.15,
                    block_light=0.55,
                    sky_light=0.25,
                    submerged_factor=1.0,
                    quality_budget=0.68,
                    optimization_pressure=0.22,
                )
                self.assertEqual(run_payload["width"], 4)
                self.assertEqual(run_payload["height"], 4)
                self.assertEqual(len(run_payload["pixels_abgr"]), 16)
                self.assertEqual(run_payload["profile"], "intel_npu_gi_v3")
                self.assertIn("backend", run_payload)

    def test_socket_shader2_round_trip(self) -> None:
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as client:
            with client.makefile("rwb") as stream:
                compile_payload = self._send_request(
                    stream,
                    {
                        "op": "shader_compile",
                        "payload": {
                            "width": 4,
                            "height": 4,
                            "device": "CPU",
                            "profile": "intel_npu_shader2_v1",
                        },
                    },
                )
                self.assertTrue(compile_payload["ok"])
                session_id = compile_payload["result"]["session_id"]

                run_payload = self._send_binary_shader_run_v3(
                    stream,
                    session_id=session_id,
                    pos_x=2.0,
                    pos_y=71.0,
                    pos_z=-5.0,
                    yaw_degrees=12.0,
                    pitch_degrees=6.0,
                    time_seconds=0.4,
                    sun_height=0.88,
                    rain_strength=0.22,
                    thunder_strength=0.08,
                    block_light=0.42,
                    sky_light=0.36,
                    submerged_factor=0.0,
                    quality_budget=0.61,
                    optimization_pressure=0.37,
                )
                self.assertEqual(run_payload["width"], 4)
                self.assertEqual(run_payload["height"], 4)
                self.assertEqual(len(run_payload["pixels_abgr"]), 16)
                self.assertEqual(run_payload["profile"], "intel_npu_shader2_v1")
                self.assertIn("backend", run_payload)
