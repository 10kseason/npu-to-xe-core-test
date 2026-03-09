from __future__ import annotations

import json
import socket
import threading
import time
import unittest
from urllib import request

import numpy as np

from npu_xmx.bridge import BridgeState, GameBridgeServer, SocketBridgeServer
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
        self.assertGreaterEqual(run_response["mean_alpha"], 0.0)
        self.assertLessEqual(run_response["mean_alpha"], 1.0)
        self.assertGreaterEqual(run_response["mean_luma"], 0.0)
        self.assertLessEqual(run_response["mean_luma"], 1.0)

        release_response = self.state.release_shader(session_id)
        self.assertTrue(release_response["released"])

        with self.assertRaises(KeyError):
            self.state.run_shader({"session_id": session_id})


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


class BridgeSocketTests(unittest.TestCase):
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
