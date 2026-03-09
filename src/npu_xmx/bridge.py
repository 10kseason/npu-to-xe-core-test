from __future__ import annotations

import json
import socketserver
import threading
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlparse

import numpy as np

from .engine import LinearBlock, OpenVINOTensorCore
import openvino as ov
from openvino.runtime import opset13


@dataclass
class _LinearSession:
    session_id: str
    block: LinearBlock


@dataclass
class _ShaderSession:
    session_id: str
    block: "_ShaderFieldBlock"
    width: int
    height: int


@dataclass
class _ShaderFieldBlock:
    runner: Callable[[np.ndarray], np.ndarray]
    device: str
    input_features: int
    output_features: int
    max_batch: int
    dtype: np.dtype

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        matrix = np.asarray(inputs, dtype=self.dtype)
        matrix = _ensure_rank2(matrix, "shader_inputs")
        if matrix.shape[1] != self.input_features:
            raise ValueError(
                f"Expected {self.input_features} shader features, got {matrix.shape[1]}"
            )
        if matrix.shape[0] != self.max_batch:
            raise ValueError(
                f"Expected shader batch {self.max_batch}, got {matrix.shape[0]}"
            )
        return np.asarray(self.runner(matrix), dtype=self.dtype)


class BridgeState:
    def __init__(self, engine: OpenVINOTensorCore | None = None) -> None:
        self.engine = engine or OpenVINOTensorCore()
        self._sessions: dict[str, _LinearSession] = {}
        self._shader_sessions: dict[str, _ShaderSession] = {}
        self._lock = threading.Lock()

    def devices_payload(self) -> list[dict[str, Any]]:
        return [self.engine.device_info(device) for device in self.engine.available_devices()]

    def health_payload(self) -> dict[str, Any]:
        return {
            "ok": True,
            "service": "npu-xmx-bridge",
            "native_npu_backend_available": self.engine.native_npu_backend_available(),
            "devices": self.devices_payload(),
        }

    def matmul(self, payload: dict[str, Any]) -> dict[str, Any]:
        lhs = _array_from_payload(payload, "lhs", ndim=2)
        rhs = _array_from_payload(payload, "rhs", ndim=2)
        output = self.engine.matmul(
            lhs,
            rhs,
            device=str(payload.get("device", "AUTO")),
            pad_to=int(payload.get("pad_to", 16)),
            performance_hint=str(payload.get("hint", "THROUGHPUT")),
            turbo=bool(payload.get("turbo", True)),
        )
        return {
            "device": self.engine.resolve_device(str(payload.get("device", "AUTO"))),
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "output": output.tolist(),
        }

    def compile_linear(self, payload: dict[str, Any]) -> dict[str, Any]:
        weight = _array_from_payload(payload, "weight", ndim=2)
        bias_payload = payload.get("bias")
        bias = None if bias_payload is None else np.asarray(bias_payload, dtype=np.float16)
        batch_size = int(payload.get("batch_size", 1))
        device = str(payload.get("device", "AUTO"))
        pad_to = int(payload.get("pad_to", 16))
        hint = str(payload.get("hint", "THROUGHPUT"))
        turbo = bool(payload.get("turbo", True))

        block = self.engine.compile_linear(
            weight,
            bias=bias,
            batch_size=batch_size,
            device=device,
            dtype=np.float16,
            pad_to=pad_to,
            performance_hint=hint,
            turbo=turbo,
        )
        session_id = uuid.uuid4().hex
        with self._lock:
            self._sessions[session_id] = _LinearSession(session_id=session_id, block=block)

        return {
            "session_id": session_id,
            "device": block.device,
            "input_features": block.input_features,
            "output_features": block.output_features,
            "max_batch": block.max_batch,
            "dtype": str(block.dtype),
        }

    def run_linear(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id", ""))
        inputs = _array_from_payload(payload, "inputs", ndim=2)
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown session_id: {session_id}")

        output = session.block(inputs)
        return {
            "session_id": session_id,
            "device": session.block.device,
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "output": output.tolist(),
        }

    def release_linear(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            removed = self._sessions.pop(session_id, None)
        return {
            "session_id": session_id,
            "released": removed is not None,
        }

    def compile_shader(self, payload: dict[str, Any]) -> dict[str, Any]:
        width = int(payload.get("width", 32))
        height = int(payload.get("height", width))
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive integers")

        device = str(payload.get("device", "AUTO"))
        hint = str(payload.get("hint", "THROUGHPUT"))
        turbo = bool(payload.get("turbo", True))
        resolved_device = self.engine.resolve_device(device)
        batch_size = width * height

        block = _compile_shader_field_block(
            engine=self.engine,
            batch_size=batch_size,
            device=resolved_device,
            performance_hint=hint,
            turbo=turbo,
        )
        session_id = uuid.uuid4().hex
        with self._lock:
            self._shader_sessions[session_id] = _ShaderSession(
                session_id=session_id,
                block=block,
                width=width,
                height=height,
            )

        return {
            "session_id": session_id,
            "device": block.device,
            "width": width,
            "height": height,
            "input_features": block.input_features,
            "output_features": block.output_features,
            "max_batch": block.max_batch,
            "dtype": str(block.dtype),
        }

    def run_shader(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id", ""))
        with self._lock:
            session = self._shader_sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown session_id: {session_id}")

        pos_x = float(payload.get("pos_x", 0.0))
        pos_y = float(payload.get("pos_y", 64.0))
        pos_z = float(payload.get("pos_z", 0.0))
        yaw_degrees = float(payload.get("yaw_degrees", 0.0))
        pitch_degrees = float(payload.get("pitch_degrees", 0.0))
        time_seconds = float(payload.get("time_seconds", 0.0))

        shader_inputs = _build_shader_inputs(
            width=session.width,
            height=session.height,
            pos_x=pos_x,
            pos_y=pos_y,
            pos_z=pos_z,
            yaw_degrees=yaw_degrees,
            pitch_degrees=pitch_degrees,
            time_seconds=time_seconds,
        )
        output = session.block(shader_inputs)
        pixels_abgr, mean_alpha, mean_luma = _pack_shader_pixels(output)
        return {
            "session_id": session_id,
            "device": session.block.device,
            "width": session.width,
            "height": session.height,
            "pixels_abgr": pixels_abgr,
            "mean_alpha": mean_alpha,
            "mean_luma": mean_luma,
        }

    def release_shader(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            removed = self._shader_sessions.pop(session_id, None)
        return {
            "session_id": session_id,
            "released": removed is not None,
        }

    def dispatch(self, op: str, payload: dict[str, Any]) -> Any:
        if op == "health":
            return self.health_payload()
        if op == "devices":
            return self.devices_payload()
        if op == "matmul":
            return self.matmul(payload)
        if op == "linear_compile":
            return self.compile_linear(payload)
        if op == "linear_run":
            return self.run_linear(payload)
        if op == "linear_release":
            return self.release_linear(str(payload.get("session_id", "")))
        if op == "shader_compile":
            return self.compile_shader(payload)
        if op == "shader_run":
            return self.run_shader(payload)
        if op == "shader_release":
            return self.release_shader(str(payload.get("session_id", "")))
        raise KeyError(f"Unknown op: {op}")


def _shader_weight_matrix() -> np.ndarray:
    rng = np.random.default_rng(19)
    return {
        "w1": (rng.standard_normal((16, 64), dtype=np.float32) * 0.34).astype(np.float16),
        "b1": (rng.standard_normal((64,), dtype=np.float32) * 0.06).astype(np.float16),
        "w2": (rng.standard_normal((64, 48), dtype=np.float32) * 0.24).astype(np.float16),
        "b2": (rng.standard_normal((48,), dtype=np.float32) * 0.05).astype(np.float16),
        "w3": (rng.standard_normal((48, 4), dtype=np.float32) * 0.18).astype(np.float16),
        "b3": np.asarray([0.0, 0.0, 0.12, -0.28], dtype=np.float16),
    }


def _compile_shader_field_block(
    *,
    engine: OpenVINOTensorCore,
    batch_size: int,
    device: str,
    performance_hint: str,
    turbo: bool,
) -> _ShaderFieldBlock:
    weights = _shader_weight_matrix()
    input_node = opset13.parameter([batch_size, 16], np.float16, name="shader_input")

    hidden1 = opset13.matmul(input_node, opset13.constant(weights["w1"]), False, False)
    hidden1 = opset13.add(hidden1, opset13.constant(weights["b1"]))
    hidden1 = opset13.relu(hidden1)

    hidden2 = opset13.matmul(hidden1, opset13.constant(weights["w2"]), False, False)
    hidden2 = opset13.add(hidden2, opset13.constant(weights["b2"]))
    hidden2 = opset13.relu(hidden2)

    output = opset13.matmul(hidden2, opset13.constant(weights["w3"]), False, False)
    output = opset13.add(output, opset13.constant(weights["b3"]))
    output = opset13.sigmoid(output)

    model = ov.Model([output], [input_node], "shader_field")
    compiled = engine.core.compile_model(
        model,
        device,
        engine._compile_config(device, performance_hint, turbo),
    )
    request = compiled.create_infer_request()
    input_name = compiled.input(0).get_any_name()
    output_port = compiled.output(0)

    def runner(shader_inputs: np.ndarray) -> np.ndarray:
        result = request.infer({input_name: shader_inputs})
        return np.array(result[output_port], copy=False)

    return _ShaderFieldBlock(
        runner=runner,
        device=device,
        input_features=16,
        output_features=4,
        max_batch=batch_size,
        dtype=np.dtype(np.float16),
    )


def _build_shader_inputs(
    *,
    width: int,
    height: int,
    pos_x: float,
    pos_y: float,
    pos_z: float,
    yaw_degrees: float,
    pitch_degrees: float,
    time_seconds: float,
) -> np.ndarray:
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    time_feature = np.full_like(grid_x, ((time_seconds * 0.17) % 2.0) - 1.0)
    yaw_feature = np.full_like(grid_x, np.clip(yaw_degrees / 180.0, -1.0, 1.0))
    pitch_feature = np.full_like(grid_x, np.clip(pitch_degrees / 90.0, -1.0, 1.0))
    pos_x_feature = np.full_like(grid_x, np.clip(pos_x / 96.0, -1.0, 1.0))
    pos_y_feature = np.full_like(grid_x, np.clip((pos_y - 64.0) / 96.0, -1.0, 1.0))
    pos_z_feature = np.full_like(grid_x, np.clip(pos_z / 96.0, -1.0, 1.0))
    u2 = grid_x * grid_x
    v2 = grid_y * grid_y
    uv = grid_x * grid_y
    time_u = time_feature * grid_x
    time_v = time_feature * grid_y
    yaw_u = yaw_feature * grid_x
    pitch_v = pitch_feature * grid_y
    radius2 = u2 + v2

    features = np.stack(
        [
            grid_x,
            grid_y,
            time_feature,
            yaw_feature,
            pitch_feature,
            pos_x_feature,
            pos_y_feature,
            pos_z_feature,
            u2,
            v2,
            uv,
            time_u,
            time_v,
            yaw_u,
            pitch_v,
            radius2,
        ],
        axis=-1,
    )
    return features.reshape(width * height, 16).astype(np.float16)


def _pack_shader_pixels(output: np.ndarray) -> tuple[list[int], float, float]:
    rgba = np.asarray(output, dtype=np.float32)
    warp_u = np.clip(rgba[:, 0], 0.0, 1.0)
    warp_v = np.clip(rgba[:, 1], 0.0, 1.0)
    tint = np.clip(rgba[:, 2], 0.0, 1.0)
    blend = np.clip(rgba[:, 3], 0.0, 1.0)

    rgb = np.stack([warp_u, warp_v, tint], axis=1)
    alpha = np.clip(0.10 + blend * 0.48, 0.08, 0.64)

    red = np.rint(rgb[:, 0] * 255.0).astype(np.uint32)
    green = np.rint(rgb[:, 1] * 255.0).astype(np.uint32)
    blue = np.rint(rgb[:, 2] * 255.0).astype(np.uint32)
    alpha_u32 = np.rint(alpha * 255.0).astype(np.uint32)
    packed = ((alpha_u32 << 24) | (blue << 16) | (green << 8) | red).astype(np.uint32)
    signed = packed.view(np.int32).astype(np.int64)
    luma = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    return signed.tolist(), float(alpha.mean()), float(luma.mean())


def _array_from_payload(payload: dict[str, Any], key: str, *, ndim: int) -> np.ndarray:
    if key not in payload:
        raise KeyError(f"Missing required field: {key}")
    array = np.asarray(payload[key], dtype=np.float16)
    if array.ndim != ndim:
        raise ValueError(f"{key} must be rank-{ndim}, got shape {array.shape}")
    return array


def _ensure_rank2(array: np.ndarray, name: str) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"{name} must be rank-2, got shape {array.shape}")
    return array


class BridgeRequestHandler(BaseHTTPRequestHandler):
    server: "GameBridgeServer"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._write_json(HTTPStatus.OK, self.server.state.health_payload())
            return
        if path == "/devices":
            self._write_json(HTTPStatus.OK, self.server.state.devices_payload())
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown path: {path}"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/matmul":
                self._write_json(HTTPStatus.OK, self.server.state.matmul(payload))
                return
            if path == "/linear/compile":
                self._write_json(HTTPStatus.OK, self.server.state.compile_linear(payload))
                return
            if path == "/linear/run":
                self._write_json(HTTPStatus.OK, self.server.state.run_linear(payload))
                return
            if path == "/shader/compile":
                self._write_json(HTTPStatus.OK, self.server.state.compile_shader(payload))
                return
            if path == "/shader/run":
                self._write_json(HTTPStatus.OK, self.server.state.run_shader(payload))
                return
            self._write_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown path: {path}"})
        except KeyError as exc:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except ValueError as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        session_id = path.rsplit("/", 1)[-1]
        if path.startswith("/linear/session/"):
            self._write_json(HTTPStatus.OK, self.server.state.release_linear(session_id))
            return
        if path.startswith("/shader/session/"):
            self._write_json(HTTPStatus.OK, self.server.state.release_shader(session_id))
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown path: {path}"})

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _write_json(self, status: HTTPStatus, payload: Any) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")


class GameBridgeServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        state: BridgeState | None = None,
    ) -> None:
        super().__init__(server_address, BridgeRequestHandler)
        self.state = state or BridgeState()


class SocketBridgeRequestHandler(socketserver.StreamRequestHandler):
    server: "SocketBridgeServer"

    def handle(self) -> None:
        while True:
            raw = self.rfile.readline()
            if not raw:
                return

            try:
                request = json.loads(raw.decode("utf-8"))
                op = str(request.get("op", ""))
                payload = request.get("payload", {})
                response = {"ok": True, "result": self.server.state.dispatch(op, payload)}
            except KeyError as exc:
                response = {"ok": False, "error": str(exc)}
            except ValueError as exc:
                response = {"ok": False, "error": str(exc)}
            except Exception as exc:
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

            encoded = json.dumps(response).encode("utf-8") + b"\n"
            self.wfile.write(encoded)
            self.wfile.flush()


class SocketBridgeServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        state: BridgeState | None = None,
    ) -> None:
        super().__init__(server_address, SocketBridgeRequestHandler)
        self.state = state or BridgeState()


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    socket_port: int = 8766,
    engine: OpenVINOTensorCore | None = None,
) -> None:
    state = BridgeState(engine=engine)
    server = GameBridgeServer((host, port), state=state)
    socket_server = SocketBridgeServer((host, socket_port), state=state)
    socket_thread = threading.Thread(target=socket_server.serve_forever, daemon=True)
    socket_thread.start()
    print(f"npu-xmx bridge listening on http://{host}:{port}")
    print(f"npu-xmx socket bridge listening on tcp://{host}:{socket_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        socket_server.shutdown()
        socket_server.server_close()
        socket_thread.join(timeout=2)
