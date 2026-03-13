from __future__ import annotations

import json
import socketserver
import struct
import threading
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from time import perf_counter
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
    profile: str


@dataclass
class _ShaderFieldBlock:
    runner: Callable[[np.ndarray], np.ndarray]
    device: str
    input_features: int
    output_features: int
    max_batch: int
    dtype: np.dtype
    profile: str
    backend: str

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


@dataclass
class _RenderedShaderFrame:
    session_id: str
    device: str
    width: int
    height: int
    pixels_abgr: np.ndarray
    mean_alpha: float
    mean_luma: float
    profile: str
    backend: str


@dataclass(frozen=True)
class _ScenePolicyHints:
    sun_height: float = 0.5
    rain_strength: float = 0.0
    thunder_strength: float = 0.0
    block_light: float = 0.0
    sky_light: float = 1.0
    submerged_factor: float = 0.0
    quality_budget: float = 1.0
    optimization_pressure: float = 0.0


@dataclass(frozen=True)
class ShaderProfileBenchmarkResult:
    device: str
    profile: str
    backend: str
    width: int
    height: int
    average_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float
    iterations: int
    updates_per_second: float
    pixels_per_second: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "profile": self.profile,
            "backend": self.backend,
            "width": self.width,
            "height": self.height,
            "average_ms": self.average_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "p95_ms": self.p95_ms,
            "iterations": self.iterations,
            "updates_per_second": self.updates_per_second,
            "pixels_per_second": self.pixels_per_second,
        }


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
        profile = str(payload.get("profile", "intel_npu_gi_v3"))
        resolved_device = self.engine.resolve_device(device)
        batch_size = width * height

        block = _compile_shader_field_block(
            engine=self.engine,
            batch_size=batch_size,
            device=resolved_device,
            performance_hint=hint,
            turbo=turbo,
            profile=profile,
        )
        session_id = uuid.uuid4().hex
        with self._lock:
            self._shader_sessions[session_id] = _ShaderSession(
                session_id=session_id,
                block=block,
                width=width,
                height=height,
                profile=block.profile,
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
            "profile": block.profile,
            "backend": block.backend,
        }

    def run_shader(self, payload: dict[str, Any]) -> dict[str, Any]:
        frame = self.render_shader_frame(payload)
        return {
            "session_id": frame.session_id,
            "device": frame.device,
            "width": frame.width,
            "height": frame.height,
            "pixels_abgr": frame.pixels_abgr.view(np.int32).astype(np.int64, copy=False).tolist(),
            "mean_alpha": frame.mean_alpha,
            "mean_luma": frame.mean_luma,
            "profile": frame.profile,
            "backend": frame.backend,
        }

    def render_shader_frame(self, payload: dict[str, Any]) -> _RenderedShaderFrame:
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
        scene_hints = _scene_policy_hints_from_payload(payload)

        shader_inputs = _build_shader_inputs(
            width=session.width,
            height=session.height,
            pos_x=pos_x,
            pos_y=pos_y,
            pos_z=pos_z,
            yaw_degrees=yaw_degrees,
            pitch_degrees=pitch_degrees,
            time_seconds=time_seconds,
            sun_height=scene_hints.sun_height,
            rain_strength=scene_hints.rain_strength,
            thunder_strength=scene_hints.thunder_strength,
            block_light=scene_hints.block_light,
            sky_light=scene_hints.sky_light,
            submerged_factor=scene_hints.submerged_factor,
            quality_budget=scene_hints.quality_budget,
            optimization_pressure=scene_hints.optimization_pressure,
            input_features=session.block.input_features,
        )
        output = _apply_realtime_budget_policy(
            session.block(shader_inputs),
            scene_hints,
            profile=session.profile,
        )
        pixels_abgr, mean_alpha, mean_luma = _pack_shader_pixels_array(output)
        return _RenderedShaderFrame(
            session_id=session_id,
            device=session.block.device,
            width=session.width,
            height=session.height,
            pixels_abgr=pixels_abgr,
            mean_alpha=mean_alpha,
            mean_luma=mean_luma,
            profile=session.profile,
            backend=session.block.backend,
        )

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


def _shader_weight_matrix() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(19)
    return {
        "w1": (rng.standard_normal((16, 64), dtype=np.float32) * 0.34).astype(np.float16),
        "b1": (rng.standard_normal((64,), dtype=np.float32) * 0.06).astype(np.float16),
        "w2": (rng.standard_normal((64, 48), dtype=np.float32) * 0.24).astype(np.float16),
        "b2": (rng.standard_normal((48,), dtype=np.float32) * 0.05).astype(np.float16),
        "w3": (rng.standard_normal((48, 4), dtype=np.float32) * 0.18).astype(np.float16),
        "b3": np.asarray([0.0, 0.0, 0.12, -0.28], dtype=np.float16),
    }


def _native_shader_weight_matrix() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(41)
    return {
        "w1": (rng.standard_normal((16, 80), dtype=np.float32) * 0.28).astype(np.float16),
        "b1": (rng.standard_normal((80,), dtype=np.float32) * 0.05).astype(np.float16),
        "w2": (rng.standard_normal((80, 48), dtype=np.float32) * 0.22).astype(np.float16),
        "b2": (rng.standard_normal((48,), dtype=np.float32) * 0.04).astype(np.float16),
        "w3": (rng.standard_normal((48, 4), dtype=np.float32) * 0.16).astype(np.float16),
        "b3": np.asarray([0.02, -0.03, 0.10, -0.24], dtype=np.float16),
    }


def _native_shader_weight_matrix_v2() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(73)
    return {
        "w1": (rng.standard_normal((24, 128), dtype=np.float32) * 0.24).astype(np.float16),
        "b1": (rng.standard_normal((128,), dtype=np.float32) * 0.04).astype(np.float16),
        "w2": (rng.standard_normal((128, 96), dtype=np.float32) * 0.18).astype(np.float16),
        "b2": (rng.standard_normal((96,), dtype=np.float32) * 0.035).astype(np.float16),
        "w3": (rng.standard_normal((96, 48), dtype=np.float32) * 0.16).astype(np.float16),
        "b3": (rng.standard_normal((48,), dtype=np.float32) * 0.03).astype(np.float16),
        "w4": (rng.standard_normal((48, 4), dtype=np.float32) * 0.12).astype(np.float16),
        "b4": np.asarray([0.10, 0.02, 0.16, -0.12], dtype=np.float16),
    }


def _native_shader_weight_matrix_v3() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(97)
    return {
        "w1": (rng.standard_normal((32, 192), dtype=np.float32) * 0.22).astype(np.float16),
        "b1": (rng.standard_normal((192,), dtype=np.float32) * 0.035).astype(np.float16),
        "w2": (rng.standard_normal((192, 160), dtype=np.float32) * 0.18).astype(np.float16),
        "b2": (rng.standard_normal((160,), dtype=np.float32) * 0.03).astype(np.float16),
        "w3": (rng.standard_normal((160, 96), dtype=np.float32) * 0.15).astype(np.float16),
        "b3": (rng.standard_normal((96,), dtype=np.float32) * 0.028).astype(np.float16),
        "w4": (rng.standard_normal((96, 48), dtype=np.float32) * 0.12).astype(np.float16),
        "b4": (rng.standard_normal((48,), dtype=np.float32) * 0.022).astype(np.float16),
        "w5": (rng.standard_normal((48, 4), dtype=np.float32) * 0.10).astype(np.float16),
        "b5": np.asarray([0.06, 0.00, 0.14, -0.08], dtype=np.float16),
    }


def _native_shader_weight_matrix_rtgl() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(131)
    return {
        "w1": (rng.standard_normal((32, 160), dtype=np.float32) * 0.20).astype(np.float16),
        "b1": (rng.standard_normal((160,), dtype=np.float32) * 0.030).astype(np.float16),
        "w2": (rng.standard_normal((160, 96), dtype=np.float32) * 0.16).astype(np.float16),
        "b2": (rng.standard_normal((96,), dtype=np.float32) * 0.025).astype(np.float16),
        "w3": (rng.standard_normal((96, 32), dtype=np.float32) * 0.14).astype(np.float16),
        "b3": (rng.standard_normal((32,), dtype=np.float32) * 0.020).astype(np.float16),
        "m1": (rng.standard_normal((32, 16), dtype=np.float32) * 0.18).astype(np.float16),
        "m2": (rng.standard_normal((16, 8), dtype=np.float32) * 0.15).astype(np.float16),
        "m3": (rng.standard_normal((8, 4), dtype=np.float32) * 0.12).astype(np.float16),
        "b4": np.asarray([0.00, 0.00, 0.18, -0.06], dtype=np.float16),
    }


def _native_shader_weight_matrix_gi() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(149)
    return {
        "w1": (rng.standard_normal((32, 96), dtype=np.float32) * 0.18).astype(np.float16),
        "b1": (rng.standard_normal((96,), dtype=np.float32) * 0.028).astype(np.float16),
        "w2": (rng.standard_normal((96, 48), dtype=np.float32) * 0.15).astype(np.float16),
        "b2": (rng.standard_normal((48,), dtype=np.float32) * 0.022).astype(np.float16),
        "m1": (rng.standard_normal((48, 16), dtype=np.float32) * 0.14).astype(np.float16),
        "m2": (rng.standard_normal((16, 4), dtype=np.float32) * 0.12).astype(np.float16),
        "b3": np.asarray([0.00, 0.00, 0.16, -0.02], dtype=np.float16),
    }


def _native_shader_weight_matrix_gi_v2() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(167)
    return {
        "w1": (rng.standard_normal((32, 88), dtype=np.float32) * 0.17).astype(np.float16),
        "b1": (rng.standard_normal((88,), dtype=np.float32) * 0.025).astype(np.float16),
        "w2": (rng.standard_normal((88, 40), dtype=np.float32) * 0.14).astype(np.float16),
        "b2": (rng.standard_normal((40,), dtype=np.float32) * 0.020).astype(np.float16),
        "m1": (rng.standard_normal((40, 12), dtype=np.float32) * 0.13).astype(np.float16),
        "m2": (rng.standard_normal((12, 4), dtype=np.float32) * 0.11).astype(np.float16),
        "b3": np.asarray([0.00, 0.00, 0.15, -0.02], dtype=np.float16),
    }


def _native_shader_weight_matrix_gi_v3() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(181)
    return {
        "w1": (rng.standard_normal((80, 128), dtype=np.float32) * 0.15).astype(np.float16),
        "b1": (rng.standard_normal((128,), dtype=np.float32) * 0.020).astype(np.float16),
        "w2": (rng.standard_normal((128, 80), dtype=np.float32) * 0.12).astype(np.float16),
        "b2": (rng.standard_normal((80,), dtype=np.float32) * 0.017).astype(np.float16),
        "w3": (rng.standard_normal((80, 48), dtype=np.float32) * 0.10).astype(np.float16),
        "b3": (rng.standard_normal((48,), dtype=np.float32) * 0.015).astype(np.float16),
        "m1": (rng.standard_normal((48, 64), dtype=np.float32) * 0.10).astype(np.float16),
        "m2": (rng.standard_normal((64, 32), dtype=np.float32) * 0.090).astype(np.float16),
        "m3": (rng.standard_normal((32, 4), dtype=np.float32) * 0.080).astype(np.float16),
        "b4": np.asarray([0.02, -0.01, 0.22, 0.02], dtype=np.float16),
    }


def _compile_shader_field_block(
    *,
    engine: OpenVINOTensorCore,
    batch_size: int,
    device: str,
    performance_hint: str,
    turbo: bool,
    profile: str,
) -> _ShaderFieldBlock:
    if profile == "assist_v1":
        return _compile_openvino_shader_field_block(
            engine=engine,
            batch_size=batch_size,
            device=device,
            performance_hint=performance_hint,
            turbo=turbo,
        )
    if profile == "intel_npu_native_v1":
        return _compile_native_shader_field_block(
            engine=engine,
            batch_size=batch_size,
            device=device,
            performance_hint=performance_hint,
            turbo=turbo,
        )
    if profile == "intel_npu_native_v2":
        return _compile_native_shader_field_block_v2(
            engine=engine,
            batch_size=batch_size,
            device=device,
            performance_hint=performance_hint,
            turbo=turbo,
        )
    if profile == "intel_npu_native_v3":
        return _compile_native_shader_field_block_v3(
            engine=engine,
            batch_size=batch_size,
            device=device,
            performance_hint=performance_hint,
            turbo=turbo,
        )
    if profile == "intel_npu_rtgl_v1":
        return _compile_native_shader_field_block_rtgl_v1(
            engine=engine,
            batch_size=batch_size,
            device=device,
            performance_hint=performance_hint,
            turbo=turbo,
        )
    if profile == "intel_npu_gi_v1":
        return _compile_native_shader_field_block_gi_v1(
            engine=engine,
            batch_size=batch_size,
            device=device,
            performance_hint=performance_hint,
            turbo=turbo,
        )
    if profile == "intel_npu_gi_v2":
        return _compile_native_shader_field_block_gi_v2(
            engine=engine,
            batch_size=batch_size,
            device=device,
            performance_hint=performance_hint,
            turbo=turbo,
        )
    if profile == "intel_npu_gi_v3":
        return _compile_native_shader_field_block_gi_v3(
            engine=engine,
            batch_size=batch_size,
            device=device,
            performance_hint=performance_hint,
            turbo=turbo,
        )
    if profile == "intel_npu_shader2_v1":
        return _compile_native_shader_field_block_shader2_v1(
            engine=engine,
            batch_size=batch_size,
            device=device,
            performance_hint=performance_hint,
            turbo=turbo,
        )
    raise ValueError(f"Unknown shader profile: {profile}")


def _compile_openvino_shader_field_block(
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
        profile="assist_v1",
        backend="openvino_graph",
    )


def _compile_native_shader_field_block(
    *,
    engine: OpenVINOTensorCore,
    batch_size: int,
    device: str,
    performance_hint: str,
    turbo: bool,
) -> _ShaderFieldBlock:
    # The native Intel NPU path currently exposes fast bias-less Linear blocks.
    # To keep the shader field on that path, bias is folded into an extra constant
    # input channel instead of using a separate bias add in the model graph.
    weights = _native_shader_weight_matrix()
    layer1 = engine.compile_linear(
        _augment_bias_weight(weights["w1"], weights["b1"]),
        batch_size=batch_size,
        device=device,
        dtype=np.float16,
        pad_to=16,
        performance_hint=performance_hint,
        turbo=turbo,
    )
    layer2 = engine.compile_linear(
        _augment_bias_weight(weights["w2"], weights["b2"]),
        batch_size=batch_size,
        device=device,
        dtype=np.float16,
        pad_to=16,
        performance_hint=performance_hint,
        turbo=turbo,
    )
    layer3 = engine.compile_linear(
        _augment_bias_weight(weights["w3"], weights["b3"]),
        batch_size=batch_size,
        device=device,
        dtype=np.float16,
        pad_to=16,
        performance_hint=performance_hint,
        turbo=turbo,
    )

    def runner(shader_inputs: np.ndarray) -> np.ndarray:
        # The dense layers below are the part we intentionally want on the NPU.
        # Future contributors should move more low-frequency policy math into these
        # layers first before trying to offload frame-local GPU work like shadow PCF.
        hidden1 = layer1(_append_bias_feature(shader_inputs))
        np.maximum(hidden1, 0.0, out=hidden1)
        hidden2 = layer2(_append_bias_feature(hidden1))
        np.maximum(hidden2, 0.0, out=hidden2)
        output = layer3(_append_bias_feature(hidden2))
        output_f32 = np.asarray(output, dtype=np.float32)
        np.clip(output_f32, -8.0, 8.0, out=output_f32)
        output_f32 = 1.0 / (1.0 + np.exp(-output_f32))
        return output_f32.astype(np.float16)

    block_backends = {layer1.backend, layer2.backend, layer3.backend}
    if block_backends == {"intel_npu_acceleration_library"}:
        backend = "native_linear_pipeline"
    elif block_backends == {"openvino"}:
        backend = "openvino_linear_pipeline"
    else:
        backend = "hybrid_linear_pipeline"

    return _ShaderFieldBlock(
        runner=runner,
        device=device,
        input_features=16,
        output_features=4,
        max_batch=batch_size,
        dtype=np.dtype(np.float16),
        profile="intel_npu_native_v1",
        backend=backend,
    )


def _compile_native_shader_field_block_v2(
    *,
    engine: OpenVINOTensorCore,
    batch_size: int,
    device: str,
    performance_hint: str,
    turbo: bool,
) -> _ShaderFieldBlock:
    # v2 pushes more of the scene policy onto the NPU by widening the input basis
    # and adding one more native linear stage. It is intentionally heavier than v1.
    weights = _native_shader_weight_matrix_v2()
    layer1 = engine.compile_linear(
        _augment_bias_weight(weights["w1"], weights["b1"]),
        batch_size=batch_size,
        device=device,
        dtype=np.float16,
        pad_to=16,
        performance_hint=performance_hint,
        turbo=turbo,
    )
    layer2 = engine.compile_linear(
        _augment_bias_weight(weights["w2"], weights["b2"]),
        batch_size=batch_size,
        device=device,
        dtype=np.float16,
        pad_to=16,
        performance_hint=performance_hint,
        turbo=turbo,
    )
    layer3 = engine.compile_linear(
        _augment_bias_weight(weights["w3"], weights["b3"]),
        batch_size=batch_size,
        device=device,
        dtype=np.float16,
        pad_to=16,
        performance_hint=performance_hint,
        turbo=turbo,
    )
    layer4 = engine.compile_linear(
        _augment_bias_weight(weights["w4"], weights["b4"]),
        batch_size=batch_size,
        device=device,
        dtype=np.float16,
        pad_to=16,
        performance_hint=performance_hint,
        turbo=turbo,
    )

    def runner(shader_inputs: np.ndarray) -> np.ndarray:
        hidden1 = layer1(_append_bias_feature(shader_inputs))
        np.maximum(hidden1, 0.0, out=hidden1)
        hidden2 = layer2(_append_bias_feature(hidden1))
        np.maximum(hidden2, 0.0, out=hidden2)
        hidden3 = layer3(_append_bias_feature(hidden2))
        np.maximum(hidden3, 0.0, out=hidden3)
        output = layer4(_append_bias_feature(hidden3))
        output_f32 = np.asarray(output, dtype=np.float32)
        np.clip(output_f32, -8.0, 8.0, out=output_f32)
        output_f32 = 1.0 / (1.0 + np.exp(-output_f32))
        return output_f32.astype(np.float16)

    block_backends = {layer1.backend, layer2.backend, layer3.backend, layer4.backend}
    if block_backends == {"intel_npu_acceleration_library"}:
        backend = "native_linear_pipeline_v2"
    elif block_backends == {"openvino"}:
        backend = "openvino_linear_pipeline_v2"
    else:
        backend = "hybrid_linear_pipeline_v2"

    return _ShaderFieldBlock(
        runner=runner,
        device=device,
        input_features=24,
        output_features=4,
        max_batch=batch_size,
        dtype=np.dtype(np.float16),
        profile="intel_npu_native_v2",
        backend=backend,
    )


def _compile_native_shader_field_block_v3(
    *,
    engine: OpenVINOTensorCore,
    batch_size: int,
    device: str,
    performance_hint: str,
    turbo: bool,
) -> _ShaderFieldBlock:
    # v3 is intentionally heavier so the NPU owns more of the water / shadow / grade policy field.
    weights = _native_shader_weight_matrix_v3()
    layer1 = engine.compile_linear(
        _augment_bias_weight(weights["w1"], weights["b1"]),
        batch_size=batch_size,
        device=device,
        dtype=np.float16,
        pad_to=16,
        performance_hint=performance_hint,
        turbo=turbo,
    )
    layer2 = engine.compile_linear(
        _augment_bias_weight(weights["w2"], weights["b2"]),
        batch_size=batch_size,
        device=device,
        dtype=np.float16,
        pad_to=16,
        performance_hint=performance_hint,
        turbo=turbo,
    )
    layer3 = engine.compile_linear(
        _augment_bias_weight(weights["w3"], weights["b3"]),
        batch_size=batch_size,
        device=device,
        dtype=np.float16,
        pad_to=16,
        performance_hint=performance_hint,
        turbo=turbo,
    )
    layer4 = engine.compile_linear(
        _augment_bias_weight(weights["w4"], weights["b4"]),
        batch_size=batch_size,
        device=device,
        dtype=np.float16,
        pad_to=16,
        performance_hint=performance_hint,
        turbo=turbo,
    )
    layer5 = engine.compile_linear(
        _augment_bias_weight(weights["w5"], weights["b5"]),
        batch_size=batch_size,
        device=device,
        dtype=np.float16,
        pad_to=16,
        performance_hint=performance_hint,
        turbo=turbo,
    )

    def runner(shader_inputs: np.ndarray) -> np.ndarray:
        hidden1 = layer1(_append_bias_feature(shader_inputs))
        np.maximum(hidden1, 0.0, out=hidden1)
        hidden2 = layer2(_append_bias_feature(hidden1))
        np.maximum(hidden2, 0.0, out=hidden2)
        hidden3 = layer3(_append_bias_feature(hidden2))
        np.maximum(hidden3, 0.0, out=hidden3)
        hidden4 = layer4(_append_bias_feature(hidden3))
        np.maximum(hidden4, 0.0, out=hidden4)
        output = layer5(_append_bias_feature(hidden4))
        output_f32 = np.asarray(output, dtype=np.float32)
        np.clip(output_f32, -8.0, 8.0, out=output_f32)
        output_f32 = 1.0 / (1.0 + np.exp(-output_f32))
        return output_f32.astype(np.float16)

    block_backends = {layer1.backend, layer2.backend, layer3.backend, layer4.backend, layer5.backend}
    if block_backends == {"intel_npu_acceleration_library"}:
        backend = "native_linear_pipeline_v3"
    elif block_backends == {"openvino"}:
        backend = "openvino_linear_pipeline_v3"
    else:
        backend = "hybrid_linear_pipeline_v3"

    return _ShaderFieldBlock(
        runner=runner,
        device=device,
        input_features=32,
        output_features=4,
        max_batch=batch_size,
        dtype=np.dtype(np.float16),
        profile="intel_npu_native_v3",
        backend=backend,
    )


def _compile_native_shader_field_block_rtgl_v1(
    *,
    engine: OpenVINOTensorCore,
    batch_size: int,
    device: str,
    performance_hint: str,
    turbo: bool,
) -> _ShaderFieldBlock:
    # RT-GL v1 keeps the low-frequency scene policy on the NPU, then explicitly
    # decodes light-path and bloom coefficients through matmul stages. This keeps
    # the "trajectory field" generation inside NPU-friendly GEMM work instead of
    # trying to offload frame-local screen tracing itself.
    weights = _native_shader_weight_matrix_rtgl()
    layer1 = engine.compile_linear(
        _augment_bias_weight(weights["w1"], weights["b1"]),
        batch_size=batch_size,
        device=device,
        dtype=np.float16,
        pad_to=16,
        performance_hint=performance_hint,
        turbo=turbo,
    )
    layer2 = engine.compile_linear(
        _augment_bias_weight(weights["w2"], weights["b2"]),
        batch_size=batch_size,
        device=device,
        dtype=np.float16,
        pad_to=16,
        performance_hint=performance_hint,
        turbo=turbo,
    )
    layer3 = engine.compile_linear(
        _augment_bias_weight(weights["w3"], weights["b3"]),
        batch_size=batch_size,
        device=device,
        dtype=np.float16,
        pad_to=16,
        performance_hint=performance_hint,
        turbo=turbo,
    )

    matmul_backend = (
        "native_matmul"
        if engine._use_native_npu_backend(device, np.dtype(np.float16))
        else "openvino_matmul"
    )

    def runner(shader_inputs: np.ndarray) -> np.ndarray:
        hidden1 = layer1(_append_bias_feature(shader_inputs))
        np.maximum(hidden1, 0.0, out=hidden1)
        hidden2 = layer2(_append_bias_feature(hidden1))
        np.maximum(hidden2, 0.0, out=hidden2)
        hidden3 = layer3(_append_bias_feature(hidden2))
        hidden3_f32 = np.asarray(hidden3, dtype=np.float32)
        np.tanh(hidden3_f32, out=hidden3_f32)

        path_state = engine.matmul(
            hidden3_f32.astype(np.float16),
            weights["m1"],
            device=device,
            dtype=np.float16,
            pad_to=16,
            performance_hint=performance_hint,
            turbo=turbo,
        )
        path_state_f32 = np.asarray(path_state, dtype=np.float32)
        np.tanh(path_state_f32, out=path_state_f32)

        bloom_state = engine.matmul(
            path_state_f32.astype(np.float16),
            weights["m2"],
            device=device,
            dtype=np.float16,
            pad_to=16,
            performance_hint=performance_hint,
            turbo=turbo,
        )
        bloom_state_f32 = np.asarray(bloom_state, dtype=np.float32)
        np.tanh(bloom_state_f32, out=bloom_state_f32)

        output = engine.matmul(
            bloom_state_f32.astype(np.float16),
            weights["m3"],
            device=device,
            dtype=np.float16,
            pad_to=16,
            performance_hint=performance_hint,
            turbo=turbo,
        )
        output_f32 = np.asarray(output, dtype=np.float32)
        output_f32 += np.asarray(weights["b4"], dtype=np.float32)
        np.clip(output_f32, -8.0, 8.0, out=output_f32)
        output_f32 = 1.0 / (1.0 + np.exp(-output_f32))
        return output_f32.astype(np.float16)

    block_backends = {layer1.backend, layer2.backend, layer3.backend}
    if block_backends == {"intel_npu_acceleration_library"} and matmul_backend == "native_matmul":
        backend = "native_rtgl_matmul_pipeline_v1"
    elif block_backends == {"openvino"} and matmul_backend == "openvino_matmul":
        backend = "openvino_rtgl_matmul_pipeline_v1"
    else:
        backend = "hybrid_rtgl_matmul_pipeline_v1"

    return _ShaderFieldBlock(
        runner=runner,
        device=device,
        input_features=32,
        output_features=4,
        max_batch=batch_size,
        dtype=np.dtype(np.float16),
        profile="intel_npu_rtgl_v1",
        backend=backend,
    )


def _compile_native_shader_field_block_gi_v1(
    *,
    engine: OpenVINOTensorCore,
    batch_size: int,
    device: str,
    performance_hint: str,
    turbo: bool,
) -> _ShaderFieldBlock:
    # GI v1 drops the wide bloom policy and keeps only the low-resolution
    # indirect-light direction and energy field. That lets the GPU do fewer
    # post taps while the NPU focuses on a smaller matmul-heavy estimate.
    weights = _native_shader_weight_matrix_gi()
    layer1 = engine.compile_linear(
        _augment_bias_weight(weights["w1"], weights["b1"]),
        batch_size=batch_size,
        device=device,
        dtype=np.float16,
        pad_to=16,
        performance_hint=performance_hint,
        turbo=turbo,
    )
    layer2 = engine.compile_linear(
        _augment_bias_weight(weights["w2"], weights["b2"]),
        batch_size=batch_size,
        device=device,
        dtype=np.float16,
        pad_to=16,
        performance_hint=performance_hint,
        turbo=turbo,
    )

    matmul_backend = (
        "native_matmul"
        if engine._use_native_npu_backend(device, np.dtype(np.float16))
        else "openvino_matmul"
    )

    def runner(shader_inputs: np.ndarray) -> np.ndarray:
        hidden1 = layer1(_append_bias_feature(shader_inputs))
        np.maximum(hidden1, 0.0, out=hidden1)
        hidden2 = layer2(_append_bias_feature(hidden1))
        hidden2_f32 = np.asarray(hidden2, dtype=np.float32)
        np.tanh(hidden2_f32, out=hidden2_f32)

        gi_state = engine.matmul(
            hidden2_f32.astype(np.float16),
            weights["m1"],
            device=device,
            dtype=np.float16,
            pad_to=16,
            performance_hint=performance_hint,
            turbo=turbo,
        )
        gi_state_f32 = np.asarray(gi_state, dtype=np.float32)
        np.tanh(gi_state_f32, out=gi_state_f32)

        output = engine.matmul(
            gi_state_f32.astype(np.float16),
            weights["m2"],
            device=device,
            dtype=np.float16,
            pad_to=16,
            performance_hint=performance_hint,
            turbo=turbo,
        )
        output_f32 = np.asarray(output, dtype=np.float32)
        output_f32 += np.asarray(weights["b3"], dtype=np.float32)
        np.clip(output_f32, -8.0, 8.0, out=output_f32)
        output_f32 = 1.0 / (1.0 + np.exp(-output_f32))
        return output_f32.astype(np.float16)

    block_backends = {layer1.backend, layer2.backend}
    if block_backends == {"intel_npu_acceleration_library"} and matmul_backend == "native_matmul":
        backend = "native_gi_matmul_pipeline_v1"
    elif block_backends == {"openvino"} and matmul_backend == "openvino_matmul":
        backend = "openvino_gi_matmul_pipeline_v1"
    else:
        backend = "hybrid_gi_matmul_pipeline_v1"

    return _ShaderFieldBlock(
        runner=runner,
        device=device,
        input_features=32,
        output_features=4,
        max_batch=batch_size,
        dtype=np.dtype(np.float16),
        profile="intel_npu_gi_v1",
        backend=backend,
    )


def _compile_native_shader_field_block_gi_v2(
    *,
    engine: OpenVINOTensorCore,
    batch_size: int,
    device: str,
    performance_hint: str,
    turbo: bool,
) -> _ShaderFieldBlock:
    # GI v2 trims the basis width slightly so a 96x96 tile fits under a 60 FPS
    # style 16.6 ms budget more comfortably on the NPU.
    weights = _native_shader_weight_matrix_gi_v2()
    layer1 = engine.compile_linear(
        _augment_bias_weight(weights["w1"], weights["b1"]),
        batch_size=batch_size,
        device=device,
        dtype=np.float16,
        pad_to=16,
        performance_hint=performance_hint,
        turbo=turbo,
    )
    layer2 = engine.compile_linear(
        _augment_bias_weight(weights["w2"], weights["b2"]),
        batch_size=batch_size,
        device=device,
        dtype=np.float16,
        pad_to=16,
        performance_hint=performance_hint,
        turbo=turbo,
    )

    matmul_backend = (
        "native_matmul"
        if engine._use_native_npu_backend(device, np.dtype(np.float16))
        else "openvino_matmul"
    )

    def runner(shader_inputs: np.ndarray) -> np.ndarray:
        hidden1 = layer1(_append_bias_feature(shader_inputs))
        np.maximum(hidden1, 0.0, out=hidden1)
        hidden2 = layer2(_append_bias_feature(hidden1))
        hidden2_f32 = np.asarray(hidden2, dtype=np.float32)
        np.tanh(hidden2_f32, out=hidden2_f32)

        gi_state = engine.matmul(
            hidden2_f32.astype(np.float16),
            weights["m1"],
            device=device,
            dtype=np.float16,
            pad_to=16,
            performance_hint=performance_hint,
            turbo=turbo,
        )
        gi_state_f32 = np.asarray(gi_state, dtype=np.float32)
        np.tanh(gi_state_f32, out=gi_state_f32)

        output = engine.matmul(
            gi_state_f32.astype(np.float16),
            weights["m2"],
            device=device,
            dtype=np.float16,
            pad_to=16,
            performance_hint=performance_hint,
            turbo=turbo,
        )
        output_f32 = np.asarray(output, dtype=np.float32)
        output_f32 += np.asarray(weights["b3"], dtype=np.float32)
        np.clip(output_f32, -8.0, 8.0, out=output_f32)
        output_f32 = 1.0 / (1.0 + np.exp(-output_f32))
        return output_f32.astype(np.float16)

    block_backends = {layer1.backend, layer2.backend}
    if block_backends == {"intel_npu_acceleration_library"} and matmul_backend == "native_matmul":
        backend = "native_gi_matmul_pipeline_v2"
    elif block_backends == {"openvino"} and matmul_backend == "openvino_matmul":
        backend = "openvino_gi_matmul_pipeline_v2"
    else:
        backend = "hybrid_gi_matmul_pipeline_v2"

    return _ShaderFieldBlock(
        runner=runner,
        device=device,
        input_features=32,
        output_features=4,
        max_batch=batch_size,
        dtype=np.dtype(np.float16),
        profile="intel_npu_gi_v2",
        backend=backend,
    )


def _compile_native_shader_field_block_gi_v3(
    *,
    engine: OpenVINOTensorCore,
    batch_size: int,
    device: str,
    performance_hint: str,
    turbo: bool,
) -> _ShaderFieldBlock:
    # GI v3 is intentionally Xe-core-biased: a wider basis plus aligned matmul-heavy
    # decode keeps more of the low-frequency reflection / fog / shadow policy field
    # inside NPU-friendly GEMM work before the GPU applies frame-local depth and
    # texture sampling.
    weights = _native_shader_weight_matrix_gi_v3()
    layer1 = engine.compile_linear(
        _augment_bias_weight(weights["w1"], weights["b1"]),
        batch_size=batch_size,
        device=device,
        dtype=np.float16,
        pad_to=16,
        performance_hint=performance_hint,
        turbo=turbo,
    )
    layer2 = engine.compile_linear(
        _augment_bias_weight(weights["w2"], weights["b2"]),
        batch_size=batch_size,
        device=device,
        dtype=np.float16,
        pad_to=16,
        performance_hint=performance_hint,
        turbo=turbo,
    )
    layer3 = engine.compile_linear(
        _augment_bias_weight(weights["w3"], weights["b3"]),
        batch_size=batch_size,
        device=device,
        dtype=np.float16,
        pad_to=16,
        performance_hint=performance_hint,
        turbo=turbo,
    )

    matmul_backend = (
        "native_matmul"
        if engine._use_native_npu_backend(device, np.dtype(np.float16))
        else "openvino_matmul"
    )

    def runner(shader_inputs: np.ndarray) -> np.ndarray:
        hidden1 = layer1(_append_bias_feature(shader_inputs))
        np.maximum(hidden1, 0.0, out=hidden1)
        hidden2 = layer2(_append_bias_feature(hidden1))
        np.maximum(hidden2, 0.0, out=hidden2)
        hidden3 = layer3(_append_bias_feature(hidden2))
        hidden3_f32 = np.asarray(hidden3, dtype=np.float32)
        np.tanh(hidden3_f32, out=hidden3_f32)

        lane_state = engine.matmul(
            hidden3_f32.astype(np.float16),
            weights["m1"],
            device=device,
            dtype=np.float16,
            pad_to=16,
            performance_hint=performance_hint,
            turbo=turbo,
        )
        lane_state_f32 = np.asarray(lane_state, dtype=np.float32)
        np.tanh(lane_state_f32, out=lane_state_f32)

        policy_state = engine.matmul(
            lane_state_f32.astype(np.float16),
            weights["m2"],
            device=device,
            dtype=np.float16,
            pad_to=16,
            performance_hint=performance_hint,
            turbo=turbo,
        )
        policy_state_f32 = np.asarray(policy_state, dtype=np.float32)
        np.tanh(policy_state_f32, out=policy_state_f32)

        output = engine.matmul(
            policy_state_f32.astype(np.float16),
            weights["m3"],
            device=device,
            dtype=np.float16,
            pad_to=16,
            performance_hint=performance_hint,
            turbo=turbo,
        )
        output_f32 = np.asarray(output, dtype=np.float32)
        output_f32 += np.asarray(weights["b4"], dtype=np.float32)
        np.clip(output_f32, -8.0, 8.0, out=output_f32)
        output_f32 = 1.0 / (1.0 + np.exp(-output_f32))
        return output_f32.astype(np.float16)

    block_backends = {layer1.backend, layer2.backend, layer3.backend}
    if block_backends == {"intel_npu_acceleration_library"} and matmul_backend == "native_matmul":
        backend = "native_gi_xe_matmul_pipeline_v3"
    elif block_backends == {"openvino"} and matmul_backend == "openvino_matmul":
        backend = "openvino_gi_xe_matmul_pipeline_v3"
    else:
        backend = "hybrid_gi_xe_matmul_pipeline_v3"

    return _ShaderFieldBlock(
        runner=runner,
        device=device,
        input_features=80,
        output_features=4,
        max_batch=batch_size,
        dtype=np.dtype(np.float16),
        profile="intel_npu_gi_v3",
        backend=backend,
    )


def _compile_native_shader_field_block_shader2_v1(
    *,
    engine: OpenVINOTensorCore,
    batch_size: int,
    device: str,
    performance_hint: str,
    turbo: bool,
) -> _ShaderFieldBlock:
    # shader2 v1 reuses the GI v3 topology and weights, but surfaces a distinct
    # profile name so the runtime can preserve more NPU-authored policy under load.
    weights = _native_shader_weight_matrix_gi_v3()
    layer1 = engine.compile_linear(
        _augment_bias_weight(weights["w1"], weights["b1"]),
        batch_size=batch_size,
        device=device,
        dtype=np.float16,
        pad_to=16,
        performance_hint=performance_hint,
        turbo=turbo,
    )
    layer2 = engine.compile_linear(
        _augment_bias_weight(weights["w2"], weights["b2"]),
        batch_size=batch_size,
        device=device,
        dtype=np.float16,
        pad_to=16,
        performance_hint=performance_hint,
        turbo=turbo,
    )
    layer3 = engine.compile_linear(
        _augment_bias_weight(weights["w3"], weights["b3"]),
        batch_size=batch_size,
        device=device,
        dtype=np.float16,
        pad_to=16,
        performance_hint=performance_hint,
        turbo=turbo,
    )

    matmul_backend = (
        "native_matmul"
        if engine._use_native_npu_backend(device, np.dtype(np.float16))
        else "openvino_matmul"
    )

    def runner(shader_inputs: np.ndarray) -> np.ndarray:
        hidden1 = layer1(_append_bias_feature(shader_inputs))
        np.maximum(hidden1, 0.0, out=hidden1)
        hidden2 = layer2(_append_bias_feature(hidden1))
        np.maximum(hidden2, 0.0, out=hidden2)
        hidden3 = layer3(_append_bias_feature(hidden2))
        hidden3_f32 = np.asarray(hidden3, dtype=np.float32)
        np.tanh(hidden3_f32, out=hidden3_f32)

        lane_state = engine.matmul(
            hidden3_f32.astype(np.float16),
            weights["m1"],
            device=device,
            dtype=np.float16,
            pad_to=16,
            performance_hint=performance_hint,
            turbo=turbo,
        )
        lane_state_f32 = np.asarray(lane_state, dtype=np.float32)
        np.tanh(lane_state_f32, out=lane_state_f32)

        policy_state = engine.matmul(
            lane_state_f32.astype(np.float16),
            weights["m2"],
            device=device,
            dtype=np.float16,
            pad_to=16,
            performance_hint=performance_hint,
            turbo=turbo,
        )
        policy_state_f32 = np.asarray(policy_state, dtype=np.float32)
        np.tanh(policy_state_f32, out=policy_state_f32)

        output = engine.matmul(
            policy_state_f32.astype(np.float16),
            weights["m3"],
            device=device,
            dtype=np.float16,
            pad_to=16,
            performance_hint=performance_hint,
            turbo=turbo,
        )
        output_f32 = np.asarray(output, dtype=np.float32)
        output_f32 += np.asarray(weights["b4"], dtype=np.float32)
        np.clip(output_f32, -8.0, 8.0, out=output_f32)
        output_f32 = 1.0 / (1.0 + np.exp(-output_f32))
        return output_f32.astype(np.float16)

    block_backends = {layer1.backend, layer2.backend, layer3.backend}
    if block_backends == {"intel_npu_acceleration_library"} and matmul_backend == "native_matmul":
        backend = "native_shader2_matmul_pipeline_v1"
    elif block_backends == {"openvino"} and matmul_backend == "openvino_matmul":
        backend = "openvino_shader2_matmul_pipeline_v1"
    else:
        backend = "hybrid_shader2_matmul_pipeline_v1"

    return _ShaderFieldBlock(
        runner=runner,
        device=device,
        input_features=80,
        output_features=4,
        max_batch=batch_size,
        dtype=np.dtype(np.float16),
        profile="intel_npu_shader2_v1",
        backend=backend,
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
    sun_height: float = 0.5,
    rain_strength: float = 0.0,
    thunder_strength: float = 0.0,
    block_light: float = 0.0,
    sky_light: float = 1.0,
    submerged_factor: float = 0.0,
    quality_budget: float = 1.0,
    optimization_pressure: float = 0.0,
    input_features: int,
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
    time_sin = np.full_like(grid_x, np.sin(time_seconds * 0.37))
    time_cos = np.full_like(grid_x, np.cos(time_seconds * 0.23))
    yaw_sin = np.full_like(grid_x, np.sin(np.deg2rad(yaw_degrees)))
    yaw_cos = np.full_like(grid_x, np.cos(np.deg2rad(yaw_degrees)))
    pitch_sin = np.full_like(grid_x, np.sin(np.deg2rad(pitch_degrees)))
    horizon = np.clip(1.0 - np.abs(grid_y), 0.0, 1.0)
    wave_u = np.sin(grid_x * np.pi * 2.0 + time_seconds * 0.42)
    wave_v = np.cos(grid_y * np.pi * 2.0 - time_seconds * 0.28)
    diag = grid_x + grid_y
    anti_diag = grid_x - grid_y
    edge_mask = np.clip(np.maximum(np.abs(grid_x), np.abs(grid_y)), 0.0, 1.0)
    pos_y_time = pos_y_feature * time_feature
    yaw_v = yaw_feature * grid_y
    pitch_u = pitch_feature * grid_x
    radius_wave = np.sin(radius2 * np.pi * 1.5 + time_seconds * 0.31)
    horizon_wave = horizon * np.cos(time_seconds * 0.11 + grid_x * np.pi)
    sun_height_feature = np.full_like(grid_x, np.clip(sun_height * 2.0 - 1.0, -1.0, 1.0))
    rain_strength_feature = np.full_like(grid_x, np.clip(rain_strength * 2.0 - 1.0, -1.0, 1.0))
    thunder_strength_feature = np.full_like(grid_x, np.clip(thunder_strength * 2.0 - 1.0, -1.0, 1.0))
    block_light_feature = np.full_like(grid_x, np.clip(block_light * 2.0 - 1.0, -1.0, 1.0))
    sky_light_feature = np.full_like(grid_x, np.clip(sky_light * 2.0 - 1.0, -1.0, 1.0))
    submerged_feature = np.full_like(grid_x, np.clip(submerged_factor * 2.0 - 1.0, -1.0, 1.0))
    quality_budget_feature = np.full_like(grid_x, np.clip(quality_budget * 2.0 - 1.0, -1.0, 1.0))
    optimization_pressure_feature = np.full_like(
        grid_x, np.clip(optimization_pressure * 2.0 - 1.0, -1.0, 1.0)
    )
    sun_horizon = sun_height_feature * horizon
    rain_horizon = rain_strength_feature * horizon
    thunder_horizon = thunder_strength_feature * horizon
    sky_horizon = sky_light_feature * horizon
    block_radius = block_light_feature * radius2
    submerged_radius = submerged_feature * radius2
    sun_grid_y = sun_height_feature * grid_y
    rain_edge = rain_strength_feature * edge_mask
    thunder_diag = thunder_strength_feature * diag
    sky_anti_diag = sky_light_feature * anti_diag
    micro_lane_u = np.sin(grid_x * np.pi * 4.0)
    micro_lane_v = np.cos(grid_y * np.pi * 4.0)
    micro_lane_cross = micro_lane_u * micro_lane_v
    micro_lane_diag = np.sin((grid_x + grid_y) * np.pi * 2.0)
    sun_time_wave = sun_height_feature * time_feature
    sky_time_wave = sky_light_feature * time_feature
    storm_time_wave = thunder_strength_feature * time_feature
    rain_radius = rain_strength_feature * radius_wave
    thunder_radius = thunder_strength_feature * radius_wave
    block_horizon = block_light_feature * horizon
    sky_edge = sky_light_feature * edge_mask
    submerged_horizon = submerged_feature * horizon
    sun_lane = sun_height_feature * micro_lane_u
    storm_lane = thunder_strength_feature * micro_lane_cross
    light_cross = block_light_feature * uv
    submerged_time = submerged_feature * time_feature
    quality_horizon = quality_budget_feature * horizon
    pressure_horizon = optimization_pressure_feature * horizon
    quality_radius = quality_budget_feature * radius2
    pressure_edge = optimization_pressure_feature * edge_mask
    quality_time = quality_budget_feature * time_feature
    pressure_time = optimization_pressure_feature * time_feature
    quality_light = quality_budget_feature * sky_light_feature
    pressure_light = optimization_pressure_feature * block_light_feature
    quality_rain = quality_budget_feature * rain_strength_feature
    pressure_rain = optimization_pressure_feature * rain_strength_feature
    quality_submerged = quality_budget_feature * submerged_feature
    pressure_submerged = optimization_pressure_feature * submerged_feature
    quality_lane = quality_budget_feature * micro_lane_cross
    pressure_diag = optimization_pressure_feature * diag

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
            time_sin,
            time_cos,
            yaw_sin,
            yaw_cos,
            pitch_sin,
            horizon,
            wave_u,
            wave_v,
            diag,
            anti_diag,
            edge_mask,
            pos_y_time,
            yaw_v,
            pitch_u,
            radius_wave,
            horizon_wave,
            sun_height_feature,
            rain_strength_feature,
            thunder_strength_feature,
            block_light_feature,
            sky_light_feature,
            submerged_feature,
            sun_horizon,
            rain_horizon,
            thunder_horizon,
            sky_horizon,
            block_radius,
            submerged_radius,
            sun_grid_y,
            rain_edge,
            thunder_diag,
            sky_anti_diag,
            micro_lane_u,
            micro_lane_v,
            micro_lane_cross,
            micro_lane_diag,
            sun_time_wave,
            sky_time_wave,
            storm_time_wave,
            rain_radius,
            thunder_radius,
            block_horizon,
            sky_edge,
            submerged_horizon,
            sun_lane,
            storm_lane,
            light_cross,
            submerged_time,
            quality_budget_feature,
            optimization_pressure_feature,
            quality_horizon,
            pressure_horizon,
            quality_radius,
            pressure_edge,
            quality_time,
            pressure_time,
            quality_light,
            pressure_light,
            quality_rain,
            pressure_rain,
            quality_submerged,
            pressure_submerged,
            quality_lane,
            pressure_diag,
        ],
        axis=-1,
    )
    return features.reshape(width * height, features.shape[-1])[:, :input_features].astype(np.float16)


def _apply_realtime_budget_policy(
    output: np.ndarray,
    scene_hints: _ScenePolicyHints,
    *,
    profile: str = "intel_npu_gi_v3",
) -> np.ndarray:
    quality_budget = float(np.clip(scene_hints.quality_budget, 0.0, 1.0))
    optimization_pressure = float(np.clip(scene_hints.optimization_pressure, 0.0, 1.0))
    if quality_budget >= 0.999 and optimization_pressure <= 0.001:
        return np.asarray(output, dtype=np.float16)

    rgba = np.asarray(output, dtype=np.float32).copy()
    if profile == "intel_npu_shader2_v1":
        trajectory_scale = max(
            (0.84 + quality_budget * 0.16) * (1.0 - optimization_pressure * 0.05),
            0.82,
        )
        energy_scale = max(
            (0.80 + quality_budget * 0.20) * (1.0 - optimization_pressure * 0.08),
            0.74,
        )
        budget_scale = max(
            quality_budget * (1.0 - optimization_pressure * 0.14),
            quality_budget * 0.72,
        )
    else:
        trajectory_scale = (0.76 + quality_budget * 0.24) * (1.0 - optimization_pressure * 0.08)
        energy_scale = (0.72 + quality_budget * 0.28) * (1.0 - optimization_pressure * 0.12)
        budget_scale = quality_budget * (1.0 - optimization_pressure * 0.22)

    rgba[:, 0:2] = 0.5 + (rgba[:, 0:2] - 0.5) * trajectory_scale
    rgba[:, 2] *= energy_scale
    rgba[:, 3] *= budget_scale
    np.clip(rgba, 0.0, 1.0, out=rgba)
    return rgba.astype(np.float16)


def _augment_bias_weight(weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    weight_matrix = np.asarray(weight, dtype=np.float16)
    bias_vector = np.asarray(bias, dtype=np.float16)
    augmented = np.zeros((weight_matrix.shape[0] + 1, weight_matrix.shape[1]), dtype=np.float16)
    augmented[:-1, :] = weight_matrix
    augmented[-1, :] = bias_vector
    return augmented


def _append_bias_feature(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float16)
    ones = np.ones((values.shape[0], 1), dtype=np.float16)
    return np.concatenate([values, ones], axis=1)


def _pack_shader_pixels_array(output: np.ndarray) -> tuple[np.ndarray, float, float]:
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
    luma = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    return packed, float(alpha.mean()), float(luma.mean())


def _pack_shader_pixels(output: np.ndarray) -> tuple[list[int], float, float]:
    packed, mean_alpha, mean_luma = _pack_shader_pixels_array(output)
    signed = packed.view(np.int32).astype(np.int64, copy=False)
    return signed.tolist(), mean_alpha, mean_luma


def benchmark_shader_profile(
    *,
    profile: str,
    width: int,
    height: int,
    devices: tuple[str, ...] = ("NPU", "GPU", "CPU"),
    iterations: int = 20,
    warmup: int = 3,
    performance_hint: str = "THROUGHPUT",
    turbo: bool = True,
    pos_x: float = 12.0,
    pos_y: float = 80.0,
    pos_z: float = -6.0,
    yaw_degrees: float = 45.0,
    pitch_degrees: float = 10.0,
    time_seconds: float = 1.25,
    sun_height: float = 0.5,
    rain_strength: float = 0.0,
    thunder_strength: float = 0.0,
    block_light: float = 0.0,
    sky_light: float = 1.0,
    submerged_factor: float = 0.0,
    quality_budget: float = 1.0,
    optimization_pressure: float = 0.0,
) -> list[ShaderProfileBenchmarkResult]:
    engine = OpenVINOTensorCore()
    batch_size = width * height
    results: list[ShaderProfileBenchmarkResult] = []
    scene_hints = _ScenePolicyHints(
        sun_height=sun_height,
        rain_strength=rain_strength,
        thunder_strength=thunder_strength,
        block_light=block_light,
        sky_light=sky_light,
        submerged_factor=submerged_factor,
        quality_budget=quality_budget,
        optimization_pressure=optimization_pressure,
    )

    for device_name in devices:
        resolved_device = engine.resolve_device(device_name)
        block = _compile_shader_field_block(
            engine=engine,
            batch_size=batch_size,
            device=resolved_device,
            performance_hint=performance_hint,
            turbo=turbo,
            profile=profile,
        )
        shader_inputs = _build_shader_inputs(
            width=width,
            height=height,
            pos_x=pos_x,
            pos_y=pos_y,
            pos_z=pos_z,
            yaw_degrees=yaw_degrees,
            pitch_degrees=pitch_degrees,
            time_seconds=time_seconds,
            sun_height=sun_height,
            rain_strength=rain_strength,
            thunder_strength=thunder_strength,
            block_light=block_light,
            sky_light=sky_light,
            submerged_factor=submerged_factor,
            quality_budget=quality_budget,
            optimization_pressure=optimization_pressure,
            input_features=block.input_features,
        )

        for _ in range(max(warmup, 0)):
            _render_shader_block(block, shader_inputs, scene_hints)

        timings: list[float] = []
        for _ in range(max(iterations, 1)):
            started_at = perf_counter()
            _render_shader_block(block, shader_inputs, scene_hints)
            timings.append((perf_counter() - started_at) * 1000.0)

        average_ms = float(np.mean(timings))
        updates_per_second = 1000.0 / average_ms if average_ms > 0.0 else float("inf")
        results.append(
            ShaderProfileBenchmarkResult(
                device=resolved_device,
                profile=block.profile,
                backend=block.backend,
                width=width,
                height=height,
                average_ms=average_ms,
                min_ms=float(np.min(timings)),
                max_ms=float(np.max(timings)),
                p95_ms=float(np.percentile(np.asarray(timings, dtype=np.float64), 95.0)),
                iterations=max(iterations, 1),
                updates_per_second=updates_per_second,
                pixels_per_second=updates_per_second * batch_size,
            )
        )

    return results


def _render_shader_block(
    block: _ShaderFieldBlock,
    shader_inputs: np.ndarray,
    scene_hints: _ScenePolicyHints = _ScenePolicyHints(),
) -> tuple[np.ndarray, float, float]:
    output = _apply_realtime_budget_policy(block(shader_inputs), scene_hints, profile=block.profile)
    return _pack_shader_pixels_array(output)


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


def _clamp_unit(value: Any, default: float) -> float:
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except (TypeError, ValueError):
        return default


def _scene_policy_hints_from_payload(payload: dict[str, Any]) -> _ScenePolicyHints:
    return _ScenePolicyHints(
        sun_height=_clamp_unit(payload.get("sun_height", 0.5), 0.5),
        rain_strength=_clamp_unit(payload.get("rain_strength", 0.0), 0.0),
        thunder_strength=_clamp_unit(payload.get("thunder_strength", 0.0), 0.0),
        block_light=_clamp_unit(payload.get("block_light", 0.0), 0.0),
        sky_light=_clamp_unit(payload.get("sky_light", 1.0), 1.0),
        submerged_factor=_clamp_unit(payload.get("submerged_factor", 0.0), 0.0),
        quality_budget=_clamp_unit(payload.get("quality_budget", 1.0), 1.0),
        optimization_pressure=_clamp_unit(payload.get("optimization_pressure", 0.0), 0.0),
    )


def _scene_policy_hints_payload(hints: _ScenePolicyHints) -> dict[str, float]:
    return {
        "sun_height": hints.sun_height,
        "rain_strength": hints.rain_strength,
        "thunder_strength": hints.thunder_strength,
        "block_light": hints.block_light,
        "sky_light": hints.sky_light,
        "submerged_factor": hints.submerged_factor,
        "quality_budget": hints.quality_budget,
        "optimization_pressure": hints.optimization_pressure,
    }


_BINARY_MAGIC = b"NPXB"
_BINARY_VERSION = 1
_BINARY_KIND_SHADER_RUN_REQUEST_V1 = 1
_BINARY_KIND_SHADER_RUN_RESPONSE = 2
_BINARY_KIND_SHADER_RUN_REQUEST_V2 = 3
_BINARY_KIND_SHADER_RUN_REQUEST_V3 = 4
_BINARY_KIND_ERROR = 255
_BINARY_HEADER = struct.Struct("<4sBBHI")
_BINARY_SHADER_RUN_REQUEST_V1 = struct.Struct("<16s6f")
_BINARY_SHADER_RUN_REQUEST_V2 = struct.Struct("<16s12f")
_BINARY_SHADER_RUN_REQUEST_V3 = struct.Struct("<16s14f")
_BINARY_SHADER_RUN_RESPONSE = struct.Struct("<IIffHHI")
_BINARY_ERROR = struct.Struct("<H")


def _read_exact(stream: Any, size: int) -> bytes:
    data = stream.read(size)
    if data is None or len(data) != size:
        raise EOFError(f"Expected {size} bytes, got {0 if data is None else len(data)}")
    return data


def _encode_binary_frame(kind: int, payload: bytes) -> bytes:
    header = _BINARY_HEADER.pack(
        _BINARY_MAGIC,
        _BINARY_VERSION,
        kind,
        0,
        len(payload),
    )
    return header + payload


def _encode_binary_error(message: str) -> bytes:
    encoded = message.encode("utf-8")
    if len(encoded) > 0xFFFF:
        encoded = encoded[:0xFFFF]
    return _encode_binary_frame(_BINARY_KIND_ERROR, _BINARY_ERROR.pack(len(encoded)) + encoded)


def _decode_binary_shader_run_request_v1(payload: bytes) -> dict[str, Any]:
    if len(payload) != _BINARY_SHADER_RUN_REQUEST_V1.size:
        raise ValueError(
            f"Invalid shader_run v1 binary payload size: expected {_BINARY_SHADER_RUN_REQUEST_V1.size}, got {len(payload)}"
        )
    session_bytes, pos_x, pos_y, pos_z, yaw_degrees, pitch_degrees, time_seconds = _BINARY_SHADER_RUN_REQUEST_V1.unpack(
        payload
    )
    request = {
        "session_id": session_bytes.hex(),
        "pos_x": pos_x,
        "pos_y": pos_y,
        "pos_z": pos_z,
        "yaw_degrees": yaw_degrees,
        "pitch_degrees": pitch_degrees,
        "time_seconds": time_seconds,
    }
    request.update(_scene_policy_hints_payload(_ScenePolicyHints()))
    return request


def _decode_binary_shader_run_request_v2(payload: bytes) -> dict[str, Any]:
    if len(payload) != _BINARY_SHADER_RUN_REQUEST_V2.size:
        raise ValueError(
            f"Invalid shader_run v2 binary payload size: expected {_BINARY_SHADER_RUN_REQUEST_V2.size}, got {len(payload)}"
        )
    unpacked = _BINARY_SHADER_RUN_REQUEST_V2.unpack(payload)
    (
        session_bytes,
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
    ) = unpacked
    request = {
        "session_id": session_bytes.hex(),
        "pos_x": pos_x,
        "pos_y": pos_y,
        "pos_z": pos_z,
        "yaw_degrees": yaw_degrees,
        "pitch_degrees": pitch_degrees,
        "time_seconds": time_seconds,
        "sun_height": sun_height,
        "rain_strength": rain_strength,
        "thunder_strength": thunder_strength,
        "block_light": block_light,
        "sky_light": sky_light,
        "submerged_factor": submerged_factor,
    }
    request.update(
        _scene_policy_hints_payload(
            _ScenePolicyHints(
                sun_height=sun_height,
                rain_strength=rain_strength,
                thunder_strength=thunder_strength,
                block_light=block_light,
                sky_light=sky_light,
                submerged_factor=submerged_factor,
            )
        )
    )
    return request


def _decode_binary_shader_run_request_v3(payload: bytes) -> dict[str, Any]:
    if len(payload) != _BINARY_SHADER_RUN_REQUEST_V3.size:
        raise ValueError(
            f"Invalid shader_run v3 binary payload size: expected {_BINARY_SHADER_RUN_REQUEST_V3.size}, got {len(payload)}"
        )
    unpacked = _BINARY_SHADER_RUN_REQUEST_V3.unpack(payload)
    (
        session_bytes,
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
    ) = unpacked
    return {
        "session_id": session_bytes.hex(),
        "pos_x": pos_x,
        "pos_y": pos_y,
        "pos_z": pos_z,
        "yaw_degrees": yaw_degrees,
        "pitch_degrees": pitch_degrees,
        "time_seconds": time_seconds,
        "sun_height": sun_height,
        "rain_strength": rain_strength,
        "thunder_strength": thunder_strength,
        "block_light": block_light,
        "sky_light": sky_light,
        "submerged_factor": submerged_factor,
        "quality_budget": quality_budget,
        "optimization_pressure": optimization_pressure,
    }


def _encode_binary_shader_run_response(frame: _RenderedShaderFrame) -> bytes:
    profile_bytes = frame.profile.encode("utf-8")
    backend_bytes = frame.backend.encode("utf-8")
    if len(profile_bytes) > 0xFFFF or len(backend_bytes) > 0xFFFF:
        raise ValueError("profile/backend metadata is too large for the binary bridge")

    pixels = np.asarray(frame.pixels_abgr, dtype="<u4")
    prefix = _BINARY_SHADER_RUN_RESPONSE.pack(
        frame.width,
        frame.height,
        float(frame.mean_alpha),
        float(frame.mean_luma),
        len(profile_bytes),
        len(backend_bytes),
        pixels.size,
    )
    payload = prefix + profile_bytes + backend_bytes + pixels.tobytes(order="C")
    return _encode_binary_frame(_BINARY_KIND_SHADER_RUN_RESPONSE, payload)


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
            first = self.rfile.read(1)
            if not first:
                return

            if first == b"{":
                self._handle_json_frame(first)
                continue

            if first == _BINARY_MAGIC[:1]:
                try:
                    self._handle_binary_frame(first)
                except EOFError:
                    return
                continue

            self.wfile.write(_encode_binary_error(f"Unknown socket frame prefix: 0x{first.hex()}"))
            self.wfile.flush()
            return

    def _handle_json_frame(self, first: bytes) -> None:
        raw = first + self.rfile.readline()
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

    def _handle_binary_frame(self, first: bytes) -> None:
        header = first + _read_exact(self.rfile, _BINARY_HEADER.size - 1)
        magic, version, kind, _reserved, payload_size = _BINARY_HEADER.unpack(header)
        if magic != _BINARY_MAGIC:
            self.wfile.write(_encode_binary_error("Invalid binary bridge magic"))
            self.wfile.flush()
            return
        if version != _BINARY_VERSION:
            self.wfile.write(_encode_binary_error(f"Unsupported binary bridge version: {version}"))
            self.wfile.flush()
            return

        payload = _read_exact(self.rfile, payload_size)
        try:
            if kind == _BINARY_KIND_SHADER_RUN_REQUEST_V1:
                frame = self.server.state.render_shader_frame(_decode_binary_shader_run_request_v1(payload))
                response = _encode_binary_shader_run_response(frame)
            elif kind == _BINARY_KIND_SHADER_RUN_REQUEST_V2:
                frame = self.server.state.render_shader_frame(_decode_binary_shader_run_request_v2(payload))
                response = _encode_binary_shader_run_response(frame)
            elif kind == _BINARY_KIND_SHADER_RUN_REQUEST_V3:
                frame = self.server.state.render_shader_frame(_decode_binary_shader_run_request_v3(payload))
                response = _encode_binary_shader_run_response(frame)
            else:
                response = _encode_binary_error(f"Unsupported binary bridge kind: {kind}")
        except KeyError as exc:
            response = _encode_binary_error(str(exc))
        except ValueError as exc:
            response = _encode_binary_error(str(exc))
        except Exception as exc:
            response = _encode_binary_error(f"{type(exc).__name__}: {exc}")

        self.wfile.write(response)
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
