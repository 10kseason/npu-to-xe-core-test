from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Sequence
import uuid

import numpy as np

DEFAULT_DEVICE_ORDER = ("NPU", "GPU", "CPU")


def _probe_native_npu_backend() -> tuple[Any, Any] | None:
    try:
        from intel_npu_acceleration_library.backend import Linear, MatMul

        return (MatMul, Linear)
    except Exception:
        return None


_NATIVE_NPU_BACKEND: tuple[Any, Any] | None = _probe_native_npu_backend()

import openvino as ov
from openvino.runtime import opset13


@dataclass(frozen=True)
class BenchmarkResult:
    device: str
    average_ms: float
    min_ms: float
    max_ms: float
    iterations: int
    output_shape: tuple[int, int]
    dtype: str
    pad_to: int
    performance_hint: str
    turbo: bool
    backend: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _MatmulKernel:
    compiled_model: ov.CompiledModel
    request: ov.InferRequest
    output: Any
    input_names: tuple[str, str]
    padded_shape: tuple[int, int, int]


@dataclass
class _NativeMatmulKernel:
    op: Any
    padded_shape: tuple[int, int, int]


def _load_native_npu_backend() -> tuple[Any, Any] | None:
    return _NATIVE_NPU_BACKEND


def _ceil_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _normalize_hint(performance_hint: str) -> str:
    normalized = performance_hint.upper()
    if normalized not in {"LATENCY", "THROUGHPUT"}:
        raise ValueError("performance_hint must be LATENCY or THROUGHPUT")
    return normalized


def _ensure_2d(array: np.ndarray, name: str) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 matrix, got shape {array.shape}")
    return array


def _coerce_dtype(dtype: np.dtype | type[np.generic] | str | None) -> np.dtype:
    if dtype is None:
        return np.dtype(np.float16)
    return np.dtype(dtype)


def _pad_matrix(matrix: np.ndarray, padded_shape: tuple[int, int]) -> np.ndarray:
    rows, cols = padded_shape
    if matrix.shape == padded_shape:
        return np.ascontiguousarray(matrix)
    padded = np.zeros(padded_shape, dtype=matrix.dtype)
    padded[: matrix.shape[0], : matrix.shape[1]] = matrix
    return padded


class LinearBlock:
    """Compiled, fixed-weight linear block for repeated NPU/GPU/CPU execution."""

    def __init__(
        self,
        *,
        runner: Callable[[np.ndarray], np.ndarray],
        device: str,
        input_features: int,
        output_features: int,
        max_batch: int,
        padded_input_features: int,
        padded_batch: int,
        dtype: np.dtype,
        pad_to: int,
        performance_hint: str,
        turbo: bool,
        backend: str,
    ) -> None:
        self._runner = runner
        self.device = device
        self.input_features = input_features
        self.output_features = output_features
        self.max_batch = max_batch
        self.padded_input_features = padded_input_features
        self.padded_batch = padded_batch
        self.dtype = dtype
        self.pad_to = pad_to
        self.performance_hint = performance_hint
        self.turbo = turbo
        self.backend = backend

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        matrix = np.asarray(inputs, dtype=self.dtype)
        matrix = _ensure_2d(matrix, "inputs")
        if matrix.shape[1] != self.input_features:
            raise ValueError(
                f"Expected {self.input_features} input features, got {matrix.shape[1]}"
            )
        if matrix.shape[0] > self.max_batch:
            raise ValueError(f"Batch {matrix.shape[0]} exceeds compiled max_batch={self.max_batch}")

        padded = _pad_matrix(matrix, (self.padded_batch, self.padded_input_features))
        output = np.asarray(self._runner(padded), dtype=self.dtype)
        return output[: matrix.shape[0], : self.output_features].copy()

    def benchmark(self, inputs: np.ndarray, *, iterations: int = 30, warmup: int = 5) -> BenchmarkResult:
        matrix = np.asarray(inputs, dtype=self.dtype)
        for _ in range(max(warmup, 0)):
            self(matrix)

        timings: list[float] = []
        last_output = self(matrix)
        for _ in range(max(iterations, 1)):
            start = perf_counter()
            last_output = self(matrix)
            timings.append((perf_counter() - start) * 1000.0)

        return BenchmarkResult(
            device=self.device,
            average_ms=float(np.mean(timings)),
            min_ms=float(np.min(timings)),
            max_ms=float(np.max(timings)),
            iterations=max(iterations, 1),
            output_shape=tuple(last_output.shape),
            dtype=str(self.dtype),
            pad_to=self.pad_to,
            performance_hint=self.performance_hint,
            turbo=self.turbo,
            backend=self.backend,
        )


class OpenVINOTensorCore:
    """
    Tensor-core-style FP16 matmul helpers on top of OpenVINO.

    This does not expose raw Xe GPU XMX instructions. Instead, it packages
    static-shape MatMul and Linear blocks so the OpenVINO NPU backend can map
    them onto the NPU execution stack while preserving a simple GEMM-oriented API.
    """

    def __init__(
        self,
        *,
        preferred_devices: Sequence[str] = DEFAULT_DEVICE_ORDER,
        cache_dir: str | Path | None = ".ov_cache",
        prefer_native_npu_backend: bool = True,
    ) -> None:
        self.core = ov.Core()
        self.preferred_devices = tuple(device.upper() for device in preferred_devices)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.prefer_native_npu_backend = prefer_native_npu_backend
        self._matmul_cache: dict[tuple[Any, ...], _MatmulKernel] = {}
        self._native_matmul_cache: dict[tuple[Any, ...], _NativeMatmulKernel] = {}

    def available_devices(self) -> tuple[str, ...]:
        return tuple(self.core.available_devices)

    def native_npu_backend_available(self) -> bool:
        return _load_native_npu_backend() is not None

    def device_info(self, device: str) -> dict[str, Any]:
        resolved = device.upper()
        return {
            "device": resolved,
            "name": self.core.get_property(resolved, "FULL_DEVICE_NAME"),
            "capabilities": tuple(self.core.get_property(resolved, "OPTIMIZATION_CAPABILITIES")),
        }

    def resolve_device(self, device: str = "AUTO") -> str:
        requested = device.upper()
        available = set(self.available_devices())

        if requested != "AUTO":
            if requested not in available:
                raise ValueError(f"Requested device {requested} is not available: {sorted(available)}")
            return requested

        for candidate in self.preferred_devices:
            if candidate in available:
                return candidate
        raise RuntimeError(f"No preferred devices available. Found: {sorted(available)}")

    def _use_native_npu_backend(self, device: str, dtype: np.dtype) -> bool:
        return (
            self.prefer_native_npu_backend
            and device == "NPU"
            and dtype == np.dtype(np.float16)
            and self.native_npu_backend_available()
        )

    def matmul(
        self,
        lhs: np.ndarray,
        rhs: np.ndarray,
        *,
        device: str = "AUTO",
        dtype: np.dtype | type[np.generic] | str | None = None,
        pad_to: int = 16,
        performance_hint: str = "THROUGHPUT",
        turbo: bool = True,
    ) -> np.ndarray:
        resolved_device = self.resolve_device(device)
        coerced_dtype = _coerce_dtype(dtype)
        hint = _normalize_hint(performance_hint)

        lhs_array = _ensure_2d(np.asarray(lhs, dtype=coerced_dtype), "lhs")
        rhs_array = _ensure_2d(np.asarray(rhs, dtype=coerced_dtype), "rhs")
        if lhs_array.shape[1] != rhs_array.shape[0]:
            raise ValueError(
                f"Incompatible matrix shapes {lhs_array.shape} and {rhs_array.shape} for matmul"
            )

        rows, inner = lhs_array.shape
        _, cols = rhs_array.shape
        padded_rows = _ceil_multiple(rows, pad_to)
        padded_inner = _ceil_multiple(inner, pad_to)
        padded_cols = _ceil_multiple(cols, pad_to)

        lhs_padded = _pad_matrix(lhs_array, (padded_rows, padded_inner))
        rhs_padded = _pad_matrix(rhs_array, (padded_inner, padded_cols))

        if self._use_native_npu_backend(resolved_device, coerced_dtype):
            kernel = self._get_or_create_native_matmul_kernel(
                lhs_padded.shape[0],
                lhs_padded.shape[1],
                rhs_padded.shape[1],
            )
            output = np.asarray(kernel.op.run(lhs_padded, np.ascontiguousarray(rhs_padded.T)), dtype=coerced_dtype)
            return output[:rows, :cols].copy()

        kernel = self._get_or_create_matmul_kernel(
            resolved_device,
            lhs_padded.shape[0],
            lhs_padded.shape[1],
            rhs_padded.shape[1],
            coerced_dtype,
            hint,
            turbo,
        )
        result = kernel.request.infer(
            {
                kernel.input_names[0]: lhs_padded,
                kernel.input_names[1]: rhs_padded,
            }
        )
        output = np.array(result[kernel.output], copy=False)
        return output[:rows, :cols].copy()

    def benchmark_matmul(
        self,
        *,
        m: int,
        k: int,
        n: int,
        devices: Iterable[str],
        dtype: np.dtype | type[np.generic] | str | None = None,
        pad_to: int = 16,
        performance_hint: str = "THROUGHPUT",
        turbo: bool = True,
        warmup: int = 5,
        iterations: int = 30,
        seed: int = 7,
    ) -> list[BenchmarkResult]:
        coerced_dtype = _coerce_dtype(dtype)
        hint = _normalize_hint(performance_hint)
        rng = np.random.default_rng(seed)
        lhs = rng.standard_normal((m, k), dtype=np.float32).astype(coerced_dtype)
        rhs = rng.standard_normal((k, n), dtype=np.float32).astype(coerced_dtype)

        results: list[BenchmarkResult] = []
        for device in devices:
            resolved = self.resolve_device(device)
            backend = (
                "intel_npu_acceleration_library"
                if self._use_native_npu_backend(resolved, coerced_dtype)
                else "openvino"
            )
            for _ in range(max(warmup, 0)):
                self.matmul(
                    lhs,
                    rhs,
                    device=resolved,
                    dtype=coerced_dtype,
                    pad_to=pad_to,
                    performance_hint=hint,
                    turbo=turbo,
                )

            timings: list[float] = []
            last_output = self.matmul(
                lhs,
                rhs,
                device=resolved,
                dtype=coerced_dtype,
                pad_to=pad_to,
                performance_hint=hint,
                turbo=turbo,
            )
            for _ in range(max(iterations, 1)):
                start = perf_counter()
                last_output = self.matmul(
                    lhs,
                    rhs,
                    device=resolved,
                    dtype=coerced_dtype,
                    pad_to=pad_to,
                    performance_hint=hint,
                    turbo=turbo,
                )
                timings.append((perf_counter() - start) * 1000.0)

            results.append(
                BenchmarkResult(
                    device=resolved,
                    average_ms=float(np.mean(timings)),
                    min_ms=float(np.min(timings)),
                    max_ms=float(np.max(timings)),
                    iterations=max(iterations, 1),
                    output_shape=tuple(last_output.shape),
                    dtype=str(coerced_dtype),
                    pad_to=pad_to,
                    performance_hint=hint if backend == "openvino" else "NATIVE_NPU",
                    turbo=bool(turbo if resolved == "NPU" and backend == "openvino" else False),
                    backend=backend,
                )
            )
        return results

    def compile_linear(
        self,
        weight: np.ndarray,
        *,
        bias: np.ndarray | None = None,
        batch_size: int = 1,
        device: str = "AUTO",
        dtype: np.dtype | type[np.generic] | str | None = None,
        pad_to: int = 16,
        performance_hint: str = "THROUGHPUT",
        turbo: bool = True,
    ) -> LinearBlock:
        resolved_device = self.resolve_device(device)
        coerced_dtype = _coerce_dtype(dtype)
        hint = _normalize_hint(performance_hint)

        weight_matrix = _ensure_2d(np.asarray(weight, dtype=coerced_dtype), "weight")
        in_features, out_features = weight_matrix.shape
        padded_batch = _ceil_multiple(batch_size, pad_to)
        padded_in = _ceil_multiple(in_features, pad_to)
        padded_out = _ceil_multiple(out_features, pad_to)

        padded_weight = _pad_matrix(weight_matrix, (padded_in, padded_out))
        if bias is not None:
            bias_array = np.asarray(bias, dtype=coerced_dtype)
            if bias_array.ndim != 1 or bias_array.shape[0] != out_features:
                raise ValueError(f"bias must have shape ({out_features},), got {bias_array.shape}")
            padded_bias = np.zeros((padded_out,), dtype=coerced_dtype)
            padded_bias[:out_features] = bias_array
        else:
            padded_bias = None

        if self._use_native_npu_backend(resolved_device, coerced_dtype) and padded_bias is None:
            _, NativeLinear = _load_native_npu_backend() or (None, None)
            if NativeLinear is None:
                raise RuntimeError("Native NPU backend unexpectedly unavailable")

            op = NativeLinear(padded_in, padded_out, padded_batch, device=resolved_device)
            weight_for_backend = np.ascontiguousarray(padded_weight.T)
            op_id = f"linear-{uuid.uuid4().hex}"

            def native_runner(padded_inputs: np.ndarray) -> np.ndarray:
                return np.asarray(op.run(padded_inputs, weight_for_backend, op_id), dtype=coerced_dtype)

            return LinearBlock(
                runner=native_runner,
                device=resolved_device,
                input_features=in_features,
                output_features=out_features,
                max_batch=batch_size,
                padded_input_features=padded_in,
                padded_batch=padded_batch,
                dtype=coerced_dtype,
                pad_to=pad_to,
                performance_hint="NATIVE_NPU",
                turbo=False,
                backend="intel_npu_acceleration_library",
            )

        input_node = opset13.parameter([padded_batch, padded_in], coerced_dtype, name="input")
        weight_node = opset13.constant(padded_weight)
        output = opset13.matmul(input_node, weight_node, False, False)
        if padded_bias is not None:
            output = opset13.add(output, opset13.constant(padded_bias))
        model = ov.Model([output], [input_node], "linear_block")
        compiled = self.core.compile_model(
            model,
            resolved_device,
            self._compile_config(resolved_device, hint, turbo),
        )
        request = compiled.create_infer_request()
        input_name = compiled.input(0).get_any_name()
        output_node = compiled.output(0)

        def openvino_runner(padded_inputs: np.ndarray) -> np.ndarray:
            result = request.infer({input_name: padded_inputs})
            return np.array(result[output_node], copy=False)

        return LinearBlock(
            runner=openvino_runner,
            device=resolved_device,
            input_features=in_features,
            output_features=out_features,
            max_batch=batch_size,
            padded_input_features=padded_in,
            padded_batch=padded_batch,
            dtype=coerced_dtype,
            pad_to=pad_to,
            performance_hint=hint,
            turbo=bool(turbo if resolved_device == "NPU" else False),
            backend="openvino",
        )

    def _compile_config(self, device: str, performance_hint: str, turbo: bool) -> dict[str, Any]:
        config: dict[str, Any] = {"PERFORMANCE_HINT": performance_hint}
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            config["CACHE_DIR"] = str(self.cache_dir)
        if device == "NPU":
            config["NPU_TURBO"] = bool(turbo)
        return config

    def _get_or_create_matmul_kernel(
        self,
        device: str,
        rows: int,
        inner: int,
        cols: int,
        dtype: np.dtype,
        performance_hint: str,
        turbo: bool,
    ) -> _MatmulKernel:
        key = (device, rows, inner, cols, dtype.str, performance_hint, bool(turbo))
        cached = self._matmul_cache.get(key)
        if cached is not None:
            return cached

        lhs = opset13.parameter([rows, inner], dtype, name="lhs")
        rhs = opset13.parameter([inner, cols], dtype, name="rhs")
        output = opset13.matmul(lhs, rhs, False, False)
        model = ov.Model([output], [lhs, rhs], "matmul")
        compiled = self.core.compile_model(
            model,
            device,
            self._compile_config(device, performance_hint, turbo),
        )
        kernel = _MatmulKernel(
            compiled_model=compiled,
            request=compiled.create_infer_request(),
            output=compiled.output(0),
            input_names=(compiled.input(0).get_any_name(), compiled.input(1).get_any_name()),
            padded_shape=(rows, inner, cols),
        )
        self._matmul_cache[key] = kernel
        return kernel

    def _get_or_create_native_matmul_kernel(
        self,
        rows: int,
        inner: int,
        cols: int,
    ) -> _NativeMatmulKernel:
        key = ("NPU", rows, inner, cols)
        cached = self._native_matmul_cache.get(key)
        if cached is not None:
            return cached

        NativeMatMul, _ = _load_native_npu_backend() or (None, None)
        if NativeMatMul is None:
            raise RuntimeError("Native NPU backend unexpectedly unavailable")

        kernel = _NativeMatmulKernel(
            op=NativeMatMul(inner, cols, rows, device="NPU"),
            padded_shape=(rows, inner, cols),
        )
        self._native_matmul_cache[key] = kernel
        return kernel
