from __future__ import annotations

import argparse
import json
from typing import Sequence

import numpy as np

from .bridge import serve
from .engine import BenchmarkResult, OpenVINOTensorCore


def _parse_devices(raw: str) -> list[str]:
    return [device.strip().upper() for device in raw.split(",") if device.strip()]


def _print_benchmarks(results: Sequence[BenchmarkResult]) -> None:
    print("device  backend                       avg_ms  min_ms  max_ms  iterations  output_shape")
    for result in results:
        print(
            f"{result.device:6}  "
            f"{result.backend:28}  "
            f"{result.average_ms:6.3f}  "
            f"{result.min_ms:6.3f}  "
            f"{result.max_ms:6.3f}  "
            f"{result.iterations:10d}  "
            f"{result.output_shape}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="npu-xmx",
        description="Tensor-core-style MatMul/Linear benchmarker across Intel OpenVINO NPU, GPU, and CPU.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    devices = subparsers.add_parser("devices", help="List available OpenVINO devices.")
    devices.add_argument("--json", action="store_true", help="Print the device list as JSON.")

    matmul = subparsers.add_parser("matmul", help="Benchmark a static-shape FP16 MatMul.")
    matmul.add_argument("--m", type=int, default=64)
    matmul.add_argument("--k", type=int, default=1024)
    matmul.add_argument("--n", type=int, default=1024)
    matmul.add_argument("--devices", default="NPU,GPU,CPU")
    matmul.add_argument("--iters", type=int, default=30)
    matmul.add_argument("--warmup", type=int, default=5)
    matmul.add_argument("--pad-to", type=int, default=16)
    matmul.add_argument("--hint", choices=["LATENCY", "THROUGHPUT"], default="THROUGHPUT")
    matmul.add_argument("--no-turbo", action="store_true")
    matmul.add_argument("--json", action="store_true")

    linear = subparsers.add_parser(
        "linear",
        help="Compile a fixed-weight linear block and benchmark repeated execution.",
    )
    linear.add_argument("--batch", type=int, default=64)
    linear.add_argument("--in-features", type=int, default=1024)
    linear.add_argument("--out-features", type=int, default=2048)
    linear.add_argument("--devices", default="NPU,GPU,CPU")
    linear.add_argument("--iters", type=int, default=30)
    linear.add_argument("--warmup", type=int, default=5)
    linear.add_argument("--pad-to", type=int, default=16)
    linear.add_argument("--hint", choices=["LATENCY", "THROUGHPUT"], default="THROUGHPUT")
    linear.add_argument("--no-turbo", action="store_true")
    linear.add_argument("--json", action="store_true")

    bridge = subparsers.add_parser(
        "serve",
        help="Run a localhost HTTP bridge for game mods or plugins.",
    )
    bridge.add_argument("--host", default="127.0.0.1")
    bridge.add_argument("--port", type=int, default=8765)
    bridge.add_argument("--socket-port", type=int, default=8766)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    engine = OpenVINOTensorCore()

    if args.command == "devices":
        payload = [engine.device_info(device) for device in engine.available_devices()]
        if args.json:
            print(
                json.dumps(
                    {
                        "devices": payload,
                        "native_npu_backend_available": engine.native_npu_backend_available(),
                    },
                    indent=2,
                )
            )
            return
        print(f"Native NPU backend: {'available' if engine.native_npu_backend_available() else 'unavailable'}")
        for item in payload:
            capabilities = ", ".join(item["capabilities"])
            print(f"{item['device']}: {item['name']} [{capabilities}]")
        return

    if args.command == "matmul":
        results = engine.benchmark_matmul(
            m=args.m,
            k=args.k,
            n=args.n,
            devices=_parse_devices(args.devices),
            dtype=np.float16,
            pad_to=args.pad_to,
            performance_hint=args.hint,
            turbo=not args.no_turbo,
            warmup=args.warmup,
            iterations=args.iters,
        )
        if args.json:
            print(json.dumps([result.to_dict() for result in results], indent=2))
            return
        _print_benchmarks(results)
        return

    if args.command == "linear":
        rng = np.random.default_rng(7)
        weights = rng.standard_normal((args.in_features, args.out_features), dtype=np.float32).astype(np.float16)
        inputs = rng.standard_normal((args.batch, args.in_features), dtype=np.float32).astype(np.float16)
        results: list[BenchmarkResult] = []
        for device in _parse_devices(args.devices):
            block = engine.compile_linear(
                weights,
                batch_size=args.batch,
                device=device,
                dtype=np.float16,
                pad_to=args.pad_to,
                performance_hint=args.hint,
                turbo=not args.no_turbo,
            )
            results.append(block.benchmark(inputs, iterations=args.iters, warmup=args.warmup))

        if args.json:
            print(json.dumps([result.to_dict() for result in results], indent=2))
            return
        _print_benchmarks(results)
        return

    if args.command == "serve":
        serve(host=args.host, port=args.port, socket_port=args.socket_port)
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
