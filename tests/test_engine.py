from __future__ import annotations

import unittest

import numpy as np

from npu_xmx.engine import OpenVINOTensorCore


class OpenVINOTensorCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = OpenVINOTensorCore(preferred_devices=("CPU",))

    def test_resolve_auto_prefers_cpu_when_requested(self) -> None:
        self.assertEqual(self.engine.resolve_device("AUTO"), "CPU")

    def test_matmul_matches_numpy_on_cpu(self) -> None:
        rng = np.random.default_rng(0)
        lhs = rng.standard_normal((7, 11), dtype=np.float32).astype(np.float16)
        rhs = rng.standard_normal((11, 5), dtype=np.float32).astype(np.float16)

        output = self.engine.matmul(lhs, rhs, device="CPU", pad_to=16)
        expected = lhs @ rhs

        np.testing.assert_allclose(output, expected, rtol=1e-2, atol=1e-2)

    def test_compile_linear_matches_numpy_on_cpu(self) -> None:
        rng = np.random.default_rng(1)
        weight = rng.standard_normal((13, 9), dtype=np.float32).astype(np.float16)
        bias = rng.standard_normal((9,), dtype=np.float32).astype(np.float16)
        inputs = rng.standard_normal((4, 13), dtype=np.float32).astype(np.float16)

        block = self.engine.compile_linear(
            weight,
            bias=bias,
            batch_size=4,
            device="CPU",
            pad_to=16,
        )
        output = block(inputs)
        expected = inputs @ weight + bias

        np.testing.assert_allclose(output, expected, rtol=1e-2, atol=1e-2)

    def test_compile_linear_rejects_wrong_feature_count(self) -> None:
        weight = np.ones((8, 4), dtype=np.float16)
        block = self.engine.compile_linear(weight, batch_size=2, device="CPU")
        with self.assertRaises(ValueError):
            block(np.ones((2, 7), dtype=np.float16))

    def test_native_npu_matmul_matches_cpu_when_available(self) -> None:
        if "NPU" not in self.engine.available_devices():
            self.skipTest("NPU device is not available")
        if not self.engine.native_npu_backend_available():
            self.skipTest("intel_npu_acceleration_library is not available")

        rng = np.random.default_rng(2)
        lhs = rng.standard_normal((8, 16), dtype=np.float32).astype(np.float16)
        rhs = rng.standard_normal((16, 8), dtype=np.float32).astype(np.float16)

        native_output = self.engine.matmul(lhs, rhs, device="NPU", pad_to=16)
        cpu_engine = OpenVINOTensorCore(preferred_devices=("CPU",), prefer_native_npu_backend=False)
        cpu_output = cpu_engine.matmul(lhs, rhs, device="CPU", pad_to=16)

        np.testing.assert_allclose(native_output, cpu_output, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
