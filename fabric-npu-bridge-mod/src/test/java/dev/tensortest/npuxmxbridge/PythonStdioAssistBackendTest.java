package dev.tensortest.npuxmxbridge;

import java.io.IOException;
import java.nio.file.Path;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PythonStdioAssistBackendTest {
	@Test
	void pythonStdioBackendRunsShaderRoundTripOnCpuFromBundledRuntime() throws Exception {
		Path bundledRoot = PythonStdioAssistBackend.ensureBundledPythonRoot();
		Assumptions.assumeTrue(canImportWorker(bundledRoot));

		try (PythonStdioAssistBackend backend = new PythonStdioAssistBackend("python", "npu_xmx.translator_worker", null)) {
			String sessionId = backend.compile(new AssistSessionSpec(4, 4, "CPU", "intel_npu_shader2_v1", "THROUGHPUT", true));
			AssistFrame frame = backend.run(
				sessionId,
				new AssistSceneState(4.0, 72.0, -8.0, 18.0, 6.0, 0.8, 0.8, 0.1, 0.0, 0.5, 0.9, 0.0, 0.7, 0.2)
			);

			assertEquals(4, frame.width());
			assertEquals(4, frame.height());
			assertEquals(16, frame.pixelsAbgr().length);
			assertEquals("intel_npu_shader2_v1", frame.profile());
			assertTrue(frame.backend().contains("pipeline") || frame.backend().contains("openvino") || frame.backend().contains("native"));
			assertTrue(backend.release(sessionId));
		}
	}

	private static boolean canImportWorker(Path repoRoot) {
		ProcessBuilder builder = new ProcessBuilder("python", "-c", "import openvino, npu_xmx.translator_worker; print('ok')");
		builder.directory(repoRoot.toFile());
		builder.environment().put(
			"PYTHONPATH",
			repoRoot.toAbsolutePath().toString()
		);
		try {
			Process process = builder.start();
			int exitCode = process.waitFor();
			return exitCode == 0;
		} catch (IOException | InterruptedException exception) {
			Thread.currentThread().interrupt();
			return false;
		}
	}
}
