package dev.tensortest.npuxmxbridge;

import java.nio.file.Path;
import java.util.Locale;

final class NpuBridgeTranslatorClient implements NpuBridgeClient {
	private final String device;
	private final String shaderProfile;
	private final String hint;
	private final boolean turbo;
	private final AssistTranslatorBackend backend;
	private final Object sessionLock = new Object();
	private String shaderSessionId;
	private int shaderWidth = -1;
	private int shaderHeight = -1;

	NpuBridgeTranslatorClient() {
		this(
			System.getProperty("npuxmxbridge.device", "NPU"),
			System.getProperty("npuxmxbridge.shaderProfile", "intel_npu_gi_v3"),
			"THROUGHPUT",
			true,
			createBackendFromSystemProperties()
		);
	}

	NpuBridgeTranslatorClient(
		String device,
		String shaderProfile,
		String hint,
		boolean turbo,
		AssistTranslatorBackend backend
	) {
		this.device = device == null || device.isBlank() ? "AUTO" : device;
		this.shaderProfile = shaderProfile == null || shaderProfile.isBlank() ? "intel_npu_gi_v3" : shaderProfile;
		this.hint = hint == null || hint.isBlank() ? "THROUGHPUT" : hint;
		this.turbo = turbo;
		this.backend = backend;
	}

	@Override
	public String transportName() {
		return "translator-java/" + backend.transportSuffix();
	}

	@Override
	public ShaderFrameResult runShaderFrame(ShaderFrameRequest request) {
		long startedAt = System.nanoTime();
		try {
			synchronized (sessionLock) {
				String sessionId = ensureShaderSession(request.width(), request.height());
				AssistFrame frame = backend.run(sessionId, AssistSceneState.fromRequest(request));
				double elapsedMs = (System.nanoTime() - startedAt) / 1_000_000.0;
				return new ShaderFrameResult(
					frame.width(),
					frame.height(),
					frame.pixelsAbgr(),
					String.format(
						Locale.ROOT,
						"[%s %.2f ms] NPU shader %s/%s %dx%d luma=%.2f alpha=%.2f",
						transportName(),
						elapsedMs,
						frame.profile(),
						frame.backend(),
						frame.width(),
						frame.height(),
						frame.meanLuma(),
						frame.meanAlpha()
					),
					elapsedMs,
					frame.meanAlpha(),
					frame.meanLuma(),
					frame.profile(),
					frame.backend()
				);
			}
		} catch (RuntimeException exception) {
			resetSession();
			throw exception;
		}
	}

	private String ensureShaderSession(int width, int height) {
		if (shaderSessionId != null && shaderWidth == width && shaderHeight == height) {
			return shaderSessionId;
		}

		releaseShaderSession();
		AssistSessionSpec spec = new AssistSessionSpec(width, height, device, shaderProfile, hint, turbo);
		shaderSessionId = backend.compile(spec);
		shaderWidth = width;
		shaderHeight = height;
		return shaderSessionId;
	}

	private void resetSession() {
		synchronized (sessionLock) {
			releaseShaderSession();
		}
	}

	private void releaseShaderSession() {
		if (shaderSessionId != null) {
			try {
				backend.release(shaderSessionId);
			} catch (RuntimeException ignored) {
				// Best-effort cleanup after translator failures.
			}
		}
		shaderSessionId = null;
		shaderWidth = -1;
		shaderHeight = -1;
	}

	private static AssistTranslatorBackend createBackendFromSystemProperties() {
		String configured = System.getProperty("npuxmxbridge.translatorBackend", "procedural").trim().toLowerCase(Locale.ROOT);
		if ("python".equals(configured) || "python-stdio".equals(configured) || "live-npu".equals(configured)) {
			return new PythonStdioAssistBackend();
		}
		if ("replay".equals(configured)) {
			String rawPath = System.getProperty("npuxmxbridge.translatorFixture", "").trim();
			if (!rawPath.isEmpty()) {
				return new ReplayAssistBackend(resolveFixturePath(rawPath));
			}
			return new ReplayAssistBackend();
		}
		if ("procedural".equals(configured)) {
			return new ProceduralAssistBackend();
		}
		throw new IllegalArgumentException("Unsupported translator backend: " + configured);
	}

	private static Path resolveFixturePath(String rawPath) {
		Path path = Path.of(rawPath);
		if (path.isAbsolute()) {
			return path;
		}
		return Path.of("").toAbsolutePath().resolve(path).normalize();
	}
}
