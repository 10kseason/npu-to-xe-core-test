package dev.tensortest.npuxmxbridge;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.Map;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

final class PythonStdioAssistBackend implements AssistTranslatorBackend, AutoCloseable {
	private static final Gson GSON = new Gson();
	private static final String BUNDLED_RESOURCE_ROOT = "/bundled-python/npu_xmx/";
	private static final String[] BUNDLED_MODULE_FILES = {
		"__init__.py",
		"bridge.py",
		"cli.py",
		"engine.py",
		"translator_worker.py",
	};
	private static final Object BUNDLED_RUNTIME_LOCK = new Object();
	private static Path bundledPythonRoot;

	private final String pythonExecutable;
	private final String pythonModule;
	private final Path workingDirectory;
	private final Object ioLock = new Object();
	private final Deque<String> stderrTail = new ArrayDeque<>();
	private Process process;
	private BufferedWriter writer;
	private BufferedReader reader;
	private Thread stderrThread;
	private Path runtimePythonRoot;

	PythonStdioAssistBackend() {
		this(
			System.getProperty("npuxmxbridge.pythonExecutable", "python"),
			System.getProperty("npuxmxbridge.pythonModule", "npu_xmx.translator_worker"),
			resolveOptionalPath(System.getProperty("npuxmxbridge.pythonWorkingDir", "").trim())
		);
	}

	PythonStdioAssistBackend(String pythonExecutable, String pythonModule, Path workingDirectory) {
		this.pythonExecutable = pythonExecutable == null || pythonExecutable.isBlank() ? "python" : pythonExecutable;
		this.pythonModule = pythonModule == null || pythonModule.isBlank() ? "npu_xmx.translator_worker" : pythonModule;
		this.workingDirectory = workingDirectory;
	}

	@Override
	public String transportSuffix() {
		return "python-stdio";
	}

	@Override
	public String compile(AssistSessionSpec spec) {
		Map<String, Object> payload = new LinkedHashMap<>();
		payload.put("width", spec.width());
		payload.put("height", spec.height());
		payload.put("device", spec.device());
		payload.put("profile", spec.profile());
		payload.put("hint", spec.hint());
		payload.put("turbo", spec.turbo());
		return request("shader_compile", payload).get("session_id").getAsString();
	}

	@Override
	public AssistFrame run(String sessionId, AssistSceneState sceneState) {
		Map<String, Object> payload = new LinkedHashMap<>();
		payload.put("session_id", sessionId);
		payload.put("pos_x", sceneState.posX());
		payload.put("pos_y", sceneState.posY());
		payload.put("pos_z", sceneState.posZ());
		payload.put("yaw_degrees", sceneState.yawDegrees());
		payload.put("pitch_degrees", sceneState.pitchDegrees());
		payload.put("time_seconds", sceneState.timeSeconds());
		payload.put("sun_height", sceneState.sunHeight());
		payload.put("rain_strength", sceneState.rainStrength());
		payload.put("thunder_strength", sceneState.thunderStrength());
		payload.put("block_light", sceneState.blockLight());
		payload.put("sky_light", sceneState.skyLight());
		payload.put("submerged_factor", sceneState.submergedFactor());
		payload.put("quality_budget", sceneState.qualityBudget());
		payload.put("optimization_pressure", sceneState.optimizationPressure());

		long startedAt = System.nanoTime();
		JsonObject response = request("shader_run", payload);
		return new AssistFrame(
			response.get("width").getAsInt(),
			response.get("height").getAsInt(),
			readIntArray(response, "pixels_abgr"),
			response.get("mean_alpha").getAsDouble(),
			response.get("mean_luma").getAsDouble(),
			response.get("profile").getAsString(),
			response.get("backend").getAsString(),
			(System.nanoTime() - startedAt) / 1_000_000.0
		);
	}

	@Override
	public boolean release(String sessionId) {
		Map<String, Object> payload = new LinkedHashMap<>();
		payload.put("session_id", sessionId);
		return request("shader_release", payload).get("released").getAsBoolean();
	}

	@Override
	public void close() {
		synchronized (ioLock) {
			if (process == null) {
				return;
			}
			try {
				sendRequestUnchecked("shutdown", Map.of());
			} catch (RuntimeException ignored) {
				// Best-effort shutdown.
			}
			closeQuietly(writer);
			closeQuietly(reader);
			process.destroy();
			process = null;
			writer = null;
			reader = null;
			stderrTail.clear();
		}
	}

	private JsonObject request(String op, Map<String, Object> payload) {
		synchronized (ioLock) {
			try {
				ensureProcess();
				return sendRequestUnchecked(op, payload);
			} catch (IOException exception) {
				String message = buildWorkerErrorMessage(exception);
				close();
				throw new RuntimeException(message, exception);
			} catch (RuntimeException exception) {
				close();
				throw exception;
			}
		}
	}

	private JsonObject sendRequestUnchecked(String op, Map<String, Object> payload) {
		try {
			return sendRequest(op, payload);
		} catch (IOException exception) {
			throw new RuntimeException(buildWorkerErrorMessage(exception), exception);
		}
	}

	private JsonObject sendRequest(String op, Map<String, Object> payload) throws IOException {
		if (writer == null || reader == null) {
			throw new IOException("Python stdio worker is not initialized.");
		}

		Map<String, Object> requestPayload = new LinkedHashMap<>();
		requestPayload.put("op", op);
		requestPayload.put("payload", payload);

		writer.write(GSON.toJson(requestPayload));
		writer.newLine();
		writer.flush();

		String rawResponse = reader.readLine();
		if (rawResponse == null) {
			throw new IOException("Python stdio worker closed the pipe.");
		}

		JsonObject response = JsonParser.parseString(rawResponse).getAsJsonObject();
		if (!response.get("ok").getAsBoolean()) {
			String detail = response.has("error") ? response.get("error").getAsString() : "Unknown python worker failure";
			throw new IOException(detail);
		}
		return response.getAsJsonObject("result");
	}

	private void ensureProcess() throws IOException {
		if (process != null && process.isAlive() && writer != null && reader != null) {
			return;
		}

		close();
		ProcessBuilder builder = new ProcessBuilder(pythonExecutable, "-m", pythonModule);
		runtimePythonRoot = resolvePythonSearchRoot();
		builder.directory(runtimePythonRoot.toFile());
		String pythonPath = runtimePythonRoot.toAbsolutePath().toString();
		String existing = builder.environment().getOrDefault("PYTHONPATH", "");
		builder.environment().put("PYTHONPATH", existing.isBlank() ? pythonPath : pythonPath + java.io.File.pathSeparator + existing);
		process = builder.start();
		writer = new BufferedWriter(new OutputStreamWriter(process.getOutputStream(), StandardCharsets.UTF_8));
		reader = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8));
		startStderrDrainer(process);
		sendRequest("health", Map.of());
	}

	private void startStderrDrainer(Process runningProcess) {
		stderrThread = new Thread(() -> {
			try (BufferedReader stderrReader = new BufferedReader(new InputStreamReader(runningProcess.getErrorStream(), StandardCharsets.UTF_8))) {
				String line;
				while ((line = stderrReader.readLine()) != null) {
					synchronized (ioLock) {
						stderrTail.addLast(line);
						while (stderrTail.size() > 24) {
							stderrTail.removeFirst();
						}
					}
				}
			} catch (IOException ignored) {
				// Best-effort worker diagnostics.
			}
		}, "npu-xmx-python-stdio-stderr");
		stderrThread.setDaemon(true);
		stderrThread.start();
	}

	private String buildWorkerErrorMessage(IOException exception) {
		StringBuilder message = new StringBuilder("Python stdio translator error");
		Path cwd = runtimePythonRoot != null ? runtimePythonRoot : workingDirectory;
		if (cwd != null) {
			message.append(" [cwd=").append(cwd).append(']');
		}
		String detail = exception.getMessage();
		if (detail != null && !detail.isBlank()) {
			message.append(": ").append(detail);
		}
		synchronized (ioLock) {
			if (!stderrTail.isEmpty()) {
				message.append(" | stderr tail: ");
				message.append(String.join(" | ", stderrTail));
			}
		}
		if (process != null && !process.isAlive()) {
			message.append(" | exit=").append(process.exitValue());
		}
		return message.toString();
	}

	private static int[] readIntArray(JsonObject response, String key) {
		int size = response.getAsJsonArray(key).size();
		int[] values = new int[size];
		for (int index = 0; index < size; index++) {
			values[index] = response.getAsJsonArray(key).get(index).getAsInt();
		}
		return values;
	}

	private static Path resolveOptionalPath(String raw) {
		if (raw == null || raw.isBlank()) {
			return null;
		}
		Path path = Path.of(raw);
		if (!path.isAbsolute()) {
			path = Path.of("").toAbsolutePath().resolve(path).normalize();
		}
		return Files.exists(path) ? path : path;
	}

	private Path resolvePythonSearchRoot() throws IOException {
		if (workingDirectory == null) {
			return ensureBundledPythonRoot();
		}

		Path sourceRoot = workingDirectory.resolve("src");
		if (Files.exists(sourceRoot.resolve("npu_xmx"))) {
			return sourceRoot.toAbsolutePath().normalize();
		}
		if (Files.exists(workingDirectory.resolve("npu_xmx"))) {
			return workingDirectory.toAbsolutePath().normalize();
		}
		return workingDirectory.toAbsolutePath().normalize();
	}

	static Path ensureBundledPythonRoot() throws IOException {
		synchronized (BUNDLED_RUNTIME_LOCK) {
			if (bundledPythonRoot != null && Files.exists(bundledPythonRoot.resolve("npu_xmx").resolve("translator_worker.py"))) {
				return bundledPythonRoot;
			}

			Path root = Files.createTempDirectory("npu-xmx-bundled-python-");
			Path packageDir = Files.createDirectories(root.resolve("npu_xmx"));
			for (String fileName : BUNDLED_MODULE_FILES) {
				Path target = packageDir.resolve(fileName);
				try (InputStream stream = PythonStdioAssistBackend.class.getResourceAsStream(BUNDLED_RESOURCE_ROOT + fileName)) {
					if (stream == null) {
						throw new IOException("Bundled python resource missing: " + BUNDLED_RESOURCE_ROOT + fileName);
					}
					Files.copy(stream, target, StandardCopyOption.REPLACE_EXISTING);
				}
			}
			bundledPythonRoot = root.toAbsolutePath().normalize();
			return bundledPythonRoot;
		}
	}

	private static void closeQuietly(AutoCloseable closeable) {
		if (closeable == null) {
			return;
		}
		try {
			closeable.close();
		} catch (Exception ignored) {
			// Best-effort cleanup.
		}
	}
}
