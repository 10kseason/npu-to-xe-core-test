package dev.tensortest.npuxmxbridge;

import java.io.IOException;
import java.net.ConnectException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Map;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

final class NpuBridgeHttpClient implements NpuBridgeClient {
	private static final Gson GSON = new Gson();

	private final HttpClient httpClient;
	private final String baseUrl;
	private final URI compileUri;
	private final URI runUri;
	private final String device;
	private final String shaderProfile;
	private final Object sessionLock = new Object();
	private volatile String shaderSessionId;
	private volatile int shaderWidth = -1;
	private volatile int shaderHeight = -1;

	NpuBridgeHttpClient() {
		this.baseUrl = System.getProperty("npuxmxbridge.url", "http://127.0.0.1:8765");
		this.device = System.getProperty("npuxmxbridge.device", "NPU");
		this.shaderProfile = System.getProperty("npuxmxbridge.shaderProfile", "intel_npu_gi_v3");
		this.httpClient = HttpClient.newBuilder()
			.connectTimeout(Duration.ofSeconds(2))
			.build();
		this.compileUri = URI.create(baseUrl + "/shader/compile");
		this.runUri = URI.create(baseUrl + "/shader/run");
	}

	@Override
	public String transportName() {
		return "http";
	}

	@Override
	public ShaderFrameResult runShaderFrame(ShaderFrameRequest requestData) {
		long startedAt = System.nanoTime();
		try {
			String activeSession = ensureShaderSession(requestData.width(), requestData.height());

			Map<String, Object> payload = new LinkedHashMap<>();
			payload.put("session_id", activeSession);
			payload.put("pos_x", requestData.sample().x());
			payload.put("pos_y", requestData.sample().y());
			payload.put("pos_z", requestData.sample().z());
			payload.put("yaw_degrees", requestData.sample().yaw());
			payload.put("pitch_degrees", requestData.sample().pitch());
			payload.put("time_seconds", requestData.timeSeconds());
			payload.put("sun_height", requestData.sample().sunHeight());
			payload.put("rain_strength", requestData.sample().rainStrength());
			payload.put("thunder_strength", requestData.sample().thunderStrength());
			payload.put("block_light", requestData.sample().blockLight());
			payload.put("sky_light", requestData.sample().skyLight());
			payload.put("submerged_factor", requestData.sample().submergedFactor());
			payload.put("quality_budget", requestData.sample().qualityBudget());
			payload.put("optimization_pressure", requestData.sample().optimizationPressure());

			JsonObject response = post(runUri, payload);
			double elapsedMs = (System.nanoTime() - startedAt) / 1_000_000.0;
			double meanLuma = response.get("mean_luma").getAsDouble();
			double meanAlpha = response.get("mean_alpha").getAsDouble();
			int width = response.get("width").getAsInt();
			int height = response.get("height").getAsInt();
			String profile = response.get("profile").getAsString();
			String backend = response.get("backend").getAsString();

			return new ShaderFrameResult(
				width,
				height,
				readIntArray(response, "pixels_abgr"),
				String.format(
					"[http %.2f ms] NPU shader %s/%s %dx%d luma=%.2f alpha=%.2f",
					elapsedMs,
					profile,
					backend,
					width,
					height,
					meanLuma,
					meanAlpha
				),
				elapsedMs,
				meanAlpha,
				meanLuma,
				profile,
				backend
			);
		} catch (InterruptedException exception) {
			Thread.currentThread().interrupt();
			resetSession();
			throw new RuntimeException("Bridge request interrupted", exception);
		} catch (IOException exception) {
			resetSession();
			throw new RuntimeException(buildBridgeErrorMessage(exception), exception);
		} catch (RuntimeException exception) {
			resetSession();
			throw exception;
		}
	}

	private int[] readIntArray(JsonObject response, String key) {
		int size = response.getAsJsonArray(key).size();
		int[] values = new int[size];
		for (int index = 0; index < size; index++) {
			values[index] = response.getAsJsonArray(key).get(index).getAsInt();
		}
		return values;
	}

	private String buildBridgeErrorMessage(IOException exception) {
		Throwable cause = exception.getCause();
		if (exception instanceof ConnectException || cause instanceof ConnectException) {
			return "Bridge not running at " + baseUrl + ". Start `npu-xmx serve --host 127.0.0.1 --port 8765` first.";
		}

		String detail = exception.getMessage();
		if (detail == null || detail.isBlank()) {
			detail = exception.getClass().getSimpleName();
		}

		return "Bridge error at " + baseUrl + ": " + detail;
	}

	private String ensureShaderSession(int width, int height) throws IOException, InterruptedException {
		if (shaderSessionId != null && shaderWidth == width && shaderHeight == height) {
			return shaderSessionId;
		}

		synchronized (sessionLock) {
			if (shaderSessionId != null && shaderWidth == width && shaderHeight == height) {
				return shaderSessionId;
			}

			Map<String, Object> payload = new LinkedHashMap<>();
			payload.put("width", width);
			payload.put("height", height);
			payload.put("device", device);
			payload.put("profile", shaderProfile);
			payload.put("pad_to", 16);
			payload.put("hint", "THROUGHPUT");
			payload.put("turbo", Boolean.TRUE);

			JsonObject response = post(compileUri, payload);
			shaderSessionId = response.get("session_id").getAsString();
			shaderWidth = width;
			shaderHeight = height;
			return shaderSessionId;
		}
	}

	private void resetSession() {
		shaderSessionId = null;
		shaderWidth = -1;
		shaderHeight = -1;
	}

	private JsonObject post(URI uri, Map<String, Object> payload) throws IOException, InterruptedException {
		HttpRequest request = HttpRequest.newBuilder(uri)
			.timeout(Duration.ofSeconds(5))
			.header("Content-Type", "application/json")
			.POST(HttpRequest.BodyPublishers.ofString(GSON.toJson(payload)))
			.build();

		HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
		if (response.statusCode() < 200 || response.statusCode() >= 300) {
			throw new IOException("Bridge returned " + response.statusCode() + ": " + response.body());
		}

		return JsonParser.parseString(response.body()).getAsJsonObject();
	}
}
