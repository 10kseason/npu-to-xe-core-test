package dev.tensortest.npuxmxbridge;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.ConnectException;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.SocketException;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

final class NpuBridgeSocketClient implements NpuBridgeClient {
	private static final Gson GSON = new Gson();

	private final String host;
	private final int port;
	private final String device;
	private final Object ioLock = new Object();
	private Socket socket;
	private BufferedReader reader;
	private BufferedWriter writer;
	private String shaderSessionId;
	private int shaderWidth = -1;
	private int shaderHeight = -1;

	NpuBridgeSocketClient() {
		this.host = System.getProperty("npuxmxbridge.socketHost", "127.0.0.1");
		this.port = Integer.getInteger("npuxmxbridge.socketPort", 8766);
		this.device = System.getProperty("npuxmxbridge.device", "NPU");
	}

	@Override
	public String transportName() {
		return "socket";
	}

	@Override
	public ShaderFrameResult runShaderFrame(ShaderFrameRequest requestData) {
		long startedAt = System.nanoTime();

		try {
			synchronized (ioLock) {
				ensureConnection();
				String activeSession = ensureShaderSession(requestData.width(), requestData.height());

				Map<String, Object> payload = new LinkedHashMap<>();
				payload.put("session_id", activeSession);
				payload.put("pos_x", requestData.sample().x());
				payload.put("pos_y", requestData.sample().y());
				payload.put("pos_z", requestData.sample().z());
				payload.put("yaw_degrees", requestData.sample().yaw());
				payload.put("pitch_degrees", requestData.sample().pitch());
				payload.put("time_seconds", requestData.timeSeconds());

				JsonObject response = request("shader_run", payload);
				double elapsedMs = (System.nanoTime() - startedAt) / 1_000_000.0;
				double meanLuma = response.get("mean_luma").getAsDouble();
				double meanAlpha = response.get("mean_alpha").getAsDouble();
				int width = response.get("width").getAsInt();
				int height = response.get("height").getAsInt();

				return new ShaderFrameResult(
					width,
					height,
					readIntArray(response, "pixels_abgr"),
					String.format(
						"[socket %.2f ms] NPU shader %dx%d luma=%.2f alpha=%.2f",
						elapsedMs,
						width,
						height,
						meanLuma,
						meanAlpha
					),
					elapsedMs,
					meanAlpha,
					meanLuma
				);
			}
		} catch (IOException exception) {
			resetConnection();
			throw new RuntimeException(buildBridgeErrorMessage(exception), exception);
		} catch (RuntimeException exception) {
			resetConnection();
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

	private void ensureConnection() throws IOException {
		if (socket != null && socket.isConnected() && !socket.isClosed()) {
			return;
		}

		resetConnection();
		socket = new Socket();
		socket.setTcpNoDelay(true);
		socket.connect(new InetSocketAddress(host, port), 2_000);
		socket.setSoTimeout(5_000);
		reader = new BufferedReader(new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8));
		writer = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream(), StandardCharsets.UTF_8));
	}

	private String ensureShaderSession(int width, int height) throws IOException {
		if (shaderSessionId != null && shaderWidth == width && shaderHeight == height) {
			return shaderSessionId;
		}

		releaseShaderSession();
		Map<String, Object> payload = new LinkedHashMap<>();
		payload.put("width", width);
		payload.put("height", height);
		payload.put("device", device);
		payload.put("pad_to", 16);
		payload.put("hint", "THROUGHPUT");
		payload.put("turbo", Boolean.TRUE);

		JsonObject response = request("shader_compile", payload);
		shaderSessionId = response.get("session_id").getAsString();
		shaderWidth = width;
		shaderHeight = height;
		return shaderSessionId;
	}

	private void releaseShaderSession() {
		if (shaderSessionId == null) {
			return;
		}

		try {
			Map<String, Object> payload = new LinkedHashMap<>();
			payload.put("session_id", shaderSessionId);
			request("shader_release", payload);
		} catch (IOException ignored) {
			// Best-effort cleanup when swapping shader session sizes.
		} finally {
			shaderSessionId = null;
			shaderWidth = -1;
			shaderHeight = -1;
		}
	}

	private JsonObject request(String op, Map<String, Object> payload) throws IOException {
		if (writer == null || reader == null) {
			throw new IOException("Socket bridge connection is not initialized.");
		}

		Map<String, Object> requestPayload = new LinkedHashMap<>();
		requestPayload.put("op", op);
		requestPayload.put("payload", payload);

		writer.write(GSON.toJson(requestPayload));
		writer.write("\n");
		writer.flush();

		String rawResponse = reader.readLine();
		if (rawResponse == null) {
			throw new EOFException("Socket bridge closed the connection.");
		}

		JsonObject response = JsonParser.parseString(rawResponse).getAsJsonObject();
		if (!response.get("ok").getAsBoolean()) {
			throw new IOException(response.get("error").getAsString());
		}

		return response.getAsJsonObject("result");
	}

	private String buildBridgeErrorMessage(IOException exception) {
		Throwable cause = exception.getCause();
		if (exception instanceof ConnectException || cause instanceof ConnectException) {
			return "Socket bridge not running at tcp://" + host + ":" + port
				+ ". Start `npu-xmx serve --host 127.0.0.1 --port 8765 --socket-port 8766` first.";
		}
		if (exception instanceof SocketException && exception.getMessage() != null && exception.getMessage().contains("Connection reset")) {
			return "Socket bridge connection was reset. Restart `npu-xmx serve` and try again.";
		}

		String detail = exception.getMessage();
		if (detail == null || detail.isBlank()) {
			detail = exception.getClass().getSimpleName();
		}

		return "Socket bridge error at tcp://" + host + ":" + port + ": " + detail;
	}

	private void resetConnection() {
		releaseShaderSession();

		if (reader != null) {
			try {
				reader.close();
			} catch (IOException ignored) {
				// Best-effort cleanup after a bridge failure.
			}
			reader = null;
		}

		if (writer != null) {
			try {
				writer.close();
			} catch (IOException ignored) {
				// Best-effort cleanup after a bridge failure.
			}
			writer = null;
		}

		if (socket != null) {
			try {
				socket.close();
			} catch (IOException ignored) {
				// Best-effort cleanup after a bridge failure.
			}
			socket = null;
		}
	}
}
