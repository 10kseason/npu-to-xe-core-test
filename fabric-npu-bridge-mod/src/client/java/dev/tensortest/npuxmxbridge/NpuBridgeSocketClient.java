package dev.tensortest.npuxmxbridge;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.ConnectException;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.SocketException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

final class NpuBridgeSocketClient implements NpuBridgeClient {
	private static final Gson GSON = new Gson();
	private static final byte[] BINARY_MAGIC = new byte[] { 'N', 'P', 'X', 'B' };
	private static final int BINARY_VERSION = 1;
	private static final int BINARY_KIND_SHADER_RUN_REQUEST_V3 = 4;
	private static final int BINARY_KIND_SHADER_RUN_RESPONSE = 2;
	private static final int BINARY_KIND_ERROR = 255;
	private static final int BINARY_HEADER_SIZE = 12;
	private static final int BINARY_SHADER_RUN_REQUEST_V3_SIZE = 16 + (Float.BYTES * 14);

	private final String host;
	private final int port;
	private final String device;
	private final String shaderProfile;
	private final Object ioLock = new Object();
	private Socket socket;
	private InputStream input;
	private OutputStream output;
	private String shaderSessionId;
	private int shaderWidth = -1;
	private int shaderHeight = -1;

	NpuBridgeSocketClient() {
		this.host = System.getProperty("npuxmxbridge.socketHost", "127.0.0.1");
		this.port = Integer.getInteger("npuxmxbridge.socketPort", 8766);
		this.device = System.getProperty("npuxmxbridge.device", "NPU");
		this.shaderProfile = System.getProperty("npuxmxbridge.shaderProfile", "intel_npu_gi_v3");
	}

	@Override
	public String transportName() {
		return "socket-binary";
	}

	@Override
	public ShaderFrameResult runShaderFrame(ShaderFrameRequest requestData) {
		long startedAt = System.nanoTime();

		try {
			synchronized (ioLock) {
				ensureConnection();
				String activeSession = ensureShaderSession(requestData.width(), requestData.height());

				ShaderFrameResult result = requestBinaryShaderRun(activeSession, requestData, startedAt);
				double elapsedMs = (System.nanoTime() - startedAt) / 1_000_000.0;

				return new ShaderFrameResult(
					result.width(),
					result.height(),
					result.pixelsAbgr(),
					String.format(
						"[socket-binary %.2f ms] NPU shader %s/%s %dx%d luma=%.2f alpha=%.2f",
						elapsedMs,
						result.profile(),
						result.backend(),
						result.width(),
						result.height(),
						result.meanLuma(),
						result.meanAlpha()
					),
					elapsedMs,
					result.meanAlpha(),
					result.meanLuma(),
					result.profile(),
					result.backend()
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

	private ShaderFrameResult requestBinaryShaderRun(String sessionId, ShaderFrameRequest requestData, long startedAt) throws IOException {
		if (input == null || output == null) {
			throw new IOException("Socket bridge connection is not initialized.");
		}

		ByteBuffer payload = ByteBuffer.allocate(BINARY_SHADER_RUN_REQUEST_V3_SIZE).order(ByteOrder.LITTLE_ENDIAN);
		payload.put(decodeSessionId(sessionId));
		payload.putFloat((float) requestData.sample().x());
		payload.putFloat((float) requestData.sample().y());
		payload.putFloat((float) requestData.sample().z());
		payload.putFloat((float) requestData.sample().yaw());
		payload.putFloat((float) requestData.sample().pitch());
		payload.putFloat((float) requestData.timeSeconds());
		payload.putFloat((float) requestData.sample().sunHeight());
		payload.putFloat((float) requestData.sample().rainStrength());
		payload.putFloat((float) requestData.sample().thunderStrength());
		payload.putFloat((float) requestData.sample().blockLight());
		payload.putFloat((float) requestData.sample().skyLight());
		payload.putFloat((float) requestData.sample().submergedFactor());
		payload.putFloat((float) requestData.sample().qualityBudget());
		payload.putFloat((float) requestData.sample().optimizationPressure());
		writeBinaryFrame(BINARY_KIND_SHADER_RUN_REQUEST_V3, payload.array());

		BinaryFrame frame = readBinaryFrame();
		if (frame.kind() == BINARY_KIND_ERROR) {
			throw new IOException(decodeBinaryError(frame.payload()));
		}
		if (frame.kind() != BINARY_KIND_SHADER_RUN_RESPONSE) {
			throw new IOException("Unexpected binary bridge frame kind: " + frame.kind());
		}

		ByteBuffer response = ByteBuffer.wrap(frame.payload()).order(ByteOrder.LITTLE_ENDIAN);
		int width = response.getInt();
		int height = response.getInt();
		double meanAlpha = response.getFloat();
		double meanLuma = response.getFloat();
		int profileLength = Short.toUnsignedInt(response.getShort());
		int backendLength = Short.toUnsignedInt(response.getShort());
		int pixelCount = response.getInt();
		byte[] profileBytes = new byte[profileLength];
		byte[] backendBytes = new byte[backendLength];
		response.get(profileBytes);
		response.get(backendBytes);

		int[] pixels = new int[pixelCount];
		for (int index = 0; index < pixelCount; index++) {
			pixels[index] = response.getInt();
		}

		return new ShaderFrameResult(
			width,
			height,
			pixels,
			String.format(
				"[socket-binary %.2f ms] NPU shader %dx%d",
				(System.nanoTime() - startedAt) / 1_000_000.0,
				width,
				height
			),
			0.0,
			meanAlpha,
			meanLuma,
			new String(profileBytes, StandardCharsets.UTF_8),
			new String(backendBytes, StandardCharsets.UTF_8)
		);
	}

	private void ensureConnection() throws IOException {
		if (socket != null && socket.isConnected() && !socket.isClosed()) {
			return;
		}

		resetConnection();
		socket = new Socket();
		socket.setTcpNoDelay(true);
		socket.connect(new InetSocketAddress(host, port), 2_000);
		socket.setSoTimeout(10_000);
		input = new BufferedInputStream(socket.getInputStream());
		output = new BufferedOutputStream(socket.getOutputStream());
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
		payload.put("profile", shaderProfile);
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
		if (output == null || input == null) {
			throw new IOException("Socket bridge connection is not initialized.");
		}

		Map<String, Object> requestPayload = new LinkedHashMap<>();
		requestPayload.put("op", op);
		requestPayload.put("payload", payload);

		byte[] encoded = GSON.toJson(requestPayload).getBytes(StandardCharsets.UTF_8);
		output.write(encoded);
		output.write('\n');
		output.flush();

		String rawResponse = readJsonLine();
		JsonObject response = JsonParser.parseString(rawResponse).getAsJsonObject();
		if (!response.get("ok").getAsBoolean()) {
			throw new IOException(response.get("error").getAsString());
		}

		return response.getAsJsonObject("result");
	}

	private void writeBinaryFrame(int kind, byte[] payload) throws IOException {
		ByteBuffer header = ByteBuffer.allocate(BINARY_HEADER_SIZE).order(ByteOrder.LITTLE_ENDIAN);
		header.put(BINARY_MAGIC);
		header.put((byte) BINARY_VERSION);
		header.put((byte) kind);
		header.putShort((short) 0);
		header.putInt(payload.length);
		output.write(header.array());
		output.write(payload);
		output.flush();
	}

	private BinaryFrame readBinaryFrame() throws IOException {
		byte[] headerBytes = readFully(BINARY_HEADER_SIZE);
		ByteBuffer header = ByteBuffer.wrap(headerBytes).order(ByteOrder.LITTLE_ENDIAN);
		byte[] magic = new byte[4];
		header.get(magic);
		int version = Byte.toUnsignedInt(header.get());
		int kind = Byte.toUnsignedInt(header.get());
		header.getShort();
		int payloadLength = header.getInt();
		if (!Arrays.equals(magic, BINARY_MAGIC)) {
			throw new IOException("Socket bridge returned an invalid binary frame.");
		}
		if (version != BINARY_VERSION) {
			throw new IOException("Unsupported binary bridge version: " + version);
		}
		return new BinaryFrame(kind, readFully(payloadLength));
	}

	private String decodeBinaryError(byte[] payload) throws IOException {
		if (payload.length < Short.BYTES) {
			throw new IOException("Socket bridge returned a truncated binary error.");
		}
		ByteBuffer buffer = ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN);
		int messageLength = Short.toUnsignedInt(buffer.getShort());
		if (messageLength > buffer.remaining()) {
			throw new IOException("Socket bridge returned a malformed binary error.");
		}
		byte[] messageBytes = new byte[messageLength];
		buffer.get(messageBytes);
		return new String(messageBytes, StandardCharsets.UTF_8);
	}

	private String readJsonLine() throws IOException {
		ByteArrayOutputStream buffer = new ByteArrayOutputStream();
		while (true) {
			int value = input.read();
			if (value < 0) {
				throw new EOFException("Socket bridge closed the connection.");
			}
			if (value == '\n') {
				break;
			}
			buffer.write(value);
		}
		return new String(buffer.toByteArray(), StandardCharsets.UTF_8);
	}

	private byte[] readFully(int length) throws IOException {
		byte[] buffer = new byte[length];
		int offset = 0;
		while (offset < length) {
			int read = input.read(buffer, offset, length - offset);
			if (read < 0) {
				throw new EOFException("Socket bridge closed the connection.");
			}
			offset += read;
		}
		return buffer;
	}

	private byte[] decodeSessionId(String sessionId) throws IOException {
		if (sessionId == null || sessionId.length() != 32) {
			throw new IOException("Binary shader_run requires a 32-character hex session id.");
		}

		byte[] decoded = new byte[16];
		for (int index = 0; index < decoded.length; index++) {
			int high = Character.digit(sessionId.charAt(index * 2), 16);
			int low = Character.digit(sessionId.charAt(index * 2 + 1), 16);
			if (high < 0 || low < 0) {
				throw new IOException("Binary shader_run requires a hex-encoded session id.");
			}
			decoded[index] = (byte) ((high << 4) | low);
		}
		return decoded;
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

		if (input != null) {
			try {
				input.close();
			} catch (IOException ignored) {
				// Best-effort cleanup after a bridge failure.
			}
			input = null;
		}

		if (output != null) {
			try {
				output.close();
			} catch (IOException ignored) {
				// Best-effort cleanup after a bridge failure.
			}
			output = null;
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

	private record BinaryFrame(int kind, byte[] payload) { }
}
