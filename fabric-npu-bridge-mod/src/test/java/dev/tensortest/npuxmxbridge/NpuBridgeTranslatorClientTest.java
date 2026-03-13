package dev.tensortest.npuxmxbridge;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NpuBridgeTranslatorClientTest {
	@Test
	void translatorClientReusesAndReleasesSessions() {
		CountingBackend backend = new CountingBackend();
		NpuBridgeTranslatorClient client = new NpuBridgeTranslatorClient(
			"CPU",
			"intel_npu_gi_v3",
			"THROUGHPUT",
			true,
			backend
		);

		client.runShaderFrame(sampleRequest(4, 4));
		client.runShaderFrame(sampleRequest(4, 4));

		assertEquals("translator-java/counting", client.transportName());
		assertEquals(1, backend.compileCalls);
		assertEquals(2, backend.runCalls);
		assertEquals(0, backend.releaseCalls);

		client.runShaderFrame(sampleRequest(8, 8));

		assertEquals(2, backend.compileCalls);
		assertEquals(3, backend.runCalls);
		assertEquals(1, backend.releaseCalls);
	}

	@Test
	void proceduralBackendIsDeterministic() {
		ProceduralAssistBackend backend = new ProceduralAssistBackend();
		String sessionId = backend.compile(new AssistSessionSpec(4, 4, "CPU", "intel_npu_shader2_v1", "THROUGHPUT", true));
		AssistSceneState state = new AssistSceneState(12.0, 80.0, -6.0, 45.0, 8.0, 1.25, 0.9, 0.1, 0.0, 0.6, 0.95, 0.0, 0.7, 0.2);

		AssistFrame first = backend.run(sessionId, state);
		AssistFrame second = backend.run(sessionId, state);

		assertArrayEquals(first.pixelsAbgr(), second.pixelsAbgr());
		assertEquals(first.meanAlpha(), second.meanAlpha(), 1e-5);
		assertEquals(first.meanLuma(), second.meanLuma(), 1e-5);
		assertTrue(backend.release(sessionId));
	}

	@Test
	void proceduralBackendSmoothsSmallSceneChanges() {
		ProceduralAssistBackend backend = new ProceduralAssistBackend();
		String sessionId = backend.compile(new AssistSessionSpec(4, 4, "CPU", "intel_npu_shader2_v1", "THROUGHPUT", true));
		AssistSceneState baseline = new AssistSceneState(12.0, 80.0, -6.0, 45.0, 8.0, 1.25, 0.9, 0.1, 0.0, 0.6, 0.95, 0.0, 0.7, 0.2);
		AssistSceneState slightDrift = new AssistSceneState(12.2, 80.0, -5.8, 47.0, 8.5, 1.33, 0.9, 0.1, 0.0, 0.6, 0.95, 0.0, 0.7, 0.2);
		AssistSceneState largerChange = new AssistSceneState(18.0, 72.0, -2.0, 92.0, -6.0, 3.5, 0.45, 0.6, 0.2, 0.2, 0.35, 1.0, 0.4, 0.7);

		AssistFrame first = backend.run(sessionId, baseline);
		AssistFrame smallDelta = backend.run(sessionId, slightDrift);
		AssistFrame bigDelta = backend.run(sessionId, largerChange);

		assertTrue(meanPixelDelta(first.pixelsAbgr(), smallDelta.pixelsAbgr()) < meanPixelDelta(first.pixelsAbgr(), bigDelta.pixelsAbgr()));
		assertTrue(backend.release(sessionId));
	}

	@Test
	void replayBackendReproducesFixtureExactly() throws IOException {
		Path fixturePath = fixturePath();
		JsonObject root = JsonParser.parseString(Files.readString(fixturePath, StandardCharsets.UTF_8)).getAsJsonObject();
		JsonObject entry = firstEntry(root.getAsJsonArray("entries"), "intel_npu_gi_v3", 4);
		JsonObject request = entry.getAsJsonObject("request");
		JsonObject response = entry.getAsJsonObject("response");

		ReplayAssistBackend backend = new ReplayAssistBackend(fixturePath);
		String sessionId = backend.compile(new AssistSessionSpec(4, 4, "CPU", "intel_npu_gi_v3", "THROUGHPUT", true));
		AssistFrame frame = backend.run(
			sessionId,
			new AssistSceneState(
				request.get("pos_x").getAsDouble(),
				request.get("pos_y").getAsDouble(),
				request.get("pos_z").getAsDouble(),
				request.get("yaw_degrees").getAsDouble(),
				request.get("pitch_degrees").getAsDouble(),
				request.get("time_seconds").getAsDouble(),
				request.get("sun_height").getAsDouble(),
				request.get("rain_strength").getAsDouble(),
				request.get("thunder_strength").getAsDouble(),
				request.get("block_light").getAsDouble(),
				request.get("sky_light").getAsDouble(),
				request.get("submerged_factor").getAsDouble(),
				request.get("quality_budget").getAsDouble(),
				request.get("optimization_pressure").getAsDouble()
			)
		);

		assertEquals(response.get("width").getAsInt(), frame.width());
		assertEquals(response.get("height").getAsInt(), frame.height());
		assertEquals(response.get("mean_alpha").getAsDouble(), frame.meanAlpha(), 1e-6);
		assertEquals(response.get("mean_luma").getAsDouble(), frame.meanLuma(), 1e-6);
		assertEquals("replay/" + response.get("backend").getAsString(), frame.backend());
		assertArrayEquals(readPixels(response.getAsJsonArray("pixels_abgr")), frame.pixelsAbgr());
	}

	@Test
	void translatorFactoryCreatesTranslatorTransport() {
		NpuBridgeClient client = NpuBridgeClientFactory.create("translator");

		assertInstanceOf(NpuBridgeTranslatorClient.class, client);
		assertFalse(client instanceof NpuBridgeSocketClient);
		assertFalse(client instanceof NpuBridgeHttpClient);
		assertTrue(client.transportName().startsWith("translator-java/"));
	}

	private static ShaderFrameRequest sampleRequest(int width, int height) {
		return new ShaderFrameRequest(
			new PlayerSample(4.0, 72.0, -8.0, 18.0, 6.0, 0.8, 0.1, 0.0, 0.5, 0.9, 0.0, 0.7, 0.2),
			0.8,
			width,
			height
		);
	}

	private static Path fixturePath() {
		Path local = Path.of("src", "main", "resources", "assets", "npu_xmx_bridge", "translator", "default_replay_fixture.json");
		if (Files.exists(local)) {
			return local;
		}
		return Path.of("fabric-npu-bridge-mod", "src", "main", "resources", "assets", "npu_xmx_bridge", "translator", "default_replay_fixture.json");
	}

	private static JsonObject firstEntry(JsonArray entries, String profile, int width) {
		for (int index = 0; index < entries.size(); index++) {
			JsonObject entry = entries.get(index).getAsJsonObject();
			if (profile.equals(entry.get("profile").getAsString()) && entry.get("width").getAsInt() == width) {
				return entry;
			}
		}
		throw new IllegalArgumentException("Fixture entry not found for profile=" + profile + " width=" + width);
	}

	private static int[] readPixels(JsonArray array) {
		int[] pixels = new int[array.size()];
		for (int index = 0; index < array.size(); index++) {
			pixels[index] = array.get(index).getAsInt();
		}
		return pixels;
	}

	private static double meanPixelDelta(int[] left, int[] right) {
		long delta = 0L;
		for (int index = 0; index < left.length; index++) {
			delta += Math.abs(left[index] - right[index]);
		}
		return (double) delta / (double) left.length;
	}

	private static final class CountingBackend implements AssistTranslatorBackend {
		private final Map<String, AssistSessionSpec> sessions = new HashMap<>();
		private int compileCalls;
		private int runCalls;
		private int releaseCalls;
		private int nextSessionId = 1;

		@Override
		public String transportSuffix() {
			return "counting";
		}

		@Override
		public String compile(AssistSessionSpec spec) {
			compileCalls++;
			String sessionId = "counting-" + nextSessionId++;
			sessions.put(sessionId, spec);
			return sessionId;
		}

		@Override
		public AssistFrame run(String sessionId, AssistSceneState sceneState) {
			runCalls++;
			AssistSessionSpec spec = sessions.get(sessionId);
			int[] pixels = new int[spec.width() * spec.height()];
			Arrays.fill(pixels, 0xCC887766);
			return new AssistFrame(spec.width(), spec.height(), pixels, 0.8, 0.55, spec.profile(), "counting_backend", 0.2);
		}

		@Override
		public boolean release(String sessionId) {
			releaseCalls++;
			return sessions.remove(sessionId) != null;
		}
	}
}
