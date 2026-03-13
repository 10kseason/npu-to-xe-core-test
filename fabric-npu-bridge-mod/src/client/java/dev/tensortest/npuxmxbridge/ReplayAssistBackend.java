package dev.tensortest.npuxmxbridge;

import java.io.IOException;
import java.io.InputStream;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import com.google.gson.Gson;

final class ReplayAssistBackend implements AssistTranslatorBackend {
	private static final Gson GSON = new Gson();
	private static final String DEFAULT_RESOURCE_PATH = "/assets/npu_xmx_bridge/translator/default_replay_fixture.json";

	private final Path fixturePath;
	private final Object lock = new Object();
	private final Map<String, AssistSessionSpec> sessions = new HashMap<>();
	private int nextSessionId = 1;
	private volatile FixtureCatalog fixtureCatalog;

	ReplayAssistBackend() {
		this.fixturePath = null;
	}

	ReplayAssistBackend(Path fixturePath) {
		this.fixturePath = fixturePath;
	}

	@Override
	public String transportSuffix() {
		return "replay";
	}

	@Override
	public String compile(AssistSessionSpec spec) {
		synchronized (lock) {
			String sessionId = String.format(Locale.ROOT, "replay-%08x", nextSessionId++);
			sessions.put(sessionId, spec);
			return sessionId;
		}
	}

	@Override
	public AssistFrame run(String sessionId, AssistSceneState sceneState) {
		AssistSessionSpec spec;
		synchronized (lock) {
			spec = sessions.get(sessionId);
		}
		if (spec == null) {
			throw new IllegalArgumentException("Unknown translator session: " + sessionId);
		}

		long startedAt = System.nanoTime();
		FixtureEntry entry = resolveCatalog().find(spec, sceneState);
		AssistFrame storedFrame = entry.frame();
		return new AssistFrame(
			storedFrame.width(),
			storedFrame.height(),
			storedFrame.pixelsAbgr().clone(),
			storedFrame.meanAlpha(),
			storedFrame.meanLuma(),
			storedFrame.profile(),
			"replay/" + storedFrame.backend(),
			(System.nanoTime() - startedAt) / 1_000_000.0
		);
	}

	@Override
	public boolean release(String sessionId) {
		synchronized (lock) {
			return sessions.remove(sessionId) != null;
		}
	}

	private FixtureCatalog resolveCatalog() {
		FixtureCatalog cached = fixtureCatalog;
		if (cached != null) {
			return cached;
		}
		synchronized (lock) {
			if (fixtureCatalog == null) {
				fixtureCatalog = loadCatalog();
			}
			return fixtureCatalog;
		}
	}

	private FixtureCatalog loadCatalog() {
		if (fixturePath != null) {
			try (Reader reader = Files.newBufferedReader(fixturePath, StandardCharsets.UTF_8)) {
				return parseCatalog(reader, fixturePath.toString());
			} catch (IOException exception) {
				throw new RuntimeException("Unable to load replay fixture from " + fixturePath, exception);
			}
		}

		try (InputStream stream = ReplayAssistBackend.class.getResourceAsStream(DEFAULT_RESOURCE_PATH)) {
			if (stream == null) {
				throw new RuntimeException("Bundled replay fixture resource not found: " + DEFAULT_RESOURCE_PATH);
			}
			return parseCatalog(new java.io.InputStreamReader(stream, StandardCharsets.UTF_8), DEFAULT_RESOURCE_PATH);
		} catch (IOException exception) {
			throw new RuntimeException("Unable to load bundled replay fixture", exception);
		}
	}

	private static FixtureCatalog parseCatalog(Reader reader, String source) throws IOException {
		try (reader) {
			FixtureFile file = GSON.fromJson(reader, FixtureFile.class);
			if (file == null || file.entries == null || file.entries.isEmpty()) {
				throw new RuntimeException("Replay fixture has no entries: " + source);
			}
			List<FixtureEntry> entries = new ArrayList<>(file.entries.size());
			for (FixtureFileEntry entry : file.entries) {
				if (entry.request == null || entry.response == null) {
					continue;
				}
				int[] pixels = entry.response.pixels_abgr == null ? new int[0] : entry.response.pixels_abgr;
				entries.add(new FixtureEntry(
					entry.profile,
					entry.width,
					entry.height,
					new AssistSceneState(
						entry.request.pos_x,
						entry.request.pos_y,
						entry.request.pos_z,
						entry.request.yaw_degrees,
						entry.request.pitch_degrees,
						entry.request.time_seconds,
						entry.request.sun_height,
						entry.request.rain_strength,
						entry.request.thunder_strength,
						entry.request.block_light,
						entry.request.sky_light,
						entry.request.submerged_factor,
						entry.request.quality_budget,
						entry.request.optimization_pressure
					),
					new AssistFrame(
						entry.response.width,
						entry.response.height,
						pixels,
						entry.response.mean_alpha,
						entry.response.mean_luma,
						entry.response.profile,
						entry.response.backend,
						0.0
					)
				));
			}
			if (entries.isEmpty()) {
				throw new RuntimeException("Replay fixture has no usable entries: " + source);
			}
			return new FixtureCatalog(entries, source);
		}
	}

	private record FixtureCatalog(List<FixtureEntry> entries, String source) {
		FixtureEntry find(AssistSessionSpec spec, AssistSceneState sceneState) {
			FixtureEntry best = null;
			double bestDistance = Double.POSITIVE_INFINITY;
			for (FixtureEntry entry : entries) {
				if (!entry.matches(spec)) {
					continue;
				}
				double distance = entry.scene().squaredDistanceTo(sceneState);
				if (distance < bestDistance) {
					bestDistance = distance;
					best = entry;
				}
			}
			if (best == null) {
				throw new IllegalArgumentException(
					"No replay fixture entry for profile=%s size=%dx%d in %s".formatted(
						spec.profile(),
						spec.width(),
						spec.height(),
						source
					)
				);
			}
			return best;
		}
	}

	private record FixtureEntry(
		String profile,
		int width,
		int height,
		AssistSceneState scene,
		AssistFrame frame
	) {
		boolean matches(AssistSessionSpec spec) {
			return profile.equals(spec.profile()) && width == spec.width() && height == spec.height();
		}
	}

	private static final class FixtureFile {
		int version;
		List<FixtureFileEntry> entries;
	}

	private static final class FixtureFileEntry {
		String profile;
		int width;
		int height;
		FixtureRequest request;
		FixtureResponse response;
	}

	private static final class FixtureRequest {
		double pos_x;
		double pos_y;
		double pos_z;
		double yaw_degrees;
		double pitch_degrees;
		double time_seconds;
		double sun_height;
		double rain_strength;
		double thunder_strength;
		double block_light;
		double sky_light;
		double submerged_factor;
		double quality_budget;
		double optimization_pressure;
	}

	private static final class FixtureResponse {
		int width;
		int height;
		int[] pixels_abgr;
		double mean_alpha;
		double mean_luma;
		String profile;
		String backend;
	}
}
