package dev.tensortest.npuxmxbridge;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.Locale;

import net.minecraft.client.Minecraft;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

final class AssistTelemetryLogger implements AutoCloseable {
	private static final Logger LOGGER = LoggerFactory.getLogger("npu_xmx_bridge.telemetry");
	static final String CSV_HEADER =
		"wall_time_utc,event,enabled,transport,elapsed_ms,frame_cpu_ms,frame_gpu_ms,upload_cpu_ms,upload_gpu_ms,preview_cpu_ms,preview_gpu_ms,frame_samples,fps_avg_window,fps_1pct_low,fps_0_1pct_low,frame_cpu_avg_ms,frame_cpu_1pct_worst_ms,frame_cpu_0_1pct_worst_ms,frame_gpu_avg_ms,frame_gpu_1pct_worst_ms,frame_gpu_0_1pct_worst_ms,fps,width,height,mean_alpha,mean_luma,sequence,time_seconds,pos_x,pos_y,pos_z,yaw,pitch,shader_ready,in_flight,assist_state,assist_age_frames,assist_updated_this_frame,warmup_remaining,mode,shader_profile,shader_backend,message,sun_height,rain_strength,thunder_strength,block_light,sky_light,submerged_factor,quality_budget,optimization_pressure";

	private final boolean enabled;
	private final Path logPath;
	private final Object lock = new Object();
	private BufferedWriter writer;

	AssistTelemetryLogger(File gameDirectory) {
		this.enabled = Boolean.parseBoolean(System.getProperty("npuxmxbridge.logEnabled", "true"));

		if (!enabled) {
			this.logPath = null;
			return;
		}

		String override = System.getProperty("npuxmxbridge.logPath", "").trim();
		if (!override.isEmpty()) {
			this.logPath = Path.of(override);
		} else {
			this.logPath = gameDirectory.toPath().resolve("logs").resolve("npu-xmx-assist.csv");
		}

		try {
			Files.createDirectories(logPath.getParent());
			boolean fileExists = Files.exists(logPath);
			writer = Files.newBufferedWriter(
				logPath,
				StandardCharsets.UTF_8,
				StandardOpenOption.CREATE,
				StandardOpenOption.APPEND
			);
			if (!fileExists || Files.size(logPath) == 0L) {
				writer.write(CSV_HEADER);
				writer.newLine();
				writer.flush();
			}
			Runtime.getRuntime().addShutdownHook(new Thread(this::closeQuietly, "npu-xmx-assist-log-close"));
		} catch (IOException exception) {
			throw new RuntimeException("Unable to initialize telemetry log at " + logPath, exception);
		}
	}

	boolean isEnabled() {
		return enabled;
	}

	Path logPath() {
		return logPath;
	}

	void logToggle(
		Minecraft client,
		boolean realtimeEnabled,
		String transport,
		long intervalMs,
		String assistState,
		long assistAgeFrames,
		boolean assistUpdatedThisFrame,
		int warmupRemaining,
		boolean useVanillaPostEffect,
		int tileSize,
		RenderTelemetrySnapshot metrics,
		FrameStatsSnapshot frameStats,
		String shaderProfile,
		String shaderBackend
	) {
		writeRow(
			"toggle",
			realtimeEnabled,
			transport,
			"",
			metrics,
			frameStats,
			Integer.toString(client.getFps()),
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			Boolean.toString(false),
			Boolean.toString(false),
			assistState,
			formatOptional(assistAgeFrames),
			Boolean.toString(assistUpdatedThisFrame),
			Integer.toString(warmupRemaining),
			modeName(useVanillaPostEffect),
			shaderProfile,
			shaderBackend,
			String.format(Locale.ROOT, "interval_ms=%d tile=%d", intervalMs, tileSize),
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			""
		);
	}

	void logFrame(
		Minecraft client,
		boolean realtimeEnabled,
		String transport,
		boolean shaderFrameReady,
		boolean requestInFlight,
		String assistState,
		long assistAgeFrames,
		boolean assistUpdatedThisFrame,
		int warmupRemaining,
		long sequence,
		ShaderFrameRequest request,
		ShaderFrameResult result,
		boolean useVanillaPostEffect,
		RenderTelemetrySnapshot metrics,
		FrameStatsSnapshot frameStats
	) {
		writeRow(
			"frame",
			realtimeEnabled,
			transport,
			formatDecimal(result.elapsedMs()),
			metrics,
			frameStats,
			Integer.toString(client.getFps()),
			Integer.toString(result.width()),
			Integer.toString(result.height()),
			formatDecimal(result.meanAlpha()),
			formatDecimal(result.meanLuma()),
			Long.toString(sequence),
			formatDecimal(request.timeSeconds()),
			formatDecimal(request.sample().x()),
			formatDecimal(request.sample().y()),
			formatDecimal(request.sample().z()),
			formatDecimal(request.sample().yaw()),
			formatDecimal(request.sample().pitch()),
			Boolean.toString(shaderFrameReady),
			Boolean.toString(requestInFlight),
			assistState,
			formatOptional(assistAgeFrames),
			Boolean.toString(assistUpdatedThisFrame),
			Integer.toString(warmupRemaining),
			modeName(useVanillaPostEffect),
			result.profile(),
			result.backend(),
			result.statusLine(),
			formatDecimal(request.sample().sunHeight()),
			formatDecimal(request.sample().rainStrength()),
			formatDecimal(request.sample().thunderStrength()),
			formatDecimal(request.sample().blockLight()),
			formatDecimal(request.sample().skyLight()),
			formatDecimal(request.sample().submergedFactor()),
			formatDecimal(request.sample().qualityBudget()),
			formatDecimal(request.sample().optimizationPressure())
		);
	}

	void logHeartbeat(
		Minecraft client,
		boolean realtimeEnabled,
		String transport,
		boolean shaderFrameReady,
		boolean requestInFlight,
		String assistState,
		long assistAgeFrames,
		boolean assistUpdatedThisFrame,
		int warmupRemaining,
		boolean useVanillaPostEffect,
		RenderTelemetrySnapshot metrics,
		FrameStatsSnapshot frameStats,
		String shaderProfile,
		String shaderBackend
	) {
		writeRow(
			"heartbeat",
			realtimeEnabled,
			transport,
			"",
			metrics,
			frameStats,
			Integer.toString(client.getFps()),
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			Boolean.toString(shaderFrameReady),
			Boolean.toString(requestInFlight),
			assistState,
			formatOptional(assistAgeFrames),
			Boolean.toString(assistUpdatedThisFrame),
			Integer.toString(warmupRemaining),
			modeName(useVanillaPostEffect),
			shaderProfile,
			shaderBackend,
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			""
		);
	}

	void logError(
		Minecraft client,
		boolean realtimeEnabled,
		String transport,
		boolean shaderFrameReady,
		boolean requestInFlight,
		String assistState,
		long assistAgeFrames,
		boolean assistUpdatedThisFrame,
		int warmupRemaining,
		long sequence,
		ShaderFrameRequest request,
		Throwable error,
		boolean useVanillaPostEffect,
		RenderTelemetrySnapshot metrics,
		FrameStatsSnapshot frameStats,
		String shaderProfile,
		String shaderBackend
	) {
		String message = error.getClass().getSimpleName();
		if (error.getMessage() != null && !error.getMessage().isBlank()) {
			message += ": " + error.getMessage();
		}
		writeRow(
			"error",
			realtimeEnabled,
			transport,
			"",
			metrics,
			frameStats,
			Integer.toString(client.getFps()),
			"",
			"",
			"",
			"",
			Long.toString(sequence),
			formatDecimal(request.timeSeconds()),
			formatDecimal(request.sample().x()),
			formatDecimal(request.sample().y()),
			formatDecimal(request.sample().z()),
			formatDecimal(request.sample().yaw()),
			formatDecimal(request.sample().pitch()),
			Boolean.toString(shaderFrameReady),
			Boolean.toString(requestInFlight),
			assistState,
			formatOptional(assistAgeFrames),
			Boolean.toString(assistUpdatedThisFrame),
			Integer.toString(warmupRemaining),
			modeName(useVanillaPostEffect),
			shaderProfile,
			shaderBackend,
			message,
			formatDecimal(request.sample().sunHeight()),
			formatDecimal(request.sample().rainStrength()),
			formatDecimal(request.sample().thunderStrength()),
			formatDecimal(request.sample().blockLight()),
			formatDecimal(request.sample().skyLight()),
			formatDecimal(request.sample().submergedFactor()),
			formatDecimal(request.sample().qualityBudget()),
			formatDecimal(request.sample().optimizationPressure())
		);
	}

	private void writeRow(
		String event,
		boolean realtimeEnabled,
		String transport,
		String elapsedMs,
		RenderTelemetrySnapshot metrics,
		FrameStatsSnapshot frameStats,
		String fps,
		String width,
		String height,
		String meanAlpha,
		String meanLuma,
		String sequence,
		String timeSeconds,
		String posX,
		String posY,
		String posZ,
		String yaw,
		String pitch,
		String shaderReady,
		String requestInFlight,
		String assistState,
		String assistAgeFrames,
		String assistUpdatedThisFrame,
		String warmupRemaining,
		String mode,
		String shaderProfile,
		String shaderBackend,
		String message,
		String sunHeight,
		String rainStrength,
		String thunderStrength,
		String blockLight,
		String skyLight,
		String submergedFactor,
		String qualityBudget,
		String optimizationPressure
	) {
		if (!enabled || writer == null) {
			return;
		}

		synchronized (lock) {
			try {
				writer.write(String.join(
					",",
					csv(Instant.now().toString()),
					csv(event),
					csv(Boolean.toString(realtimeEnabled)),
					csv(transport),
					csv(elapsedMs),
					csv(formatOptional(metrics.frameCpuMs())),
					csv(formatOptional(metrics.frameGpuMs())),
					csv(formatOptional(metrics.uploadCpuMs())),
					csv(formatOptional(metrics.uploadGpuMs())),
					csv(formatOptional(metrics.previewCpuMs())),
					csv(formatOptional(metrics.previewGpuMs())),
					csv(formatOptional(frameStats.frameSamples())),
					csv(formatOptional(frameStats.fpsAvg())),
					csv(formatOptional(frameStats.fps1PctLow())),
					csv(formatOptional(frameStats.fps0_1PctLow())),
					csv(formatOptional(frameStats.frameCpuAvgMs())),
					csv(formatOptional(frameStats.frameCpu1PctWorstMs())),
					csv(formatOptional(frameStats.frameCpu0_1PctWorstMs())),
					csv(formatOptional(frameStats.frameGpuAvgMs())),
					csv(formatOptional(frameStats.frameGpu1PctWorstMs())),
					csv(formatOptional(frameStats.frameGpu0_1PctWorstMs())),
					csv(fps),
					csv(width),
					csv(height),
					csv(meanAlpha),
					csv(meanLuma),
					csv(sequence),
					csv(timeSeconds),
					csv(posX),
					csv(posY),
					csv(posZ),
					csv(yaw),
					csv(pitch),
					csv(shaderReady),
					csv(requestInFlight),
					csv(assistState),
					csv(assistAgeFrames),
					csv(assistUpdatedThisFrame),
					csv(warmupRemaining),
					csv(mode),
					csv(shaderProfile),
					csv(shaderBackend),
					csv(message),
					csv(sunHeight),
					csv(rainStrength),
					csv(thunderStrength),
					csv(blockLight),
					csv(skyLight),
					csv(submergedFactor),
					csv(qualityBudget),
					csv(optimizationPressure)
				));
				writer.newLine();
				writer.flush();
			} catch (IOException exception) {
				LOGGER.warn("Unable to append telemetry to {}", logPath, exception);
			}
		}
	}

	private static String csv(String value) {
		if (value == null) {
			return "\"\"";
		}
		return "\"" + value.replace("\"", "\"\"") + "\"";
	}

	private static String formatDecimal(double value) {
		return String.format(Locale.ROOT, "%.4f", value);
	}

	private static String formatOptional(double value) {
		if (Double.isNaN(value) || Double.isInfinite(value)) {
			return "";
		}
		return formatDecimal(value);
	}

	private static String formatOptional(long value) {
		if (value < 0L) {
			return "";
		}
		return Long.toString(value);
	}

	private static String formatOptional(int value) {
		if (value <= 0) {
			return "";
		}
		return Integer.toString(value);
	}

	private static String modeName(boolean useVanillaPostEffect) {
		return useVanillaPostEffect ? "vanilla_post_effect" : "iris_shaderpack";
	}

	void logSyntheticRow(String transport, String message) {
		writeRow(
			"frame",
			true,
			transport,
			"1.2500",
			RenderTelemetrySnapshot.empty(),
			FrameStatsSnapshot.empty(),
			"60",
			"4",
			"4",
			"0.5000",
			"0.5000",
			"1",
			"1.0000",
			"0.0000",
			"64.0000",
			"0.0000",
			"0.0000",
			"0.0000",
			"true",
			"false",
			"active",
			"0",
			"true",
			"0",
			"iris_shaderpack",
			"intel_npu_gi_v3",
			"translator_fixture",
			message,
			"0.5000",
			"0.0000",
			"0.0000",
			"0.0000",
			"1.0000",
			"0.0000",
			"1.0000",
			"0.0000"
		);
	}

	private void closeQuietly() {
		try {
			close();
		} catch (IOException exception) {
			LOGGER.debug("Ignoring telemetry log close failure", exception);
		}
	}

	@Override
	public void close() throws IOException {
		synchronized (lock) {
			if (writer != null) {
				writer.flush();
				writer.close();
				writer = null;
			}
		}
	}
}
