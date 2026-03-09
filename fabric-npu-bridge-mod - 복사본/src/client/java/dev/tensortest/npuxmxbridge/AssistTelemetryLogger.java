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
				writer.write(
					"wall_time_utc,event,enabled,transport,elapsed_ms,frame_cpu_ms,frame_gpu_ms,upload_cpu_ms,upload_gpu_ms,fps,width,height,mean_alpha,mean_luma,sequence,time_seconds,pos_x,pos_y,pos_z,yaw,pitch,shader_ready,in_flight,assist_state,assist_age_frames,assist_updated_this_frame,warmup_remaining,mode,message"
				);
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
		RenderTelemetrySnapshot metrics
	) {
		writeRow(
			"toggle",
			realtimeEnabled,
			transport,
			"",
			formatOptional(metrics.frameCpuMs()),
			formatOptional(metrics.frameGpuMs()),
			formatOptional(metrics.uploadCpuMs()),
			formatOptional(metrics.uploadGpuMs()),
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
			String.format(Locale.ROOT, "interval_ms=%d tile=%d", intervalMs, tileSize)
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
		RenderTelemetrySnapshot metrics
	) {
		writeRow(
			"frame",
			realtimeEnabled,
			transport,
			formatDecimal(result.elapsedMs()),
			formatOptional(metrics.frameCpuMs()),
			formatOptional(metrics.frameGpuMs()),
			formatOptional(metrics.uploadCpuMs()),
			formatOptional(metrics.uploadGpuMs()),
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
			result.statusLine()
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
		RenderTelemetrySnapshot metrics
	) {
		writeRow(
			"heartbeat",
			realtimeEnabled,
			transport,
			"",
			formatOptional(metrics.frameCpuMs()),
			formatOptional(metrics.frameGpuMs()),
			formatOptional(metrics.uploadCpuMs()),
			formatOptional(metrics.uploadGpuMs()),
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
		RenderTelemetrySnapshot metrics
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
			formatOptional(metrics.frameCpuMs()),
			formatOptional(metrics.frameGpuMs()),
			formatOptional(metrics.uploadCpuMs()),
			formatOptional(metrics.uploadGpuMs()),
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
			message
		);
	}

	private void writeRow(
		String event,
		boolean realtimeEnabled,
		String transport,
		String elapsedMs,
		String frameCpuMs,
		String frameGpuMs,
		String uploadCpuMs,
		String uploadGpuMs,
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
		String message
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
					csv(frameCpuMs),
					csv(frameGpuMs),
					csv(uploadCpuMs),
					csv(uploadGpuMs),
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
					csv(message)
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

	private static String modeName(boolean useVanillaPostEffect) {
		return useVanillaPostEffect ? "vanilla_post_effect" : "iris_shaderpack";
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
