package dev.tensortest.npuxmxbridge;

import java.util.ArrayDeque;

import org.lwjgl.opengl.GL;
import org.lwjgl.opengl.GL15C;
import org.lwjgl.opengl.GL33C;
import org.lwjgl.opengl.GLCapabilities;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

final class GpuTimerQueries {
	private static final Logger LOGGER = LoggerFactory.getLogger("npu_xmx_bridge.gpu_timers");

	private final boolean enabled = Boolean.parseBoolean(System.getProperty("npuxmxbridge.gpuTimersEnabled", "true"));
	private final ArrayDeque<QuerySpan> pendingFrameSpans = new ArrayDeque<>();
	private final ArrayDeque<QuerySpan> pendingUploadSpans = new ArrayDeque<>();
	private final ArrayDeque<QuerySpan> pendingPreviewSpans = new ArrayDeque<>();

	private boolean capabilityChecked;
	private boolean available;
	private boolean frameSpanOpen;
	private int frameStartQueryId;
	private double lastFrameCpuMs = Double.NaN;
	private double lastFrameGpuMs = Double.NaN;
	private double lastUploadCpuMs = Double.NaN;
	private double lastUploadGpuMs = Double.NaN;
	private double lastPreviewCpuMs = Double.NaN;
	private double lastPreviewGpuMs = Double.NaN;
	private long lastHudSampleAtNanos;

	boolean isEnabled() {
		return enabled;
	}

	void onWorldRenderStart() {
		if (!ensureAvailable()) {
			return;
		}

		poll();
		if (frameSpanOpen) {
			return;
		}

		frameStartQueryId = GL15C.glGenQueries();
		GL33C.glQueryCounter(frameStartQueryId, GL33C.GL_TIMESTAMP);
		frameSpanOpen = true;
	}

	void onHudRender() {
		noteFrameCpuTime();
		if (!ensureAvailable()) {
			return;
		}

		if (frameSpanOpen) {
			int endQueryId = GL15C.glGenQueries();
			GL33C.glQueryCounter(endQueryId, GL33C.GL_TIMESTAMP);
			pendingFrameSpans.addLast(new QuerySpan(frameStartQueryId, endQueryId));
			frameSpanOpen = false;
			frameStartQueryId = 0;
		}

		poll();
	}

	void measureUpload(Runnable uploadAction) {
		long cpuStart = System.nanoTime();
		if (!ensureAvailable()) {
			uploadAction.run();
			lastUploadCpuMs = nanosToMillis(System.nanoTime() - cpuStart);
			lastUploadGpuMs = Double.NaN;
			return;
		}

		poll();
		int startQueryId = GL15C.glGenQueries();
		GL33C.glQueryCounter(startQueryId, GL33C.GL_TIMESTAMP);
		uploadAction.run();
		lastUploadCpuMs = nanosToMillis(System.nanoTime() - cpuStart);
		int endQueryId = GL15C.glGenQueries();
		GL33C.glQueryCounter(endQueryId, GL33C.GL_TIMESTAMP);
		pendingUploadSpans.addLast(new QuerySpan(startQueryId, endQueryId));
	}

	void clearUploadMetrics() {
		lastUploadCpuMs = Double.NaN;
		lastUploadGpuMs = Double.NaN;
	}

	void clearAssistMetrics() {
		clearUploadMetrics();
		lastPreviewCpuMs = Double.NaN;
		lastPreviewGpuMs = Double.NaN;
	}

	void measurePreviewDraw(Runnable drawAction) {
		long cpuStart = System.nanoTime();
		if (!ensureAvailable()) {
			drawAction.run();
			lastPreviewCpuMs = nanosToMillis(System.nanoTime() - cpuStart);
			lastPreviewGpuMs = Double.NaN;
			return;
		}

		poll();
		int startQueryId = GL15C.glGenQueries();
		GL33C.glQueryCounter(startQueryId, GL33C.GL_TIMESTAMP);
		drawAction.run();
		lastPreviewCpuMs = nanosToMillis(System.nanoTime() - cpuStart);
		int endQueryId = GL15C.glGenQueries();
		GL33C.glQueryCounter(endQueryId, GL33C.GL_TIMESTAMP);
		pendingPreviewSpans.addLast(new QuerySpan(startQueryId, endQueryId));
	}

	RenderTelemetrySnapshot snapshot() {
		return new RenderTelemetrySnapshot(
			lastFrameCpuMs,
			lastFrameGpuMs,
			lastUploadCpuMs,
			lastUploadGpuMs,
			lastPreviewCpuMs,
			lastPreviewGpuMs
		);
	}

	private void noteFrameCpuTime() {
		long now = System.nanoTime();
		if (lastHudSampleAtNanos != 0L) {
			lastFrameCpuMs = nanosToMillis(now - lastHudSampleAtNanos);
		}
		lastHudSampleAtNanos = now;
	}

	private void poll() {
		pollQueue(pendingFrameSpans, QueryKind.FRAME);
		pollQueue(pendingUploadSpans, QueryKind.UPLOAD);
		pollQueue(pendingPreviewSpans, QueryKind.PREVIEW);
	}

	private void pollQueue(ArrayDeque<QuerySpan> queue, QueryKind kind) {
		while (!queue.isEmpty()) {
			QuerySpan span = queue.peekFirst();
			int availableFlag = GL15C.glGetQueryObjecti(span.endQueryId(), GL15C.GL_QUERY_RESULT_AVAILABLE);
			if (availableFlag == 0) {
				return;
			}

			long start = GL33C.glGetQueryObjecti64(span.startQueryId(), GL15C.GL_QUERY_RESULT);
			long end = GL33C.glGetQueryObjecti64(span.endQueryId(), GL15C.GL_QUERY_RESULT);
			double elapsedMs = nanosToMillis(end - start);
			if (kind == QueryKind.FRAME) {
				lastFrameGpuMs = elapsedMs;
			} else if (kind == QueryKind.UPLOAD) {
				lastUploadGpuMs = elapsedMs;
			} else {
				lastPreviewGpuMs = elapsedMs;
			}

			GL15C.glDeleteQueries(span.startQueryId());
			GL15C.glDeleteQueries(span.endQueryId());
			queue.removeFirst();
		}
	}

	private boolean ensureAvailable() {
		if (!enabled) {
			return false;
		}
		if (capabilityChecked) {
			return available;
		}

		capabilityChecked = true;
		try {
			GLCapabilities capabilities = GL.getCapabilities();
			available = capabilities != null && (capabilities.OpenGL33 || capabilities.GL_ARB_timer_query);
		} catch (IllegalStateException exception) {
			available = false;
		}

		if (available) {
			LOGGER.info("GPU timestamp queries enabled for NPU assist telemetry.");
		} else {
			LOGGER.warn("GPU timestamp queries are unavailable; only CPU-side assist timings will be recorded.");
		}
		return available;
	}

	private static double nanosToMillis(long nanos) {
		return nanos / 1_000_000.0;
	}

	private enum QueryKind {
		FRAME,
		UPLOAD,
		PREVIEW
	}

	private record QuerySpan(int startQueryId, int endQueryId) { }
}
