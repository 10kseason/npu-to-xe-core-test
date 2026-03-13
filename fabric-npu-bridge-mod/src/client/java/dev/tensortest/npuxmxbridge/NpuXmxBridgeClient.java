package dev.tensortest.npuxmxbridge;

import com.mojang.blaze3d.platform.NativeImage;
import java.lang.reflect.Method;
import java.util.Locale;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

import com.mojang.blaze3d.platform.InputConstants;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.fabricmc.fabric.api.client.rendering.v1.hud.HudElementRegistry;
import net.fabricmc.fabric.api.client.rendering.v1.world.WorldRenderEvents;
import net.minecraft.ChatFormatting;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.Minecraft;
import net.minecraft.network.chat.Component;
import net.minecraft.client.gui.GuiGraphics;
import net.minecraft.client.renderer.RenderPipelines;
import net.minecraft.client.renderer.texture.DynamicTexture;
import net.minecraft.resources.Identifier;

import org.lwjgl.glfw.GLFW;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class NpuXmxBridgeClient implements ClientModInitializer {
	public static final String MOD_ID = "npu_xmx_bridge";
	public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);
	private static final long UPDATE_INTERVAL_MS = Long.getLong("npuxmxbridge.intervalMs", 75L);
	private static final int UPDATE_EVERY_N_FRAMES = Math.max(1, Integer.getInteger("npuxmxbridge.updateEveryNFrames", 1));
	private static final int WARMUP_REQUESTS = Math.max(0, Integer.getInteger("npuxmxbridge.warmupRequests", 3));
	private static final int MAX_ASSIST_AGE_FRAMES = Math.max(1, Integer.getInteger("npuxmxbridge.maxAssistAgeFrames", 4));
	private static final double MIN_POSITION_DELTA = getDoubleProperty("npuxmxbridge.minPositionDelta", 0.75);
	private static final double MIN_ANGLE_DELTA = getDoubleProperty("npuxmxbridge.minAngleDelta", 3.0);
	private static final double TARGET_FRAME_MS = getDoubleProperty("npuxmxbridge.targetFrameMs", 16.6);
	private static final double TARGET_ASSIST_MS = getDoubleProperty("npuxmxbridge.targetAssistMs", 12.0);
	private static final double BUDGET_RESPONSE = getDoubleProperty("npuxmxbridge.budgetResponse", 0.28);
	private static final double PRESSURE_RESPONSE = getDoubleProperty("npuxmxbridge.pressureResponse", 0.34);
	private static final int SHADER_TILE_SIZE = Integer.getInteger("npuxmxbridge.shaderTileSize", 80);
	private static final int SHADER_PREVIEW_SCALE = Integer.getInteger("npuxmxbridge.shaderPreviewScale", 6);
	private static final boolean SHOW_PREVIEW_HUD = Boolean.getBoolean("npuxmxbridge.showPreviewHud");
	private static final boolean USE_VANILLA_POST_EFFECT = Boolean.getBoolean("npuxmxbridge.useVanillaPostEffect");
	private static final String CONFIGURED_SHADER_PROFILE = System.getProperty("npuxmxbridge.shaderProfile", "intel_npu_gi_v3");
	private static final int NEUTRAL_ASSIST_ABGR = 0x00808080;
	private static final Identifier SHADER_TEXTURE_ID = Identifier.fromNamespaceAndPath(MOD_ID, "shader_preview");
	private static final Identifier HUD_LAYER_ID = Identifier.fromNamespaceAndPath(MOD_ID, "shader_preview_hud");
	private static final Identifier POST_EFFECT_ID = Identifier.fromNamespaceAndPath(MOD_ID, "npu_assist");
	private static final Method SET_POST_EFFECT_METHOD = resolveSetPostEffectMethod();

	private static final KeyMapping RUN_SAMPLE_KEY = KeyBindingHelper.registerKeyBinding(
		new KeyMapping(
			"key.npu_xmx_bridge.run_sample",
			InputConstants.Type.KEYSYM,
			GLFW.GLFW_KEY_N,
			KeyMapping.Category.MISC
		)
	);
	private static final AtomicBoolean REQUEST_IN_FLIGHT = new AtomicBoolean(false);
	private static final NpuBridgeClient BRIDGE_CLIENT = NpuBridgeClientFactory.createFromSystemProperties();
	private static final ExecutorService BRIDGE_EXECUTOR = Executors.newSingleThreadExecutor(runnable -> {
		Thread thread = new Thread(runnable, "npu-xmx-bridge-worker");
		thread.setDaemon(true);
		return thread;
	});
	private static final Object TEXTURE_LOCK = new Object();
	private static volatile long lastDispatchAtNanos;
	private static volatile OverlayStatus overlayStatus = new OverlayStatus("", ChatFormatting.GRAY);
	private static volatile long frameSequence;
	private static volatile DynamicTexture shaderTexture;
	private static volatile int shaderTextureWidth = SHADER_TILE_SIZE * Math.max(1, SHADER_PREVIEW_SCALE);
	private static volatile int shaderTextureHeight = SHADER_TILE_SIZE * Math.max(1, SHADER_PREVIEW_SCALE);
	private static volatile boolean shaderFrameReady;
	private static volatile Identifier previousPostEffectId;
	private static volatile AssistTelemetryLogger telemetryLogger;
	private static volatile boolean telemetryInitFailed;
	private static volatile long lastTelemetryHeartbeatAtNanos;
	private static volatile AssistMode assistMode = AssistMode.OFF;
	private static volatile int warmupRemaining;
	private static volatile long assistGeneration;
	private static volatile long renderFrameIndex;
	private static volatile long lastDispatchRenderFrame = -1L;
	private static volatile long lastAppliedRenderFrame = -1L;
	private static volatile PlayerSample lastRealtimeDispatchSample;
	private static volatile double lastRealtimeDispatchTimeSeconds = Double.NaN;
	private static volatile String lastShaderProfile = CONFIGURED_SHADER_PROFILE;
	private static volatile String lastShaderBackend = "";
	private static volatile double lastBridgeElapsedMs = Double.NaN;
	private static volatile double lastQualityBudget = 1.0;
	private static volatile double lastOptimizationPressure = 0.0;
	private static final GpuTimerQueries GPU_TIMERS = new GpuTimerQueries();
	private static final FrameStatsAccumulator ACTIVE_FRAME_STATS = new FrameStatsAccumulator();
	private static final FrameStatsAccumulator INACTIVE_FRAME_STATS = new FrameStatsAccumulator();

	@Override
	public void onInitializeClient() {
		INACTIVE_FRAME_STATS.reset();
		ACTIVE_FRAME_STATS.reset();
		WorldRenderEvents.START_MAIN.register(context -> GPU_TIMERS.onWorldRenderStart());
		HudElementRegistry.addFirst(Identifier.fromNamespaceAndPath(MOD_ID, "frame_telemetry"), NpuXmxBridgeClient::sampleRenderTelemetry);
		HudElementRegistry.addLast(HUD_LAYER_ID, NpuXmxBridgeClient::renderShaderPreview);
		ClientTickEvents.END_CLIENT_TICK.register(client -> {
			ensureAssistTexture(client);
			tickTelemetry(client);
			while (RUN_SAMPLE_KEY.consumeClick()) {
				toggleRealtime(client);
			}
			tickRealtime(client);
		});

		LOGGER.info(
			"NPU XMX Bridge client initialized with {} transport. Press N to toggle real-time NPU assist feed at {} ms cadence, {}-frame reuse, and {}-frame max assist age.",
			BRIDGE_CLIENT.transportName(),
			UPDATE_INTERVAL_MS,
			UPDATE_EVERY_N_FRAMES,
			MAX_ASSIST_AGE_FRAMES
		);
	}

	private static void toggleRealtime(Minecraft client) {
		if (client.player == null) {
			return;
		}

		if (assistMode == AssistMode.OFF) {
			beginWarmup(client);
			return;
		}

		boolean wasActive = assistMode == AssistMode.ACTIVE;
		disableAssist(client, wasActive ? "NPU shader assist disabled." : "NPU shader assist warm-up canceled.");
	}

	private static void tickTelemetry(Minecraft client) {
		if (client.player == null) {
			return;
		}

		long now = System.nanoTime();
		if (lastTelemetryHeartbeatAtNanos != 0L && now - lastTelemetryHeartbeatAtNanos < 1_000_000_000L) {
			return;
		}
		lastTelemetryHeartbeatAtNanos = now;

		AssistTelemetryLogger logger = telemetryLogger(client);
		if (logger == null || !logger.isEnabled()) {
			return;
		}

		logger.logHeartbeat(
			client,
			assistMode == AssistMode.ACTIVE,
			BRIDGE_CLIENT.transportName(),
			shaderFrameReady,
			REQUEST_IN_FLIGHT.get(),
			assistStateName(),
			currentAssistAgeFrames(),
			assistUpdatedThisFrame(),
			warmupRemaining,
			USE_VANILLA_POST_EFFECT,
			GPU_TIMERS.snapshot(),
			currentFrameStats(),
			lastShaderProfile,
			lastShaderBackend
		);
	}

	private static void tickRealtime(Minecraft client) {
		if (client.player == null) {
			return;
		}

		if (assistMode == AssistMode.OFF) {
			return;
		}

		if (assistMode == AssistMode.ACTIVE && USE_VANILLA_POST_EFFECT) {
			ensureHybridPostEffect(client);
		}
		showOverlayStatus(client, overlayStatus.message(), overlayStatus.color());

		long now = System.nanoTime();
		if (REQUEST_IN_FLIGHT.get()) {
			return;
		}

		if (assistMode == AssistMode.WARMING_UP) {
			runBridgeSample(client, now, DispatchKind.WARMUP, assistGeneration);
			return;
		}

		if (!shouldDispatchRealtime(client, now)) {
			return;
		}

		runBridgeSample(client, now, DispatchKind.REALTIME, assistGeneration);
	}

	private static void runBridgeSample(Minecraft client, long startedAtNanos, DispatchKind dispatchKind, long generation) {
		if (!REQUEST_IN_FLIGHT.compareAndSet(false, true)) {
			return;
		}

		lastDispatchAtNanos = startedAtNanos;
		lastDispatchRenderFrame = renderFrameIndex;
		RealtimeBudgetState budgetState = computeRealtimeBudgetState();
		PlayerSample sample = PlayerSample.capture(client, budgetState.qualityBudget(), budgetState.optimizationPressure());
		double timeSeconds = client.level != null ? client.level.getGameTime() / 20.0 : frameSequence / 20.0;
		if (dispatchKind == DispatchKind.REALTIME) {
			lastRealtimeDispatchSample = sample;
			lastRealtimeDispatchTimeSeconds = timeSeconds;
		}
		ShaderFrameRequest request = new ShaderFrameRequest(sample, timeSeconds, SHADER_TILE_SIZE, SHADER_TILE_SIZE);
		long currentFrame = dispatchKind == DispatchKind.REALTIME ? ++frameSequence : frameSequence;
		if (dispatchKind == DispatchKind.WARMUP) {
			overlayStatus = new OverlayStatus(
				String.format(
					Locale.ROOT,
					"Warming up NPU assist (%d/%d)...",
					Math.max(0, WARMUP_REQUESTS - warmupRemaining + 1),
					Math.max(1, WARMUP_REQUESTS)
				),
				ChatFormatting.GOLD
			);
		} else {
			overlayStatus = new OverlayStatus("NPU shader assist frame in flight...", ChatFormatting.DARK_AQUA);
		}

		CompletableFuture
			.supplyAsync(() -> BRIDGE_CLIENT.runShaderFrame(request), BRIDGE_EXECUTOR)
			.whenComplete((frame, error) -> client.execute(() -> {
				REQUEST_IN_FLIGHT.set(false);
				if (generation != assistGeneration || assistMode == AssistMode.OFF) {
					return;
				}
				if (error != null) {
					Throwable cause = error.getCause() != null ? error.getCause() : error;
					LOGGER.warn("Bridge sample failed", cause);
					disableAssistInternal(client, false);
					logError(client, currentFrame, request, cause);
					overlayStatus = new OverlayStatus("Bridge call failed: " + cause.getMessage(), ChatFormatting.RED);
					showChatStatus(client, overlayStatus.message(), overlayStatus.color());
					return;
				}

				lastBridgeElapsedMs = frame.elapsedMs();
				if (dispatchKind == DispatchKind.WARMUP) {
					handleWarmupCompletion(client, request, frame, generation);
					return;
				}

				applyShaderFrame(client, frame);
				logFrame(client, currentFrame, request, frame);
				overlayStatus = new OverlayStatus(frame.statusLine(), ChatFormatting.AQUA);
			}));
	}

	private static void beginWarmup(Minecraft client) {
		assistGeneration++;
		assistMode = AssistMode.WARMING_UP;
		warmupRemaining = WARMUP_REQUESTS;
		lastDispatchAtNanos = 0L;
		lastDispatchRenderFrame = -1L;
		lastAppliedRenderFrame = -1L;
		lastRealtimeDispatchSample = null;
		lastRealtimeDispatchTimeSeconds = Double.NaN;
		lastBridgeElapsedMs = Double.NaN;
		lastQualityBudget = 1.0;
		lastOptimizationPressure = 0.0;
		frameSequence = 0L;
		shaderFrameReady = false;
		ensureShaderTexture(client, shaderTextureWidth, shaderTextureHeight);
		resetShaderTexture(client);
		overlayStatus = new OverlayStatus(
			WARMUP_REQUESTS > 0
				? String.format(Locale.ROOT, "Warming up NPU assist (%d requests)...", WARMUP_REQUESTS)
				: "Preparing NPU shader assist...",
			ChatFormatting.GOLD
		);
		showChatStatus(client, overlayStatus.message(), overlayStatus.color());
		if (WARMUP_REQUESTS == 0) {
			activateRealtime(client);
		}
	}

	private static void activateRealtime(Minecraft client) {
		assistMode = AssistMode.ACTIVE;
		ACTIVE_FRAME_STATS.reset();
		warmupRemaining = 0;
		lastDispatchAtNanos = 0L;
		lastDispatchRenderFrame = -1L;
		lastRealtimeDispatchSample = null;
		lastRealtimeDispatchTimeSeconds = Double.NaN;
		lastBridgeElapsedMs = Double.NaN;
		lastQualityBudget = 1.0;
		lastOptimizationPressure = 0.0;
		if (USE_VANILLA_POST_EFFECT) {
			enableHybridPostEffect(client);
		}
		overlayStatus = new OverlayStatus(
			String.format(
				Locale.ROOT,
				USE_VANILLA_POST_EFFECT
					? "Vanilla post-effect NPU assist enabled via %s (%d ms, every %d frames)."
					: "Iris shaderpack NPU assist enabled via %s (%d ms, every %d frames).",
				BRIDGE_CLIENT.transportName(),
				UPDATE_INTERVAL_MS,
				UPDATE_EVERY_N_FRAMES
			),
			ChatFormatting.GREEN
		);
		showChatStatus(client, overlayStatus.message(), overlayStatus.color());
		logToggle(client, true);
	}

	private static void handleWarmupCompletion(Minecraft client, ShaderFrameRequest request, ShaderFrameResult frame, long generation) {
		if (generation != assistGeneration || assistMode != AssistMode.WARMING_UP) {
			return;
		}
		lastShaderProfile = frame.profile();
		lastShaderBackend = frame.backend();

		if (warmupRemaining > 0) {
			warmupRemaining--;
		}
		if (warmupRemaining > 0) {
			overlayStatus = new OverlayStatus(
				String.format(
					Locale.ROOT,
					"Warming up NPU assist (%d/%d)...",
					WARMUP_REQUESTS - warmupRemaining,
					Math.max(1, WARMUP_REQUESTS)
				),
				ChatFormatting.GOLD
			);
			return;
		}

		LOGGER.info(
			"NPU assist warm-up complete at {} ms for {}x{} tile; entering decimated realtime mode.",
			String.format(Locale.ROOT, "%.3f", frame.elapsedMs()),
			request.width(),
			request.height()
		);
		activateRealtime(client);
	}

	private static boolean shouldDispatchRealtime(Minecraft client, long now) {
		if (lastDispatchAtNanos != 0L && now - lastDispatchAtNanos < UPDATE_INTERVAL_MS * 1_000_000L) {
			return false;
		}

		long assistAgeFrames = currentAssistAgeFrames();
		if (!shaderFrameReady || assistAgeFrames < 0L) {
			return true;
		}
		if (assistAgeFrames >= MAX_ASSIST_AGE_FRAMES) {
			return true;
		}
		if (lastDispatchRenderFrame >= 0L && renderFrameIndex - lastDispatchRenderFrame < UPDATE_EVERY_N_FRAMES) {
			return false;
		}

		PlayerSample currentSample = PlayerSample.capture(client);
		if (lastRealtimeDispatchSample == null) {
			return true;
		}

		double positionDelta = currentSample.distanceTo(lastRealtimeDispatchSample);
		double angleDelta = currentSample.maxAngleDeltaTo(lastRealtimeDispatchSample);
		if (positionDelta >= MIN_POSITION_DELTA || angleDelta >= MIN_ANGLE_DELTA) {
			return true;
		}

		double timeSeconds = client.level != null ? client.level.getGameTime() / 20.0 : frameSequence / 20.0;
		return Double.isNaN(lastRealtimeDispatchTimeSeconds)
			|| timeSeconds - lastRealtimeDispatchTimeSeconds >= Math.max(0.25, MAX_ASSIST_AGE_FRAMES / 20.0);
	}

	private static void disableAssist(Minecraft client, String message) {
		boolean wasActive = assistMode == AssistMode.ACTIVE;
		disableAssistInternal(client, true);
		overlayStatus = new OverlayStatus(message, wasActive ? ChatFormatting.YELLOW : ChatFormatting.GRAY);
		showChatStatus(client, overlayStatus.message(), overlayStatus.color());
		if (wasActive) {
			logToggle(client, false);
		}
	}

	private static void disableAssistInternal(Minecraft client, boolean resetStatusMessage) {
		assistGeneration++;
		assistMode = AssistMode.OFF;
		INACTIVE_FRAME_STATS.reset();
		warmupRemaining = 0;
		shaderFrameReady = false;
		resetShaderTexture(client);
		GPU_TIMERS.clearAssistMetrics();
		lastDispatchRenderFrame = -1L;
		lastAppliedRenderFrame = -1L;
		lastRealtimeDispatchSample = null;
		lastRealtimeDispatchTimeSeconds = Double.NaN;
		lastBridgeElapsedMs = Double.NaN;
		lastQualityBudget = 1.0;
		lastOptimizationPressure = 0.0;
		if (USE_VANILLA_POST_EFFECT) {
			disableHybridPostEffect(client);
		}
		if (resetStatusMessage) {
			overlayStatus = new OverlayStatus("NPU shader assist disabled.", ChatFormatting.YELLOW);
		}
	}

	private static void showChatStatus(Minecraft client, String message, ChatFormatting color) {
		if (client.player != null) {
			client.player.displayClientMessage(Component.literal(message).withStyle(color), false);
		}
	}

	private static void showOverlayStatus(Minecraft client, String message, ChatFormatting color) {
		if (client.player != null) {
			client.player.displayClientMessage(Component.literal(message).withStyle(color), true);
		}
	}

	private static void applyShaderFrame(Minecraft client, ShaderFrameResult frame) {
		lastShaderProfile = frame.profile();
		lastShaderBackend = frame.backend();
		int previewScale = Math.max(1, SHADER_PREVIEW_SCALE);
		int previewWidth = frame.width() * previewScale;
		int previewHeight = frame.height() * previewScale;
		ensureShaderTexture(client, previewWidth, previewHeight);
		synchronized (TEXTURE_LOCK) {
			if (shaderTexture == null) {
				return;
			}

			NativeImage pixels = shaderTexture.getPixels();
			if (pixels == null) {
				return;
			}

			int index = 0;
			for (int y = 0; y < frame.height(); y++) {
				for (int x = 0; x < frame.width(); x++) {
					int abgr = frame.pixelsAbgr()[index++];
					int baseX = x * previewScale;
					int baseY = y * previewScale;
					for (int offsetY = 0; offsetY < previewScale; offsetY++) {
						for (int offsetX = 0; offsetX < previewScale; offsetX++) {
							pixels.setPixelABGR(baseX + offsetX, baseY + offsetY, abgr);
						}
					}
				}
			}
			GPU_TIMERS.measureUpload(shaderTexture::upload);
			shaderFrameReady = true;
			lastAppliedRenderFrame = renderFrameIndex;
		}
	}

	private static void enableHybridPostEffect(Minecraft client) {
		if (SET_POST_EFFECT_METHOD == null) {
			return;
		}

		Identifier current = client.gameRenderer.currentPostEffect();
		if (POST_EFFECT_ID.equals(current)) {
			return;
		}
		if (current != null) {
			previousPostEffectId = current;
		}
		invokeSetPostEffect(client, POST_EFFECT_ID);
	}

	private static void ensureHybridPostEffect(Minecraft client) {
		if (SET_POST_EFFECT_METHOD == null) {
			return;
		}
		Identifier current = client.gameRenderer.currentPostEffect();
		if (!POST_EFFECT_ID.equals(current)) {
			invokeSetPostEffect(client, POST_EFFECT_ID);
		}
	}

	private static void disableHybridPostEffect(Minecraft client) {
		if (POST_EFFECT_ID.equals(client.gameRenderer.currentPostEffect())) {
			client.gameRenderer.clearPostEffect();
		}

		if (SET_POST_EFFECT_METHOD != null && previousPostEffectId != null && !POST_EFFECT_ID.equals(previousPostEffectId)) {
			invokeSetPostEffect(client, previousPostEffectId);
		}
		previousPostEffectId = null;
	}

	private static void invokeSetPostEffect(Minecraft client, Identifier postEffectId) {
		if (SET_POST_EFFECT_METHOD == null) {
			return;
		}

		try {
			SET_POST_EFFECT_METHOD.invoke(client.gameRenderer, postEffectId);
		} catch (ReflectiveOperationException exception) {
			LOGGER.warn("Unable to activate post effect {}", postEffectId, exception);
		}
	}

	private static Method resolveSetPostEffectMethod() {
		try {
			Method method = net.minecraft.client.renderer.GameRenderer.class.getDeclaredMethod("setPostEffect", Identifier.class);
			method.setAccessible(true);
			return method;
		} catch (ReflectiveOperationException exception) {
			LOGGER.warn("Unable to access GameRenderer.setPostEffect; the optional vanilla post-effect demo path will stay disabled.", exception);
			return null;
		}
	}

	private static void ensureShaderTexture(Minecraft client, int width, int height) {
		synchronized (TEXTURE_LOCK) {
			if (shaderTexture != null && shaderTextureWidth == width && shaderTextureHeight == height) {
				return;
			}

			if (shaderTexture != null) {
				client.getTextureManager().release(SHADER_TEXTURE_ID);
				shaderTexture.close();
			}

			shaderTexture = new DynamicTexture(() -> "npu_xmx_shader_preview", width, height, false);
			shaderTextureWidth = width;
			shaderTextureHeight = height;
			NativeImage pixels = shaderTexture.getPixels();
			if (pixels != null) {
				fillTexturePixels(pixels, width, height, NEUTRAL_ASSIST_ABGR);
				GPU_TIMERS.measureUpload(shaderTexture::upload);
			}
			client.getTextureManager().register(SHADER_TEXTURE_ID, shaderTexture);
		}
	}

	private static void ensureAssistTexture(Minecraft client) {
		ensureShaderTexture(client, shaderTextureWidth, shaderTextureHeight);
	}

	private static void resetShaderTexture(Minecraft client) {
		ensureShaderTexture(client, shaderTextureWidth, shaderTextureHeight);
		synchronized (TEXTURE_LOCK) {
			if (shaderTexture == null) {
				return;
			}

			NativeImage pixels = shaderTexture.getPixels();
			if (pixels == null) {
				return;
			}

			fillTexturePixels(pixels, shaderTextureWidth, shaderTextureHeight, NEUTRAL_ASSIST_ABGR);
			GPU_TIMERS.measureUpload(shaderTexture::upload);
		}
	}

	private static void fillTexturePixels(NativeImage pixels, int width, int height, int abgr) {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				pixels.setPixelABGR(x, y, abgr);
			}
		}
	}

	private static AssistTelemetryLogger telemetryLogger(Minecraft client) {
		if (telemetryInitFailed) {
			return null;
		}
		if (telemetryLogger != null) {
			return telemetryLogger;
		}

		synchronized (TEXTURE_LOCK) {
			if (telemetryLogger != null || telemetryInitFailed) {
				return telemetryLogger;
			}

			try {
				telemetryLogger = new AssistTelemetryLogger(client.gameDirectory);
				if (telemetryLogger.isEnabled()) {
					LOGGER.info("NPU assist telemetry logging to {}", telemetryLogger.logPath());
				}
			} catch (RuntimeException exception) {
				telemetryInitFailed = true;
				LOGGER.warn("Unable to initialize NPU assist telemetry logging", exception);
			}
			return telemetryLogger;
		}
	}

	private static void logToggle(Minecraft client, boolean realtimeEnabled) {
		AssistTelemetryLogger logger = telemetryLogger(client);
		if (logger == null || !logger.isEnabled()) {
			return;
		}

		logger.logToggle(
			client,
			realtimeEnabled,
			BRIDGE_CLIENT.transportName(),
			UPDATE_INTERVAL_MS,
			assistStateName(),
			currentAssistAgeFrames(),
			assistUpdatedThisFrame(),
			warmupRemaining,
			USE_VANILLA_POST_EFFECT,
			SHADER_TILE_SIZE,
			GPU_TIMERS.snapshot(),
			currentFrameStats(),
			lastShaderProfile,
			lastShaderBackend
		);
	}

	private static void logFrame(Minecraft client, long sequence, ShaderFrameRequest request, ShaderFrameResult frame) {
		AssistTelemetryLogger logger = telemetryLogger(client);
		if (logger == null || !logger.isEnabled()) {
			return;
		}

		logger.logFrame(
			client,
			assistMode == AssistMode.ACTIVE,
			BRIDGE_CLIENT.transportName(),
			shaderFrameReady,
			REQUEST_IN_FLIGHT.get(),
			assistStateName(),
			currentAssistAgeFrames(),
			assistUpdatedThisFrame(),
			warmupRemaining,
			sequence,
			request,
			frame,
			USE_VANILLA_POST_EFFECT,
			GPU_TIMERS.snapshot(),
			currentFrameStats()
		);
		GPU_TIMERS.clearAssistMetrics();
	}

	private static void logError(Minecraft client, long sequence, ShaderFrameRequest request, Throwable error) {
		AssistTelemetryLogger logger = telemetryLogger(client);
		if (logger == null || !logger.isEnabled()) {
			return;
		}

		logger.logError(
			client,
			assistMode == AssistMode.ACTIVE,
			BRIDGE_CLIENT.transportName(),
			shaderFrameReady,
			REQUEST_IN_FLIGHT.get(),
			assistStateName(),
			currentAssistAgeFrames(),
			assistUpdatedThisFrame(),
			warmupRemaining,
			sequence,
			request,
			error,
			USE_VANILLA_POST_EFFECT,
			GPU_TIMERS.snapshot(),
			currentFrameStats(),
			lastShaderProfile,
			lastShaderBackend
		);
		GPU_TIMERS.clearAssistMetrics();
	}

	private static void sampleRenderTelemetry(GuiGraphics context, net.minecraft.client.DeltaTracker tickCounter) {
		renderFrameIndex++;
		GPU_TIMERS.onHudRender();
		RenderTelemetrySnapshot snapshot = GPU_TIMERS.snapshot();
		if (assistMode == AssistMode.ACTIVE) {
			ACTIVE_FRAME_STATS.record(snapshot.frameCpuMs(), snapshot.frameGpuMs());
		} else if (assistMode == AssistMode.OFF) {
			INACTIVE_FRAME_STATS.record(snapshot.frameCpuMs(), snapshot.frameGpuMs());
		}
	}

	private static void renderShaderPreview(GuiGraphics context, net.minecraft.client.DeltaTracker tickCounter) {
		if (!SHOW_PREVIEW_HUD || assistMode != AssistMode.ACTIVE || !shaderFrameReady || SHADER_PREVIEW_SCALE <= 0) {
			return;
		}

		int previewWidth = shaderTextureWidth;
		int previewHeight = shaderTextureHeight;
		int x = context.guiWidth() - previewWidth - 12;
		int y = 12;

		GPU_TIMERS.measurePreviewDraw(() -> {
			context.fill(x - 4, y - 4, x + previewWidth + 4, y + previewHeight + 18, 0x8A000000);
			context.blit(
				RenderPipelines.GUI_TEXTURED,
				SHADER_TEXTURE_ID,
				x,
				y,
				0.0F,
				0.0F,
				previewWidth,
				previewHeight,
				previewWidth,
				previewHeight
			);
			context.drawString(Minecraft.getInstance().font, "NPU shader preview", x, y + previewHeight + 4, 0xE8F7FF, false);
		});
	}

	private static String assistStateName() {
		return assistMode.name().toLowerCase(Locale.ROOT);
	}

	private static long currentAssistAgeFrames() {
		if (!shaderFrameReady || lastAppliedRenderFrame < 0L) {
			return -1L;
		}
		return Math.max(0L, renderFrameIndex - lastAppliedRenderFrame);
	}

	private static boolean assistUpdatedThisFrame() {
		return shaderFrameReady && lastAppliedRenderFrame >= 0L && renderFrameIndex == lastAppliedRenderFrame;
	}

	private static FrameStatsSnapshot currentFrameStats() {
		if (assistMode == AssistMode.ACTIVE) {
			return ACTIVE_FRAME_STATS.snapshot();
		}
		if (assistMode == AssistMode.OFF) {
			return INACTIVE_FRAME_STATS.snapshot();
		}
		return FrameStatsSnapshot.empty();
	}

	private static RealtimeBudgetState computeRealtimeBudgetState() {
		RenderTelemetrySnapshot metrics = GPU_TIMERS.snapshot();
		FrameStatsSnapshot frameStats = currentFrameStats();
		long assistAgeFrames = currentAssistAgeFrames();

		double gpuFrameMs = maxFinite(metrics.frameGpuMs(), frameStats.frameGpuAvgMs(), scaleFinite(frameStats.frameGpu1PctWorstMs(), 0.92));
		double cpuFrameMs = maxFinite(metrics.frameCpuMs(), frameStats.frameCpuAvgMs(), scaleFinite(frameStats.frameCpu1PctWorstMs(), 0.92));
		double uploadAndPreviewMs = maxFinite(metrics.uploadCpuMs(), metrics.uploadGpuMs())
			+ maxFinite(metrics.previewCpuMs(), metrics.previewGpuMs()) * 0.5;
		double gpuPressure = pressureForMs(gpuFrameMs, TARGET_FRAME_MS * 0.94);
		double cpuPressure = pressureForMs(cpuFrameMs, TARGET_FRAME_MS);
		double assistPressure = pressureForMs(lastBridgeElapsedMs, TARGET_ASSIST_MS);
		double uploadPressure = pressureForMs(uploadAndPreviewMs, 2.25);
		double stalePressure = assistAgeFrames < 0L
			? 0.0
			: clamp01((assistAgeFrames - 1.0) / Math.max(1.0, MAX_ASSIST_AGE_FRAMES - 1.0));

		double pressureTarget = clamp01(
			gpuPressure * 0.44
				+ cpuPressure * 0.16
				+ assistPressure * 0.24
				+ uploadPressure * 0.10
				+ stalePressure * 0.06
		);
		if (assistMode == AssistMode.WARMING_UP) {
			pressureTarget = Math.max(pressureTarget, 0.10);
		}
		lastOptimizationPressure = blend(lastOptimizationPressure, pressureTarget, assistMode == AssistMode.WARMING_UP ? 0.55 : PRESSURE_RESPONSE);

		double qualityTarget = clamp01(1.0 - lastOptimizationPressure * 0.84);
		if (!shaderFrameReady && assistMode == AssistMode.ACTIVE) {
			qualityTarget = Math.min(qualityTarget, 0.88);
		}
		if (assistAgeFrames >= MAX_ASSIST_AGE_FRAMES - 1L) {
			qualityTarget = Math.min(qualityTarget, 0.74);
		}
		lastQualityBudget = blend(lastQualityBudget, qualityTarget, assistMode == AssistMode.WARMING_UP ? 0.42 : BUDGET_RESPONSE);
		return new RealtimeBudgetState(lastQualityBudget, lastOptimizationPressure);
	}

	private enum AssistMode {
		OFF,
		WARMING_UP,
		ACTIVE
	}

	private enum DispatchKind {
		WARMUP,
		REALTIME
	}

	private static double getDoubleProperty(String name, double fallback) {
		String value = System.getProperty(name);
		if (value == null || value.isBlank()) {
			return fallback;
		}
		try {
			return Double.parseDouble(value);
		} catch (NumberFormatException exception) {
			LOGGER.warn("Invalid double system property {}={}; using {}", name, value, fallback);
			return fallback;
		}
	}

	private static double clamp01(double value) {
		return Math.max(0.0, Math.min(1.0, value));
	}

	private static double pressureForMs(double value, double budgetMs) {
		if (!Double.isFinite(value) || budgetMs <= 0.0) {
			return 0.0;
		}
		return clamp01((value - budgetMs * 0.75) / Math.max(budgetMs * 0.75, 0.001));
	}

	private static double maxFinite(double... values) {
		double best = Double.NaN;
		for (double value : values) {
			if (!Double.isFinite(value)) {
				continue;
			}
			best = Double.isFinite(best) ? Math.max(best, value) : value;
		}
		return Double.isFinite(best) ? best : 0.0;
	}

	private static double scaleFinite(double value, double scale) {
		return Double.isFinite(value) ? value * scale : Double.NaN;
	}

	private static double blend(double current, double target, double response) {
		double mix = clamp01(response);
		double base = Double.isFinite(current) ? current : target;
		return base + (target - base) * mix;
	}

	private static record RealtimeBudgetState(double qualityBudget, double optimizationPressure) { }
	private static record OverlayStatus(String message, ChatFormatting color) { }
}
