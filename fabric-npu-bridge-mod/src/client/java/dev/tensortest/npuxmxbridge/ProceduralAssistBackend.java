package dev.tensortest.npuxmxbridge;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

final class ProceduralAssistBackend implements AssistTranslatorBackend {
	private static final String BACKEND_NAME = "procedural_policy_field_v2";
	private final Object lock = new Object();
	private final Map<String, SessionState> sessions = new HashMap<>();
	private int nextSessionId = 1;

	@Override
	public String transportSuffix() {
		return "procedural";
	}

	@Override
	public String compile(AssistSessionSpec spec) {
		synchronized (lock) {
			String sessionId = String.format(Locale.ROOT, "proc-%08x", nextSessionId++);
			sessions.put(sessionId, new SessionState(spec));
			return sessionId;
		}
	}

	@Override
	public AssistFrame run(String sessionId, AssistSceneState sceneState) {
		SessionState session;
		synchronized (lock) {
			session = sessions.get(sessionId);
		}
		if (session == null) {
			throw new IllegalArgumentException("Unknown translator session: " + sessionId);
		}
		AssistSessionSpec spec = session.spec;

		long startedAt = System.nanoTime();
		int[] pixels = new int[spec.width() * spec.height()];
		float[] filteredChannels = session.filteredChannels;
		if (filteredChannels == null || filteredChannels.length != pixels.length * 4) {
			filteredChannels = new float[pixels.length * 4];
		}
		double alphaSum = 0.0;
		double lumaSum = 0.0;
		boolean shader2 = "intel_npu_shader2_v1".equals(spec.profile());
		double yawRadians = Math.toRadians(sceneState.yawDegrees());
		double pitchRadians = Math.toRadians(sceneState.pitchDegrees());
		double weatherPenalty = (sceneState.rainStrength() * 0.65) + (sceneState.thunderStrength() * 0.35);
		double timePhase = sceneState.timeSeconds() * 0.12;
		double positionPhaseX = sceneState.posX() * 0.008;
		double positionPhaseZ = sceneState.posZ() * 0.007;
		double positionPhaseY = sceneState.posY() * 0.004;
		double motionScore = session.lastScene == null ? 4.0 : session.lastScene.changeScoreTo(sceneState);
		double baseHistory = clamp(shader2 ? 0.84 : 0.78, 0.0, 0.95);
		double history = clamp(baseHistory - motionScore * 0.10, 0.48, shader2 ? 0.90 : 0.84);
		double trajectoryHistory = clamp(history + (shader2 ? 0.05 : 0.03), 0.0, 0.93);
		double energyHistory = clamp(history - 0.04, 0.0, 0.88);
		double budgetHistory = clamp(history - 0.08, 0.0, 0.84);

		for (int y = 0; y < spec.height(); y++) {
			double v = spec.height() == 1 ? 0.5 : (double) y / (double) (spec.height() - 1);
			for (int x = 0; x < spec.width(); x++) {
				double u = spec.width() == 1 ? 0.5 : (double) x / (double) (spec.width() - 1);
				double trajectoryX = clamp01(
					0.5 + (0.18 * Math.sin((u * 3.2) + timePhase + (yawRadians * 0.16)
						+ positionPhaseX - (weatherPenalty * 0.45)))
				);
				double trajectoryY = clamp01(
					0.5 + (0.16 * Math.cos((v * 3.7) - (sceneState.timeSeconds() * 0.08) + (pitchRadians * 0.14)
						+ positionPhaseZ + (sceneState.submergedFactor() * 0.26)))
				);
				double energy = clamp01(
					0.50
						+ (0.08 * Math.sin((u * 4.4) + (sceneState.timeSeconds() * 0.06) + positionPhaseY))
						+ (0.12 * sceneState.sunHeight())
						+ (0.08 * sceneState.blockLight())
						+ (0.06 * sceneState.skyLight())
						- (0.10 * weatherPenalty)
						- (0.06 * sceneState.submergedFactor())
						+ (0.05 * sceneState.qualityBudget())
						- (0.04 * sceneState.optimizationPressure())
				);
				double budget = clamp01(
					0.56
						+ (0.10 * Math.cos((v * 4.1) - (sceneState.timeSeconds() * 0.05) + (sceneState.qualityBudget() * 1.2)
							- (sceneState.optimizationPressure() * 0.9)))
						+ (0.18 * sceneState.qualityBudget())
						- (0.12 * sceneState.optimizationPressure())
						+ (0.04 * sceneState.skyLight())
						- (0.03 * weatherPenalty)
				);

				if (shader2) {
					trajectoryX = boostAroundMidpoint(trajectoryX, 1.06);
					trajectoryY = boostAroundMidpoint(trajectoryY, 1.04);
					energy = clamp01(boostAroundMidpoint(energy, 1.04) + (0.02 * sceneState.qualityBudget()));
					budget = clamp01(boostAroundMidpoint(budget, 1.02) + (0.02 * sceneState.qualityBudget()));
				} else {
					trajectoryX = boostAroundMidpoint(trajectoryX, 0.92);
					trajectoryY = boostAroundMidpoint(trajectoryY, 0.90);
					energy = clamp01(boostAroundMidpoint(energy, 0.92));
					budget = clamp01(boostAroundMidpoint(budget, 0.96));
				}

				int channelIndex = ((y * spec.width()) + x) * 4;
				if (session.lastScene != null) {
					trajectoryX = mix(filteredChannels[channelIndex], trajectoryX, 1.0 - trajectoryHistory);
					trajectoryY = mix(filteredChannels[channelIndex + 1], trajectoryY, 1.0 - trajectoryHistory);
					energy = mix(filteredChannels[channelIndex + 2], energy, 1.0 - energyHistory);
					budget = mix(filteredChannels[channelIndex + 3], budget, 1.0 - budgetHistory);
				}
				filteredChannels[channelIndex] = (float) trajectoryX;
				filteredChannels[channelIndex + 1] = (float) trajectoryY;
				filteredChannels[channelIndex + 2] = (float) energy;
				filteredChannels[channelIndex + 3] = (float) budget;

				int abgr = packAbgr(trajectoryX, trajectoryY, energy, budget);
				pixels[(y * spec.width()) + x] = abgr;
				alphaSum += budget;
				lumaSum += (0.2126 * trajectoryX) + (0.7152 * trajectoryY) + (0.0722 * energy);
			}
		}
		session.filteredChannels = filteredChannels;
		session.lastScene = sceneState;

		double pixelCount = pixels.length;
		return new AssistFrame(
			spec.width(),
			spec.height(),
			pixels,
			alphaSum / pixelCount,
			lumaSum / pixelCount,
			spec.profile(),
			BACKEND_NAME,
			(System.nanoTime() - startedAt) / 1_000_000.0
		);
	}

	@Override
	public boolean release(String sessionId) {
		synchronized (lock) {
			return sessions.remove(sessionId) != null;
		}
	}

	private static double mix(double start, double end, double amount) {
		return start + (end - start) * clamp(amount, 0.0, 1.0);
	}

	private static double clamp01(double value) {
		return Math.max(0.0, Math.min(1.0, value));
	}

	private static double clamp(double value, double min, double max) {
		return Math.max(min, Math.min(max, value));
	}

	private static double boostAroundMidpoint(double value, double gain) {
		return clamp01(0.5 + ((value - 0.5) * gain));
	}

	private static int packAbgr(double red, double green, double blue, double alpha) {
		int r = channel(red);
		int g = channel(green);
		int b = channel(blue);
		int a = channel(alpha);
		return (a << 24) | (b << 16) | (g << 8) | r;
	}

	private static int channel(double value) {
		return Math.max(0, Math.min(255, (int) Math.round(clamp01(value) * 255.0)));
	}

	private static final class SessionState {
		private final AssistSessionSpec spec;
		private float[] filteredChannels;
		private AssistSceneState lastScene;

		private SessionState(AssistSessionSpec spec) {
			this.spec = spec;
		}
	}
}
