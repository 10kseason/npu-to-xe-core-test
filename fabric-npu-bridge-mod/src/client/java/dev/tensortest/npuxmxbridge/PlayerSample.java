package dev.tensortest.npuxmxbridge;

import net.minecraft.client.Minecraft;
import net.minecraft.core.BlockPos;
import net.minecraft.world.level.LightLayer;

record PlayerSample(
	double x,
	double y,
	double z,
	double yaw,
	double pitch,
	double sunHeight,
	double rainStrength,
	double thunderStrength,
	double blockLight,
	double skyLight,
	double submergedFactor,
	double qualityBudget,
	double optimizationPressure
) {
	static PlayerSample capture(Minecraft client) {
		return capture(client, 1.0, 0.0);
	}

	static PlayerSample capture(Minecraft client, double qualityBudget, double optimizationPressure) {
		double sunHeight = computeSunHeight(client);
		double rainStrength = client.level != null ? clamp01(client.level.getRainLevel(1.0F)) : 0.0;
		double thunderStrength = client.level != null ? clamp01(client.level.getThunderLevel(1.0F)) : 0.0;
		BlockPos samplePos = BlockPos.containing(client.player.getX(), client.player.getEyeY(), client.player.getZ());
		double blockLight = client.level != null ? clamp01(client.level.getBrightness(LightLayer.BLOCK, samplePos) / 15.0) : 0.0;
		double skyLight = client.level != null ? clamp01(client.level.getBrightness(LightLayer.SKY, samplePos) / 15.0) : 1.0;
		double submergedFactor = client.player.isUnderWater() ? 1.0 : 0.0;
		return new PlayerSample(
			client.player.getX(),
			client.player.getY(),
			client.player.getZ(),
			client.player.getYRot(),
			client.player.getXRot(),
			sunHeight,
			rainStrength,
			thunderStrength,
			blockLight,
			skyLight,
			submergedFactor,
			clamp01(qualityBudget),
			clamp01(optimizationPressure)
		);
	}

	double distanceTo(PlayerSample other) {
		double dx = x - other.x;
		double dy = y - other.y;
		double dz = z - other.z;
		return Math.sqrt(dx * dx + dy * dy + dz * dz);
	}

	double maxAngleDeltaTo(PlayerSample other) {
		return Math.max(wrappedAngleDelta(yaw, other.yaw), wrappedAngleDelta(pitch, other.pitch));
	}

	private static double computeSunHeight(Minecraft client) {
		if (client.level == null) {
			return 0.5;
		}

		long dayTime = client.level.getDayTime();
		double dayPhase = Math.floorMod(dayTime, 24000L) / 24000.0;
		double cycle = (dayPhase - 0.25) * Math.PI * 2.0;
		return clamp01(0.5 + 0.5 * Math.cos(cycle));
	}

	private static double clamp01(double value) {
		return Math.max(0.0, Math.min(1.0, value));
	}

	private static double wrappedAngleDelta(double a, double b) {
		double delta = Math.abs(a - b) % 360.0;
		return delta > 180.0 ? 360.0 - delta : delta;
	}
}
