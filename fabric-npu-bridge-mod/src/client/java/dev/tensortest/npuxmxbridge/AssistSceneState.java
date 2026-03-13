package dev.tensortest.npuxmxbridge;

record AssistSceneState(
	double posX,
	double posY,
	double posZ,
	double yawDegrees,
	double pitchDegrees,
	double timeSeconds,
	double sunHeight,
	double rainStrength,
	double thunderStrength,
	double blockLight,
	double skyLight,
	double submergedFactor,
	double qualityBudget,
	double optimizationPressure
) {
	static AssistSceneState fromRequest(ShaderFrameRequest request) {
		return new AssistSceneState(
			request.sample().x(),
			request.sample().y(),
			request.sample().z(),
			request.sample().yaw(),
			request.sample().pitch(),
			request.timeSeconds(),
			request.sample().sunHeight(),
			request.sample().rainStrength(),
			request.sample().thunderStrength(),
			request.sample().blockLight(),
			request.sample().skyLight(),
			request.sample().submergedFactor(),
			request.sample().qualityBudget(),
			request.sample().optimizationPressure()
		);
	}

	double squaredDistanceTo(AssistSceneState other) {
		double distance = 0.0;
		distance += squaredDelta(posX, other.posX);
		distance += squaredDelta(posY, other.posY);
		distance += squaredDelta(posZ, other.posZ);
		distance += squaredDelta(yawDegrees, other.yawDegrees);
		distance += squaredDelta(pitchDegrees, other.pitchDegrees);
		distance += squaredDelta(timeSeconds, other.timeSeconds);
		distance += squaredDelta(sunHeight, other.sunHeight);
		distance += squaredDelta(rainStrength, other.rainStrength);
		distance += squaredDelta(thunderStrength, other.thunderStrength);
		distance += squaredDelta(blockLight, other.blockLight);
		distance += squaredDelta(skyLight, other.skyLight);
		distance += squaredDelta(submergedFactor, other.submergedFactor);
		distance += squaredDelta(qualityBudget, other.qualityBudget);
		distance += squaredDelta(optimizationPressure, other.optimizationPressure);
		return distance;
	}

	double changeScoreTo(AssistSceneState other) {
		double score = 0.0;
		score += squaredDelta(posX, other.posX) * 0.0012;
		score += squaredDelta(posY, other.posY) * 0.0010;
		score += squaredDelta(posZ, other.posZ) * 0.0012;
		score += squaredDelta(yawDegrees, other.yawDegrees) * 0.0009;
		score += squaredDelta(pitchDegrees, other.pitchDegrees) * 0.0012;
		score += squaredDelta(timeSeconds, other.timeSeconds) * 0.0500;
		score += squaredDelta(sunHeight, other.sunHeight) * 0.8000;
		score += squaredDelta(rainStrength, other.rainStrength) * 1.0000;
		score += squaredDelta(thunderStrength, other.thunderStrength) * 1.0000;
		score += squaredDelta(blockLight, other.blockLight) * 0.8000;
		score += squaredDelta(skyLight, other.skyLight) * 0.8000;
		score += squaredDelta(submergedFactor, other.submergedFactor) * 1.3000;
		score += squaredDelta(qualityBudget, other.qualityBudget) * 1.8000;
		score += squaredDelta(optimizationPressure, other.optimizationPressure) * 1.8000;
		return Math.sqrt(score);
	}

	private static double squaredDelta(double left, double right) {
		double delta = left - right;
		return delta * delta;
	}
}
