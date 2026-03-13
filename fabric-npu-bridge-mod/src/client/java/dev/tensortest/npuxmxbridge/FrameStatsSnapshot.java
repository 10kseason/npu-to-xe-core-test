package dev.tensortest.npuxmxbridge;

record FrameStatsSnapshot(
	int frameSamples,
	double frameCpuAvgMs,
	double frameCpu1PctWorstMs,
	double frameCpu0_1PctWorstMs,
	double frameGpuAvgMs,
	double frameGpu1PctWorstMs,
	double frameGpu0_1PctWorstMs,
	double fpsAvg,
	double fps1PctLow,
	double fps0_1PctLow
) {
	static FrameStatsSnapshot empty() {
		return new FrameStatsSnapshot(
			0,
			Double.NaN,
			Double.NaN,
			Double.NaN,
			Double.NaN,
			Double.NaN,
			Double.NaN,
			Double.NaN,
			Double.NaN,
			Double.NaN
		);
	}
}
