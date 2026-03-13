package dev.tensortest.npuxmxbridge;

record RenderTelemetrySnapshot(
	double frameCpuMs,
	double frameGpuMs,
	double uploadCpuMs,
	double uploadGpuMs,
	double previewCpuMs,
	double previewGpuMs
) {
	static RenderTelemetrySnapshot empty() {
		return new RenderTelemetrySnapshot(Double.NaN, Double.NaN, Double.NaN, Double.NaN, Double.NaN, Double.NaN);
	}
}
