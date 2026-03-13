package dev.tensortest.npuxmxbridge;

record AssistFrame(
	int width,
	int height,
	int[] pixelsAbgr,
	double meanAlpha,
	double meanLuma,
	String profile,
	String backend,
	double elapsedMs
) {
	AssistFrame {
		if (width <= 0 || height <= 0) {
			throw new IllegalArgumentException("width and height must be positive");
		}
		if (pixelsAbgr == null || pixelsAbgr.length != width * height) {
			throw new IllegalArgumentException("pixelsAbgr must match width * height");
		}
		profile = profile == null || profile.isBlank() ? "intel_npu_gi_v3" : profile;
		backend = backend == null || backend.isBlank() ? "translator" : backend;
	}
}
