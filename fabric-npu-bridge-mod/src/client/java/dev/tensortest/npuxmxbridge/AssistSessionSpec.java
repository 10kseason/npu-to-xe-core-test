package dev.tensortest.npuxmxbridge;

record AssistSessionSpec(
	int width,
	int height,
	String device,
	String profile,
	String hint,
	boolean turbo
) {
	AssistSessionSpec {
		if (width <= 0 || height <= 0) {
			throw new IllegalArgumentException("width and height must be positive");
		}
		device = normalize(device, "AUTO");
		profile = normalize(profile, "intel_npu_gi_v3");
		hint = normalize(hint, "THROUGHPUT");
	}

	private static String normalize(String value, String fallback) {
		if (value == null) {
			return fallback;
		}
		String trimmed = value.trim();
		return trimmed.isEmpty() ? fallback : trimmed;
	}
}
