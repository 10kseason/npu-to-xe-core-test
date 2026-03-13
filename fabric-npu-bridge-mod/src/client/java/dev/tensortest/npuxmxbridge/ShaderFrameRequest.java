package dev.tensortest.npuxmxbridge;

record ShaderFrameRequest(
	PlayerSample sample,
	double timeSeconds,
	int width,
	int height
) { }
