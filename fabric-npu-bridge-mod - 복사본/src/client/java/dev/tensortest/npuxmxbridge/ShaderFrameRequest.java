package dev.tensortest.npuxmxbridge;

record ShaderFrameRequest(
	NpuXmxBridgeClient.PlayerSample sample,
	double timeSeconds,
	int width,
	int height
) { }
