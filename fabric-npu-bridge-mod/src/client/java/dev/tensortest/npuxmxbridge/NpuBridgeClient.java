package dev.tensortest.npuxmxbridge;

interface NpuBridgeClient {
	String transportName();

	ShaderFrameResult runShaderFrame(ShaderFrameRequest request);
}
