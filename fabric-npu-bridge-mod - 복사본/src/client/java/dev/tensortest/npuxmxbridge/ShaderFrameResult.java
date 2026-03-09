package dev.tensortest.npuxmxbridge;

record ShaderFrameResult(
	int width,
	int height,
	int[] pixelsAbgr,
	String statusLine,
	double elapsedMs,
	double meanAlpha,
	double meanLuma
) { }
