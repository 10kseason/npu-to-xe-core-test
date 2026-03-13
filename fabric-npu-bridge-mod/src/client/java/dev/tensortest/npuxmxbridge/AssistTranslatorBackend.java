package dev.tensortest.npuxmxbridge;

interface AssistTranslatorBackend {
	String transportSuffix();

	String compile(AssistSessionSpec spec);

	AssistFrame run(String sessionId, AssistSceneState sceneState);

	boolean release(String sessionId);
}
