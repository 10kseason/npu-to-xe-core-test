package dev.tensortest.npuxmxbridge;

import java.util.Locale;

final class NpuBridgeClientFactory {
	private NpuBridgeClientFactory() {
	}

	static NpuBridgeClient createFromSystemProperties() {
		return create(System.getProperty("npuxmxbridge.transport", "socket"));
	}

	static NpuBridgeClient create(String configuredTransport) {
		String transport = configuredTransport == null
			? "socket"
			: configuredTransport.trim().toLowerCase(Locale.ROOT);
		if ("http".equals(transport)) {
			return new NpuBridgeHttpClient();
		}
		if ("translator".equals(transport)) {
			return new NpuBridgeTranslatorClient();
		}
		return new NpuBridgeSocketClient();
	}
}
