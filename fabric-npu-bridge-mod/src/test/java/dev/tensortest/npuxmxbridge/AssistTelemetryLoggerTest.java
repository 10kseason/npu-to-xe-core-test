package dev.tensortest.npuxmxbridge;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AssistTelemetryLoggerTest {
	@TempDir
	Path tempDir;

	@Test
	void telemetryCsvKeepsColumnCountAndTranslatorTransport() throws IOException {
		String previous = System.getProperty("npuxmxbridge.logEnabled");
		System.setProperty("npuxmxbridge.logEnabled", "true");
		try (AssistTelemetryLogger logger = new AssistTelemetryLogger(tempDir.toFile())) {
			logger.logSyntheticRow("translator-java/procedural", "translator synthetic row");
		} finally {
			restoreProperty("npuxmxbridge.logEnabled", previous);
		}

		Path logPath = tempDir.resolve("logs").resolve("npu-xmx-assist.csv");
		List<String> lines = Files.readAllLines(logPath, StandardCharsets.UTF_8);
		assertEquals(2, lines.size());
		assertEquals(columnCount(AssistTelemetryLogger.CSV_HEADER), columnCount(lines.get(0)));
		assertEquals(columnCount(lines.get(0)), columnCount(lines.get(1)));
		assertTrue(lines.get(1).contains("\"translator-java/procedural\""));
	}

	private static int columnCount(String csvLine) {
		return csvLine.split(",", -1).length;
	}

	private static void restoreProperty(String name, String previous) {
		if (previous == null) {
			System.clearProperty(name);
			return;
		}
		System.setProperty(name, previous);
	}
}
