package dev.tensortest.npuxmxbridge;

import java.util.Arrays;

final class FrameStatsAccumulator {
	private double[] frameCpuSamples = new double[256];
	private double[] frameGpuSamples = new double[256];
	private int frameCpuCount;
	private int frameGpuCount;

	synchronized void record(double frameCpuMs, double frameGpuMs) {
		if (isFinitePositive(frameCpuMs)) {
			frameCpuSamples = ensureCapacity(frameCpuSamples, frameCpuCount + 1);
			frameCpuSamples[frameCpuCount++] = frameCpuMs;
		}
		if (isFinitePositive(frameGpuMs)) {
			frameGpuSamples = ensureCapacity(frameGpuSamples, frameGpuCount + 1);
			frameGpuSamples[frameGpuCount++] = frameGpuMs;
		}
	}

	synchronized void reset() {
		frameCpuCount = 0;
		frameGpuCount = 0;
	}

	synchronized FrameStatsSnapshot snapshot() {
		double frameCpuAvgMs = average(frameCpuSamples, frameCpuCount);
		double frameCpu1PctWorstMs = worstBucketAverage(frameCpuSamples, frameCpuCount, 0.01);
		double frameCpu0_1PctWorstMs = worstBucketAverage(frameCpuSamples, frameCpuCount, 0.001);
		double frameGpuAvgMs = average(frameGpuSamples, frameGpuCount);
		double frameGpu1PctWorstMs = worstBucketAverage(frameGpuSamples, frameGpuCount, 0.01);
		double frameGpu0_1PctWorstMs = worstBucketAverage(frameGpuSamples, frameGpuCount, 0.001);
		return new FrameStatsSnapshot(
			frameCpuCount,
			frameCpuAvgMs,
			frameCpu1PctWorstMs,
			frameCpu0_1PctWorstMs,
			frameGpuAvgMs,
			frameGpu1PctWorstMs,
			frameGpu0_1PctWorstMs,
			fpsFromFrameMs(frameCpuAvgMs),
			fpsFromFrameMs(frameCpu1PctWorstMs),
			fpsFromFrameMs(frameCpu0_1PctWorstMs)
		);
	}

	private static double[] ensureCapacity(double[] array, int minimumLength) {
		if (minimumLength <= array.length) {
			return array;
		}
		return Arrays.copyOf(array, Math.max(array.length * 2, minimumLength));
	}

	private static boolean isFinitePositive(double value) {
		return Double.isFinite(value) && value > 0.0;
	}

	private static double average(double[] values, int count) {
		if (count <= 0) {
			return Double.NaN;
		}
		double sum = 0.0;
		for (int index = 0; index < count; index++) {
			sum += values[index];
		}
		return sum / count;
	}

	private static double worstBucketAverage(double[] values, int count, double fraction) {
		if (count <= 0) {
			return Double.NaN;
		}

		double[] ordered = Arrays.copyOf(values, count);
		Arrays.sort(ordered);
		int bucketSize = Math.max(1, (int) Math.ceil(count * fraction));
		double sum = 0.0;
		for (int index = count - bucketSize; index < count; index++) {
			sum += ordered[index];
		}
		return sum / bucketSize;
	}

	private static double fpsFromFrameMs(double frameMs) {
		if (!isFinitePositive(frameMs)) {
			return Double.NaN;
		}
		return 1000.0 / frameMs;
	}
}
