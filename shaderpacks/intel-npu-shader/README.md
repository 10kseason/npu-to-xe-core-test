# Intel NPU Shader

`intel-npu-shader` is a standalone Iris shaderpack built for this repository.

It does not bundle or derive from a third-party shaderpack. Instead, it consumes
the dynamic `npuAssist` texture exposed by the Fabric bridge mod and turns it
into a small original rendering pipeline with:

- HDR-style lighting accumulation in `composite`
- sun-directed shadow mapping through a real `shadow` pass
- ACES-like tone mapping in `final`
- NPU-driven color / glow / softness shaping on top

## What it does

- writes terrain, textured geometry, entities, and hand passes into a small G-buffer
- shades the scene with sun-direction lighting and shadow map visibility
- tone-maps the HDR result back to display space
- uses the low-resolution NPU field as a live flow / tint / mask texture

## Requirements

- Minecraft 1.21.11
- Fabric + Iris + Sodium
- the `fabric-npu-bridge-mod` from this repository
- the local bridge service running via `npu-xmx serve`

## Recommended JVM properties

```text
-Dnpuxmxbridge.shaderProfile=intel_npu_native_v2
-Dnpuxmxbridge.shaderTileSize=96
-Dnpuxmxbridge.shaderPreviewScale=0
```

The `intel_npu_native_v2` profile routes a wider, deeper dense field through the
bridge's native-biased linear pipeline when Intel NPU native support is available.
Compared with `v1`, it uses more input features and one more linear stage so the
NPU contributes more strongly to shadow, water, and grading policy.

## How to move more work to the NPU

Use the NPU for low-frequency control fields, not for the whole frame.

Good candidates:

- shadow softness or shadow budget maps
- exposure, glare, and palette coefficients
- fog, horizon glow, and day/night grading controls
- reflection or volumetric sample budgets

Keep on the GPU:

- current-frame depth reconstruction
- shadow map sampling and PCF compares
- HDR buffer neighbor taps
- full-resolution tone mapping and final framebuffer writes

Rule of thumb:

- if the math depends on `depthtex0`, `shadowtex0`, or multiple live HDR taps, keep it on the GPU
- if the math can be represented as a low-resolution policy texture updated a few times per second, it is a good NPU target

## Install

1. Copy this folder into your Minecraft `shaderpacks` directory.
2. Start the bridge with `npu-xmx serve --host 127.0.0.1 --port 8765 --socket-port 8766`.
3. Launch Minecraft with the JVM properties above.
4. Select `intel-npu-shader` in Iris.
5. Press `N` in game to enable the live NPU assist feed.
