# Intel NPU Shader

`intel-npu-shader` is a standalone Iris shaderpack built for this repository.

It does not bundle or derive from a third-party shaderpack. Instead, it consumes
the dynamic `npuAssist` texture exposed by the Fabric bridge mod and turns it
into a small original rendering pipeline with:

- HDR-style lighting accumulation in `composite`
- sun-directed shadow mapping through a real `shadow` pass
- ACES-like tone mapping in `final`
- NPU-guided GI estimation and light-trajectory tracing on top

## What it does

- writes terrain, textured geometry, entities, and hand passes into a small G-buffer
- shades the scene with sun-direction lighting and shadow map visibility
- tone-maps the HDR result back to display space
- uses the low-resolution NPU field as a live GI / water-reflection policy texture

## Requirements

- Minecraft 1.21.11
- Fabric + Iris + Sodium
- the `fabric-npu-bridge-mod` from this repository
- the local bridge service running via `npu-xmx serve`

## Recommended JVM properties

```text
-Dnpuxmxbridge.shaderProfile=intel_npu_gi_v2
-Dnpuxmxbridge.shaderTileSize=96
-Dnpuxmxbridge.shaderPreviewScale=0
-Dnpuxmxbridge.intervalMs=75
```

The `intel_npu_gi_v2` profile routes the shader field through a lighter hybrid
NPU pipeline: native-biased linear blocks build a scene policy basis, then
explicit NPU `matmul` stages decode that basis into indirect-light direction and
GI energy coefficients. The GPU still performs full-resolution HDR fetches and
shadow/depth work, but the NPU now owns more of the low-frequency GI policy while
staying closer to a 16.6 ms 60 FPS-style budget at `96x96`.

## How to move more work to the NPU

Use the NPU for low-frequency control fields, not for the whole frame.

Good candidates:

- shadow softness or shadow budget maps
- exposure, glare, indirect-light thresholds, and palette coefficients
- fog, horizon glow, and day/night grading controls
- reflection or volumetric sample budgets
- low-resolution indirect-light trajectory / reflection path fields

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
