# Iris + NPU Direction

## Current status

- The end-to-end path works: Java -> bridge -> NPU -> dynamic texture -> Iris shaderpack.
- Recent telemetry shows the bridge is no longer the main problem.
- Update cadence is low enough to reuse results across frames.
- Upload cost is negligible compared with total frame time.
- The remaining issue is shader target selection, not transport.
- The current runtime reports OpenGL 3.3, so a new compute-shader merge stage is not a safe default on this machine.

## What the Iris pipeline implies

- `gbuffers_*` passes are geometry-facing passes. They rely on current-frame material inputs, vertex outputs, depth, normals, and texture sampling behavior that is tightly bound to the GPU pipeline.
- `deferred*`, `composite*`, and `final` are fullscreen screen-space stages.
- Iris supports custom textures, custom images, SSBOs, and compute passes.
- Compute passes and persistent images/SSBOs make it possible to build low-resolution policy maps on the GPU and reuse them across frames.

## What should stay on the GPU

- Anything that depends directly on current-frame depth/normal/G-buffer data.
- Anything that relies on implicit derivatives, mip selection, or heavy texture sampling context.
- Shadow lookups, ray steps, and full-resolution screen reconstruction.
- Water shading details that need exact current-frame geometry context.

## What can move toward the NPU

- Low-frequency control fields.
- Budget maps for SSR, water reflection, volumetric light, and clouds.
- Temporal policy outputs that change at 5-10 Hz instead of every frame.
- Small scene-level quality governors driven by pose, time, weather, dimension, and coarse scene state.

## Recommended architecture

1. Keep NPU output low-resolution and low-frequency.
2. Treat NPU output as policy, not as final shading.
3. Consume that policy in Iris compute/deferred/composite stages.
4. Store the merged policy in a persistent custom image or SSBO.
5. Let heavy GPU passes read the persistent policy to reduce sample count, refinement count, blur radius, or step budget.

## Immediate implementation direction

1. Stop trying to replace cheap post-FX math.
2. Focus on reflection, volumetric light, and cloud step budgets.
3. On hardware that exposes the needed GL feature set, add a compute or image-store merge pass that:
   - reads the NPU assist texture,
   - blends it with GPU-visible masks already available in the shaderpack,
   - writes a persistent low-resolution budget image.
4. On the current OpenGL 3.3 path, keep the same budget logic in shader helper code first:
   - sample raw NPU assist,
   - combine it with depth, smoothness, and sky visibility in-place,
   - use that merged result inside reflection, volumetric, and cloud budget helpers.
5. Make reflection and volumetric passes consume the merged budget source first.
6. Keep cloud control only if it shows measurable gain after the above change.

## Why not direct GPU -> NPU -> GPU

- In this stack, Minecraft/Iris runs on OpenGL.
- The practical OpenVINO sharing paths are documented around GPU and NPU runtime APIs, but not as a drop-in OpenGL-to-NPU screen-space shader path.
- That makes direct current-frame G-buffer offload unrealistic here.
- The high-probability path is hybrid: NPU for policy, GPU for execution.

## Success criteria

- Lower GPU ms in the targeted heavy pass, not just lower bridge RTT.
- Stable benefit in fixed-scene ABAB runs.
- No visible shader compile instability.
- Benefit remains after warm-up and result reuse are enabled.
