# Iris + NPU Direction

## Current status

- The end-to-end path works with both transports:
  - Java -> socket bridge -> Python/OpenVINO -> dynamic texture -> Iris shaderpack
  - Java -> in-process translator -> `python-stdio` worker -> dynamic texture -> Iris shaderpack
- Recent telemetry shows the TCP bridge is no longer the only viable path.
- Update cadence is low enough to reuse results across frames.
- Upload cost is negligible compared with total frame time.
- The remaining issue is shader target selection and temporal stability on dynamic silhouettes, not raw transport wiring.
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
3. Consume that policy in Iris deferred/composite/final stages first, and only use compute/image-store merge paths on hardware that exposes the right GL feature set.
4. Store the merged policy in a persistent custom image or SSBO only on the higher-feature path.
5. Let heavy GPU passes read the policy to reduce sample count, refinement count, blur radius, or step budget.

## Immediate implementation direction

1. Stop trying to replace cheap post-FX math.
2. Focus on reflection, volumetric light, fog, and GI budgets.
3. On hardware that exposes the needed GL feature set, add a compute or image-store merge pass that:
   - reads the NPU assist texture,
   - blends it with GPU-visible masks already available in the shaderpack,
   - writes a persistent low-resolution budget image.
4. On the current OpenGL 3.3 path, keep the same budget logic in shader helper code first:
   - sample raw NPU assist,
   - combine it with depth, smoothness, and sky visibility in-place,
   - suppress or weaken the assist near entity pixels and silhouette discontinuities,
   - use that merged result inside reflection, volumetric, and cloud budget helpers.
5. Make reflection, volumetric, and GI-heavy passes consume the merged budget source first.
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
- Dynamic objects do not show obvious double-image ghosting after entity masking and depth/normal discontinuity rejection are applied.

## Current practical recommendation

- Use `npuxmxbridge.transport=translator` for local testing unless you explicitly need the socket bridge.
- Use `npuxmxbridge.translatorBackend=procedural` or `replay` for bridge-free smoke tests.
- Use `npuxmxbridge.translatorBackend=python-stdio` for the live NPU/OpenVINO path without TCP sockets.
- For distributed `jar + shaderpack` setups, omit `pythonWorkingDir` and let the mod extract the bundled worker from the mod jar automatically.
- Prefer `intel-npu-shader2` with `intel_npu_shader2_v1` when testing "more NPU-directed" policy behavior.
- If moving mobs or held items smear, shorten assist reuse before changing the model:
  - `-Dnpuxmxbridge.intervalMs=50`
  - `-Dnpuxmxbridge.maxAssistAgeFrames=2`
  - `-Dnpuxmxbridge.shaderTileSize=64`

## Distribution note

The practical public shape is now:

- one built mod jar from `fabric-npu-bridge-mod/build/libs/`
- one original in-repo shaderpack
- launcher JVM args that use `translator/python-stdio`
- no repo checkout path and normally no `pythonWorkingDir`

That matters because the transport contract is now stable enough to document as an end-user flow, not only as a developer workflow. The shaderpack still reads the same logical `npuAssist` texture either way.

For installation and packaging details, see [`standalone-jar-setup.md`](standalone-jar-setup.md).
