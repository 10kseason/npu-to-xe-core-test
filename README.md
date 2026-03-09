# NPU XMX Prototype

Hybrid Intel NPU + Minecraft shader experiment.

This repository started as a small OpenVINO `MatMul` / `Linear` wrapper for Intel `NPU`, `GPU`, and `CPU`, then grew into an end-to-end prototype that feeds a low-resolution NPU-generated control field into a Minecraft Fabric + Iris shader pipeline.

The short version:

- The architecture works.
- The bridge overhead is now small enough.
- The hard part is not transport anymore.
- The hard part is choosing shader work that is actually expensive enough to justify NPU involvement.

## Current status

This project is no longer at the "can this be wired up?" stage.

It already has:

- a Python bridge that can run small FP16 tensor workloads on Intel NPU
- a Fabric client mod that streams real-time assist data into Minecraft
- an Iris shaderpack integration path that consumes the NPU output
- telemetry for bridge time, frame time, upload time, and assist reuse

What is still missing is a repeatable net rendering win in a heavy real-world scene. The current prototype proves feasibility, but it does not yet prove that the chosen shader substitutions beat their orchestration cost in practice.

## What this repo contains

- `src/npu_xmx/engine.py`
  Tensor-style `MatMul` / `Linear` execution over OpenVINO, with optional native Intel NPU acceleration for supported cases.
- `src/npu_xmx/bridge.py`
  Local companion service with HTTP and low-overhead socket transports.
- `src/npu_xmx/cli.py`
  Device listing, microbenchmarks, and bridge startup.
- `fabric-npu-bridge-mod/`
  Client-side Fabric mod that talks to the bridge, updates a dynamic texture, and records telemetry.
- `tools/analyze_assist_log.py`
  Summarizes the latest NPU assist session from CSV logs.
- `docs/iris-npu-direction.md`
  Design notes and the current architectural direction.

Important note:

- The live Complementary/Euphoria shaderpack edits used during local profiling currently live in the author's local Minecraft profile, not as a packaged shader patch inside this repo yet.

## Tested stack

- Windows
- Python 3.10+
- OpenVINO 2025.4+
- Minecraft 1.21.11
- Fabric Loader 0.18.2
- Iris 1.10.4+mc1.21.11
- Sodium
- Intel Core Ultra platform with Intel NPU and Intel Arc integrated graphics

## What we learned

### 1. The NPU can be used as a real assist device

The project can:

- compile small fixed-shape workloads for Intel NPU
- run them in a separate companion process
- stream results into a Fabric mod
- expose them as a dynamic texture to an Iris shaderpack

That part is real and works end to end.

### 2. HTTP was too slow, socket transport was good enough

Early bridge numbers made this obvious:

- direct NPU tiny linear: about `0.16 ms`
- socket bridge: about `0.38 ms`
- HTTP bridge: about `6.8 ms`

HTTP was fine for inspection, but not for a hot path.

### 3. Upload cost is not the main problem anymore

Recent telemetry runs show texture upload is tiny compared with total frame time:

- `upload_cpu_ms_avg`: about `0.15 ms`
- `upload_gpu_ms_avg`: about `0.02 ms`

So the current bottleneck is not "getting the NPU result into OpenGL."

### 4. Cold start mattered a lot

The first NPU request could spike badly, so the Fabric mod now warms up before enabling the live assist path. That removed misleading first-use spikes from the measured path.

### 5. Reuse matters more than raw dispatch speed

The assist path originally refreshed too often. Motion-gated reuse and frame/time decimation were necessary. Current sessions can run at only a few assist updates per second while reusing the latest result across multiple render frames.

### 6. Cheap post-FX substitution was the wrong target

Replacing decorative fullscreen math like scanlines, noise overlays, or similar post effects was interesting as a demo, but it did not produce a reliable frame-time win.

This was the biggest lesson.

### 7. Sodium changes the interpretation

Minecraft here is running with Sodium. That means the main renderer and chunk path are already being optimized on the GPU/renderer side. The NPU is not a replacement for Sodium's job.

The realistic target is:

- keep Sodium doing geometry/chunk rendering
- keep Iris doing the actual shading
- use the NPU only to generate low-frequency control fields that help Iris do less expensive screen-space work

### 8. "Direct GLSL on NPU" is not the right mental model

The NPU is not directly executing GLSL.

The practical high-probability architecture is:

- NPU generates a low-resolution policy / budget field
- GPU shader code reads that field
- expensive GPU passes reduce sample count, refinement count, blur radius, or step count

That is very different from "move the shader to the NPU."

### 9. The current best direction is NPU policy + GPU execution

The most promising targets are not cheap effects. They are passes like:

- reflections / SSR / water reflection budget
- volumetric light step budget
- cloud step budget

The NPU should decide where the GPU can spend less work, not try to replace all pixel work outright.

## Architecture

Today the prototype looks like this:

1. Fabric mod captures lightweight scene state.
2. The mod sends that state to the local bridge over a socket.
3. The bridge builds a compact basis and runs a small NPU model.
4. The result comes back as a low-resolution RGBA field.
5. Minecraft uploads it as a dynamic texture.
6. Iris shader code turns that texture into reflection / volumetric / cloud budgets.

This is a hybrid assist path, not direct GPU-to-NPU screen-space offload.

## Why not direct GPU -> NPU -> GPU?

Because this stack is constrained by reality:

- Minecraft + Iris here runs on OpenGL
- the tested runtime reports OpenGL 3.3
- OpenVINO NPU integration is not a drop-in path for "take current G-buffer, run NPU, feed it back into GLSL"

That means direct current-frame G-buffer offload is not the high-probability route in this environment.

The practical route is still:

- small NPU output
- low-frequency updates
- GPU-local execution of the expensive shading

## Current shader direction

The current experiment is moving away from post-FX replacement and toward budget shaping:

- reflection budget
- volumetric light budget
- cloud budget

The helper that consumes the NPU assist now also mixes in GPU-visible scene context such as:

- depth
- sky visibility
- material smoothness

That keeps the NPU result coarse and reusable while still letting the GPU adapt it to the current frame.

## Install

```powershell
python -m pip install -e .
```

Optional native NPU backend:

```powershell
python -m pip install -e ".[native-npu]"
```

## CLI

List devices:

```powershell
npu-xmx devices
```

Benchmark static `MatMul`:

```powershell
npu-xmx matmul --m 64 --k 1024 --n 1024 --devices NPU,GPU,CPU --iters 30
```

Benchmark fixed-weight `Linear`:

```powershell
npu-xmx linear --batch 64 --in-features 1024 --out-features 2048 --devices NPU,GPU,CPU --iters 30
```

Run the local bridge:

```powershell
npu-xmx serve --host 127.0.0.1 --port 8765 --socket-port 8766
```

Windows helper:

```powershell
.\start-npu-bridge.bat
```

## Fabric sample mod

The Fabric sample mod lives in [`fabric-npu-bridge-mod/README.md`](fabric-npu-bridge-mod/README.md).

At a high level it:

- binds `N` to toggle the assist path
- warms up the NPU path before going live
- updates a dynamic texture for Iris
- records telemetry into `logs/npu-xmx-assist.csv`

Useful JVM properties include:

- `-Dnpuxmxbridge.intervalMs=150`
- `-Dnpuxmxbridge.updateEveryNFrames=4`
- `-Dnpuxmxbridge.maxAssistAgeFrames=10`
- `-Dnpuxmxbridge.minPositionDelta=0.75`
- `-Dnpuxmxbridge.minAngleDelta=3.0`

## Telemetry

The assist logger records:

- `toggle`
- `heartbeat`
- `frame`
- `error`

And includes fields such as:

- bridge elapsed time
- frame CPU and GPU time
- upload CPU and GPU time
- assist age in frames
- whether the assist was updated this frame
- pose and timing context

Summarize the latest session:

```powershell
python .\tools\analyze_assist_log.py "<minecraft profile>\\logs\\npu-xmx-assist.csv"
```

## Safety and scope

This project does not support:

- process injection
- memory hooks
- anti-cheat bypass
- arbitrary binary patching of games

The intended integration path is:

- official mod/plugin/script API
- local companion process over `localhost`
- explicit shaderpack-side consumption of assist data

## Publishing

Before making the repository public, follow [`docs/github-publish-checklist.md`](docs/github-publish-checklist.md).

## Honest project assessment

Right now the best description is:

- feasibility: proven
- transport overhead: mostly solved
- shader integration: proven
- consistent performance win: not proven yet

So this repository should be read as a serious working prototype and research log, not as a finished performance mod.

## Next steps

- keep narrowing focus toward reflection and volumetric budgets
- keep cloud control only if it shows measurable benefit
- package the local shaderpack edits into a reproducible distributable patch
- continue ABAB telemetry runs in heavy water / reflection / volumetric scenes

## References

- [OpenVINO supported devices](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes.html)
- [OpenVINO NPU device](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)
- [OpenVINO stateful models](https://docs.openvino.ai/nightly/openvino-workflow/running-inference/inference-request/stateful-models.html)
- [Iris ShaderDoc](https://shaders.properties/)
- [Khronos OpenGL Samplers](https://wikis.khronos.org/opengl/Sampler_%28GLSL%29)
- [Khronos OpenGL Compute Shader](https://wikis.khronos.org/opengl/Compute_Shader)
