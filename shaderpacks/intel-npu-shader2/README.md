# Intel NPU Shader 2 / 인텔 NPU 셰이더 2

`intel-npu-shader2` is a standalone Iris shaderpack variant built for this repository.

It uses the same live `npuAssist` texture exposed by the Fabric bridge mod, but it
leans harder on the NPU-authored policy field. The GPU still keeps current-frame
depth reconstruction, shadow map comparisons, HDR sampling, and final framebuffer
writes, while the NPU policy more directly drives GI, fog, reflection, and shadow
budget decisions.

## Korean Summary / 한국어 요약

`intel-npu-shader2`는 이 저장소에서 만든 NPU 지향 Iris 셰이더팩입니다. 기존 `intel-npu-shader`보다 `npuAssist`를 더 직접적으로 믿고, GI, 안개, 그림자 budget, 물 반사 방향과 샘플 수에 NPU policy를 더 강하게 반영합니다.

현재 팩 특징:

- GPU의 local heuristic merge를 줄였습니다.
- `intel_npu_shader2_v1` 프로필을 권장합니다.
- 동적 객체/원거리 실루엣 고스팅을 줄이기 위해 entity mask, depth/normal/luma discontinuity 억제를 넣었습니다.
- 물 반사에는 water-only reflection + bloom/sheen/clarity post-processing을 넣었습니다.

## Quick Start / 빠른 시작

Recommended launch args for this checkout:

```text
-Dnpuxmxbridge.transport=translator -Dnpuxmxbridge.translatorBackend=python-stdio -Dnpuxmxbridge.device=NPU -Dnpuxmxbridge.shaderProfile=intel_npu_shader2_v1 -Dnpuxmxbridge.shaderTileSize=64 -Dnpuxmxbridge.shaderPreviewScale=0 -Dnpuxmxbridge.intervalMs=50 -Dnpuxmxbridge.maxAssistAgeFrames=2
```

사용 순서:

1. `python -m pip install numpy openvino`
2. 이 폴더를 Minecraft `shaderpacks` 폴더에 넣습니다.
3. Fabric 모드 jar를 `mods` 폴더에 넣습니다.
4. 런처의 Java/JVM 인자 칸에 위 문자열을 넣습니다.
5. Iris에서 `intel-npu-shader2`를 선택합니다.
6. 게임 안에서 `N` 키를 눌러 assist를 활성화합니다.

Standalone setup details for other users:

- [`../../docs/standalone-jar-setup.md`](../../docs/standalone-jar-setup.md)

## Shared Build Setup / 배포용 설치

To let another user run this pack, give them:

- the mod jar from `fabric-npu-bridge-mod/build/libs/npu-xmx-bridge-fabric-0.1.0.jar`
- this shaderpack folder

They do not need a repo checkout if they use `translator/python-stdio` without `pythonWorkingDir`.

Minimum end-user setup:

1. Install Python and run `python -m pip install numpy openvino`.
2. Copy the mod jar into `.minecraft/mods/`.
3. Copy this folder into `.minecraft/shaderpacks/`.
4. Paste the recommended JVM args below into the launcher.
5. Select `intel-npu-shader2` in Iris and press `N`.

Korean summary:

- 다른 사람에게는 최신 mod jar와 이 shaderpack 폴더만 전달하면 됩니다.
- 일반 사용자는 `pythonWorkingDir`를 넣지 않습니다.
- `-Dnpuxmxbridge.*` 값은 환경 변수가 아니라 런처의 JVM arguments 칸에 넣습니다.

## What is different from `intel-npu-shader`

- weaker GPU-side reinterpretation of `npuAssist`
- stronger use of raw NPU trajectory, energy, and budget channels
- lower GI, fog, reflection, and screen-shadow tap counts on the GPU
- separate recommended bridge profile: `intel_npu_shader2_v1`

## Requirements

- Minecraft 1.21.11
- Fabric + Iris + Sodium
- the `fabric-npu-bridge-mod` from this repository
- one assist transport:
  - socket bridge via `npu-xmx serve`
  - `translator/procedural` or `translator/replay` for smoke tests
  - `translator/python-stdio` for the live Python/OpenVINO path without TCP sockets

## Recommended JVM properties

These are JVM system properties. Put them in the Minecraft launcher's Java arguments, not in Windows environment variables.

For a live NPU test without the TCP bridge:

```text
-Dnpuxmxbridge.transport=translator
-Dnpuxmxbridge.translatorBackend=python-stdio
-Dnpuxmxbridge.device=NPU
-Dnpuxmxbridge.shaderProfile=intel_npu_shader2_v1
-Dnpuxmxbridge.shaderTileSize=64
-Dnpuxmxbridge.shaderPreviewScale=0
-Dnpuxmxbridge.intervalMs=50
-Dnpuxmxbridge.maxAssistAgeFrames=2
```

Generic distributed example:

```text
-Dnpuxmxbridge.transport=translator -Dnpuxmxbridge.translatorBackend=python-stdio -Dnpuxmxbridge.device=NPU -Dnpuxmxbridge.shaderProfile=intel_npu_shader2_v1 -Dnpuxmxbridge.shaderTileSize=64 -Dnpuxmxbridge.shaderPreviewScale=0 -Dnpuxmxbridge.intervalMs=50 -Dnpuxmxbridge.maxAssistAgeFrames=2
```

`pythonWorkingDir` is optional. If it is omitted, the mod extracts the bundled worker from the mod jar automatically.

For a more detailed explanation of shared-jar setup, prerequisites, and troubleshooting, see [`../../docs/standalone-jar-setup.md`](../../docs/standalone-jar-setup.md).

`intel_npu_shader2_v1` keeps the same wire format and matmul topology as
`intel_npu_gi_v3`, but the bridge preserves more of the low-resolution NPU policy
under realtime optimization pressure. This pack then consumes that policy more
directly instead of heavily merging it back with local GPU heuristics.

For bridge-free smoke tests that do not touch the real NPU, swap only the backend:

```text
-Dnpuxmxbridge.translatorBackend=procedural
```

or:

```text
-Dnpuxmxbridge.translatorBackend=replay
```

## GPU vs NPU split

Keep on the NPU side:

- low-resolution GI direction and energy policy
- shadow softness and shadow budget guidance
- fog density and scattering budget guidance
- water reflection direction and sample-budget guidance

Keep on the GPU side:

- live depth reconstruction
- shadow map sampling and PCF compares
- HDR neighborhood fetches
- full-resolution tone mapping and framebuffer writes
- dynamic-object masking and silhouette rejection around unstable low-resolution policy samples

## Current artifact notes

`intel-npu-shader2` now carries an entity/hand mask through the G-buffer and suppresses assist-driven GI/exposure around entity silhouettes and depth/normal discontinuities. That substantially reduces the old double-image ghosting on mobs and held items, but it does not make a slow assist feed invisible.

If moving entities still smear, lower the assist staleness before changing the model:

- shorten `intervalMs`
- lower `maxAssistAgeFrames`
- reduce `shaderTileSize`
- fall back to `replay` or `procedural` if you only need shader debugging instead of live NPU execution

## Install

1. Copy this folder into your Minecraft `shaderpacks` directory.
2. Choose one transport:
   - Socket bridge: start `npu-xmx serve --host 127.0.0.1 --port 8765 --socket-port 8766`
   - Translator: use the JVM properties above and do not start the bridge service
3. Launch Minecraft with the JVM properties above.
4. Select `intel-npu-shader2` in Iris.
5. Press `N` in game to enable the live NPU assist feed.
