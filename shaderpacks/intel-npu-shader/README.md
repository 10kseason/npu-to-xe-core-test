# Intel NPU Shader / 인텔 NPU 셰이더

`intel-npu-shader` is a standalone Iris shaderpack built for this repository.

It does not bundle or derive from a third-party shaderpack. Instead, it consumes
the dynamic `npuAssist` texture exposed by the Fabric bridge mod and turns it
into a small original rendering pipeline with:

- HDR-style lighting accumulation in `composite`
- sun-directed shadow mapping through a real `shadow` pass
- ACES-like tone mapping in `final`
- NPU-guided GI estimation and light-trajectory tracing on top

## Korean Summary / 한국어 요약

`intel-npu-shader`는 이 저장소의 기본 원본 Iris 셰이더팩입니다. `npuAssist` 텍스처를 받아 GI, 반사, 안개, 그림자 budget에 쓰지만, `shader2`보다 GPU 쪽 보정 비중이 더 높고 더 보수적으로 동작합니다.

권장 용도:

- 비교 기준 pack
- `intel_npu_gi_v3` 프로필 테스트
- `shader2`보다 덜 공격적인 assist 사용이 필요할 때

## Quick Start / 빠른 시작

Recommended launch args for the original pack:

```text
-Dnpuxmxbridge.transport=translator -Dnpuxmxbridge.translatorBackend=python-stdio -Dnpuxmxbridge.device=NPU -Dnpuxmxbridge.shaderProfile=intel_npu_gi_v3 -Dnpuxmxbridge.shaderTileSize=80 -Dnpuxmxbridge.shaderPreviewScale=0 -Dnpuxmxbridge.intervalMs=75
```

사용 순서:

1. `python -m pip install numpy openvino`
2. 이 폴더를 Minecraft `shaderpacks` 폴더에 넣습니다.
3. Fabric 모드 jar를 `mods` 폴더에 넣습니다.
4. 런처의 Java/JVM 인자 칸에 위 문자열을 넣습니다.
5. Iris에서 `intel-npu-shader`를 선택합니다.
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
5. Select `intel-npu-shader` in Iris and press `N`.

Korean summary:

- 다른 사람에게는 최신 mod jar와 이 shaderpack 폴더만 전달하면 됩니다.
- 일반 사용자는 `pythonWorkingDir`를 넣지 않습니다.
- `-Dnpuxmxbridge.*` 값은 환경 변수가 아니라 런처의 JVM arguments 칸에 넣습니다.

## What it does

- writes terrain, textured geometry, entities, and hand passes into a small G-buffer
- shades the scene with sun-direction lighting and shadow map visibility
- tone-maps the HDR result back to display space
- uses the low-resolution NPU field as a live GI / water-reflection policy texture

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

For the original pack over the live translator path:

```text
-Dnpuxmxbridge.transport=translator
-Dnpuxmxbridge.translatorBackend=python-stdio
-Dnpuxmxbridge.device=NPU
-Dnpuxmxbridge.shaderProfile=intel_npu_gi_v3
-Dnpuxmxbridge.shaderTileSize=80
-Dnpuxmxbridge.shaderPreviewScale=0
-Dnpuxmxbridge.intervalMs=75
```

`pythonWorkingDir` is optional. If it is omitted, the mod extracts the bundled worker from the mod jar automatically.

For a more detailed explanation of shared-jar setup, prerequisites, and troubleshooting, see [`../../docs/standalone-jar-setup.md`](../../docs/standalone-jar-setup.md).

The `intel_npu_gi_v3` profile widens the NPU input basis with low-frequency scene
hints plus micro-tile lane features, then runs a deeper matmul-heavy decode that
is intentionally more Xe-core-like in spirit: the NPU builds more of the GI /
reflection / fog / shadow policy field up front, while the GPU keeps the
full-resolution HDR fetches, shadow/depth work, and final framebuffer writes.
The Fabric client now also sends a realtime `quality_budget` / `optimization_pressure`
signal so the shader can reduce actual GI, fog, and shadow tap counts under load.
The default target is an `80x80` assist tile.

For bridge-free validation that does not use the real NPU, replace the backend with `procedural` or `replay`.

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
2. Choose one transport:
   - Socket bridge: start `npu-xmx serve --host 127.0.0.1 --port 8765 --socket-port 8766`
   - Translator: use the JVM properties above and do not start the bridge service
3. Launch Minecraft with the JVM properties above.
4. Select `intel-npu-shader` in Iris.
5. Press `N` in game to enable the live NPU assist feed.
