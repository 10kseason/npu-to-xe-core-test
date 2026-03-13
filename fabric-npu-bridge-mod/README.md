# NPU XMX Bridge Fabric Mod / Fabric 브리지 모드

Client-side Fabric test mod for the `npu-xmx` bridge and the in-process translator transport.

## Korean Summary / 한국어 요약

이 모드는 Minecraft 쪽에서 `N` 키로 NPU assist를 켜고 끄는 클라이언트 모드입니다. 결과를 `npu_xmx_bridge:shader_preview` 동적 텍스처로 업로드하고, 로그를 `logs/npu-xmx-assist.csv`에 기록합니다.

현재 권장 경로:

- `npuxmxbridge.transport=translator`
- `npuxmxbridge.translatorBackend=python-stdio`
- `npuxmxbridge.shaderProfile=intel_npu_shader2_v1`

이 조합은 TCP 브리지 없이 로컬 Python worker를 `stdio`로 띄워 실제 `npu_xmx` 셰이더 경로를 사용합니다.
이제 `pythonWorkingDir`를 주지 않아도 mod jar 안에 번들된 Python worker를 자동 추출해서 실행할 수 있습니다.

Detailed standalone-distribution notes:

- [`../docs/standalone-jar-setup.md`](../docs/standalone-jar-setup.md)

## Quick Start / 빠른 시작

Build:

```powershell
.\gradlew.bat build
```

Recommended launch args for this checkout:

```powershell
-Dnpuxmxbridge.transport=translator -Dnpuxmxbridge.translatorBackend=python-stdio -Dnpuxmxbridge.device=NPU -Dnpuxmxbridge.shaderProfile=intel_npu_shader2_v1 -Dnpuxmxbridge.shaderTileSize=64 -Dnpuxmxbridge.shaderPreviewScale=0 -Dnpuxmxbridge.intervalMs=50 -Dnpuxmxbridge.maxAssistAgeFrames=2
```

사용 순서:

1. `python -m pip install numpy openvino`
2. `.\gradlew.bat build`
3. `fabric-npu-bridge-mod\build\libs`의 최신 jar를 `mods` 폴더에 넣습니다.
4. 런처의 Java/JVM 인자 칸에 위 `-D...` 문자열을 넣습니다.
5. Minecraft 실행 후 `N` 키로 assist를 켭니다.

## Shared Build Packaging / 배포용 파일 안내

If you are packaging this mod for another user, the file to distribute is:

- `fabric-npu-bridge-mod/build/libs/npu-xmx-bridge-fabric-0.1.0.jar`

That jar already contains:

- the Fabric client mod
- the bundled replay fixture
- the bundled Python worker used by `translator/python-stdio`

When both a repo-root `build/libs/` jar and a `fabric-npu-bridge-mod/build/libs/` jar exist, distribute the one under `fabric-npu-bridge-mod/build/libs/`.

End-user install flow:

1. Install Python and run `python -m pip install numpy openvino`.
2. Copy this jar into `.minecraft/mods/`.
3. Copy a supported shaderpack such as `intel-npu-shader2` into `.minecraft/shaderpacks/`.
4. Paste the recommended `-Dnpuxmxbridge.*` line into the launcher's Java arguments.
5. Start Minecraft, select the shaderpack in Iris, then press `N`.

Important:

- `pythonWorkingDir` should normally be omitted for end users.
- `python-stdio` extracts the bundled worker automatically when `pythonWorkingDir` is not set.
- These `-D...` values are JVM properties, not Windows environment variables.

## What it does

- Registers a keybind on `N`
- Toggles a real-time NPU assist feed to either the local bridge or the in-process translator
- Runs a short NPU warm-up before the assist actually goes live, so cold-start spikes stay out of the measured path
- Keeps a dynamic `npu_xmx_bridge:shader_preview` texture updated for the active Iris shaderpack patch
- Displays live status in the action bar and can optionally draw a HUD preview tile for debugging
- Reuses the last completed assist field across multiple render frames instead of dispatching every frame
- Skips new NPU dispatches while the camera is effectively stationary, then refreshes when pose drift or assist age crosses a threshold
- Writes assist telemetry to `logs/npu-xmx-assist.csv` so you can compare `N` on/off runs afterward

The assist path turns player pose, world time, and a few low-frequency scene hints
(sun height, rain/thunder strength, local light, submerged state) into a compact
NPU policy field that the shaderpack samples through a dynamic texture.

When `npuxmxbridge.transport=socket`, the hot path uses the socket bridge's binary framing instead of JSON pixel arrays. Session setup and teardown still use JSON because they are infrequent.

When `npuxmxbridge.transport=translator`, the mod stays in-process and uses a Java translator backend:

- `procedural`: deterministic fake policy field for bridge-free smoke tests
- `replay`: replays checked-in golden frames from `assets/npu_xmx_bridge/translator/default_replay_fixture.json`
- `python-stdio`: launches a local Python worker over stdio and uses the real `npu_xmx` shader compile/run path without TCP sockets

All `-Dnpuxmxbridge.*` values below are JVM system properties. Put them in the Minecraft launcher's Java arguments, not in Windows environment variables.

## Run

1. Choose a transport:

   Bridge transport:

   ```powershell
   npu-xmx serve --host 127.0.0.1 --port 8765 --socket-port 8766
   ```

   Or on Windows:

   ```powershell
   ..\start-npu-bridge.bat
   ```

   In-process translator transport:

   ```powershell
   -Dnpuxmxbridge.transport=translator -Dnpuxmxbridge.translatorBackend=procedural
   ```

   Live NPU translator transport:

   ```powershell
   -Dnpuxmxbridge.transport=translator -Dnpuxmxbridge.translatorBackend=python-stdio -Dnpuxmxbridge.device=NPU
   ```

   Generic `intel-npu-shader2` example:

   ```powershell
   -Dnpuxmxbridge.transport=translator -Dnpuxmxbridge.translatorBackend=python-stdio -Dnpuxmxbridge.device=NPU -Dnpuxmxbridge.shaderProfile=intel_npu_shader2_v1 -Dnpuxmxbridge.shaderTileSize=64 -Dnpuxmxbridge.shaderPreviewScale=0 -Dnpuxmxbridge.intervalMs=50 -Dnpuxmxbridge.maxAssistAgeFrames=2
   ```

2. Build the mod:

   ```powershell
   .\gradlew.bat build
   ```

3. Drop the jar from `fabric-npu-bridge-mod\build\libs` into your Fabric `mods` folder.

4. Launch Minecraft 1.21.11 with Fabric Loader 0.18.2 and press `N` to turn the NPU shader assist on or off.

If you are sharing a prebuilt setup with another user, they do not need the repo checkout. They only need this jar, a shaderpack, Python with `numpy` and `openvino`, and the JVM args above.

## Telemetry log

- Default path: `<minecraft profile>/logs/npu-xmx-assist.csv`
- Records `toggle`, `heartbeat`, `frame`, and `error` events
- Heartbeats are written about once per second even when the assist is off, so you can compare FPS before and after enabling `N`
- Heartbeat rows now also include cumulative segment stats for:
  - average FPS
  - 1% low FPS
  - 0.1% low FPS
  - average / worst frame CPU and GPU time
- Frame rows include transport, bridge elapsed time, frame CPU+GPU time, assist upload CPU+GPU time, assist preview draw CPU+GPU time, assist age in frames, whether the assist was updated this frame, tile result stats, shader profile/backend, player pose, and appended scene-hint columns

## Optional JVM properties

- `-Dnpuxmxbridge.transport=socket`
- `-Dnpuxmxbridge.transport=translator`
- `-Dnpuxmxbridge.socketHost=127.0.0.1`
- `-Dnpuxmxbridge.socketPort=8766`
- `-Dnpuxmxbridge.url=http://127.0.0.1:8765`
- `-Dnpuxmxbridge.device=NPU`
- `-Dnpuxmxbridge.shaderProfile=intel_npu_gi_v3`
- `-Dnpuxmxbridge.translatorBackend=procedural`
- `-Dnpuxmxbridge.translatorBackend=replay`
- `-Dnpuxmxbridge.translatorBackend=python-stdio`
- `-Dnpuxmxbridge.translatorFixture=C:\path\to\translator-fixture.json`
- `-Dnpuxmxbridge.pythonExecutable=python`
- `-Dnpuxmxbridge.pythonModule=npu_xmx.translator_worker`
- `-Dnpuxmxbridge.pythonWorkingDir=C:\path\to\Tensor-test`
- `-Dnpuxmxbridge.intervalMs=75`
- `-Dnpuxmxbridge.updateEveryNFrames=1`
- `-Dnpuxmxbridge.warmupRequests=3`
- `-Dnpuxmxbridge.maxAssistAgeFrames=4`
- `-Dnpuxmxbridge.minPositionDelta=0.75`
- `-Dnpuxmxbridge.minAngleDelta=3.0`
- `-Dnpuxmxbridge.shaderTileSize=80`
- `-Dnpuxmxbridge.shaderPreviewScale=6`
- `-Dnpuxmxbridge.showPreviewHud=true`
- `-Dnpuxmxbridge.logEnabled=true`
- `-Dnpuxmxbridge.logPath=C:\path\to\npu-xmx-assist.csv`
- `-Dnpuxmxbridge.useVanillaPostEffect=true`

If live `python-stdio` runs show moving-entity smearing, start with:

- `-Dnpuxmxbridge.intervalMs=50`
- `-Dnpuxmxbridge.maxAssistAgeFrames=2`
- `-Dnpuxmxbridge.shaderTileSize=64`
- `-Dnpuxmxbridge.shaderProfile=intel_npu_shader2_v1`

`pythonWorkingDir` is optional. Use it only when you explicitly want to point the mod at a repo checkout instead of the bundled worker inside the mod jar.

For packaging/sharing guidance, including what the jar bundles and what still has to be installed separately, see [`../docs/standalone-jar-setup.md`](../docs/standalone-jar-setup.md).

## License

This mod template inherits the CC0 license from the Fabric example mod skeleton.
