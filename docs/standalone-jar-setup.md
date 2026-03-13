# Standalone Jar Setup / 배포용 Jar 실행 가이드

## Summary / 요약

You can now share the built Fabric mod jar and one shaderpack folder without requiring the other user to clone this repository.

이제 다른 사람에게 실행용 파일을 줄 때, 이 저장소 전체를 받게 할 필요가 없습니다. 최신 Fabric mod jar 하나와 shaderpack 폴더 하나만 전달하면 됩니다.

The intended distribution shape is:

- one mod jar from `fabric-npu-bridge-mod/build/libs/`
- one shaderpack folder from `shaderpacks/`
- Python installed on the target machine
- `numpy` and `openvino` installed into that Python

배포 형태는 다음이 권장됩니다.

- `fabric-npu-bridge-mod/build/libs/` 안의 mod jar 하나
- `shaderpacks/` 안의 shaderpack 폴더 하나
- 대상 PC에 설치된 Python
- 그 Python에 설치된 `numpy`, `openvino`

## Exactly What To Share / 정확히 무엇을 전달할지

If you are sending a runnable package to another person, send only:

- `fabric-npu-bridge-mod/build/libs/npu-xmx-bridge-fabric-0.1.0.jar`
- one shaderpack folder:
  - `shaderpacks/intel-npu-shader2/`, or
  - `shaderpacks/intel-npu-shader/`

Do not tell them to:

- clone the repo
- set `pythonWorkingDir`
- run the old socket bridge for the normal live translator path

다른 사람에게 실행용으로 줄 파일은 이것이면 충분합니다.

- `fabric-npu-bridge-mod/build/libs/npu-xmx-bridge-fabric-0.1.0.jar`
- shaderpack 폴더 하나

일반 사용자는 다음을 할 필요가 없습니다.

- repo clone
- `pythonWorkingDir` 지정
- 예전 socket bridge 실행

Important:

- If both `build/libs/...` and `fabric-npu-bridge-mod/build/libs/...` exist, distribute the jar inside `fabric-npu-bridge-mod/build/libs/`.

중요:

- 루트 `build/libs/`와 모듈 `fabric-npu-bridge-mod/build/libs/`가 둘 다 보이면, 배포용은 모듈 쪽 jar입니다.

## Target Folder Layout / 대상 폴더 배치 예시

The target Minecraft profile should end up looking roughly like this:

```text
.minecraft/
  mods/
    npu-xmx-bridge-fabric-0.1.0.jar
  shaderpacks/
    intel-npu-shader2/
      pack.mcmeta
      shaders/
```

For the original pack, replace `intel-npu-shader2/` with `intel-npu-shader/`.

대상 폴더 구조는 대략 이렇게 됩니다.

```text
.minecraft/
  mods/
    npu-xmx-bridge-fabric-0.1.0.jar
  shaderpacks/
    intel-npu-shader2/
      pack.mcmeta
      shaders/
```

원본 팩을 쓸 때는 `intel-npu-shader2/` 대신 `intel-npu-shader/`를 넣으면 됩니다.

## What The Mod Jar Contains / mod jar에 포함된 것

The built mod jar includes:

- the Fabric client mod classes
- the bundled replay fixture
- the bundled Python worker under `bundled-python/npu_xmx/`

The bundled worker is extracted automatically when:

- `npuxmxbridge.transport=translator`
- `npuxmxbridge.translatorBackend=python-stdio`
- `npuxmxbridge.pythonWorkingDir` is not set

빌드된 mod jar에는 다음이 들어 있습니다.

- Fabric 클라이언트 모드 클래스
- replay fixture
- `bundled-python/npu_xmx/` 아래의 번들 Python worker

아래 조건이면 이 worker가 자동으로 임시 폴더에 풀려서 실행됩니다.

- `npuxmxbridge.transport=translator`
- `npuxmxbridge.translatorBackend=python-stdio`
- `npuxmxbridge.pythonWorkingDir`를 비움

## What Is Not Bundled / jar에 포함되지 않는 것

The jar does **not** bundle:

- Python itself
- `numpy`
- `openvino`
- Intel NPU drivers/runtime from the target machine
- Minecraft/Fabric/Iris/Sodium

즉 jar 하나만으로 모든 것이 끝나지는 않습니다. 아래 항목은 대상 PC에 따로 있어야 합니다.

- Python 실행기
- `numpy`
- `openvino`
- Intel NPU 드라이버/런타임
- Minecraft/Fabric/Iris/Sodium

## Minimum Prerequisites / 최소 전제 조건

Install on the target machine:

```powershell
python -m pip install numpy openvino
```

If you want the advanced repo workflow with editable installs or optional native extras, that is a separate setup and not part of the simple shared-jar flow.

대상 PC에서는 최소한 이것만 설치하면 됩니다.

```powershell
python -m pip install numpy openvino
```

repo 개발용 `pip install -e .` 경로나 optional native extras는 별도 고급 설정이며, 단순 배포용 jar 실행 경로에는 포함되지 않습니다.

## Recommended Launch Args / 권장 실행 인자

These are JVM system properties. Put them in the Minecraft launcher's Java/JVM arguments field, not in Windows environment variables.

이 값들은 JVM system property입니다. Windows 환경 변수에 넣는 것이 아니라 Minecraft 런처의 Java/JVM arguments 칸에 넣어야 합니다.

### `intel-npu-shader2`

```text
-Dnpuxmxbridge.transport=translator -Dnpuxmxbridge.translatorBackend=python-stdio -Dnpuxmxbridge.device=NPU -Dnpuxmxbridge.shaderProfile=intel_npu_shader2_v1 -Dnpuxmxbridge.shaderTileSize=64 -Dnpuxmxbridge.shaderPreviewScale=0 -Dnpuxmxbridge.intervalMs=50 -Dnpuxmxbridge.maxAssistAgeFrames=2
```

### `intel-npu-shader`

```text
-Dnpuxmxbridge.transport=translator -Dnpuxmxbridge.translatorBackend=python-stdio -Dnpuxmxbridge.device=NPU -Dnpuxmxbridge.shaderProfile=intel_npu_gi_v3 -Dnpuxmxbridge.shaderTileSize=80 -Dnpuxmxbridge.shaderPreviewScale=0 -Dnpuxmxbridge.intervalMs=75
```

Do not add `pythonWorkingDir` for normal users. Leaving it out is what makes the mod extract and launch the bundled worker automatically.

일반 사용자는 `pythonWorkingDir`를 넣지 않습니다. 이 값을 비워야 mod jar 안의 번들 worker가 자동으로 풀려서 실행됩니다.

## Installation Flow / 설치 순서

1. Install Python on the target machine.
2. Run `python -m pip install numpy openvino`.
3. Copy the built mod jar into Minecraft `mods`.
4. Copy the desired shaderpack into Minecraft `shaderpacks`.
5. Put the launch args above into the launcher JVM arguments.
6. Start Minecraft, select the shaderpack in Iris, then press `N`.

설치 순서:

1. 대상 PC에 Python 설치
2. `python -m pip install numpy openvino` 실행
3. mod jar를 Minecraft `mods` 폴더에 복사
4. shaderpack 폴더를 Minecraft `shaderpacks` 폴더에 복사
5. 위 실행 인자를 런처 JVM arguments 칸에 입력
6. Minecraft 실행 후 Iris에서 shaderpack 선택, `N` 키로 assist 활성화

## What To Expect On First Run / 첫 실행 시 기대 동작

With the recommended `translator/python-stdio` setup:

1. Minecraft starts without needing the old socket bridge.
2. Pressing `N` triggers a short warm-up.
3. The mod extracts the bundled worker into a temp directory.
4. Python loads `openvino` and starts serving `shader_compile` / `shader_run` over stdio.
5. The shaderpack begins reading `npu_xmx_bridge:shader_preview`.

If that sequence does not happen, check the troubleshooting section below before changing shader code.

권장 `translator/python-stdio` 설정에서는 다음 흐름이 보여야 합니다.

1. 예전 socket bridge 없이 게임이 실행됨
2. `N`을 누르면 짧은 warm-up 진행
3. mod가 번들 worker를 임시 폴더로 풀기 시작함
4. Python이 `openvino`를 로드하고 stdio worker로 뜸
5. shaderpack이 `npu_xmx_bridge:shader_preview`를 읽기 시작함

## When To Use `pythonWorkingDir` / `pythonWorkingDir`가 필요한 경우

You only need `-Dnpuxmxbridge.pythonWorkingDir=...` when:

- you are developing against a repo checkout
- you want the mod to use a specific local source tree instead of the bundled worker

Normal end users should omit it.

`-Dnpuxmxbridge.pythonWorkingDir=...`는 다음 경우에만 필요합니다.

- repo checkout 기준으로 개발 중일 때
- jar 안의 번들 worker 대신 특정 로컬 소스 트리를 강제로 쓰고 싶을 때

일반 사용자는 이 값을 빼는 것이 맞습니다.

If you include `pythonWorkingDir` in shared instructions by mistake, users often point it at a non-existent repo path and the live translator path fails for the wrong reason.

## Troubleshooting / 문제 해결

### I pasted the args into Windows environment variables

- remove them from environment variables
- put the same `-Dnpuxmxbridge.*` string into the Minecraft launcher's Java arguments field instead

### 환경 변수에 넣었는데요

- 환경 변수에서 빼세요
- 같은 `-Dnpuxmxbridge.*` 문자열을 런처의 Java arguments 칸으로 옮기세요

### I pasted the args into launcher settings but the game still wants a socket bridge

- confirm the line really includes `-Dnpuxmxbridge.transport=translator`
- confirm the jar in `mods` is the current one from `fabric-npu-bridge-mod/build/libs/`
- remove older copies of the mod jar so the wrong build does not win class loading

### 런처에 넣었는데도 socket bridge를 찾습니다

- 인자에 `-Dnpuxmxbridge.transport=translator`가 실제로 들어 있는지 확인
- `mods` 폴더의 jar가 `fabric-npu-bridge-mod/build/libs/`에서 나온 최신 빌드인지 확인
- 예전 mod jar가 여러 개 있으면 지워서 잘못된 빌드가 로드되지 않게 하기

### `python` not found

- Install Python
- or set `-Dnpuxmxbridge.pythonExecutable=python`
- or point it to a full path such as `C:\Python313\python.exe`

### `No module named openvino`

- run `python -m pip install openvino numpy`
- verify the same Python executable is the one Minecraft will launch

### NPU device does not show up

- confirm Windows device driver/runtime is installed
- try `-Dnpuxmxbridge.device=CPU` first to confirm the transport path works at all
- if CPU works and NPU does not, the issue is device/runtime availability rather than the mod jar

### The game still looks for a socket bridge

- verify the launcher JVM args contain `-Dnpuxmxbridge.transport=translator`
- remove old socket-only jars from `mods`
- make sure the latest jar from `fabric-npu-bridge-mod/build/libs/` is the one being used

### Moving-object ghosting or far-distance smearing

- lower `intervalMs`
- lower `maxAssistAgeFrames`
- lower `shaderTileSize`
- for `shader2`, start from the recommended `64 / 50 / 2` values

## Internal Behavior / 내부 동작

At runtime the flow is:

1. Java translator backend starts.
2. The mod extracts `bundled-python/npu_xmx/*` from the jar into a temp directory.
3. It launches `python -m npu_xmx.translator_worker`.
4. The worker imports `openvino` and the bundled `npu_xmx` modules.
5. Java and Python exchange `shader_compile`, `shader_run`, and `shader_release` over stdio.

내부 실행 흐름은 다음과 같습니다.

1. Java translator backend 시작
2. mod가 `bundled-python/npu_xmx/*`를 임시 폴더로 추출
3. `python -m npu_xmx.translator_worker` 실행
4. worker가 `openvino`와 번들 `npu_xmx` 모듈 import
5. Java와 Python이 stdio로 `shader_compile`, `shader_run`, `shader_release`를 주고받음
