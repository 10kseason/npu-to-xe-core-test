# NPU XMX Bridge Fabric Mod

Client-side Fabric test mod for the local `npu-xmx` bridge.

## What it does

- Registers a keybind on `N`
- Toggles a real-time NPU assist feed to the local bridge
- Runs a short NPU warm-up before the assist actually goes live, so cold-start spikes stay out of the measured path
- Keeps a dynamic `npu_xmx_bridge:shader_preview` texture updated for the active Iris shaderpack patch
- Displays live status in the action bar and can optionally draw a HUD preview tile for debugging
- Reuses the last completed assist field across multiple render frames instead of dispatching every frame
- Skips new NPU dispatches while the camera is effectively stationary, then refreshes when pose drift or assist age crosses a threshold
- Writes assist telemetry to `logs/npu-xmx-assist.csv` so you can compare `N` on/off runs afterward

The bridge turns player position, view direction, and world time into a small polynomial basis, runs a compact NPU MLP field on top of it, and returns pixels for a dynamic texture that the patched shaderpack samples.

## Run

1. Start the local bridge from the Python project root:

   ```powershell
   npu-xmx serve --host 127.0.0.1 --port 8765 --socket-port 8766
   ```

   Or on Windows:

   ```powershell
   ..\start-npu-bridge.bat
   ```

2. Build the mod:

   ```powershell
   .\gradlew.bat build
   ```

3. Drop the jar from `build\libs` into your Fabric `mods` folder.

4. Launch Minecraft 1.21.11 with Fabric Loader 0.18.2 and press `N` to turn the NPU shader assist on or off.

## Telemetry log

- Default path: `<minecraft profile>/logs/npu-xmx-assist.csv`
- Records `toggle`, `heartbeat`, `frame`, and `error` events
- Heartbeats are written about once per second even when the assist is off, so you can compare FPS before and after enabling `N`
- Frame rows include transport, bridge elapsed time, frame/upload CPU+GPU time, assist age in frames, whether the assist was updated this frame, tile result stats, and player pose

## Optional JVM properties

- `-Dnpuxmxbridge.transport=socket`
- `-Dnpuxmxbridge.socketHost=127.0.0.1`
- `-Dnpuxmxbridge.socketPort=8766`
- `-Dnpuxmxbridge.url=http://127.0.0.1:8765`
- `-Dnpuxmxbridge.device=NPU`
- `-Dnpuxmxbridge.intervalMs=150`
- `-Dnpuxmxbridge.updateEveryNFrames=4`
- `-Dnpuxmxbridge.warmupRequests=3`
- `-Dnpuxmxbridge.maxAssistAgeFrames=10`
- `-Dnpuxmxbridge.minPositionDelta=0.75`
- `-Dnpuxmxbridge.minAngleDelta=3.0`
- `-Dnpuxmxbridge.shaderTileSize=32`
- `-Dnpuxmxbridge.shaderPreviewScale=6`
- `-Dnpuxmxbridge.showPreviewHud=true`
- `-Dnpuxmxbridge.logEnabled=true`
- `-Dnpuxmxbridge.logPath=C:\path\to\npu-xmx-assist.csv`
- `-Dnpuxmxbridge.useVanillaPostEffect=true`

## License

This mod template inherits the CC0 license from the Fabric example mod skeleton.
