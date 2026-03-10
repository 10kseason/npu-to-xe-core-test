# Shaderpack Integration Notes

This repository does **not** ship a third-party shaderpack, compiled GLSL output, or a bundled patch set for a specific external pack.

That is intentional.

The recommended public-facing approach is:

- keep the repository focused on the bridge, mod, telemetry, and integration contract
- optionally ship a small original shaderpack that is authored in-repo
- document how a shaderpack can consume the NPU assist texture
- avoid redistributing third-party shader code unless licensing and packaging are handled explicitly

This repository now includes one original in-repo pack:

- `shaderpacks/intel-npu-shader/`

That pack is safe to publish because it is repository-authored code, not a copied patch set from another shaderpack.

## Integration contract

The Fabric mod exposes a dynamic texture with this logical role:

- low-resolution NPU-generated assist field
- updated at low frequency
- consumed by the shaderpack as a policy / budget source

In the local experiment, that texture is registered as:

- `npu_xmx_bridge:shader_preview`

## Minimal Iris-side hookup

### 1. Add a custom texture binding

In `shaders.properties`:

```properties
customTexture.npuAssist=npu_xmx_bridge:shader_preview
```

### 2. Declare the sampler

In the shaderpack uniform declarations:

```glsl
uniform sampler2D npuAssist;
```

### 3. Sample it through a helper

Do not scatter raw `texture(npuAssist, uv)` calls everywhere.

Wrap it in a helper that:

- clamps UVs
- interprets the low-resolution assist field
- converts it into budgets such as reflection, volumetric, and cloud budgets
- optionally mixes in scene-local GPU context like depth, smoothness, or sky visibility

### 4. Apply it only to expensive passes

Use the helper in places that can actually save meaningful work:

- reflection / SSR sample count
- water reflection refinement count
- volumetric light step count
- cloud step count

Avoid using it mainly for decorative post-FX replacement. That path was useful for demos, but not for reliable frame-time wins.

## What not to publish in this repo

Avoid committing:

- a full third-party shaderpack copy
- compiled or generated shader output
- local Minecraft profile dumps
- local Modrinth or AppData paths
- one-off private experiments tied to a specific local install

## What to publish instead

Prefer publishing:

- the bridge code
- the Fabric mod
- telemetry tools
- original shaderpacks authored in this repository
- integration notes
- a small patch guide or a clean patch set if licensing allows it

## Why this is the safer public approach

It keeps the repository:

- easier to review
- easier to license correctly
- easier to reproduce
- less tied to one local machine
- less likely to leak private local setup details
