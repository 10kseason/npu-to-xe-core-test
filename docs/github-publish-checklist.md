# GitHub Publish Checklist

Use this checklist before creating the first public commit.

## Privacy cleanup already applied

- local Windows user paths were removed from repository docs
- generated build output is ignored
- local cache directories are ignored
- Fabric Gradle output is ignored
- telemetry output like `npu-xmx-assist.csv` is ignored

## Manual checks before publish

1. Make sure no local Minecraft logs are copied into the repo.
2. Make sure no local shaderpack folder dump from `AppData` is copied into the repo unless it has been intentionally sanitized.
3. Review `README.md` and `fabric-npu-bridge-mod/README.md` one more time for machine-specific paths or usernames.
4. Review `git status --short` and confirm only source, docs, and intentional assets are staged.

## Recommended first publish flow

```powershell
git status --short
git add .
git commit -m "Initial public prototype"
git branch -M main
git remote add origin https://github.com/<username>/<repo>.git
git push -u origin main
```

## Optional pre-push review

Search the staged repo for local path fragments:

```powershell
git grep -n "C:\\Users\\"
git grep -n "AppData"
git grep -n "ModrinthApp"
```

These commands should return nothing before you publish.
