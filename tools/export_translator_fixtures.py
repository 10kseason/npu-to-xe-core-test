from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from npu_xmx.bridge import BridgeState  # noqa: E402
from npu_xmx.engine import OpenVINOTensorCore  # noqa: E402


def _default_output_path() -> Path:
	return REPO_ROOT / "fabric-npu-bridge-mod" / "src" / "main" / "resources" / "assets" / "npu_xmx_bridge" / "translator" / "default_replay_fixture.json"


def _scene_catalog() -> list[dict[str, float]]:
	return [
		{
			"pos_x": 12.0,
			"pos_y": 78.0,
			"pos_z": -6.0,
			"yaw_degrees": 42.0,
			"pitch_degrees": 11.5,
			"time_seconds": 0.75,
			"sun_height": 0.92,
			"rain_strength": 0.00,
			"thunder_strength": 0.00,
			"block_light": 0.55,
			"sky_light": 0.95,
			"submerged_factor": 0.00,
			"quality_budget": 0.92,
			"optimization_pressure": 0.08,
		},
		{
			"pos_x": -24.0,
			"pos_y": 65.0,
			"pos_z": 18.0,
			"yaw_degrees": -126.0,
			"pitch_degrees": 6.0,
			"time_seconds": 6.25,
			"sun_height": 0.34,
			"rain_strength": 0.78,
			"thunder_strength": 0.42,
			"block_light": 0.20,
			"sky_light": 0.30,
			"submerged_factor": 0.00,
			"quality_budget": 0.56,
			"optimization_pressure": 0.44,
		},
		{
			"pos_x": 3.0,
			"pos_y": 52.0,
			"pos_z": 9.0,
			"yaw_degrees": 178.0,
			"pitch_degrees": -24.0,
			"time_seconds": 11.5,
			"sun_height": 0.12,
			"rain_strength": 0.18,
			"thunder_strength": 0.08,
			"block_light": 0.82,
			"sky_light": 0.10,
			"submerged_factor": 1.00,
			"quality_budget": 0.48,
			"optimization_pressure": 0.72,
		},
	]


def export_fixtures(output_path: Path, sizes: list[int], profiles: list[str]) -> dict[str, object]:
	state = BridgeState(engine=OpenVINOTensorCore(preferred_devices=("CPU",)))
	entries: list[dict[str, object]] = []
	for profile in profiles:
		for size in sizes:
			compile_response = state.compile_shader(
				{
					"width": size,
					"height": size,
					"device": "CPU",
					"profile": profile,
					"hint": "THROUGHPUT",
					"turbo": True,
				}
			)
			session_id = compile_response["session_id"]
			try:
				for scene in _scene_catalog():
					run_payload = {"session_id": session_id, **scene}
					response = state.run_shader(run_payload)
					entries.append(
						{
							"profile": profile,
							"width": size,
							"height": size,
							"request": dict(scene),
							"response": {
								"width": response["width"],
								"height": response["height"],
								"pixels_abgr": response["pixels_abgr"],
								"mean_alpha": response["mean_alpha"],
								"mean_luma": response["mean_luma"],
								"profile": response["profile"],
								"backend": response["backend"],
							},
						}
					)
			finally:
				state.release_shader(session_id)

	return {
		"version": 1,
		"generated_at_utc": datetime.now(timezone.utc).isoformat(),
		"device": "CPU",
		"entries": entries,
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Export golden replay fixtures for the Fabric translator backend.")
	parser.add_argument(
		"--output",
		type=Path,
		default=_default_output_path(),
		help="Where to write the JSON fixture file.",
	)
	parser.add_argument(
		"--sizes",
		default="4,80",
		help="Comma-separated square tile sizes to export.",
	)
	parser.add_argument(
		"--profiles",
		default="intel_npu_gi_v3,intel_npu_shader2_v1",
		help="Comma-separated shader profiles to export.",
	)
	args = parser.parse_args()

	sizes = [int(part.strip()) for part in args.sizes.split(",") if part.strip()]
	profiles = [part.strip() for part in args.profiles.split(",") if part.strip()]
	payload = export_fixtures(args.output, sizes=sizes, profiles=profiles)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
	print(f"Wrote {len(payload['entries'])} fixture entries to {args.output}")


if __name__ == "__main__":
	main()
