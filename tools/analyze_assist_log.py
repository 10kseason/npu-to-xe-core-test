from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from npu_xmx.bridge import benchmark_shader_profile


def parse_float(value: str) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return float(value)


def parse_int(value: str) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return int(value)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(math.floor((len(ordered) - 1) * p))
    return ordered[index]


def parse_wall_time(value: str) -> datetime:
    trimmed = value
    if "." in value:
        head, tail = value[:-1].split(".", 1)
        trimmed = f"{head}.{tail[:6]}Z"
    return datetime.strptime(trimmed, "%Y-%m-%dT%H:%M:%S.%fZ")


def collect_metric(rows: list[dict[str, str]], name: str) -> list[float]:
    return [value for row in rows if (value := parse_float(row.get(name, ""))) is not None]


def safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def format_optional(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"


def add_stat(lines: list[str], name: str, value: float | int | None) -> None:
    lines.append(f"{name}={format_optional(value)}")


@dataclass(frozen=True)
class SessionBoundaries:
    start_time: str
    end_time: str
    start_index: int
    end_index: int


def find_latest_completed_session(rows: list[dict[str, str]]) -> SessionBoundaries:
    toggles = [
        (index, row)
        for index, row in enumerate(rows)
        if row.get("event") == "toggle" and row.get("enabled", "").lower() in {"true", "false"}
    ]
    for start_pos in range(len(toggles) - 1, -1, -1):
        index, row = toggles[start_pos]
        if row.get("enabled", "").lower() != "true":
            continue
        for end_pos in range(start_pos + 1, len(toggles)):
            end_index, end_row = toggles[end_pos]
            if end_row.get("enabled", "").lower() == "false":
                return SessionBoundaries(
                    start_time=row["wall_time_utc"],
                    end_time=end_row["wall_time_utc"],
                    start_index=index,
                    end_index=end_index,
                )
    raise SystemExit("Need at least one completed ON -> OFF toggle pair in the CSV.")


def latest_row(rows: list[dict[str, str]], *, event: str, enabled: str | None = None) -> dict[str, str] | None:
    for row in reversed(rows):
        if row.get("event") != event:
            continue
        if enabled is not None and row.get("enabled", "").lower() != enabled:
            continue
        return row
    return None


def summarize_mode_row(row: dict[str, str] | None, prefix: str, lines: list[str]) -> None:
    if row is None:
        lines.append(f"{prefix}=n/a")
        return

    add_stat(lines, f"{prefix}_frame_samples", parse_int(row.get("frame_samples", "")))
    add_stat(lines, f"{prefix}_fps_avg", parse_float(row.get("fps_avg_window", "")) or parse_float(row.get("fps", "")))
    add_stat(lines, f"{prefix}_fps_1pct_low", parse_float(row.get("fps_1pct_low", "")))
    add_stat(lines, f"{prefix}_fps_0_1pct_low", parse_float(row.get("fps_0_1pct_low", "")))
    add_stat(lines, f"{prefix}_frame_cpu_avg_ms", parse_float(row.get("frame_cpu_avg_ms", "")) or parse_float(row.get("frame_cpu_ms", "")))
    add_stat(lines, f"{prefix}_frame_cpu_1pct_worst_ms", parse_float(row.get("frame_cpu_1pct_worst_ms", "")))
    add_stat(lines, f"{prefix}_frame_cpu_0_1pct_worst_ms", parse_float(row.get("frame_cpu_0_1pct_worst_ms", "")))
    add_stat(lines, f"{prefix}_frame_gpu_avg_ms", parse_float(row.get("frame_gpu_avg_ms", "")) or parse_float(row.get("frame_gpu_ms", "")))
    add_stat(lines, f"{prefix}_frame_gpu_1pct_worst_ms", parse_float(row.get("frame_gpu_1pct_worst_ms", "")))
    add_stat(lines, f"{prefix}_frame_gpu_0_1pct_worst_ms", parse_float(row.get("frame_gpu_0_1pct_worst_ms", "")))


def summarize_rows(
    rows: list[dict[str, str]],
    *,
    benchmark_equivalent: bool,
    benchmark_devices: tuple[str, ...],
    benchmark_iters: int,
    benchmark_warmup: int,
) -> list[str]:
    session = find_latest_completed_session(rows)
    start_time = session.start_time
    end_time = session.end_time

    session_rows = [row for row in rows if start_time <= row["wall_time_utc"] <= end_time]
    pre_rows = rows[: session.start_index]
    post_rows = rows[session.end_index + 1 :]

    frame_rows = [row for row in session_rows if row.get("event") == "frame"]
    heartbeat_on_rows = [
        row for row in session_rows if row.get("event") == "heartbeat" and row.get("enabled", "").lower() == "true"
    ]
    heartbeat_off_pre_rows = [
        row for row in pre_rows if row.get("event") == "heartbeat" and row.get("enabled", "").lower() == "false"
    ]
    heartbeat_off_post_rows = [
        row for row in post_rows if row.get("event") == "heartbeat" and row.get("enabled", "").lower() == "false"
    ]

    frame_elapsed = collect_metric(frame_rows, "elapsed_ms")
    frame_cpu = collect_metric(frame_rows, "frame_cpu_ms")
    frame_gpu = collect_metric(frame_rows, "frame_gpu_ms")
    upload_cpu = collect_metric(frame_rows, "upload_cpu_ms")
    upload_gpu = collect_metric(frame_rows, "upload_gpu_ms")
    preview_cpu = collect_metric(frame_rows, "preview_cpu_ms")
    preview_gpu = collect_metric(frame_rows, "preview_gpu_ms")
    assist_age = collect_metric(frame_rows, "assist_age_frames")
    mean_luma = collect_metric(frame_rows, "mean_luma")
    mean_alpha = collect_metric(frame_rows, "mean_alpha")
    assist_updated_count = sum(1 for row in frame_rows if row.get("assist_updated_this_frame", "").strip().lower() == "true")

    lines = [
        f"session_start={start_time}",
        f"session_end={end_time}",
        f"frames={len(frame_rows)}",
        f"heartbeat_on={len(heartbeat_on_rows)}",
        f"heartbeat_off_pre={len(heartbeat_off_pre_rows)}",
        f"heartbeat_off_post={len(heartbeat_off_post_rows)}",
    ]

    session_duration_s = (parse_wall_time(end_time) - parse_wall_time(start_time)).total_seconds()
    if session_duration_s > 0:
        add_stat(lines, "session_duration_s", session_duration_s)
        add_stat(lines, "frame_updates_per_s", len(frame_rows) / session_duration_s)

    def add_series(prefix: str, values: list[float]) -> None:
        add_stat(lines, f"{prefix}_avg", safe_mean(values))
        add_stat(lines, f"{prefix}_p95", percentile(values, 0.95))
        add_stat(lines, f"{prefix}_max", max(values) if values else None)

    add_series("assist_elapsed_ms", frame_elapsed)
    add_series("assist_frame_cpu_ms", frame_cpu)
    add_series("assist_frame_gpu_ms", frame_gpu)
    add_series("assist_upload_cpu_ms", upload_cpu)
    add_series("assist_upload_gpu_ms", upload_gpu)
    add_series("assist_preview_cpu_ms", preview_cpu)
    add_series("assist_preview_gpu_ms", preview_gpu)
    add_series("assist_age_frames", assist_age)
    add_series("assist_mean_luma", mean_luma)
    add_series("assist_mean_alpha", mean_alpha)
    if frame_rows:
        add_stat(lines, "assist_updated_frame_ratio", assist_updated_count / len(frame_rows))
    assist_cpu_path = []
    assist_gpu_path = []
    for row in frame_rows:
        upload_cpu_value = parse_float(row.get("upload_cpu_ms", "")) or 0.0
        upload_gpu_value = parse_float(row.get("upload_gpu_ms", "")) or 0.0
        preview_cpu_value = parse_float(row.get("preview_cpu_ms", "")) or 0.0
        preview_gpu_value = parse_float(row.get("preview_gpu_ms", "")) or 0.0
        assist_cpu_path.append(upload_cpu_value + preview_cpu_value)
        assist_gpu_path.append(upload_gpu_value + preview_gpu_value)
    add_stat(lines, "assist_cpu_path_ms_avg", safe_mean(assist_cpu_path))
    add_stat(lines, "assist_gpu_path_ms_avg", safe_mean(assist_gpu_path))

    summarize_mode_row(latest_row(heartbeat_off_pre_rows, event="heartbeat", enabled="false"), "off_pre", lines)
    summarize_mode_row(latest_row(heartbeat_on_rows, event="heartbeat", enabled="true"), "on", lines)
    summarize_mode_row(latest_row(heartbeat_off_post_rows, event="heartbeat", enabled="false"), "off_post", lines)

    on_row = latest_row(heartbeat_on_rows, event="heartbeat", enabled="true")
    off_pre_row = latest_row(heartbeat_off_pre_rows, event="heartbeat", enabled="false")
    off_post_row = latest_row(heartbeat_off_post_rows, event="heartbeat", enabled="false")
    if on_row and off_pre_row:
        add_stat(
            lines,
            "on_vs_off_pre_fps_avg_delta",
            (parse_float(on_row.get("fps_avg_window", "")) or parse_float(on_row.get("fps", "")))
            - (parse_float(off_pre_row.get("fps_avg_window", "")) or parse_float(off_pre_row.get("fps", ""))),
        )
        add_stat(
            lines,
            "on_vs_off_pre_frame_gpu_avg_ms_delta",
            (parse_float(on_row.get("frame_gpu_avg_ms", "")) or parse_float(on_row.get("frame_gpu_ms", "")))
            - (parse_float(off_pre_row.get("frame_gpu_avg_ms", "")) or parse_float(off_pre_row.get("frame_gpu_ms", ""))),
        )
    if on_row and off_post_row:
        add_stat(
            lines,
            "on_vs_off_post_fps_avg_delta",
            (parse_float(on_row.get("fps_avg_window", "")) or parse_float(on_row.get("fps", "")))
            - (parse_float(off_post_row.get("fps_avg_window", "")) or parse_float(off_post_row.get("fps", ""))),
        )
        add_stat(
            lines,
            "on_vs_off_post_frame_gpu_avg_ms_delta",
            (parse_float(on_row.get("frame_gpu_avg_ms", "")) or parse_float(on_row.get("frame_gpu_ms", "")))
            - (parse_float(off_post_row.get("frame_gpu_avg_ms", "")) or parse_float(off_post_row.get("frame_gpu_ms", ""))),
        )

    if benchmark_equivalent and frame_rows:
        last_frame = frame_rows[-1]
        profile = last_frame.get("shader_profile", "").strip() or "intel_npu_gi_v3"
        width = parse_int(last_frame.get("width", "")) or 96
        height = parse_int(last_frame.get("height", "")) or width
        benchmark_rows = benchmark_shader_profile(
            profile=profile,
            width=width,
            height=height,
            devices=benchmark_devices,
            iterations=benchmark_iters,
            warmup=benchmark_warmup,
        )
        add_stat(lines, "equivalent_width", width)
        add_stat(lines, "equivalent_height", height)
        lines.append(f"equivalent_profile={profile}")
        for result in benchmark_rows:
            label = result.device.lower()
            add_stat(lines, f"equivalent_{label}_avg_ms", result.average_ms)
            add_stat(lines, f"equivalent_{label}_p95_ms", result.p95_ms)
            add_stat(lines, f"equivalent_{label}_updates_per_s", result.updates_per_second)
            add_stat(lines, f"equivalent_{label}_pixels_per_s", result.pixels_per_second)
        benchmark_map = {result.device: result for result in benchmark_rows}
        if "NPU" in benchmark_map and "GPU" in benchmark_map:
            add_stat(
                lines,
                "gpu_vs_npu_update_ratio",
                benchmark_map["GPU"].updates_per_second / benchmark_map["NPU"].updates_per_second,
            )
            add_stat(
                lines,
                "gpu_minus_npu_avg_ms",
                benchmark_map["GPU"].average_ms - benchmark_map["NPU"].average_ms,
            )

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the latest NPU assist CSV session.")
    parser.add_argument("csv_path", type=Path, help="Path to npu-xmx-assist.csv")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip NPU/GPU/CPU equivalent shader-profile benchmarking")
    parser.add_argument("--benchmark-devices", default="NPU,GPU,CPU", help="Comma-separated devices for equivalent-profile benchmarking")
    parser.add_argument("--benchmark-iters", type=int, default=12, help="Iterations for equivalent-profile benchmarking")
    parser.add_argument("--benchmark-warmup", type=int, default=3, help="Warmup iterations for equivalent-profile benchmarking")
    args = parser.parse_args()

    with args.csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    benchmark_devices = tuple(device.strip().upper() for device in args.benchmark_devices.split(",") if device.strip())
    for line in summarize_rows(
        rows,
        benchmark_equivalent=not args.skip_benchmark,
        benchmark_devices=benchmark_devices,
        benchmark_iters=args.benchmark_iters,
        benchmark_warmup=args.benchmark_warmup,
    ):
        print(line)


if __name__ == "__main__":
    main()
