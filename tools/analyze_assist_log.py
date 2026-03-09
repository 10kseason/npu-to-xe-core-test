from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean
from datetime import datetime


def parse_float(value: str) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return float(value)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(math.floor((len(ordered) - 1) * p))
    return ordered[index]


def summarize_rows(rows: list[dict[str, str]], *, fps_tail: int = 10) -> list[str]:
    toggles = [row for row in rows if row.get("event") == "toggle"]
    if len(toggles) < 2:
        raise SystemExit("Need at least one ON toggle and one OFF toggle in the CSV.")

    start_toggle = toggles[-2]
    end_toggle = toggles[-1]
    start_time = start_toggle["wall_time_utc"]
    end_time = end_toggle["wall_time_utc"]

    session_rows = [
        row
        for row in rows
        if start_time <= row["wall_time_utc"] <= end_time
    ]
    off_rows = [row for row in rows if row["wall_time_utc"] > end_time]

    frame_rows = [row for row in session_rows if row["event"] == "frame"]
    heartbeat_on_rows = [
        row for row in session_rows if row["event"] == "heartbeat" and row["enabled"] == "true"
    ]
    heartbeat_off_rows = [
        row for row in off_rows if row["event"] == "heartbeat" and row["enabled"] == "false"
    ]

    def collect(name: str, source: list[dict[str, str]]) -> list[float]:
        return [value for row in source if (value := parse_float(row.get(name, ""))) is not None]

    frame_elapsed = collect("elapsed_ms", frame_rows)
    frame_cpu = collect("frame_cpu_ms", frame_rows)
    frame_gpu = collect("frame_gpu_ms", frame_rows)
    upload_cpu = collect("upload_cpu_ms", frame_rows)
    upload_gpu = collect("upload_gpu_ms", frame_rows)
    assist_age = collect("assist_age_frames", frame_rows)
    heartbeat_assist_age = collect("assist_age_frames", heartbeat_on_rows)
    fps_on = collect("fps", heartbeat_on_rows)
    fps_off = collect("fps", heartbeat_off_rows)
    assist_updated_count = sum(1 for row in frame_rows if row.get("assist_updated_this_frame", "").strip().lower() == "true")
    heartbeat_updated_count = sum(1 for row in heartbeat_on_rows if row.get("assist_updated_this_frame", "").strip().lower() == "true")

    lines = [
        f"session_start={start_time}",
        f"session_end={end_time}",
        f"frames={len(frame_rows)}",
        f"heartbeat_on={len(heartbeat_on_rows)}",
        f"heartbeat_off={len(heartbeat_off_rows)}",
        f"toggle_on_fps={start_toggle.get('fps', '')}",
        f"toggle_off_fps={end_toggle.get('fps', '')}",
    ]

    def parse_wall_time(value: str) -> datetime:
        trimmed = value
        if "." in value:
            head, tail = value[:-1].split(".", 1)
            trimmed = f"{head}.{tail[:6]}Z"
        return datetime.strptime(trimmed, "%Y-%m-%dT%H:%M:%S.%fZ")

    session_duration_s = (parse_wall_time(end_time) - parse_wall_time(start_time)).total_seconds()
    if session_duration_s > 0:
        lines.append(f"session_duration_s={session_duration_s:.3f}")
        lines.append(f"frame_updates_per_s={len(frame_rows) / session_duration_s:.3f}")

    def add_stats(prefix: str, values: list[float], *, tail: int | None = None) -> None:
        if not values:
            lines.append(f"{prefix}=n/a")
            return
        lines.append(f"{prefix}_avg={mean(values):.3f}")
        lines.append(f"{prefix}_p95={percentile(values, 0.95):.3f}")
        lines.append(f"{prefix}_max={max(values):.3f}")
        if tail and len(values) >= tail:
            tail_values = values[-tail:]
            lines.append(f"{prefix}_tail{tail}_avg={mean(tail_values):.3f}")

    add_stats("frame_elapsed_ms", frame_elapsed)
    add_stats("frame_cpu_ms", frame_cpu)
    add_stats("frame_gpu_ms", frame_gpu)
    add_stats("upload_cpu_ms", upload_cpu)
    add_stats("upload_gpu_ms", upload_gpu)
    add_stats("assist_age_frames", assist_age)
    add_stats("heartbeat_assist_age_frames", heartbeat_assist_age)
    add_stats("heartbeat_on_fps", fps_on, tail=fps_tail)
    add_stats("heartbeat_off_fps", fps_off, tail=fps_tail)
    if frame_rows:
        lines.append(f"assist_updated_frame_ratio={assist_updated_count / len(frame_rows):.3f}")
    if heartbeat_on_rows:
        lines.append(f"assist_updated_heartbeat_ratio={heartbeat_updated_count / len(heartbeat_on_rows):.3f}")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the latest NPU assist CSV session.")
    parser.add_argument("csv_path", type=Path, help="Path to npu-xmx-assist.csv")
    parser.add_argument("--fps-tail", type=int, default=10, help="Tail window for warm-up-trimmed FPS average")
    args = parser.parse_args()

    with args.csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    for line in summarize_rows(rows, fps_tail=args.fps_tail):
        print(line)


if __name__ == "__main__":
    main()
