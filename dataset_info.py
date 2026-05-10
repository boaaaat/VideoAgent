#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from action_space import game_data_root, selected_game


IGNORED_COLUMNS = {"timestamp", "dt", "delta_x", "delta_y"}


@dataclass
class ClipInfo:
    clip_id: str
    csv_path: Path
    video_path: Optional[Path]
    rows: int
    duration_seconds: float
    fps_estimate: Optional[float]
    started_at: Optional[datetime]
    schema: Tuple[str, ...]
    action_frames: Counter
    mouse_move_frames: int
    abs_delta_x: float
    abs_delta_y: float
    max_abs_delta_x: float
    max_abs_delta_y: float


def default_dataset_dir() -> Path:
    candidates = (Path(game_data_root(selected_game)), Path("data"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path("data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a dataset of run_*.mp4 and run_*.csv clips."
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default=str(default_dataset_dir()),
        help="Directory containing the dataset clips. Defaults to data/<selected_game> when present.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of longest clips to show in the text summary.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of the human-readable report.",
    )
    return parser.parse_args()


def iter_paths(root: Path, pattern: str, recursive: bool) -> List[Path]:
    iterator = root.rglob(pattern) if recursive else root.glob(pattern)
    return sorted(path for path in iterator if path.is_file())


def normalize_stem(path: Path) -> str:
    stem = path.stem
    if path.suffix.lower() == ".mp4" and stem.endswith("_512"):
        return stem[:-4]
    return stem


def clip_key(root: Path, path: Path) -> str:
    relative_parent = path.parent.relative_to(root)
    stem = normalize_stem(path)
    if relative_parent == Path("."):
        return stem
    return f"{relative_parent.as_posix()}/{stem}"


def safe_float(value: Optional[str]) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def truthy(value: Optional[str]) -> bool:
    parsed = safe_float(value)
    return parsed is not None and parsed > 0.5


def parse_started_at(path: Path) -> Optional[datetime]:
    stem = normalize_stem(path)
    if not stem.startswith("run_"):
        return None
    stamp = stem[4:]
    try:
        return datetime.strptime(stamp, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def median_or_none(values: Sequence[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def find_all_files(root: Path, recursive: bool) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
    csvs: Dict[str, List[Path]] = {}
    videos: Dict[str, List[Path]] = {}

    for path in iter_paths(root, "run_*.csv", recursive):
        csvs.setdefault(clip_key(root, path), []).append(path)

    for path in iter_paths(root, "run_*.mp4", recursive):
        videos.setdefault(clip_key(root, path), []).append(path)

    return csvs, videos


def scan_csv(clip_id: str, csv_path: Path, video_path: Optional[Path]) -> ClipInfo:
    rows = 0
    min_timestamp: Optional[float] = None
    max_timestamp: Optional[float] = None
    positive_dts: List[float] = []
    schema: Tuple[str, ...] = ()
    action_frames: Counter = Counter()
    mouse_move_frames = 0
    abs_delta_x = 0.0
    abs_delta_y = 0.0
    max_abs_delta_x = 0.0
    max_abs_delta_y = 0.0

    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        schema = tuple(fieldnames)
        action_columns = [name for name in fieldnames if name not in IGNORED_COLUMNS]

        for row in reader:
            rows += 1

            timestamp = safe_float(row.get("timestamp"))
            if timestamp is not None:
                min_timestamp = timestamp if min_timestamp is None else min(min_timestamp, timestamp)
                max_timestamp = timestamp if max_timestamp is None else max(max_timestamp, timestamp)

            dt = safe_float(row.get("dt"))
            if dt is not None and 0 < dt < 1:
                positive_dts.append(dt)

            delta_x = safe_float(row.get("delta_x")) or 0.0
            delta_y = safe_float(row.get("delta_y")) or 0.0
            abs_delta_x += abs(delta_x)
            abs_delta_y += abs(delta_y)
            max_abs_delta_x = max(max_abs_delta_x, abs(delta_x))
            max_abs_delta_y = max(max_abs_delta_y, abs(delta_y))
            if delta_x != 0.0 or delta_y != 0.0:
                mouse_move_frames += 1

            for column in action_columns:
                if truthy(row.get(column)):
                    action_frames[column] += 1

    duration_seconds = 0.0
    if max_timestamp is not None:
        if min_timestamp is None:
            duration_seconds = max_timestamp
        else:
            duration_seconds = max(max_timestamp, max_timestamp - min_timestamp)
    elif positive_dts:
        duration_seconds = sum(positive_dts)

    fps_estimate: Optional[float] = None
    if rows > 1 and duration_seconds > 0:
        fps_estimate = rows / duration_seconds
    elif positive_dts:
        average_dt = statistics.mean(positive_dts)
        if average_dt > 0:
            fps_estimate = 1.0 / average_dt

    return ClipInfo(
        clip_id=clip_id,
        csv_path=csv_path,
        video_path=video_path,
        rows=rows,
        duration_seconds=duration_seconds,
        fps_estimate=fps_estimate,
        started_at=parse_started_at(csv_path),
        schema=schema,
        action_frames=action_frames,
        mouse_move_frames=mouse_move_frames,
        abs_delta_x=abs_delta_x,
        abs_delta_y=abs_delta_y,
        max_abs_delta_x=max_abs_delta_x,
        max_abs_delta_y=max_abs_delta_y,
    )


def format_seconds(total_seconds: float) -> str:
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds - (hours * 3600 + minutes * 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:05.2f}s"
    if minutes:
        return f"{minutes:d}m {seconds:05.2f}s"
    return f"{seconds:.2f}s"


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def path_bytes(paths: Iterable[Path]) -> int:
    total = 0
    for path in paths:
        try:
            total += path.stat().st_size
        except FileNotFoundError:
            continue
    return total


def build_summary(root: Path, recursive: bool, top_n: int) -> Dict[str, object]:
    csvs, videos = find_all_files(root, recursive)
    all_clip_ids = sorted(set(csvs) | set(videos))
    paired_clip_ids = sorted(set(csvs) & set(videos))
    csv_only_clip_ids = sorted(set(csvs) - set(videos))
    video_only_clip_ids = sorted(set(videos) - set(csvs))

    duplicate_csvs = {clip_id: paths for clip_id, paths in csvs.items() if len(paths) > 1}
    duplicate_videos = {clip_id: paths for clip_id, paths in videos.items() if len(paths) > 1}

    clip_infos: List[ClipInfo] = []
    schema_counts: Counter = Counter()
    action_frame_totals: Counter = Counter()
    action_clip_totals: Counter = Counter()

    for clip_id in sorted(csvs):
        csv_path = csvs[clip_id][0]
        video_path = videos.get(clip_id, [None])[0]
        clip = scan_csv(clip_id, csv_path, video_path)
        clip_infos.append(clip)
        schema_counts[clip.schema] += 1
        action_frame_totals.update(clip.action_frames)
        for action in clip.action_frames:
            action_clip_totals[action] += 1

    total_rows = sum(clip.rows for clip in clip_infos)
    durations = [clip.duration_seconds for clip in clip_infos if clip.duration_seconds > 0]
    total_duration = sum(durations)
    fps_values = [clip.fps_estimate for clip in clip_infos if clip.fps_estimate]
    mouse_move_frames = sum(clip.mouse_move_frames for clip in clip_infos)
    total_abs_delta_x = sum(clip.abs_delta_x for clip in clip_infos)
    total_abs_delta_y = sum(clip.abs_delta_y for clip in clip_infos)
    max_abs_delta_x = max((clip.max_abs_delta_x for clip in clip_infos), default=0.0)
    max_abs_delta_y = max((clip.max_abs_delta_y for clip in clip_infos), default=0.0)

    started_at_values = [clip.started_at for clip in clip_infos if clip.started_at is not None]
    started_at_values.sort()

    top_clips = sorted(
        clip_infos,
        key=lambda clip: (clip.duration_seconds, clip.rows, clip.clip_id),
        reverse=True,
    )[: max(top_n, 0)]

    all_csv_paths = [paths[0] for paths in csvs.values()]
    all_video_paths = [paths[0] for paths in videos.values()]

    actions_summary = {}
    for action in sorted(action_frame_totals):
        active_frames = action_frame_totals[action]
        actions_summary[action] = {
            "active_frames": active_frames,
            "active_frame_pct": (active_frames / total_rows * 100.0) if total_rows else 0.0,
            "clips_with_action": action_clip_totals[action],
        }

    schema_breakdown = [
        {
            "count": count,
            "columns": list(schema),
        }
        for schema, count in schema_counts.most_common()
    ]

    return {
        "dataset_dir": str(root.resolve()),
        "recursive": recursive,
        "files": {
            "clip_ids": len(all_clip_ids),
            "paired_clips": len(paired_clip_ids),
            "csv_clips": len(csvs),
            "video_clips": len(videos),
            "csv_only_clips": len(csv_only_clip_ids),
            "video_only_clips": len(video_only_clip_ids),
            "duplicate_csv_keys": len(duplicate_csvs),
            "duplicate_video_keys": len(duplicate_videos),
            "csv_size_bytes": path_bytes(all_csv_paths),
            "video_size_bytes": path_bytes(all_video_paths),
            "total_size_bytes": path_bytes(all_csv_paths) + path_bytes(all_video_paths),
        },
        "labels": {
            "total_rows": total_rows,
            "avg_rows_per_csv": (total_rows / len(clip_infos)) if clip_infos else 0.0,
            "schema_variants": len(schema_counts),
            "schema_breakdown": schema_breakdown,
            "columns": sorted({column for clip in clip_infos for column in clip.action_frames} | {column for clip in clip_infos for column in clip.schema if column not in IGNORED_COLUMNS}),
        },
        "duration": {
            "total_seconds": total_duration,
            "total_hours": total_duration / 3600.0,
            "avg_seconds_per_csv": mean_or_none(durations),
            "median_seconds_per_csv": median_or_none(durations),
            "min_seconds_per_csv": min(durations) if durations else None,
            "max_seconds_per_csv": max(durations) if durations else None,
            "avg_fps_estimate": mean_or_none(fps_values),
        },
        "mouse": {
            "move_frames": mouse_move_frames,
            "move_frame_pct": (mouse_move_frames / total_rows * 100.0) if total_rows else 0.0,
            "total_abs_delta_x": total_abs_delta_x,
            "total_abs_delta_y": total_abs_delta_y,
            "max_abs_delta_x": max_abs_delta_x,
            "max_abs_delta_y": max_abs_delta_y,
        },
        "actions": actions_summary,
        "date_range": {
            "first_clip_started_at": started_at_values[0].isoformat(sep=" ") if started_at_values else None,
            "last_clip_started_at": started_at_values[-1].isoformat(sep=" ") if started_at_values else None,
        },
        "top_clips": [
            {
                "clip_id": clip.clip_id,
                "duration_seconds": clip.duration_seconds,
                "rows": clip.rows,
                "fps_estimate": clip.fps_estimate,
                "csv_path": str(clip.csv_path),
                "video_path": str(clip.video_path) if clip.video_path else None,
            }
            for clip in top_clips
        ],
        "missing": {
            "csv_only_clip_ids": csv_only_clip_ids,
            "video_only_clip_ids": video_only_clip_ids,
        },
        "duplicates": {
            "csv": {clip_id: [str(path) for path in paths] for clip_id, paths in duplicate_csvs.items()},
            "video": {clip_id: [str(path) for path in paths] for clip_id, paths in duplicate_videos.items()},
        },
    }


def print_text_summary(summary: Dict[str, object]) -> None:
    files = summary["files"]
    labels = summary["labels"]
    duration = summary["duration"]
    mouse = summary["mouse"]
    actions = summary["actions"]
    top_clips = summary["top_clips"]
    date_range = summary["date_range"]

    print(f"Dataset: {summary['dataset_dir']}")
    print(f"Recursive scan: {'yes' if summary['recursive'] else 'no'}")
    print()

    print("Files")
    print(f"  clip ids:          {files['clip_ids']}")
    print(f"  paired clips:      {files['paired_clips']}")
    print(f"  csv clips:         {files['csv_clips']}")
    print(f"  video clips:       {files['video_clips']}")
    print(f"  csv-only clips:    {files['csv_only_clips']}")
    print(f"  video-only clips:  {files['video_only_clips']}")
    print(f"  total size:        {format_bytes(files['total_size_bytes'])}")
    print(f"  video size:        {format_bytes(files['video_size_bytes'])}")
    print(f"  csv size:          {format_bytes(files['csv_size_bytes'])}")
    print()

    print("Duration")
    print(f"  total labeled time: {format_seconds(duration['total_seconds'])} ({duration['total_hours']:.4f} hours)")
    print(f"  avg per csv:        {format_seconds(duration['avg_seconds_per_csv'] or 0.0)}")
    print(f"  median per csv:     {format_seconds(duration['median_seconds_per_csv'] or 0.0)}")
    print(f"  shortest csv:       {format_seconds(duration['min_seconds_per_csv'] or 0.0)}")
    print(f"  longest csv:        {format_seconds(duration['max_seconds_per_csv'] or 0.0)}")
    avg_fps = duration["avg_fps_estimate"]
    print(f"  avg fps estimate:   {avg_fps:.2f}" if avg_fps else "  avg fps estimate:   n/a")
    print()

    print("Labels")
    print(f"  total rows:         {labels['total_rows']}")
    print(f"  avg rows per csv:   {labels['avg_rows_per_csv']:.1f}")
    print(f"  schema variants:    {labels['schema_variants']}")
    columns = labels["columns"]
    print(f"  action columns:     {', '.join(columns) if columns else 'none'}")
    schema_breakdown = labels["schema_breakdown"]
    if schema_breakdown:
        for index, schema_info in enumerate(schema_breakdown, start=1):
            print(
                f"  schema {index}:         {schema_info['count']} csvs "
                f"({', '.join(schema_info['columns'])})"
            )
    print()

    print("Mouse")
    print(f"  move frames:        {mouse['move_frames']} ({mouse['move_frame_pct']:.2f}%)")
    print(f"  total |delta_x|:    {mouse['total_abs_delta_x']:.0f}")
    print(f"  total |delta_y|:    {mouse['total_abs_delta_y']:.0f}")
    print(f"  max |delta_x|:      {mouse['max_abs_delta_x']:.0f}")
    print(f"  max |delta_y|:      {mouse['max_abs_delta_y']:.0f}")
    print()

    print("Actions")
    if not actions:
        print("  none")
    else:
        sorted_actions = sorted(
            actions.items(),
            key=lambda item: (-item[1]["active_frames"], item[0]),
        )
        for action, stats in sorted_actions:
            print(
                f"  {action:<15} {stats['active_frames']:>7} frames  "
                f"{stats['active_frame_pct']:>6.2f}%  {stats['clips_with_action']:>3} clips"
            )
    print()

    print("Dates")
    print(f"  first clip:         {date_range['first_clip_started_at'] or 'n/a'}")
    print(f"  last clip:          {date_range['last_clip_started_at'] or 'n/a'}")
    print()

    print("Longest clips")
    if not top_clips:
        print("  none")
    else:
        for clip in top_clips:
            fps_text = f"{clip['fps_estimate']:.2f} fps" if clip["fps_estimate"] else "fps n/a"
            print(
                f"  {clip['clip_id']}: {format_seconds(clip['duration_seconds'])}, "
                f"{clip['rows']} rows, {fps_text}"
            )


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_dir).expanduser().resolve()

    if not root.exists():
        raise SystemExit(f"Dataset directory does not exist: {root}")
    if not root.is_dir():
        raise SystemExit(f"Dataset path is not a directory: {root}")

    summary = build_summary(root=root, recursive=args.recursive, top_n=args.top)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_text_summary(summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
