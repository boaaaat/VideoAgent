#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Optional

from action_space import game_data_root, selected_game


COMMON_FFMPEG_PATHS = (
    Path("/mnt/c/ffmpeg/bin/ffmpeg.exe"),
    Path("C:/ffmpeg/bin/ffmpeg.exe"),
)
COMMON_FFPROBE_PATHS = (
    Path("/mnt/c/ffmpeg/bin/ffprobe.exe"),
    Path("C:/ffmpeg/bin/ffprobe.exe"),
)
DEFAULT_FPS_TOLERANCE = 0.05


@dataclass
class VideoInfo:
    fps: float
    duration_seconds: float
    frame_count: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retime dataset videos so playback fps matches the real capture fps."
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default=game_data_root(selected_game),
        help="Directory containing run_*.mp4 videos.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=16.0,
        help="Desired playback fps for the fixed videos.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively.",
    )
    parser.add_argument(
        "--ffmpeg",
        default=None,
        help="Optional path to ffmpeg.",
    )
    parser.add_argument(
        "--ffprobe",
        default=None,
        help="Optional path to ffprobe.",
    )
    parser.add_argument(
        "--pattern",
        default="run_*.mp4",
        help="Glob pattern for the video files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without rewriting any files.",
    )
    parser.add_argument(
        "--fps-tolerance",
        type=float,
        default=DEFAULT_FPS_TOLERANCE,
        help="Skip videos already within this many fps of the target.",
    )
    return parser.parse_args()


def find_tool(explicit: Optional[str], common_paths: Iterable[Path], tool_name: str) -> str:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    path_hit = shutil.which(tool_name)
    if path_hit:
        candidates.append(Path(path_hit))
    candidates.extend(common_paths)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise SystemExit(
        f"Could not find {tool_name}. Pass --{tool_name} explicitly or install it in PATH."
    )


def is_windows_exe(tool_path: str) -> bool:
    return tool_path.lower().endswith(".exe")


def tool_path_arg(path: Path, tool_path: str) -> str:
    if is_windows_exe(tool_path) and path.is_absolute() and str(path).startswith("/mnt/"):
        return subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()
    return str(path)


def run_json(cmd: list[str]) -> dict:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def probe_video(video_path: Path, ffprobe_path: str) -> VideoInfo:
    probe_cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,nb_frames",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        tool_path_arg(video_path.resolve(), ffprobe_path),
    ]
    data = run_json(probe_cmd)
    stream = data["streams"][0]
    avg_fps = float(Fraction(stream["avg_frame_rate"]))
    duration_seconds = float(data["format"]["duration"])
    frame_count_raw = stream.get("nb_frames")
    frame_count = int(frame_count_raw) if frame_count_raw not in (None, "N/A") else None
    return VideoInfo(fps=avg_fps, duration_seconds=duration_seconds, frame_count=frame_count)


def retime_video(video_path: Path, target_fps: float, ffmpeg_path: str, current_info: VideoInfo) -> None:
    scale = current_info.fps / target_fps
    temp_path = video_path.with_name(f"{video_path.stem}.retime_tmp{video_path.suffix}")
    ffmpeg_cmd = [
        ffmpeg_path,
        "-y",
        "-itsscale",
        f"{scale:.12g}",
        "-i",
        tool_path_arg(video_path.resolve(), ffmpeg_path),
        "-c",
        "copy",
        tool_path_arg(temp_path.resolve(), ffmpeg_path),
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    temp_path.replace(video_path)


def iter_videos(root: Path, pattern: str, recursive: bool) -> list[Path]:
    iterator = root.rglob(pattern) if recursive else root.glob(pattern)
    return sorted(path for path in iterator if path.is_file())


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory does not exist: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset path is not a directory: {dataset_dir}")
    if args.target_fps <= 0:
        raise SystemExit("--target-fps must be > 0")
    if args.fps_tolerance < 0:
        raise SystemExit("--fps-tolerance must be >= 0")

    ffmpeg_path = find_tool(args.ffmpeg, COMMON_FFMPEG_PATHS, "ffmpeg")
    ffprobe_path = find_tool(args.ffprobe, COMMON_FFPROBE_PATHS, "ffprobe")

    videos = iter_videos(dataset_dir, args.pattern, args.recursive)
    if not videos:
        print("No matching videos found.")
        return 0

    changed = 0
    skipped = 0
    for video_path in videos:
        before = probe_video(video_path, ffprobe_path)
        if abs(before.fps - args.target_fps) <= args.fps_tolerance:
            skipped += 1
            print(f"[SKIP] {video_path.name}: already {before.fps:.3f} fps")
            continue

        print(
            f"[FIX]  {video_path.name}: {before.fps:.3f} fps, "
            f"{before.duration_seconds:.3f}s -> {args.target_fps:.3f} fps"
        )
        if args.dry_run:
            changed += 1
            continue

        retime_video(video_path, args.target_fps, ffmpeg_path, before)
        after = probe_video(video_path, ffprobe_path)
        print(
            f"       now {after.fps:.3f} fps, {after.duration_seconds:.3f}s, "
            f"{after.frame_count if after.frame_count is not None else 'N/A'} frames"
        )
        changed += 1

    print()
    print(f"Processed: {len(videos)}")
    print(f"Changed:   {changed}")
    print(f"Skipped:   {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
