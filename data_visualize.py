import argparse
import csv
import os
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

import cv2
from tqdm import tqdm

from action_space import game_data_root, selected_game


KEY_DISPLAY = {
    "Key.shift": "SHIFT",
    "Key.space": "SPACE",
    "Key.ctrl_l": "CTRL",
    "Key.ctrl_r": "CTRL",
    "Key.alt_l": "ALT",
    "Key.alt_r": "ALT",
    "Key.tab": "TAB",
    "Key.esc": "ESC",
}


class FFmpegWriter:
    def __init__(
        self,
        out_path: str,
        fps: float,
        width: int,
        height: int,
        ffmpeg_path: Optional[str] = None,
        use_nvenc_codec: str = "hevc_nvenc",
        container_ext: str = "mp4",
        cq: int = 20,
        preset: str = "p4",
        pix_fmt_in: str = "bgr24",
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.pix_fmt_in = pix_fmt_in
        self.container_ext = container_ext

        self.out_file = (
            out_path if out_path.endswith(f".{container_ext}") else f"{out_path}.{container_ext}"
        )

        self.ffmpeg_path = ffmpeg_path or shutil.which("ffmpeg")
        if not self.ffmpeg_path:
            raise RuntimeError("FFmpeg not found. Add ffmpeg to PATH or set --ffmpeg-path.")

        self.cmd = [
            self.ffmpeg_path,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            self.pix_fmt_in,
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            use_nvenc_codec,
            "-cq",
            str(cq),
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
            self.out_file,
        ]

        self.proc = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write(self, frame_bgr) -> None:
        if self.proc and self.proc.stdin:
            self.proc.stdin.write(frame_bgr.tobytes())

    def release(self) -> None:
        if self.proc:
            try:
                if self.proc.stdin:
                    self.proc.stdin.flush()
                    self.proc.stdin.close()
            except Exception:
                pass
            self.proc.wait(timeout=5)
            self.proc = None


def _to_float(value: Optional[str]) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_int(value: Optional[str]) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _display_key(name: str) -> str:
    if name in KEY_DISPLAY:
        return KEY_DISPLAY[name]
    if name.startswith("Key."):
        return name.split(".", 1)[1].upper()
    if len(name) == 1:
        return name.upper()
    return name


def _truthy(value: Optional[str]) -> bool:
    return _to_float(value) > 0.5


def find_csv_for_video(video_path: str, csv_ext: str = ".csv") -> str:
    base = os.path.splitext(video_path)[0]
    candidates = [
        base + csv_ext,
        base.replace("_512", "") + csv_ext,
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"CSV not found for video. Tried: {', '.join(candidates)}"
    )


def load_csv_rows(csv_path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return fieldnames, rows


def split_columns(fieldnames: List[str]) -> Tuple[List[str], List[str]]:
    ignore = {"timestamp", "dt", "delta_x", "delta_y"}
    keys: List[str] = []
    clicks: List[str] = []
    for name in fieldnames:
        if name in ignore:
            continue
        if name.endswith("_click"):
            clicks.append(name)
        else:
            keys.append(name)
    return keys, clicks


def parse_row(
    row: Dict[str, str],
    key_names: List[str],
    click_names: List[str],
) -> Tuple[Dict[str, bool], Dict[str, bool], Tuple[int, int], float]:
    keys = {k: _truthy(row.get(k)) for k in key_names}
    clicks = {c: _truthy(row.get(c)) for c in click_names}
    dx = _to_int(row.get("delta_x"))
    dy = _to_int(row.get("delta_y"))
    timestamp = _to_float(row.get("timestamp"))
    return keys, clicks, (dx, dy), timestamp


def transition_between(
    previous: Dict[str, bool],
    current: Dict[str, bool],
) -> Tuple[List[str], List[str]]:
    pressed = [name for name, active in current.items() if active and not previous.get(name, False)]
    released = [name for name, active in previous.items() if active and not current.get(name, False)]
    return pressed, released


def draw_overlay(
    frame,
    *,
    frame_idx: int,
    total_frames: int,
    fps: float,
    label_idx: Optional[int],
    prediction_horizon: int,
    action_label_offset: int,
    keys: Optional[Dict[str, bool]],
    clicks: Optional[Dict[str, bool]],
    pressed: Optional[List[str]],
    released: Optional[List[str]],
    mouse_delta: Tuple[int, int],
    label_timestamp: Optional[float],
) -> None:
    h, w = frame.shape[:2]
    box_w = min(520, w - 20)
    box_h = min(215, h - 20)
    x0, y0 = 10, 10
    x1, y1 = x0 + box_w, y0 + box_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    y = y0 + 22
    frame_time = (frame_idx / fps) if fps > 0 else 0.0
    total_str = f"{total_frames - 1}" if total_frames > 0 else "?"
    cv2.putText(
        frame,
        f"Frame: {frame_idx}/{total_str}  Video: {frame_time:.3f}s",
        (x0 + 10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    y += 22

    label_str = f"{label_idx}" if label_idx is not None else "N/A"
    cv2.putText(
        frame,
        f"Target idx: {label_str} = frame + frame_offset - 1 + label_offset",
        (x0 + 10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )
    y += 22

    cv2.putText(
        frame,
        f"Frame offset: {prediction_horizon:+d}  label_offset: {action_label_offset:+d}",
        (x0 + 10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )
    y += 22

    if label_timestamp is None:
        label_ts_str = "N/A"
    else:
        label_ts_str = f"{label_timestamp:.3f}s"
    cv2.putText(
        frame,
        f"Label time: {label_ts_str}",
        (x0 + 10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )
    y += 22

    if keys is None or clicks is None:
        cv2.putText(
            frame,
            "Keys: N/A",
            (x0 + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        y += 22
        cv2.putText(
            frame,
            "Click: N/A",
            (x0 + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 255),
            1,
        )
        y += 22
    else:
        active_keys = [_display_key(k) for k, v in keys.items() if v]
        cv2.putText(
            frame,
            f"Keys: {' '.join(active_keys) or 'None'}",
            (x0 + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        y += 22
        click_labels = {
            "left_click": "L",
            "right_click": "R",
            "middle_click": "M",
        }
        click_tags = [
            click_labels.get(name, _display_key(name))
            for name, active in clicks.items()
            if active
        ]
        cv2.putText(
            frame,
            f"Click: {' '.join(click_tags) or 'None'}",
            (x0 + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 255),
            1,
        )
        y += 22

    if pressed is not None and released is not None:
        press_labels = [_display_key(name) for name in pressed]
        release_labels = [_display_key(name) for name in released]
        cv2.putText(
            frame,
            f"Press: {' '.join(press_labels) or 'None'}",
            (x0 + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 255, 80),
            1,
        )
        y += 22
        cv2.putText(
            frame,
            f"Release: {' '.join(release_labels) or 'None'}",
            (x0 + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 160, 255),
            1,
        )
        y += 22

    dx, dy = mouse_delta
    cv2.putText(
        frame,
        f"Mouse: ({dx:+d}, {dy:+d})",
        (x0 + 10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 200, 0),
        1,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize CSV key presses over video frames."
    )
    parser.add_argument(
        "--video",
        help="Path to the video file.",
        default=r'C:\Users\Abhil\Desktop\Github_Projects\VideoAgent\data\greenville\run_20260528_121826.mp4',
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        default=None,
        help="Optional path to CSV. If omitted, auto-matched from video name.",
    )
    parser.add_argument(
        "--label-shift",
        type=int,
        default=None,
        help="Legacy direct label shift. Prefer --prediction-horizon and --action-label-offset.",
    )
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=None,
        help="Direct future frame offset override. Target idx = frame + offset - 1 + action_label_offset.",
    )
    parser.add_argument(
        "--prediction-horizon-offsets",
        default="1,2,3,5,7,10,13,16,20,24",
        help="Comma-separated training horizon frame offsets.",
    )
    parser.add_argument(
        "--command-horizon",
        type=int,
        default=10,
        help="1-based horizon head to visualize when --prediction-horizon is omitted.",
    )
    parser.add_argument(
        "--action-label-offset",
        type=int,
        default=-1,
        help="Same offset used by train.py. Target idx = frame + frame_offset - 1 + label_offset.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = all).",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Write overlay video using FFmpeg/NVENC (default: <video>_keys.mp4).",
    )
    parser.add_argument(
        "--ffmpeg-path",
        default=None,
        help="Optional explicit ffmpeg path.",
    )
    parser.add_argument(
        "--nvenc-codec",
        default="hevc_nvenc",
        help="NVENC codec (e.g. hevc_nvenc, h264_nvenc).",
    )
    parser.add_argument(
        "--cq",
        type=int,
        default=20,
        help="Constant quality for NVENC (lower = higher quality).",
    )
    parser.add_argument(
        "--preset",
        default="p4",
        help="NVENC preset (e.g. p1..p7).",
    )
    args = parser.parse_args()

    video_path = args.video
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    csv_path = args.csv_path or find_csv_for_video(video_path)
    fieldnames, rows = load_csv_rows(csv_path)
    if not rows:
        raise RuntimeError(f"CSV has no rows: {csv_path}")

    key_names, click_names = split_columns(fieldnames)
    if not key_names and not click_names:
        raise RuntimeError(f"No key/click columns found in {csv_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frame_idx = 0
    current_idx = -1
    frame_bgr = None
    writer = None
    pbar = None
    if args.out:
        out_path = args.out
    else:
        base, _ = os.path.splitext(video_path)
        out_path = f"{base}_keys"

    try:
        while True:
            frame_updated = False
            ret, frame_bgr = cap.read()
            if not ret:
                break
            current_idx = frame_idx
            frame_idx += 1
            frame_updated = True

            if frame_bgr is None:
                break

            if writer is None and out_path:
                h, w = frame_bgr.shape[:2]
                writer = FFmpegWriter(
                    out_path=out_path,
                    fps=fps,
                    width=w,
                    height=h,
                    ffmpeg_path=args.ffmpeg_path,
                    use_nvenc_codec=args.nvenc_codec,
                    cq=args.cq,
                    preset=args.preset,
                )
                pbar = tqdm(
                    total=total_frames if total_frames > 0 else None,
                    unit="frame",
                    desc="Writing",
                )

            horizon_offsets = tuple(
                int(part.strip())
                for part in str(args.prediction_horizon_offsets).split(",")
                if part.strip()
            )
            if not horizon_offsets:
                raise ValueError("--prediction-horizon-offsets must contain at least one frame offset.")
            command_idx = max(0, min(int(args.command_horizon) - 1, len(horizon_offsets) - 1))
            prediction_horizon = (
                max(1, int(args.prediction_horizon))
                if args.prediction_horizon is not None
                else int(horizon_offsets[command_idx])
            )
            if args.label_shift is None:
                action_label_offset = int(args.action_label_offset)
                label_idx = current_idx + prediction_horizon - 1 + action_label_offset
            else:
                action_label_offset = int(args.label_shift) - prediction_horizon + 1
                label_idx = current_idx + int(args.label_shift)

            if 0 <= label_idx < len(rows):
                keys, clicks, mouse_delta, label_ts = parse_row(
                    rows[label_idx], key_names, click_names
                )
                if label_idx > 0:
                    prev_keys, prev_clicks, _, _ = parse_row(
                        rows[label_idx - 1], key_names, click_names
                    )
                    all_prev = {**prev_keys, **prev_clicks}
                    all_current = {**keys, **clicks}
                    pressed, released = transition_between(all_prev, all_current)
                else:
                    pressed, released = [], []
            else:
                keys, clicks, mouse_delta, label_ts = None, None, (0, 0), None
                pressed, released = None, None
                label_idx = None

            draw_overlay(
                frame_bgr,
                frame_idx=current_idx,
                total_frames=total_frames,
                fps=fps,
                label_idx=label_idx,
                prediction_horizon=prediction_horizon,
                action_label_offset=action_label_offset,
                keys=keys,
                clicks=clicks,
                pressed=pressed,
                released=released,
                mouse_delta=mouse_delta,
                label_timestamp=label_ts,
            )

            if writer and frame_updated:
                writer.write(frame_bgr)
                if pbar:
                    pbar.update(1)

            if frame_updated and args.max_frames and (current_idx + 1) >= args.max_frames:
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        if pbar:
            pbar.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
