import glob
import os
import subprocess
from pathlib import Path

# Path to your dataset.
DATA_ROOT = str(Path(__file__).resolve().parent / "data" / "greenville_test")

# Output resolution (no aspect ratio preserved)
TARGET_W = 256
TARGET_H = 256

# FFmpeg / NVENC settings. H.264 with short GOPs is easier for DALI random
# window reads than long-GOP HEVC.
CODEC = "h264_nvenc"
PRESET = "p5"
CQ = "23"
GOP_SIZE = "2"
TEMP_SUFFIX = "_temp"
FFMPEG = "ffmpeg"      # FFmpeg should be in PATH


def temp_video_path(in_path: str) -> str:
    path = Path(in_path)
    return str(path.with_name(f"{path.stem}{TEMP_SUFFIX}{path.suffix}"))


def process_video(in_path: str) -> None:
    """Resize to EXACT 512x512 and re-encode in place with DALI-friendly GOPs."""

    path = Path(in_path)
    if path.stem.endswith(TEMP_SUFFIX):
        print(f"[SKIP] Temp file: {in_path}")
        return

    temp_path = temp_video_path(in_path)

    # This works on almost any FFmpeg build:
    # -hwaccel cuda is optional; remove it if it errors on your setup.
    cmd = [
        FFMPEG,
        "-y",

        # OPTIONAL: try GPU-accelerated decode. If this causes errors, delete the next 2 args.
        "-hwaccel", "cuda",

        # Input file
        "-i", in_path,

        # Force resize to EXACT 512x512 (no aspect ratio preservation)
        "-vf", f"scale={TARGET_W}:{TARGET_H}",

        # NVENC GPU encoder
        "-c:v", CODEC,
        "-preset", PRESET,
        "-rc", "constqp",
        "-qp", CQ,
        "-pix_fmt", "yuv420p",

        # Short, simple GOPs make DALI frame-window reads cheaper and more robust.
        "-g", GOP_SIZE,
        "-keyint_min", GOP_SIZE,
        "-bf", "0",
        "-forced-idr", "1",

        # Dataset clips do not need audio, and faststart keeps MP4 metadata up front.
        "-an",
        "-movflags", "+faststart",

        temp_path,
    ]

    print("\n[PROCESS]", in_path)
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    os.replace(temp_path, in_path)
    print("[DONE]", in_path)


def main() -> None:
    # Adjust glob if your naming differs
    video_files = [
        path
        for path in sorted(glob.glob(os.path.join(DATA_ROOT, "run_*.mp4")))
        if not Path(path).stem.endswith(TEMP_SUFFIX)
    ]

    if not video_files:
        print("No videos found.")
        return

    print(f"Found {len(video_files)} videos\n")

    for v in video_files:
        process_video(v)


if __name__ == "__main__":
    main()
