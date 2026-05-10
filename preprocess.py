import os
import glob
import subprocess

# Path to your dataset (WSL-visible)
DATA_ROOT = "C:/Users/Abhil/Desktop/vs code stuff/python/ai/data/flee_the_facility"

# Output resolution (no aspect ratio preserved)
TARGET_W = 512
TARGET_H = 512

# FFmpeg / NVENC settings
CODEC = "hevc_nvenc"   # or "h264_nvenc"
PRESET = "p5"
CQ = "28"
FFMPEG = "ffmpeg"      # FFmpeg should be in PATH


def process_video(in_path: str):
    """Resize to EXACT 512×512 and re-encode with NVENC."""

    base, _ = os.path.splitext(in_path)
    out_path = base + "_512.mp4"

    if os.path.exists(out_path):
        print(f"[SKIP] Already exists: {out_path}")
        return

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
        "-cq", CQ,

        out_path,
    ]

    print("\n[PROCESS]", in_path)
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[DONE]", out_path)


def main():
    # Adjust glob if your naming differs
    video_files = sorted(glob.glob(os.path.join(DATA_ROOT, "run_*.mp4")))

    if not video_files:
        print("No videos found.")
        return

    print(f"Found {len(video_files)} videos\n")

    for v in video_files:
        process_video(v)


if __name__ == "__main__":
    main()
