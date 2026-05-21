import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import DrivingVideoPolicy, ModelConfig  # noqa: E402


def find_first_video(data_root: str, video_ext: str) -> str:
    direct = sorted(glob.glob(os.path.join(data_root, f"run_*{video_ext}")))
    if direct:
        return direct[0]

    recursive = sorted(glob.glob(os.path.join(data_root, "**", f"run_*{video_ext}"), recursive=True))
    if recursive:
        return recursive[0]

    raise FileNotFoundError(f"No run_*{video_ext} files found under {data_root!r}.")


def read_first_frame(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read first frame from video: {video_path}")
        return frame
    finally:
        cap.release()


def mask_frame_bgr(frame_bgr: np.ndarray, model: DrivingVideoPolicy, model_size: int) -> tuple[np.ndarray, np.ndarray]:
    resized_bgr = cv2.resize(frame_bgr, (model_size, model_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    frame = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        masked = model._apply_masks(model._normalize_frames(frame))[0]

    masked_rgb = (masked.permute(1, 2, 0).detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
    masked_bgr = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR)
    return resized_bgr, masked_bgr


def build_mask_visual(original_bgr: np.ndarray, masked_bgr: np.ndarray) -> np.ndarray:
    changed = np.any(original_bgr != masked_bgr, axis=2)
    mask = np.zeros_like(original_bgr)
    mask[changed] = (0, 0, 255)

    overlay = original_bgr.copy()
    overlay[changed] = (original_bgr[changed].astype(np.float32) * 0.35 + mask[changed].astype(np.float32) * 0.65).astype(
        np.uint8
    )

    return np.concatenate([original_bgr, mask, overlay, masked_bgr], axis=1)


def default_output_path(video_path: str) -> str:
    base = Path(video_path)
    return str(base.with_name(f"{base.stem}_mask_preview.png"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize the policy input mask from models.py on a video's first frame.")
    parser.add_argument("--video", default=None, help="Video to read. Defaults to the first run_*.mp4 in cfg.data_root.")
    parser.add_argument("--data-root", default=None, help="Data root used when --video is omitted.")
    parser.add_argument("--model-size", type=int, default=None, help="Square resize size before applying the mask.")
    parser.add_argument("--selected-game", default=None, help="Game config name. Defaults to action_space.selected_game.")
    parser.add_argument("--output", default=None, help="Output PNG path. Defaults next to the selected video.")
    parser.add_argument("--show", action="store_true", help="Open an OpenCV preview window after writing the PNG.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg_kwargs = {}
    if args.selected_game is not None:
        cfg_kwargs["selected_game"] = str(args.selected_game)
    if args.data_root is not None:
        cfg_kwargs["data_root"] = str(args.data_root)
    if args.model_size is not None:
        cfg_kwargs["model_size"] = int(args.model_size)

    cfg = ModelConfig(**cfg_kwargs)
    model = DrivingVideoPolicy(cfg)
    model.eval()

    video_path = str(args.video) if args.video is not None else find_first_video(str(cfg.data_root), str(cfg.video_ext))
    frame_bgr = read_first_frame(video_path)
    original_bgr, masked_bgr = mask_frame_bgr(frame_bgr, model, int(cfg.model_size))
    visual = build_mask_visual(original_bgr, masked_bgr)

    output_path = str(args.output) if args.output is not None else default_output_path(video_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if not cv2.imwrite(output_path, visual):
        raise RuntimeError(f"Could not write output image: {output_path}")

    print(f"Video: {video_path}")
    print(f"Output: {output_path}")
    print("Columns: original | mask | overlay | masked model input")

    if args.show:
        cv2.imshow("models.py mask preview", visual)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
