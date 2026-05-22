import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

import torch
import torch.nn.functional as F


class VideoAugmentConfig(Protocol):
    aug_brightness: float
    aug_contrast: float
    aug_noise_std: float
    aug_gray_prob: float
    aug_translate_frac: float
    aug_scale_frac: float
    aug_edges_crop_prob: float
    aug_edges_crop_min_frac: float
    aug_edges_crop_max_frac: float
    aug_cutout_prob: float
    aug_cutout_min_frac: float
    aug_cutout_max_frac: float
    aug_cutout_count: int


@dataclass
class SimpleVideoAugmentConfig:
    aug_brightness: float = 0.08
    aug_contrast: float = 0.10
    aug_noise_std: float = 0.006
    aug_gray_prob: float = 0.02
    aug_translate_frac: float = 0.02
    aug_scale_frac: float = 0.03
    aug_edges_crop_prob: float = 0.25
    aug_edges_crop_min_frac: float = 0.02
    aug_edges_crop_max_frac: float = 0.06
    aug_cutout_prob: float = 0.20
    aug_cutout_min_frac: float = 0.04
    aug_cutout_max_frac: float = 0.12
    aug_cutout_count: int = 1


AUGMENTATION_PRESETS: Dict[str, Dict[str, float | int]] = {
    "none": {},
    "color": {
        "aug_brightness": 0.08,
        "aug_contrast": 0.10,
        "aug_noise_std": 0.006,
        "aug_gray_prob": 0.02,
    },
    "spatial": {
        "aug_translate_frac": 0.02,
        "aug_scale_frac": 0.03,
    },
    "edges_crop": {
        "aug_edges_crop_prob": 1.0,
        "aug_edges_crop_min_frac": 0.02,
        "aug_edges_crop_max_frac": 0.06,
    },
    "cutout": {
        "aug_cutout_prob": 1.0,
        "aug_cutout_min_frac": 0.04,
        "aug_cutout_max_frac": 0.12,
        "aug_cutout_count": 1,
    },
    "all": {
        "aug_brightness": 0.08,
        "aug_contrast": 0.10,
        "aug_noise_std": 0.006,
        "aug_gray_prob": 0.02,
        "aug_translate_frac": 0.02,
        "aug_scale_frac": 0.03,
        "aug_edges_crop_prob": 0.25,
        "aug_edges_crop_min_frac": 0.02,
        "aug_edges_crop_max_frac": 0.06,
        "aug_cutout_prob": 0.20,
        "aug_cutout_min_frac": 0.04,
        "aug_cutout_max_frac": 0.12,
        "aug_cutout_count": 1,
    },
}


def _sample_time_shape(frames: torch.Tensor, same_over_time: bool) -> tuple[int, int]:
    b, t = frames.shape[:2]
    return b, 1 if same_over_time else t


def _spatial_jitter(frames: torch.Tensor, cfg: VideoAugmentConfig, *, same_over_time: bool) -> torch.Tensor:
    translate_frac = float(cfg.aug_translate_frac)
    scale_frac = float(cfg.aug_scale_frac)
    if translate_frac <= 0.0 and scale_frac <= 0.0:
        return frames

    b, t, c, h, w = frames.shape
    work = frames.float()

    sample_shape = _sample_time_shape(frames, same_over_time)
    scale = torch.ones(sample_shape, device=frames.device, dtype=torch.float32)
    if scale_frac > 0.0:
        scale.uniform_(1.0 - scale_frac, 1.0 + scale_frac)
    scale = scale.expand(b, t)

    tx = torch.zeros(sample_shape, device=frames.device, dtype=torch.float32)
    ty = torch.zeros(sample_shape, device=frames.device, dtype=torch.float32)
    if translate_frac > 0.0:
        tx.uniform_(-2.0 * translate_frac, 2.0 * translate_frac)
        ty.uniform_(-2.0 * translate_frac, 2.0 * translate_frac)
    tx = tx.expand(b, t)
    ty = ty.expand(b, t)

    theta = torch.zeros((b, t, 2, 3), device=frames.device, dtype=torch.float32)
    theta[:, :, 0, 0] = scale
    theta[:, :, 1, 1] = scale
    theta[:, :, 0, 2] = tx
    theta[:, :, 1, 2] = ty
    theta = theta.reshape(b * t, 2, 3)

    flat = work.reshape(b * t, c, h, w)
    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    jittered = F.grid_sample(flat, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return jittered.reshape(b, t, c, h, w).to(dtype=frames.dtype)


def _edges_crop_blackout(frames: torch.Tensor, cfg: VideoAugmentConfig, *, same_over_time: bool) -> torch.Tensor:
    prob = float(cfg.aug_edges_crop_prob)
    min_frac = float(cfg.aug_edges_crop_min_frac)
    max_frac = float(cfg.aug_edges_crop_max_frac)
    if prob <= 0.0 or max_frac <= 0.0:
        return frames

    b, t, c, h, w = frames.shape
    min_frac = min(max(min_frac, 0.0), 0.45)
    max_frac = min(max(max_frac, min_frac), 0.45)
    if max_frac <= 0.0:
        return frames

    sample_shape = _sample_time_shape(frames, same_over_time)
    crop_frac = torch.empty(sample_shape, device=frames.device, dtype=torch.float32).uniform_(min_frac, max_frac)
    edge_w = torch.clamp((crop_frac * float(w)).round().long(), min=1, max=max(1, w // 2))
    xx = torch.arange(w, device=frames.device).view(1, 1, 1, 1, w)
    keep = (xx >= edge_w.view(*sample_shape, 1, 1, 1)) & (xx < (w - edge_w).view(*sample_shape, 1, 1, 1))
    cropped = torch.where(keep, frames, frames.new_zeros(()))

    if prob >= 1.0:
        return cropped
    mask = (torch.rand((*sample_shape, 1, 1, 1), device=frames.device) < prob).to(dtype=torch.bool)
    return torch.where(mask, cropped, frames)


def _cutout(frames: torch.Tensor, cfg: VideoAugmentConfig, *, same_over_time: bool) -> torch.Tensor:
    prob = float(cfg.aug_cutout_prob)
    min_frac = float(cfg.aug_cutout_min_frac)
    max_frac = float(cfg.aug_cutout_max_frac)
    count = int(cfg.aug_cutout_count)
    if prob <= 0.0 or max_frac <= 0.0 or count <= 0:
        return frames

    b, t, c, h, w = frames.shape
    min_frac = min(max(min_frac, 0.0), 1.0)
    max_frac = min(max(max_frac, min_frac), 1.0)
    count = max(1, min(count, 16))
    out = frames.clone()
    yy = torch.arange(h, device=frames.device).view(1, 1, 1, h, 1)
    xx = torch.arange(w, device=frames.device).view(1, 1, 1, 1, w)
    sample_shape = _sample_time_shape(frames, same_over_time)
    edge_x = max(1, int(round(w * 0.15)))
    edge_y = max(1, int(round(h * 0.15)))
    center_x0 = edge_x
    center_x1 = max(center_x0 + 1, w - edge_x)
    center_y0 = edge_y
    center_y1 = max(center_y0 + 1, h - edge_y)

    for _ in range(count):
        active = (torch.rand(sample_shape, device=frames.device) < prob).view(*sample_shape, 1, 1, 1)
        frac = torch.empty(sample_shape, device=frames.device, dtype=torch.float32).uniform_(min_frac, max_frac)
        cut_h = torch.clamp((frac * float(h)).round().long(), min=1, max=h)
        cut_w = torch.clamp((frac * float(w)).round().long(), min=1, max=w)
        band = torch.randint(0, 4, sample_shape, device=frames.device)

        zeros = torch.zeros(sample_shape, device=frames.device, dtype=torch.long)
        full_w = torch.full(sample_shape, w, device=frames.device, dtype=torch.long)
        full_h = torch.full(sample_shape, h, device=frames.device, dtype=torch.long)
        cx0 = torch.full(sample_shape, center_x0, device=frames.device, dtype=torch.long)
        cx1 = torch.full(sample_shape, center_x1, device=frames.device, dtype=torch.long)
        cy0 = torch.full(sample_shape, center_y0, device=frames.device, dtype=torch.long)
        cy1 = torch.full(sample_shape, center_y1, device=frames.device, dtype=torch.long)

        band_x0 = torch.where((band == 2), zeros, torch.where((band == 3), cx1, zeros))
        band_x1 = torch.where((band == 2), cx0, torch.where((band == 3), full_w, full_w))
        band_y0 = torch.where((band == 0), zeros, torch.where((band == 1), cy1, cy0))
        band_y1 = torch.where((band == 0), cy0, torch.where((band == 1), full_h, cy1))

        band_w = (band_x1 - band_x0).clamp(min=1)
        band_h = (band_y1 - band_y0).clamp(min=1)
        cut_w = torch.minimum(cut_w, band_w)
        cut_h = torch.minimum(cut_h, band_h)

        span_x = (band_w - cut_w + 1).clamp(min=1)
        span_y = (band_h - cut_h + 1).clamp(min=1)
        x0 = band_x0 + torch.floor(torch.rand(sample_shape, device=frames.device) * span_x.float()).long()
        y0 = band_y0 + torch.floor(torch.rand(sample_shape, device=frames.device) * span_y.float()).long()
        x1 = x0 + cut_w
        y1 = y0 + cut_h

        hole = (
            active
            & (yy >= y0.view(*sample_shape, 1, 1, 1))
            & (yy < y1.view(*sample_shape, 1, 1, 1))
            & (xx >= x0.view(*sample_shape, 1, 1, 1))
            & (xx < x1.view(*sample_shape, 1, 1, 1))
        )
        out = torch.where(hole, out.new_zeros(()), out)
    return out


def augment_frames(frames: torch.Tensor, cfg: VideoAugmentConfig, *, same_over_time: bool = True) -> torch.Tensor:
    """Video-safe training augmentations. No flips: those would require action remapping."""
    if not frames.is_floating_point():
        return frames
    device = frames.device
    sample_shape = _sample_time_shape(frames, same_over_time)
    out = _spatial_jitter(frames, cfg, same_over_time=same_over_time)
    out = _edges_crop_blackout(out, cfg, same_over_time=same_over_time)

    if cfg.aug_contrast > 0.0:
        contrast = torch.empty((*sample_shape, 1, 1, 1), device=device, dtype=torch.float32).uniform_(
            1.0 - float(cfg.aug_contrast), 1.0 + float(cfg.aug_contrast)
        ).to(dtype=out.dtype)
        mean = out.mean(dim=(-1, -2), keepdim=True)
        out = (out - mean) * contrast + mean

    if cfg.aug_brightness > 0.0:
        gain = torch.empty((*sample_shape, 1, 1, 1), device=device, dtype=torch.float32).uniform_(
            1.0 - float(cfg.aug_brightness), 1.0 + float(cfg.aug_brightness)
        ).to(dtype=out.dtype)
        bias = torch.empty((*sample_shape, 1, 1, 1), device=device, dtype=torch.float32).uniform_(
            -float(cfg.aug_brightness), float(cfg.aug_brightness)
        ).to(dtype=out.dtype)
        out = out * gain + bias

    if cfg.aug_gray_prob > 0.0:
        mask = (torch.rand((*sample_shape, 1, 1, 1), device=device) < float(cfg.aug_gray_prob)).to(dtype=torch.bool)
        gray = out.mean(dim=2, keepdim=True).expand_as(out)
        out = torch.where(mask, gray, out)

    if cfg.aug_noise_std > 0.0:
        noise_shape = (out.size(0), 1, *out.shape[2:]) if same_over_time else out.shape
        noise = torch.randn(noise_shape, device=device, dtype=torch.float32) * float(cfg.aug_noise_std)
        out = out + noise.to(dtype=out.dtype)

    out = _cutout(out, cfg, same_over_time=same_over_time)
    return out.clamp_(0.0, 1.0)


def build_preset_config(preset: str, **overrides: Optional[float | int]) -> SimpleVideoAugmentConfig:
    if preset not in AUGMENTATION_PRESETS:
        raise ValueError(f"Unknown augmentation preset {preset!r}. Choose one of: {', '.join(AUGMENTATION_PRESETS)}")
    values = dict(AUGMENTATION_PRESETS[preset])
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    return SimpleVideoAugmentConfig(**values)


def _read_sample_frames(video_path: str, frame_start: int, num_frames: int, resize_size: int) -> tuple[List[object], torch.Tensor]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        if frame_start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_start))
        bgr_frames: List[object] = []
        rgb_tensors: List[torch.Tensor] = []
        for _ in range(max(1, int(num_frames))):
            ret, frame_bgr = cap.read()
            if not ret:
                break
            resized_bgr = cv2.resize(
                frame_bgr,
                (int(resize_size), int(resize_size)),
                interpolation=cv2.INTER_AREA,
            )
            frame_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
            bgr_frames.append(resized_bgr)
            rgb_tensors.append(torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0)
    finally:
        cap.release()
    if not rgb_tensors:
        raise RuntimeError(f"No frames could be read from {video_path}")
    return bgr_frames, torch.stack(rgb_tensors, dim=0).unsqueeze(0)


def _tensor_frames_to_bgr(frames: torch.Tensor) -> List[object]:
    import cv2

    frames = frames.detach().cpu().float().clamp(0.0, 1.0)
    if frames.dim() != 5 or frames.size(0) != 1 or frames.size(2) != 3:
        raise ValueError(f"Expected frames with shape [1,T,3,H,W], got {tuple(frames.shape)}.")
    bgr_frames: List[object] = []
    for idx in range(frames.size(1)):
        rgb = (frames[0, idx].permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
        bgr_frames.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return bgr_frames


def _draw_label(image: object, text: str) -> object:
    import cv2

    cv2.rectangle(image, (0, 0), (image.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(image, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def write_augmentation_sheet(
    original_bgr: List[object],
    augmented_bgr: List[object],
    out_path: str,
    *,
    title: str,
) -> None:
    import cv2
    import numpy as np

    columns = min(len(original_bgr), len(augmented_bgr))
    if columns <= 0:
        raise ValueError("Cannot write an empty augmentation sheet.")
    top = [_draw_label(original_bgr[idx].copy(), "original") for idx in range(columns)]
    bottom = [_draw_label(augmented_bgr[idx].copy(), title) for idx in range(columns)]
    sheet = np.vstack([np.hstack(top), np.hstack(bottom)])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if not cv2.imwrite(out_path, sheet):
        raise RuntimeError(f"Failed to write augmentation sheet: {out_path}")


def _default_sheet_path(video_path: str, preset: str) -> str:
    base, _ = os.path.splitext(video_path)
    return f"{base}_aug_{preset}.jpg"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize training video augmentations on real video frames.")
    parser.add_argument("--video", default=r'C:\Users\Abhil\Desktop\Github_Projects\VideoAgent\data\greenville\run_20260520_214152.mp4', help="Input video path.")
    parser.add_argument("--out", default=None, help="Output image path. Defaults next to the input video.")
    parser.add_argument("--preset", choices=sorted(AUGMENTATION_PRESETS), default="edges_crop")
    parser.add_argument("--list-augmentations", action="store_true", help="Print available presets and exit.")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--aug-brightness", type=float, default=None)
    parser.add_argument("--aug-contrast", type=float, default=None)
    parser.add_argument("--aug-noise-std", type=float, default=None)
    parser.add_argument("--aug-gray-prob", type=float, default=None)
    parser.add_argument("--aug-translate-frac", type=float, default=None)
    parser.add_argument("--aug-scale-frac", type=float, default=None)
    parser.add_argument("--aug-edges-crop-prob", type=float, default=None)
    parser.add_argument("--aug-edges-crop-min-frac", type=float, default=None)
    parser.add_argument("--aug-edges-crop-max-frac", type=float, default=None)
    parser.add_argument("--aug-cutout-prob", type=float, default=None)
    parser.add_argument("--aug-cutout-min-frac", type=float, default=None)
    parser.add_argument("--aug-cutout-max-frac", type=float, default=None)
    parser.add_argument("--aug-cutout-count", type=int, default=None)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.list_augmentations:
        for name in sorted(AUGMENTATION_PRESETS):
            print(name)
        return 0
    if not args.video:
        parser.error("--video is required unless --list-augmentations is used.")

    cfg = build_preset_config(
        args.preset,
        aug_brightness=args.aug_brightness,
        aug_contrast=args.aug_contrast,
        aug_noise_std=args.aug_noise_std,
        aug_gray_prob=args.aug_gray_prob,
        aug_translate_frac=args.aug_translate_frac,
        aug_scale_frac=args.aug_scale_frac,
        aug_edges_crop_prob=args.aug_edges_crop_prob,
        aug_edges_crop_min_frac=args.aug_edges_crop_min_frac,
        aug_edges_crop_max_frac=args.aug_edges_crop_max_frac,
        aug_cutout_prob=args.aug_cutout_prob,
        aug_cutout_min_frac=args.aug_cutout_min_frac,
        aug_cutout_max_frac=args.aug_cutout_max_frac,
        aug_cutout_count=args.aug_cutout_count,
    )
    torch.manual_seed(int(args.seed))
    original_bgr, frames = _read_sample_frames(args.video, args.frame_start, args.num_frames, args.resize_size)
    augmented = augment_frames(frames, cfg, same_over_time=False)
    augmented_bgr = _tensor_frames_to_bgr(augmented)
    out_path = args.out or _default_sheet_path(args.video, args.preset)
    write_augmentation_sheet(original_bgr, augmented_bgr, out_path, title=f"augmented: {args.preset}")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
