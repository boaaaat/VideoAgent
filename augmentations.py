from typing import Protocol

import torch
import torch.nn.functional as F


class VideoAugmentConfig(Protocol):
    aug_brightness: float
    aug_contrast: float
    aug_noise_std: float
    aug_gray_prob: float
    aug_translate_frac: float
    aug_scale_frac: float


def _spatial_jitter(frames: torch.Tensor, cfg: VideoAugmentConfig) -> torch.Tensor:
    translate_frac = float(cfg.aug_translate_frac)
    scale_frac = float(cfg.aug_scale_frac)
    if translate_frac <= 0.0 and scale_frac <= 0.0:
        return frames

    b, t, c, h, w = frames.shape
    work = frames.float()

    scale = torch.ones((b,), device=frames.device, dtype=torch.float32)
    if scale_frac > 0.0:
        scale.uniform_(1.0 - scale_frac, 1.0 + scale_frac)

    tx = torch.zeros((b,), device=frames.device, dtype=torch.float32)
    ty = torch.zeros((b,), device=frames.device, dtype=torch.float32)
    if translate_frac > 0.0:
        tx.uniform_(-2.0 * translate_frac, 2.0 * translate_frac)
        ty.uniform_(-2.0 * translate_frac, 2.0 * translate_frac)

    theta = torch.zeros((b, 2, 3), device=frames.device, dtype=torch.float32)
    theta[:, 0, 0] = scale
    theta[:, 1, 1] = scale
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    theta = theta.repeat_interleave(t, dim=0)

    flat = work.reshape(b * t, c, h, w)
    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    jittered = F.grid_sample(flat, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return jittered.reshape(b, t, c, h, w).to(dtype=frames.dtype)


def augment_frames(frames: torch.Tensor, cfg: VideoAugmentConfig) -> torch.Tensor:
    """Video-safe training augmentations. No flips: those would require action remapping."""
    if not frames.is_floating_point():
        return frames
    b = frames.size(0)
    device = frames.device
    out = _spatial_jitter(frames, cfg)

    if cfg.aug_contrast > 0.0:
        contrast = torch.empty((b, 1, 1, 1, 1), device=device, dtype=torch.float32).uniform_(
            1.0 - float(cfg.aug_contrast), 1.0 + float(cfg.aug_contrast)
        ).to(dtype=out.dtype)
        mean = out.mean(dim=(-1, -2), keepdim=True)
        out = (out - mean) * contrast + mean

    if cfg.aug_brightness > 0.0:
        gain = torch.empty((b, 1, 1, 1, 1), device=device, dtype=torch.float32).uniform_(
            1.0 - float(cfg.aug_brightness), 1.0 + float(cfg.aug_brightness)
        ).to(dtype=out.dtype)
        bias = torch.empty((b, 1, 1, 1, 1), device=device, dtype=torch.float32).uniform_(
            -float(cfg.aug_brightness), float(cfg.aug_brightness)
        ).to(dtype=out.dtype)
        out = out * gain + bias

    if cfg.aug_gray_prob > 0.0:
        mask = (torch.rand((b, 1, 1, 1, 1), device=device) < float(cfg.aug_gray_prob)).to(dtype=torch.bool)
        if bool(mask.any().item()):
            gray = out.mean(dim=2, keepdim=True).expand_as(out)
            out = torch.where(mask, gray, out)

    if cfg.aug_noise_std > 0.0:
        noise = torch.randn_like(out, dtype=torch.float32) * float(cfg.aug_noise_std)
        out = out + noise.to(dtype=out.dtype)

    return out.clamp_(0.0, 1.0)
