import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from action_space import (
    game_data_root,
    get_key_names,
    get_mouse_button_names,
    normalize_game_name,
    selected_game as ACTION_SELECTED_GAME,
)


INVERSE_VISUAL_ENCODER_NAME = "simple_cnn_transformer_v1"


def validate_inverse_visual_encoder_name(name: str) -> str:
    name = str(name).strip()
    if name != INVERSE_VISUAL_ENCODER_NAME:
        raise ValueError(
            f"Unsupported inverse-dynamics architecture {name!r}. "
            f"Expected {INVERSE_VISUAL_ENCODER_NAME!r}."
        )
    return name


def inverse_checkpoint_family_mismatch_reason(config_dict: object) -> Optional[str]:
    if not isinstance(config_dict, dict):
        return "Checkpoint is missing inverse-dynamics config metadata."
    checkpoint_encoder = str(config_dict.get("visual_encoder_name", "")).strip()
    if checkpoint_encoder != INVERSE_VISUAL_ENCODER_NAME:
        return (
            "Inverse checkpoint architecture mismatch: "
            f"ckpt={checkpoint_encoder!r} current={INVERSE_VISUAL_ENCODER_NAME!r}."
        )
    return None


def center_window_bounds(total_len: int, output_len: int) -> Tuple[int, int]:
    total_len = int(total_len)
    output_len = int(output_len)
    if total_len <= 0:
        raise ValueError("total_len must be positive.")
    if output_len <= 0 or output_len > total_len:
        raise ValueError(f"output_len must be in [1, {total_len}], got {output_len}.")
    margin = total_len - output_len
    if margin % 2 != 0:
        raise ValueError(
            "Centered output requires seq_len - output_seq_len to be even, "
            f"got seq_len={total_len}, output_seq_len={output_len}."
        )
    start = margin // 2
    return start, start + output_len


@dataclass
class InverseDynamicsConfig:
    selected_game: str = ACTION_SELECTED_GAME
    data_root: Optional[str] = None
    video_ext: str = ".mp4"
    csv_ext: str = ".csv"

    model_size: int = 256
    seq_len: int = 24
    output_seq_len: int = 8
    train_seq_stride: int = 8
    val_seq_stride: int = 8
    prediction_dt: float = 1.0 / 20.0

    key_names: Optional[List[str]] = None
    mouse_button_names: Optional[List[str]] = None
    scroll_action_names: Tuple[str, ...] = ("scroll_up", "scroll_down")
    binary_mouse_button_names: Optional[List[str]] = None

    visual_encoder_name: str = INVERSE_VISUAL_ENCODER_NAME
    d_model: int = 256
    cnn_width: int = 48
    cnn_depth: int = 2
    transformer_layers: int = 4
    transformer_heads: int = 8
    dropout: float = 0.1

    button_state_threshold: float = 0.5
    mouse_active_threshold: float = 0.5
    mouse_active_epsilon: float = 2.0
    mouse_delta_scales: Optional[Sequence[float]] = None
    scroll_delta_scales: Optional[Sequence[float]] = None

    num_bin: int = 0
    num_scroll: int = 0

    def __post_init__(self) -> None:
        self.selected_game = normalize_game_name(self.selected_game)
        if self.data_root is None:
            self.data_root = game_data_root(self.selected_game)

        self.model_size = max(1, int(self.model_size))
        self.seq_len = max(1, int(self.seq_len))
        self.output_seq_len = max(1, int(self.output_seq_len))
        self.train_seq_stride = max(1, int(self.train_seq_stride))
        self.val_seq_stride = max(1, int(self.val_seq_stride))
        self.prediction_dt = max(1.0 / 240.0, float(self.prediction_dt))
        center_window_bounds(self.seq_len, self.output_seq_len)

        self.visual_encoder_name = validate_inverse_visual_encoder_name(self.visual_encoder_name)
        self.d_model = max(16, int(self.d_model))
        self.cnn_width = max(8, int(self.cnn_width))
        self.cnn_depth = max(1, int(self.cnn_depth))
        self.transformer_layers = max(1, int(self.transformer_layers))
        self.transformer_heads = max(1, int(self.transformer_heads))
        if self.d_model % self.transformer_heads != 0:
            raise ValueError(
                f"d_model must be divisible by transformer_heads, got "
                f"d_model={self.d_model}, transformer_heads={self.transformer_heads}."
            )
        self.dropout = float(min(max(self.dropout, 0.0), 0.9))
        self.button_state_threshold = float(min(max(self.button_state_threshold, 0.0), 1.0))
        self.mouse_active_threshold = float(min(max(self.mouse_active_threshold, 0.0), 1.0))
        self.mouse_active_epsilon = max(0.0, float(self.mouse_active_epsilon))

        if self.key_names is None:
            self.key_names = get_key_names(self.selected_game)
        else:
            self.key_names = [str(name) for name in self.key_names]

        if self.mouse_button_names is None:
            self.mouse_button_names = get_mouse_button_names(self.selected_game)
        else:
            self.mouse_button_names = [str(name) for name in self.mouse_button_names]

        configured_scroll = set(str(name) for name in self.scroll_action_names)
        self.scroll_action_names = tuple(name for name in self.mouse_button_names if name in configured_scroll)
        self.binary_mouse_button_names = [
            name for name in self.mouse_button_names if name not in set(self.scroll_action_names)
        ]
        self.num_bin = len(self.key_names) + len(self.binary_mouse_button_names)
        self.num_scroll = len(self.scroll_action_names)

        if self.mouse_delta_scales is None:
            self.mouse_delta_scales = (1.0, 1.0)
        else:
            mouse_scales = tuple(float(value) for value in self.mouse_delta_scales)
            if len(mouse_scales) != 2:
                raise ValueError("mouse_delta_scales must contain exactly two values.")
            if not all(math.isfinite(value) and value > 0.0 for value in mouse_scales):
                raise ValueError(f"mouse_delta_scales must be finite and positive, got {mouse_scales!r}.")
            self.mouse_delta_scales = mouse_scales

        if self.scroll_delta_scales is None:
            self.scroll_delta_scales = tuple(1.0 for _ in range(self.num_scroll))
        else:
            scroll_scales = tuple(float(value) for value in self.scroll_delta_scales)
            if len(scroll_scales) != self.num_scroll:
                raise ValueError(
                    f"scroll_delta_scales must contain {self.num_scroll} values, got {len(scroll_scales)}."
                )
            if not all(math.isfinite(value) and value > 0.0 for value in scroll_scales):
                raise ValueError(f"scroll_delta_scales must be finite and positive, got {scroll_scales!r}.")
            self.scroll_delta_scales = scroll_scales


@dataclass
class InverseDynamicsOutput:
    button_logits: torch.Tensor
    button_state: torch.Tensor
    mouse_active_logits: torch.Tensor
    mouse_delta: torch.Tensor
    scroll_delta: torch.Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed.to(dtype=dtype) * self.weight.to(dtype=dtype))


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.norm(self.conv(x)), inplace=True)


class DepthwiseResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
        self.pw2 = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.drop = nn.Dropout2d(float(dropout)) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = F.silu(self.norm(y), inplace=True)
        y = F.silu(self.pw1(y), inplace=True)
        y = self.drop(self.pw2(y))
        return x + y


class FrameMotionEncoder(nn.Module):
    def __init__(self, cfg: InverseDynamicsConfig):
        super().__init__()
        widths = [cfg.cnn_width, cfg.cnn_width * 2, cfg.cnn_width * 4, cfg.cnn_width * 6]
        layers: List[nn.Module] = [ConvNormAct(6, widths[0], stride=2)]
        in_channels = widths[0]
        for stage_idx, out_channels in enumerate(widths):
            if stage_idx > 0:
                layers.append(ConvNormAct(in_channels, out_channels, stride=2))
                in_channels = out_channels
            for _ in range(cfg.cnn_depth):
                layers.append(DepthwiseResidualBlock(in_channels, cfg.dropout * 0.25))
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.norm = RMSNorm(cfg.d_model)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = frames.shape
        if c != 3:
            raise ValueError(f"Expected RGB frames with shape [B,T,3,H,W], got {tuple(frames.shape)}.")
        if frames.is_cuda and torch.is_autocast_enabled("cuda"):
            target_dtype = torch.get_autocast_dtype("cuda")
        else:
            target_dtype = torch.float32
        was_uint8 = frames.dtype == torch.uint8
        frames = frames.to(dtype=target_dtype)
        if was_uint8:
            frames = frames * (1.0 / 255.0)
        frames = (frames * 2.0) - 1.0
        prev = torch.cat([frames[:, :1], frames[:, :-1]], dim=1)
        motion = frames - prev
        x = torch.cat([frames, motion], dim=2).reshape(b * t, 6, h, w)
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        x = self.net(x)
        x = self.pool(x)
        x = self.proj(x).view(b, t, -1)
        return self.norm(x)


class InverseDynamicsModel(nn.Module):
    def __init__(self, cfg: Optional[InverseDynamicsConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else InverseDynamicsConfig()
        validate_inverse_visual_encoder_name(self.cfg.visual_encoder_name)

        self.frame_encoder = FrameMotionEncoder(self.cfg)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.cfg.seq_len, self.cfg.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.d_model,
            nhead=self.cfg.transformer_heads,
            dim_feedforward=self.cfg.d_model * 4,
            dropout=self.cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.transformer_layers)
        self.final_norm = RMSNorm(self.cfg.d_model)

        self.button_head = nn.Linear(self.cfg.d_model, self.cfg.num_bin)
        self.mouse_active_head = nn.Linear(self.cfg.d_model, 1)
        self.mouse_delta_head = nn.Linear(self.cfg.d_model, 2)
        self.scroll_delta_head = nn.Linear(self.cfg.d_model, self.cfg.num_scroll) if self.cfg.num_scroll > 0 else None

        self.register_buffer(
            "mouse_delta_scale",
            torch.tensor(list(self.cfg.mouse_delta_scales), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "scroll_delta_scale",
            torch.tensor(list(self.cfg.scroll_delta_scales), dtype=torch.float32),
            persistent=True,
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.constant_(self.button_head.bias, -1.0)
        nn.init.constant_(self.mouse_active_head.bias, 0.0)
        nn.init.zeros_(self.mouse_delta_head.bias)
        if self.scroll_delta_head is not None:
            nn.init.constant_(self.scroll_delta_head.bias, -4.0)

    def parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        params = [param for param in self.parameters() if param.requires_grad]
        return {"model": params, "controller": params}

    def _scale_like(self, scale: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        scale = scale.to(device=x.device, dtype=x.dtype)
        while scale.dim() < x.dim():
            scale = scale.unsqueeze(0)
        return scale

    def forward(self, frames: torch.Tensor, *, dt: Optional[torch.Tensor] = None) -> InverseDynamicsOutput:
        del dt
        if frames.dim() == 4:
            frames = frames.unsqueeze(1)
        if frames.dim() != 5:
            raise ValueError(f"Expected frames with shape [B,T,C,H,W], got {tuple(frames.shape)}.")

        b, t, _, _, _ = frames.shape
        if t > self.cfg.seq_len:
            raise ValueError(f"Expected at most {self.cfg.seq_len} frames, got {t}.")

        x = self.frame_encoder(frames)
        x = x + self.pos_embed[:, :t].to(device=x.device, dtype=x.dtype)
        x = self.temporal(x)
        x = self.final_norm(x)

        output_len = min(int(self.cfg.output_seq_len), t)
        output_start, output_end = center_window_bounds(t, output_len)
        h = x[:, output_start:output_end]

        button_logits = self.button_head(h)
        button_state = torch.sigmoid(button_logits.float()).to(dtype=button_logits.dtype)
        mouse_active_logits = self.mouse_active_head(h)
        mouse_delta_norm = self.mouse_delta_head(h)
        mouse_delta = mouse_delta_norm * self._scale_like(self.mouse_delta_scale, mouse_delta_norm)

        if self.scroll_delta_head is None:
            scroll_delta = h.new_zeros((b, output_len, 0))
        else:
            scroll_norm = F.softplus(self.scroll_delta_head(h))
            scroll_delta = scroll_norm * self._scale_like(self.scroll_delta_scale, scroll_norm)

        return InverseDynamicsOutput(
            button_logits=button_logits,
            button_state=button_state,
            mouse_active_logits=mouse_active_logits,
            mouse_delta=mouse_delta,
            scroll_delta=scroll_delta,
        )


def initialize_inverse_lazy_layers(
    model: InverseDynamicsModel,
    cfg: InverseDynamicsConfig,
    device: torch.device,
) -> None:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy_frames = torch.zeros(1, cfg.seq_len, 3, cfg.model_size, cfg.model_size, device=device)
        _ = model(dummy_frames)
    model.train(was_training)


if __name__ == "__main__":
    cfg = InverseDynamicsConfig(model_size=128)
    model = InverseDynamicsModel(cfg)
    x = torch.rand(2, cfg.seq_len, 3, cfg.model_size, cfg.model_size)
    out = model(x)
    print("button_logits", tuple(out.button_logits.shape))
    print("button_state", tuple(out.button_state.shape))
    print("mouse_active_logits", tuple(out.mouse_active_logits.shape))
    print("mouse_delta", tuple(out.mouse_delta.shape))
    print("scroll_delta", tuple(out.scroll_delta.shape))
