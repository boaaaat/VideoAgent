import math
from dataclasses import dataclass, field
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

POLICY_MODEL_FAMILY = "policy_spatial_temporal_v2"


@dataclass
class ModelConfig:
    selected_game: str = ACTION_SELECTED_GAME
    data_root: Optional[str] = None
    video_ext: str = ".mp4"
    csv_ext: str = ".csv"

    policy_model_family: str = POLICY_MODEL_FAMILY

    model_size: int = 256
    seq_len: int = 128
    train_seq_stride: int = 32
    val_seq_stride: int = 128
    prediction_dt: float = 1.0 / 20.0
    prediction_horizon: int = 8

    key_names: Optional[List[str]] = None
    mouse_button_names: Optional[List[str]] = None

    d_model: int = 256
    temporal_heads: int = 8
    temporal_layers: int = 6
    spatial_heads: int = 8
    spatial_tokens: int = 32
    dropout: float = 0.05
    encode_chunk_size: int = 8
    max_context: int = 128

    button_state_threshold: float = 0.5
    transition_threshold: float = 0.5
    mouse_active_threshold: float = 0.5
    mouse_velocity_scales: Optional[Sequence[float]] = None

    # Compatibility fields used by older runtime/visualization helpers.
    token_count: int = 32
    local_context_frames: int = 32
    activation_checkpointing: bool = False
    backbone_activation_checkpointing: bool = False
    temporal_activation_checkpointing: bool = False
    pretrained_backbone: bool = False
    require_pretrained_backbone: bool = False
    backbone_variant: str = "custom"
    future_horizons: Tuple[int, ...] = field(default_factory=tuple)

    num_bin: int = 0
    prev_action_dim: int = 0

    def __post_init__(self) -> None:
        self.selected_game = normalize_game_name(self.selected_game)
        if self.data_root is None:
            self.data_root = game_data_root(self.selected_game)
        self.policy_model_family = validate_policy_model_family(self.policy_model_family)

        self.model_size = max(32, int(self.model_size))
        self.seq_len = max(1, int(self.seq_len))
        self.train_seq_stride = max(1, int(self.train_seq_stride))
        self.val_seq_stride = max(1, int(self.val_seq_stride))
        self.prediction_dt = max(1.0 / 240.0, float(self.prediction_dt))
        self.prediction_horizon = max(1, int(self.prediction_horizon))

        self.d_model = max(64, int(self.d_model))
        self.temporal_heads = max(1, int(self.temporal_heads))
        self.temporal_layers = max(1, int(self.temporal_layers))
        self.spatial_heads = max(1, int(self.spatial_heads))
        self.spatial_tokens = max(1, int(self.spatial_tokens))
        self.token_count = self.spatial_tokens
        self.dropout = float(min(max(self.dropout, 0.0), 0.9))
        self.encode_chunk_size = max(1, int(self.encode_chunk_size))
        self.max_context = max(1, int(self.max_context))
        self.local_context_frames = max(1, int(self.local_context_frames))

        if self.d_model % self.temporal_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by temporal_heads={self.temporal_heads}.")
        if self.d_model % self.spatial_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by spatial_heads={self.spatial_heads}.")

        if self.key_names is None:
            self.key_names = get_key_names(self.selected_game)
        else:
            self.key_names = list(self.key_names)

        if self.mouse_button_names is None:
            self.mouse_button_names = get_mouse_button_names(self.selected_game)
        else:
            self.mouse_button_names = list(self.mouse_button_names)

        if self.mouse_velocity_scales is None:
            self.mouse_velocity_scales = (1.0, 1.0)
        else:
            scales = tuple(float(x) for x in self.mouse_velocity_scales)
            if len(scales) != 2:
                raise ValueError("mouse_velocity_scales must contain exactly 2 values.")
            self.mouse_velocity_scales = scales

        self.future_horizons = tuple(range(1, self.prediction_horizon + 1))
        self.num_bin = len(self.key_names) + len(self.mouse_button_names)
        self.prev_action_dim = self.num_bin + 2


@dataclass
class PolicyOutput:
    button_logits: torch.Tensor
    press_logits: torch.Tensor
    release_logits: torch.Tensor
    mouse_active_logits: torch.Tensor
    mouse_delta: torch.Tensor
    horizon_button_logits: torch.Tensor
    horizon_mouse_active_logits: torch.Tensor
    horizon_mouse_delta: torch.Tensor
    future_button_logits: Dict[int, torch.Tensor] = field(default_factory=dict)
    future_mouse_active_logits: Dict[int, torch.Tensor] = field(default_factory=dict)
    future_mouse_mu: Dict[int, torch.Tensor] = field(default_factory=dict)
    future_mouse_log_b: Dict[int, torch.Tensor] = field(default_factory=dict)
    mouse_mu: Optional[torch.Tensor] = None
    mouse_log_b: Optional[torch.Tensor] = None


@dataclass
class TemporalState:
    cached_summaries: Optional[torch.Tensor] = None
    cached_spatial_tokens: Optional[torch.Tensor] = None
    prev_frame: Optional[torch.Tensor] = None
    prev_button_state: Optional[torch.Tensor] = None
    prev_mouse_action: Optional[torch.Tensor] = None
    steps: int = 0
    # Kept so old checkpoints/runtime code that references these fields does not fail.
    cached_keys: Optional[List[Optional[torch.Tensor]]] = None
    cached_values: Optional[List[Optional[torch.Tensor]]] = None
    cached_inputs: Optional[torch.Tensor] = None
    prev_visual_embedding: Optional[torch.Tensor] = None


def validate_policy_model_family(name: str) -> str:
    name = str(name).strip()
    if name != POLICY_MODEL_FAMILY:
        raise ValueError(
            f"Unsupported policy model family {name!r}. Expected {POLICY_MODEL_FAMILY!r}; "
            "start a new policy training run."
        )
    return name


def policy_checkpoint_family_mismatch_reason(config_dict: object) -> Optional[str]:
    if not isinstance(config_dict, dict):
        return (
            f"Legacy policy checkpoint detected: missing config. Expected "
            f"policy_model_family={POLICY_MODEL_FAMILY!r}; retrain from scratch."
        )
    family = config_dict.get("policy_model_family")
    if family is None:
        return (
            f"Legacy policy checkpoint detected: missing policy_model_family. Expected "
            f"{POLICY_MODEL_FAMILY!r}; retrain from scratch."
        )
    family = str(family).strip()
    if family != POLICY_MODEL_FAMILY:
        return (
            f"Policy checkpoint family mismatch: ckpt={family!r}, expected={POLICY_MODEL_FAMILY!r}; "
            "retrain from scratch."
        )
    return None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        x_norm = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x_norm.to(dtype=dtype) * self.weight.to(dtype=dtype)


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.norm(self.conv(x)), inplace=True)


class SpatialResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False)
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.pw1 = nn.Conv2d(channels, channels * 4, kernel_size=1)
        self.pw2 = nn.Conv2d(channels * 4, channels, kernel_size=1)
        self.drop = nn.Dropout2d(float(dropout)) if dropout > 0.0 else nn.Identity()
        self.gamma = nn.Parameter(torch.ones(channels) * 1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = F.silu(self.norm(y), inplace=True)
        y = self.pw2(F.silu(self.pw1(y), inplace=True))
        y = self.drop(y)
        return x + y * self.gamma.to(dtype=y.dtype).view(1, -1, 1, 1)


class SpatialGridEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        widths = [64, 128, 192, cfg.d_model]
        layers: List[nn.Module] = []
        in_channels = 6
        for stage_idx, out_channels in enumerate(widths):
            layers.append(ConvNormAct(in_channels, out_channels, stride=2))
            layers.append(SpatialResidualBlock(out_channels, cfg.dropout * 0.25))
            layers.append(SpatialResidualBlock(out_channels, cfg.dropout * 0.25))
            in_channels = out_channels
        self.net = nn.Sequential(*layers)
        grid = max(1, cfg.model_size // 16)
        self.grid_size = grid
        self.pos_embed = nn.Parameter(torch.zeros(1, grid * grid, cfg.d_model))
        self.pre_attn = RMSNorm(cfg.d_model)
        self.spatial_attn = nn.MultiheadAttention(
            cfg.d_model,
            cfg.spatial_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.post_attn = RMSNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 4),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * 4, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def _pos_for(self, token_count: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if token_count == self.pos_embed.size(1):
            return self.pos_embed.to(device=device, dtype=dtype)
        src_size = int(math.sqrt(self.pos_embed.size(1)))
        dst_size = int(math.sqrt(token_count))
        pos = self.pos_embed.transpose(1, 2).reshape(1, -1, src_size, src_size)
        pos = F.interpolate(pos.float(), size=(dst_size, dst_size), mode="bicubic", align_corners=False)
        return pos.reshape(1, -1, token_count).transpose(1, 2).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        tokens = tokens + self._pos_for(h * w, tokens.dtype, tokens.device)
        attn_in = self.pre_attn(tokens)
        attn_out, _ = self.spatial_attn(attn_in, attn_in, attn_in, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.ff(self.post_attn(tokens))
        return tokens


class SpatialTokenSampler(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, cfg.spatial_tokens, cfg.d_model) * 0.02)
        self.query_norm = RMSNorm(cfg.d_model)
        self.grid_norm = RMSNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model,
            cfg.spatial_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.out_norm = RMSNorm(cfg.d_model)

    def forward(self, grid_tokens: torch.Tensor) -> torch.Tensor:
        query = self.query.expand(grid_tokens.size(0), -1, -1).to(dtype=grid_tokens.dtype)
        out, _ = self.attn(self.query_norm(query), self.grid_norm(grid_tokens), self.grid_norm(grid_tokens), need_weights=False)
        return self.out_norm(out + query)


class CausalTemporalEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.max_context, cfg.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.temporal_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.net = nn.TransformerEncoder(layer, num_layers=cfg.temporal_layers)
        self.norm = RMSNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        if t > self.pos_embed.size(1):
            raise ValueError(f"Temporal length {t} exceeds max_context={self.pos_embed.size(1)}.")
        x = x + self.pos_embed[:, :t].to(device=x.device, dtype=x.dtype)
        mask = torch.full((t, t), float("-inf"), device=x.device, dtype=torch.float32)
        mask = torch.triu(mask, diagonal=1)
        x = self.net(x, mask=mask)
        return self.norm(x)


class ActionDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.query_norm = RMSNorm(cfg.d_model)
        self.token_norm = RMSNorm(cfg.d_model)
        self.cross_attn = nn.MultiheadAttention(
            cfg.d_model,
            cfg.spatial_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.head_norm = RMSNorm(cfg.d_model)
        self.head_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
        )
        self.button_head = nn.Linear(cfg.d_model, cfg.prediction_horizon * cfg.num_bin)
        self.mouse_active_head = nn.Linear(cfg.d_model, cfg.prediction_horizon)
        self.mouse_delta_head = nn.Linear(cfg.d_model, cfg.prediction_horizon * 2)
        self.press_head = nn.Linear(cfg.d_model, cfg.num_bin)
        self.release_head = nn.Linear(cfg.d_model, cfg.num_bin)

        nn.init.constant_(self.button_head.bias, -1.0)
        nn.init.constant_(self.mouse_active_head.bias, -1.0)
        nn.init.zeros_(self.mouse_delta_head.bias)
        nn.init.constant_(self.press_head.bias, -3.0)
        nn.init.constant_(self.release_head.bias, -3.0)

    def forward(self, temporal: torch.Tensor, spatial_tokens: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        b, t, k, d = spatial_tokens.shape
        query = self.query_norm(temporal.reshape(b * t, 1, d))
        kv = self.token_norm(spatial_tokens.reshape(b * t, k, d))
        cross, _ = self.cross_attn(query, kv, kv, need_weights=False)
        h = temporal.reshape(b * t, d) + cross.squeeze(1)
        h = self.head_mlp(self.head_norm(h)).view(b, t, d)

        button_logits = self.button_head(h).view(b, t, self.cfg.prediction_horizon, self.cfg.num_bin)
        mouse_active_logits = self.mouse_active_head(h).view(b, t, self.cfg.prediction_horizon, 1)
        mouse_delta_norm = self.mouse_delta_head(h).view(b, t, self.cfg.prediction_horizon, 2)
        press_logits = self.press_head(h)
        release_logits = self.release_head(h)
        return button_logits, mouse_active_logits, mouse_delta_norm, press_logits, release_logits


def transition_button_state(
    prev_state: torch.Tensor,
    press_logits: torch.Tensor,
    release_logits: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    press_prob = torch.sigmoid(press_logits)
    release_prob = torch.sigmoid(release_logits)
    press_on = press_prob >= threshold
    release_on = release_prob >= threshold
    next_state = prev_state
    next_state = torch.where(press_on & ~release_on, torch.ones_like(next_state), next_state)
    next_state = torch.where(release_on & ~press_on, torch.zeros_like(next_state), next_state)
    conflict = press_on & release_on
    prefer_press = press_prob >= release_prob
    next_state = torch.where(conflict & prefer_press, torch.ones_like(next_state), next_state)
    next_state = torch.where(conflict & ~prefer_press, torch.zeros_like(next_state), next_state)
    return next_state


class ActionConditionedVideoPolicy(nn.Module):
    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else ModelConfig()
        validate_policy_model_family(self.cfg.policy_model_family)

        self.frame_encoder = SpatialGridEncoder(self.cfg)
        self.spatial_sampler = SpatialTokenSampler(self.cfg)
        self.summary_proj = nn.Sequential(
            RMSNorm(self.cfg.d_model),
            nn.Linear(self.cfg.d_model, self.cfg.d_model),
            nn.GELU(),
            nn.Linear(self.cfg.d_model, self.cfg.d_model),
        )
        self.prev_action_embed = nn.Sequential(
            nn.Linear(self.cfg.prev_action_dim, self.cfg.d_model),
            nn.GELU(),
            nn.Linear(self.cfg.d_model, self.cfg.d_model),
        )
        self.dt_embed = nn.Sequential(
            nn.Linear(3, self.cfg.d_model),
            nn.GELU(),
            nn.Linear(self.cfg.d_model, self.cfg.d_model),
        )
        self.temporal = CausalTemporalEncoder(self.cfg)
        self.decoder = ActionDecoder(self.cfg)

        self.register_buffer(
            "mouse_velocity_scale",
            torch.tensor(list(self.cfg.mouse_velocity_scales), dtype=torch.float32),
            persistent=True,
        )

    @property
    def num_buttons(self) -> int:
        return int(self.cfg.num_bin)

    def parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        params = [p for p in self.parameters() if p.requires_grad]
        return {"backbone": params, "controller": params, "model": params}

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> TemporalState:
        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else next(self.parameters()).dtype
        return TemporalState(
            cached_summaries=None,
            cached_spatial_tokens=None,
            prev_frame=None,
            prev_button_state=torch.zeros(batch_size, self.num_buttons, device=device, dtype=dtype),
            prev_mouse_action=torch.zeros(batch_size, 2, device=device, dtype=dtype),
            steps=0,
        )

    def _normalize_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0
        return frames.clamp(0.0, 1.0)

    def _dt_features(self, dt: torch.Tensor) -> torch.Tensor:
        if dt.dim() == 3 and dt.size(-1) == 1:
            dt = dt[..., 0]
        dt = dt.clamp(min=1.0 / 240.0, max=0.5)
        features = torch.stack([dt, torch.log(dt), 1.0 / dt], dim=-1)
        target_dtype = next(self.dt_embed.parameters()).dtype
        return self.dt_embed(features.to(dtype=target_dtype))

    def _scale_mouse(self, mouse_delta_norm: torch.Tensor) -> torch.Tensor:
        scale = self.mouse_velocity_scale.to(device=mouse_delta_norm.device, dtype=mouse_delta_norm.dtype)
        while scale.dim() < mouse_delta_norm.dim():
            scale = scale.unsqueeze(0)
        return mouse_delta_norm * scale

    def _encode_frames(
        self,
        frames: torch.Tensor,
        *,
        previous_frame: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if frames.dim() == 4:
            frames = frames.unsqueeze(1)
        if frames.dim() != 5:
            raise ValueError(f"Expected frames [B,T,3,H,W], got {tuple(frames.shape)}.")
        b, t, c, h, w = frames.shape
        if c != 3:
            raise ValueError(f"Expected RGB frames, got shape {tuple(frames.shape)}.")

        frames = self._normalize_frames(frames)
        if previous_frame is None:
            prev = torch.cat([frames[:, :1], frames[:, :-1]], dim=1)
            motion = frames - prev
            motion[:, 0] = 0.0
        else:
            prev0 = previous_frame.to(device=frames.device, dtype=frames.dtype).unsqueeze(1)
            prev = torch.cat([prev0, frames[:, :-1]], dim=1)
            motion = frames - prev

        x = torch.cat([frames, motion], dim=2).reshape(b * t, 6, h, w)
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)

        token_chunks: List[torch.Tensor] = []
        for start in range(0, x.size(0), int(self.cfg.encode_chunk_size)):
            grid = self.frame_encoder(x[start : start + int(self.cfg.encode_chunk_size)])
            sampled = self.spatial_sampler(grid)
            token_chunks.append(sampled)
        spatial = torch.cat(token_chunks, dim=0).view(b, t, self.cfg.spatial_tokens, self.cfg.d_model)
        summary = self.summary_proj(spatial.mean(dim=2))
        return summary, spatial, frames[:, -1].detach()

    def _build_summaries(
        self,
        visual_summary: torch.Tensor,
        prev_actions: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        if prev_actions.shape[-1] != self.cfg.prev_action_dim:
            raise ValueError(
                f"prev_actions last dim={prev_actions.shape[-1]} does not match expected {self.cfg.prev_action_dim}."
            )
        if dt.dim() == 1:
            dt = dt.unsqueeze(1).expand(prev_actions.shape[0], prev_actions.shape[1])
        action_dtype = next(self.prev_action_embed.parameters()).dtype
        action_emb = self.prev_action_embed(prev_actions.to(dtype=action_dtype))
        dt_emb = self._dt_features(dt)
        return visual_summary + action_emb.to(dtype=visual_summary.dtype) + dt_emb.to(dtype=visual_summary.dtype)

    def _pack_output(
        self,
        temporal: torch.Tensor,
        spatial: torch.Tensor,
    ) -> PolicyOutput:
        h_button, h_active, h_mouse_norm, press, release = self.decoder(temporal, spatial)
        h_mouse = self._scale_mouse(h_mouse_norm)
        button = h_button[:, :, 0]
        active = h_active[:, :, 0]
        mouse = h_mouse[:, :, 0]
        future_button = {idx + 1: h_button[:, :, idx] for idx in range(self.cfg.prediction_horizon)}
        future_active = {idx + 1: h_active[:, :, idx] for idx in range(self.cfg.prediction_horizon)}
        future_mu = {idx + 1: h_mouse[:, :, idx] for idx in range(self.cfg.prediction_horizon)}
        future_log_b = {idx + 1: torch.zeros_like(h_mouse[:, :, idx]) for idx in range(self.cfg.prediction_horizon)}
        return PolicyOutput(
            button_logits=button,
            press_logits=press,
            release_logits=release,
            mouse_active_logits=active,
            mouse_delta=mouse,
            horizon_button_logits=h_button,
            horizon_mouse_active_logits=h_active,
            horizon_mouse_delta=h_mouse,
            future_button_logits=future_button,
            future_mouse_active_logits=future_active,
            future_mouse_mu=future_mu,
            future_mouse_log_b=future_log_b,
            mouse_mu=mouse,
            mouse_log_b=torch.zeros_like(mouse),
        )

    def forward(
        self,
        frames: torch.Tensor,
        prev_actions: torch.Tensor,
        dt: torch.Tensor,
        state: Optional[TemporalState] = None,
        return_aux: bool = False,
    ) -> PolicyOutput:
        del state, return_aux
        visual_summary, spatial, _ = self._encode_frames(frames)
        summaries = self._build_summaries(visual_summary, prev_actions, dt)
        temporal = self.temporal(summaries)
        return self._pack_output(temporal, spatial)

    def _encode_prev_mouse_action(self, mouse_delta: torch.Tensor) -> torch.Tensor:
        scale = self.mouse_velocity_scale.to(device=mouse_delta.device, dtype=mouse_delta.dtype)
        while scale.dim() < mouse_delta.dim():
            scale = scale.unsqueeze(0)
        return mouse_delta / scale.clamp(min=1e-6)

    def forward_step(
        self,
        frame: torch.Tensor,
        dt: torch.Tensor,
        state: TemporalState,
        return_aux: bool = False,
    ) -> Tuple[PolicyOutput, TemporalState]:
        del return_aux
        if frame.dim() != 4:
            raise ValueError(f"forward_step expects frame shape [B,3,H,W], got {tuple(frame.shape)}.")
        if state.prev_button_state is None or state.prev_mouse_action is None:
            state = self.init_state(batch_size=frame.size(0), device=frame.device, dtype=frame.dtype)

        visual_summary, spatial, last_frame = self._encode_frames(frame.unsqueeze(1), previous_frame=state.prev_frame)
        prev_actions = torch.cat([state.prev_button_state, state.prev_mouse_action], dim=-1).unsqueeze(1)
        step_summary = self._build_summaries(visual_summary, prev_actions, dt.reshape(frame.size(0), 1))

        if state.cached_summaries is None:
            summaries = step_summary
            spatial_cache = spatial
        else:
            summaries = torch.cat([state.cached_summaries.to(step_summary.device, step_summary.dtype), step_summary], dim=1)
            spatial_cache = torch.cat([state.cached_spatial_tokens.to(spatial.device, spatial.dtype), spatial], dim=1)
        if summaries.size(1) > int(self.cfg.max_context):
            summaries = summaries[:, -int(self.cfg.max_context) :]
            spatial_cache = spatial_cache[:, -int(self.cfg.max_context) :]

        temporal = self.temporal(summaries)
        output = self._pack_output(temporal[:, -1:], spatial_cache[:, -1:])
        next_button_state = transition_button_state(
            state.prev_button_state,
            output.press_logits[:, 0],
            output.release_logits[:, 0],
            threshold=float(self.cfg.transition_threshold),
        )
        next_mouse_action = self._encode_prev_mouse_action(output.mouse_delta[:, 0])
        new_state = TemporalState(
            cached_summaries=summaries.detach(),
            cached_spatial_tokens=spatial_cache.detach(),
            prev_frame=last_frame.detach(),
            prev_button_state=next_button_state.detach(),
            prev_mouse_action=next_mouse_action.detach(),
            steps=int(state.steps) + 1,
        )
        squeezed = PolicyOutput(
            button_logits=output.button_logits[:, 0],
            press_logits=output.press_logits[:, 0],
            release_logits=output.release_logits[:, 0],
            mouse_active_logits=output.mouse_active_logits[:, 0],
            mouse_delta=output.mouse_delta[:, 0],
            horizon_button_logits=output.horizon_button_logits[:, 0],
            horizon_mouse_active_logits=output.horizon_mouse_active_logits[:, 0],
            horizon_mouse_delta=output.horizon_mouse_delta[:, 0],
            future_button_logits={k: v[:, 0] for k, v in output.future_button_logits.items()},
            future_mouse_active_logits={k: v[:, 0] for k, v in output.future_mouse_active_logits.items()},
            future_mouse_mu={k: v[:, 0] for k, v in output.future_mouse_mu.items()},
            future_mouse_log_b={k: v[:, 0] for k, v in output.future_mouse_log_b.items()},
            mouse_mu=output.mouse_delta[:, 0],
            mouse_log_b=torch.zeros_like(output.mouse_delta[:, 0]),
        )
        return squeezed, new_state


RealTimeTemporalControlNet = ActionConditionedVideoPolicy


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = ModelConfig(model_size=256, seq_len=8, prediction_horizon=4, max_context=8, temporal_layers=2)
    model = ActionConditionedVideoPolicy(cfg)
    x = torch.rand(1, cfg.seq_len, 3, cfg.model_size, cfg.model_size)
    prev = torch.zeros(1, cfg.seq_len, cfg.prev_action_dim)
    dt = torch.full((1, cfg.seq_len), cfg.prediction_dt)
    out = model(x, prev_actions=prev, dt=dt)
    print(tuple(out.horizon_button_logits.shape), tuple(out.horizon_mouse_delta.shape))
