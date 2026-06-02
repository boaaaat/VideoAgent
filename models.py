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


SPATIAL_FEATURE_CHANNELS = 80
CNN_FEATURE_CHANNELS = SPATIAL_FEATURE_CHANNELS
POLICY_INPUT_CHANNELS = 3
LAST_ACTION_EMBEDDING_DROPOUT = 0.25
LAST_ACTION_FEATURE_SCALE = 1.0
GROUP_NORM_GROUPS = 8
DEFAULT_HORIZON_OFFSETS = (1, 2, 3, 5, 7, 10, 13, 16, 20, 24)
LEGACY_LEARNED_POOLING_STATE_PREFIXES = (
    "spatial_queries",
    "spatial_query_norm.",
    "spatial_cell_norm.",
    "spatial_cross_attn.",
    "frame_query",
    "frame_query_norm.",
    "frame_token_norm.",
    "frame_pool_attn.",
    "motion_query",
    "motion_query_norm.",
    "motion_token_norm.",
    "motion_pool_attn.",
)


def is_legacy_learned_pooling_state_key(name: str) -> bool:
    for prefix in LEGACY_LEARNED_POOLING_STATE_PREFIXES:
        if prefix.endswith("."):
            if name.startswith(prefix):
                return True
        elif name == prefix:
            return True
    return False


def normalize_pooling_shape(pooling: object) -> Tuple[int, int]:
    if isinstance(pooling, str):
        parts = tuple(part.strip() for part in pooling.lower().replace("x", ",").split(",") if part.strip())
        if len(parts) == 1:
            size = int(parts[0])
            return max(1, size), max(1, size)
        if len(parts) == 2:
            return max(1, int(parts[0])), max(1, int(parts[1]))
        raise ValueError(f"pooling must be H,W or HxW, got {pooling!r}.")
    if isinstance(pooling, int):
        size = max(1, int(pooling))
        return size, size
    values = tuple(int(value) for value in pooling)  # type: ignore[arg-type]
    if len(values) != 2:
        raise ValueError(f"pooling must contain 2 values, got {values}.")
    return max(1, values[0]), max(1, values[1])


def _largest_valid_head_count(channels: int, requested_heads: int) -> int:
    requested_heads = max(1, min(int(requested_heads), int(channels)))
    for heads in range(requested_heads, 0, -1):
        if channels % heads == 0:
            return heads
    return 1


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = max(1, min(int(GROUP_NORM_GROUPS), int(channels)))
    while int(channels) % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, int(channels))


def _default_horizon_offsets(prediction_horizon: int) -> Tuple[int, ...]:
    horizon = max(1, int(prediction_horizon))
    if horizon <= len(DEFAULT_HORIZON_OFFSETS):
        return tuple(DEFAULT_HORIZON_OFFSETS[:horizon])
    offsets = list(DEFAULT_HORIZON_OFFSETS)
    while len(offsets) < horizon:
        offsets.append(offsets[-1] + 4)
    return tuple(offsets)


@dataclass
class ModelConfig:
    selected_game: str = ACTION_SELECTED_GAME
    data_root: Optional[str] = None
    video_ext: str = ".mp4"
    csv_ext: str = ".csv"

    model_size: int = 256
    seq_len: int = 80
    train_seq_stride: int = 40
    val_seq_stride: int = 80
    prediction_dt: float = 1.0 / 20.0
    prediction_horizon: int = 10
    prediction_horizon_offsets: Optional[Sequence[int]] = None

    key_names: Optional[List[str]] = None
    mouse_button_names: Optional[List[str]] = None

    d_model: int = 256
    spatial_dropout: float = 0.05
    head_dropout: float = 0.1

    fastvit_depth: int = 2
    fastvit_kernel_size: int = 3
    temporal_layers: int = 4
    temporal_heads: int = 4
    temporal_context: int = 80

    pooling: Tuple[int, int] = (16, 16)
    recent_spatial_context: int = 10
    high_res_spatial_context: int = 20
    high_res_pooling: Tuple[int, int] = (5, 5)
    low_res_pooling: Tuple[int, int] = (3, 3)

    button_state_threshold: float = 0.5
    button_state_thresholds: Optional[Sequence[float]] = None
    num_bin: int = 0

    def __post_init__(self) -> None:
        self.selected_game = normalize_game_name(self.selected_game)
        if self.data_root is None:
            self.data_root = game_data_root(self.selected_game)

        self.model_size = max(32, int(self.model_size))
        self.seq_len = max(1, int(self.seq_len))
        self.train_seq_stride = max(1, int(self.train_seq_stride))
        self.val_seq_stride = max(1, int(self.val_seq_stride))
        self.prediction_dt = max(1.0 / 240.0, float(self.prediction_dt))
        self.prediction_horizon = max(1, int(self.prediction_horizon))
        if self.prediction_horizon_offsets is None:
            self.prediction_horizon_offsets = _default_horizon_offsets(self.prediction_horizon)
        else:
            offsets = tuple(int(offset) for offset in self.prediction_horizon_offsets)
            if not offsets:
                raise ValueError("prediction_horizon_offsets must contain at least one frame offset.")
            if any(offset <= 0 for offset in offsets):
                raise ValueError(f"prediction_horizon_offsets must be positive, got {offsets}.")
            if any(curr <= prev for prev, curr in zip(offsets, offsets[1:])):
                raise ValueError(f"prediction_horizon_offsets must be strictly increasing, got {offsets}.")
            self.prediction_horizon_offsets = offsets
            self.prediction_horizon = len(offsets)

        self.d_model = max(64, int(self.d_model))
        self.spatial_dropout = float(min(max(self.spatial_dropout, 0.0), 0.9))
        self.head_dropout = float(min(max(self.head_dropout, 0.0), 0.9))
        self.fastvit_depth = max(0, int(self.fastvit_depth))
        self.fastvit_kernel_size = max(3, int(self.fastvit_kernel_size))
        if self.fastvit_kernel_size % 2 == 0:
            self.fastvit_kernel_size += 1
        self.temporal_layers = max(1, int(self.temporal_layers))
        self.temporal_heads = _largest_valid_head_count(self.d_model, int(self.temporal_heads))
        self.temporal_context = max(1, int(self.temporal_context))
        self.pooling = normalize_pooling_shape(self.pooling)
        self.recent_spatial_context = max(1, int(self.recent_spatial_context))
        self.high_res_spatial_context = max(1, int(self.high_res_spatial_context))
        self.high_res_pooling = normalize_pooling_shape(self.high_res_pooling)
        self.low_res_pooling = normalize_pooling_shape(self.low_res_pooling)

        if self.key_names is None:
            self.key_names = get_key_names(self.selected_game)
        else:
            self.key_names = list(self.key_names)

        if self.mouse_button_names is None:
            self.mouse_button_names = get_mouse_button_names(self.selected_game)
        else:
            self.mouse_button_names = list(self.mouse_button_names)

        self.num_bin = len(self.key_names) + len(self.mouse_button_names)

        self.button_state_threshold = float(min(max(self.button_state_threshold, 0.0), 1.0))
        if self.button_state_thresholds is None:
            self.button_state_thresholds = tuple(float(self.button_state_threshold) for _ in range(self.num_bin))
        else:
            thresholds = tuple(float(x) for x in self.button_state_thresholds)
            if len(thresholds) != self.num_bin:
                raise ValueError(
                    f"button_state_thresholds must contain {self.num_bin} values, got {len(thresholds)}."
                )
            self.button_state_thresholds = tuple(min(max(x, 0.0), 1.0) for x in thresholds)


@dataclass
class PolicyOutput:
    button_logits: torch.Tensor
    horizon_button_logits: torch.Tensor
    future_button_logits: Dict[int, torch.Tensor] = field(default_factory=dict)


@dataclass
class TemporalState:
    hidden_state: Optional[torch.Tensor] = None
    visual_tokens: Optional[torch.Tensor] = None
    low_res_visual_tokens: Optional[torch.Tensor] = None
    high_res_visual_tokens: Optional[torch.Tensor] = None
    previous_frame: Optional[torch.Tensor] = None


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class CustomSpatialEncoder(nn.Module):
    """
    Exact-width residual CNN encoder.
    Outputs [B, SPATIAL_FEATURE_CHANNELS, H/4, W/4] for square policy inputs.
    """
    def __init__(self, in_channels: int = 3, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        self.detail = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
        )
        self.residual_blocks = nn.Sequential(
            ResidualBlock(64, dropout),
            ResidualBlock(64, dropout),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(64, SPATIAL_FEATURE_CHANNELS, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(SPATIAL_FEATURE_CHANNELS),
            nn.SiLU(inplace=True),
        )

    def feature_stages(self, x: torch.Tensor) -> List[torch.Tensor]:
        stem = self.stem(x)
        detail = self.detail(stem)
        residual = self.residual_blocks(detail)
        projected = self.proj(residual)
        return [stem, detail, residual, projected]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_stages(x)[-1]


class DrivingVideoPolicy(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.car_y_min_pct = 0.50
        self.car_y_max_pct = 0.80
        self.car_x_min_pct = 0.40
        self.car_x_max_pct = 0.60

        self.spatial_encoder = CustomSpatialEncoder(
            in_channels=POLICY_INPUT_CHANNELS,
            dropout=cfg.spatial_dropout,
        )
        self.feat_channels = SPATIAL_FEATURE_CHANNELS
        self.tokens_per_frame = int(cfg.high_res_pooling[0]) * int(cfg.high_res_pooling[1])

        self.cell_projector = nn.Sequential(
            nn.Linear(self.feat_channels, self.cfg.d_model),
            nn.LayerNorm(self.cfg.d_model),
            nn.SiLU(inplace=True),
            nn.Dropout(cfg.head_dropout * 0.5),
        )
        self.spatial_coord_projector = nn.Linear(4, self.cfg.d_model)
        nn.init.normal_(self.spatial_coord_projector.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.spatial_coord_projector.bias)

        self.low_res_token_embed = nn.Parameter(torch.empty(1, 1, 1, self.cfg.d_model))
        self.high_res_token_embed = nn.Parameter(torch.empty(1, 1, 1, self.cfg.d_model))
        nn.init.normal_(self.low_res_token_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.high_res_token_embed, mean=0.0, std=0.02)

        self.spatial_token_norm = nn.LayerNorm(self.cfg.d_model)
        self.spatial_token_ff = nn.Sequential(
            nn.LayerNorm(self.cfg.d_model),
            nn.Linear(self.cfg.d_model, max(self.cfg.d_model * 2, 256)),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout * 0.5),
            nn.Linear(max(self.cfg.d_model * 2, 256), self.cfg.d_model),
            nn.Dropout(cfg.head_dropout * 0.5),
        )

        dt_hidden = max(16, min(64, self.cfg.d_model // 4))
        self.dt_encoder = nn.Sequential(
            nn.Linear(1, dt_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(dt_hidden, self.cfg.d_model),
        )

        action_hidden = max(4, min(32, cfg.num_bin * 2, self.cfg.d_model // 4))
        self.last_action_encoder = nn.Sequential(
            nn.Linear(cfg.num_bin, action_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(LAST_ACTION_EMBEDDING_DROPOUT),
            nn.Linear(action_hidden, self.cfg.d_model),
        )
        self.context_delta = nn.Sequential(
            nn.LayerNorm(self.cfg.d_model * 2),
            nn.Linear(self.cfg.d_model * 2, self.cfg.d_model),
            nn.SiLU(inplace=True),
            nn.Dropout(cfg.head_dropout * 0.5),
            nn.Linear(self.cfg.d_model, self.cfg.d_model),
        )
        self.context_gate = nn.Linear(self.cfg.d_model * 3, self.cfg.d_model)
        nn.init.zeros_(self.context_gate.weight)
        nn.init.constant_(self.context_gate.bias, -3.0)
        self.fused_token_norm = nn.LayerNorm(self.cfg.d_model)

        self.temporal_frame_count = max(int(self.cfg.seq_len), int(self.cfg.temporal_context))
        self.temporal_pos_embed = nn.Parameter(torch.empty(1, self.temporal_frame_count, self.cfg.d_model))
        nn.init.normal_(self.temporal_pos_embed, mean=0.0, std=0.02)

        temporal_heads = _largest_valid_head_count(self.cfg.d_model, self.cfg.temporal_heads)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.d_model,
            nhead=temporal_heads,
            dim_feedforward=max(self.cfg.d_model * 4, 512),
            dropout=cfg.head_dropout * 0.5,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=self.cfg.temporal_layers,
            norm=nn.LayerNorm(self.cfg.d_model),
            enable_nested_tensor=False,
        )
        self.temporal_norm = nn.LayerNorm(self.cfg.d_model)

        self.horizon_queries = nn.Parameter(torch.empty(self.cfg.prediction_horizon, self.cfg.d_model))
        nn.init.normal_(self.horizon_queries, mean=0.0, std=0.02)
        horizon_hidden = max(64, self.cfg.d_model // 2)
        self.button_head = nn.Sequential(
            nn.LayerNorm(self.cfg.d_model),
            nn.Linear(self.cfg.d_model, horizon_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(cfg.head_dropout * 0.5),
            nn.Linear(horizon_hidden, self.cfg.num_bin),
        )
        self.persistence_logit_gate = nn.Parameter(torch.tensor(0.0))
        final_button_layer = self.button_head[-1]
        if isinstance(final_button_layer, nn.Linear):
            nn.init.zeros_(final_button_layer.bias)

    def _max_context_tokens(self) -> int:
        return int(self.cfg.temporal_context)

    def _max_recent_visual_frames(self) -> int:
        return max(1, int(self.cfg.high_res_spatial_context))

    def _pooling_token_count(self) -> int:
        pool_h, pool_w = self._spatial_pool_shape()
        return pool_h * pool_w

    def _spatial_pool_shape(self) -> Tuple[int, int]:
        return normalize_pooling_shape(self.cfg.high_res_pooling)

    def _high_res_pool_shape(self) -> Tuple[int, int]:
        return normalize_pooling_shape(self.cfg.high_res_pooling)

    def _low_res_pool_shape(self) -> Tuple[int, int]:
        return normalize_pooling_shape(self.cfg.low_res_pooling)

    def _spatial_coord_tokens(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, int(height), device=device)
        x = torch.linspace(-1.0, 1.0, int(width), device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack((xx, yy, xx * yy, xx.square() + yy.square()), dim=-1)
        weight_dtype = self.spatial_coord_projector.weight.dtype
        coords = coords.reshape(1, int(height) * int(width), 4).to(dtype=weight_dtype)
        return self.spatial_coord_projector(coords).to(dtype=dtype)

    def _project_spatial_features(self, spatial_feats: torch.Tensor) -> torch.Tensor:
        if spatial_feats.dim() != 4:
            raise ValueError(f"Expected spatial features [B,C,H,W], got {tuple(spatial_feats.shape)}.")
        b, c, h, w = spatial_feats.shape
        if int(c) != int(self.feat_channels):
            raise ValueError(f"Expected {self.feat_channels} feature channels, got {c}.")
        cells = spatial_feats.flatten(2).transpose(1, 2)
        cell_tokens = self.cell_projector(cells)
        pos = self._spatial_coord_tokens(
            int(h),
            int(w),
            device=cell_tokens.device,
            dtype=cell_tokens.dtype,
        )
        tokens = cell_tokens + pos
        return self.spatial_token_norm(tokens + self.spatial_token_ff(tokens))

    def _normalize_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0
        return frames.clamp(0.0, 1.0)

    def _apply_masks(self, frames: torch.Tensor) -> torch.Tensor:
        h, w = frames.shape[-2:]
        masked_frames = frames.clone()

        hud_y1 = int(h * 0.96)
        masked_frames[..., hud_y1:, :] = 0.0

        map_y1, map_y2 = int(h * 0.05), int(h * 0.2)
        map_x1 = int(w * 0.75)
        masked_frames[..., map_y1:map_y2, map_x1:] = 0.0

        roblox_ui_y2 = int(h * 0.1)
        roblox_ui_x2 = int(w * 0.1)
        masked_frames[..., :roblox_ui_y2, :roblox_ui_x2] = 0.0

        return masked_frames

    def _dt_features(
        self,
        dt: Optional[torch.Tensor],
        batch_size: int,
        time_steps: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if dt is None:
            dt_values = torch.full((batch_size, time_steps), float(self.cfg.prediction_dt), device=device, dtype=dtype)
        else:
            dt_values = dt.to(device=device, dtype=dtype)
            if dt_values.dim() == 0:
                dt_values = dt_values.reshape(1, 1).expand(batch_size, time_steps)
            elif dt_values.dim() == 1:
                if int(dt_values.numel()) == batch_size and time_steps == 1:
                    dt_values = dt_values.view(batch_size, 1)
                elif int(dt_values.numel()) == time_steps and batch_size == 1:
                    dt_values = dt_values.view(1, time_steps)
                elif int(dt_values.numel()) == batch_size:
                    dt_values = dt_values.view(batch_size, 1).expand(batch_size, time_steps)
                else:
                    raise ValueError(f"Expected dt with {batch_size} or {time_steps} values, got {tuple(dt_values.shape)}.")
            elif dt_values.dim() == 2:
                if tuple(dt_values.shape) != (batch_size, time_steps):
                    raise ValueError(f"Expected dt shape {(batch_size, time_steps)}, got {tuple(dt_values.shape)}.")
            else:
                raise ValueError(f"Expected scalar, [B], [T], or [B,T] dt, got {tuple(dt_values.shape)}.")

        base_dt = max(float(self.cfg.prediction_dt), 1.0 / 240.0)
        dt_scaled = (dt_values / base_dt).clamp(0.25, 4.0) - 1.0
        return self.dt_encoder(dt_scaled.reshape(batch_size, time_steps, 1)).to(dtype=dtype)

    def _last_action_features(
        self,
        prev_action: Optional[torch.Tensor],
        batch_size: int,
        time_steps: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if prev_action is None:
            action_values = torch.zeros((batch_size, time_steps, self.cfg.num_bin), device=device, dtype=dtype)
        else:
            action_values = prev_action.to(device=device, dtype=dtype)
            if action_values.dim() == 2:
                if tuple(action_values.shape) != (batch_size, self.cfg.num_bin):
                    raise ValueError(
                        f"Expected prev_action shape {(batch_size, self.cfg.num_bin)} or "
                        f"{(batch_size, time_steps, self.cfg.num_bin)}, got {tuple(action_values.shape)}."
                    )
                action_values = action_values.view(batch_size, 1, self.cfg.num_bin).expand(
                    batch_size,
                    time_steps,
                    self.cfg.num_bin,
                )
            elif action_values.dim() == 3:
                if tuple(action_values.shape) != (batch_size, time_steps, self.cfg.num_bin):
                    raise ValueError(
                        f"Expected prev_action shape {(batch_size, time_steps, self.cfg.num_bin)}, "
                        f"got {tuple(action_values.shape)}."
                    )
            else:
                raise ValueError(f"Expected prev_action with 2 or 3 dims, got {tuple(action_values.shape)}.")

        action_values = action_values.clamp(0.0, 1.0).reshape(batch_size * time_steps, self.cfg.num_bin)
        features = self.last_action_encoder(action_values).reshape(batch_size, time_steps, self.cfg.d_model)
        return features.to(dtype=dtype) * LAST_ACTION_FEATURE_SCALE

    def _fuse_token_context(
        self,
        tokens: torch.Tensor,
        dt: Optional[torch.Tensor],
        prev_action: Optional[torch.Tensor],
        prev_action_scale: float = 1.0,
    ) -> torch.Tensor:
        if tokens.dim() != 4:
            raise ValueError(f"Expected tokens [B,T,K,D], got {tuple(tokens.shape)}.")
        b, t, _, _ = tokens.shape
        dt_feat = self._dt_features(dt, b, t, device=tokens.device, dtype=tokens.dtype)
        action_feat = self._last_action_features(prev_action, b, t, device=tokens.device, dtype=tokens.dtype)
        action_scale = torch.as_tensor(prev_action_scale, device=tokens.device, dtype=tokens.dtype)
        action_feat = action_feat * action_scale
        context = torch.cat([action_feat, dt_feat], dim=-1)
        context_delta = self.context_delta(context).unsqueeze(2)
        action_dt = torch.cat([action_feat, dt_feat], dim=-1).unsqueeze(2).expand(-1, -1, tokens.size(2), -1)
        gate = torch.sigmoid(self.context_gate(torch.cat([tokens, action_dt], dim=-1)))
        return self.fused_token_norm(tokens + gate * context_delta)

    def _cnn_features_from_masked_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dim() != 5:
            raise ValueError(f"Expected frames [B,T,3,H,W], got {tuple(frames.shape)}.")
        b, t, c, h, w = frames.shape
        if int(c) != POLICY_INPUT_CHANNELS:
            raise ValueError(f"Expected RGB frames with shape [B,T,3,H,W], got {tuple(frames.shape)}.")
        x = frames.reshape(b * t, c, h, w)
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        spatial_feats = self.spatial_encoder(x)
        _, feat_c, feat_h, feat_w = spatial_feats.shape
        return spatial_feats.reshape(b, t, feat_c, feat_h, feat_w)

    def _tokens_from_features(
        self,
        features: torch.Tensor,
        pool_shape: Tuple[int, int],
        token_embed: torch.Tensor,
    ) -> torch.Tensor:
        if features.dim() != 5:
            raise ValueError(f"Expected features [B,T,C,H,W], got {tuple(features.shape)}.")
        b, t, c, h, w = features.shape
        pool_h, pool_w = normalize_pooling_shape(pool_shape)
        flat = features.reshape(b * t, c, h, w)
        pooled = F.adaptive_avg_pool2d(flat, (pool_h, pool_w))
        tokens = self._project_spatial_features(pooled).reshape(b, t, pool_h * pool_w, self.cfg.d_model)
        return tokens + token_embed.to(device=tokens.device, dtype=tokens.dtype)

    def _token_grids_from_masked_frames(
        self,
        frames: torch.Tensor,
        dt: Optional[torch.Tensor],
        prev_action: Optional[torch.Tensor],
        prev_action_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self._cnn_features_from_masked_frames(frames)
        low_tokens = self._tokens_from_features(
            features,
            self._low_res_pool_shape(),
            self.low_res_token_embed,
        )
        high_tokens = self._tokens_from_features(
            features,
            self._high_res_pool_shape(),
            self.high_res_token_embed,
        )
        low_tokens = self._fuse_token_context(low_tokens, dt, prev_action, prev_action_scale=prev_action_scale)
        high_tokens = self._fuse_token_context(high_tokens, dt, prev_action, prev_action_scale=prev_action_scale)
        return low_tokens, high_tokens

    def _visual_cells_from_masked_frames(self, frames: torch.Tensor) -> torch.Tensor:
        features = self._cnn_features_from_masked_frames(frames)
        b, t, c, h, w = features.shape
        pool_h, pool_w = self._high_res_pool_shape()
        pooled = F.adaptive_avg_pool2d(features.reshape(b * t, c, h, w), (pool_h, pool_w))
        tokens = self._project_spatial_features(pooled)
        return tokens.reshape(b, t, pool_h * pool_w, self.cfg.d_model)

    def _visual_tokens_from_masked_frames(self, frames: torch.Tensor) -> torch.Tensor:
        return self._visual_cells_from_masked_frames(frames)

    def _prepare_grid_state(
        self,
        state_tokens: Optional[torch.Tensor],
        batch_size: int,
        tokens_per_frame: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if state_tokens is None:
            return torch.zeros(
                batch_size,
                0,
                tokens_per_frame,
                self.cfg.d_model,
                device=device,
                dtype=dtype,
            )
        tokens = state_tokens.to(device=device, dtype=dtype)
        if tokens.dim() != 4:
            raise ValueError(f"Expected token state [B,T,K,D], got {tuple(tokens.shape)}.")
        if int(tokens.size(0)) != int(batch_size):
            raise ValueError(f"Expected token state batch {batch_size}, got {tokens.size(0)}.")
        if int(tokens.size(2)) != int(tokens_per_frame):
            raise ValueError(f"Expected {tokens_per_frame} tokens per frame, got {tokens.size(2)}.")
        if int(tokens.size(-1)) != int(self.cfg.d_model):
            raise ValueError(f"Expected token dim {self.cfg.d_model}, got {tokens.size(-1)}.")
        return tokens[:, -self._max_context_tokens():]

    def _prepare_low_high_state(
        self,
        state: Optional[TemporalState],
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        low_k = self._low_res_pool_shape()[0] * self._low_res_pool_shape()[1]
        high_k = self._high_res_pool_shape()[0] * self._high_res_pool_shape()[1]
        low_state = None if state is None else state.low_res_visual_tokens
        high_state = None if state is None else state.high_res_visual_tokens
        return (
            self._prepare_grid_state(low_state, batch_size, low_k, device=device, dtype=dtype),
            self._prepare_grid_state(high_state, batch_size, high_k, device=device, dtype=dtype),
        )

    def _select_variable_grid_tokens(
        self,
        low_tokens: torch.Tensor,
        high_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        if low_tokens.dim() != 4 or high_tokens.dim() != 4:
            raise ValueError(
                f"Expected low/high tokens [B,T,K,D], got {tuple(low_tokens.shape)} and {tuple(high_tokens.shape)}."
            )
        if int(low_tokens.size(0)) != int(high_tokens.size(0)) or int(low_tokens.size(1)) != int(high_tokens.size(1)):
            raise ValueError("Low- and high-resolution token states must share batch and frame dimensions.")
        total_frames = int(low_tokens.size(1))
        high_start = max(0, total_frames - self._max_recent_visual_frames())

        chunks: List[torch.Tensor] = []
        frame_indices: List[torch.Tensor] = []
        spans: List[Tuple[int, int]] = []
        cursor = 0
        for frame_idx in range(total_frames):
            frame_tokens = high_tokens[:, frame_idx] if frame_idx >= high_start else low_tokens[:, frame_idx]
            token_count = int(frame_tokens.size(1))
            chunks.append(frame_tokens)
            frame_indices.append(
                torch.full((token_count,), frame_idx, device=low_tokens.device, dtype=torch.long)
            )
            spans.append((cursor, cursor + token_count))
            cursor += token_count
        if not chunks:
            raise ValueError("Temporal transformer requires at least one frame.")
        return torch.cat(chunks, dim=1), torch.cat(frame_indices, dim=0), spans

    def _causal_attention_mask(self, frame_indices: torch.Tensor) -> torch.Tensor:
        return frame_indices.unsqueeze(0) > frame_indices.unsqueeze(1)

    def _encode_variable_tokens(
        self,
        low_tokens: torch.Tensor,
        high_tokens: torch.Tensor,
    ) -> torch.Tensor:
        token_memory, frame_indices, spans = self._select_variable_grid_tokens(low_tokens, high_tokens)
        total_frames = int(low_tokens.size(1))
        if total_frames > self.temporal_frame_count:
            raise ValueError(
                f"Temporal frame count {total_frames} exceeds configured capacity {self.temporal_frame_count}."
            )
        pos = self.temporal_pos_embed[:, :total_frames].to(device=token_memory.device, dtype=token_memory.dtype)
        token_memory = token_memory + pos[:, frame_indices]
        encoded = self.temporal_encoder(
            token_memory,
            mask=self._causal_attention_mask(frame_indices),
            is_causal=False,
        )
        encoded = self.temporal_norm(encoded)
        frame_features = [encoded[:, start:end].mean(dim=1) for start, end in spans]
        return torch.stack(frame_features, dim=1)

    def _horizon_button_logits(
        self,
        frame_features: torch.Tensor,
        prev_action: Optional[torch.Tensor] = None,
        persistence_scale: float = 1.0,
    ) -> torch.Tensor:
        if frame_features.dim() == 2:
            frame_features = frame_features.unsqueeze(1)
        if frame_features.dim() != 3:
            raise ValueError(f"Expected frame features [B,T,D], got {tuple(frame_features.shape)}.")
        if int(frame_features.size(-1)) != int(self.cfg.d_model):
            raise ValueError(f"Expected final feature dim {self.cfg.d_model}, got {frame_features.size(-1)}.")

        b, t, d = frame_features.shape
        horizon_queries = self.horizon_queries.to(device=frame_features.device, dtype=frame_features.dtype)
        query_features = frame_features.unsqueeze(2) + horizon_queries.reshape(
            1,
            1,
            self.cfg.prediction_horizon,
            d,
        )
        logits = self.button_head(query_features)
        prior = self._persistence_prior_logits(
            prev_action,
            batch_size=b,
            time_steps=t,
            device=logits.device,
            dtype=logits.dtype,
            persistence_scale=persistence_scale,
        )
        return logits + prior

    def _persistence_prior_logits(
        self,
        prev_action: Optional[torch.Tensor],
        *,
        batch_size: int,
        time_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        persistence_scale: float = 1.0,
    ) -> torch.Tensor:
        if prev_action is None:
            action_values = torch.zeros((batch_size, time_steps, self.cfg.num_bin), device=device, dtype=dtype)
        else:
            action_values = prev_action.to(device=device, dtype=dtype)
            if action_values.dim() == 2:
                if tuple(action_values.shape) != (batch_size, self.cfg.num_bin):
                    raise ValueError(
                        f"Expected prev_action shape {(batch_size, self.cfg.num_bin)} or "
                        f"{(batch_size, time_steps, self.cfg.num_bin)}, got {tuple(action_values.shape)}."
                    )
                action_values = action_values.view(batch_size, 1, self.cfg.num_bin).expand(
                    batch_size,
                    time_steps,
                    self.cfg.num_bin,
                )
            elif action_values.dim() == 3:
                if int(action_values.size(0)) != int(batch_size) or int(action_values.size(-1)) != int(self.cfg.num_bin):
                    raise ValueError(
                        f"Expected prev_action batch/classes {(batch_size, self.cfg.num_bin)}, got {tuple(action_values.shape)}."
                    )
                if int(action_values.size(1)) < int(time_steps):
                    pad = action_values[:, :1].expand(batch_size, time_steps - int(action_values.size(1)), self.cfg.num_bin)
                    action_values = torch.cat([pad, action_values], dim=1)
                action_values = action_values[:, -time_steps:, :]
            else:
                raise ValueError(f"Expected prev_action with 2 or 3 dims, got {tuple(action_values.shape)}.")

        prior_logits = action_values.clamp(0.0, 1.0).mul(2.0).sub(1.0) * 2.0
        horizon_decay = self._persistence_horizon_decay(device=device, dtype=dtype)
        persistence_scale_tensor = torch.as_tensor(persistence_scale, device=device, dtype=dtype)
        gate = torch.sigmoid(self.persistence_logit_gate).to(dtype=dtype) * persistence_scale_tensor
        return gate * prior_logits.unsqueeze(2) * horizon_decay.view(1, 1, -1, 1)

    def _persistence_horizon_decay(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        offsets = torch.tensor(
            tuple(int(offset) for offset in self.cfg.prediction_horizon_offsets),
            device=device,
            dtype=dtype,
        )
        first = offsets[:1].clamp(min=1.0)
        span = (offsets[-1:] - first).clamp(min=1.0)
        progress = ((offsets - first) / span).clamp(0.0, 1.0)
        return (1.0 - progress).pow(2.0).clamp(min=0.05, max=1.0)

    def _make_output(self, button: torch.Tensor) -> PolicyOutput:
        return PolicyOutput(
            button_logits=button[:, :, 0] if button.dim() == 4 else button[:, 0],
            horizon_button_logits=button,
            future_button_logits={
                int(offset): button[:, :, idx] if button.dim() == 4 else button[:, idx]
                for idx, offset in enumerate(self.cfg.prediction_horizon_offsets)
            },
        )

    def forward(
        self,
        frames: torch.Tensor,
        dt: torch.Tensor,
        state: Optional[TemporalState] = None,
        return_aux: bool = False,
        prev_action: Optional[torch.Tensor] = None,
        prev_action_scale: float = 1.0,
        persistence_scale: float = 1.0,
    ):
        del return_aux
        if frames.dim() != 5:
            raise ValueError(f"Expected frames [B,T,3,H,W], got {tuple(frames.shape)}.")
        b, t, c, _, _ = frames.shape
        if int(c) != POLICY_INPUT_CHANNELS:
            raise ValueError(f"Expected RGB frames with shape [B,T,3,H,W], got {tuple(frames.shape)}.")
        if t > self.temporal_frame_count:
            raise ValueError(
                f"Input sequence length {t} exceeds temporal frame capacity {self.temporal_frame_count}."
            )

        frames = self._apply_masks(self._normalize_frames(frames))
        low_tokens, high_tokens = self._token_grids_from_masked_frames(
            frames,
            dt,
            prev_action,
            prev_action_scale=prev_action_scale,
        )
        prior_low, prior_high = self._prepare_low_high_state(
            state,
            b,
            device=frames.device,
            dtype=low_tokens.dtype,
        )
        if prior_low.size(1) > 0:
            keep_prior = max(0, self._max_context_tokens() - int(low_tokens.size(1)))
            prior_low = prior_low[:, -keep_prior:] if keep_prior > 0 else prior_low[:, :0]
            prior_high = prior_high[:, -keep_prior:] if keep_prior > 0 else prior_high[:, :0]
            low_tokens = torch.cat([prior_low, low_tokens], dim=1)
            high_tokens = torch.cat([prior_high, high_tokens], dim=1)

        frame_features = self._encode_variable_tokens(low_tokens, high_tokens)
        current_features = frame_features[:, -t:]
        button = self._horizon_button_logits(
            current_features,
            prev_action=prev_action,
            persistence_scale=persistence_scale,
        )
        return self._make_output(button)

    def forward_step(
        self,
        frame: torch.Tensor,
        dt: torch.Tensor,
        state: TemporalState,
        return_aux: bool = False,
        prev_action: Optional[torch.Tensor] = None,
        prev_action_scale: float = 1.0,
        persistence_scale: float = 1.0,
    ):
        del return_aux
        if frame.dim() != 4 or int(frame.size(1)) != POLICY_INPUT_CHANNELS:
            raise ValueError(f"Expected current frame [B,3,H,W], got {tuple(frame.shape)}.")
        b = int(frame.size(0))

        frame_norm = self._normalize_frames(frame)
        masked_frame = self._apply_masks(frame_norm)
        current_low, current_high = self._token_grids_from_masked_frames(
            masked_frame.unsqueeze(1),
            dt,
            prev_action,
            prev_action_scale=prev_action_scale,
        )
        prior_low, prior_high = self._prepare_low_high_state(
            state,
            b,
            device=frame.device,
            dtype=current_low.dtype,
        )
        low_memory = torch.cat([prior_low, current_low], dim=1)[:, -self._max_context_tokens():]
        high_memory = torch.cat([prior_high, current_high], dim=1)[:, -self._max_context_tokens():]

        frame_features = self._encode_variable_tokens(low_memory, high_memory)
        button = self._horizon_button_logits(
            frame_features[:, -1:],
            prev_action=prev_action,
            persistence_scale=persistence_scale,
        )[:, 0]

        squeezed = self._make_output(button)
        new_state = TemporalState(
            hidden_state=frame_features.detach(),
            visual_tokens=high_memory.detach(),
            low_res_visual_tokens=low_memory.detach(),
            high_res_visual_tokens=high_memory.detach(),
            previous_frame=None,
        )
        return squeezed, new_state
