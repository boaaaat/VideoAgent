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


CNN_FEATURE_CHANNELS = 72
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
    spatial_token_count: int = 81
    recent_spatial_context: int = 10

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
        self.spatial_token_count = max(1, int(self.spatial_token_count))
        self.recent_spatial_context = max(1, int(self.recent_spatial_context))

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
    previous_frame: Optional[torch.Tensor] = None


class ResBlock(nn.Module):
    """ Lightweight 2D Residual block for regularizing spatial primitives """
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            _group_norm(channels),
            nn.ELU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            _group_norm(channels),
            nn.Dropout2d(dropout)
        )
        self.elu = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.elu(x + self.conv(x))


class CustomSpatialEncoder(nn.Module):
    """
    Preserves /4 detail features and fuses them into /8 semantic features.
    Outputs a feature map scale of [B, CNN_FEATURE_CHANNELS, H/8, W/8].
    """
    def __init__(self, in_channels: int = 3, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=5, stride=2, padding=2, bias=False),  # /2
            _group_norm(24),
            nn.ELU(inplace=True),
        )
        self.detail = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1, bias=False),          # /4
            _group_norm(48),
            nn.ELU(inplace=True),
            ResBlock(48, dropout),
        )
        self.semantic = nn.Sequential(
            nn.Conv2d(48, CNN_FEATURE_CHANNELS, kernel_size=3, stride=2, padding=1, bias=False),          # /8
            _group_norm(CNN_FEATURE_CHANNELS),
            nn.ELU(inplace=True),
            ResBlock(CNN_FEATURE_CHANNELS, dropout),
        )
        self.detail_to_semantic = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1, bias=False),          # /4 -> /8
            _group_norm(48),
            nn.ELU(inplace=True),
            ResBlock(48, dropout * 0.5),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(CNN_FEATURE_CHANNELS + 48, CNN_FEATURE_CHANNELS, kernel_size=1, bias=False),
            _group_norm(CNN_FEATURE_CHANNELS),
            nn.ELU(inplace=True),
            ResBlock(CNN_FEATURE_CHANNELS, dropout)
        )

    def feature_stages(self, x: torch.Tensor) -> List[torch.Tensor]:
        stem = self.stem(x)
        detail = self.detail(stem)
        semantic = self.semantic(detail)
        detail_semantic = self.detail_to_semantic(detail)
        fused = self.fuse(torch.cat([semantic, detail_semantic], dim=1))
        return [stem, detail, semantic, fused]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_stages(x)[-1]


class FastVitStyleBlock(nn.Module):
    """
    Compact FastViT-style local token mixer for post-CNN feature maps.
    """
    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        kernel_size = max(3, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = kernel_size // 2
        hidden_channels = max(int(channels) * 2, int(channels))

        self.token_mixer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels, bias=False),
            _group_norm(channels),
            nn.GELU(),
            nn.Dropout2d(dropout * 0.5),
        )
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            _group_norm(hidden_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            _group_norm(channels),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mixer(x)
        return x + self.channel_mixer(x)


class FastVitSpatialMixer(nn.Module):
    def __init__(self, channels: int, depth: int, kernel_size: int, dropout: float):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                FastVitStyleBlock(channels, kernel_size=kernel_size, dropout=dropout)
                for _ in range(max(0, int(depth)))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class DrivingVideoPolicy(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        
        # --- Center Car Mask Boundaries (Percentages) ---
        self.car_y_min_pct = 0.50
        self.car_y_max_pct = 0.80
        self.car_x_min_pct = 0.40
        self.car_x_max_pct = 0.60
        
        self.spatial_encoder = CustomSpatialEncoder(in_channels=POLICY_INPUT_CHANNELS, dropout=cfg.spatial_dropout)
        self.feat_channels = CNN_FEATURE_CHANNELS
        self.fastvit_mixer = FastVitSpatialMixer(
            self.feat_channels,
            depth=cfg.fastvit_depth,
            kernel_size=cfg.fastvit_kernel_size,
            dropout=cfg.spatial_dropout,
        )
        
        self.tokens_per_frame = 1

        self.cell_projector = nn.Sequential(
            nn.Linear(self.feat_channels, self.cfg.d_model),
            nn.LayerNorm(self.cfg.d_model),
            nn.ELU(inplace=True),
            nn.Dropout(cfg.head_dropout * 0.5),
        )
        self.spatial_coord_projector = nn.Linear(4, self.cfg.d_model)
        self.spatial_token_norm = nn.LayerNorm(self.cfg.d_model)
        self.spatial_token_ff = nn.Sequential(
            nn.LayerNorm(self.cfg.d_model),
            nn.Linear(self.cfg.d_model, max(self.cfg.d_model * 2, 256)),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout * 0.5),
            nn.Linear(max(self.cfg.d_model * 2, 256), self.cfg.d_model),
            nn.Dropout(cfg.head_dropout * 0.5),
        )
        nn.init.normal_(self.spatial_coord_projector.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.spatial_coord_projector.bias)

        self.visual_frame_norm = nn.LayerNorm(self.cfg.d_model)
        self.visual_frame_ff = nn.Sequential(
            nn.LayerNorm(self.cfg.d_model),
            nn.Linear(self.cfg.d_model, max(self.cfg.d_model * 2, 256)),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout * 0.5),
            nn.Linear(max(self.cfg.d_model * 2, 256), self.cfg.d_model),
            nn.Dropout(cfg.head_dropout * 0.5),
        )

        self.visual_motion_fuser = nn.Sequential(
            nn.LayerNorm(self.cfg.d_model * 2),
            nn.Linear(self.cfg.d_model * 2, self.cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout * 0.5),
            nn.Linear(self.cfg.d_model, self.cfg.d_model),
            nn.Dropout(cfg.head_dropout * 0.5),
        )
        self.visual_motion_gate = nn.Linear(self.cfg.d_model * 2, self.cfg.d_model)
        self.visual_motion_norm = nn.LayerNorm(self.cfg.d_model)
        nn.init.zeros_(self.visual_motion_gate.weight)
        nn.init.constant_(self.visual_motion_gate.bias, -1.0)

        dt_hidden = max(16, min(64, self.cfg.d_model // 4))
        self.dt_encoder = nn.Sequential(
            nn.Linear(1, dt_hidden),
            nn.ELU(inplace=True),
            nn.Linear(dt_hidden, self.cfg.d_model),
        )

        action_hidden = max(4, min(32, cfg.num_bin * 2, self.cfg.d_model // 4))
        self.last_action_encoder = nn.Sequential(
            nn.Linear(cfg.num_bin, action_hidden),
            nn.ELU(inplace=True),
            nn.Dropout(LAST_ACTION_EMBEDDING_DROPOUT),
            nn.Linear(action_hidden, self.cfg.d_model),
        )
        self.context_delta = nn.Sequential(
            nn.LayerNorm(self.cfg.d_model * 2),
            nn.Linear(self.cfg.d_model * 2, self.cfg.d_model),
            nn.ELU(inplace=True),
            nn.Dropout(cfg.head_dropout * 0.5),
            nn.Linear(self.cfg.d_model, self.cfg.d_model),
        )
        self.context_gate = nn.Linear(self.cfg.d_model * 3, self.cfg.d_model)
        nn.init.zeros_(self.context_gate.weight)
        nn.init.constant_(self.context_gate.bias, -3.0)
        self.fused_frame_norm = nn.LayerNorm(self.cfg.d_model)

        self.temporal_frame_count = max(int(self.cfg.seq_len), int(self.cfg.temporal_context))
        self.temporal_position_count = self.temporal_frame_count
        self.temporal_pos_embed = nn.Parameter(torch.empty(1, self.temporal_position_count, self.cfg.d_model))
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
        decoder_heads = _largest_valid_head_count(self.cfg.d_model, 4)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.cfg.d_model,
            nhead=decoder_heads,
            dim_feedforward=max(self.cfg.d_model * 4, 512),
            dropout=cfg.head_dropout * 0.5,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.horizon_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=2,
            norm=nn.LayerNorm(self.cfg.d_model),
        )
        self.current_memory_embed = nn.Parameter(torch.empty(1, 1, self.cfg.d_model))
        self.history_memory_embed = nn.Parameter(torch.empty(1, 1, self.cfg.d_model))
        self.recent_visual_age_embed = nn.Parameter(
            torch.empty(self._max_recent_visual_frames(), self.cfg.d_model)
        )
        nn.init.normal_(self.current_memory_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.history_memory_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.recent_visual_age_embed, mean=0.0, std=0.02)
        horizon_hidden = max(64, self.cfg.d_model // 2)
        self.button_head = nn.Sequential(
            nn.LayerNorm(self.cfg.d_model),
            nn.Linear(self.cfg.d_model, horizon_hidden),
            nn.ELU(inplace=True),
            nn.Dropout(cfg.head_dropout * 0.5),
            nn.Linear(horizon_hidden, self.cfg.num_bin),
        )
        self.persistence_logit_gate = nn.Parameter(torch.tensor(0.0))
        final_button_layer = self.button_head[-1]
        if isinstance(final_button_layer, nn.Linear):
            nn.init.zeros_(final_button_layer.bias)

    def _initial_temporal_state(
        self,
        batch_size: int,
        *unused_spatial_shape: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            0,
            self.cfg.d_model,
            device=device,
            dtype=dtype,
        )

    def _prepare_temporal_state(
        self,
        state: Optional[TemporalState],
        batch_size: int,
        *unused_spatial_shape: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if state is None or state.hidden_state is None:
            return self._initial_temporal_state(batch_size, device=device, dtype=dtype)

        hidden = state.hidden_state.to(device=device, dtype=dtype)
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(1)
        if hidden.dim() != 3:
            raise ValueError(f"Expected temporal token state [B,N,D], got {tuple(hidden.shape)}.")
        if hidden.size(0) != batch_size:
            raise ValueError(f"Expected temporal state batch {batch_size}, got {hidden.size(0)}.")
        if hidden.size(-1) != self.cfg.d_model:
            raise ValueError(f"Expected temporal token dim {self.cfg.d_model}, got {hidden.size(-1)}.")
        max_tokens = self._max_context_tokens()
        if hidden.size(1) > max_tokens:
            hidden = hidden[:, -max_tokens:]
        return hidden

    def _max_context_tokens(self) -> int:
        return int(self.cfg.temporal_context)

    def _max_recent_visual_frames(self) -> int:
        return max(1, int(self.cfg.recent_spatial_context))

    def _prepare_visual_state(
        self,
        state: Optional[TemporalState],
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if state is None or state.visual_tokens is None:
            return torch.zeros(
                batch_size,
                0,
                int(self.cfg.spatial_token_count),
                self.cfg.d_model,
                device=device,
                dtype=dtype,
            )

        visual = state.visual_tokens.to(device=device, dtype=dtype)
        if visual.dim() == 3:
            visual = visual.unsqueeze(1)
        if visual.dim() != 4:
            raise ValueError(f"Expected visual token state [B,T,K,D], got {tuple(visual.shape)}.")
        if int(visual.size(0)) != int(batch_size):
            raise ValueError(f"Expected visual state batch {batch_size}, got {visual.size(0)}.")
        if int(visual.size(2)) != int(self.cfg.spatial_token_count):
            raise ValueError(f"Expected {self.cfg.spatial_token_count} visual tokens, got {visual.size(2)}.")
        if int(visual.size(-1)) != int(self.cfg.d_model):
            raise ValueError(f"Expected visual token dim {self.cfg.d_model}, got {visual.size(-1)}.")
        max_prior_frames = max(0, self._max_recent_visual_frames() - 1)
        if max_prior_frames <= 0:
            return visual[:, :0]
        return visual[:, -max_prior_frames:]

    def _flatten_frame_tokens(self, frame_tokens: torch.Tensor) -> torch.Tensor:
        if frame_tokens.dim() == 3:
            if int(frame_tokens.size(-1)) != int(self.cfg.d_model):
                raise ValueError(f"Expected token dim {self.cfg.d_model}, got {frame_tokens.size(-1)}.")
            return frame_tokens
        if frame_tokens.dim() != 4:
            raise ValueError(f"Expected frame tokens [B,T,K,D], got {tuple(frame_tokens.shape)}.")
        b, t, k, d = frame_tokens.shape
        if int(d) != int(self.cfg.d_model):
            raise ValueError(f"Expected token dim {self.cfg.d_model}, got {d}.")
        return frame_tokens.reshape(b, t * k, d)

    def _causal_attention_mask(self, token_count: int, device: torch.device) -> torch.Tensor:
        token_idx = torch.arange(int(token_count), device=device)
        return token_idx.unsqueeze(0) > token_idx.unsqueeze(1)

    def _encode_temporal_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"Expected temporal tokens [B,T,D], got {tuple(tokens.shape)}.")
        token_count = int(tokens.size(1))
        if token_count <= 0:
            raise ValueError("Temporal transformer requires at least one token.")
        if token_count > self.temporal_position_count:
            raise ValueError(
                f"Temporal token length {token_count} exceeds configured position capacity "
                f"{self.temporal_position_count}."
            )
        pos = self.temporal_pos_embed[:, :token_count].to(device=tokens.device, dtype=tokens.dtype)
        causal_mask = self._causal_attention_mask(token_count, tokens.device)
        encoded = self.temporal_encoder(tokens + pos, mask=causal_mask, is_causal=False)
        return self.temporal_norm(encoded)

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

    def _spatial_pool_shape(self) -> Tuple[int, int]:
        token_count = max(1, int(self.cfg.spatial_token_count))
        rows = max(1, int(math.isqrt(token_count)))
        while rows > 1 and token_count % rows != 0:
            rows -= 1
        return rows, token_count // rows

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
        return cell_tokens + pos

    def _visual_cells_from_masked_frames(self, frames: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = frames.shape
        x = frames.reshape(b * t, c, h, w)
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        spatial_feats = self.fastvit_mixer(self.spatial_encoder(x))
        pool_h, pool_w = self._spatial_pool_shape()
        spatial_feats = F.adaptive_avg_pool2d(spatial_feats, (pool_h, pool_w))
        cell_tokens = self._project_spatial_features(spatial_feats)
        return cell_tokens.reshape(b, t, pool_h * pool_w, self.cfg.d_model)

    def _spatial_tokens_from_cells(self, cell_tokens: torch.Tensor) -> torch.Tensor:
        b, t, k, d = cell_tokens.shape
        expected_tokens = int(self.cfg.spatial_token_count)
        if int(k) != expected_tokens:
            raise ValueError(f"Expected {expected_tokens} averaged spatial tokens, got {k}.")
        if int(d) != int(self.cfg.d_model):
            raise ValueError(f"Expected token dim {self.cfg.d_model}, got {d}.")
        spatial_tokens = self.spatial_token_norm(cell_tokens)
        spatial_tokens = spatial_tokens + self.spatial_token_ff(spatial_tokens)
        return spatial_tokens

    def _visual_tokens_from_masked_frames(self, frames: torch.Tensor) -> torch.Tensor:
        return self._spatial_tokens_from_cells(self._visual_cells_from_masked_frames(frames))

    def _visual_frame_tokens_from_cells(self, cell_tokens: torch.Tensor) -> torch.Tensor:
        b, t, k, d = cell_tokens.shape
        if int(d) != int(self.cfg.d_model):
            raise ValueError(f"Expected token dim {self.cfg.d_model}, got {d}.")
        frame_token = self.visual_frame_norm(cell_tokens.mean(dim=2))
        frame_token = frame_token + self.visual_frame_ff(frame_token)
        return frame_token

    def _motion_frame_tokens_from_spatial_tokens(
        self,
        spatial_tokens: torch.Tensor,
        previous_spatial_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if spatial_tokens.dim() != 4:
            raise ValueError(f"Expected spatial tokens [B,T,K,D], got {tuple(spatial_tokens.shape)}.")
        b, t, k, d = spatial_tokens.shape
        delta = torch.zeros_like(spatial_tokens)
        if previous_spatial_tokens is not None and int(previous_spatial_tokens.size(1)) > 0:
            previous = previous_spatial_tokens[:, -1:].to(
                device=spatial_tokens.device,
                dtype=spatial_tokens.dtype,
            )
            if int(previous.size(0)) != b or int(previous.size(2)) != k or int(previous.size(3)) != d:
                raise ValueError(
                    f"Previous spatial token shape {tuple(previous.shape)} does not match "
                    f"current shape {(b, t, k, d)}."
                )
            delta[:, :1] = spatial_tokens[:, :1] - previous
        if t > 1:
            delta[:, 1:] = spatial_tokens[:, 1:] - spatial_tokens[:, :-1]

        return delta.mean(dim=2)

    def _add_visual_motion_context(
        self,
        visual_frame_tokens: torch.Tensor,
        spatial_tokens: torch.Tensor,
        previous_spatial_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        motion_tokens = self._motion_frame_tokens_from_spatial_tokens(
            spatial_tokens,
            previous_spatial_tokens=previous_spatial_tokens,
        )
        context = torch.cat([visual_frame_tokens, motion_tokens], dim=-1)
        motion_delta = self.visual_motion_fuser(context)
        gate = torch.sigmoid(self.visual_motion_gate(context))
        return self.visual_motion_norm(visual_frame_tokens + gate * motion_delta)

    def _visual_frame_tokens_from_masked_frames(self, frames: torch.Tensor) -> torch.Tensor:
        cell_tokens = self._visual_cells_from_masked_frames(frames)
        spatial_tokens = self._spatial_tokens_from_cells(cell_tokens)
        visual_frame_tokens = self._visual_frame_tokens_from_cells(spatial_tokens)
        return self._add_visual_motion_context(visual_frame_tokens, spatial_tokens)

    def _fuse_frame_context(
        self,
        visual_token: torch.Tensor,
        dt: Optional[torch.Tensor],
        prev_action: Optional[torch.Tensor],
        prev_action_scale: float = 1.0,
    ) -> torch.Tensor:
        b, t = int(visual_token.size(0)), int(visual_token.size(1))
        dt_feat = self._dt_features(dt, b, t, device=visual_token.device, dtype=visual_token.dtype)
        action_feat = self._last_action_features(prev_action, b, t, device=visual_token.device, dtype=visual_token.dtype)
        action_scale = torch.as_tensor(prev_action_scale, device=action_feat.device, dtype=action_feat.dtype)
        action_feat = action_feat * action_scale
        context = torch.cat([action_feat, dt_feat], dim=-1)
        context_delta = self.context_delta(context)
        gate = torch.sigmoid(self.context_gate(torch.cat([visual_token, action_feat, dt_feat], dim=-1)))
        return self.fused_frame_norm(visual_token + gate * context_delta)

    def _fused_frame_tokens(
        self,
        frames: torch.Tensor,
        dt: Optional[torch.Tensor],
        prev_action: Optional[torch.Tensor],
        prev_action_scale: float = 1.0,
    ) -> torch.Tensor:
        visual_token = self._visual_frame_tokens_from_masked_frames(frames)
        return self._fuse_frame_context(visual_token, dt, prev_action, prev_action_scale=prev_action_scale)

    def _visual_memory_and_frame_tokens(
        self,
        frames: torch.Tensor,
        dt: Optional[torch.Tensor],
        prev_action: Optional[torch.Tensor],
        prev_action_scale: float = 1.0,
        previous_visual_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cell_tokens = self._visual_cells_from_masked_frames(frames)
        current_visual_tokens = self._spatial_tokens_from_cells(cell_tokens)
        visual_frame_tokens = self._visual_frame_tokens_from_cells(current_visual_tokens)
        visual_frame_tokens = self._add_visual_motion_context(
            visual_frame_tokens,
            current_visual_tokens,
            previous_spatial_tokens=previous_visual_tokens,
        )
        fused_frame_tokens = self._fuse_frame_context(
            visual_frame_tokens,
            dt,
            prev_action,
            prev_action_scale=prev_action_scale,
        )
        return current_visual_tokens, fused_frame_tokens

    def _recent_visual_memory(
        self,
        visual_tokens: torch.Tensor,
        *,
        total_frames: int,
        current_frame_count: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if visual_tokens.dim() != 4:
            raise ValueError(f"Expected visual tokens [B,T,K,D], got {tuple(visual_tokens.shape)}.")
        b, visual_frames, visual_tokens_per_frame, d = visual_tokens.shape
        total_frames = int(total_frames)
        current_frame_count = int(current_frame_count)
        if visual_frames <= 0:
            raise ValueError("Recent visual memory requires at least one visual frame.")
        if visual_frames > total_frames:
            visual_tokens = visual_tokens[:, -total_frames:]
            visual_frames = int(visual_tokens.size(1))

        recent_frames = min(self._max_recent_visual_frames(), total_frames)
        device = visual_tokens.device
        query_frame_idx = torch.arange(
            total_frames - current_frame_count,
            total_frames,
            device=device,
        )
        recent_offsets = torch.arange(recent_frames, device=device) - (recent_frames - 1)
        recent_frame_idx = query_frame_idx.unsqueeze(1) + recent_offsets.unsqueeze(0)
        visual_start_idx = total_frames - int(visual_frames)

        visual_gather_idx = (recent_frame_idx - visual_start_idx).clamp(0, int(visual_frames) - 1).long()
        visual_source = visual_tokens.unsqueeze(1).expand(
            b,
            current_frame_count,
            int(visual_frames),
            int(visual_tokens_per_frame),
            d,
        )
        gather_idx = visual_gather_idx.view(1, current_frame_count, recent_frames, 1, 1).expand(
            b,
            current_frame_count,
            recent_frames,
            int(visual_tokens_per_frame),
            d,
        )
        recent_memory = torch.gather(visual_source, dim=2, index=gather_idx)
        age_embed = self.recent_visual_age_embed[-recent_frames:].to(
            device=recent_memory.device,
            dtype=recent_memory.dtype,
        )
        recent_memory = recent_memory + age_embed.view(1, 1, recent_frames, 1, d)
        recent_memory = recent_memory.reshape(
            b,
            current_frame_count,
            recent_frames * int(visual_tokens_per_frame),
            d,
        )

        recent_valid = (recent_frame_idx >= visual_start_idx) & (recent_frame_idx >= 0)
        recent_padding_mask = ~recent_valid.unsqueeze(-1).expand(
            current_frame_count,
            recent_frames,
            int(visual_tokens_per_frame),
        ).reshape(current_frame_count, recent_frames * int(visual_tokens_per_frame))

        source_frame_idx = torch.arange(total_frames, device=device)
        future_or_current = source_frame_idx.unsqueeze(0) >= query_frame_idx.unsqueeze(1)
        visual_available_start = torch.maximum(
            query_frame_idx - (recent_frames - 1),
            torch.full_like(query_frame_idx, visual_start_idx),
        )
        recent_compressed_duplicate = (
            (source_frame_idx.unsqueeze(0) >= visual_available_start.unsqueeze(1))
            & (source_frame_idx.unsqueeze(0) < query_frame_idx.unsqueeze(1))
        )
        history_padding_mask = future_or_current | recent_compressed_duplicate
        return recent_memory, recent_padding_mask, history_padding_mask

    def _horizon_button_logits(
        self,
        encoded_tokens: torch.Tensor,
        current_frame_count: Optional[int] = None,
        prev_action: Optional[torch.Tensor] = None,
        current_visual_tokens: Optional[torch.Tensor] = None,
        persistence_scale: float = 1.0,
    ) -> torch.Tensor:
        if encoded_tokens.size(-1) != self.cfg.d_model:
            raise ValueError(f"Expected final feature dim {self.cfg.d_model}, got {encoded_tokens.size(-1)}.")

        if encoded_tokens.dim() == 2:
            memory = encoded_tokens.unsqueeze(1)
            queries = self.horizon_queries.to(device=encoded_tokens.device, dtype=encoded_tokens.dtype)
            queries = encoded_tokens.unsqueeze(1) + queries.unsqueeze(0).expand(memory.size(0), -1, -1)
            decoded = self.horizon_decoder(tgt=queries, memory=memory)
            logits = self.button_head(decoded)
            prior = self._persistence_prior_logits(
                prev_action,
                batch_size=encoded_tokens.size(0),
                time_steps=1,
                device=logits.device,
                dtype=logits.dtype,
                persistence_scale=persistence_scale,
            )
            return logits + prior[:, 0]

        if encoded_tokens.dim() != 3:
            raise ValueError(f"Expected encoded tokens [B,N,D] or [B,D], got {tuple(encoded_tokens.shape)}.")

        if current_visual_tokens is None:
            return self._compressed_horizon_button_logits(
                encoded_tokens,
                current_frame_count=current_frame_count,
                prev_action=prev_action,
                persistence_scale=persistence_scale,
            )

        if current_visual_tokens.dim() != 4:
            raise ValueError(f"Expected current visual tokens [B,T,K,D], got {tuple(current_visual_tokens.shape)}.")

        b, token_count, d = encoded_tokens.shape
        if int(current_visual_tokens.size(0)) != b or int(current_visual_tokens.size(-1)) != d:
            raise ValueError(
                f"Current visual token shape {tuple(current_visual_tokens.shape)} does not match encoded "
                f"batch/dim {(b, d)}."
            )
        total_frames = int(token_count)
        if current_frame_count is None:
            current_frame_count = int(current_visual_tokens.size(1))
        current_frame_count = max(1, min(int(current_frame_count), total_frames))
        if int(current_visual_tokens.size(1)) < current_frame_count:
            raise ValueError(
                f"Need at least {current_frame_count} current visual frames, got {current_visual_tokens.size(1)}."
            )

        recent_visual_memory, recent_visual_mask, history_mask = self._recent_visual_memory(
            current_visual_tokens,
            total_frames=total_frames,
            current_frame_count=current_frame_count,
        )
        frame_anchor = encoded_tokens[:, -current_frame_count:, :]
        horizon_queries = self.horizon_queries.to(device=encoded_tokens.device, dtype=encoded_tokens.dtype)
        tgt = frame_anchor.unsqueeze(2) + horizon_queries.reshape(1, 1, self.cfg.prediction_horizon, d)
        tgt = tgt.reshape(b * current_frame_count, self.cfg.prediction_horizon, d)

        memory_key_padding_mask = torch.cat([recent_visual_mask, history_mask], dim=1)
        memory_key_padding_mask = (
            memory_key_padding_mask.unsqueeze(0)
            .expand(b, -1, -1)
            .reshape(b * current_frame_count, -1)
        )

        current_memory = recent_visual_memory + self.current_memory_embed.to(
            device=recent_visual_memory.device,
            dtype=recent_visual_memory.dtype,
        )
        history_memory = encoded_tokens.unsqueeze(1).expand(b, current_frame_count, total_frames, d)
        history_memory = history_memory + self.history_memory_embed.to(
            device=encoded_tokens.device,
            dtype=encoded_tokens.dtype,
        )
        memory = torch.cat([current_memory, history_memory], dim=2)
        memory = memory.reshape(b * current_frame_count, memory.size(2), d)

        decoded = self.horizon_decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=False,
            memory_is_causal=False,
        )
        logits = self.button_head(decoded).reshape(
            b,
            current_frame_count,
            self.cfg.prediction_horizon,
            self.cfg.num_bin,
        )
        prior = self._persistence_prior_logits(
            prev_action,
            batch_size=b,
            time_steps=current_frame_count,
            device=logits.device,
            dtype=logits.dtype,
            persistence_scale=persistence_scale,
        )
        return logits + prior

    def _compressed_horizon_button_logits(
        self,
        encoded_tokens: torch.Tensor,
        current_frame_count: Optional[int],
        prev_action: Optional[torch.Tensor],
        persistence_scale: float = 1.0,
    ) -> torch.Tensor:
        b, token_count, d = encoded_tokens.shape
        total_frames = int(token_count)
        if current_frame_count is None:
            current_frame_count = total_frames
        current_frame_count = max(1, min(int(current_frame_count), total_frames))

        frame_anchor = encoded_tokens[:, -current_frame_count:, :]
        horizon_queries = self.horizon_queries.to(device=encoded_tokens.device, dtype=encoded_tokens.dtype)
        tgt = frame_anchor.unsqueeze(2) + horizon_queries.reshape(1, 1, self.cfg.prediction_horizon, d)
        tgt = tgt.reshape(b, current_frame_count * self.cfg.prediction_horizon, d)

        query_frame_idx = torch.arange(
            total_frames - current_frame_count,
            total_frames,
            device=encoded_tokens.device,
        ).repeat_interleave(int(self.cfg.prediction_horizon))
        source_frame_idx = torch.arange(total_frames, device=encoded_tokens.device)
        tgt_mask = query_frame_idx.unsqueeze(0) > query_frame_idx.unsqueeze(1)
        memory_mask = source_frame_idx.unsqueeze(0) > query_frame_idx.unsqueeze(1)

        decoded = self.horizon_decoder(
            tgt=tgt,
            memory=encoded_tokens,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_is_causal=False,
            memory_is_causal=False,
        )
        logits = self.button_head(decoded).reshape(b, current_frame_count, self.cfg.prediction_horizon, self.cfg.num_bin)
        prior = self._persistence_prior_logits(
            prev_action,
            batch_size=b,
            time_steps=current_frame_count,
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
                action_values = action_values.view(batch_size, 1, self.cfg.num_bin).expand(batch_size, time_steps, self.cfg.num_bin)
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

    def _normalize_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0
        return frames.clamp(0.0, 1.0)

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
                action_values = action_values.view(batch_size, 1, self.cfg.num_bin).expand(batch_size, time_steps, self.cfg.num_bin)
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

    def _apply_masks(self, frames: torch.Tensor) -> torch.Tensor:
        h, w = frames.shape[-2:]
        masked_frames = frames.clone()

        # 1. Mask the Bottom Left HUD (Speedometer)
        hud_y1 = int(h * 0.96)
        masked_frames[..., hud_y1:, :] = 0.0

        # 2. Mask the Minimap (Mid-Right)
        map_y1, map_y2 = int(h * 0.05), int(h * 0.2)
        map_x1 = int(w * 0.75)
        masked_frames[..., map_y1:map_y2, map_x1:] = 0.0

        # 3. Top Left Roblox UI
        roblox_ui_y2 = int(h * 0.1)
        roblox_ui_x2 = int(w * 0.1)
        masked_frames[..., :roblox_ui_y2, :roblox_ui_x2] = 0.0

        return masked_frames

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
        b, t, c, h, w = frames.shape
        if c != 3:
            raise ValueError(f"Expected RGB frames with shape [B,T,3,H,W], got {tuple(frames.shape)}.")
        
        frames = self._normalize_frames(frames)
        frames = self._apply_masks(frames)

        prior_visual_tokens = self._prepare_visual_state(
            state,
            b,
            device=frames.device,
            dtype=frames.dtype,
        )
        current_visual_tokens, current_tokens = self._visual_memory_and_frame_tokens(
            frames,
            dt,
            prev_action,
            prev_action_scale=prev_action_scale,
            previous_visual_tokens=prior_visual_tokens,
        )
        prior_tokens = self._prepare_temporal_state(
            state,
            b,
            device=frames.device,
            dtype=current_tokens.dtype,
        )
        if t > self.temporal_frame_count:
            raise ValueError(
                f"Input sequence length {t} exceeds temporal frame capacity {self.temporal_frame_count}."
            )
        if prior_tokens.size(1) > 0:
            keep_prior = max(0, self.temporal_position_count - int(current_tokens.size(1)))
            prior_tokens = prior_tokens[:, -keep_prior:] if keep_prior > 0 else prior_tokens[:, :0]
            if prior_visual_tokens.size(1) > 0:
                keep_visual = min(int(prior_visual_tokens.size(1)), int(prior_tokens.size(1)))
                prior_visual_tokens = prior_visual_tokens[:, -keep_visual:] if keep_visual > 0 else prior_visual_tokens[:, :0]
            current_tokens = torch.cat([prior_tokens, current_tokens], dim=1)
            if prior_visual_tokens.size(1) > 0:
                current_visual_tokens = torch.cat([prior_visual_tokens, current_visual_tokens], dim=1)
        temporal_out = self._encode_temporal_tokens(current_tokens)
        
        button = self._horizon_button_logits(
            temporal_out,
            current_frame_count=t,
            prev_action=prev_action,
            current_visual_tokens=current_visual_tokens,
            persistence_scale=persistence_scale,
        )
        
        step_button = button[:, :, 0]
        output = PolicyOutput(
            button_logits=step_button,
            horizon_button_logits=button,
            future_button_logits={
                int(offset): button[:, :, idx]
                for idx, offset in enumerate(self.cfg.prediction_horizon_offsets)
            },
        )
        return output

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
        b = frame.shape[0]

        # 1. Normalize and mask out the car layout exactly once
        frame_norm = self._normalize_frames(frame)
        masked_frame = self._apply_masks(frame_norm)

        prior_visual_tokens = self._prepare_visual_state(
            state,
            b,
            device=frame.device,
            dtype=masked_frame.dtype,
        )
        current_visual_tokens, current_tokens = self._visual_memory_and_frame_tokens(
            masked_frame.unsqueeze(1),
            dt,
            prev_action,
            prev_action_scale=prev_action_scale,
            previous_visual_tokens=prior_visual_tokens,
        )
        prior_tokens = self._prepare_temporal_state(
            state,
            b,
            device=frame.device,
            dtype=current_tokens.dtype,
        )
        token_memory = torch.cat([prior_tokens, current_tokens], dim=1)
        token_memory = token_memory[:, -self._max_context_tokens():]
        visual_memory = torch.cat([prior_visual_tokens, current_visual_tokens], dim=1)
        visual_memory = visual_memory[:, -min(self._max_recent_visual_frames(), int(token_memory.size(1))):]
        temporal_out = self._encode_temporal_tokens(token_memory)

        # 5. Project to action space values
        button = self._horizon_button_logits(
            temporal_out,
            current_frame_count=1,
            prev_action=prev_action,
            current_visual_tokens=visual_memory,
            persistence_scale=persistence_scale,
        )[:, 0]

        squeezed = PolicyOutput(
            button_logits=button[:, 0],
            horizon_button_logits=button,
            future_button_logits={
                int(offset): button[:, idx]
                for idx, offset in enumerate(self.cfg.prediction_horizon_offsets)
            },
        )

        # 6. Detach hidden state to prevent backpropagation graph memory leaks
        new_state = TemporalState(
            hidden_state=token_memory.detach(),
            visual_tokens=visual_memory.detach(),
            previous_frame=None,
        )
        return squeezed, new_state
