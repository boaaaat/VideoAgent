from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from action_space import (
    game_data_root,
    get_key_names,
    get_mouse_button_names,
    normalize_game_name,
    selected_game as ACTION_SELECTED_GAME,
)


CNN_FEATURE_CHANNELS = 72
POLICY_INPUT_CHANNELS = 3
TEMPORAL_RNN_LAYERS = 1
LAST_ACTION_EMBEDDING_DROPOUT = 0.25
LAST_ACTION_FEATURE_SCALE = 1.0
GROUP_NORM_GROUPS = 8
DEFAULT_HORIZON_OFFSETS = (1, 2, 3, 5, 7, 10, 13, 16, 20, 24)


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
    prediction_horizon: int = 1
    prediction_horizon_offsets: Optional[Sequence[int]] = None

    key_names: Optional[List[str]] = None
    mouse_button_names: Optional[List[str]] = None

    d_model: int = 256
    spatial_dropout = 0.10
    head_dropout = 0.20
    zoneout = 0.10

    pooling: Tuple[int, int] = (16, 16)
    pool_heads: int = 4

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
        self.zoneout = float(min(max(self.zoneout, 0.0), 0.9))
        pool_h, pool_w = self.pooling
        self.pooling = (max(1, int(pool_h)), max(1, int(pool_w)))
        self.pool_heads = _largest_valid_head_count(CNN_FEATURE_CHANNELS, int(self.pool_heads))

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


class ConvGRUCell(nn.Module):
    """
    A GRU cell that replaces standard Linear matrix multiplications with Conv2d loops,
    preserving structural 2D coordinates across time.
    """
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, zoneout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.zoneout = float(min(max(zoneout, 0.0), 0.9))
        padding = kernel_size // 2
        
        self.gates_conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=2 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
        self.candidate_conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.gates_conv(combined)
        r_gate, z_gate = torch.chunk(gates, 2, dim=1)
        
        r_gate = torch.sigmoid(r_gate)
        z_gate = torch.sigmoid(z_gate)
        
        combined_candidate = torch.cat([x, r_gate * h_prev], dim=1)
        candidate = torch.tanh(self.candidate_conv(combined_candidate))
        
        h_next = (1.0 - z_gate) * h_prev + z_gate * candidate
        if self.zoneout <= 0.0:
            return h_next
        if self.training:
            keep_previous = torch.rand_like(h_next) < self.zoneout
            return torch.where(keep_previous, h_prev, h_next)
        return (1.0 - self.zoneout) * h_next + self.zoneout * h_prev


class LearnedMultiHeadSpatialPool2d(nn.Module):
    """
    Learns spatial attention pools while preserving an AdaptiveAvgPool2d-style output shape.
    """
    def __init__(self, channels: int, output_size: Sequence[int], num_heads: int):
        super().__init__()
        output_h, output_w = output_size
        self.channels = int(channels)
        self.output_size = (max(1, int(output_h)), max(1, int(output_w)))
        self.num_heads = _largest_valid_head_count(self.channels, int(num_heads))
        self.head_dim = self.channels // self.num_heads
        self.num_slots = self.output_size[0] * self.output_size[1]

        self.attn_logits = nn.Conv2d(
            self.channels,
            self.num_heads * self.num_slots,
            kernel_size=1,
            bias=True,
        )
        nn.init.zeros_(self.attn_logits.weight)
        nn.init.zeros_(self.attn_logits.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, spatial_h, spatial_w = x.shape
        if c != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {c}.")

        spatial_size = spatial_h * spatial_w
        logits = self.attn_logits(x).reshape(b, self.num_heads, self.num_slots, spatial_size)
        weights = torch.softmax(logits, dim=-1)

        values = x.reshape(b, self.num_heads, self.head_dim, spatial_size)
        pooled = torch.einsum("bhsn,bhdn->bhsd", weights, values)
        pooled = pooled.permute(0, 1, 3, 2).reshape(b, c, self.output_size[0], self.output_size[1])
        return pooled


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
        
        # ConvGRU tracking state
        self.feat_channels = CNN_FEATURE_CHANNELS
        self.temporal_rnns = nn.ModuleList(
            [
                ConvGRUCell(
                    input_dim=self.feat_channels,
                    hidden_dim=self.feat_channels,
                    kernel_size=3,
                    zoneout=cfg.zoneout,
                )
                for _ in range(TEMPORAL_RNN_LAYERS)
            ]
        )
        
        # Preserve a real fixed spatial grid for the classifier head.
        self.pool = nn.AdaptiveAvgPool2d(cfg.pooling)
        
        self.fc_features = nn.Sequential(
            nn.Linear(self.feat_channels * cfg.pooling[0] * cfg.pooling[1], self.cfg.d_model),
            nn.ELU(inplace=True),
            nn.Dropout(cfg.head_dropout)
        )
        self.dt_encoder = nn.Linear(1, self.cfg.d_model)
        nn.init.zeros_(self.dt_encoder.weight)
        nn.init.zeros_(self.dt_encoder.bias)

        action_hidden = max(4, min(32, cfg.num_bin * 2, self.cfg.d_model // 4))
        self.last_action_encoder = nn.Sequential(
            nn.Linear(cfg.num_bin, action_hidden),
            nn.ELU(inplace=True),
            nn.Dropout(LAST_ACTION_EMBEDDING_DROPOUT),
            nn.Linear(action_hidden, self.cfg.d_model),
        )
        final_action_layer = self.last_action_encoder[-1]
        if isinstance(final_action_layer, nn.Linear):
            nn.init.zeros_(final_action_layer.weight)
            nn.init.zeros_(final_action_layer.bias)

        self.head_fusion = nn.Sequential(
            nn.LayerNorm(self.cfg.d_model * 3),
            nn.Linear(self.cfg.d_model * 3, self.cfg.d_model),
            nn.ELU(inplace=True),
        )
        
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
        horizon_hidden = max(64, self.cfg.d_model // 2)
        self.button_head = nn.Sequential(
            nn.LayerNorm(self.cfg.d_model),
            nn.Linear(self.cfg.d_model, horizon_hidden),
            nn.ELU(inplace=True),
            nn.Dropout(cfg.head_dropout * 0.5),
            nn.Linear(horizon_hidden, self.cfg.num_bin),
        )
        final_button_layer = self.button_head[-1]
        if isinstance(final_button_layer, nn.Linear):
            nn.init.constant_(final_button_layer.bias, -1.0)

    def _initial_temporal_state(
        self,
        batch_size: int,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(
            len(self.temporal_rnns),
            batch_size,
            self.feat_channels,
            height,
            width,
            device=device,
            dtype=dtype,
        )

    def _prepare_temporal_state(
        self,
        state: Optional[TemporalState],
        batch_size: int,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if state is None or state.hidden_state is None:
            return self._initial_temporal_state(batch_size, height, width, device=device, dtype=dtype)

        hidden = state.hidden_state.to(device=device, dtype=dtype)
        if hidden.dim() == 4:
            first = hidden
            remaining = self._initial_temporal_state(batch_size, height, width, device=device, dtype=dtype)[1:]
            return torch.cat([first.unsqueeze(0), remaining], dim=0)
        if hidden.dim() != 5:
            raise ValueError(f"Expected temporal hidden state [L,B,C,H,W], got {tuple(hidden.shape)}.")
        if hidden.size(0) != len(self.temporal_rnns):
            raise ValueError(f"Expected {len(self.temporal_rnns)} temporal layers, got {hidden.size(0)}.")
        return hidden

    def _temporal_step(self, x_t: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        next_states = []
        for layer_idx, rnn in enumerate(self.temporal_rnns):
            x_t = rnn(x_t, hidden_state[layer_idx])
            next_states.append(x_t)
        return x_t, torch.stack(next_states, dim=0)

    def _horizon_button_logits(self, features: torch.Tensor) -> torch.Tensor:
        if features.size(-1) != self.cfg.d_model:
            raise ValueError(f"Expected final feature dim {self.cfg.d_model}, got {features.size(-1)}.")
        leading_shape = tuple(features.shape[:-1])
        memory = features.reshape(-1, 1, self.cfg.d_model)
        queries = self.horizon_queries.to(device=features.device, dtype=features.dtype)
        queries = queries.unsqueeze(0).expand(memory.size(0), -1, -1)
        decoded = self.horizon_decoder(tgt=queries, memory=memory)
        logits = self.button_head(decoded)
        return logits.reshape(*leading_shape, self.cfg.prediction_horizon, self.cfg.num_bin)

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

        # 1. Mask the Car (Center)
        car_y1, car_y2 = int(h * self.car_y_min_pct), int(h * self.car_y_max_pct)
        car_x1, car_x2 = int(w * self.car_x_min_pct), int(w * self.car_x_max_pct)
        masked_frames[..., car_y1:car_y2, car_x1:car_x2] = 0.0

        # 2. Mask the Bottom HUD (Speedometer, Gear, etc.)
        hud_y1 = int(h * 0.80)
        masked_frames[..., hud_y1:, :] = 0.0

        # 3. Mask the Minimap (Mid-Right)
        map_y1, map_y2 = int(h * 0.05), int(h * 0.2)
        map_x1 = int(w * 0.75)
        masked_frames[..., map_y1:map_y2, map_x1:] = 0.0

        # 4. Top Left Roblox UI
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
    ):
        b, t, c, h, w = frames.shape
        if c != 3:
            raise ValueError(f"Expected RGB frames with shape [B,T,3,H,W], got {tuple(frames.shape)}.")
        
        frames = self._normalize_frames(frames)
        frames = self._apply_masks(frames)
        
        # 1. Spatial Processing
        x = frames.reshape(b * t, POLICY_INPUT_CHANNELS, h, w)
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        spatial_feats = self.spatial_encoder(x)
        _, cf, hf, wf = spatial_feats.shape
        spatial_feats = spatial_feats.reshape(b, t, cf, hf, wf)
        
        # 2. Temporal ConvGRU Rollout Loop
        h_t = self._prepare_temporal_state(
            state,
            b,
            hf,
            wf,
            device=frames.device,
            dtype=spatial_feats.dtype,
        )
            
        temporal_outputs = []
        for step in range(t):
            temporal_feat, h_t = self._temporal_step(spatial_feats[:, step], h_t)
            temporal_outputs.append(temporal_feat.unsqueeze(1))
            
        temporal_out = torch.cat(temporal_outputs, dim=1)  # [B, T, C_feat, H_feat, W_feat]
        
        # 3. Linear Downsampling for Classifier Heads
        temporal_flat = temporal_out.reshape(b * t, cf, hf, wf)
        pooled = self.pool(temporal_flat)
        pooled_flat = pooled.reshape(pooled.shape[0], -1)
        
        visual_feat = self.fc_features(pooled_flat).reshape(b, t, self.cfg.d_model)
        dt_feat = self._dt_features(dt, b, t, device=frames.device, dtype=visual_feat.dtype)
        action_feat = self._last_action_features(prev_action, b, t, device=frames.device, dtype=visual_feat.dtype)
        fc_out = self.head_fusion(torch.cat([visual_feat, dt_feat, action_feat], dim=-1))
        
        # 4. Action Mapping Prediction
        button = self._horizon_button_logits(fc_out)
        
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
    ):
        b = frame.shape[0]

        # 1. Normalize and mask out the car layout exactly once
        frame_norm = self._normalize_frames(frame)
        masked_frame = self._apply_masks(frame_norm)

        # 2. Extract spatial primitives
        x = masked_frame
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        spatial_feat = self.spatial_encoder(x)
        cf, hf, wf = spatial_feat.shape[1:]

        # 3. Evaluate a single temporal rollout transition step
        h_t = self._prepare_temporal_state(
            state,
            b,
            hf,
            wf,
            device=frame.device,
            dtype=spatial_feat.dtype,
        )

        temporal_feat, new_hidden = self._temporal_step(spatial_feat, h_t)

        # 4. Map the new hidden states through the learned spatial pooling layout
        pooled = self.pool(temporal_feat)
        pooled_flat = pooled.reshape(b, -1)

        visual_feat = self.fc_features(pooled_flat)
        dt_feat = self._dt_features(dt, b, 1, device=frame.device, dtype=visual_feat.dtype).reshape(b, self.cfg.d_model)
        action_feat = self._last_action_features(prev_action, b, 1, device=frame.device, dtype=visual_feat.dtype).reshape(
            b,
            self.cfg.d_model,
        )
        fc_out = self.head_fusion(torch.cat([visual_feat, dt_feat, action_feat], dim=-1))

        # 5. Project to action space values
        button = self._horizon_button_logits(fc_out)

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
            hidden_state=new_hidden.detach(),
            previous_frame=None,
        )
        return squeezed, new_state
