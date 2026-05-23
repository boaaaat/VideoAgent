from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from action_space import (
    game_data_root,
    get_key_names,
    get_mouse_button_names,
    normalize_game_name,
    selected_game as ACTION_SELECTED_GAME,
)


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

    key_names: Optional[List[str]] = None
    mouse_button_names: Optional[List[str]] = None

    d_model: int = 192
    dropout: float = 0.10

    pooling = (5, 5)

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

        self.d_model = max(64, int(self.d_model))
        self.dropout = float(min(max(self.dropout, 0.0), 0.9))

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


class ResBlock(nn.Module):
    """ Lightweight 2D Residual block for regularizing spatial primitives """
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ELU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Dropout2d(dropout)
        )
        self.elu = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.elu(x + self.conv(x))


class CustomSpatialEncoder(nn.Module):
    """
    Downsamples the screen while embedding strong spatial features.
    Outputs a feature map scale of (Batch, 48, H/16, W/16).
    """
    def __init__(self, in_channels: int = 3, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=7, stride=2, padding=3, bias=False),  # /2
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),          # /4
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            ResBlock(32, dropout),
            
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1, bias=False),          # /8
            nn.BatchNorm2d(48),
            nn.ELU(inplace=True),
            ResBlock(48, dropout),
            
            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1, bias=False),          # /16
            nn.BatchNorm2d(48),
            nn.ELU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvGRUCell(nn.Module):
    """
    A GRU cell that replaces standard Linear matrix multiplications with Conv2d loops,
    preserving structural 2D coordinates across time.
    """
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
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
        return h_next


class DrivingVideoPolicy(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        
        # --- Center Car Mask Boundaries (Percentages) ---
        self.car_y_min_pct = 0.50
        self.car_y_max_pct = 0.80
        self.car_x_min_pct = 0.40
        self.car_x_max_pct = 0.60
        
        self.spatial_encoder = CustomSpatialEncoder(in_channels=3, dropout=cfg.dropout)
        
        # ConvGRU tracking state
        self.feat_channels = 48
        self.temporal_rnn = ConvGRUCell(input_dim=self.feat_channels, hidden_dim=self.feat_channels, kernel_size=3)
        
        # Retain your custom 6x3 pooling layout for driving horizon bias
        self.pool = nn.AdaptiveAvgPool2d(cfg.pooling)
        
        self.fc_features = nn.Sequential(
            nn.Linear(self.feat_channels * cfg.pooling[0] * cfg.pooling[1], self.cfg.d_model),
            nn.ELU(inplace=True),
            nn.Dropout(cfg.dropout)
        )
        
        self.button_head = nn.Linear(self.cfg.d_model, self.cfg.prediction_horizon * self.cfg.num_bin)
        nn.init.constant_(self.button_head.bias, -1.0)

    def _normalize_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0
        return frames.clamp(0.0, 1.0)

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

    def forward(self, frames: torch.Tensor, dt: torch.Tensor, state: Optional[TemporalState] = None, return_aux: bool = False):
        b, t, c, h, w = frames.shape
        
        frames = self._normalize_frames(frames)
        frames = self._apply_masks(frames)
        
        # 1. Spatial Processing
        x = frames.reshape(b * t, c, h, w)
        spatial_feats = self.spatial_encoder(x)
        _, cf, hf, wf = spatial_feats.shape
        spatial_feats = spatial_feats.reshape(b, t, cf, hf, wf)
        
        # 2. Temporal ConvGRU Rollout Loop
        if state is not None and state.hidden_state is not None:
            h_t = state.hidden_state
        else:
            h_t = torch.zeros(b, self.feat_channels, hf, wf, device=frames.device, dtype=spatial_feats.dtype)
            
        temporal_outputs = []
        for step in range(t):
            h_t = self.temporal_rnn(spatial_feats[:, step], h_t)
            temporal_outputs.append(h_t.unsqueeze(1))
            
        temporal_out = torch.cat(temporal_outputs, dim=1)  # [B, T, C_feat, H_feat, W_feat]
        
        # 3. Linear Downsampling for Classifier Heads
        temporal_flat = temporal_out.reshape(b * t, cf, hf, wf)
        pooled = self.pool(temporal_flat)
        pooled_flat = pooled.reshape(pooled.shape[0], -1)
        
        fc_out = self.fc_features(pooled_flat)
        fc_out = fc_out.reshape(b, t, self.cfg.d_model)
        
        # 4. Action Mapping Prediction
        button = self.button_head(fc_out).reshape(b, t, self.cfg.prediction_horizon, self.cfg.num_bin)
        
        step_button = button[:, :, 0]
        output = PolicyOutput(
            button_logits=step_button,
            horizon_button_logits=button,
            future_button_logits={idx + 1: button[:, :, idx] for idx in range(self.cfg.prediction_horizon)},
        )
        return output

    def forward_step(self, frame: torch.Tensor, dt: torch.Tensor, state: TemporalState, return_aux: bool = False):
            b = frame.shape[0]
            
            # 1. Normalize and mask out the car layout exactly once
            frame_norm = self._normalize_frames(frame)
            masked_frame = self._apply_masks(frame_norm)
            
            # 2. Extract spatial primitives
            spatial_feat = self.spatial_encoder(masked_frame)
            cf, hf, wf = spatial_feat.shape[1:]
            
            # 3. Evaluate a single temporal rollout transition step
            if state is not None and state.hidden_state is not None:
                h_t = state.hidden_state
            else:
                h_t = torch.zeros(b, self.feat_channels, hf, wf, device=frame.device, dtype=spatial_feat.dtype)
                
            new_hidden = self.temporal_rnn(spatial_feat, h_t)
            
            # 4. Map the new hidden states through your 5x5 pooling layout
            pooled = self.pool(new_hidden)
            pooled_flat = pooled.reshape(b, -1)
            
            fc_out = self.fc_features(pooled_flat)
            
            # 5. Project to action space values
            button = self.button_head(fc_out).reshape(b, self.cfg.prediction_horizon, self.cfg.num_bin)
            
            squeezed = PolicyOutput(
                button_logits=button[:, 0],
                horizon_button_logits=button,
                future_button_logits={idx + 1: button[:, idx] for idx in range(self.cfg.prediction_horizon)},
            )
            
            # 6. Detach hidden state to prevent backpropagation graph memory leaks
            new_state = TemporalState(
                hidden_state=new_hidden.detach(),
            )
            return squeezed, new_state