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
    frame_spatial_pool: int = 8
    frame_spatial_channels: int = 32
    spatial_attention_tokens: int = 8
    spatial_attention_heads: int = 4
    spatial_temporal_grid: int = 2
    temporal_layers: int = 2
    temporal_heads: int = 4
    temporal_mlp_ratio: float = 2.0
    dropout: float = 0.10
    coord_scale: float = 1.0
    coord_dropout: float = 0.1
    encode_chunk_size: int = 16
    max_context: int = 100

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
        self.frame_spatial_pool = max(1, int(self.frame_spatial_pool))
        self.frame_spatial_channels = int(self.frame_spatial_channels)
        if self.frame_spatial_channels <= 0:
            self.frame_spatial_channels = max(16, self.d_model // 4)
        self.frame_spatial_channels = max(8, min(int(self.frame_spatial_channels), int(self.d_model)))
        self.spatial_attention_tokens = max(1, int(self.spatial_attention_tokens))
        self.spatial_attention_heads = max(1, int(self.spatial_attention_heads))
        while self.d_model % self.spatial_attention_heads != 0 and self.spatial_attention_heads > 1:
            self.spatial_attention_heads -= 1
        self.spatial_temporal_grid = max(1, int(self.spatial_temporal_grid))
        self.temporal_layers = max(1, int(self.temporal_layers))
        self.temporal_heads = max(1, int(self.temporal_heads))
        while self.d_model % self.temporal_heads != 0 and self.temporal_heads > 1:
            self.temporal_heads -= 1
        self.temporal_mlp_ratio = float(min(max(self.temporal_mlp_ratio, 1.0), 8.0))
        self.dropout = float(min(max(self.dropout, 0.0), 0.9))
        self.coord_scale = float(min(max(self.coord_scale, 0.0), 2.0))
        self.coord_dropout = float(min(max(self.coord_dropout, 0.0), 1.0))
        self.encode_chunk_size = max(1, int(self.encode_chunk_size))
        self.max_context = max(self.seq_len, int(self.max_context))

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
    prev_frame: Optional[torch.Tensor] = None
    hidden_state: Optional[torch.Tensor] = None
    steps: int = 0


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1

class PilotNetBackbone(nn.Module):
    """
    Based on NVIDIA's PilotNet architecture, heavily optimized for detecting 
    lane lines and road boundaries without washing out spatial features.
    """
    def __init__(self, in_channels: int = 3, dropout: float = 0.2):
        super().__init__()
        # Standard PilotNet Conv stack
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm2d(24),
            nn.ELU(inplace=True),
            
            nn.Conv2d(24, 36, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm2d(36),
            nn.ELU(inplace=True),
            
            nn.Conv2d(36, 48, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm2d(48),
            nn.ELU(inplace=True),
            
            nn.Conv2d(48, 64, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            
            nn.Dropout2d(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_layers(x)

class DrivingVideoPolicy(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        
        # Crop parameters: e.g., if model_size is 256, top_crop=100 removes the sky
        self.top_crop = int(self.cfg.model_size * 0.40) 
        
        # We only pass the current RGB frame to prevent motion/coord artifacts from distracting
        self.cnn = PilotNetBackbone(in_channels=3, dropout=cfg.dropout)
        
        # Calculate flattened size dynamically based on crop and model size
        dummy_h = self.cfg.model_size - self.top_crop
        dummy_w = self.cfg.model_size
        dummy_input = torch.zeros(1, 3, dummy_h, dummy_w)
        with torch.no_grad():
            flattened_size = self.cnn(dummy_input).flatten(1).size(1)
            
        self.fc_features = nn.Sequential(
            nn.Linear(flattened_size, self.cfg.d_model),
            nn.ELU(inplace=True),
            nn.Dropout(cfg.dropout)
        )
        
        # RNN for temporal consistency (smoother steering than raw transformers)
        self.temporal_rnn = nn.GRU(
            input_size=self.cfg.d_model, 
            hidden_size=self.cfg.d_model, 
            num_layers=1, 
            batch_first=True
        )
        
        # Final projection to button/steering logits
        self.button_head = nn.Linear(self.cfg.d_model, self.cfg.prediction_horizon * self.cfg.num_bin)
        nn.init.constant_(self.button_head.bias, -1.0)

    def _normalize_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0
        return frames.clamp(0.0, 1.0)

    def forward(self, frames: torch.Tensor, dt: torch.Tensor, state: Optional[TemporalState] = None, return_aux: bool = False):
        # frames shape: [B, T, C, H, W]
        b, t, c, h, w = frames.shape
        
        # 1. Normalize and Crop the sky
        frames = self._normalize_frames(frames)
        frames = frames[:, :, :, self.top_crop:, :] # Removes top portion of the image
        
        # 2. Extract spatial features
        x = frames.reshape(b * t, c, h - self.top_crop, w)
        x = self.cnn(x)
        x = x.flatten(1) # Flatten spatial dimensions entirely
        x = self.fc_features(x)
        x = x.view(b, t, self.cfg.d_model)
        
        # 3. Temporal aggregation
        # We ignore dt here as standard driving models respond to visual layout, 
        # but you can concatenate dt_features to 'x' before the RNN if framerate varies heavily.
        if state is not None and state.hidden_state is not None:
            temporal, hidden = self.temporal_rnn(x, state.hidden_state)
        else:
            temporal, hidden = self.temporal_rnn(x)
            
        # 4. Predict
        button = self.button_head(temporal).view(b, t, self.cfg.prediction_horizon, self.cfg.num_bin)
        
        step_button = button[:, :, 0]
        output = PolicyOutput(
            button_logits=step_button,
            horizon_button_logits=button,
            future_button_logits={idx + 1: button[:, :, idx] for idx in range(self.cfg.prediction_horizon)},
        )
        return output

    def forward_step(self, frame: torch.Tensor, dt: torch.Tensor, state: TemporalState, return_aux: bool = False):
        # frame shape: [B, C, H, W] -> make it [B, 1, C, H, W]
        output = self.forward(frame.unsqueeze(1), dt.unsqueeze(1), state)
        
        # Update hidden state for inference
        step_features = self.cnn(self._normalize_frames(frame)[:, :, self.top_crop:, :]).flatten(1)
        _, hidden = self.temporal_rnn(self.fc_features(step_features).unsqueeze(1), state.hidden_state)
        
        new_state = TemporalState(
            prev_frame=frame.detach(),
            hidden_state=hidden.detach(),
            steps=state.steps + 1,
        )
        
        squeezed = PolicyOutput(
            button_logits=output.button_logits[:, 0],
            horizon_button_logits=output.horizon_button_logits[:, 0],
            future_button_logits={k: v[:, 0] for k, v in output.future_button_logits.items()},
        )
        return squeezed, new_state


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = ModelConfig(model_size=256, seq_len=100, prediction_horizon=1, max_context=100)
    model = PilotNetBackbone(cfg)
    x = torch.rand(1, cfg.seq_len, 3, cfg.model_size, cfg.model_size)
    dt = torch.full((1, cfg.seq_len), cfg.prediction_dt)
    out = model(x, dt=dt)
    print(tuple(out.horizon_button_logits.shape))
