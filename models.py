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


class PilotNetBackbone(nn.Module):
    def __init__(self, in_channels: int = 3, dropout: float = 0.2):
        super().__init__()
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
        
        # --- Center Car Mask Boundaries (Percentages) ---
        # Adjust these percentages to draw a tight box around your car in Greenville.
        # 0.0 is top/left, 1.0 is bottom/right.
        self.car_y_min_pct = 0.50  # Top of the car
        self.car_y_max_pct = 0.80  # Bottom of the car (leaves the bottom 20% for UI)
        self.car_x_min_pct = 0.40  # Left side of the car
        self.car_x_max_pct = 0.60  # Right side of the car
        
        self.cnn = PilotNetBackbone(in_channels=3, dropout=cfg.dropout)
        
        # We pass the full model size now because we aren't changing the tensor shape
        dummy_input = torch.zeros(1, 3, self.cfg.model_size, self.cfg.model_size)
        with torch.no_grad():
            flattened_size = self.cnn(dummy_input).reshape(1, -1).size(1)
            
        self.fc_features = nn.Sequential(
            nn.Linear(flattened_size, self.cfg.d_model),
            nn.ELU(inplace=True),
            nn.Dropout(cfg.dropout)
        )
        
        self.temporal_rnn = nn.GRU(
            input_size=self.cfg.d_model, 
            hidden_size=self.cfg.d_model, 
            num_layers=1, 
            batch_first=True
        )
        
        self.button_head = nn.Linear(self.cfg.d_model, self.cfg.prediction_horizon * self.cfg.num_bin)
        nn.init.constant_(self.button_head.bias, -1.0)

    def _normalize_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0
        return frames.clamp(0.0, 1.0)

    def _apply_masks(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Blacks out the car, the bottom HUD, and the minimap to force the CNN
        to look at the road and lane lines.
        """
        h, w = frames.shape[-2:]
        masked_frames = frames.clone()

        # 1. Mask the Car (Center)
        # Blocks out the 3rd-person car model so the AI stops copying its rotation.
        car_y1, car_y2 = int(h * self.car_y_min_pct), int(h * self.car_y_max_pct)
        car_x1, car_x2 = int(w * self.car_x_min_pct), int(w * self.car_x_max_pct)
        masked_frames[..., car_y1:car_y2, car_x1:car_x2] = 0.0

        # 2. Mask the Bottom HUD (Speedometer, Gear, etc.)
        # Blocks the entire bottom 20% so it cannot cheat by reading speed.
        hud_y1 = int(h * 0.80)
        masked_frames[..., hud_y1:, :] = 0.0

        # 3. Mask the Minimap (Mid-Right)
        # Blocks the right side where the map and money UI appear.
        map_y1, map_y2 = int(h * 0.05), int(h * 0.2)
        map_x1 = int(w * 0.75)
        masked_frames[..., map_y1:map_y2, map_x1:] = 0.0

        # 4. Top Left Roblox UI (Optional but recommended)
        roblox_ui_y2 = int(h * 0.1)
        roblox_ui_x2 = int(w * 0.1)
        masked_frames[..., :roblox_ui_y2, :roblox_ui_x2] = 0.0

        return masked_frames

    def forward(self, frames: torch.Tensor, dt: torch.Tensor, state: Optional[TemporalState] = None, return_aux: bool = False):
        b, t, c, h, w = frames.shape
        
        # 1. Normalize and mask out the car
        frames = self._normalize_frames(frames)
        frames = self._apply_masks(frames)
        
        # 2. Extract spatial features
        x = frames.reshape(b * t, c, h, w)
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_features(x)
        x = x.reshape(b, t, self.cfg.d_model)
        
        # 3. Temporal aggregation
        if state is not None and state.hidden_state is not None:
            temporal, _ = self.temporal_rnn(x, state.hidden_state)
        else:
            temporal, _ = self.temporal_rnn(x)
            
        # 4. Predict
        button = self.button_head(temporal).reshape(b, t, self.cfg.prediction_horizon, self.cfg.num_bin)
        
        step_button = button[:, :, 0]
        output = PolicyOutput(
            button_logits=step_button,
            horizon_button_logits=button,
            future_button_logits={idx + 1: button[:, :, idx] for idx in range(self.cfg.prediction_horizon)},
        )
        return output

    def forward_step(self, frame: torch.Tensor, dt: torch.Tensor, state: TemporalState, return_aux: bool = False):
        output = self.forward(frame.unsqueeze(1), dt.unsqueeze(1), state)
        
        # Process frame for hidden state update
        frame_norm = self._normalize_frames(frame)
        masked_frame = self._apply_masks(frame_norm)

        step_features = self.cnn(masked_frame)
        x = self.fc_features(step_features.reshape(step_features.shape[0], -1)).unsqueeze(1)
        _, hidden = self.temporal_rnn(x, state.hidden_state)
        
        new_state = TemporalState(
            hidden_state=hidden.detach(),
        )
        
        squeezed = PolicyOutput(
            button_logits=output.button_logits[:, 0],
            horizon_button_logits=output.horizon_button_logits[:, 0],
            future_button_logits={k: v[:, 0] for k, v in output.future_button_logits.items()},
        )
        return squeezed, new_state
