from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple

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


SPATIAL_FEATURE_CHANNELS = 128
TEMPORAL_HIDDEN_CHANNELS = 128
POLICY_INPUT_CHANNELS = 6
TEMPORAL_RNN_LAYERS = 1
KEYPOINT_HEATMAP_CHANNELS = 32
FPN_FUSION_CHANNELS = 128
GRU_MEMORY_MIN_FRAMES = 2.0
GRU_MEMORY_MAX_FRAMES = 40.0
LAST_ACTION_EMBEDDING_DROPOUT = 0.25
DEFAULT_PREDICTION_OFFSET = 1


def _require_int_at_least(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc
    try:
        if float(result) != float(value):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}.")
    return result


def _require_float_range(name: str, value: object, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result) or result < minimum or result > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {value!r}.")
    return result


def _single_prediction_offset(prediction_horizon: int, offsets: Optional[Sequence[int]]) -> Tuple[int, ...]:
    horizon = int(prediction_horizon)
    if offsets is None:
        if horizon != 1:
            raise ValueError("This policy is single-horizon only; prediction_horizon must be 1.")
        return (DEFAULT_PREDICTION_OFFSET,)

    parsed = tuple(
        _require_int_at_least(f"prediction_horizon_offsets[{idx}]", offset, 1)
        for idx, offset in enumerate(offsets)
    )
    if len(parsed) != 1:
        raise ValueError(f"This policy is single-horizon only; provide exactly one offset, got {parsed}.")
    return parsed


@dataclass
class ModelConfig:
    selected_game: str = ACTION_SELECTED_GAME
    data_root: Optional[str] = None
    video_ext: str = ".mp4"
    csv_ext: str = ".csv"

    model_size: int = 256
    seq_len: int = 80
    train_seq_stride: int = 20
    val_seq_stride: int = 80
    prediction_horizon: int = 1
    prediction_horizon_offsets: Optional[Sequence[int]] = None

    key_names: Optional[List[str]] = None
    mouse_button_names: Optional[List[str]] = None

    d_model: int = 128
    spatial_dropout: float = 0.10
    head_dropout: float = 0.20
    zoneout: float = 0.0
    # Feed the previous action into the heads. Default True so legacy
    # checkpoints (which have last_action_encoder weights) keep loading;
    # TrainConfig overrides this to False for new runs because the copy
    # shortcut it creates dominates the +1 head (causal confusion).
    last_action_conditioning: bool = True

    button_state_threshold: float = 0.5
    button_state_thresholds: Optional[Sequence[float]] = None
    num_bin: int = 0

    def __post_init__(self) -> None:
        self.selected_game = normalize_game_name(self.selected_game)
        if self.data_root is None:
            self.data_root = game_data_root(self.selected_game)

        self.model_size = _require_int_at_least("model_size", self.model_size, 32)
        self.seq_len = _require_int_at_least("seq_len", self.seq_len, 1)
        self.train_seq_stride = _require_int_at_least("train_seq_stride", self.train_seq_stride, 1)
        self.val_seq_stride = _require_int_at_least("val_seq_stride", self.val_seq_stride, 1)
        self.prediction_horizon = _require_int_at_least("prediction_horizon", self.prediction_horizon, 1)
        self.prediction_horizon_offsets = _single_prediction_offset(
            self.prediction_horizon,
            self.prediction_horizon_offsets,
        )
        self.prediction_horizon = 1

        self.d_model = _require_int_at_least("d_model", self.d_model, 64)
        if self.d_model % 4 != 0:
            raise ValueError(f"d_model must be divisible by 4, got {self.d_model}.")
        self.spatial_dropout = _require_float_range("spatial_dropout", self.spatial_dropout, 0.0, 0.9)
        self.head_dropout = _require_float_range("head_dropout", self.head_dropout, 0.0, 0.9)
        self.zoneout = _require_float_range("zoneout", self.zoneout, 0.0, 0.9)
        self.last_action_conditioning = bool(self.last_action_conditioning)

        if self.key_names is None:
            self.key_names = get_key_names(self.selected_game)
        else:
            self.key_names = list(self.key_names)

        if self.mouse_button_names is None:
            self.mouse_button_names = get_mouse_button_names(self.selected_game)
        else:
            self.mouse_button_names = list(self.mouse_button_names)

        self.num_bin = len(self.key_names) + len(self.mouse_button_names)
        if self.num_bin <= 0:
            raise ValueError("At least one action key/button is required.")

        self.button_state_threshold = _require_float_range(
            "button_state_threshold",
            self.button_state_threshold,
            0.0,
            1.0,
        )
        if self.button_state_thresholds is None:
            self.button_state_thresholds = tuple(float(self.button_state_threshold) for _ in range(self.num_bin))
        else:
            thresholds = tuple(float(x) for x in self.button_state_thresholds)
            if len(thresholds) != self.num_bin:
                raise ValueError(
                    f"button_state_thresholds must contain {self.num_bin} values, got {len(thresholds)}."
                )
            for idx, threshold in enumerate(thresholds):
                _require_float_range(f"button_state_thresholds[{idx}]", threshold, 0.0, 1.0)
            self.button_state_thresholds = thresholds


@dataclass
class PolicyOutput:
    button_logits: torch.Tensor


@dataclass
class TemporalState:
    hidden_state: Optional[torch.Tensor] = None
    prev_frame: Optional[torch.Tensor] = None


class ResidualBlock(nn.Module):
    """2D residual block used by the policy encoder."""

    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        self.conv = nn.Sequential(*layers)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv(x))


class CustomSpatialEncoder(nn.Module):
    """
    GroupNorm residual CNN encoder with a compact FPN-style fusion neck.
    It keeps H/4, H/8, and H/16 feature stages, then fuses them into an H/8
    grid so lane and curb geometry is not forced through a 16x16 bottleneck.
    GroupNorm keeps statistics batch-independent (training runs at batch_size=1).
    """
    def __init__(self, in_channels: int = 3, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),   # /2
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),            # /4
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            ResidualBlock(64, dropout),
            ResidualBlock(64, dropout),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),            # /8
            nn.GroupNorm(8, 96),
            nn.SiLU(inplace=True),
            ResidualBlock(96, dropout),
            ResidualBlock(96, dropout),
            ResidualBlock(96, dropout),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(96, SPATIAL_FEATURE_CHANNELS, kernel_size=3, stride=2, padding=1, bias=False),  # /16
            nn.GroupNorm(8, SPATIAL_FEATURE_CHANNELS),
            nn.SiLU(inplace=True),
            ResidualBlock(SPATIAL_FEATURE_CHANNELS, dropout),
            ResidualBlock(SPATIAL_FEATURE_CHANNELS, dropout),
        )
        self.fpn_64_to_32 = nn.Sequential(
            nn.Conv2d(64, FPN_FUSION_CHANNELS, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, FPN_FUSION_CHANNELS),
            nn.SiLU(inplace=True),
        )
        self.fpn_32 = nn.Sequential(
            nn.Conv2d(96, FPN_FUSION_CHANNELS, kernel_size=1, bias=False),
            nn.GroupNorm(8, FPN_FUSION_CHANNELS),
            nn.SiLU(inplace=True),
            ResidualBlock(FPN_FUSION_CHANNELS, dropout),
            ResidualBlock(FPN_FUSION_CHANNELS, dropout),
        )
        self.fpn_16 = nn.Sequential(
            nn.Conv2d(SPATIAL_FEATURE_CHANNELS, FPN_FUSION_CHANNELS, kernel_size=1, bias=False),
            nn.GroupNorm(8, FPN_FUSION_CHANNELS),
            nn.SiLU(inplace=True),
        )
        self.fpn_refine = nn.Sequential(
            nn.Conv2d(FPN_FUSION_CHANNELS * 3, FPN_FUSION_CHANNELS, kernel_size=1, bias=False),
            nn.GroupNorm(8, FPN_FUSION_CHANNELS),
            nn.SiLU(inplace=True),
            ResidualBlock(FPN_FUSION_CHANNELS, dropout),
            ResidualBlock(FPN_FUSION_CHANNELS, dropout),
        )

    def _fuse_stages(
        self,
        stage1: torch.Tensor,
        stage2: torch.Tensor,
        stage3: torch.Tensor,
    ) -> torch.Tensor:
        stage1_32 = self.fpn_64_to_32(stage1)
        stage2_32 = self.fpn_32(stage2)
        stage3_32 = F.interpolate(
            self.fpn_16(stage3),
            size=stage2_32.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.fpn_refine(torch.cat([stage1_32, stage2_32, stage3_32], dim=1))

    def feature_stages(self, x: torch.Tensor) -> List[torch.Tensor]:
        stem = self.stem(x)
        stage1 = self.stage1(stem)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        fused = self._fuse_stages(stage1, stage2, stage3)
        return [stage1, stage2, stage3, fused]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem = self.stem(x)
        stage1 = self.stage1(stem)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        return self._fuse_stages(stage1, stage2, stage3)


def apply_static_masks(frames: torch.Tensor) -> torch.Tensor:
    """Zero out the HUD, minimap, and Roblox UI regions (relative coords)."""
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


class ConvGRUCell(nn.Module):
    """
    A GRU cell that replaces standard Linear matrix multiplications with Conv2d,
    preserving structural 2D coordinates across time.
    """
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, zoneout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.zoneout = _require_float_range("zoneout", zoneout, 0.0, 0.9)
        padding = kernel_size // 2

        self.gates_conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=2 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        self.candidate_conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.gates_conv.bias is not None:
            nn.init.zeros_(self.gates_conv.bias)
            # Per-channel memory timescales log-spaced over 2-40 frames:
            # update-gate bias -log(T-1) gives z = sigmoid(b) = 1/T, i.e. a
            # ~T-frame memory. Log spacing keeps most channels fast (median
            # ~9 frames, healthy z(1-z) gate gradients) while the tail reaches
            # 40 frames; a uniform slow init (chrono-style) starved the cell
            # of input and froze the gates.
            with torch.no_grad():
                log_timescales = torch.linspace(
                    math.log(GRU_MEMORY_MIN_FRAMES), math.log(GRU_MEMORY_MAX_FRAMES), self.hidden_dim
                )
                self.gates_conv.bias[self.hidden_dim:] = -torch.log(torch.exp(log_timescales) - 1.0)
        if self.candidate_conv.bias is not None:
            nn.init.zeros_(self.candidate_conv.bias)

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


class DrivingVideoPolicy(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.spatial_encoder = CustomSpatialEncoder(in_channels=POLICY_INPUT_CHANNELS, dropout=cfg.spatial_dropout)

        # ConvGRU tracking state
        self.spatial_feat_channels = SPATIAL_FEATURE_CHANNELS
        self.feat_channels = TEMPORAL_HIDDEN_CHANNELS
        self.temporal_rnns = nn.ModuleList(
            [
                ConvGRUCell(
                    input_dim=self.spatial_feat_channels if layer_idx == 0 else self.feat_channels,
                    hidden_dim=self.feat_channels,
                    kernel_size=3,
                    zoneout=cfg.zoneout,
                )
                for layer_idx in range(TEMPORAL_RNN_LAYERS)
            ]
        )
        self.temporal_spatial_fusion = (
            nn.Identity()
            if self.spatial_feat_channels == self.feat_channels
            else nn.Conv2d(self.spatial_feat_channels, self.feat_channels, kernel_size=1, bias=False)
        )
        self.fused_norm = nn.GroupNorm(8, self.feat_channels)

        # Spatial-softmax keypoint pooling: each heatmap channel reduces to its
        # expected (x, y), preserving precise lane/object positions for steering.
        self.keypoint_heatmaps = nn.Conv2d(self.feat_channels, KEYPOINT_HEATMAP_CHANNELS, kernel_size=1)
        pooled_dim = self.feat_channels + 2 * KEYPOINT_HEATMAP_CHANNELS
        self.fc_features = nn.Sequential(
            nn.Linear(pooled_dim, self.cfg.d_model),
            nn.SiLU(inplace=True),
            nn.Dropout(cfg.head_dropout),
        )

        if cfg.last_action_conditioning:
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
        else:
            self.last_action_encoder = None

        fusion_in = self.cfg.d_model * 2 if cfg.last_action_conditioning else self.cfg.d_model
        self.head_fusion = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, self.cfg.d_model),
            nn.ELU(inplace=True),
        )

        head_hidden = max(128, self.cfg.d_model)
        self.button_head = nn.Sequential(
            nn.LayerNorm(self.cfg.d_model),
            nn.Linear(self.cfg.d_model, head_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(cfg.head_dropout * 0.5),
            nn.Linear(head_hidden, self.cfg.num_bin),
        )
        output_layer = self.button_head[-1]
        if isinstance(output_layer, nn.Linear):
            nn.init.constant_(output_layer.bias, -1.0)

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
        expected_single = (batch_size, self.feat_channels, height, width)
        if hidden.dim() == 4:
            if tuple(hidden.shape) != expected_single:
                raise ValueError(
                    f"Expected temporal hidden state {expected_single}, got {tuple(hidden.shape)}."
                )
            first = hidden
            remaining = self._initial_temporal_state(batch_size, height, width, device=device, dtype=dtype)[1:]
            return torch.cat([first.unsqueeze(0), remaining], dim=0)
        if hidden.dim() != 5:
            raise ValueError(f"Expected temporal hidden state [L,B,C,H,W], got {tuple(hidden.shape)}.")
        if hidden.size(0) != len(self.temporal_rnns):
            raise ValueError(f"Expected {len(self.temporal_rnns)} temporal layers, got {hidden.size(0)}.")
        expected_stacked = (len(self.temporal_rnns), *expected_single)
        if tuple(hidden.shape) != expected_stacked:
            raise ValueError(
                f"Expected temporal hidden state {expected_stacked}, got {tuple(hidden.shape)}."
            )
        return hidden

    def _temporal_step(self, x_t: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        next_states = []
        for layer_idx, rnn in enumerate(self.temporal_rnns):
            x_t = rnn(x_t, hidden_state[layer_idx])
            next_states.append(x_t)
        return x_t, torch.stack(next_states, dim=0)

    def _pool_features(self, fused: torch.Tensor) -> torch.Tensor:
        fused = self.fused_norm(fused)
        heatmaps = self.keypoint_heatmaps(fused)
        b, k, h, w = heatmaps.shape
        attention = torch.softmax(heatmaps.reshape(b, k, h * w).float(), dim=-1)
        ys = torch.linspace(-1.0, 1.0, h, device=fused.device, dtype=attention.dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=fused.device, dtype=attention.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        expected_x = (attention * grid_x.reshape(1, 1, h * w)).sum(dim=-1)
        expected_y = (attention * grid_y.reshape(1, 1, h * w)).sum(dim=-1)
        keypoints = torch.cat([expected_x, expected_y], dim=1).to(dtype=fused.dtype)
        context = fused.mean(dim=(-2, -1))
        return self.fc_features(torch.cat([keypoints, context], dim=1))

    def _button_logits(self, features: torch.Tensor) -> torch.Tensor:
        if features.size(-1) != self.cfg.d_model:
            raise ValueError(f"Expected final feature dim {self.cfg.d_model}, got {features.size(-1)}.")
        return self.button_head(features)

    def _normalize_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0
        elif not torch.is_floating_point(frames):
            raise TypeError(f"frames must be a floating point or uint8 tensor, got {frames.dtype}.")
        return frames

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

        action_values = action_values.reshape(batch_size * time_steps, self.cfg.num_bin)
        features = self.last_action_encoder(action_values).reshape(batch_size, time_steps, self.cfg.d_model)
        return features.to(dtype=dtype)

    def _apply_masks(self, frames: torch.Tensor) -> torch.Tensor:
        return apply_static_masks(frames)

    def _previous_frame_from_state(
        self,
        state: Optional[TemporalState],
        expected_shape: Tuple[int, int, int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if state is None or state.prev_frame is None:
            return None
        previous = state.prev_frame.to(device=device, dtype=dtype)
        if tuple(previous.shape) != expected_shape:
            raise ValueError(f"Expected previous frame shape {expected_shape}, got {tuple(previous.shape)}.")
        return previous

    def _frames_with_motion(
        self,
        frames: torch.Tensor,
        state: Optional[TemporalState],
    ) -> torch.Tensor:
        b, t, c, h, w = frames.shape
        previous_first = self._previous_frame_from_state(
            state,
            (b, c, h, w),
            device=frames.device,
            dtype=frames.dtype,
        )
        if previous_first is None:
            previous_first = frames[:, 0]
        previous_first = previous_first.unsqueeze(1)
        previous = previous_first if t == 1 else torch.cat([previous_first, frames[:, :-1]], dim=1)
        motion = frames - previous
        return torch.cat([frames, motion], dim=2)

    def _frame_with_motion(
        self,
        frame: torch.Tensor,
        state: Optional[TemporalState],
    ) -> torch.Tensor:
        previous = self._previous_frame_from_state(
            state,
            tuple(frame.shape),
            device=frame.device,
            dtype=frame.dtype,
        )
        if previous is None:
            previous = frame
        motion = frame - previous
        return torch.cat([frame, motion], dim=1)

    def forward(
        self,
        frames: torch.Tensor,
        state: Optional[TemporalState] = None,
        return_aux: bool = False,
        prev_action: Optional[torch.Tensor] = None,
    ):
        if frames.dim() != 5:
            raise ValueError(f"Expected RGB frames with shape [B,T,3,H,W], got {tuple(frames.shape)}.")
        b, t, c, h, w = frames.shape
        if c != 3:
            raise ValueError(f"Expected RGB frames with shape [B,T,3,H,W], got {tuple(frames.shape)}.")

        frames = self._normalize_frames(frames)
        frames = self._apply_masks(frames)
        model_input = self._frames_with_motion(frames, state)

        # 1. Spatial Processing
        x = model_input.reshape(b * t, POLICY_INPUT_CHANNELS, h, w)
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        spatial_feats = self.spatial_encoder(x)
        _, spatial_channels, hf, wf = spatial_feats.shape
        spatial_feats = spatial_feats.reshape(b, t, spatial_channels, hf, wf)

        # 2. Temporal ConvGRU Rollout Loop
        h_t = self._prepare_temporal_state(
            state,
            b,
            hf,
            wf,
            device=frames.device,
            dtype=spatial_feats.dtype,
        )

        visual_steps = []
        for step in range(t):
            spatial_step = spatial_feats[:, step]
            temporal_feat, h_t = self._temporal_step(spatial_step, h_t)
            fused = temporal_feat + self.temporal_spatial_fusion(spatial_step)
            visual_steps.append(self._pool_features(fused))

        # 3. Fuse pooled visual features with last-action context (when enabled)
        visual_feat = torch.stack(visual_steps, dim=1)
        if self.last_action_encoder is not None:
            action_feat = self._last_action_features(prev_action, b, t, device=frames.device, dtype=visual_feat.dtype)
            fc_out = self.head_fusion(torch.cat([visual_feat, action_feat], dim=-1))
        else:
            fc_out = self.head_fusion(visual_feat)

        # 4. Action Mapping Prediction
        output = PolicyOutput(
            button_logits=self._button_logits(fc_out),
        )
        if return_aux:
            return output, TemporalState(hidden_state=h_t.detach(), prev_frame=frames[:, -1].detach())
        return output

    def forward_step(
        self,
        frame: torch.Tensor,
        state: TemporalState,
        prev_action: Optional[torch.Tensor] = None,
    ):
        if frame.dim() != 4 or frame.size(1) != 3:
            raise ValueError(f"Expected RGB frame with shape [B,3,H,W], got {tuple(frame.shape)}.")
        b = frame.shape[0]

        # 1. Normalize and mask out the car layout exactly once
        frame_norm = self._normalize_frames(frame)
        masked_frame = self._apply_masks(frame_norm)
        model_input = self._frame_with_motion(masked_frame, state)

        # 2. Extract spatial primitives
        x = model_input
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        spatial_feat = self.spatial_encoder(x)
        _, hf, wf = spatial_feat.shape[1:]

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
        fused = temporal_feat + self.temporal_spatial_fusion(spatial_feat)

        # 4. Pool the fused features and add last-action context (when enabled)
        visual_feat = self._pool_features(fused)
        if self.last_action_encoder is not None:
            action_feat = self._last_action_features(prev_action, b, 1, device=frame.device, dtype=visual_feat.dtype).reshape(
                b,
                self.cfg.d_model,
            )
            fc_out = self.head_fusion(torch.cat([visual_feat, action_feat], dim=-1))
        else:
            fc_out = self.head_fusion(visual_feat)

        # 5. Project to action space values
        squeezed = PolicyOutput(
            button_logits=self._button_logits(fc_out),
        )

        # 6. Detach hidden state to prevent backpropagation graph memory leaks
        new_state = TemporalState(hidden_state=new_hidden.detach(), prev_frame=masked_frame.detach())
        return squeezed, new_state
