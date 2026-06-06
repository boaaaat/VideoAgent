from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from action_space import (
    game_data_root,
    get_key_names,
    get_mouse_button_names,
    normalize_game_name,
    selected_game as ACTION_SELECTED_GAME,
)


MODEL_FAMILY = "causal_multihorizon_video_policy"
ARCHITECTURE_VERSION = 4
DEFAULT_HORIZON_OFFSETS = (1, 2, 3, 5, 7, 10)
BURN_IN_FRAMES = 20
POLICY_INPUT_CHANNELS = 6
GROUP_NORM_GROUPS = 8


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(int(GROUP_NORM_GROUPS), int(channels))
    while groups > 1 and int(channels) % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, int(channels))


def _coerce_horizon_offsets(offsets: Optional[Sequence[int]]) -> Tuple[int, ...]:
    values = DEFAULT_HORIZON_OFFSETS if offsets is None else tuple(int(offset) for offset in offsets)
    if not values:
        raise ValueError("prediction_horizon_offsets must contain at least one future frame offset.")
    if any(offset <= 0 for offset in values):
        raise ValueError(f"prediction_horizon_offsets must be positive, got {values}.")
    if any(current <= previous for previous, current in zip(values, values[1:])):
        raise ValueError(f"prediction_horizon_offsets must be strictly increasing, got {values}.")
    if values != DEFAULT_HORIZON_OFFSETS:
        raise ValueError(
            f"This policy family requires the six strict horizon offsets {DEFAULT_HORIZON_OFFSETS}, got {values}."
        )
    return tuple(values)


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
    prediction_horizon: int = len(DEFAULT_HORIZON_OFFSETS)
    prediction_horizon_offsets: Optional[Sequence[int]] = None
    action_label_offset: int = 0

    key_names: Optional[List[str]] = None
    mouse_button_names: Optional[List[str]] = None
    num_bin: int = 0

    cnn_channels: Tuple[int, int, int] = (32, 64, 128)
    spatial_channels: int = 128
    compressor_channels: int = 8
    spatial_pool_size: int = 8
    d_model: int = 512

    spatial_dropout: float = 0.05
    temporal_dropout: float = 0.10
    head_dropout: float = 0.15
    action_sequence_dropout: float = 0.25
    action_key_dropout: float = 0.10

    button_state_threshold: float = 0.5
    button_state_thresholds: Optional[Sequence[float]] = None
    horizon_button_thresholds: Optional[Sequence[Sequence[float]]] = None

    def __post_init__(self) -> None:
        self.selected_game = normalize_game_name(self.selected_game)
        if self.data_root is None:
            self.data_root = game_data_root(self.selected_game)

        self.model_size = int(self.model_size)
        if self.model_size != 256:
            raise ValueError(f"This policy family requires model_size=256 for a 64x64 spatial map, got {self.model_size}.")
        self.seq_len = max(1, int(self.seq_len))
        self.train_seq_stride = max(1, int(self.train_seq_stride))
        self.val_seq_stride = max(1, int(self.val_seq_stride))
        self.prediction_dt = max(1.0 / 240.0, float(self.prediction_dt))
        self.prediction_horizon_offsets = _coerce_horizon_offsets(self.prediction_horizon_offsets)
        self.prediction_horizon = len(self.prediction_horizon_offsets)
        self.action_label_offset = int(self.action_label_offset)
        effective_offsets = tuple(
            int(offset) + self.action_label_offset
            for offset in self.prediction_horizon_offsets
        )
        if any(offset <= 0 for offset in effective_offsets):
            raise ValueError(
                "Every horizon plus action_label_offset must remain strictly future-facing, "
                f"got effective offsets {effective_offsets}."
            )

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
            raise ValueError("The policy requires at least one action key or mouse button.")

        channels = tuple(max(8, int(value)) for value in self.cnn_channels)
        if len(channels) != 3:
            raise ValueError(f"cnn_channels must contain three stages, got {channels}.")
        self.cnn_channels = channels
        self.spatial_channels = max(32, int(self.spatial_channels))
        if self.spatial_channels != self.cnn_channels[-1]:
            raise ValueError(
                f"spatial_channels must match the final CNN channel count, got "
                f"{self.spatial_channels} and {self.cnn_channels[-1]}."
            )
        self.compressor_channels = max(1, int(self.compressor_channels))
        self.spatial_pool_size = max(1, int(self.spatial_pool_size))
        self.d_model = max(64, int(self.d_model))

        self.spatial_dropout = float(min(max(self.spatial_dropout, 0.0), 0.9))
        self.temporal_dropout = float(min(max(self.temporal_dropout, 0.0), 0.9))
        self.head_dropout = float(min(max(self.head_dropout, 0.0), 0.9))
        self.action_sequence_dropout = float(min(max(self.action_sequence_dropout, 0.0), 1.0))
        self.action_key_dropout = float(min(max(self.action_key_dropout, 0.0), 1.0))

        self.button_state_threshold = float(min(max(self.button_state_threshold, 0.0), 1.0))
        if self.button_state_thresholds is None:
            self.button_state_thresholds = tuple(self.button_state_threshold for _ in range(self.num_bin))
        else:
            thresholds = tuple(float(value) for value in self.button_state_thresholds)
            if len(thresholds) != self.num_bin:
                raise ValueError(f"button_state_thresholds must contain {self.num_bin} values, got {len(thresholds)}.")
            self.button_state_thresholds = tuple(min(max(value, 0.0), 1.0) for value in thresholds)

        if self.horizon_button_thresholds is None:
            self.horizon_button_thresholds = tuple(
                tuple(float(value) for value in self.button_state_thresholds)
                for _ in self.prediction_horizon_offsets
            )
        else:
            rows = tuple(tuple(float(value) for value in row) for row in self.horizon_button_thresholds)
            expected = (self.prediction_horizon, self.num_bin)
            if len(rows) != expected[0] or any(len(row) != expected[1] for row in rows):
                raise ValueError(f"horizon_button_thresholds must have shape {expected}.")
            self.horizon_button_thresholds = tuple(
                tuple(min(max(value, 0.0), 1.0) for value in row)
                for row in rows
            )


def _freeze_checkpoint_value(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_checkpoint_value(item) for item in value)
    return value


def validate_policy_checkpoint(
    state: object,
    cfg: ModelConfig,
    *,
    require_optimizer_state: bool = False,
) -> None:
    if not isinstance(state, dict):
        raise RuntimeError("Policy checkpoint must be a dictionary.")
    if state.get("model_family") != MODEL_FAMILY or int(state.get("architecture_version", -1)) != ARCHITECTURE_VERSION:
        raise RuntimeError(f"Checkpoint must be {MODEL_FAMILY} v{ARCHITECTURE_VERSION}; legacy checkpoints are unsupported.")

    required_dicts = ["config", "model_state", "ema_model_state"]
    if require_optimizer_state:
        required_dicts.append("optimizer_state")
    missing = [name for name in required_dicts if not isinstance(state.get(name), dict)]
    if missing:
        raise RuntimeError(f"Policy checkpoint is missing required dictionary fields: {', '.join(missing)}.")

    checkpoint_config = state["config"]
    architecture_fields = (
        "selected_game",
        "model_size",
        "seq_len",
        "prediction_dt",
        "action_label_offset",
        "cnn_channels",
        "spatial_channels",
        "compressor_channels",
        "spatial_pool_size",
        "d_model",
        "spatial_dropout",
        "temporal_dropout",
        "head_dropout",
        "action_sequence_dropout",
        "action_key_dropout",
    )
    for name in architecture_fields:
        if name not in checkpoint_config:
            raise RuntimeError(f"Policy checkpoint config is missing architecture field {name!r}.")
        stored_value = _freeze_checkpoint_value(checkpoint_config[name])
        expected_value = _freeze_checkpoint_value(getattr(cfg, name))
        if stored_value != expected_value:
            raise RuntimeError(
                f"Checkpoint architecture field {name!r} does not match: "
                f"checkpoint={stored_value!r}, expected={expected_value!r}."
            )
    expected_actions = tuple(list(cfg.key_names) + list(cfg.mouse_button_names))
    expected_offsets = tuple(int(offset) for offset in cfg.prediction_horizon_offsets)
    expected_thresholds = tuple(
        tuple(float(value) for value in row)
        for row in cfg.horizon_button_thresholds
    )
    try:
        stored_thresholds = tuple(
            tuple(float(value) for value in row)
            for row in state.get("horizon_button_thresholds", ())
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Policy checkpoint has invalid horizon_button_thresholds metadata.") from exc
    try:
        config_thresholds = tuple(
            tuple(float(value) for value in row)
            for row in checkpoint_config.get("horizon_button_thresholds", ())
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Policy checkpoint config has invalid horizon_button_thresholds.") from exc

    if tuple(state.get("action_names", ())) != expected_actions:
        raise RuntimeError(f"Checkpoint action names do not match config: expected {expected_actions}.")
    if tuple(state.get("horizon_offsets", ())) != expected_offsets:
        raise RuntimeError(f"Checkpoint horizons do not match config: expected {expected_offsets}.")
    if len(stored_thresholds) != cfg.prediction_horizon or any(
        len(row) != cfg.num_bin for row in stored_thresholds
    ):
        raise RuntimeError(
            "Checkpoint horizon thresholds have the wrong shape: "
            f"expected {(cfg.prediction_horizon, cfg.num_bin)}."
        )
    if stored_thresholds != expected_thresholds:
        raise RuntimeError("Checkpoint horizon thresholds do not match its config.")
    if config_thresholds != stored_thresholds:
        raise RuntimeError("Checkpoint top-level horizon thresholds do not match its saved config.")


@dataclass
class PolicyOutput:
    horizon_button_logits: torch.Tensor
    horizon_change_logits: torch.Tensor


@dataclass
class TemporalState:
    previous_frame: Optional[torch.Tensor] = None
    spatial_hidden: Optional[torch.Tensor] = None
    temporal_tokens: Optional[torch.Tensor] = None
    temporal_lengths: Optional[torch.Tensor] = None


class TokenGroupNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.norm = _group_norm(self.channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if int(x.size(-1)) != self.channels:
            raise ValueError(f"Expected final token dimension {self.channels}, got {tuple(x.shape)}.")
        original_shape = x.shape
        flat = x.reshape(-1, self.channels, 1)
        return self.norm(flat).reshape(original_shape)


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = int(dilation) * (kernel_size // 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=int(dilation),
            bias=False,
        )
        self.norm = _group_norm(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualSpatialBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels)
        self.dropout = nn.Dropout2d(float(dropout))
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _group_norm(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.norm2(self.conv2(self.dropout(self.conv1(x))))
        return self.act(x + residual)


class SpatialEncoder(nn.Module):
    def __init__(self, channels: Sequence[int], dropout: float) -> None:
        super().__init__()
        c1, c2, c3 = (int(value) for value in channels)
        self.stage1 = nn.Sequential(
            ConvNormAct(POLICY_INPUT_CHANNELS, c1, stride=2),
            ResidualSpatialBlock(c1, dropout),
        )
        self.stage2 = nn.Sequential(
            ConvNormAct(c1, c2, stride=2),
            ResidualSpatialBlock(c2, dropout),
        )
        self.stage3 = nn.Sequential(
            ConvNormAct(c2, c3, stride=1, dilation=2),
            ResidualSpatialBlock(c3, dropout),
            ConvNormAct(c3, c3, stride=1, dilation=4),
            ResidualSpatialBlock(c3, dropout),
        )

    def feature_stages(self, x: torch.Tensor) -> List[torch.Tensor]:
        stage1 = self.stage1(x)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        return [stage1, stage2, stage3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_stages(x)[-1]


class SpatialConvGRUCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        padding = int(kernel_size) // 2
        combined_channels = self.in_channels + self.hidden_channels
        self.gates = nn.Conv2d(
            combined_channels,
            self.hidden_channels * 2,
            kernel_size=int(kernel_size),
            padding=padding,
            bias=False,
        )
        self.gates_norm = _group_norm(self.hidden_channels * 2)
        self.candidate = nn.Conv2d(
            combined_channels,
            self.hidden_channels,
            kernel_size=int(kernel_size),
            padding=padding,
            bias=False,
        )
        self.candidate_norm = _group_norm(self.hidden_channels)

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor]) -> torch.Tensor:
        if hidden is None:
            hidden = x.new_zeros((int(x.size(0)), self.hidden_channels, int(x.size(2)), int(x.size(3))))
        if int(x.size(1)) != self.in_channels:
            raise ValueError(f"Expected ConvGRU input channels={self.in_channels}, got {tuple(x.shape)}.")
        if int(hidden.size(1)) != self.hidden_channels:
            raise ValueError(f"Expected ConvGRU hidden channels={self.hidden_channels}, got {tuple(hidden.shape)}.")
        combined = torch.cat((x, hidden), dim=1)
        update, reset = self.gates_norm(self.gates(combined)).chunk(2, dim=1)
        update = torch.sigmoid(update)
        reset = torch.sigmoid(reset)
        candidate_input = torch.cat((x, reset * hidden), dim=1)
        candidate = torch.tanh(self.candidate_norm(self.candidate(candidate_input)))
        return (1.0 - update) * hidden + update * candidate


class DrivingVideoPolicy(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_family = MODEL_FAMILY
        self.architecture_version = ARCHITECTURE_VERSION
        self.feature_size = int(cfg.model_size) // 4
        self.compressed_feature_dim = int(cfg.compressor_channels) * int(cfg.spatial_pool_size) * int(cfg.spatial_pool_size)

        self.spatial_encoder = SpatialEncoder(cfg.cnn_channels, cfg.spatial_dropout)
        self.spatial_gru = SpatialConvGRUCell(cfg.spatial_channels, cfg.spatial_channels)
        self.channel_compressor = nn.Sequential(
            nn.Conv2d(cfg.spatial_channels, cfg.compressor_channels, kernel_size=1, bias=False),
            _group_norm(cfg.compressor_channels),
            nn.SiLU(inplace=True),
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((cfg.spatial_pool_size, cfg.spatial_pool_size))
        self.frame_projector = nn.Sequential(
            nn.LayerNorm(self.compressed_feature_dim),
            nn.Linear(self.compressed_feature_dim, cfg.d_model),
            nn.SiLU(inplace=True),
            nn.Dropout(cfg.temporal_dropout),
        )

        context_hidden = max(64, cfg.d_model // 2)
        self.dt_encoder = nn.Sequential(
            nn.Linear(2, context_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(context_hidden, cfg.d_model),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(cfg.num_bin, context_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(context_hidden, cfg.d_model),
        )
        self.context_delta = nn.Sequential(
            TokenGroupNorm(cfg.d_model * 2),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.SiLU(inplace=True),
            nn.Dropout(cfg.temporal_dropout),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.context_gate = nn.Linear(cfg.d_model * 3, cfg.d_model)
        nn.init.zeros_(self.context_gate.weight)
        nn.init.constant_(self.context_gate.bias, -2.0)
        self.fused_norm = TokenGroupNorm(cfg.d_model)

        self.horizon_queries = nn.Parameter(torch.empty(cfg.prediction_horizon, cfg.d_model))
        nn.init.normal_(self.horizon_queries, mean=0.0, std=0.02)
        self.horizon_time_encoder = nn.Sequential(
            nn.Linear(2, context_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(context_hidden, cfg.d_model),
        )
        self.horizon_norm = TokenGroupNorm(cfg.d_model)
        head_hidden = max(128, cfg.d_model // 2)
        self.button_head = nn.Sequential(
            TokenGroupNorm(cfg.d_model),
            nn.Linear(cfg.d_model, head_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(head_hidden, cfg.num_bin),
        )
        self.change_head = nn.Sequential(
            TokenGroupNorm(cfg.d_model),
            nn.Linear(cfg.d_model, head_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(head_hidden, cfg.num_bin),
        )

    def _normalize_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dtype == torch.uint8:
            frames = frames.float().div(255.0)
        return frames.clamp(0.0, 1.0)

    def _apply_masks(self, frames: torch.Tensor) -> torch.Tensor:
        height, width = frames.shape[-2:]
        masked = frames.clone()
        masked[..., int(height * 0.96):, :] = 0.0
        masked[..., int(height * 0.05):int(height * 0.20), int(width * 0.75):] = 0.0
        masked[..., :int(height * 0.10), :int(width * 0.10)] = 0.0
        return masked

    def prepare_visual_input(
        self,
        frame: torch.Tensor,
        previous_frame: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if frame.dim() != 4 or int(frame.size(1)) != 3:
            raise ValueError(f"Expected frame [B,3,H,W], got {tuple(frame.shape)}.")
        current = self._apply_masks(self._normalize_frames(frame))
        if previous_frame is None:
            previous = torch.zeros_like(current)
        else:
            previous = previous_frame.to(device=current.device, dtype=current.dtype)
            if tuple(previous.shape) != tuple(current.shape):
                raise ValueError(f"Expected previous_frame shape {tuple(current.shape)}, got {tuple(previous.shape)}.")
        motion = current - previous
        return torch.cat((current, motion), dim=1), current

    def extract_spatial_features(
        self,
        frame: torch.Tensor,
        previous_frame: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        visual_input, current = self.prepare_visual_input(frame, previous_frame)
        if visual_input.is_cuda:
            visual_input = visual_input.contiguous(memory_format=torch.channels_last)
        return self.spatial_encoder.feature_stages(visual_input), current

    def _coerce_reset_mask(
        self,
        reset_mask: Optional[torch.Tensor],
        batch_size: int,
        *,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if reset_mask is None:
            return None
        values = reset_mask.to(device=device, dtype=torch.bool).reshape(-1)
        if int(values.numel()) != batch_size:
            raise ValueError(f"Expected reset_mask with {batch_size} values, got {tuple(values.shape)}.")
        return values

    def _reset_batch_rows(
        self,
        value: Optional[torch.Tensor],
        reset_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if value is None or reset_mask is None or not bool(reset_mask.any().item()):
            return value
        view_shape = [int(reset_mask.numel())] + [1] * (value.dim() - 1)
        keep = (~reset_mask).to(device=value.device).view(*view_shape)
        return value * keep.to(dtype=value.dtype)

    def _prepare_sequence_visual_input(
        self,
        frames: torch.Tensor,
        previous_frame: Optional[torch.Tensor] = None,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if frames.dim() != 5 or int(frames.size(2)) != 3:
            raise ValueError(f"Expected frames [B,T,3,H,W], got {tuple(frames.shape)}.")
        current = self._apply_masks(self._normalize_frames(frames))
        batch_size = int(current.size(0))
        if previous_frame is None:
            first_previous = torch.zeros_like(current[:, :1])
        else:
            first_previous = previous_frame.to(device=current.device, dtype=current.dtype)
            if tuple(first_previous.shape) != tuple(current[:, 0].shape):
                raise ValueError(
                    f"Expected previous_frame shape {tuple(current[:, 0].shape)}, got {tuple(first_previous.shape)}."
                )
            reset_mask = self._coerce_reset_mask(reset_mask, batch_size, device=current.device)
            first_previous = self._reset_batch_rows(first_previous, reset_mask)
            first_previous = first_previous.unsqueeze(1)
        previous = torch.cat((first_previous, current[:, :-1]), dim=1)
        motion = current - previous
        return torch.cat((current, motion), dim=2), current

    def _spatial_features(self, visual_input: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels, height, width = visual_input.shape
        flat = visual_input.reshape(batch_size * time_steps, channels, height, width)
        if flat.is_cuda:
            flat = flat.contiguous(memory_format=torch.channels_last)
        features = self.spatial_encoder(flat)
        return features.reshape(
            batch_size,
            time_steps,
            self.cfg.spatial_channels,
            self.feature_size,
            self.feature_size,
        )

    def _run_spatial_gru(
        self,
        features: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs: List[torch.Tensor] = []
        for step in range(int(features.size(1))):
            hidden = self.spatial_gru(features[:, step], hidden)
            outputs.append(hidden)
        if hidden is None:
            raise RuntimeError("Spatial ConvGRU received an empty sequence.")
        return torch.stack(outputs, dim=1), hidden

    def _pool_spatial_states(self, states: torch.Tensor) -> torch.Tensor:
        if states.dim() != 5:
            raise ValueError(f"Expected spatial states [B,T,C,H,W], got {tuple(states.shape)}.")
        batch_size, time_steps, channels, height, width = states.shape
        flat = states.reshape(batch_size * time_steps, channels, height, width)
        compressed = self.channel_compressor(flat)
        pooled = self.spatial_pool(compressed)
        flattened = pooled.flatten(1)
        projected = self.frame_projector(flattened)
        return projected.reshape(batch_size, time_steps, self.cfg.d_model)

    def _coerce_dt(
        self,
        dt: Optional[torch.Tensor],
        batch_size: int,
        time_steps: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if dt is None:
            return torch.full((batch_size, time_steps), float(self.cfg.prediction_dt), device=device, dtype=dtype)
        values = dt.to(device=device, dtype=dtype)
        if values.dim() == 0:
            return values.reshape(1, 1).expand(batch_size, time_steps)
        if values.dim() == 1:
            if int(values.numel()) == batch_size and time_steps == 1:
                return values.reshape(batch_size, 1)
            if int(values.numel()) == time_steps and batch_size == 1:
                return values.reshape(1, time_steps)
            if int(values.numel()) == batch_size:
                return values.reshape(batch_size, 1).expand(batch_size, time_steps)
        if values.dim() == 2 and tuple(values.shape) == (batch_size, time_steps):
            return values
        raise ValueError(f"Expected dt scalar, [B], [T], or [B,T], got {tuple(values.shape)}.")

    def _coerce_prev_action(
        self,
        prev_action: Optional[torch.Tensor],
        batch_size: int,
        time_steps: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if prev_action is None:
            return torch.zeros((batch_size, time_steps, self.cfg.num_bin), device=device, dtype=dtype)
        values = prev_action.to(device=device, dtype=dtype)
        if values.dim() == 2 and tuple(values.shape) == (batch_size, self.cfg.num_bin):
            values = values.unsqueeze(1).expand(batch_size, time_steps, self.cfg.num_bin)
        if values.dim() != 3 or tuple(values.shape) != (batch_size, time_steps, self.cfg.num_bin):
            raise ValueError(
                f"Expected prev_action shape {(batch_size, self.cfg.num_bin)} or "
                f"{(batch_size, time_steps, self.cfg.num_bin)}, got {tuple(values.shape)}."
            )
        if self.training:
            if self.cfg.action_sequence_dropout > 0.0:
                keep_sequence = (
                    torch.rand((batch_size, 1, 1), device=device) >= self.cfg.action_sequence_dropout
                ).to(dtype=dtype)
                values = values * keep_sequence
            if self.cfg.action_key_dropout > 0.0:
                keep_key = (
                    torch.rand((batch_size, 1, self.cfg.num_bin), device=device) >= self.cfg.action_key_dropout
                ).to(dtype=dtype)
                values = values * keep_key
        return values

    def _fuse_context(
        self,
        frame_tokens: torch.Tensor,
        dt: Optional[torch.Tensor],
        prev_action: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, time_steps, _ = frame_tokens.shape
        dt_values = self._coerce_dt(
            dt,
            batch_size,
            time_steps,
            device=frame_tokens.device,
            dtype=frame_tokens.dtype,
        ).clamp(min=1.0 / 240.0, max=0.5)
        dt_ratio = dt_values / float(self.cfg.prediction_dt)
        dt_features = torch.stack((dt_ratio, torch.log(dt_ratio.clamp(min=1e-4))), dim=-1)
        dt_features = self.dt_encoder(dt_features)
        action_values = self._coerce_prev_action(
            prev_action,
            batch_size,
            time_steps,
            device=frame_tokens.device,
            dtype=frame_tokens.dtype,
        )
        action_features = self.action_encoder(action_values)
        context = torch.cat((dt_features, action_features), dim=-1)
        delta = self.context_delta(context)
        gate = torch.sigmoid(self.context_gate(torch.cat((frame_tokens, dt_features, action_features), dim=-1)))
        return self.fused_norm(frame_tokens + gate * delta)

    def _encode_temporal(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"Expected temporal tokens [B,T,D], got {tuple(tokens.shape)}.")
        return tokens

    def _decode_horizons(
        self,
        temporal_features: torch.Tensor,
        dt: Optional[torch.Tensor],
    ) -> PolicyOutput:
        squeeze_time = temporal_features.dim() == 2
        if squeeze_time:
            temporal_features = temporal_features.unsqueeze(1)
        if temporal_features.dim() != 3:
            raise ValueError(f"Expected temporal features [B,T,D], got {tuple(temporal_features.shape)}.")

        batch_size, time_steps, d_model = temporal_features.shape
        offsets = torch.tensor(
            tuple(
                int(offset) + int(self.cfg.action_label_offset)
                for offset in self.cfg.prediction_horizon_offsets
            ),
            device=temporal_features.device,
            dtype=temporal_features.dtype,
        )
        max_offset = offsets[-1].clamp(min=1.0)
        dt_values = self._coerce_dt(
            dt,
            batch_size,
            time_steps,
            device=temporal_features.device,
            dtype=temporal_features.dtype,
        ).clamp(min=1.0 / 240.0, max=0.5)
        normalized_offsets = (offsets / max_offset).reshape(1, 1, self.cfg.prediction_horizon).expand(
            batch_size,
            time_steps,
            -1,
        )
        horizon_seconds = dt_values.unsqueeze(-1) * offsets.reshape(1, 1, self.cfg.prediction_horizon)
        horizon_time = torch.stack((normalized_offsets, horizon_seconds), dim=-1)
        horizon_time = self.horizon_time_encoder(horizon_time)
        horizon_queries = self.horizon_queries.to(device=temporal_features.device, dtype=temporal_features.dtype)
        horizon_base = horizon_queries.reshape(1, 1, self.cfg.prediction_horizon, d_model) + horizon_time

        decoded = self.horizon_norm(temporal_features.unsqueeze(2) + horizon_base)
        button = self.button_head(decoded)
        change = self.change_head(decoded)

        if squeeze_time:
            button = button[:, 0]
            change = change[:, 0]
        return PolicyOutput(
            horizon_button_logits=button,
            horizon_change_logits=change,
        )

    def forward(
        self,
        frames: torch.Tensor,
        dt: Optional[torch.Tensor] = None,
        prev_action: Optional[torch.Tensor] = None,
        state: Optional[TemporalState] = None,
        reset_mask: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ):
        if state is not None or reset_mask is not None or return_state:
            output, next_state = self.forward_sequence_with_state(
                frames,
                dt=dt,
                state=state,
                prev_action=prev_action,
                reset_mask=reset_mask,
            )
            if return_state:
                return output, next_state
            return output
        visual_input, _ = self._prepare_sequence_visual_input(frames)
        features = self._spatial_features(visual_input)
        spatial_states, _ = self._run_spatial_gru(features)
        frame_tokens = self._pool_spatial_states(spatial_states)
        fused = self._fuse_context(frame_tokens, dt, prev_action)
        temporal = self._encode_temporal(fused)
        return self._decode_horizons(temporal, dt)

    def forward_sequence_with_state(
        self,
        frames: torch.Tensor,
        dt: Optional[torch.Tensor],
        state: Optional[TemporalState],
        prev_action: Optional[torch.Tensor] = None,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[PolicyOutput, TemporalState]:
        state = state if state is not None else TemporalState()
        if frames.dim() != 5:
            raise ValueError(f"Expected frames [B,T,3,H,W], got {tuple(frames.shape)}.")
        batch_size = int(frames.size(0))
        reset_mask = self._coerce_reset_mask(reset_mask, batch_size, device=frames.device)

        previous_frame = state.previous_frame
        if previous_frame is not None:
            previous_frame = previous_frame.to(device=frames.device)
        visual_input, current = self._prepare_sequence_visual_input(frames, previous_frame, reset_mask)
        features = self._spatial_features(visual_input)

        spatial_hidden = state.spatial_hidden
        if spatial_hidden is not None:
            spatial_hidden = spatial_hidden.to(device=features.device, dtype=features.dtype)
            if int(spatial_hidden.size(0)) != batch_size:
                raise ValueError(f"Spatial state batch mismatch: expected {batch_size}, got {spatial_hidden.size(0)}.")
            spatial_hidden = self._reset_batch_rows(spatial_hidden, reset_mask)

        spatial_states, spatial_hidden = self._run_spatial_gru(features, spatial_hidden)
        frame_tokens = self._pool_spatial_states(spatial_states)
        fused = self._fuse_context(frame_tokens, dt, prev_action)
        temporal = self._encode_temporal(fused)
        output = self._decode_horizons(temporal, dt)
        next_state = TemporalState(
            previous_frame=current[:, -1].detach(),
            spatial_hidden=spatial_hidden.detach(),
            temporal_tokens=None,
            temporal_lengths=None,
        )
        return output, next_state

    def forward_step(
        self,
        frame: torch.Tensor,
        dt: Optional[torch.Tensor],
        state: Optional[TemporalState],
        prev_action: Optional[torch.Tensor] = None,
    ) -> Tuple[PolicyOutput, TemporalState]:
        state = state if state is not None else TemporalState()
        visual_input, current = self.prepare_visual_input(frame, state.previous_frame)
        if visual_input.is_cuda:
            visual_input = visual_input.contiguous(memory_format=torch.channels_last)
        features = self.spatial_encoder(visual_input)
        spatial_hidden = self.spatial_gru(features, state.spatial_hidden)
        frame_token = self._pool_spatial_states(spatial_hidden.unsqueeze(1))
        fused = self._fuse_context(frame_token, dt, prev_action)
        temporal = self._encode_temporal(fused)
        output = self._decode_horizons(temporal[:, -1], dt)
        next_state = TemporalState(
            previous_frame=current.detach(),
            spatial_hidden=spatial_hidden.detach(),
            temporal_tokens=None,
            temporal_lengths=None,
        )
        return output, next_state
