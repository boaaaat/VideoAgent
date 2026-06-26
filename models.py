"""Streaming driving policy using RGB vision, ConvGRU memory, and action queries.

Training can read a causal action logit at every frame in a window; forward_step
exposes the same computation one frame at a time for online control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from action_space import game_data_root, get_key_names, normalize_game_name, selected_game


RGB_CHANNELS = 3
STEM_CHANNELS = 32
LOW_CHANNELS = 48
MID_CHANNELS = 64
DEEP_CHANNELS = 80
FUSED_CHANNELS = 80
READOUT_CHANNELS = 128
DETAIL_CHANNELS = 64
DEFAULT_SEQUENCE_LENGTH = 80
DEFAULT_ACTION_NAMES = ("w", "a", "s", "d", "z", "c")
ARCHITECTURE_VERSION = "fpn_convgru_attention_v18_32x32_16x16_temporal"
DETAIL_INITIAL_RESIDUAL_GATE = 0.10
TEMPORAL_STATE_LAYERS = 2


def _as_int(name: str, value: object, minimum: int) -> int:
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


def _as_float(name: str, value: object, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, got {value!r}.")
    result = float(value)
    if not math.isfinite(result) or result < minimum or result > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {value!r}.")
    return result


def _prediction_offsets(prediction_horizon: object, value: Optional[Sequence[int]]) -> Tuple[int, ...]:
    if value is None:
        return (_as_int("prediction_horizon", prediction_horizon, 1),)
    offsets = tuple(_as_int(f"prediction_horizon_offsets[{idx}]", item, 1) for idx, item in enumerate(value))
    if len(offsets) != 1:
        raise ValueError(f"The policy has one control head and requires one prediction offset, got {offsets}.")
    return offsets


@dataclass
class ModelConfig:
    """Model and data contract shared by training and online inference."""

    selected_game: str = selected_game
    data_root: Optional[str] = None
    video_ext: str = ".mp4"
    csv_ext: str = ".csv"

    model_size: int = 256
    seq_len: int = DEFAULT_SEQUENCE_LENGTH
    train_seq_stride: int = DEFAULT_SEQUENCE_LENGTH
    val_seq_stride: int = DEFAULT_SEQUENCE_LENGTH
    # Future target measured from the source frame. action_offset=1 means
    # frame[i] is supervised with action[i + 1].
    action_offset: int = 1
    prediction_horizon: int = 1
    prediction_horizon_offsets: Optional[Sequence[int]] = None
    # Retained so existing checkpoint readers can deserialize their configs.
    sequence_output_tail_frames: int = 0

    # The recommended controller intentionally has exactly these six outputs.
    key_names: Optional[List[str]] = None
    mouse_button_names: Optional[List[str]] = None
    num_bin: int = 0

    spatial_dropout: float = 0.10
    head_dropout: float = 0.20
    zoneout: float = 0.0
    attention_temperature: float = 1.0

    button_state_threshold: float = 0.5
    button_state_thresholds: Optional[Sequence[float]] = None
    architecture_version: str = ARCHITECTURE_VERSION

    # Compatibility fields consumed by run.py and checkpoint utilities.
    d_model: int = READOUT_CHANNELS
    action_decoder: str = "spatial_attention"
    action_query_heads: int = 4
    action_query_layers: int = 1
    # Previous controls are a constrained readout-only input. They never enter
    # the visual encoder or ConvGRU state, so vision remains the source of scene
    # understanding while the head can cheaply model key persistence.
    last_action_conditioning: bool = True
    last_action_fusion: str = "bounded_visual_residual"
    last_action_residual_cap: float = 0.75
    last_action_prior_logit: float = 0.0
    last_action_absence_prior_logit: float = 0.0

    def __post_init__(self) -> None:
        self.selected_game = normalize_game_name(self.selected_game)
        if self.data_root is None:
            self.data_root = game_data_root(self.selected_game)

        self.model_size = _as_int("model_size", self.model_size, 32)
        if self.model_size % 16 != 0:
            raise ValueError(f"model_size must be divisible by 16, got {self.model_size}.")
        self.seq_len = _as_int("seq_len", self.seq_len, 1)
        self.train_seq_stride = _as_int("train_seq_stride", self.train_seq_stride, 1)
        self.val_seq_stride = _as_int("val_seq_stride", self.val_seq_stride, 1)
        self.sequence_output_tail_frames = _as_int(
            "sequence_output_tail_frames", self.sequence_output_tail_frames, 0
        )
        self.action_offset = _as_int("action_offset", self.action_offset, 1)
        # Keep the legacy horizon fields synchronized for runtime/debug tools
        # that still read them from checkpoints.
        self.prediction_horizon = self.action_offset
        self.prediction_horizon_offsets = (self.action_offset,)

        available_keys = set(get_key_names(self.selected_game))
        if self.key_names is None:
            self.key_names = list(DEFAULT_ACTION_NAMES)
        else:
            self.key_names = [str(name) for name in self.key_names]
        unknown_keys = [name for name in self.key_names if name not in available_keys]
        if unknown_keys:
            raise ValueError(
                f"Configured action keys are not available for {self.selected_game!r}: {unknown_keys}."
            )
        if tuple(self.key_names) != DEFAULT_ACTION_NAMES:
            raise ValueError(
                "This policy uses the fixed action order "
                f"{list(DEFAULT_ACTION_NAMES)}, got {self.key_names}."
            )

        self.mouse_button_names = [] if self.mouse_button_names is None else list(self.mouse_button_names)
        if self.mouse_button_names:
            raise ValueError("The six-action FPN/ConvGRU policy does not support mouse-button outputs.")
        self.num_bin = len(self.key_names)

        self.spatial_dropout = _as_float("spatial_dropout", self.spatial_dropout, 0.0, 0.9)
        self.head_dropout = _as_float("head_dropout", self.head_dropout, 0.0, 0.9)
        self.zoneout = _as_float("zoneout", self.zoneout, 0.0, 0.9)
        self.attention_temperature = _as_float(
            "attention_temperature", self.attention_temperature, 0.05, 10.0
        )
        self.button_state_threshold = _as_float(
            "button_state_threshold", self.button_state_threshold, 0.0, 1.0
        )
        if self.button_state_thresholds is None:
            self.button_state_thresholds = tuple(
                float(self.button_state_threshold) for _ in range(self.num_bin)
            )
        else:
            thresholds = tuple(float(item) for item in self.button_state_thresholds)
            if len(thresholds) != self.num_bin:
                raise ValueError(
                    f"button_state_thresholds must contain {self.num_bin} values, got {len(thresholds)}."
                )
            for idx, threshold in enumerate(thresholds):
                _as_float(f"button_state_thresholds[{idx}]", threshold, 0.0, 1.0)
            self.button_state_thresholds = thresholds

        self.d_model = _as_int("d_model", self.d_model, 1)
        self.action_query_heads = _as_int("action_query_heads", self.action_query_heads, 1)
        self.action_query_layers = _as_int("action_query_layers", self.action_query_layers, 1)
        self.last_action_conditioning = bool(self.last_action_conditioning)
        self.last_action_fusion = str(self.last_action_fusion).strip().lower()
        if self.last_action_fusion != "bounded_visual_residual":
            raise ValueError(
                "last_action_fusion must be 'bounded_visual_residual', "
                f"got {self.last_action_fusion!r}."
            )
        self.last_action_residual_cap = _as_float(
            "last_action_residual_cap", self.last_action_residual_cap, 0.0, 5.0
        )
        _as_float("last_action_prior_logit", self.last_action_prior_logit, 0.0, 5.0)
        _as_float("last_action_absence_prior_logit", self.last_action_absence_prior_logit, 0.0, 5.0)
        if str(self.architecture_version) != ARCHITECTURE_VERSION:
            raise ValueError(
                f"Expected architecture_version={ARCHITECTURE_VERSION!r}, "
                f"got {self.architecture_version!r}."
            )


@dataclass
class PolicyOutput:
    """Final-frame logits, with optional causal logits for every input frame."""

    button_logits: torch.Tensor
    sequence_button_logits: Optional[torch.Tensor] = None
    vision_button_logits: Optional[torch.Tensor] = None
    sequence_vision_button_logits: Optional[torch.Tensor] = None
    next_feedback_action: Optional[torch.Tensor] = None


@dataclass
class TemporalState:
    """Packed ConvGRU state with shape [2, B, FUSED_CHANNELS, H/8, W/8].

    Layer 0 stores the 32x32 recurrent state. Layer 1 stores the 16x16
    recurrent state in the top-left H/16 x W/16 slice.
    """

    hidden_state: Optional[torch.Tensor] = None


def make_norm(channels: int) -> nn.Module:
    """GroupNorm remains stable when video batch sizes are small."""

    groups = min(8, int(channels))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activate: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.norm = make_norm(out_channels)
        self.act = nn.SiLU(inplace=True) if activate else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Low-cost spatial convolution followed by channel mixing."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1, activate: bool = True) -> None:
        super().__init__()
        self.depthwise = ConvNormAct(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            groups=in_channels,
            activate=True,
        )
        self.pointwise = ConvNormAct(
            in_channels,
            out_channels,
            kernel_size=1,
            activate=activate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class LightweightResidualBlock(nn.Module):
    """Residual block using depthwise-separable convolutions."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, stride=stride, activate=True)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, activate=False)
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else ConvNormAct(in_channels, out_channels, kernel_size=1, stride=stride, activate=False)
        )
        self.out_act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_act(self.conv2(self.conv1(x)) + self.skip(x))


class SharedFrameEncoder(nn.Module):
    """Shared per-frame RGB encoder preserving detail until the first downsample."""

    def __init__(self) -> None:
        super().__init__()
        # The first two convolutions intentionally stay at 256x256.
        self.pre_stem = nn.Sequential(
            ConvNormAct(RGB_CHANNELS, STEM_CHANNELS, kernel_size=3, stride=1),
            ConvNormAct(STEM_CHANNELS, STEM_CHANNELS, kernel_size=3, stride=1),
        )
        self.stem_down = nn.Sequential(
            ConvNormAct(STEM_CHANNELS, STEM_CHANNELS, kernel_size=3, stride=2),
            LightweightResidualBlock(STEM_CHANNELS, STEM_CHANNELS),
        )
        self.low = nn.Sequential(
            LightweightResidualBlock(STEM_CHANNELS, LOW_CHANNELS, stride=2),
            LightweightResidualBlock(LOW_CHANNELS, LOW_CHANNELS),
        )
        self.mid = nn.Sequential(
            LightweightResidualBlock(LOW_CHANNELS, MID_CHANNELS, stride=2),
            LightweightResidualBlock(MID_CHANNELS, MID_CHANNELS),
        )
        self.deep = nn.Sequential(
            LightweightResidualBlock(MID_CHANNELS, DEEP_CHANNELS, stride=2),
            LightweightResidualBlock(DEEP_CHANNELS, DEEP_CHANNELS),
        )

    def feature_stages(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.dim() != 4 or x.size(1) != RGB_CHANNELS:
            raise ValueError(f"Expected RGB images [B,3,H,W], got {tuple(x.shape)}.")
        pre_stem = self.pre_stem(x)
        stem = self.stem_down(pre_stem)
        low = self.low(stem)
        mid = self.mid(low)
        deep = self.deep(mid)
        return [pre_stem, stem, low, mid, deep]

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pre_stem, stem, low, mid, deep = self.feature_stages(x)
        return pre_stem, stem, low, mid, deep


class FPNFusion(nn.Module):
    """Build temporal 32x32/16x16 maps plus current-frame 64x64 detail features."""

    def __init__(self) -> None:
        super().__init__()
        self.low_lateral = ConvNormAct(LOW_CHANNELS, FUSED_CHANNELS, kernel_size=1, activate=False)
        self.mid_lateral = ConvNormAct(MID_CHANNELS, FUSED_CHANNELS, kernel_size=1, activate=False)
        self.deep_lateral = ConvNormAct(DEEP_CHANNELS, FUSED_CHANNELS, kernel_size=1, activate=False)

        self.smooth32 = LightweightResidualBlock(FUSED_CHANNELS, FUSED_CHANNELS)
        self.smooth64 = LightweightResidualBlock(FUSED_CHANNELS, FUSED_CHANNELS)
        self.down64_to_16 = nn.Sequential(
            DepthwiseSeparableConv(FUSED_CHANNELS, FUSED_CHANNELS, stride=2),
            DepthwiseSeparableConv(FUSED_CHANNELS, FUSED_CHANNELS, stride=2),
        )
        # Reuse the 256->128 projection for both the original FPN residual and
        # a direct current-frame detail path. The detail path keeps enough
        # resolution for thin road markings, but presents a more semantic
        # 64x64 map to the action readout.
        self.pre_stem_to_128 = DepthwiseSeparableConv(STEM_CHANNELS, LOW_CHANNELS, stride=2)
        # These widths match the v7 256->16 residual, so the current-frame
        # detail readout does not weaken the 256->16 temporal path.
        self.pre_128_to_64 = DepthwiseSeparableConv(LOW_CHANNELS, MID_CHANNELS, stride=2)
        self.pre_64_to_32 = DepthwiseSeparableConv(MID_CHANNELS, DEEP_CHANNELS, stride=2)
        self.pre_32_to_16 = DepthwiseSeparableConv(DEEP_CHANNELS, FUSED_CHANNELS, stride=2)
        self.low_detail_lateral = ConvNormAct(LOW_CHANNELS, DETAIL_CHANNELS, kernel_size=1, activate=False)
        self.detail64_output = LightweightResidualBlock(DETAIL_CHANNELS, DETAIL_CHANNELS)
        self.temporal32_output = LightweightResidualBlock(FUSED_CHANNELS, FUSED_CHANNELS)
        self.output = LightweightResidualBlock(FUSED_CHANNELS, FUSED_CHANNELS)

    def forward(
        self,
        pre_stem: torch.Tensor,
        low: torch.Tensor,
        mid: torch.Tensor,
        deep: torch.Tensor,
        *,
        return_detail: bool = False,
        return_temporal32: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        p16 = self.deep_lateral(deep)
        p32 = self.mid_lateral(mid) + F.interpolate(p16, size=mid.shape[-2:], mode="bilinear", align_corners=False)
        p32 = self.smooth32(p32)
        p64 = self.low_lateral(low) + F.interpolate(p32, size=low.shape[-2:], mode="bilinear", align_corners=False)
        p64 = self.smooth64(p64)

        pre128 = self.pre_stem_to_128(pre_stem)
        pre64 = self.pre_128_to_64(pre128)
        detail64 = self.detail64_output(pre64 + self.low_detail_lateral(low))
        pre32 = self.pre_64_to_32(pre64)
        pre16 = self.pre_32_to_16(pre32)
        temporal32 = self.temporal32_output(p32 + pre32)

        # p64 already incorporates p32 and p16 through the top-down path. The
        # separate detail64 return is consumed directly by the action readout,
        # not collapsed into the temporal 16x16 feature map.
        fused16 = self.output(
            p16
            + self.down64_to_16(p64)
            + pre16
        )
        outputs: List[torch.Tensor] = [fused16]
        if return_detail:
            outputs.append(detail64)
        if return_temporal32:
            outputs.append(temporal32)
        return tuple(outputs) if len(outputs) > 1 else fused16


class ConvGRUCell(nn.Module):
    """Causal ConvGRU update that preserves the 2-D feature layout."""

    def __init__(self, channels: int = FUSED_CHANNELS) -> None:
        super().__init__()
        self.channels = int(channels)
        self.gates = nn.Conv2d(2 * self.channels, 2 * self.channels, kernel_size=3, padding=1)
        self.candidate = nn.Conv2d(2 * self.channels, self.channels, kernel_size=3, padding=1)
        with torch.no_grad():
            # This implementation uses the first gate as the candidate-write
            # amount: h = (1 - update) * previous + update * candidate. A
            # negative initial bias therefore preserves memory instead of
            # overwriting it every frame.
            self.gates.bias[: self.channels].fill_(-2.0)

    def forward(self, x: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        if x.shape != previous.shape:
            raise ValueError(
                f"ConvGRU input/state shapes must match, got input={tuple(x.shape)} state={tuple(previous.shape)}."
            )
        update, reset = self.gates(torch.cat((x, previous), dim=1)).chunk(2, dim=1)
        update = torch.sigmoid(update)
        reset = torch.sigmoid(reset)
        candidate = torch.tanh(self.candidate(torch.cat((x, reset * previous), dim=1)))
        return (1.0 - update) * previous + update * candidate


class LearnedSpatialQueryPool(nn.Module):
    """Action-specific learned queries over the final spatial map."""

    def __init__(self, channels: int, num_actions: int, dropout: float, temperature: float) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_actions = int(num_actions)
        self.temperature = float(temperature)
        self.token_norm = nn.LayerNorm(self.channels)
        self.value = nn.Linear(self.channels, self.channels, bias=False)
        self.queries = nn.Parameter(torch.empty(self.num_actions, self.channels))
        self.position = nn.Parameter(torch.empty(1, 16 * 16, self.channels))
        self.out_norm = nn.LayerNorm(self.channels)
        self.dropout = nn.Dropout(float(dropout))
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def _position_for(self, height: int, width: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        position = self.position.reshape(1, 16, 16, self.channels).permute(0, 3, 1, 2)
        if (height, width) != (16, 16):
            position = F.interpolate(position, size=(height, width), mode="bilinear", align_corners=False)
        return position.permute(0, 2, 3, 1).reshape(1, height * width, self.channels).to(
            device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.size(1) != self.channels:
            raise ValueError(f"Expected readout map [B,{self.channels},H,W], got {tuple(x.shape)}.")
        batch, _, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        keys = self.token_norm(tokens + self._position_for(height, width, dtype=x.dtype, device=x.device))
        values = self.value(keys)
        queries = self.queries.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(batch, -1, -1)
        scores = torch.matmul(queries, keys.transpose(1, 2))
        scores = scores / (math.sqrt(self.channels) * self.temperature)
        weights = torch.softmax(scores, dim=-1)
        return self.out_norm(self.dropout(torch.matmul(weights, values)))


class TemporalConditionedDetailPool(nn.Module):
    """Pool a current 64x64 detail map with temporal action-specific queries."""

    def __init__(self, detail_channels: int, query_channels: int, dropout: float, temperature: float) -> None:
        super().__init__()
        self.detail_channels = int(detail_channels)
        self.query_channels = int(query_channels)
        self.temperature = float(temperature)
        self.token_norm = nn.LayerNorm(self.detail_channels)
        self.query = nn.Linear(self.query_channels, self.detail_channels, bias=False)
        self.value = nn.Linear(self.detail_channels, self.detail_channels, bias=False)
        # A compact 16x16 learned grid is interpolated for the 64x64 map. This
        # provides position information without a large 64x64 parameter table.
        self.position = nn.Parameter(torch.empty(1, 16 * 16, self.detail_channels))
        self.out_norm = nn.LayerNorm(self.detail_channels)
        self.dropout = nn.Dropout(float(dropout))
        nn.init.trunc_normal_(self.position, std=0.02)

    def _position_for(self, height: int, width: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        position = self.position.reshape(1, 16, 16, self.detail_channels).permute(0, 3, 1, 2)
        if (height, width) != (16, 16):
            position = F.interpolate(position, size=(height, width), mode="bilinear", align_corners=False)
        return position.permute(0, 2, 3, 1).reshape(1, height * width, self.detail_channels).to(
            device=device, dtype=dtype
        )

    def forward(self, detail_map: torch.Tensor, temporal_action_features: torch.Tensor) -> torch.Tensor:
        if detail_map.dim() != 4 or detail_map.size(1) != self.detail_channels:
            raise ValueError(
                f"Expected detail map [B,{self.detail_channels},H,W], got {tuple(detail_map.shape)}."
            )
        if (
            temporal_action_features.dim() != 3
            or temporal_action_features.size(0) != detail_map.size(0)
            or temporal_action_features.size(-1) != self.query_channels
        ):
            raise ValueError(
                "Expected temporal action features "
                f"[B,A,{self.query_channels}], got {tuple(temporal_action_features.shape)}."
            )

        batch, _, height, width = detail_map.shape
        tokens = detail_map.flatten(2).transpose(1, 2)
        keys = self.token_norm(tokens + self._position_for(height, width, dtype=detail_map.dtype, device=detail_map.device))
        values = self.value(keys)
        queries = self.query(temporal_action_features)
        # SDPA selects a fused CUDA kernel when available and avoids allocating
        # [B, actions, 4096] attention weights for every video timestep.
        pooled = F.scaled_dot_product_attention(
            queries.unsqueeze(1),
            keys.unsqueeze(1),
            values.unsqueeze(1),
            dropout_p=0.0,
            scale=1.0 / (math.sqrt(self.detail_channels) * self.temperature),
        ).squeeze(1)
        return self.out_norm(self.dropout(pooled))


def apply_static_masks(frames: torch.Tensor, *, clone: bool = True) -> torch.Tensor:
    """Mask game UI regions while retaining a three-channel RGB input."""

    if frames.dim() < 4:
        raise ValueError(f"Expected image tensor with at least four dimensions, got {tuple(frames.shape)}.")
    height, width = frames.shape[-2:]
    output = frames.clone() if clone else frames
    output[..., int(height * 0.96) :, :] = 0.0
    output[..., int(height * 0.05) : int(height * 0.20), int(width * 0.75) :] = 0.0
    output[..., : int(height * 0.10), : int(width * 0.10)] = 0.0
    return output


class DrivingVideoPolicy(nn.Module):
    """Causal behavioral-cloning policy with streaming inference."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.spatial_encoder = SharedFrameEncoder()
        self.fpn = FPNFusion()
        self.gru32 = ConvGRUCell(FUSED_CHANNELS)
        self.temporal32_to_16 = DepthwiseSeparableConv(FUSED_CHANNELS, FUSED_CHANNELS, stride=2)
        self.temporal16_fusion = LightweightResidualBlock(FUSED_CHANNELS, FUSED_CHANNELS)
        self.gru1 = ConvGRUCell(FUSED_CHANNELS)
        self.readout_fusion = nn.Sequential(
            ConvNormAct(2 * FUSED_CHANNELS, READOUT_CHANNELS, kernel_size=1),
            DepthwiseSeparableConv(READOUT_CHANNELS, READOUT_CHANNELS),
        )
        self.spatial_dropout = nn.Dropout2d(float(cfg.spatial_dropout))
        self.query_pool = LearnedSpatialQueryPool(
            READOUT_CHANNELS,
            cfg.num_bin,
            dropout=float(cfg.head_dropout),
            temperature=float(cfg.attention_temperature),
        )
        self.detail_pool = TemporalConditionedDetailPool(
            DETAIL_CHANNELS,
            READOUT_CHANNELS,
            dropout=float(cfg.head_dropout),
            temperature=float(cfg.attention_temperature),
        )
        self.detail_projection = nn.Sequential(
            nn.LayerNorm(DETAIL_CHANNELS),
            nn.Linear(DETAIL_CHANNELS, READOUT_CHANNELS, bias=False),
        )
        self.detail_gate = nn.Linear(READOUT_CHANNELS, READOUT_CHANNELS)
        self.control_head = nn.Sequential(
            nn.LayerNorm(READOUT_CHANNELS),
            nn.Linear(READOUT_CHANNELS, READOUT_CHANNELS),
            nn.SiLU(inplace=True),
            nn.Dropout(float(cfg.head_dropout)),
            nn.Linear(READOUT_CHANNELS, 1),
        )
        self.action_context = nn.Sequential(
            nn.Linear(cfg.num_bin, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, cfg.num_bin),
        )
        self.action_context_gate = nn.Linear(READOUT_CHANNELS, 1)
        action_codes = torch.arange(1 << cfg.num_bin, dtype=torch.long)
        action_bits = torch.tensor([1 << index for index in range(cfg.num_bin)], dtype=torch.long)
        action_states = ((action_codes.unsqueeze(1) & action_bits.unsqueeze(0)) != 0).float()
        self.register_buffer("_action_context_states", action_states, persistent=False)
        self.register_buffer("_action_context_bits", action_bits, persistent=False)
        final = self.control_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.constant_(final.bias, -1.0)
        context_final = self.action_context[-1]
        if isinstance(context_final, nn.Linear):
            nn.init.zeros_(context_final.weight)
            nn.init.zeros_(context_final.bias)
        # Start the 64x64 detail path as a small residual instead of gating it
        # off entirely; otherwise the detail pool/projection receive no useful
        # gradients at initialization. Zero weights keep the initial scale
        # uniform while the model learns channel-specific gates.
        nn.init.zeros_(self.detail_gate.weight)
        nn.init.constant_(self.detail_gate.bias, math.atanh(DETAIL_INITIAL_RESIDUAL_GATE))

        # Private compatibility attributes used by existing visual tooling.
        self.feat_channels = FUSED_CHANNELS
        self.temporal_spatial_fusion = nn.Identity()

    def _normalize_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dtype == torch.uint8:
            return frames.float().div_(255.0)
        if not torch.is_floating_point(frames):
            raise TypeError(f"frames must be uint8 or floating point, got {frames.dtype}.")
        return frames

    def _apply_masks(self, frames: torch.Tensor, *, clone: bool = True) -> torch.Tensor:
        return apply_static_masks(frames, clone=clone)

    def _initial_temporal_state(
        self,
        batch_size: int,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        high_height: Optional[int] = None,
        high_width: Optional[int] = None,
        **_: object,
    ) -> torch.Tensor:
        high_height = int(high_height) if high_height is not None else int(height) * 2
        high_width = int(high_width) if high_width is not None else int(width) * 2
        return torch.zeros(
            TEMPORAL_STATE_LAYERS,
            int(batch_size),
            FUSED_CHANNELS,
            high_height,
            high_width,
            device=device,
            dtype=dtype,
        )

    def _pack_temporal_state(
        self,
        high32: torch.Tensor,
        low16: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, high_height, high_width = high32.shape
        _, _, low_height, low_width = low16.shape
        packed = high32.new_zeros(
            TEMPORAL_STATE_LAYERS,
            batch,
            FUSED_CHANNELS,
            high_height,
            high_width,
        )
        packed[0] = high32
        packed[1, :, :, :low_height, :low_width] = low16
        return packed

    def _unpack_temporal_state(
        self,
        hidden_state: torch.Tensor,
        batch_size: int,
        low_height: int,
        low_width: int,
        high_height: int,
        high_width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        expected = (
            TEMPORAL_STATE_LAYERS,
            int(batch_size),
            FUSED_CHANNELS,
            int(high_height),
            int(high_width),
        )
        if tuple(hidden_state.shape) != expected:
            raise ValueError(f"Expected temporal state {expected}, got {tuple(hidden_state.shape)}.")
        high32 = hidden_state[0]
        low16 = hidden_state[1, :, :, :low_height, :low_width].contiguous()
        return high32.contiguous(), low16

    def _prepare_temporal_state(
        self,
        state: Optional[TemporalState],
        batch_size: int,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        high_height: Optional[int] = None,
        high_width: Optional[int] = None,
        **_: object,
    ) -> torch.Tensor:
        high_height = int(high_height) if high_height is not None else int(height) * 2
        high_width = int(high_width) if high_width is not None else int(width) * 2
        if state is None or state.hidden_state is None:
            return self._initial_temporal_state(
                batch_size,
                height,
                width,
                device=device,
                dtype=dtype,
                high_height=high_height,
                high_width=high_width,
            )
        hidden = state.hidden_state.to(device=device, dtype=dtype)
        if hidden.dim() == 4:
            if tuple(hidden.shape) == (batch_size, FUSED_CHANNELS, high_height, high_width):
                packed = self._initial_temporal_state(
                    batch_size,
                    height,
                    width,
                    device=device,
                    dtype=dtype,
                    high_height=high_height,
                    high_width=high_width,
                )
                packed[0] = hidden
                return packed
            if tuple(hidden.shape) != (batch_size, FUSED_CHANNELS, height, width):
                raise ValueError(f"Invalid temporal state shape {tuple(hidden.shape)}.")
            packed = self._initial_temporal_state(
                batch_size,
                height,
                width,
                device=device,
                dtype=dtype,
                high_height=high_height,
                high_width=high_width,
            )
            packed[1, :, :, :height, :width] = hidden
            return packed
        expected = (
            TEMPORAL_STATE_LAYERS,
            int(batch_size),
            FUSED_CHANNELS,
            int(high_height),
            int(high_width),
        )
        if tuple(hidden.shape) != expected:
            if hidden.dim() == 5 and hidden.size(0) == 2:
                expected_legacy = (2, int(batch_size), FUSED_CHANNELS, int(height), int(width))
                if tuple(hidden.shape) == expected_legacy:
                    packed = self._initial_temporal_state(
                        batch_size,
                        height,
                        width,
                        device=device,
                        dtype=dtype,
                        high_height=high_height,
                        high_width=high_width,
                    )
                    packed[1, :, :, :height, :width] = hidden[-1]
                    return packed
            if hidden.dim() == 5 and hidden.size(0) == 3:
                expected_previous = (3, int(batch_size), FUSED_CHANNELS, int(high_height), int(high_width))
                if tuple(hidden.shape) == expected_previous:
                    packed = self._initial_temporal_state(
                        batch_size,
                        height,
                        width,
                        device=device,
                        dtype=dtype,
                        high_height=high_height,
                        high_width=high_width,
                    )
                    packed[0] = hidden[0]
                    packed[1, :, :, :height, :width] = hidden[2, :, :, :height, :width]
                    return packed
            raise ValueError(f"Expected temporal state {expected}, got {tuple(hidden.shape)}.")
        return hidden

    def _zoneout(self, candidate: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        if not self.training or self.cfg.zoneout <= 0.0:
            return candidate
        keep = torch.empty_like(candidate).bernoulli_(1.0 - float(self.cfg.zoneout))
        return candidate * keep + previous * (1.0 - keep)

    def _temporal_step(
        self,
        fused_frame: torch.Tensor,
        temporal32_frame: torch.Tensor,
        hidden_state: torch.Tensor,
        **_: object,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if fused_frame.dim() != 4 or fused_frame.size(1) != FUSED_CHANNELS:
            raise ValueError(f"Expected FPN feature [B,{FUSED_CHANNELS},H,W], got {tuple(fused_frame.shape)}.")
        if temporal32_frame.dim() != 4 or temporal32_frame.size(1) != FUSED_CHANNELS:
            raise ValueError(
                f"Expected 32x32 temporal feature [B,{FUSED_CHANNELS},H,W], got {tuple(temporal32_frame.shape)}."
            )
        hidden_state = self._prepare_temporal_state(
            TemporalState(hidden_state=hidden_state),
            fused_frame.size(0),
            fused_frame.size(-2),
            fused_frame.size(-1),
            device=fused_frame.device,
            dtype=fused_frame.dtype,
            high_height=temporal32_frame.size(-2),
            high_width=temporal32_frame.size(-1),
        )
        h32_prev, h16_prev = self._unpack_temporal_state(
            hidden_state,
            fused_frame.size(0),
            fused_frame.size(-2),
            fused_frame.size(-1),
            temporal32_frame.size(-2),
            temporal32_frame.size(-1),
        )
        h32 = self._zoneout(self.gru32(temporal32_frame, h32_prev), h32_prev)
        h32_down = self.temporal32_to_16(h32)
        if h32_down.shape[-2:] != fused_frame.shape[-2:]:
            h32_down = F.interpolate(h32_down, size=fused_frame.shape[-2:], mode="bilinear", align_corners=False)
        fused16 = self.temporal16_fusion(fused_frame + h32_down)
        h16 = self._zoneout(self.gru1(fused16, h16_prev), h16_prev)
        return h16, self._pack_temporal_state(h32, h16)

    def _temporal_sequence(
        self,
        fused: torch.Tensor,
        temporal32: torch.Tensor,
        hidden_state: torch.Tensor,
        *,
        return_steps: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if fused.dim() != 5 or temporal32.dim() != 5:
            raise ValueError(
                "Expected temporal feature sequences [B,T,C,H,W], got "
                f"{tuple(fused.shape)} and {tuple(temporal32.shape)}."
            )
        if fused.size(0) != temporal32.size(0) or fused.size(1) != temporal32.size(1):
            raise ValueError(
                "FPN and 32x32 temporal sequences must share batch/time dimensions, got "
                f"{tuple(fused.shape[:2])} and {tuple(temporal32.shape[:2])}."
            )
        if fused.size(2) != FUSED_CHANNELS or temporal32.size(2) != FUSED_CHANNELS:
            raise ValueError(
                f"Expected {FUSED_CHANNELS} temporal channels, got "
                f"{fused.size(2)} and {temporal32.size(2)}."
            )

        batch, steps, _, height, width = fused.shape
        high_height, high_width = temporal32.shape[-2:]
        h32, h16 = self._unpack_temporal_state(
            hidden_state,
            batch,
            height,
            width,
            high_height,
            high_width,
        )
        step_outputs = [] if return_steps else None
        for index in range(steps):
            h32 = self._zoneout(self.gru32(temporal32[:, index], h32), h32)
            h32_down = self.temporal32_to_16(h32)
            if h32_down.shape[-2:] != fused.shape[-2:]:
                h32_down = F.interpolate(h32_down, size=fused.shape[-2:], mode="bilinear", align_corners=False)
            fused16 = self.temporal16_fusion(fused[:, index] + h32_down)
            h16 = self._zoneout(self.gru1(fused16, h16), h16)
            if step_outputs is not None:
                step_outputs.append(h16)
        hidden_sequence = torch.stack(step_outputs, dim=1) if step_outputs is not None else None
        return h16, self._pack_temporal_state(h32, h16), hidden_sequence

    def _pool_features(self, spatial_features: torch.Tensor) -> torch.Tensor:
        if spatial_features.size(1) != READOUT_CHANNELS:
            raise ValueError(
                f"Expected {READOUT_CHANNELS} readout channels for attention pooling, got "
                f"{spatial_features.size(1)}."
            )
        return self.query_pool(spatial_features)

    def _features_to_logits(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() in (3, 4):
            return self.control_head(features).squeeze(-1)
        raise ValueError(f"Expected action features [B,6,C] or [B,T,6,C], got {tuple(features.shape)}.")

    def _validate_action_context(
        self,
        prev_action: torch.Tensor,
        action_features: torch.Tensor,
    ) -> torch.Tensor:
        expected = tuple(action_features.shape[:-1])
        if tuple(prev_action.shape) != expected:
            raise ValueError(
                "Previous action context must match action-feature batch/time/action dimensions; "
                f"expected {expected}, got {tuple(prev_action.shape)}."
            )
        if prev_action.size(-1) != self.cfg.num_bin:
            raise ValueError(
                f"Expected {self.cfg.num_bin} previous-action values, got {prev_action.size(-1)}."
            )
        return prev_action.to(device=action_features.device, dtype=action_features.dtype)

    def _fuse_action_context(
        self,
        vision_logits: torch.Tensor,
        action_features: torch.Tensor,
        prev_action: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply a bounded, visual-gated one-step action residual."""

        if not self.cfg.last_action_conditioning or prev_action is None:
            return vision_logits
        context = self._validate_action_context(prev_action, action_features)
        residual = self.action_context(context)
        gate = torch.sigmoid(self.action_context_gate(action_features)).squeeze(-1)
        cap = float(self.cfg.last_action_residual_cap)
        return vision_logits + gate * cap * torch.tanh(residual)

    def _action_state_codes(self, action: torch.Tensor) -> torch.Tensor:
        """Encode one hard six-key state per batch row as an integer lookup key."""

        if action.dim() != 2 or action.size(-1) != self.cfg.num_bin:
            raise ValueError(
                f"Expected hard action states [B,{self.cfg.num_bin}], got {tuple(action.shape)}."
            )
        bits = self._action_context_bits.to(device=action.device)
        return ((action >= 0.5).to(dtype=torch.long) * bits.view(1, -1)).sum(dim=-1)

    def _encode_frame(self, frame: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre_stem, _stem, low, mid, deep = self.spatial_encoder(frame)
        fused, detail64, temporal32 = self.fpn(
            pre_stem,
            low,
            mid,
            deep,
            return_detail=True,
            return_temporal32=True,
        )
        return fused, detail64, temporal32

    def _readout_features(
        self,
        current_fpn: torch.Tensor,
        hidden: torch.Tensor,
        detail64: torch.Tensor,
    ) -> torch.Tensor:
        readout = self.readout_fusion(torch.cat((hidden, current_fpn), dim=1))
        readout = self.spatial_dropout(readout)
        temporal_features = self.query_pool(readout)
        detail_features = self.detail_pool(detail64, temporal_features)
        detail_delta = self.detail_projection(detail_features)
        detail_gate = torch.tanh(self.detail_gate(temporal_features))
        return temporal_features + detail_gate * detail_delta

    def _readout_logits(
        self,
        current_fpn: torch.Tensor,
        hidden: torch.Tensor,
        detail64: torch.Tensor,
        prev_action: Optional[torch.Tensor] = None,
        *,
        return_vision: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        features = self._readout_features(current_fpn, hidden, detail64)
        vision_logits = self._features_to_logits(features)
        fused_logits = self._fuse_action_context(vision_logits, features, prev_action)
        return (fused_logits, vision_logits) if return_vision else fused_logits

    def _readout_sequence_features(
        self,
        current_fpn: torch.Tensor,
        hidden: torch.Tensor,
        detail64: torch.Tensor,
    ) -> torch.Tensor:
        """Read all timestep logits in one batched head invocation.

        The ConvGRU must remain sequential, but the readout has no dependency
        between timesteps. Flattening B and T avoids compiling and launching
        the fusion, attention, and control head once per frame.
        """
        if current_fpn.dim() != 5 or hidden.dim() != 5 or current_fpn.shape != hidden.shape:
            raise ValueError(
                "Expected matching current/hidden feature sequences [B,T,C,H,W], got "
                f"{tuple(current_fpn.shape)} and {tuple(hidden.shape)}."
            )
        batch, steps, channels, height, width = current_fpn.shape
        if channels != FUSED_CHANNELS:
            raise ValueError(f"Expected {FUSED_CHANNELS} feature channels, got {channels}.")
        if (
            detail64.dim() != 5
            or tuple(detail64.shape[:2]) != (batch, steps)
            or detail64.size(2) != DETAIL_CHANNELS
            or tuple(detail64.shape[-2:]) != (height * 4, width * 4)
        ):
            raise ValueError(
                "Expected matching 64x64 detail sequence [B,T,64,4H,4W], got "
                f"{tuple(detail64.shape)}."
            )
        flat_current = current_fpn.reshape(batch * steps, channels, height, width)
        flat_hidden = hidden.reshape(batch * steps, channels, height, width)
        flat_detail = detail64.reshape(batch * steps, DETAIL_CHANNELS, height * 4, width * 4)
        flat_features = self._readout_features(flat_current, flat_hidden, flat_detail)
        return flat_features.reshape(batch, steps, self.cfg.num_bin, READOUT_CHANNELS)

    def _readout_sequence_logits(
        self,
        current_fpn: torch.Tensor,
        hidden: torch.Tensor,
        detail64: torch.Tensor,
        prev_action: Optional[torch.Tensor] = None,
        *,
        return_vision: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        features = self._readout_sequence_features(current_fpn, hidden, detail64)
        vision_logits = self._features_to_logits(features)
        fused_logits = self._fuse_action_context(vision_logits, features, prev_action)
        return (fused_logits, vision_logits) if return_vision else fused_logits

    def _autoregressive_sequence_logits(
        self,
        current_fpn: torch.Tensor,
        hidden: torch.Tensor,
        detail64: torch.Tensor,
        initial_prev_action: Optional[torch.Tensor],
        feedback_thresholds: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply detached hard feedback without allowing gradients through actions."""

        features = self._readout_sequence_features(current_fpn, hidden, detail64)
        vision_logits = self._features_to_logits(features)
        batch, steps, actions = vision_logits.shape
        if initial_prev_action is None:
            previous = vision_logits.new_zeros((batch, actions))
        else:
            if tuple(initial_prev_action.shape) != (batch, actions):
                raise ValueError(
                    "Autoregressive initial action must have shape "
                    f"[{batch},{actions}], got {tuple(initial_prev_action.shape)}."
                )
            previous = initial_prev_action.to(device=vision_logits.device, dtype=vision_logits.dtype)
        if feedback_thresholds is None:
            thresholds = vision_logits.new_full((actions,), 0.5)
        else:
            thresholds = feedback_thresholds.to(device=vision_logits.device, dtype=vision_logits.dtype).reshape(-1)
            if thresholds.numel() != actions:
                raise ValueError(f"Expected {actions} feedback thresholds, got {thresholds.numel()}.")
        if not self.cfg.last_action_conditioning:
            final_feedback = (torch.sigmoid(vision_logits[:, -1].detach()) >= thresholds.view(1, -1)).to(
                dtype=vision_logits.dtype
            )
            return vision_logits, vision_logits, final_feedback

        context_states = self._action_context_states.to(device=vision_logits.device, dtype=vision_logits.dtype)
        residual_table = float(self.cfg.last_action_residual_cap) * torch.tanh(self.action_context(context_states))
        visual_gates = torch.sigmoid(self.action_context_gate(features)).squeeze(-1)
        previous_code = self._action_state_codes(previous)
        fused_steps = []
        for index in range(steps):
            residual = residual_table.index_select(0, previous_code)
            fused = vision_logits[:, index] + visual_gates[:, index] * residual
            fused_steps.append(fused)
            previous = (torch.sigmoid(fused.detach()) >= thresholds.view(1, -1)).to(dtype=vision_logits.dtype)
            previous_code = self._action_state_codes(previous)
        return torch.stack(fused_steps, dim=1), vision_logits, previous.detach()

    def forward_step(
        self,
        frame: torch.Tensor,
        state: Optional[TemporalState] = None,
        *,
        prev_action: Optional[torch.Tensor] = None,
    ) -> Tuple[PolicyOutput, TemporalState]:
        """Run one causal RGB frame and return its six future-action logits."""

        if frame.dim() != 4 or frame.size(1) != RGB_CHANNELS:
            raise ValueError(f"Expected frame [B,3,H,W], got {tuple(frame.shape)}.")
        frame = self._normalize_frames(frame)
        frame = self._apply_masks(frame, clone=bool(frame.requires_grad))
        if frame.is_cuda:
            frame = frame.contiguous(memory_format=torch.channels_last)
        fused, detail64, temporal32 = self._encode_frame(frame)
        hidden = self._prepare_temporal_state(
            state,
            fused.size(0),
            fused.size(-2),
            fused.size(-1),
            device=fused.device,
            dtype=fused.dtype,
            high_height=temporal32.size(-2),
            high_width=temporal32.size(-1),
        )
        final_hidden, next_hidden = self._temporal_step(fused, temporal32, hidden)
        logits, vision_logits = self._readout_logits(
            fused,
            final_hidden,
            detail64,
            prev_action=prev_action,
            return_vision=True,
        )
        return (
            PolicyOutput(button_logits=logits, vision_button_logits=vision_logits),
            TemporalState(hidden_state=next_hidden.detach()),
        )

    def forward(
        self,
        frames: torch.Tensor,
        state: Optional[TemporalState] = None,
        *,
        return_aux: bool = False,
        return_sequence_logits: bool = False,
        prev_action: Optional[torch.Tensor] = None,
        feedback_mask: Optional[torch.Tensor] = None,
        feedback_thresholds: Optional[torch.Tensor] = None,
        soft_feedback: bool = False,
        autoregressive_feedback: bool = False,
    ) -> PolicyOutput | Tuple[PolicyOutput, TemporalState]:
        """Run a causal sequence and optionally return logits at every timestep."""

        del feedback_mask
        if soft_feedback:
            raise ValueError("Soft previous-action feedback is disabled for this architecture.")
        if frames.dim() != 5 or frames.size(2) != RGB_CHANNELS:
            raise ValueError(f"Expected frames [B,T,3,H,W], got {tuple(frames.shape)}.")
        if frames.size(1) <= 0:
            raise ValueError("The frame sequence must contain at least one frame.")
        frames = self._normalize_frames(frames)
        frames = self._apply_masks(frames, clone=bool(frames.requires_grad))
        batch, steps, _, height, width = frames.shape
        flat = frames.reshape(batch * steps, RGB_CHANNELS, height, width)
        if flat.is_cuda:
            flat = flat.contiguous(memory_format=torch.channels_last)
        pre_stem, _stem, low, mid, deep = self.spatial_encoder(flat)
        fused, detail64, temporal32 = self.fpn(
            pre_stem,
            low,
            mid,
            deep,
            return_detail=True,
            return_temporal32=True,
        )
        fused = fused.reshape(
            batch, steps, FUSED_CHANNELS, deep.size(-2), deep.size(-1)
        )
        temporal32 = temporal32.reshape(
            batch, steps, FUSED_CHANNELS, mid.size(-2), mid.size(-1)
        )
        detail64 = detail64.reshape(
            batch, steps, DETAIL_CHANNELS, low.size(-2), low.size(-1)
        )
        hidden = self._prepare_temporal_state(
            state,
            batch,
            fused.size(-2),
            fused.size(-1),
            device=fused.device,
            dtype=fused.dtype,
            high_height=temporal32.size(-2),
            high_width=temporal32.size(-1),
        )
        final_hidden, hidden, hidden_sequence = self._temporal_sequence(
            fused,
            temporal32,
            hidden,
            return_steps=return_sequence_logits or autoregressive_feedback,
        )
        if hidden_sequence is None:
            final_prev_action = prev_action
            if prev_action is not None and prev_action.dim() == 3:
                expected = (batch, steps, self.cfg.num_bin)
                if tuple(prev_action.shape) != expected:
                    raise ValueError(
                        f"Expected sequence previous actions {expected}, got {tuple(prev_action.shape)}."
                    )
                final_prev_action = prev_action[:, -1]
            final_logits, final_vision_logits = self._readout_logits(
                fused[:, -1],
                final_hidden,
                detail64[:, -1],
                prev_action=final_prev_action,
                return_vision=True,
            )
            next_feedback_action = None
            dense_logits = None
            dense_vision_logits = None
        else:
            if autoregressive_feedback:
                dense_logits, dense_vision_logits, next_feedback_action = self._autoregressive_sequence_logits(
                    fused,
                    hidden_sequence,
                    detail64,
                    prev_action,
                    feedback_thresholds,
                )
            else:
                dense_logits, dense_vision_logits = self._readout_sequence_logits(
                    fused,
                    hidden_sequence,
                    detail64,
                    prev_action=prev_action,
                    return_vision=True,
                )
                next_feedback_action = None
            final_logits = dense_logits[:, -1]
            final_vision_logits = dense_vision_logits[:, -1]
            if not return_sequence_logits:
                dense_logits = None
                dense_vision_logits = None
        output = PolicyOutput(
            button_logits=final_logits,
            sequence_button_logits=dense_logits,
            vision_button_logits=final_vision_logits,
            sequence_vision_button_logits=dense_vision_logits,
            next_feedback_action=next_feedback_action,
        )
        next_state = TemporalState(hidden_state=hidden.detach())
        return (output, next_state) if return_aux else output
