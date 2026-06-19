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


RGB_INPUT_CHANNELS = 3
POLICY_INPUT_CHANNELS = RGB_INPUT_CHANNELS
STEM_CHANNELS = 32
STAGE64_CHANNELS = 48
STAGE32_CHANNELS = 96
POLICY_FEATURE_CHANNELS = 96
SPATIAL_GRID_SIZE = 16
SPATIAL_GRID_COORD_CHANNELS = 2
PERSPECTIVE_COORD_CHANNELS = 4
POLICY_POOLED_FEATURES = (POLICY_FEATURE_CHANNELS + SPATIAL_GRID_COORD_CHANNELS) * SPATIAL_GRID_SIZE * SPATIAL_GRID_SIZE
POLICY_HEAD_HIDDEN = 256
POLICY_HEAD_FEATURES = 128
ACTION_QUERY_DECODER_HEADS = 4
ACTION_QUERY_DECODER_LAYERS = 2
TEMPORAL_STATE_LAYERS = 3
PACKED_STATE_CHANNELS = POLICY_FEATURE_CHANNELS
LAST_ACTION_EMBEDDING_DROPOUT = 0.25
DEFAULT_PREDICTION_OFFSET = 1
ACTION_DECODER_MLP = "mlp"
ACTION_DECODER_ACTION_QUERY = "action_query"
ACTION_DECODERS = (ACTION_DECODER_MLP, ACTION_DECODER_ACTION_QUERY)


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
    action_decoder: str = ACTION_DECODER_MLP
    action_query_heads: int = ACTION_QUERY_DECODER_HEADS
    action_query_layers: int = ACTION_QUERY_DECODER_LAYERS
    # Feed the previous action into the heads. Training can replace teacher
    # actions with detached model outputs, so this remains enabled for current
    # runs while legacy checkpoints with last_action_encoder weights still load.
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
        self.action_decoder = str(self.action_decoder).strip().lower()
        if self.action_decoder not in ACTION_DECODERS:
            raise ValueError(f"action_decoder must be one of {ACTION_DECODERS}, got {self.action_decoder!r}.")
        self.action_query_heads = _require_int_at_least("action_query_heads", self.action_query_heads, 1)
        if POLICY_HEAD_FEATURES % self.action_query_heads != 0:
            raise ValueError(
                f"action_query_heads must divide {POLICY_HEAD_FEATURES}, got {self.action_query_heads}."
            )
        self.action_query_layers = _require_int_at_least("action_query_layers", self.action_query_layers, 1)
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


def make_norm(channels: int, groups: int = 8) -> nn.Module:
    """GroupNorm is stable for the small video batches used here."""
    group_count = min(int(groups), int(channels))
    while channels % group_count != 0 and group_count > 1:
        group_count -= 1
    return nn.GroupNorm(group_count, channels)


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        act: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = ((kernel_size - 1) // 2) * dilation
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.norm = make_norm(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class BasicResBlock(nn.Module):
    """Two 3x3 convolutions with a projection skip when shape changes."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(in_ch, out_ch, kernel_size=3, stride=stride, dilation=dilation, act=True)
        self.conv2 = ConvNormAct(out_ch, out_ch, kernel_size=3, stride=1, dilation=dilation, act=False)
        self.proj = None
        if stride != 1 or in_ch != out_ch:
            self.proj = ConvNormAct(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, act=False)
        self.out_act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.proj is None else self.proj(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.out_act(x + identity)


class ConvGRUCell(nn.Module):
    """ConvGRU cell with 3x3 spatial transitions."""

    def __init__(self, input_ch: int, hidden_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("Use odd kernel sizes for same-shape ConvGRU.")
        self.input_ch = int(input_ch)
        self.hidden_ch = int(hidden_ch)
        padding = kernel_size // 2

        self.gates = nn.Conv2d(
            input_ch + hidden_ch,
            2 * hidden_ch,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        self.candidate = nn.Conv2d(
            input_ch + hidden_ch,
            hidden_ch,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

        with torch.no_grad():
            self.gates.bias[:hidden_ch].fill_(1.0)

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        if h_prev is None:
            h_prev = torch.zeros(
                x.shape[0],
                self.hidden_ch,
                x.shape[2],
                x.shape[3],
                device=x.device,
                dtype=x.dtype,
            )

        combined = torch.cat([x, h_prev], dim=1)
        z_r = self.gates(combined)
        z_gate, r_gate = torch.split(z_r, self.hidden_ch, dim=1)
        z_gate = torch.sigmoid(z_gate)
        r_gate = torch.sigmoid(r_gate)

        candidate = torch.tanh(self.candidate(torch.cat([x, r_gate * h_prev], dim=1)))
        return (1.0 - z_gate) * h_prev + z_gate * candidate


class SpatialContextBlock(nn.Module):
    """Dilated residual context at 32x32 before spatial pooling."""

    def __init__(self, channels: int = POLICY_FEATURE_CHANNELS) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            BasicResBlock(channels, channels, stride=1, dilation=1),
            BasicResBlock(channels, channels, stride=1, dilation=2),
            BasicResBlock(channels, channels, stride=1, dilation=4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


def perspective_coord_grid(
    batch_size: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y_coords = torch.linspace(-1.0, 1.0, int(height), device=device, dtype=dtype)
    x_coords = torch.linspace(-1.0, 1.0, int(width), device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    bottom_weight = (yy + 1.0) * 0.5
    horizon_weight = torch.exp(-((yy + 0.15) / 0.35).square())
    grid = torch.stack((xx, yy, bottom_weight, horizon_weight), dim=0).unsqueeze(0)
    return grid.expand(int(batch_size), -1, -1, -1)


class MultiScaleSpatialFusion(nn.Module):
    def __init__(
        self,
        temporal_channels: int = POLICY_FEATURE_CHANNELS,
        high_channels: int = STAGE64_CHANNELS,
        out_channels: int = POLICY_FEATURE_CHANNELS,
    ) -> None:
        super().__init__()
        self.high_to_low = ConvNormAct(high_channels, out_channels, kernel_size=3, stride=2)
        self.global_proj = ConvNormAct(temporal_channels, out_channels, kernel_size=1, stride=1, padding=0)
        fusion_channels = temporal_channels + out_channels + out_channels + PERSPECTIVE_COORD_CHANNELS
        self.fuse = nn.Sequential(
            ConvNormAct(fusion_channels, out_channels, kernel_size=1, stride=1, padding=0),
            BasicResBlock(out_channels, out_channels, stride=1, dilation=1),
        )

    def forward(self, temporal_spatial: torch.Tensor, high_spatial: torch.Tensor) -> torch.Tensor:
        high = self.high_to_low(high_spatial)
        if high.shape[-2:] != temporal_spatial.shape[-2:]:
            high = F.interpolate(high, size=temporal_spatial.shape[-2:], mode="bilinear", align_corners=False)
        global_context = F.adaptive_avg_pool2d(temporal_spatial, 1)
        global_context = self.global_proj(global_context).expand(-1, -1, temporal_spatial.size(-2), temporal_spatial.size(-1))
        coords = perspective_coord_grid(
            temporal_spatial.size(0),
            temporal_spatial.size(-2),
            temporal_spatial.size(-1),
            device=temporal_spatial.device,
            dtype=temporal_spatial.dtype,
        )
        return self.fuse(torch.cat([temporal_spatial, high, global_context, coords], dim=1))


class SpatialGridPool(nn.Module):
    """Coordinate-aware grid pooling that preserves coarse scene layout."""

    def __init__(self, channels: int, grid_size: int = SPATIAL_GRID_SIZE) -> None:
        super().__init__()
        self.channels = int(channels)
        self.grid_size = _require_int_at_least("grid_size", grid_size, 1)
        self.pre = ConvNormAct(channels, channels, kernel_size=3, stride=1)
        coords = torch.linspace(-1.0, 1.0, self.grid_size, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("_coord_template", torch.stack((xx, yy), dim=0).unsqueeze(0), persistent=False)

    def _coord_grid(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        grid = self._coord_template.to(device=device, dtype=dtype)
        return grid.expand(batch_size, -1, -1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.pre(x)
        pooled = F.adaptive_avg_pool2d(feat, (self.grid_size, self.grid_size))
        coords = self._coord_grid(pooled.size(0), device=pooled.device, dtype=pooled.dtype)
        return torch.cat([pooled, coords], dim=1).flatten(1)


class ActionQueryDecoderLayer(nn.Module):
    def __init__(self, features: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(features)
        self.token_norm = nn.LayerNorm(features)
        self.cross_attn = nn.MultiheadAttention(
            features,
            int(heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(features)
        hidden = max(features * 2, POLICY_HEAD_HIDDEN)
        self.ffn = nn.Sequential(
            nn.Linear(features, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, features),
        )

    def forward(self, queries: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        norm_tokens = self.token_norm(tokens)
        attn_out, _ = self.cross_attn(
            self.query_norm(queries),
            norm_tokens,
            norm_tokens,
            need_weights=False,
        )
        queries = queries + attn_out
        return queries + self.ffn(self.ffn_norm(queries))


class ActionQueryDecoder(nn.Module):
    """Per-action cross-attention over coordinate-aware spatial tokens."""

    def __init__(
        self,
        channels: int,
        num_actions: int,
        *,
        grid_size: int = SPATIAL_GRID_SIZE,
        features: int = POLICY_HEAD_FEATURES,
        heads: int = ACTION_QUERY_DECODER_HEADS,
        layers: int = ACTION_QUERY_DECODER_LAYERS,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_actions = _require_int_at_least("num_actions", num_actions, 1)
        self.grid_size = _require_int_at_least("grid_size", grid_size, 1)
        self.features = _require_int_at_least("features", features, 1)
        self.heads = _require_int_at_least("heads", heads, 1)
        self.layers_count = _require_int_at_least("layers", layers, 1)
        if self.features % self.heads != 0:
            raise ValueError(f"heads must divide features, got heads={self.heads} features={self.features}.")

        self.pre = ConvNormAct(channels, channels, kernel_size=3, stride=1)
        self.token_proj = nn.Linear(channels, self.features)
        self.coord_proj = nn.Sequential(
            nn.Linear(SPATIAL_GRID_COORD_CHANNELS, self.features),
            nn.SiLU(inplace=True),
            nn.Linear(self.features, self.features),
        )
        self.token_norm = nn.LayerNorm(self.features)
        self.action_queries = nn.Parameter(torch.empty(self.num_actions, self.features))
        self.layers = nn.ModuleList(
            ActionQueryDecoderLayer(self.features, self.heads, float(dropout))
            for _ in range(self.layers_count)
        )
        self.out_norm = nn.LayerNorm(self.features)
        coords = torch.linspace(-1.0, 1.0, self.grid_size, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("_coord_template", torch.stack((xx, yy), dim=0).unsqueeze(0), persistent=False)
        nn.init.normal_(self.action_queries, mean=0.0, std=0.02)

    def _coord_grid(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        grid = self._coord_template.to(device=device, dtype=dtype)
        return grid.expand(batch_size, -1, -1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.pre(x)
        pooled = F.adaptive_avg_pool2d(feat, (self.grid_size, self.grid_size))
        coords = self._coord_grid(pooled.size(0), device=pooled.device, dtype=pooled.dtype)
        feature_tokens = pooled.flatten(2).transpose(1, 2)
        coord_tokens = coords.flatten(2).transpose(1, 2)
        tokens = self.token_norm(self.token_proj(feature_tokens) + self.coord_proj(coord_tokens))
        queries = self.action_queries.to(device=tokens.device, dtype=tokens.dtype)
        queries = queries.unsqueeze(0).expand(tokens.size(0), -1, -1).contiguous()
        for layer in self.layers:
            queries = layer(queries, tokens)
        return self.out_norm(queries)


class SharedEncoder(nn.Module):
    """
    Shared per-frame encoder.

    At the default 256x256 input size this produces 64x64 and 32x32 feature
    maps, matching the reference model's recurrent streams.
    """

    def __init__(self, in_channels: int = POLICY_INPUT_CHANNELS) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.stem = ConvNormAct(in_channels, STEM_CHANNELS, kernel_size=3, stride=2)
        self.stem_block = BasicResBlock(STEM_CHANNELS, STEM_CHANNELS, stride=1, dilation=1)

        self.stage64_down = BasicResBlock(STEM_CHANNELS, STAGE64_CHANNELS, stride=2, dilation=1)
        self.stage64_block = BasicResBlock(STAGE64_CHANNELS, STAGE64_CHANNELS, stride=1, dilation=1)

        self.stage32_down = BasicResBlock(STAGE64_CHANNELS, STAGE32_CHANNELS, stride=2, dilation=1)
        self.stage32_block = BasicResBlock(STAGE32_CHANNELS, STAGE32_CHANNELS, stride=1, dilation=2)

    def feature_stages(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.size(1) != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.size(1)}.")
        stem = self.stem_block(self.stem(x))
        s64 = self.stage64_block(self.stage64_down(stem))
        s32 = self.stage32_block(self.stage32_down(s64))
        return [stem, s64, s32]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, s64, s32 = self.feature_stages(x)
        return s64, s32


class ZeroSpatialFusion(nn.Module):
    """Compatibility module for callers that add a spatial residual before pooling."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def apply_static_masks(frames: torch.Tensor, *, clone: bool = True) -> torch.Tensor:
    """Zero out the HUD, minimap, and Roblox UI regions (relative coords)."""
    h, w = frames.shape[-2:]
    masked_frames = frames.clone() if clone else frames

    hud_y1 = int(h * 0.96)
    masked_frames[..., hud_y1:, :] = 0.0

    map_y1, map_y2 = int(h * 0.05), int(h * 0.2)
    map_x1 = int(w * 0.75)
    masked_frames[..., map_y1:map_y2, map_x1:] = 0.0

    roblox_ui_y2 = int(h * 0.1)
    roblox_ui_x2 = int(w * 0.1)
    masked_frames[..., :roblox_ui_y2, :roblox_ui_x2] = 0.0

    return masked_frames


class DrivingVideoPolicy(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.spatial_encoder = SharedEncoder(in_channels=POLICY_INPUT_CHANNELS)

        self.gru64 = ConvGRUCell(input_ch=STAGE64_CHANNELS, hidden_ch=STAGE64_CHANNELS, kernel_size=3)
        self.down64_to_32 = ConvNormAct(STAGE64_CHANNELS, STAGE64_CHANNELS, kernel_size=3, stride=2)
        self.fuse32 = ConvNormAct(STAGE32_CHANNELS + STAGE64_CHANNELS, POLICY_FEATURE_CHANNELS, kernel_size=1, stride=1, padding=0)
        self.gru32_1 = ConvGRUCell(input_ch=POLICY_FEATURE_CHANNELS, hidden_ch=POLICY_FEATURE_CHANNELS, kernel_size=3)
        self.gru32_2 = ConvGRUCell(input_ch=POLICY_FEATURE_CHANNELS, hidden_ch=POLICY_FEATURE_CHANNELS, kernel_size=3)

        self.spatial_feat_channels = STAGE32_CHANNELS
        self.feat_channels = POLICY_FEATURE_CHANNELS
        self.spatial_grid_size = SPATIAL_GRID_SIZE
        self.pooled_feat_channels = POLICY_POOLED_FEATURES
        self.temporal_spatial_fusion = (
            nn.Identity()
            if STAGE32_CHANNELS == POLICY_FEATURE_CHANNELS
            else ConvNormAct(STAGE32_CHANNELS, POLICY_FEATURE_CHANNELS, kernel_size=1, stride=1, padding=0, act=False)
        )
        self.multi_scale_fusion = MultiScaleSpatialFusion(
            temporal_channels=POLICY_FEATURE_CHANNELS,
            high_channels=STAGE64_CHANNELS,
            out_channels=POLICY_FEATURE_CHANNELS,
        )
        self.action_decoder = str(cfg.action_decoder)

        self.spatial_context = SpatialContextBlock(POLICY_FEATURE_CHANNELS)
        self.spatial_dropout = nn.Dropout2d(float(cfg.spatial_dropout))
        self.head_dropout = nn.Dropout(float(cfg.head_dropout))
        if self.action_decoder == ACTION_DECODER_ACTION_QUERY:
            self.action_query_decoder = ActionQueryDecoder(
                POLICY_FEATURE_CHANNELS,
                self.cfg.num_bin,
                grid_size=SPATIAL_GRID_SIZE,
                features=POLICY_HEAD_FEATURES,
                heads=int(self.cfg.action_query_heads),
                layers=int(self.cfg.action_query_layers),
                dropout=float(cfg.head_dropout),
            )
            self.pooled_feat_channels = self.cfg.num_bin * POLICY_HEAD_FEATURES
        else:
            self.grid_pool = SpatialGridPool(POLICY_FEATURE_CHANNELS, grid_size=SPATIAL_GRID_SIZE)
            self.fc1 = nn.Linear(POLICY_POOLED_FEATURES, POLICY_HEAD_HIDDEN)
            self.fc2 = nn.Linear(POLICY_HEAD_HIDDEN, POLICY_HEAD_FEATURES)

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

        fusion_in = POLICY_HEAD_FEATURES + self.cfg.d_model if cfg.last_action_conditioning else POLICY_HEAD_FEATURES
        self.head_fusion = (
            nn.Sequential(
                nn.LayerNorm(fusion_in),
                nn.Linear(fusion_in, POLICY_HEAD_FEATURES),
                nn.ELU(inplace=True),
            )
            if cfg.last_action_conditioning
            else nn.Identity()
        )
        if self.action_decoder == ACTION_DECODER_ACTION_QUERY:
            self.action_query_head = nn.Linear(POLICY_HEAD_FEATURES, 1)
            nn.init.constant_(self.action_query_head.bias, -1.0)
        else:
            self.button_head = nn.Linear(POLICY_HEAD_FEATURES, self.cfg.num_bin)
            nn.init.constant_(self.button_head.bias, -1.0)

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
    ) -> torch.Tensor:
        high_height = int(high_height) if high_height is not None else int(height) * 2
        high_width = int(high_width) if high_width is not None else int(width) * 2
        return torch.zeros(
            TEMPORAL_STATE_LAYERS,
            batch_size,
            PACKED_STATE_CHANNELS,
            high_height,
            high_width,
            device=device,
            dtype=dtype,
        )

    def _zoneout_state(self, candidate: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        prob = float(self.cfg.zoneout)
        if prob <= 0.0 or not self.training:
            return candidate
        if candidate.shape != previous.shape:
            return candidate
        keep = torch.empty_like(candidate).bernoulli_(1.0 - prob)
        return candidate * keep + previous * (1.0 - keep)

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
        if hidden.dim() == 4 and hidden.size(0) == TEMPORAL_STATE_LAYERS and batch_size == 1:
            hidden = hidden.unsqueeze(1)
        if hidden.dim() == 4 and hidden.size(0) == batch_size:
            return self._pack_legacy_feature_state(
                hidden,
                batch_size,
                int(height),
                int(width),
                high_height,
                high_width,
                device=device,
                dtype=dtype,
            )
        if hidden.dim() != 5:
            raise ValueError(f"Expected temporal hidden state [L,B,C,H,W], got {tuple(hidden.shape)}.")
        if hidden.size(0) != TEMPORAL_STATE_LAYERS:
            raise ValueError(f"Expected {TEMPORAL_STATE_LAYERS} temporal layers, got {hidden.size(0)}.")
        if hidden.size(1) != batch_size:
            raise ValueError(f"Expected temporal hidden batch size {batch_size}, got {hidden.size(1)}.")
        if hidden.size(2) != PACKED_STATE_CHANNELS:
            raise ValueError(
                f"Expected packed temporal state channels {PACKED_STATE_CHANNELS}, got {hidden.size(2)}."
            )
        if hidden.size(3) < high_height or hidden.size(4) < high_width:
            raise ValueError(
                "Packed temporal state is too small for the current feature maps: "
                f"state={tuple(hidden.shape)} required_spatial=({high_height}, {high_width})."
            )
        if hidden.size(3) != high_height or hidden.size(4) != high_width:
            hidden = hidden[:, :, :, :high_height, :high_width].contiguous()
        return hidden

    def _pack_legacy_feature_state(
        self,
        hidden: torch.Tensor,
        batch_size: int,
        height: int,
        width: int,
        high_height: int,
        high_width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        packed = self._initial_temporal_state(
            batch_size,
            height,
            width,
            device=device,
            dtype=dtype,
            high_height=high_height,
            high_width=high_width,
        )
        if tuple(hidden.shape[1:]) == (POLICY_FEATURE_CHANNELS, height, width):
            packed[1, :, :, :height, :width] = hidden
            packed[2, :, :, :height, :width] = hidden
            return packed
        if tuple(hidden.shape[1:]) == (STAGE64_CHANNELS, high_height, high_width):
            packed[0, :, :STAGE64_CHANNELS, :high_height, :high_width] = hidden
            return packed
        raise ValueError(
            "Expected legacy temporal hidden state shape "
            f"{(batch_size, POLICY_FEATURE_CHANNELS, height, width)} or "
            f"{(batch_size, STAGE64_CHANNELS, high_height, high_width)}, got {tuple(hidden.shape)}."
        )

    def _pack_temporal_state(
        self,
        h64: torch.Tensor,
        h32_1: torch.Tensor,
        h32_2: torch.Tensor,
    ) -> torch.Tensor:
        b, _, h64_h, h64_w = h64.shape
        _, _, h32_h, h32_w = h32_1.shape
        packed = h64.new_zeros(TEMPORAL_STATE_LAYERS, b, PACKED_STATE_CHANNELS, h64_h, h64_w)
        packed[0, :, :STAGE64_CHANNELS, :h64_h, :h64_w] = h64
        packed[1, :, :POLICY_FEATURE_CHANNELS, :h32_h, :h32_w] = h32_1
        packed[2, :, :POLICY_FEATURE_CHANNELS, :h32_h, :h32_w] = h32_2
        return packed

    def _unpack_temporal_state(
        self,
        hidden_state: torch.Tensor,
        batch_size: int,
        high_height: int,
        high_width: int,
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = hidden_state
        if hidden.dim() != 5:
            raise ValueError(f"Expected temporal hidden state [L,B,C,H,W], got {tuple(hidden.shape)}.")
        if (
            hidden.size(0) != TEMPORAL_STATE_LAYERS
            or hidden.size(1) != batch_size
            or hidden.size(2) != PACKED_STATE_CHANNELS
            or hidden.size(3) < high_height
            or hidden.size(4) < high_width
        ):
            raise ValueError(
                "Unexpected packed temporal hidden state shape "
                f"{tuple(hidden.shape)} for batch={batch_size}, high=({high_height}, {high_width})."
            )
        h64 = hidden[0, :, :STAGE64_CHANNELS, :high_height, :high_width]
        h32_1 = hidden[1, :, :POLICY_FEATURE_CHANNELS, :height, :width]
        h32_2 = hidden[2, :, :POLICY_FEATURE_CHANNELS, :height, :width]
        return h64.contiguous(), h32_1.contiguous(), h32_2.contiguous()

    def _recurrent_step(
        self,
        spatial_step: torch.Tensor,
        h64_prev: torch.Tensor,
        h32_1_prev: torch.Tensor,
        h32_2_prev: torch.Tensor,
        s64_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = int(spatial_step.shape[0])
        if s64_t is None:
            high_height = int(h64_prev.size(-2))
            high_width = int(h64_prev.size(-1))
            s64_t = spatial_step.new_zeros(b, STAGE64_CHANNELS, high_height, high_width)

        h64 = self._zoneout_state(self.gru64(s64_t, h64_prev), h64_prev)
        h64_down = self.down64_to_32(h64)
        if h64_down.shape[-2:] != spatial_step.shape[-2:]:
            h64_down = F.interpolate(h64_down, size=spatial_step.shape[-2:], mode="bilinear", align_corners=False)
        fused32 = self.fuse32(torch.cat([spatial_step, h64_down], dim=1))
        h32_1 = self._zoneout_state(self.gru32_1(fused32, h32_1_prev), h32_1_prev)
        h32_2 = self._zoneout_state(self.gru32_2(h32_1, h32_2_prev), h32_2_prev)
        return h64, h32_1, h32_2

    def _context_features(self, h32_2: torch.Tensor) -> torch.Tensor:
        return self.spatial_dropout(self.spatial_context(h32_2))

    def _fuse_current_spatial(self, temporal_feat: torch.Tensor, spatial_step: torch.Tensor) -> torch.Tensor:
        return temporal_feat + self.temporal_spatial_fusion(spatial_step)

    def _head_spatial_features(
        self,
        temporal_feat: torch.Tensor,
        spatial_step: torch.Tensor,
        s64_t: torch.Tensor,
    ) -> torch.Tensor:
        return self.multi_scale_fusion(self._fuse_current_spatial(temporal_feat, spatial_step), s64_t)

    def _temporal_step(
        self,
        spatial_step: torch.Tensor,
        hidden_state: torch.Tensor,
        s64_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _, h32, w32 = spatial_step.shape
        if s64_t is None:
            high_height = int(hidden_state.size(-2)) if hidden_state.dim() == 5 else h32 * 2
            high_width = int(hidden_state.size(-1)) if hidden_state.dim() == 5 else w32 * 2
            s64_for_head = spatial_step.new_zeros(b, STAGE64_CHANNELS, high_height, high_width)
        else:
            high_height, high_width = s64_t.shape[-2:]
            s64_for_head = s64_t
        h64_prev, h32_1_prev, h32_2_prev = self._unpack_temporal_state(
            hidden_state,
            b,
            int(high_height),
            int(high_width),
            int(h32),
            int(w32),
        )
        h64, h32_1, h32_2 = self._recurrent_step(
            spatial_step,
            h64_prev,
            h32_1_prev,
            h32_2_prev,
            s64_t=s64_for_head,
        )
        context = self._context_features(self._head_spatial_features(h32_2, spatial_step, s64_for_head))
        return context, self._pack_temporal_state(h64, h32_1, h32_2)

    def _pool_features(self, fused: torch.Tensor) -> torch.Tensor:
        if self.action_decoder == ACTION_DECODER_ACTION_QUERY:
            return self.action_query_decoder(fused)
        pooled = self.grid_pool(fused)
        z = F.silu(self.fc1(pooled))
        z = self.head_dropout(z)
        return F.silu(self.fc2(z))

    def _button_logits(self, features: torch.Tensor) -> torch.Tensor:
        if features.size(-1) != POLICY_HEAD_FEATURES:
            raise ValueError(f"Expected final feature dim {POLICY_HEAD_FEATURES}, got {features.size(-1)}.")
        if features.dim() == 4:
            if features.size(2) != self.cfg.num_bin:
                raise ValueError(f"Expected {self.cfg.num_bin} action features, got {features.size(2)}.")
            return self.action_query_head(features).squeeze(-1)
        if features.dim() != 3:
            raise ValueError(f"Expected features with shape [B,T,F] or [B,T,A,F], got {tuple(features.shape)}.")
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
        if self.last_action_encoder is None:
            raise RuntimeError("last_action_conditioning is disabled for this policy.")
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

        action_values = action_values.reshape(batch_size * time_steps, self.cfg.num_bin)
        features = self.last_action_encoder(action_values).reshape(batch_size, time_steps, self.cfg.d_model)
        return features.to(dtype=dtype)

    def _apply_masks(self, frames: torch.Tensor, *, clone: bool = True) -> torch.Tensor:
        return apply_static_masks(frames, clone=clone)

    def encode_sequence(self, model_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, c, h, w = model_input.shape
        frames = model_input.reshape(b * t, c, h, w)
        if frames.is_cuda:
            frames = frames.contiguous(memory_format=torch.channels_last)
        s64, s32 = self.spatial_encoder(frames)
        s64 = s64.reshape(b, t, STAGE64_CHANNELS, s64.shape[-2], s64.shape[-1])
        s32 = s32.reshape(b, t, POLICY_FEATURE_CHANNELS, s32.shape[-2], s32.shape[-1])
        return s64, s32

    def _features_to_logits(
        self,
        features: torch.Tensor,
        prev_action: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if features.dim() not in (3, 4):
            raise ValueError(f"Expected features with shape [B,T,F] or [B,T,A,F], got {tuple(features.shape)}.")
        b, t = int(features.shape[0]), int(features.shape[1])
        if self.last_action_encoder is not None:
            action_feat = self._last_action_features(prev_action, b, t, device=features.device, dtype=features.dtype)
            if features.dim() == 4:
                action_feat = action_feat.unsqueeze(2).expand(b, t, features.size(2), self.cfg.d_model)
            features = self.head_fusion(torch.cat([features, action_feat], dim=-1))
        else:
            features = self.head_fusion(features)
        return self._button_logits(features)

    def _sequence_visual_features(
        self,
        frames: torch.Tensor,
        state: Optional[TemporalState],
    ) -> Tuple[torch.Tensor, TemporalState]:
        if frames.dim() != 5:
            raise ValueError(f"Expected RGB frames with shape [B,T,{RGB_INPUT_CHANNELS},H,W], got {tuple(frames.shape)}.")
        b, _, c, _, _ = frames.shape
        if c != RGB_INPUT_CHANNELS:
            raise ValueError(f"Expected RGB frames with shape [B,T,{RGB_INPUT_CHANNELS},H,W], got {tuple(frames.shape)}.")

        frames = self._normalize_frames(frames)
        frames = self._apply_masks(frames, clone=bool(frames.requires_grad))
        s64_seq, s32_seq = self.encode_sequence(frames)

        high_height = int(s64_seq.shape[-2])
        high_width = int(s64_seq.shape[-1])
        low_height = int(s32_seq.shape[-2])
        low_width = int(s32_seq.shape[-1])
        h_t = self._prepare_temporal_state(
            state,
            b,
            low_height,
            low_width,
            device=frames.device,
            dtype=s32_seq.dtype,
            high_height=high_height,
            high_width=high_width,
        )
        h64_t, h32_1_t, h32_2_t = self._unpack_temporal_state(
            h_t,
            b,
            high_height,
            high_width,
            low_height,
            low_width,
        )

        recurrent_steps = []
        for step in range(s32_seq.size(1)):
            h64_t, h32_1_t, h32_2_t = self._recurrent_step(
                s32_seq[:, step],
                h64_t,
                h32_1_t,
                h32_2_t,
                s64_t=s64_seq[:, step],
            )
            recurrent_steps.append(h32_2_t)

        temporal_seq = torch.stack(recurrent_steps, dim=1)
        t_steps = int(temporal_seq.size(1))
        flat_temporal = temporal_seq.reshape(
            b * t_steps,
            POLICY_FEATURE_CHANNELS,
            temporal_seq.size(-2),
            temporal_seq.size(-1),
        )
        flat_spatial = s32_seq.reshape(
            b * t_steps,
            POLICY_FEATURE_CHANNELS,
            s32_seq.size(-2),
            s32_seq.size(-1),
        )
        flat_high = s64_seq.reshape(
            b * t_steps,
            STAGE64_CHANNELS,
            s64_seq.size(-2),
            s64_seq.size(-1),
        )
        flat_head = self._head_spatial_features(flat_temporal, flat_spatial, flat_high)
        flat_context = self._context_features(flat_head)
        visual_flat = self._pool_features(flat_context)
        if self.action_decoder == ACTION_DECODER_ACTION_QUERY:
            visual_feat = visual_flat.reshape(b, t_steps, self.cfg.num_bin, POLICY_HEAD_FEATURES)
        else:
            visual_feat = visual_flat.reshape(b, t_steps, POLICY_HEAD_FEATURES)
        next_hidden = self._pack_temporal_state(h64_t, h32_1_t, h32_2_t)
        return visual_feat, TemporalState(hidden_state=next_hidden.detach())

    def _teacher_action_sequence(
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

        action_values = prev_action.to(device=device, dtype=dtype)
        if action_values.dim() == 2:
            if tuple(action_values.shape) != (batch_size, self.cfg.num_bin):
                raise ValueError(
                    f"Expected prev_action shape {(batch_size, self.cfg.num_bin)} or "
                    f"{(batch_size, time_steps, self.cfg.num_bin)}, got {tuple(action_values.shape)}."
                )
            return action_values.view(batch_size, 1, self.cfg.num_bin).expand(batch_size, time_steps, self.cfg.num_bin)

        if action_values.dim() == 3:
            if tuple(action_values.shape) != (batch_size, time_steps, self.cfg.num_bin):
                raise ValueError(
                    f"Expected prev_action shape {(batch_size, time_steps, self.cfg.num_bin)}, "
                    f"got {tuple(action_values.shape)}."
                )
            return action_values

        raise ValueError(f"Expected prev_action with 2 or 3 dims, got {tuple(action_values.shape)}.")

    def _prediction_as_prev_action(
        self,
        logits: torch.Tensor,
        thresholds: Optional[torch.Tensor],
        *,
        soft_feedback: bool,
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits.float()).to(dtype=logits.dtype)
        if soft_feedback:
            return probs.detach()
        if thresholds is None:
            threshold_values = torch.full(
                (1, self.cfg.num_bin),
                float(self.cfg.button_state_threshold),
                device=logits.device,
                dtype=probs.dtype,
            )
        else:
            threshold_values = thresholds.to(device=logits.device, dtype=probs.dtype).view(1, self.cfg.num_bin)
        return (probs >= threshold_values).to(dtype=logits.dtype).detach()

    def forward_with_action_feedback(
        self,
        frames: torch.Tensor,
        state: Optional[TemporalState] = None,
        return_aux: bool = False,
        prev_action: Optional[torch.Tensor] = None,
        feedback_mask: Optional[torch.Tensor] = None,
        feedback_thresholds: Optional[torch.Tensor] = None,
        soft_feedback: bool = False,
    ):
        visual_feat, next_state = self._sequence_visual_features(frames, state)
        logits = self._features_to_logits_with_feedback(
            visual_feat,
            prev_action,
            feedback_mask,
            feedback_thresholds,
            soft_feedback=soft_feedback,
        )
        output = PolicyOutput(button_logits=logits)
        if return_aux:
            return output, next_state
        return output

    def _features_to_logits_with_feedback(
        self,
        visual_feat: torch.Tensor,
        prev_action: Optional[torch.Tensor],
        feedback_mask: Optional[torch.Tensor],
        feedback_thresholds: Optional[torch.Tensor],
        *,
        soft_feedback: bool,
    ) -> torch.Tensor:
        if self.last_action_encoder is None or feedback_mask is None:
            return self._features_to_logits(visual_feat, prev_action)
        if visual_feat.dim() not in (3, 4):
            raise ValueError(
                f"Expected visual features with shape [B,T,F] or [B,T,A,F], got {tuple(visual_feat.shape)}."
            )
        b, t = int(visual_feat.shape[0]), int(visual_feat.shape[1])
        teacher = self._teacher_action_sequence(
            prev_action,
            b,
            t,
            device=visual_feat.device,
            dtype=visual_feat.dtype,
        )
        mask = feedback_mask.to(device=visual_feat.device, dtype=torch.bool)
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        if mask.dim() != 3 or tuple(mask.shape[:2]) != (b, t) or mask.size(-1) not in (1, self.cfg.num_bin):
            raise ValueError(
                "feedback_mask must have shape [B,T], [B,T,1], or [B,T,num_bin], "
                f"got {tuple(mask.shape)}."
            )

        logits_steps = []
        feedback_action = teacher[:, 0]
        for step in range(t):
            if step == 0:
                action_input = teacher[:, step]
            else:
                action_input = torch.where(mask[:, step], feedback_action, teacher[:, step])
            logits = self._features_to_logits(visual_feat[:, step : step + 1], action_input).reshape(b, self.cfg.num_bin)
            logits_steps.append(logits)
            feedback_action = self._prediction_as_prev_action(
                logits,
                feedback_thresholds,
                soft_feedback=bool(soft_feedback),
            )

        return torch.stack(logits_steps, dim=1)

    def forward(
        self,
        frames: torch.Tensor,
        state: Optional[TemporalState] = None,
        return_aux: bool = False,
        prev_action: Optional[torch.Tensor] = None,
        feedback_mask: Optional[torch.Tensor] = None,
        feedback_thresholds: Optional[torch.Tensor] = None,
        soft_feedback: bool = False,
    ):
        visual_feat, next_state = self._sequence_visual_features(frames, state)
        logits = self._features_to_logits_with_feedback(
            visual_feat,
            prev_action,
            feedback_mask,
            feedback_thresholds,
            soft_feedback=soft_feedback,
        )
        output = PolicyOutput(button_logits=logits)
        if return_aux:
            return output, next_state
        return output

    def forward_step(
        self,
        frame: torch.Tensor,
        state: TemporalState,
        prev_action: Optional[torch.Tensor] = None,
    ):
        if frame.dim() != 4 or frame.size(1) != RGB_INPUT_CHANNELS:
            raise ValueError(f"Expected RGB frame with shape [B,{RGB_INPUT_CHANNELS},H,W], got {tuple(frame.shape)}.")
        b = frame.shape[0]

        frame_norm = self._normalize_frames(frame)
        masked_frame = self._apply_masks(frame_norm, clone=bool(frame_norm.requires_grad))
        if masked_frame.is_cuda:
            masked_frame = masked_frame.contiguous(memory_format=torch.channels_last)

        s64_t, s32_t = self.spatial_encoder(masked_frame)
        h_t = self._prepare_temporal_state(
            state,
            b,
            s32_t.shape[-2],
            s32_t.shape[-1],
            device=frame.device,
            dtype=s32_t.dtype,
            high_height=s64_t.shape[-2],
            high_width=s64_t.shape[-1],
        )
        temporal_feat, new_hidden = self._temporal_step(s32_t, h_t, s64_t=s64_t)
        visual_flat = self._pool_features(temporal_feat)
        if self.action_decoder == ACTION_DECODER_ACTION_QUERY:
            visual_feat = visual_flat.reshape(b, 1, self.cfg.num_bin, POLICY_HEAD_FEATURES)
        else:
            visual_feat = visual_flat.reshape(b, 1, POLICY_HEAD_FEATURES)
        squeezed = PolicyOutput(button_logits=self._features_to_logits(visual_feat, prev_action).reshape(b, self.cfg.num_bin))
        new_state = TemporalState(hidden_state=new_hidden.detach())
        return squeezed, new_state
