"""CNN grid-token causal transformer driving policy.

The public classes in this file are kept compatible with the existing trainer
and runtime entrypoints, while the learned architecture is a per-frame CNN
encoder plus a causal temporal transformer over visual grid tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from action_space import game_data_root, get_key_names, normalize_game_name, selected_game


RGB_CHANNELS = 3
STEM_CHANNELS = 32
LOW_CHANNELS = 48
MID_CHANNELS = 96
DEEP_CHANNELS = 192
FUSED_CHANNELS = 128
READOUT_CHANNELS = 256
DEFAULT_MODEL_SIZE = 256
FUSED_SPATIAL_SIZE = 128
DEFAULT_CONTEXT_LENGTH = 40
DEFAULT_SEQUENCE_LENGTH = 40
DEFAULT_ACTION_NAMES = ("w", "a", "s", "d", "z", "c")
TOKEN_GRID_SIZE = 8
NUM_VISUAL_TOKENS = TOKEN_GRID_SIZE * TOKEN_GRID_SIZE
TOKENS_PER_STEP = NUM_VISUAL_TOKENS
TEMPORAL_HEADS = 8
TEMPORAL_LAYERS = 6
ARCHITECTURE_VERSION = "cnn_grid_causal_transformer_v11_ctx40_256_fusion128_onset_offset_heads"


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


@dataclass
class ModelConfig:
    """Model and data contract shared by training and online inference."""

    selected_game: str = selected_game
    data_root: Optional[str] = None
    video_ext: str = ".mp4"
    csv_ext: str = ".csv"

    model_size: int = DEFAULT_MODEL_SIZE
    seq_len: int = DEFAULT_SEQUENCE_LENGTH
    train_seq_stride: int = DEFAULT_SEQUENCE_LENGTH
    val_seq_stride: int = DEFAULT_SEQUENCE_LENGTH
    action_offset: int = 1
    prediction_horizon: int = 1
    prediction_horizon_offsets: Optional[Sequence[int]] = None
    sequence_output_tail_frames: int = 0

    key_names: Optional[list[str]] = None
    mouse_button_names: Optional[list[str]] = None
    num_bin: int = 0

    spatial_dropout: float = 0.10
    head_dropout: float = 0.10
    zoneout: float = 0.0
    attention_temperature: float = 1.0

    button_state_threshold: float = 0.5
    button_state_thresholds: Optional[Sequence[float]] = None
    press_threshold: float = 0.5
    press_thresholds: Optional[Sequence[float]] = None
    release_threshold: float = 0.5
    release_thresholds: Optional[Sequence[float]] = None
    architecture_version: str = ARCHITECTURE_VERSION

    # Compatibility fields retained for existing configs and checkpoint tools.
    d_model: int = READOUT_CHANNELS
    action_decoder: str = "causal_transformer"
    action_query_heads: int = TEMPORAL_HEADS
    action_query_layers: int = 1
    last_action_conditioning: bool = False
    last_action_fusion: str = "bounded_visual_residual"
    last_action_residual_cap: float = 0.5
    last_action_prior_logit: float = 0.0
    last_action_absence_prior_logit: float = 0.0
    use_activation_checkpointing: bool = True

    def __post_init__(self) -> None:
        self.selected_game = normalize_game_name(self.selected_game)
        if self.data_root is None:
            self.data_root = game_data_root(self.selected_game)

        self.model_size = _as_int("model_size", self.model_size, 1)
        if self.model_size != DEFAULT_MODEL_SIZE:
            raise ValueError(f"model_size must be {DEFAULT_MODEL_SIZE}, got {self.model_size}.")
        self.seq_len = _as_int("seq_len", self.seq_len, 1)
        if self.seq_len > DEFAULT_CONTEXT_LENGTH:
            raise ValueError(f"seq_len must be <= {DEFAULT_CONTEXT_LENGTH}, got {self.seq_len}.")
        self.train_seq_stride = _as_int("train_seq_stride", self.train_seq_stride, 1)
        self.val_seq_stride = _as_int("val_seq_stride", self.val_seq_stride, 1)
        self.sequence_output_tail_frames = _as_int(
            "sequence_output_tail_frames", self.sequence_output_tail_frames, 0
        )
        self.action_offset = _as_int("action_offset", self.action_offset, 1)
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
            raise ValueError("This six-action policy does not support mouse-button outputs.")
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

        self.press_threshold = _as_float("press_threshold", self.press_threshold, 0.0, 1.0)
        if self.press_thresholds is None:
            self.press_thresholds = tuple(float(self.press_threshold) for _ in range(self.num_bin))
        else:
            thresholds = tuple(float(item) for item in self.press_thresholds)
            if len(thresholds) != self.num_bin:
                raise ValueError(
                    f"press_thresholds must contain {self.num_bin} values, got {len(thresholds)}."
                )
            for idx, threshold in enumerate(thresholds):
                _as_float(f"press_thresholds[{idx}]", threshold, 0.0, 1.0)
            self.press_thresholds = thresholds

        self.release_threshold = _as_float("release_threshold", self.release_threshold, 0.0, 1.0)
        if self.release_thresholds is None:
            self.release_thresholds = tuple(float(self.release_threshold) for _ in range(self.num_bin))
        else:
            thresholds = tuple(float(item) for item in self.release_thresholds)
            if len(thresholds) != self.num_bin:
                raise ValueError(
                    f"release_thresholds must contain {self.num_bin} values, got {len(thresholds)}."
                )
            for idx, threshold in enumerate(thresholds):
                _as_float(f"release_thresholds[{idx}]", threshold, 0.0, 1.0)
            self.release_thresholds = thresholds

        self.d_model = _as_int("d_model", self.d_model, 1)
        if self.d_model != READOUT_CHANNELS:
            raise ValueError(f"d_model must be {READOUT_CHANNELS}, got {self.d_model}.")
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
        self.use_activation_checkpointing = bool(self.use_activation_checkpointing)
        if str(self.architecture_version) != ARCHITECTURE_VERSION:
            raise ValueError(
                f"Expected architecture_version={ARCHITECTURE_VERSION!r}, "
                f"got {self.architecture_version!r}."
            )


@dataclass
class PolicyOutput:
    """Final-frame logits, with optional dense causal logits."""

    button_logits: torch.Tensor
    sequence_button_logits: Optional[torch.Tensor] = None
    vision_button_logits: Optional[torch.Tensor] = None
    sequence_vision_button_logits: Optional[torch.Tensor] = None
    onset_logits: Optional[torch.Tensor] = None
    sequence_onset_logits: Optional[torch.Tensor] = None
    offset_logits: Optional[torch.Tensor] = None
    sequence_offset_logits: Optional[torch.Tensor] = None
    change_logits: Optional[torch.Tensor] = None
    sequence_change_logits: Optional[torch.Tensor] = None
    next_feedback_action: Optional[torch.Tensor] = None


@dataclass
class TemporalState:
    """Rolling transformer context for online inference.

    visual_tokens is the fast streaming cache. frames and hidden_state are
    retained so older call sites can still construct the dataclass.
    """

    frames: Optional[torch.Tensor] = None
    prev_actions: Optional[torch.Tensor] = None
    hidden_state: Optional[torch.Tensor] = None
    visual_tokens: Optional[torch.Tensor] = None


class BlurPool2d(nn.Module):
    """Anti-aliased downsampling."""

    def __init__(self, channels: int, stride: int = 2):
        super().__init__()
        k = torch.tensor([1.0, 2.0, 1.0])
        filt = k[:, None] * k[None, :]
        filt = filt / filt.sum()
        self.register_buffer("filt", filt[None, None].repeat(channels, 1, 1, 1))
        self.channels = int(channels)
        self.stride = int(stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.filt, stride=self.stride, padding=1, groups=self.channels)


class ConvGNAct(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p, bias=False)
        num_groups = max(1, min(8, cout))
        while cout % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups, cout)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ConvNeXtBlock2D(nn.Module):
    """Lightweight ConvNeXt-style 2-D block."""

    def __init__(self, channels: int, expansion: int = 4, drop: float = 0.0):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = nn.GroupNorm(1, channels)
        self.pw1 = nn.Conv2d(channels, channels * expansion, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(channels * expansion, channels, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.norm(y)
        y = self.pw1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.pw2(y)
        return x + y


class DownStage(nn.Module):
    """Conv projection, blur-pool downsample, then local feature blocks."""

    def __init__(self, cin: int, cout: int, n_blocks: int):
        super().__init__()
        self.pre = ConvGNAct(cin, cout, 3, 1, 1)
        self.down = BlurPool2d(cout, stride=2)
        self.blocks = nn.Sequential(*[ConvNeXtBlock2D(cout) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        x = self.down(x)
        return self.blocks(x)


class SpatialGridTokenizer(nn.Module):
    """Convert each fused 128x128 feature map into deterministic 8x8 grid tokens."""

    def __init__(
        self,
        cin: int = FUSED_CHANNELS,
        d_model: int = READOUT_CHANNELS,
        num_tokens: int = NUM_VISUAL_TOKENS,
    ):
        super().__init__()
        grid = int(round(math.sqrt(int(num_tokens))))
        if grid * grid != int(num_tokens):
            raise ValueError(f"num_tokens must be a square grid, got {num_tokens}.")
        self.grid_size = grid
        self.num_tokens = int(num_tokens)
        self.proj = nn.Conv2d(cin, d_model, kernel_size=1)
        self.token_mix = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=1),
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        if fmap.dim() != 4:
            raise ValueError(f"Expected feature map [B,C,H,W], got {tuple(fmap.shape)}.")
        x = self.proj(fmap)
        pooled = torch.cat(
            [
                F.adaptive_avg_pool2d(x, (self.grid_size, self.grid_size)),
                F.adaptive_max_pool2d(x, (self.grid_size, self.grid_size)),
            ],
            dim=1,
        )
        tokens = self.token_mix(pooled).flatten(2).transpose(1, 2)
        return self.ln(tokens)


class CausalTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int = READOUT_CHANNELS,
        n_heads: int = TEMPORAL_HEADS,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        y_in = self.ln1(x)
        y, _ = self.attn(y_in, y_in, y_in, attn_mask=attn_mask, need_weights=False)
        x = x + y
        return x + self.mlp(self.ln2(x))


class FrameEncoder(nn.Module):
    """High-resolution per-frame CNN front-end with 128x128 multi-scale fusion."""

    def __init__(self, d_model: int = READOUT_CHANNELS, num_tokens: int = NUM_VISUAL_TOKENS):
        super().__init__()
        self.stem = nn.Sequential(
            ConvGNAct(RGB_CHANNELS, STEM_CHANNELS, 3, 1, 1),
            ConvGNAct(STEM_CHANNELS, STEM_CHANNELS, 3, 1, 1),
        )
        self.stage1 = DownStage(STEM_CHANNELS, LOW_CHANNELS, n_blocks=2)
        self.stage2 = DownStage(LOW_CHANNELS, MID_CHANNELS, n_blocks=2)
        self.stage3 = DownStage(MID_CHANNELS, DEEP_CHANNELS, n_blocks=4)

        self.p0 = nn.Sequential(
            nn.Conv2d(STEM_CHANNELS, 64, kernel_size=1),
            ConvGNAct(64, 64, 3, 2, 1),
        )
        self.p1 = nn.Conv2d(LOW_CHANNELS, 64, kernel_size=1)
        self.p2 = nn.Conv2d(MID_CHANNELS, 64, kernel_size=1)
        self.p3 = nn.Conv2d(DEEP_CHANNELS, 64, kernel_size=1)
        self.fuse = nn.Sequential(
            ConvGNAct(64 * 4, FUSED_CHANNELS, 3, 1, 1),
            ConvNeXtBlock2D(FUSED_CHANNELS),
        )
        self.tokenizer = SpatialGridTokenizer(
            cin=FUSED_CHANNELS,
            d_model=d_model,
            num_tokens=num_tokens,
        )

    def encode_feature_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 4 or x.size(1) != RGB_CHANNELS:
            raise ValueError(f"Expected RGB images [B,3,H,W], got {tuple(x.shape)}.")
        if x.size(-2) != DEFAULT_MODEL_SIZE or x.size(-1) != DEFAULT_MODEL_SIZE:
            raise ValueError(
                f"Expected {DEFAULT_MODEL_SIZE}x{DEFAULT_MODEL_SIZE} frames, got {tuple(x.shape[-2:])}."
            )
        x = self.stem(x)
        b1 = self.stage1(x)
        b2 = self.stage2(b1)
        b3 = self.stage3(b2)

        p0 = self.p0(x)
        p1 = self.p1(b1)
        target_size = p1.shape[-2:]
        p2 = F.interpolate(self.p2(b2), size=target_size, mode="bilinear", align_corners=False)
        p3 = F.interpolate(self.p3(b3), size=target_size, mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat([p0, p1, p2, p3], dim=1))
        return x, b1, b2, b3, fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, _, _, fused = self.encode_feature_maps(x)
        return self.tokenizer(fused)


class GreenvilleBCFormer(nn.Module):
    """End-to-end causal behavioral cloning transformer."""

    def __init__(
        self,
        context_len: int = DEFAULT_CONTEXT_LENGTH,
        num_visual_tokens: int = NUM_VISUAL_TOKENS,
        d_model: int = READOUT_CHANNELS,
        n_heads: int = TEMPORAL_HEADS,
        n_layers: int = TEMPORAL_LAYERS,
        dropout: float = 0.1,
        last_action_conditioning: bool = False,
        last_action_residual_cap: float = 0.5,
        use_activation_checkpointing: bool = True,
    ):
        super().__init__()
        self.context_len = int(context_len)
        self.num_visual_tokens = int(num_visual_tokens)
        self.tokens_per_step = self.num_visual_tokens
        self.d_model = int(d_model)
        self.last_action_conditioning = bool(last_action_conditioning)
        self.last_action_residual_cap = float(last_action_residual_cap)
        self.use_activation_checkpointing = bool(use_activation_checkpointing)

        self.frame_encoder = FrameEncoder(d_model=d_model, num_tokens=num_visual_tokens)
        self.temporal_pos = nn.Embedding(self.context_len, d_model)
        self.spatial_token_pos = nn.Parameter(torch.zeros(1, 1, self.num_visual_tokens, d_model))
        self.blocks = nn.ModuleList(
            [
                CausalTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    mlp_ratio=4,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.readout_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.readout_token_ln = nn.LayerNorm(d_model)
        self.readout_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.readout_state_ln = nn.LayerNorm(d_model)
        self.readout_ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.vision_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, len(DEFAULT_ACTION_NAMES)),
        )
        self.onset_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, len(DEFAULT_ACTION_NAMES)),
        )
        self.offset_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, len(DEFAULT_ACTION_NAMES)),
        )
        self.action_residual_mlp = nn.Sequential(
            nn.Linear(len(DEFAULT_ACTION_NAMES), 64),
            nn.GELU(),
            nn.Linear(64, len(DEFAULT_ACTION_NAMES)),
        )
        nn.init.trunc_normal_(self.spatial_token_pos, std=0.02)
        nn.init.zeros_(self.action_residual_mlp[-1].weight)
        nn.init.zeros_(self.action_residual_mlp[-1].bias)

    def _frame_causal_mask(self, steps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        frame_ids = torch.arange(int(steps), device=device).repeat_interleave(self.tokens_per_step)
        future = frame_ids.view(1, -1) > frame_ids.view(-1, 1)
        mask = torch.zeros((frame_ids.numel(), frame_ids.numel()), device=device, dtype=dtype)
        return mask.masked_fill(future, float("-inf"))

    def encode_visual_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dim() != 5 or frames.size(2) != RGB_CHANNELS:
            raise ValueError(f"Expected frames [B,T,3,H,W], got {tuple(frames.shape)}.")
        batch, steps, channels, height, width = frames.shape
        if steps <= 0:
            raise ValueError("The frame sequence must contain at least one frame.")
        if steps > self.context_len:
            raise ValueError(f"Sequence length {steps} exceeds context length {self.context_len}.")
        flat = frames.reshape(batch * steps, channels, height, width)
        if flat.is_cuda:
            flat = flat.contiguous(memory_format=torch.channels_last)
        visual = self.frame_encoder(flat)
        return visual.reshape(batch, steps, self.num_visual_tokens, self.d_model)

    def tokens_from_visual(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        if visual_tokens.dim() != 4:
            raise ValueError(f"Expected visual tokens [B,T,M,D], got {tuple(visual_tokens.shape)}.")
        batch, steps, num_tokens, d_model = visual_tokens.shape
        if num_tokens != self.num_visual_tokens or d_model != self.d_model:
            raise ValueError(
                f"Expected visual tokens [B,T,{self.num_visual_tokens},{self.d_model}], "
                f"got {tuple(visual_tokens.shape)}."
            )

        time_ids = torch.arange(steps, device=visual_tokens.device)
        time_emb = self.temporal_pos(time_ids).to(dtype=visual_tokens.dtype).view(1, steps, 1, self.d_model)
        spatial_emb = self.spatial_token_pos.to(dtype=visual_tokens.dtype)
        x = visual_tokens + time_emb + spatial_emb
        return x.reshape(batch, steps * self.tokens_per_step, self.d_model)

    def temporal_features_from_visual(
        self,
        visual_tokens: torch.Tensor,
    ) -> torch.Tensor:
        x = self.tokens_from_visual(visual_tokens)
        attn_mask = self._frame_causal_mask(visual_tokens.size(1), x.device, x.dtype)
        for block in self.blocks:
            if self.use_activation_checkpointing and self.training and torch.is_grad_enabled():
                x = activation_checkpoint(block, x, attn_mask, use_reentrant=False)
            else:
                x = block(x, attn_mask)
        batch, steps = visual_tokens.shape[:2]
        return x.reshape(batch, steps, self.tokens_per_step, self.d_model)

    def action_states_from_visual(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        x = self.temporal_features_from_visual(visual_tokens)
        batch, steps, num_tokens, d_model = x.shape
        frame_tokens = x.reshape(batch * steps, num_tokens, d_model)
        query = self.readout_query.to(dtype=x.dtype).expand(batch * steps, -1, -1)
        readout, _ = self.readout_attn(
            query,
            self.readout_token_ln(frame_tokens),
            self.readout_token_ln(frame_tokens),
            need_weights=False,
        )
        state = readout.squeeze(1)
        state = state + self.readout_ffn(self.readout_state_ln(state))
        return state.reshape(batch, steps, d_model)

    def action_residual(self, prev_actions: Optional[torch.Tensor], *, dtype: torch.dtype) -> torch.Tensor:
        if not self.last_action_conditioning or prev_actions is None:
            raise ValueError("Feedback actions are required to compute action residuals.")
        residual = self.action_residual_mlp(prev_actions.to(dtype=dtype))
        return float(self.last_action_residual_cap) * torch.tanh(residual)

    def logits_from_visual(self, visual_tokens: torch.Tensor, prev_actions: torch.Tensor) -> torch.Tensor:
        logits, _ = self.logits_and_vision_from_visual(visual_tokens, prev_actions)
        return logits

    def logits_and_vision_from_visual(
        self,
        visual_tokens: torch.Tensor,
        prev_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, vision_logits, _onset_logits, _offset_logits = self.logits_vision_events_from_visual(
            visual_tokens,
            prev_actions,
        )
        return logits, vision_logits

    def logits_vision_events_from_visual(
        self,
        visual_tokens: torch.Tensor,
        prev_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_states = self.action_states_from_visual(visual_tokens)
        vision_logits = self.vision_head(action_states)
        onset_logits = self.onset_head(action_states)
        offset_logits = self.offset_head(action_states)
        if not self.last_action_conditioning:
            return vision_logits, vision_logits, onset_logits, offset_logits
        expected_actions = (visual_tokens.size(0), visual_tokens.size(1), len(DEFAULT_ACTION_NAMES))
        if tuple(prev_actions.shape) != expected_actions:
            raise ValueError(f"Expected feedback actions {expected_actions}, got {tuple(prev_actions.shape)}.")
        logits = vision_logits + self.action_residual(prev_actions, dtype=vision_logits.dtype)
        return logits, vision_logits, onset_logits, offset_logits

    def logits_vision_change_from_visual(
        self,
        visual_tokens: torch.Tensor,
        prev_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, vision_logits, onset_logits, offset_logits = self.logits_vision_events_from_visual(
            visual_tokens,
            prev_actions,
        )
        return logits, vision_logits, torch.maximum(onset_logits, offset_logits)

    def vision_logits_from_visual(self, visual_tokens: torch.Tensor, prev_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        del prev_actions
        return self.vision_head(self.action_states_from_visual(visual_tokens))

    def onset_offset_logits_from_visual(self, visual_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_states = self.action_states_from_visual(visual_tokens)
        return self.onset_head(action_states), self.offset_head(action_states)

    def change_logits_from_visual(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        onset_logits, offset_logits = self.onset_offset_logits_from_visual(visual_tokens)
        return torch.maximum(onset_logits, offset_logits)

    def forward(self, frames: torch.Tensor, prev_actions: torch.Tensor) -> torch.Tensor:
        visual_tokens = self.encode_visual_tokens(frames)
        return self.logits_from_visual(visual_tokens, prev_actions)


def apply_static_masks(frames: torch.Tensor, *, clone: bool = True) -> torch.Tensor:
    """Mean-fill game UI regions while retaining a three-channel RGB input."""

    if frames.dim() < 4:
        raise ValueError(f"Expected image tensor with at least four dimensions, got {tuple(frames.shape)}.")
    height, width = frames.shape[-2:]
    output = frames.clone() if clone else frames
    fill = output.mean(dim=(-2, -1), keepdim=True)

    def fill_region(y0: int, y1: int, x0: int, x1: int) -> None:
        y0 = max(0, min(height, int(y0)))
        y1 = max(y0, min(height, int(y1)))
        x0 = max(0, min(width, int(x0)))
        x1 = max(x0, min(width, int(x1)))
        if y1 > y0 and x1 > x0:
            output[..., y0:y1, x0:x1] = fill

    fill_region(int(height * 0.96), height, 0, width)
    fill_region(int(height * 0.78), int(height * 0.97), int(width * 0.33), int(width * 0.45))
    fill_region(int(height * 0.78), int(height * 0.97), int(width * 0.56), int(width * 0.67))
    fill_region(int(height * 0.08), int(height * 0.14), int(width * 0.66), int(width * 0.79))
    fill_region(int(height * 0.05), int(height * 0.20), int(width * 0.75), width)
    fill_region(0, int(height * 0.10), 0, int(width * 0.10))
    return output


class DrivingVideoPolicy(nn.Module):
    """Compatibility wrapper around the CNN grid-token causal transformer."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.policy = GreenvilleBCFormer(
            context_len=DEFAULT_CONTEXT_LENGTH,
            num_visual_tokens=NUM_VISUAL_TOKENS,
            d_model=READOUT_CHANNELS,
            n_heads=TEMPORAL_HEADS,
            n_layers=TEMPORAL_LAYERS,
            dropout=float(cfg.head_dropout),
            last_action_conditioning=bool(cfg.last_action_conditioning),
            last_action_residual_cap=float(cfg.last_action_residual_cap),
            use_activation_checkpointing=bool(cfg.use_activation_checkpointing),
        )

        # Expose the new encoder for direct inspection without legacy private APIs.
        self.frame_encoder = self.policy.frame_encoder
        self.feat_channels = FUSED_CHANNELS

    @property
    def context_len(self) -> int:
        return self.policy.context_len

    def _normalize_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dtype == torch.uint8:
            return frames.float().div_(255.0)
        if not torch.is_floating_point(frames):
            raise TypeError(f"frames must be uint8 or floating point, got {frames.dtype}.")
        return frames

    def _apply_masks(self, frames: torch.Tensor, *, clone: bool = True) -> torch.Tensor:
        return apply_static_masks(frames, clone=clone)

    def _prepare_frames(self, frames: torch.Tensor) -> torch.Tensor:
        frames = self._normalize_frames(frames)
        frames = self._apply_masks(frames, clone=bool(frames.requires_grad))
        if frames.is_cuda and frames.dim() == 4:
            frames = frames.contiguous(memory_format=torch.channels_last)
        return frames

    def _empty_actions(
        self,
        batch: int,
        steps: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(batch, steps, self.cfg.num_bin, device=device, dtype=dtype)

    def _sequence_prev_actions(
        self,
        prev_action: Optional[torch.Tensor],
        batch: int,
        steps: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not self.cfg.last_action_conditioning or prev_action is None:
            return self._empty_actions(batch, steps, device=device, dtype=dtype)
        prev_action = prev_action.to(device=device, dtype=dtype)
        if prev_action.dim() == 3:
            expected = (batch, steps, self.cfg.num_bin)
            if tuple(prev_action.shape) != expected:
                raise ValueError(f"Expected sequence feedback actions {expected}, got {tuple(prev_action.shape)}.")
            return prev_action
        if prev_action.dim() == 2 and steps == 1:
            expected = (batch, self.cfg.num_bin)
            if tuple(prev_action.shape) != expected:
                raise ValueError(f"Expected feedback action {expected}, got {tuple(prev_action.shape)}.")
            return prev_action.unsqueeze(1)
        raise ValueError(
            "Feedback actions must be [B,T,6], or [B,6] only for single-step/non-sequence calls; "
            f"got {tuple(prev_action.shape)}."
        )

    def _initial_feedback_action(
        self,
        prev_action: Optional[torch.Tensor],
        batch: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not self.cfg.last_action_conditioning or prev_action is None:
            return torch.zeros(batch, self.cfg.num_bin, device=device, dtype=dtype)
        prev_action = prev_action.to(device=device, dtype=dtype)
        expected = (batch, self.cfg.num_bin)
        if tuple(prev_action.shape) != expected:
            raise ValueError(f"Autoregressive initial action must have shape {expected}, got {tuple(prev_action.shape)}.")
        return prev_action

    def _state_prefix(
        self,
        state: Optional[TemporalState],
        batch: int,
        current_steps: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_prefix = max(0, self.context_len - int(current_steps))
        if (
            max_prefix <= 0
            or state is None
            or state.frames is None
            or state.prev_actions is None
        ):
            return (
                torch.empty(batch, 0, RGB_CHANNELS, DEFAULT_MODEL_SIZE, DEFAULT_MODEL_SIZE, device=device, dtype=dtype),
                torch.empty(batch, 0, self.cfg.num_bin, device=device, dtype=dtype),
            )

        frames = state.frames.to(device=device, dtype=dtype)
        actions = state.prev_actions.to(device=device, dtype=dtype)
        if frames.dim() != 5 or actions.dim() != 3:
            raise ValueError(
                f"TemporalState must contain frames [B,T,3,{DEFAULT_MODEL_SIZE},{DEFAULT_MODEL_SIZE}] "
                "and prev_actions [B,T,6], "
                f"got {tuple(frames.shape)} and {tuple(actions.shape)}."
            )
        if frames.size(0) != batch or actions.size(0) != batch:
            raise ValueError(
                f"TemporalState batch size must be {batch}, got frames={frames.size(0)} actions={actions.size(0)}."
            )
        if frames.size(1) != actions.size(1):
            raise ValueError(
                f"TemporalState frame/action lengths differ: {frames.size(1)} vs {actions.size(1)}."
            )
        if frames.size(2) != RGB_CHANNELS or tuple(frames.shape[-2:]) != (DEFAULT_MODEL_SIZE, DEFAULT_MODEL_SIZE):
            raise ValueError(f"Invalid TemporalState frame shape {tuple(frames.shape)}.")
        if actions.size(2) != self.cfg.num_bin:
            raise ValueError(f"Invalid TemporalState action shape {tuple(actions.shape)}.")
        return frames[:, -max_prefix:], actions[:, -max_prefix:]

    def _state_visual_prefix(
        self,
        state: Optional[TemporalState],
        batch: int,
        current_steps: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_prefix = max(0, self.context_len - int(current_steps))
        empty_visual = torch.empty(
            batch,
            0,
            NUM_VISUAL_TOKENS,
            READOUT_CHANNELS,
            device=device,
            dtype=dtype,
        )
        empty_actions = torch.empty(batch, 0, self.cfg.num_bin, device=device, dtype=dtype)
        if max_prefix <= 0 or state is None or state.prev_actions is None:
            return empty_visual, empty_actions

        actions = state.prev_actions.to(device=device, dtype=dtype)
        if state.visual_tokens is not None:
            visual = state.visual_tokens.to(device=device, dtype=dtype)
            if visual.dim() != 4 or actions.dim() != 3:
                raise ValueError(
                    f"TemporalState must contain visual_tokens [B,T,{NUM_VISUAL_TOKENS},{READOUT_CHANNELS}] "
                    "and prev_actions [B,T,6], "
                    f"got {tuple(visual.shape)} and {tuple(actions.shape)}."
                )
            if visual.size(0) != batch or actions.size(0) != batch:
                raise ValueError(
                    f"TemporalState batch size must be {batch}, got visual={visual.size(0)} actions={actions.size(0)}."
                )
            if visual.size(1) != actions.size(1):
                raise ValueError(
                    f"TemporalState visual/action lengths differ: {visual.size(1)} vs {actions.size(1)}."
                )
            if visual.size(2) != NUM_VISUAL_TOKENS or visual.size(3) != READOUT_CHANNELS:
                raise ValueError(f"Invalid TemporalState visual token shape {tuple(visual.shape)}.")
            if actions.size(2) != self.cfg.num_bin:
                raise ValueError(f"Invalid TemporalState action shape {tuple(actions.shape)}.")
            return visual[:, -max_prefix:], actions[:, -max_prefix:]

        prefix_frames, prefix_actions = self._state_prefix(
            state,
            batch,
            current_steps,
            device=device,
            dtype=dtype,
        )
        if prefix_frames.size(1) == 0:
            return empty_visual, empty_actions
        return self.policy.encode_visual_tokens(prefix_frames), prefix_actions.to(dtype=dtype)

    def _make_state(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
        *,
        keep_cache: bool,
    ) -> TemporalState:
        if not keep_cache:
            return TemporalState()
        return TemporalState(
            frames=frames[:, -self.context_len :].detach(),
            prev_actions=actions[:, -self.context_len :].detach(),
        )

    def _thresholds(
        self,
        feedback_thresholds: Optional[torch.Tensor],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if feedback_thresholds is None:
            return torch.full((self.cfg.num_bin,), 0.5, device=device, dtype=torch.float32)
        thresholds = feedback_thresholds.to(device=device, dtype=torch.float32).reshape(-1)
        if thresholds.numel() != self.cfg.num_bin:
            raise ValueError(f"Expected {self.cfg.num_bin} feedback thresholds, got {thresholds.numel()}.")
        return thresholds

    def _logits_with_zero_action_aux(
        self,
        visual_tokens: torch.Tensor,
        actions: torch.Tensor,
        *,
        current_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, vision_logits, onset_logits, offset_logits = self.policy.logits_vision_events_from_visual(
            visual_tokens,
            actions,
        )
        return (
            logits[:, -current_steps:],
            vision_logits[:, -current_steps:],
            onset_logits[:, -current_steps:],
            offset_logits[:, -current_steps:],
        )

    def _autoregressive_logits(
        self,
        visual_tokens: torch.Tensor,
        prefix_actions: torch.Tensor,
        initial_prev_action: torch.Tensor,
        thresholds: torch.Tensor,
        current_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        generated_actions = []
        previous = initial_prev_action
        prefix_len = prefix_actions.size(1)
        vision_logits = self.policy.vision_logits_from_visual(visual_tokens)

        with torch.no_grad():
            for index in range(current_steps):
                generated_actions.append(previous)
                residual = self.policy.action_residual(previous.unsqueeze(1), dtype=vision_logits.dtype)[:, 0]
                step_logits = vision_logits[:, prefix_len + index] + residual
                previous = (torch.sigmoid(step_logits.float()) >= thresholds.view(1, -1)).to(
                    dtype=initial_prev_action.dtype
                )

        current_actions = torch.stack(generated_actions, dim=1)
        actions = torch.cat([prefix_actions, current_actions], dim=1)
        current_logits = vision_logits[:, -current_steps:] + self.policy.action_residual(
            current_actions,
            dtype=vision_logits.dtype,
        )
        return current_logits, current_actions, previous.detach()

    def forward_step(
        self,
        frame: torch.Tensor,
        state: Optional[TemporalState] = None,
        *,
        prev_action: Optional[torch.Tensor] = None,
        return_vision_aux: bool = True,
    ) -> Tuple[PolicyOutput, TemporalState]:
        """Run one RGB frame with a rolling causal context."""

        if frame.dim() != 4 or frame.size(1) != RGB_CHANNELS:
            raise ValueError(f"Expected frame [B,3,H,W], got {tuple(frame.shape)}.")
        frame = self._prepare_frames(frame).unsqueeze(1)
        batch = frame.size(0)
        current_visual = self.policy.encode_visual_tokens(frame)
        action = self._sequence_prev_actions(
            prev_action,
            batch,
            1,
            device=frame.device,
            dtype=current_visual.dtype,
        )
        prefix_visual, prefix_actions = self._state_visual_prefix(
            state,
            batch,
            1,
            device=frame.device,
            dtype=current_visual.dtype,
        )
        visual_tokens = torch.cat([prefix_visual, current_visual], dim=1)
        actions = torch.cat([prefix_actions, action], dim=1)
        logits, sequence_vision_logits, sequence_onset_logits, sequence_offset_logits = self.policy.logits_vision_events_from_visual(
            visual_tokens,
            actions,
        )
        logits = logits[:, -1]
        vision_logits = sequence_vision_logits[:, -1]
        onset_logits = sequence_onset_logits[:, -1]
        offset_logits = sequence_offset_logits[:, -1]
        change_logits = torch.maximum(onset_logits, offset_logits)
        next_state = TemporalState(
            prev_actions=actions[:, -self.context_len :].detach(),
            visual_tokens=visual_tokens[:, -self.context_len :].detach(),
        )
        output = PolicyOutput(
            button_logits=logits,
            vision_button_logits=vision_logits,
            onset_logits=onset_logits,
            offset_logits=offset_logits,
            change_logits=change_logits,
        )
        return output, next_state

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
        """Run a causal frame sequence and optionally return dense logits."""

        del feedback_mask
        if soft_feedback:
            raise ValueError("Soft action feedback is disabled for this architecture.")
        if frames.dim() != 5 or frames.size(2) != RGB_CHANNELS:
            raise ValueError(f"Expected frames [B,T,3,H,W], got {tuple(frames.shape)}.")
        if frames.size(1) <= 0:
            raise ValueError("The frame sequence must contain at least one frame.")

        frames = self._prepare_frames(frames)
        batch, steps = frames.shape[:2]
        prefix_frames, prefix_actions = self._state_prefix(
            state,
            batch,
            steps,
            device=frames.device,
            dtype=frames.dtype,
        )
        prefix_len = prefix_frames.size(1)
        all_frames = torch.cat([prefix_frames, frames], dim=1)
        visual_tokens = self.policy.encode_visual_tokens(all_frames)

        next_feedback_action = None
        if autoregressive_feedback:
            thresholds = self._thresholds(feedback_thresholds, device=frames.device)
            initial_action = self._initial_feedback_action(
                prev_action,
                batch,
                device=frames.device,
                dtype=frames.dtype,
            )
            logits, current_actions, next_feedback_action = self._autoregressive_logits(
                visual_tokens,
                prefix_actions,
                initial_action,
                thresholds,
                steps,
            )
            all_actions = torch.cat([prefix_actions, current_actions], dim=1)
            vision_logits = self.policy.vision_logits_from_visual(visual_tokens)[:, -steps:]
            onset_logits, offset_logits = self.policy.onset_offset_logits_from_visual(visual_tokens)
            onset_logits = onset_logits[:, -steps:]
            offset_logits = offset_logits[:, -steps:]
        else:
            current_actions = self._sequence_prev_actions(
                prev_action,
                batch,
                steps,
                device=frames.device,
                dtype=frames.dtype,
            )
            all_actions = torch.cat([prefix_actions, current_actions], dim=1)
            logits, vision_logits, onset_logits, offset_logits = self._logits_with_zero_action_aux(
                visual_tokens,
                all_actions,
                current_steps=steps,
            )
        change_logits = torch.maximum(onset_logits, offset_logits)

        next_state = self._make_state(
            all_frames,
            all_actions,
            keep_cache=steps < self.context_len or prefix_len > 0,
        )
        output = PolicyOutput(
            button_logits=logits[:, -1],
            sequence_button_logits=logits if return_sequence_logits else None,
            vision_button_logits=vision_logits[:, -1],
            sequence_vision_button_logits=vision_logits if return_sequence_logits else None,
            onset_logits=onset_logits[:, -1],
            sequence_onset_logits=onset_logits if return_sequence_logits else None,
            offset_logits=offset_logits[:, -1],
            sequence_offset_logits=offset_logits if return_sequence_logits else None,
            change_logits=change_logits[:, -1],
            sequence_change_logits=change_logits if return_sequence_logits else None,
            next_feedback_action=next_feedback_action,
        )
        return (output, next_state) if return_aux else output
