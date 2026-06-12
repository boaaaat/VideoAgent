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
    prediction_horizon: int = 5

    key_names: Optional[List[str]] = None
    mouse_button_names: Optional[List[str]] = None

    d_model: int = 192
    frame_spatial_pool: int = 8
    # Kept for checkpoint/config compatibility; the spatial encoder now works
    # directly at d_model width, so this value is unused by the model.
    frame_spatial_channels: int = 32
    spatial_layers: int = 2
    spatial_heads: int = 4
    spatial_mlp_ratio: float = 2.0
    summary_tokens: int = 4
    temporal_layers: int = 4
    temporal_heads: int = 4
    temporal_mlp_ratio: float = 2.0
    dropout: float = 0.10
    coord_scale: float = 1.0
    coord_dropout: float = 0.1
    encode_chunk_size: int = 16
    max_context: int = 80

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
        self.spatial_layers = max(1, int(self.spatial_layers))
        self.spatial_heads = max(1, int(self.spatial_heads))
        while self.d_model % self.spatial_heads != 0 and self.spatial_heads > 1:
            self.spatial_heads -= 1
        self.spatial_mlp_ratio = float(min(max(self.spatial_mlp_ratio, 1.0), 8.0))
        self.summary_tokens = max(1, int(self.summary_tokens))
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


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.drop = nn.Dropout2d(float(dropout)) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(F.silu(self.norm(self.conv(x)), inplace=True))


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, *, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(channels, channels, stride=1, dropout=dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.gamma = nn.Parameter(torch.ones(channels) * 1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return x + y * self.gamma.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)


class FrameCNN(nn.Module):
    """Stride-16 conv backbone producing a d_model-wide spatial token grid."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        widths = [
            max(32, cfg.d_model // 4),
            max(48, cfg.d_model // 2),
            max(96, (3 * cfg.d_model) // 4),
            cfg.d_model,
        ]
        layers: List[nn.Module] = []
        in_channels = 8
        for idx, out_channels in enumerate(widths):
            layers.append(ConvBlock(in_channels, out_channels, stride=2, dropout=cfg.dropout * 0.2))
            layers.append(ResidualConvBlock(out_channels, dropout=cfg.dropout * 0.2))
            if idx >= 1:
                layers.append(ResidualConvBlock(out_channels, dropout=cfg.dropout * 0.2))
            in_channels = out_channels
        self.net = nn.Sequential(*layers)
        self.pool_size = int(cfg.frame_spatial_pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = F.adaptive_avg_pool2d(x, output_size=(self.pool_size, self.pool_size))
        return x.flatten(2).transpose(1, 2).contiguous()


class TransformerBlock(nn.Module):
    """Pre-LN self-attention block; causal when an attn_mask is supplied."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        *,
        depth: int = 1,
    ):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(d_model)
        hidden_dim = max(d_model, int(round(d_model * float(mlp_ratio))))
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        residual_std = 0.02 / math.sqrt(2.0 * max(1, int(depth)))
        nn.init.trunc_normal_(self.attn.out_proj.weight, std=residual_std)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.trunc_normal_(self.mlp[3].weight, std=residual_std)
        nn.init.zeros_(self.mlp[3].bias)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.attn_norm(x)
        y, _ = self.attn(y, y, y, attn_mask=attn_mask, need_weights=False)
        x = x + self.attn_drop(y)
        return x + self.mlp(self.mlp_norm(x))


class SpatialEncoder(nn.Module):
    """Per-frame spatial reasoning: 2D pos embeddings, self-attention over the
    token grid, then attention pooling into a compact frame summary."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        grid = int(cfg.frame_spatial_pool)
        d = int(cfg.d_model)
        self.grid = grid
        self.row_embed = nn.Parameter(torch.zeros(grid, d))
        self.col_embed = nn.Parameter(torch.zeros(grid, d))
        self.blocks = nn.ModuleList(
            TransformerBlock(
                d,
                cfg.spatial_heads,
                cfg.spatial_mlp_ratio,
                cfg.dropout,
                depth=cfg.spatial_layers,
            )
            for _ in range(cfg.spatial_layers)
        )
        self.norm = nn.LayerNorm(d)
        self.queries = nn.Parameter(torch.zeros(int(cfg.summary_tokens), d))
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=cfg.spatial_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.Linear(int(cfg.summary_tokens) * d, d),
            nn.LayerNorm(d),
        )
        nn.init.trunc_normal_(self.row_embed, std=0.02)
        nn.init.trunc_normal_(self.col_embed, std=0.02)
        nn.init.trunc_normal_(self.queries, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        n = tokens.size(0)
        pos = (self.row_embed.unsqueeze(1) + self.col_embed.unsqueeze(0)).view(self.grid * self.grid, -1)
        x = tokens + pos.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        queries = self.queries.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(n, -1, -1)
        pooled, _ = self.pool_attn(queries, x, x, need_weights=False)
        return self.out(pooled.flatten(1))


class CausalAttentionTemporal(nn.Module):
    def __init__(self, cfg: ModelConfig, *, input_dim: int):
        super().__init__()
        self.cfg = cfg
        self.max_context = int(cfg.max_context)
        self.in_norm = nn.LayerNorm(input_dim)
        self.in_proj = nn.Linear(input_dim, cfg.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(self.max_context, cfg.d_model))
        self.blocks = nn.ModuleList(
            TransformerBlock(
                cfg.d_model,
                cfg.temporal_heads,
                cfg.temporal_mlp_ratio,
                cfg.dropout,
                depth=cfg.temporal_layers,
            )
            for _ in range(cfg.temporal_layers)
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _encode_context(self, x: torch.Tensor) -> torch.Tensor:
        length = int(x.size(1))
        if length > self.max_context:
            raise ValueError(f"Temporal context length {length} exceeds max_context={self.max_context}.")
        y = self.in_proj(self.in_norm(x))
        pos = self.pos_embed[:length].to(device=y.device, dtype=y.dtype).unsqueeze(0)
        y = y + pos
        attn_mask = torch.ones((length, length), device=y.device, dtype=torch.bool).triu(1)
        for block in self.blocks:
            y = block(y, attn_mask)
        return self.norm(y)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        *,
        update_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if hidden is not None:
            hidden = hidden.to(device=x.device, dtype=x.dtype)
            context = torch.cat([hidden, x], dim=1)
        else:
            context = x
        if context.size(1) > self.max_context:
            context = context[:, -self.max_context :].contiguous()
        y = self._encode_context(context)
        if hidden is None and not update_cache:
            return y, None
        return y[:, -x.size(1) :].contiguous(), context.detach()


class ActionConditionedVideoPolicy(nn.Module):
    """CNN + spatial transformer + causal temporal transformer driving policy.

    Streaming note: `TemporalState.hidden_state` caches the post-fusion temporal
    inputs (one d_model vector per frame), which are exactly the tensors that
    `forward()` feeds into the temporal stack — so `forward_step` reproduces
    `forward()` logits while only encoding the newest frame each step. Once the
    cache exceeds `max_context`, the window slides and positional embeddings
    realign to the truncated context (same behavior as full-sequence forward).
    """

    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else ModelConfig()
        self.frame_tokens = int(self.cfg.frame_spatial_pool) * int(self.cfg.frame_spatial_pool)
        self.frame_feature_dim = int(self.cfg.d_model)

        self.frame_encoder = FrameCNN(self.cfg)
        self.spatial_encoder = SpatialEncoder(self.cfg)
        self.dt_embed = nn.Sequential(
            nn.Linear(3, self.cfg.d_model),
            nn.GELU(),
            nn.Linear(self.cfg.d_model, self.cfg.d_model),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(self.cfg.d_model),
            nn.Linear(self.cfg.d_model, self.cfg.d_model),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.d_model, self.cfg.d_model),
        )
        self.temporal = CausalAttentionTemporal(self.cfg, input_dim=self.cfg.d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(self.cfg.d_model),
            nn.Linear(self.cfg.d_model, self.cfg.d_model),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
        )
        self.button_head = nn.Linear(self.cfg.d_model, self.cfg.prediction_horizon * self.cfg.num_bin)

        nn.init.constant_(self.button_head.bias, -1.0)

    @property
    def num_buttons(self) -> int:
        return int(self.cfg.num_bin)

    def parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        params = [p for p in self.parameters() if p.requires_grad]
        return {"model": params, "backbone": params, "controller": params}

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> TemporalState:
        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else next(self.parameters()).dtype
        return TemporalState(
            prev_frame=None,
            hidden_state=None,
            steps=0,
        )

    def _normalize_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0
        return frames.clamp(0.0, 1.0)

    def _dt_features(self, dt: torch.Tensor) -> torch.Tensor:
        if dt.dim() == 3 and dt.size(-1) == 1:
            dt = dt[..., 0]
        dt = dt.clamp(min=1.0 / 240.0, max=0.5)
        features = torch.stack([dt, torch.log(dt), 1.0 / dt], dim=-1)
        return self.dt_embed(features.to(dtype=next(self.dt_embed.parameters()).dtype))

    def _button_thresholds(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        thresholds = self.cfg.button_state_thresholds
        if thresholds is None:
            return torch.full((self.cfg.num_bin,), float(self.cfg.button_state_threshold), device=device, dtype=dtype)
        return torch.tensor(list(thresholds), device=device, dtype=dtype)

    def _coord_channels(
        self,
        *,
        batch: int,
        steps: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype).view(1, 1, 1, height, 1)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype).view(1, 1, 1, 1, width)
        x = x.expand(batch, steps, 1, height, width)
        y = y.expand(batch, steps, 1, height, width)
        coords = torch.cat([x, y], dim=2)
        if float(self.cfg.coord_scale) != 1.0:
            coords = coords * float(self.cfg.coord_scale)
        if self.training and float(self.cfg.coord_dropout) > 0.0:
            keep = torch.rand((batch, 1, 1, 1, 1), device=device) >= float(self.cfg.coord_dropout)
            coords = coords * keep.to(dtype=dtype)
        return coords

    def _encode_frames(
        self,
        frames: torch.Tensor,
        *,
        previous_frame: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if frames.dim() == 4:
            frames = frames.unsqueeze(1)
        if frames.dim() != 5:
            raise ValueError(f"Expected frames [B,T,3,H,W], got {tuple(frames.shape)}.")
        b, t, c, h, w = frames.shape
        if c != 3:
            raise ValueError(f"Expected RGB frames, got shape {tuple(frames.shape)}.")

        frames = self._normalize_frames(frames)
        if previous_frame is None:
            prev = torch.cat([frames[:, :1], frames[:, :-1]], dim=1)
            motion = frames - prev
            motion[:, 0] = 0.0
        else:
            prev0 = previous_frame.to(device=frames.device, dtype=frames.dtype).unsqueeze(1)
            prev = torch.cat([prev0, frames[:, :-1]], dim=1)
            motion = frames - prev

        coords = self._coord_channels(
            batch=b,
            steps=t,
            height=h,
            width=w,
            device=frames.device,
            dtype=frames.dtype,
        )
        x = torch.cat([frames, motion, coords], dim=2).reshape(b * t, 8, h, w)
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)

        chunks: List[torch.Tensor] = []
        for start in range(0, x.size(0), int(self.cfg.encode_chunk_size)):
            tokens = self.frame_encoder(x[start : start + int(self.cfg.encode_chunk_size)])
            chunks.append(self.spatial_encoder(tokens))
        visual = torch.cat(chunks, dim=0).view(b, t, self.cfg.d_model)
        return visual, frames[:, -1].detach()

    def _build_inputs(self, visual: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        if dt.dim() == 1:
            dt = dt.unsqueeze(1).expand(visual.shape[0], visual.shape[1])
        dt_emb = self._dt_features(dt).to(dtype=visual.dtype)
        x = visual + dt_emb
        return x + self.fuse(x).to(dtype=visual.dtype)

    def _run_temporal(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        *,
        update_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.temporal(x, hidden, update_cache=update_cache)

    def _pack_output(self, temporal: torch.Tensor) -> PolicyOutput:
        b, t, d = temporal.shape
        h = self.head(temporal)
        button = self.button_head(h).view(b, t, self.cfg.prediction_horizon, self.cfg.num_bin)

        step_button = button[:, :, 0]
        return PolicyOutput(
            button_logits=step_button,
            horizon_button_logits=button,
            future_button_logits={idx + 1: button[:, :, idx] for idx in range(self.cfg.prediction_horizon)},
        )

    def forward(
        self,
        frames: torch.Tensor,
        dt: torch.Tensor,
        state: Optional[TemporalState] = None,
        return_aux: bool = False,
    ) -> PolicyOutput:
        del state, return_aux
        visual, _ = self._encode_frames(frames)
        temporal_in = self._build_inputs(visual, dt)
        temporal, _ = self._run_temporal(temporal_in)
        return self._pack_output(temporal)

    def forward_step(
        self,
        frame: torch.Tensor,
        dt: torch.Tensor,
        state: TemporalState,
        return_aux: bool = False,
    ) -> Tuple[PolicyOutput, TemporalState]:
        del return_aux
        if frame.dim() != 4:
            raise ValueError(f"forward_step expects frame shape [B,3,H,W], got {tuple(frame.shape)}.")
        visual, last_frame = self._encode_frames(frame.unsqueeze(1), previous_frame=state.prev_frame)
        step_input = self._build_inputs(visual, dt.reshape(frame.size(0), 1))

        temporal, hidden = self._run_temporal(step_input, state.hidden_state, update_cache=True)
        output = self._pack_output(temporal)

        new_state = TemporalState(
            prev_frame=last_frame.detach(),
            hidden_state=None if hidden is None else hidden.detach(),
            steps=int(state.steps) + 1,
        )
        squeezed = PolicyOutput(
            button_logits=output.button_logits[:, 0],
            horizon_button_logits=output.horizon_button_logits[:, 0],
            future_button_logits={k: v[:, 0] for k, v in output.future_button_logits.items()},
        )
        return squeezed, new_state


RealTimeTemporalControlNet = ActionConditionedVideoPolicy


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = ModelConfig(model_size=256, seq_len=16, prediction_horizon=5, max_context=16)
    model = ActionConditionedVideoPolicy(cfg).eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"params={params / 1e6:.2f}M num_bin={cfg.num_bin}")

    x = torch.rand(1, cfg.seq_len, 3, cfg.model_size, cfg.model_size)
    dt = torch.full((1, cfg.seq_len), cfg.prediction_dt)
    with torch.no_grad():
        out = model(x, dt=dt)
    assert out.button_logits.shape == (1, cfg.seq_len, cfg.num_bin), tuple(out.button_logits.shape)
    assert out.horizon_button_logits.shape == (1, cfg.seq_len, cfg.prediction_horizon, cfg.num_bin)

    state = model.init_state(1)
    with torch.no_grad():
        for t in range(cfg.seq_len):
            step_out, state = model.forward_step(x[:, t], dt[:, t], state)
            ref = out.horizon_button_logits[:, t]
            diff = (step_out.horizon_button_logits - ref).abs().max()
            assert torch.allclose(step_out.horizon_button_logits, ref, atol=1e-4), f"step {t}: max diff {diff}"
    print("streaming equivalence OK")

    x_u8 = (x * 255.0).to(torch.uint8)
    with torch.no_grad():
        out_u8 = model(x_u8, dt=dt)
    assert out_u8.horizon_button_logits.shape == out.horizon_button_logits.shape
    print("uint8 path OK:", tuple(out.horizon_button_logits.shape))
