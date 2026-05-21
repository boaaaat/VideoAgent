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
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        widths = [
            max(32, cfg.d_model // 8),
            max(48, cfg.d_model // 4),
            max(96, cfg.d_model // 2),
            cfg.d_model,
        ]
        layers: List[nn.Module] = []
        in_channels = 8
        for idx, out_channels in enumerate(widths):
            layers.append(ConvBlock(in_channels, out_channels, stride=2, dropout=cfg.dropout * 0.2))
            layers.append(ResidualConvBlock(out_channels, dropout=cfg.dropout * 0.2))
            in_channels = out_channels
            if idx >= 1:
                layers.append(ResidualConvBlock(out_channels, dropout=cfg.dropout * 0.2))
        self.net = nn.Sequential(*layers)
        self.pool_size = int(cfg.frame_spatial_pool)
        spatial_channels = int(cfg.frame_spatial_channels)
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(cfg.d_model, cfg.d_model, kernel_size=3, padding=1, groups=cfg.d_model, bias=False),
            nn.GroupNorm(_group_count(cfg.d_model), cfg.d_model),
            nn.SiLU(inplace=True),
            nn.Conv2d(cfg.d_model, spatial_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(spatial_channels), spatial_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = self.spatial_proj(x)
        x = F.adaptive_avg_pool2d(x, output_size=(self.pool_size, self.pool_size))
        return x.flatten(2).transpose(1, 2).contiguous()


class SpatialAttentionPool(nn.Module):
    def __init__(self, cfg: ModelConfig, *, num_source_tokens: int):
        super().__init__()
        self.num_queries = int(cfg.spatial_attention_tokens)
        self.grid_tokens = int(cfg.spatial_temporal_grid) * int(cfg.spatial_temporal_grid)
        self.token_norm = nn.LayerNorm(cfg.frame_spatial_channels)
        self.token_proj = nn.Linear(cfg.frame_spatial_channels, cfg.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(num_source_tokens, cfg.d_model))
        self.query_embed = nn.Parameter(torch.zeros(self.num_queries, cfg.d_model))
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.spatial_attention_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.mlp_norm = nn.LayerNorm(cfg.d_model)
        hidden_dim = max(cfg.d_model, int(round(cfg.d_model * float(cfg.temporal_mlp_ratio))))
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden_dim, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
        self.out_norm = nn.LayerNorm(cfg.d_model)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.query_embed, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_proj(self.token_norm(tokens))
        pos = self.pos_embed.to(device=x.device, dtype=x.dtype).unsqueeze(0)
        x = x + pos
        query = self.query_embed.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(x.size(0), -1, -1)
        pooled, _ = self.attn(query, x, x, need_weights=False)
        pooled = query + self.attn_drop(pooled)
        pooled = pooled + self.mlp(self.mlp_norm(pooled))
        pooled = self.out_norm(pooled)
        if pooled.size(1) == self.grid_tokens:
            return pooled.contiguous()
        pooled = pooled.transpose(1, 2)
        pooled = F.adaptive_avg_pool1d(pooled, self.grid_tokens)
        return pooled.transpose(1, 2).contiguous()


class CausalAttentionBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.temporal_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.mlp_norm = nn.LayerNorm(cfg.d_model)
        hidden_dim = max(cfg.d_model, int(round(cfg.d_model * float(cfg.temporal_mlp_ratio))))
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden_dim, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        y = self.attn_norm(x)
        y, _ = self.attn(y, y, y, attn_mask=attn_mask, need_weights=False)
        x = x + self.attn_drop(y)
        return x + self.mlp(self.mlp_norm(x))


class CausalAttentionTemporal(nn.Module):
    def __init__(self, cfg: ModelConfig, *, input_dim: int):
        super().__init__()
        self.cfg = cfg
        self.max_context = int(cfg.max_context)
        self.in_norm = nn.LayerNorm(input_dim)
        self.in_proj = nn.Linear(input_dim, cfg.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(self.max_context, cfg.d_model))
        self.blocks = nn.ModuleList(CausalAttentionBlock(cfg) for _ in range(cfg.temporal_layers))
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
        if update_cache:
            context = x if hidden is None else torch.cat([hidden.to(device=x.device, dtype=x.dtype), x], dim=1)
            if context.size(1) > self.max_context:
                context = context[:, -self.max_context :].contiguous()
            y = self._encode_context(context)
            return y[:, -x.size(1) :].contiguous(), context.detach()
        if hidden is not None:
            hidden = hidden.to(device=x.device, dtype=x.dtype)
            context = torch.cat([hidden, x], dim=1)
            if context.size(1) > self.max_context:
                context = context[:, -self.max_context :].contiguous()
            y = self._encode_context(context)
            return y[:, -x.size(1) :].contiguous(), context.detach()
        return self._encode_context(x), None


class ActionConditionedVideoPolicy(nn.Module):
    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else ModelConfig()
        self.frame_tokens = int(self.cfg.frame_spatial_pool) * int(self.cfg.frame_spatial_pool)
        self.spatial_grid_tokens = int(self.cfg.spatial_temporal_grid) * int(self.cfg.spatial_temporal_grid)
        self.frame_feature_dim = self.spatial_grid_tokens * int(self.cfg.d_model)

        self.frame_encoder = FrameCNN(self.cfg)
        self.spatial_pool = SpatialAttentionPool(self.cfg, num_source_tokens=self.frame_tokens)
        self.dt_embed = nn.Sequential(
            nn.Linear(3, self.cfg.d_model),
            nn.GELU(),
            nn.Linear(self.cfg.d_model, self.frame_feature_dim),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(self.frame_feature_dim),
            nn.Linear(self.frame_feature_dim, self.cfg.d_model),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.d_model, self.frame_feature_dim),
        )
        self.temporal = CausalAttentionTemporal(self.cfg, input_dim=self.frame_feature_dim)
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
            chunks.append(self.frame_encoder(x[start : start + int(self.cfg.encode_chunk_size)]))
        spatial_tokens = torch.cat(chunks, dim=0)
        visual = self.spatial_pool(spatial_tokens).view(b, t, self.frame_feature_dim)
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
    cfg = ModelConfig(model_size=256, seq_len=100, prediction_horizon=1, max_context=100)
    model = ActionConditionedVideoPolicy(cfg)
    x = torch.rand(1, cfg.seq_len, 3, cfg.model_size, cfg.model_size)
    dt = torch.full((1, cfg.seq_len), cfg.prediction_dt)
    out = model(x, dt=dt)
    print(tuple(out.horizon_button_logits.shape))
