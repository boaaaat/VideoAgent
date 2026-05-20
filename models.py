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
    seq_len: int = 100
    train_seq_stride: int = 50
    val_seq_stride: int = 100
    prediction_dt: float = 1.0 / 20.0
    prediction_horizon: int = 5

    key_names: Optional[List[str]] = None
    mouse_button_names: Optional[List[str]] = None

    d_model: int = 128
    temporal_backend: str = "gru"
    temporal_layers: int = 3
    temporal_kernel_size: int = 5
    dropout: float = 0.20
    encode_chunk_size: int = 16
    max_context: int = 100

    button_state_threshold: float = 0.5
    button_state_thresholds: Optional[Sequence[float]] = None
    transition_threshold: float = 0.5
    mouse_active_threshold: float = 0.5
    mouse_velocity_scales: Optional[Sequence[float]] = None

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
        self.temporal_backend = str(self.temporal_backend).strip().lower()
        if self.temporal_backend not in {"gru", "tcn"}:
            raise ValueError(f"temporal_backend must be 'gru' or 'tcn', got {self.temporal_backend!r}.")
        self.temporal_layers = max(1, int(self.temporal_layers))
        self.temporal_kernel_size = max(2, int(self.temporal_kernel_size))
        self.dropout = float(min(max(self.dropout, 0.0), 0.9))
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

        if self.mouse_velocity_scales is None:
            self.mouse_velocity_scales = (1.0, 1.0)
        else:
            scales = tuple(float(x) for x in self.mouse_velocity_scales)
            if len(scales) != 2:
                raise ValueError("mouse_velocity_scales must contain exactly 2 values.")
            self.mouse_velocity_scales = scales

        self.button_state_threshold = float(min(max(self.button_state_threshold, 0.0), 1.0))
        self.transition_threshold = float(min(max(self.transition_threshold, 0.0), 1.0))
        self.mouse_active_threshold = float(min(max(self.mouse_active_threshold, 0.0), 1.0))
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
    press_logits: torch.Tensor
    release_logits: torch.Tensor
    mouse_active_logits: torch.Tensor
    mouse_delta: torch.Tensor
    horizon_button_logits: torch.Tensor
    horizon_press_logits: torch.Tensor
    horizon_release_logits: torch.Tensor
    horizon_mouse_active_logits: torch.Tensor
    horizon_mouse_delta: torch.Tensor
    future_button_logits: Dict[int, torch.Tensor] = field(default_factory=dict)
    future_press_logits: Dict[int, torch.Tensor] = field(default_factory=dict)
    future_release_logits: Dict[int, torch.Tensor] = field(default_factory=dict)
    future_mouse_active_logits: Dict[int, torch.Tensor] = field(default_factory=dict)
    future_mouse_mu: Dict[int, torch.Tensor] = field(default_factory=dict)
    future_mouse_log_b: Dict[int, torch.Tensor] = field(default_factory=dict)
    mouse_mu: Optional[torch.Tensor] = None
    mouse_log_b: Optional[torch.Tensor] = None


@dataclass
class TemporalState:
    cached_summaries: Optional[torch.Tensor] = None
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
        in_channels = 6
        for idx, out_channels in enumerate(widths):
            layers.append(ConvBlock(in_channels, out_channels, stride=2, dropout=cfg.dropout * 0.2))
            layers.append(ResidualConvBlock(out_channels, dropout=cfg.dropout * 0.2))
            in_channels = out_channels
            if idx >= 1:
                layers.append(ResidualConvBlock(out_channels, dropout=cfg.dropout * 0.2))
        self.net = nn.Sequential(*layers)
        self.proj = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return self.proj(x)


class CausalTCNBlock(nn.Module):
    def __init__(self, dim: int, *, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(
            dim,
            dim * 2,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            groups=dim,
        )
        self.mix = nn.Sequential(
            nn.Conv1d(dim, dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(dim * 2, dim, kernel_size=1),
        )
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        y = self.norm(x).transpose(1, 2)
        y = F.pad(y, ((self.kernel_size - 1) * self.dilation, 0))
        y = self.depthwise(y)
        a, b = y.chunk(2, dim=1)
        y = a * torch.sigmoid(b)
        y = self.mix(y).transpose(1, 2)
        return x + y * self.gamma.to(device=x.device, dtype=dtype).view(1, 1, -1)


class CausalTCN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                CausalTCNBlock(
                    cfg.d_model,
                    kernel_size=cfg.temporal_kernel_size,
                    dilation=2 ** idx,
                    dropout=cfg.dropout,
                )
                for idx in range(cfg.temporal_layers)
            ]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
        )
        self.out_gamma = nn.Parameter(torch.ones(cfg.d_model) * 1e-3)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = x + self.out(x) * self.out_gamma.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
        return self.norm(x)


class CausalGRU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.in_norm = nn.LayerNorm(cfg.d_model)
        self.gru = nn.GRU(
            input_size=cfg.d_model,
            hidden_size=cfg.d_model,
            num_layers=cfg.temporal_layers,
            dropout=cfg.dropout if cfg.temporal_layers > 1 else 0.0,
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
        )
        self.out_gamma = nn.Parameter(torch.ones(cfg.d_model) * 1e-3)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden is not None:
            hidden = hidden.to(device=x.device, dtype=x.dtype)
        y, hidden = self.gru(self.in_norm(x), hidden)
        x = x + self.out(y) * self.out_gamma.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
        return self.norm(x), hidden


class ActionConditionedVideoPolicy(nn.Module):
    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else ModelConfig()

        self.frame_encoder = FrameCNN(self.cfg)
        self.dt_embed = nn.Sequential(
            nn.Linear(3, self.cfg.d_model),
            nn.GELU(),
            nn.Linear(self.cfg.d_model, self.cfg.d_model),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(self.cfg.d_model * 2),
            nn.Linear(self.cfg.d_model * 2, self.cfg.d_model),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.d_model, self.cfg.d_model),
        )
        if self.cfg.temporal_backend == "gru":
            self.temporal = CausalGRU(self.cfg)
        else:
            self.temporal = CausalTCN(self.cfg)
        self.head = nn.Sequential(
            nn.LayerNorm(self.cfg.d_model),
            nn.Linear(self.cfg.d_model, self.cfg.d_model),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
        )
        self.button_head = nn.Linear(self.cfg.d_model, self.cfg.prediction_horizon * self.cfg.num_bin)
        self.press_head = nn.Linear(self.cfg.d_model, self.cfg.prediction_horizon * self.cfg.num_bin)
        self.release_head = nn.Linear(self.cfg.d_model, self.cfg.prediction_horizon * self.cfg.num_bin)
        # Mouse heads are disabled for the current Greenville keyboard-only setup.
        # Re-enable these lines and the matching _pack_output lines when training mouse actions again.
        # self.mouse_active_head = nn.Linear(self.cfg.d_model, self.cfg.prediction_horizon)
        # self.mouse_delta_head = nn.Linear(self.cfg.d_model, self.cfg.prediction_horizon * 2)

        nn.init.constant_(self.button_head.bias, -1.0)
        nn.init.constant_(self.press_head.bias, -2.0)
        nn.init.constant_(self.release_head.bias, -2.0)
        # nn.init.constant_(self.mouse_active_head.bias, -1.0)
        # nn.init.zeros_(self.mouse_delta_head.bias)

        self.register_buffer(
            "mouse_velocity_scale",
            torch.tensor(list(self.cfg.mouse_velocity_scales), dtype=torch.float32),
            persistent=True,
        )

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
            cached_summaries=None,
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

    def _scale_mouse(self, mouse_delta_norm: torch.Tensor) -> torch.Tensor:
        scale = self.mouse_velocity_scale.to(device=mouse_delta_norm.device, dtype=mouse_delta_norm.dtype)
        while scale.dim() < mouse_delta_norm.dim():
            scale = scale.unsqueeze(0)
        return mouse_delta_norm * scale

    def _button_thresholds(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        thresholds = self.cfg.button_state_thresholds
        if thresholds is None:
            return torch.full((self.cfg.num_bin,), float(self.cfg.button_state_threshold), device=device, dtype=dtype)
        return torch.tensor(list(thresholds), device=device, dtype=dtype)

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

        x = torch.cat([frames, motion], dim=2).reshape(b * t, 6, h, w)
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)

        chunks: List[torch.Tensor] = []
        for start in range(0, x.size(0), int(self.cfg.encode_chunk_size)):
            chunks.append(self.frame_encoder(x[start : start + int(self.cfg.encode_chunk_size)]))
        visual = torch.cat(chunks, dim=0).view(b, t, self.cfg.d_model)
        return visual, frames[:, -1].detach()

    def _build_inputs(self, visual: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        if dt.dim() == 1:
            dt = dt.unsqueeze(1).expand(visual.shape[0], visual.shape[1])
        dt_emb = self._dt_features(dt).to(dtype=visual.dtype)
        return visual + self.fuse(torch.cat([visual, dt_emb], dim=-1)).to(dtype=visual.dtype)

    def _run_temporal(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.cfg.temporal_backend == "gru":
            return self.temporal(x, hidden)
        return self.temporal(x), None

    def _pack_output(self, temporal: torch.Tensor) -> PolicyOutput:
        b, t, d = temporal.shape
        h = self.head(temporal)
        button = self.button_head(h).view(b, t, self.cfg.prediction_horizon, self.cfg.num_bin)
        press = self.press_head(h).view(b, t, self.cfg.prediction_horizon, self.cfg.num_bin)
        release = self.release_head(h).view(b, t, self.cfg.prediction_horizon, self.cfg.num_bin)
        # active = self.mouse_active_head(h).view(b, t, self.cfg.prediction_horizon, 1)
        # mouse = self._scale_mouse(self.mouse_delta_head(h).view(b, t, self.cfg.prediction_horizon, 2))
        active = h.new_zeros((b, t, self.cfg.prediction_horizon, 1))
        mouse = h.new_zeros((b, t, self.cfg.prediction_horizon, 2))

        step_button = button[:, :, 0]
        step_press = press[:, :, 0]
        step_release = release[:, :, 0]
        step_active = active[:, :, 0]
        step_mouse = mouse[:, :, 0]
        return PolicyOutput(
            button_logits=step_button,
            press_logits=step_press,
            release_logits=step_release,
            mouse_active_logits=step_active,
            mouse_delta=step_mouse,
            horizon_button_logits=button,
            horizon_press_logits=press,
            horizon_release_logits=release,
            horizon_mouse_active_logits=active,
            horizon_mouse_delta=mouse,
            future_button_logits={idx + 1: button[:, :, idx] for idx in range(self.cfg.prediction_horizon)},
            future_press_logits={idx + 1: press[:, :, idx] for idx in range(self.cfg.prediction_horizon)},
            future_release_logits={idx + 1: release[:, :, idx] for idx in range(self.cfg.prediction_horizon)},
            future_mouse_active_logits={idx + 1: active[:, :, idx] for idx in range(self.cfg.prediction_horizon)},
            future_mouse_mu={idx + 1: mouse[:, :, idx] for idx in range(self.cfg.prediction_horizon)},
            future_mouse_log_b={idx + 1: torch.zeros_like(mouse[:, :, idx]) for idx in range(self.cfg.prediction_horizon)},
            mouse_mu=step_mouse,
            mouse_log_b=torch.zeros_like(step_mouse),
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

        if self.cfg.temporal_backend == "gru":
            temporal, hidden = self._run_temporal(step_input, state.hidden_state)
            cache = None
            output = self._pack_output(temporal)
        else:
            hidden = None
            if state.cached_summaries is None:
                cache = step_input
            else:
                cache = torch.cat([state.cached_summaries.to(step_input.device, step_input.dtype), step_input], dim=1)
            if cache.size(1) > int(self.cfg.max_context):
                cache = cache[:, -int(self.cfg.max_context) :]

            temporal, _ = self._run_temporal(cache)
            output = self._pack_output(temporal[:, -1:])

        new_state = TemporalState(
            cached_summaries=None if cache is None else cache.detach(),
            prev_frame=last_frame.detach(),
            hidden_state=None if hidden is None else hidden.detach(),
            steps=int(state.steps) + 1,
        )
        squeezed = PolicyOutput(
            button_logits=output.button_logits[:, 0],
            press_logits=output.press_logits[:, 0],
            release_logits=output.release_logits[:, 0],
            mouse_active_logits=output.mouse_active_logits[:, 0],
            mouse_delta=output.mouse_delta[:, 0],
            horizon_button_logits=output.horizon_button_logits[:, 0],
            horizon_press_logits=output.horizon_press_logits[:, 0],
            horizon_release_logits=output.horizon_release_logits[:, 0],
            horizon_mouse_active_logits=output.horizon_mouse_active_logits[:, 0],
            horizon_mouse_delta=output.horizon_mouse_delta[:, 0],
            future_button_logits={k: v[:, 0] for k, v in output.future_button_logits.items()},
            future_press_logits={k: v[:, 0] for k, v in output.future_press_logits.items()},
            future_release_logits={k: v[:, 0] for k, v in output.future_release_logits.items()},
            future_mouse_active_logits={k: v[:, 0] for k, v in output.future_mouse_active_logits.items()},
            future_mouse_mu={k: v[:, 0] for k, v in output.future_mouse_mu.items()},
            future_mouse_log_b={k: v[:, 0] for k, v in output.future_mouse_log_b.items()},
            mouse_mu=output.mouse_delta[:, 0],
            mouse_log_b=torch.zeros_like(output.mouse_delta[:, 0]),
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
    print(tuple(out.horizon_button_logits.shape), tuple(out.horizon_mouse_delta.shape))
