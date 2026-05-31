import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inverse_dynamics import (  # noqa: E402
    InverseDynamicsConfig,
    InverseDynamicsModel,
    initialize_inverse_lazy_layers,
    inverse_checkpoint_family_mismatch_reason,
)
from models import (  # noqa: E402
    DrivingVideoPolicy,
    ModelConfig,
    TemporalState,
    get_key_names as MODEL_GET_KEY_NAMES,
    get_mouse_button_names as MODEL_GET_MOUSE_BUTTON_NAMES,
)


FeatureLayer = Literal["motion", "stage1", "stage2", "stage3", "stage4", "spatial", "tokens"]
ModelKind = Literal["auto", "policy", "inverse"]
LoadedModelKind = Literal["policy", "inverse"]
VisualModel = DrivingVideoPolicy | InverseDynamicsModel
VisualConfig = ModelConfig | InverseDynamicsConfig


class FFmpegWriter:
    def __init__(
        self,
        out_path: str,
        fps: float,
        width: int,
        height: int,
        *,
        ffmpeg_path: Optional[str] = None,
        codec: str = "hevc_nvenc",
        preset: str = "p4",
        quality: int = 20,
        pix_fmt_in: str = "bgr24",
        pix_fmt_out: str = "yuv420p",
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.out_path = out_path

        self.ffmpeg_path = ffmpeg_path or shutil.which("ffmpeg")
        if not self.ffmpeg_path:
            raise RuntimeError("ffmpeg not found. Install ffmpeg or pass --ffmpeg-path.")

        quality_flag = "-cq" if "nvenc" in str(codec).lower() else "-crf"
        self.cmd = [
            self.ffmpeg_path,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix_fmt_in,
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            codec,
            "-preset",
            str(preset),
            quality_flag,
            str(int(quality)),
            "-pix_fmt",
            pix_fmt_out,
            self.out_path,
        ]

        self.proc = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write(self, frame_bgr: np.ndarray) -> None:
        if self.proc is None or self.proc.stdin is None:
            return
        self.proc.stdin.write(frame_bgr.tobytes())

    def release(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.flush()
                self.proc.stdin.close()
        except Exception:
            pass
        self.proc.wait(timeout=10)
        self.proc = None


def _coerce_config_types(cfg: ModelConfig) -> ModelConfig:
    if cfg.key_names is None:
        cfg.key_names = MODEL_GET_KEY_NAMES(cfg.selected_game)
    else:
        cfg.key_names = list(cfg.key_names)

    if cfg.mouse_button_names is None:
        cfg.mouse_button_names = MODEL_GET_MOUSE_BUTTON_NAMES(cfg.selected_game)
    else:
        cfg.mouse_button_names = list(cfg.mouse_button_names)

    cfg.seq_len = int(cfg.seq_len)
    cfg.train_seq_stride = int(cfg.train_seq_stride)
    cfg.val_seq_stride = int(cfg.val_seq_stride)
    cfg.model_size = int(cfg.model_size)
    cfg.prediction_horizon = int(getattr(cfg, "prediction_horizon", 1))
    offsets = getattr(cfg, "prediction_horizon_offsets", None)
    if offsets is None:
        cfg.prediction_horizon_offsets = tuple(range(1, cfg.prediction_horizon + 1))
    else:
        cfg.prediction_horizon_offsets = tuple(int(offset) for offset in offsets)
        if not cfg.prediction_horizon_offsets:
            cfg.prediction_horizon_offsets = tuple(range(1, cfg.prediction_horizon + 1))
        if any(offset <= 0 for offset in cfg.prediction_horizon_offsets):
            raise ValueError(f"prediction_horizon_offsets must be positive, got {cfg.prediction_horizon_offsets}.")
        if any(curr <= prev for prev, curr in zip(cfg.prediction_horizon_offsets, cfg.prediction_horizon_offsets[1:])):
            raise ValueError(
                f"prediction_horizon_offsets must be strictly increasing, got {cfg.prediction_horizon_offsets}."
            )
        cfg.prediction_horizon = len(cfg.prediction_horizon_offsets)
    cfg.d_model = int(cfg.d_model)
    cfg.num_bin = len(cfg.key_names) + len(cfg.mouse_button_names)
    return cfg


def _apply_config_overrides(cfg: ModelConfig, overrides: Dict) -> ModelConfig:
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return _coerce_config_types(cfg)


def _load_checkpoint_state(ckpt_path: str, device: torch.device):
    return torch.load(ckpt_path, map_location=device)


def _extract_checkpoint_config(state) -> Dict:
    if isinstance(state, dict) and isinstance(state.get("config"), dict):
        return dict(state["config"])
    return {}


def _extract_model_state(state):
    if isinstance(state, dict) and "model_state" in state:
        return state["model_state"]
    return state


def _detect_checkpoint_kind(state) -> LoadedModelKind:
    config_dict = _extract_checkpoint_config(state)
    inverse_keys = {field.name for field in fields(InverseDynamicsConfig)}
    policy_keys = {field.name for field in fields(ModelConfig)}

    inverse_markers = {"output_seq_len", "visual_encoder_name", "cnn_channels", "gru_hidden_size", "gru_layers"}
    policy_markers = {"prediction_horizon", "model_size", "d_model"}
    if any(key in config_dict for key in inverse_markers):
        return "inverse"
    if any(key in config_dict for key in policy_markers):
        return "policy"

    inverse_hits = sum(1 for key in config_dict if key in inverse_keys)
    policy_hits = sum(1 for key in config_dict if key in policy_keys)
    if inverse_hits > policy_hits:
        return "inverse"
    if policy_hits > inverse_hits:
        return "policy"

    model_state = _extract_model_state(state)
    if isinstance(model_state, dict):
        inverse_state_markers = ("visual_encoder.", "frame_projector.", "temporal_model.", "action_head.")
        if any(str(key).startswith(inverse_state_markers) for key in model_state):
            return "inverse"
    return "policy"


def _find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    search_dirs = [
        ckpt_dir,
        os.path.join(os.getcwd(), ckpt_dir),
        os.path.join(os.path.dirname(__file__), ckpt_dir),
    ]
    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue
        best_ckpt = os.path.join(directory, "model_best.pt")
        if os.path.exists(best_ckpt):
            return best_ckpt
        epoch_ckpts = glob.glob(os.path.join(directory, "model_epoch_*.pt"))
        if epoch_ckpts:
            epoch_ckpts.sort(key=os.path.getmtime)
            return epoch_ckpts[-1]
    return None


def initialize_model_lazy_layers(
    model: DrivingVideoPolicy,
    cfg: ModelConfig,
    device: torch.device,
) -> None:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy_frames = torch.zeros(1, 1, 3, cfg.model_size, cfg.model_size, device=device)
        _ = model(dummy_frames)
    model.train(was_training)


def load_model_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
) -> Tuple[DrivingVideoPolicy, ModelConfig]:
    state = _load_checkpoint_state(ckpt_path, device)
    cfg = ModelConfig()
    config_dict = _extract_checkpoint_config(state)
    if config_dict:
        cfg = _apply_config_overrides(cfg, config_dict)
    else:
        cfg = _coerce_config_types(cfg)

    model = DrivingVideoPolicy(cfg=cfg).to(device)
    initialize_model_lazy_layers(model, cfg, device)
    model_state = _extract_model_state(state)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing or unexpected:
        print("Checkpoint load warnings:")
        if missing:
            print("  Missing:", missing)
        if unexpected:
            print("  Unexpected:", unexpected)

    model.eval()
    return model, cfg


def load_inverse_model_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
) -> Tuple[InverseDynamicsModel, InverseDynamicsConfig]:
    state = _load_checkpoint_state(ckpt_path, device)
    config_dict = _extract_checkpoint_config(state)
    family_reason = inverse_checkpoint_family_mismatch_reason(config_dict)
    if family_reason is not None:
        raise RuntimeError(f"Cannot load inverse checkpoint {ckpt_path}: {family_reason}")
    valid_keys = {field.name for field in fields(InverseDynamicsConfig)}
    filtered = {key: value for key, value in config_dict.items() if key in valid_keys}
    if "output_seq_len" not in filtered:
        legacy_center_only = config_dict.get("predict_center_frame_only")
        if legacy_center_only is True:
            filtered["output_seq_len"] = 1
        else:
            filtered["output_seq_len"] = int(filtered.get("seq_len", InverseDynamicsConfig().seq_len))
    cfg = InverseDynamicsConfig(**filtered)
    if isinstance(state, dict) and "velocity_scales" in state:
        cfg.mouse_velocity_scales = tuple(float(item) for item in state["velocity_scales"])
        cfg.__post_init__()

    model = InverseDynamicsModel(cfg=cfg).to(device)
    initialize_inverse_lazy_layers(model, cfg, device)
    model_state = _extract_model_state(state)
    model.load_state_dict(model_state, strict=True)

    model.eval()
    return model, cfg


def load_visual_model_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
    requested_kind: ModelKind,
) -> Tuple[VisualModel, VisualConfig, LoadedModelKind]:
    state = _load_checkpoint_state(ckpt_path, device)
    detected_kind = _detect_checkpoint_kind(state)
    model_kind: LoadedModelKind = detected_kind if requested_kind == "auto" else str(requested_kind)  # type: ignore[assignment]
    if model_kind == "inverse":
        model, cfg = load_inverse_model_from_checkpoint(ckpt_path, device)
    else:
        model, cfg = load_model_from_checkpoint(ckpt_path, device)
    return model, cfg, model_kind


def _stage_start_conv(module: torch.nn.Module) -> Optional[torch.nn.Conv2d]:
    if isinstance(module, torch.nn.Conv2d):
        conv = module
    else:
        conv = getattr(module, "conv", None)
    if isinstance(conv, torch.nn.Conv2d):
        return conv
    return None


def _activation_stages(net: torch.nn.Sequential, x: torch.Tensor) -> List[torch.Tensor]:
    stages: List[torch.Tensor] = []
    y = x
    saw_stage = False
    for module in net:
        starts_new_stage = _stage_start_conv(module) is not None
        if starts_new_stage and saw_stage:
            stages.append(y)
        if starts_new_stage:
            saw_stage = True
        y = module(y)
    stages.append(y)
    return stages


def _policy_encoder_input(
    model: DrivingVideoPolicy,
    frame_rgb: torch.Tensor,
    previous_frame_rgb: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    del previous_frame_rgb
    if frame_rgb.dim() != 4 or frame_rgb.size(1) != 3:
        raise ValueError(f"Expected current frame [B,3,H,W], got {tuple(frame_rgb.shape)}.")
    frame_rgb = model._normalize_frames(frame_rgb)
    masked_frame = model._apply_masks(frame_rgb)
    x = masked_frame
    if x.is_cuda:
        x = x.contiguous(memory_format=torch.channels_last)
    return x, frame_rgb.detach()


def _inverse_encoder_input(
    frame_rgb: torch.Tensor,
    previous_frame_rgb: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if frame_rgb.dim() != 4 or frame_rgb.size(1) != 3:
        raise ValueError(f"Expected current frame [B,3,H,W], got {tuple(frame_rgb.shape)}.")
    frame_rgb = frame_rgb.clamp(0.0, 1.0)
    frame_scaled = (frame_rgb * 2.0) - 1.0
    if previous_frame_rgb is None:
        motion = torch.zeros_like(frame_scaled)
    else:
        prev_frame_rgb = previous_frame_rgb.to(device=frame_rgb.device, dtype=frame_rgb.dtype).clamp(0.0, 1.0)
        prev_scaled = (prev_frame_rgb * 2.0) - 1.0
        motion = frame_scaled - prev_scaled
    x = torch.cat([frame_scaled, motion], dim=1)
    if x.is_cuda:
        x = x.contiguous(memory_format=torch.channels_last)
    return x, frame_rgb.detach()


def _compute_feature_map(
    model: VisualModel,
    frame_rgb: torch.Tensor,
    previous_frame_rgb: Optional[torch.Tensor],
    layer: FeatureLayer,
    policy_state: Optional[TemporalState] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[TemporalState]]:
    if isinstance(model, DrivingVideoPolicy):
        x, current = _policy_encoder_input(model, frame_rgb, previous_frame_rgb)
        if layer == "motion":
            return x.abs(), current, policy_state
        if hasattr(model.spatial_encoder, "feature_stages"):
            stages = model.spatial_encoder.feature_stages(x)
        else:
            stages = _activation_stages(model.spatial_encoder.net, x)
        if layer.startswith("stage"):
            stage_idx = int(layer.removeprefix("stage")) - 1
            if stage_idx < 0 or stage_idx >= len(stages):
                raise ValueError(f"Policy ConvGRU encoder has {len(stages)} stages; cannot show {layer!r}.")
            return stages[stage_idx], current, policy_state
        if layer == "spatial":
            return stages[-1], current, policy_state
        if layer == "tokens":
            spatial_feat = stages[-1]
            b, _, hf, wf = spatial_feat.shape
            if hasattr(model, "_prepare_temporal_state") and hasattr(model, "_temporal_step"):
                h_t = model._prepare_temporal_state(
                    policy_state,
                    b,
                    hf,
                    wf,
                    device=spatial_feat.device,
                    dtype=spatial_feat.dtype,
                )
                hidden, new_state = model._temporal_step(spatial_feat, h_t)
                return hidden, current, TemporalState(hidden_state=new_state.detach())
            if policy_state is not None and policy_state.hidden_state is not None:
                h_t = policy_state.hidden_state.to(device=spatial_feat.device, dtype=spatial_feat.dtype)
            else:
                h_t = torch.zeros(
                    b,
                    int(model.feat_channels),
                    hf,
                    wf,
                    device=spatial_feat.device,
                    dtype=spatial_feat.dtype,
                )
            hidden = model.temporal_rnn(spatial_feat, h_t)
            return hidden, current, TemporalState(hidden_state=hidden.detach())
        raise ValueError(f"Unknown policy ConvGRU layer {layer!r}.")

    x, current = _inverse_encoder_input(frame_rgb, previous_frame_rgb)
    if layer == "motion":
        return x[:, 3:6].abs(), current, policy_state
    if layer in ("spatial", "tokens"):
        raise ValueError(f"Inverse dynamics CNN does not have a {layer!r} layer; use motion or stage1-stage4.")
    stages = _activation_stages(model.frame_encoder.net, x)
    if layer.startswith("stage"):
        stage_idx = int(layer.removeprefix("stage")) - 1
        if stage_idx < 0 or stage_idx >= len(stages):
            raise ValueError(f"Inverse dynamics CNN has {len(stages)} stages; cannot show {layer!r}.")
        return stages[stage_idx], current, policy_state
    raise ValueError(f"Unknown inverse CNN layer {layer!r}.")


def _feature_to_heat_color(
    feat: torch.Tensor,
    width: int,
    height: int,
    *,
    robust_norm: bool,
    q_low: float,
    q_high: float,
    gamma: float,
) -> np.ndarray:
    heat = torch.linalg.vector_norm(feat.float(), ord=2, dim=1)[0]
    if robust_norm:
        lo = torch.quantile(heat.flatten(), float(q_low))
        hi = torch.quantile(heat.flatten(), float(q_high))
        heat = (heat - lo) / (hi - lo + 1e-6)
    else:
        heat_min = heat.min()
        heat_max = heat.max()
        heat = (heat - heat_min) / (heat_max - heat_min + 1e-6)
    heat = heat.clamp(0.0, 1.0)
    if float(gamma) != 1.0:
        heat = heat.pow(float(gamma))

    heat_np = (heat.detach().cpu().numpy() * 255.0).astype(np.uint8)
    heat_up = cv2.resize(heat_np, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
    return cv2.applyColorMap(heat_up, cv2.COLORMAP_JET)


def _feature_heatmap_for_frame(
    frame_bgr: np.ndarray,
    model: VisualModel,
    device: torch.device,
    *,
    previous_frame_rgb: Optional[torch.Tensor] = None,
    layer: FeatureLayer = "tokens",
    resize_to: Optional[int] = None,
    robust_norm: bool = True,
    q_low: float = 0.05,
    q_high: float = 0.95,
    gamma: float = 0.75,
    amp_dtype: torch.dtype = torch.bfloat16,
    use_autocast: bool = True,
    policy_state: Optional[TemporalState] = None,
) -> Tuple[np.ndarray, torch.Tensor, Optional[TemporalState]]:
    orig_h, orig_w = frame_bgr.shape[:2]
    proc = frame_bgr
    if resize_to is not None and (orig_h != resize_to or orig_w != resize_to):
        proc = cv2.resize(proc, (int(resize_to), int(resize_to)), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    if x.is_cuda:
        x = x.contiguous(memory_format=torch.channels_last)

    with torch.inference_mode():
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_autocast and x.is_cuda):
            feat, current_frame_rgb, policy_state = _compute_feature_map(
                model,
                x,
                previous_frame_rgb,
                layer=layer,
                policy_state=policy_state,
            )

    heat_color = _feature_to_heat_color(
        feat,
        orig_w,
        orig_h,
        robust_norm=robust_norm,
        q_low=q_low,
        q_high=q_high,
        gamma=gamma,
    )
    return heat_color, current_frame_rgb.detach(), policy_state


def _action_index(cfg: ModelConfig, action_name: str) -> Optional[int]:
    wanted = str(action_name).lower()
    for idx, name in enumerate(list(cfg.key_names) + list(cfg.mouse_button_names)):
        normalized = str(name).split(".", 1)[-1].lower()
        if normalized == wanted:
            return idx
    return None


def _button_thresholds(cfg: ModelConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    thresholds = getattr(cfg, "button_state_thresholds", None)
    if thresholds is None or len(tuple(thresholds)) != int(cfg.num_bin):
        thresholds = tuple(float(cfg.button_state_threshold) for _ in range(int(cfg.num_bin)))
    return torch.tensor(tuple(float(value) for value in thresholds), device=device, dtype=dtype)


def _button_threshold_value(cfg: ModelConfig, action_idx: Optional[int]) -> float:
    if action_idx is None:
        return float(getattr(cfg, "button_state_threshold", 0.5))
    thresholds = getattr(cfg, "button_state_thresholds", None)
    if thresholds is not None:
        values = tuple(float(value) for value in thresholds)
        if 0 <= int(action_idx) < len(values):
            return values[int(action_idx)]
    return float(getattr(cfg, "button_state_threshold", 0.5))


def _truthy_action_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "0", "0.0", "false", "none", "no", "off"}:
            return False
        if text in {"1", "1.0", "true", "yes", "on"}:
            return True
        value = text
    try:
        return float(value) > 0.5
    except (TypeError, ValueError):
        return False


def _find_csv_for_video(video_path: str, csv_ext: str = ".csv") -> Optional[str]:
    base, _ = os.path.splitext(video_path)
    candidates = [
        base + csv_ext,
        base.replace("_512", "") + csv_ext,
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _load_real_button_sequence(csv_path: str, cfg: ModelConfig) -> np.ndarray:
    names = list(cfg.key_names) + list(cfg.mouse_button_names)
    with open(csv_path, "r", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {csv_path}")
        fieldnames = set(reader.fieldnames)
        missing = [name for name in names if name not in fieldnames]
        if missing:
            raise RuntimeError(f"CSV schema mismatch for {csv_path}: missing action columns={missing}")
        rows = list(reader)

    buttons = np.zeros((len(rows), int(cfg.num_bin)), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        for col_idx, name in enumerate(names):
            buttons[row_idx, col_idx] = 1.0 if _truthy_action_value(row.get(name)) else 0.0
    return buttons


def _real_trajectory_probs_for_frame(
    buttons: Optional[np.ndarray],
    frame_idx: int,
    cfg: ModelConfig,
    action_label_offset: int,
) -> Optional[np.ndarray]:
    if buttons is None or buttons.size == 0:
        return None
    offsets = tuple(int(offset) for offset in getattr(cfg, "prediction_horizon_offsets", ()))
    if not offsets:
        offsets = tuple(range(1, int(cfg.prediction_horizon) + 1))

    rows: List[np.ndarray] = []
    for offset in offsets:
        target_idx = int(frame_idx) + int(offset) + int(action_label_offset)
        if target_idx < 0 or target_idx >= int(buttons.shape[0]):
            break
        rows.append(buttons[target_idx])
    if len(rows) < min(10, len(offsets)):
        return None
    return np.stack(rows, axis=0).astype(np.float32)


def _policy_visuals_for_frame(
    frame_bgr: np.ndarray,
    model: DrivingVideoPolicy,
    cfg: ModelConfig,
    device: torch.device,
    *,
    state: Optional[TemporalState],
    prev_action: Optional[torch.Tensor],
    layer: FeatureLayer,
    resize_to: Optional[int],
    robust_norm: bool,
    q_low: float,
    q_high: float,
    gamma: float,
    amp_dtype: torch.dtype,
    use_autocast: bool,
    need_trajectory: bool,
) -> Tuple[np.ndarray, torch.Tensor, Optional[TemporalState], Optional[np.ndarray], Optional[torch.Tensor]]:
    orig_h, orig_w = frame_bgr.shape[:2]
    proc = frame_bgr
    if resize_to is not None and (orig_h != resize_to or orig_w != resize_to):
        proc = cv2.resize(proc, (int(resize_to), int(resize_to)), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    if x.is_cuda:
        x = x.contiguous(memory_format=torch.channels_last)

    trajectory_probs: Optional[np.ndarray] = None
    next_action: Optional[torch.Tensor] = prev_action

    with torch.inference_mode():
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast and x.is_cuda):
            frame_rgb = model._normalize_frames(x)
            masked_frame = model._apply_masks(frame_rgb)
            if masked_frame.is_cuda:
                masked_frame = masked_frame.contiguous(memory_format=torch.channels_last)

            stages: Optional[List[torch.Tensor]] = None
            if layer == "motion":
                feat = masked_frame.abs()
            else:
                if hasattr(model.spatial_encoder, "feature_stages"):
                    stages = model.spatial_encoder.feature_stages(masked_frame)
                else:
                    stages = _activation_stages(model.spatial_encoder.net, masked_frame)

                if layer.startswith("stage"):
                    stage_idx = int(layer.removeprefix("stage")) - 1
                    if stage_idx < 0 or stage_idx >= len(stages):
                        raise ValueError(f"Policy ConvGRU encoder has {len(stages)} stages; cannot show {layer!r}.")
                    feat = stages[stage_idx]
                elif layer == "spatial":
                    feat = stages[-1]
                elif layer == "tokens":
                    feat = None
                else:
                    raise ValueError(f"Unknown policy ConvGRU layer {layer!r}.")

            needs_temporal = bool(need_trajectory) or layer == "tokens"
            if needs_temporal:
                if stages is None:
                    if hasattr(model.spatial_encoder, "feature_stages"):
                        stages = model.spatial_encoder.feature_stages(masked_frame)
                    else:
                        stages = _activation_stages(model.spatial_encoder.net, masked_frame)
                spatial_feat = stages[-1]
                b, _, hf, wf = spatial_feat.shape
                h_t = model._prepare_temporal_state(
                    state,
                    b,
                    hf,
                    wf,
                    device=spatial_feat.device,
                    dtype=spatial_feat.dtype,
                )
                temporal_feat, new_hidden = model._temporal_step(spatial_feat, h_t)
                state = TemporalState(hidden_state=new_hidden.detach())
                if layer == "tokens":
                    feat = temporal_feat

                if bool(need_trajectory):
                    pooled = model.pool(temporal_feat)
                    visual_feat = model.fc_features(pooled.reshape(b, -1))
                    if prev_action is None:
                        prev_for_head = torch.zeros((b, int(cfg.num_bin)), device=visual_feat.device, dtype=visual_feat.dtype)
                    else:
                        prev_for_head = prev_action.to(device=visual_feat.device, dtype=visual_feat.dtype)
                    action_feat = model._last_action_features(
                        prev_for_head,
                        b,
                        1,
                        device=visual_feat.device,
                        dtype=visual_feat.dtype,
                    ).reshape(b, cfg.d_model)
                    fc_out = model.head_fusion(torch.cat([visual_feat, action_feat], dim=-1))
                    logits = model._horizon_button_logits(fc_out)[0].detach().float()
                    trajectory_probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
                    command_idx = min(int(cfg.prediction_horizon) - 1, 9)
                    thresholds = _button_thresholds(cfg, device=logits.device, dtype=logits.dtype)
                    next_action = (
                        torch.sigmoid(logits[command_idx]) >= thresholds
                    ).to(dtype=visual_feat.dtype).reshape(1, int(cfg.num_bin)).detach()

            if feat is None:
                raise RuntimeError(f"Policy layer {layer!r} did not produce a feature map.")

    heat_color = _feature_to_heat_color(
        feat,
        orig_w,
        orig_h,
        robust_norm=robust_norm,
        q_low=q_low,
        q_high=q_high,
        gamma=gamma,
    )
    return heat_color, frame_rgb.detach(), state, trajectory_probs, next_action


def _trajectory_points(
    probs: np.ndarray,
    cfg: ModelConfig,
    width: int,
    height: int,
    *,
    x_offset_px: float = 0.0,
) -> Optional[np.ndarray]:
    if probs.ndim != 2 or probs.shape[0] < 10:
        return None

    offsets = tuple(int(offset) for offset in getattr(cfg, "prediction_horizon_offsets", ()))
    if len(offsets) != probs.shape[0]:
        offsets = tuple(range(1, probs.shape[0] + 1))
    max_offset = max(float(offsets[-1]), 1.0)

    a_idx = _action_index(cfg, "a")
    d_idx = _action_index(cfg, "d")
    w_idx = _action_index(cfg, "w")
    s_idx = _action_index(cfg, "s")

    base_x = float(width) * 0.5 + float(x_offset_px)
    base_y = float(height) * 0.48
    horizon_y = float(height) * 0.24
    reverse_span = float(height) * 0.14
    forward_depth = 0.0
    reverse_depth = 0.0
    lateral_world = 0.0
    heading = 0.0
    previous_offset = 0
    points = [(base_x, base_y)]

    w_threshold = _button_threshold_value(cfg, w_idx)
    s_threshold = _button_threshold_value(cfg, s_idx)
    a_threshold = _button_threshold_value(cfg, a_idx)
    d_threshold = _button_threshold_value(cfg, d_idx)

    first_stop_idx: Optional[int] = None
    if s_idx is not None:
        stop_hits = np.flatnonzero(probs[: len(offsets), s_idx] >= s_threshold)
        if stop_hits.size > 0:
            first_stop_idx = int(stop_hits[0])
            if first_stop_idx == 0:
                return None

    for horizon_idx, offset in enumerate(offsets):
        if first_stop_idx is not None and horizon_idx > first_stop_idx:
            break

        step_frac = max(float(offset - previous_offset) / max_offset, 0.0)
        previous_offset = int(offset)
        is_stop_horizon = first_stop_idx is not None and horizon_idx == first_stop_idx

        right_intent = float(probs[horizon_idx, d_idx]) if d_idx is not None else 0.0
        left_intent = float(probs[horizon_idx, a_idx]) if a_idx is not None else 0.0
        right_active = max(right_intent - d_threshold, 0.0) / max(1.0 - d_threshold, 1e-3)
        left_active = max(left_intent - a_threshold, 0.0) / max(1.0 - a_threshold, 1e-3)
        threshold_steer = right_active - left_active
        raw_steer = right_intent - left_intent
        steer = threshold_steer if abs(threshold_steer) > abs(raw_steer) else raw_steer
        steer = min(max(steer, -1.0), 1.0)

        forward_intent = float(probs[horizon_idx, w_idx]) if w_idx is not None else 0.0
        reverse_intent = float(probs[horizon_idx, s_idx]) if s_idx is not None else 0.0
        coasting_forward = (
            (w_idx is None or forward_intent < w_threshold)
            and (s_idx is None or reverse_intent < s_threshold)
        )
        if is_stop_horizon:
            signed_drive = max(forward_intent, 0.42)
        elif coasting_forward:
            signed_drive = 0.42
        elif w_idx is None and s_idx is None:
            signed_drive = 0.55
        else:
            signed_drive = forward_intent - reverse_intent
        signed_drive = min(max(signed_drive, -1.0), 1.0)

        travel = signed_drive * step_frac
        motion_strength = min(max(abs(signed_drive), 0.18), 1.0)
        heading += steer * step_frac * 2.3 * motion_strength
        if travel >= 0.0:
            forward_depth += travel
            reverse_depth = max(0.0, reverse_depth - step_frac * 0.65)
        else:
            reverse_depth += -travel
            forward_depth = max(0.0, forward_depth + travel * 0.35)

        forward_depth = min(max(forward_depth, 0.0), 1.0)
        reverse_depth = min(max(reverse_depth, 0.0), 0.75)
        lateral_world += (
            np.sin(heading) * abs(travel) * 1.35
            + steer * step_frac * 0.72 * motion_strength
        )
        lateral_world = min(max(lateral_world, -1.25), 1.25)

        if forward_depth >= reverse_depth:
            ground_progress = forward_depth / (forward_depth + 0.34)
            y = base_y - (base_y - horizon_y) * ground_progress
            perspective = 0.28 + 1.18 * ground_progress
            x = base_x + lateral_world * float(width) * 0.23 * perspective
        else:
            reverse_progress = reverse_depth / (reverse_depth + 0.38)
            y = base_y + reverse_span * reverse_progress
            perspective = 0.55 + 0.45 * reverse_progress
            x = base_x + lateral_world * float(width) * 0.14 * perspective

        x = min(max(x, float(width) * 0.14), float(width) * 0.86)
        y = min(max(y, float(height) * 0.18), float(height) * 0.86)
        points.append((x, y))

    return np.asarray(points, dtype=np.float32)


def _sample_polyline(points: np.ndarray, sample_count: int = 96) -> np.ndarray:
    if points.shape[0] < 2:
        return points
    deltas = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    distance = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = float(distance[-1])
    if total <= 1e-3:
        return points

    samples = np.linspace(0.0, total, max(2, int(sample_count)), dtype=np.float32)
    x = np.interp(samples, distance, points[:, 0])
    y = np.interp(samples, distance, points[:, 1])
    sampled = np.stack([x, y], axis=1).astype(np.float32)
    if sampled.shape[0] >= 7:
        kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
        kernel /= float(kernel.sum())
        padded = np.pad(sampled, ((2, 2), (0, 0)), mode="edge")
        smoothed = np.stack(
            [
                np.convolve(padded[:, axis], kernel, mode="valid")
                for axis in range(2)
            ],
            axis=1,
        ).astype(np.float32)
        smoothed[0] = sampled[0]
        smoothed[-1] = sampled[-1]
        return smoothed
    return sampled


def _ribbon_polygon(centerline: np.ndarray, widths: np.ndarray) -> np.ndarray:
    tangents = np.gradient(centerline, axis=0)
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-3)
    normals = np.concatenate([-tangents[:, 1:2], tangents[:, 0:1]], axis=1) / lengths
    half_width = widths.reshape(-1, 1) * 0.5
    left = centerline + normals * half_width
    right = centerline - normals * half_width
    return np.round(np.concatenate([left, right[::-1]], axis=0)).astype(np.int32)


def _blend_ribbon(
    image: np.ndarray,
    centerline: np.ndarray,
    widths: np.ndarray,
    *,
    color: Tuple[int, int, int],
    alpha: float,
    blur: int = 0,
) -> None:
    polygon = _ribbon_polygon(centerline, widths)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255, lineType=cv2.LINE_AA)
    if blur > 0:
        blur = int(blur) | 1
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)

    alpha_map = (mask.astype(np.float32) / 255.0) * float(alpha)
    if not bool(np.any(alpha_map > 0.0)):
        return
    image_float = image.astype(np.float32)
    color_value = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    image[:] = np.clip(
        image_float * (1.0 - alpha_map[..., None]) + color_value * alpha_map[..., None],
        0.0,
        255.0,
    ).astype(np.uint8)


def _draw_policy_trajectory(
    image: np.ndarray,
    probs: Optional[np.ndarray],
    cfg: ModelConfig,
    *,
    alpha: float = 0.42,
    color: Tuple[int, int, int] = (255, 135, 24),
    x_offset_px: float = 0.0,
) -> np.ndarray:
    if probs is None:
        return image
    height, width = image.shape[:2]
    points = _trajectory_points(probs, cfg, width, height, x_offset_px=x_offset_px)
    if points is None or points.shape[0] < 2:
        return image

    centerline = _sample_polyline(points)
    count = int(centerline.shape[0])
    progress = np.linspace(0.0, 1.0, count, dtype=np.float32)
    widths = (
        max(34.0, width * 0.038 + 10.0) * (1.0 - progress)
        + max(17.0, width * 0.010 + 10.0) * progress
    )

    _blend_ribbon(
        image,
        centerline,
        widths,
        color=color,
        alpha=float(alpha),
        blur=3,
    )

    return image


def _default_output_path(input_path: str, layer: str, mode: str, model_kind: LoadedModelKind) -> str:
    base, _ = os.path.splitext(input_path)
    return f"{base}_visualized_{model_kind}_cnn_{layer}_{mode}.mp4"


def process_video_cnn(
    input_path: str,
    output_path: str,
    model: VisualModel,
    device: torch.device,
    *,
    model_kind: LoadedModelKind,
    layer: FeatureLayer = "tokens",
    mode: Literal["heat", "overlay", "side_by_side", "triple"] = "side_by_side",
    alpha: float = 0.5,
    resize_to: Optional[int] = None,
    robust_norm: bool = True,
    q_low: float = 0.05,
    q_high: float = 0.95,
    gamma: float = 0.75,
    amp_dtype: torch.dtype = torch.bfloat16,
    use_autocast: bool = True,
    ffmpeg_path: Optional[str] = None,
    ffmpeg_codec: str = "hevc_nvenc",
    ffmpeg_preset: str = "p4",
    ffmpeg_crf: int = 20,
    max_frames: Optional[int] = None,
    draw_trajectory: bool = True,
    draw_real_trajectory: bool = True,
    real_csv_path: Optional[str] = None,
    action_label_offset: int = 0,
    trajectory_alpha: float = 0.42,
) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_w = width
    if mode == "side_by_side":
        out_w = width * 2
    elif mode == "triple":
        out_w = width * 3
    out_h = height

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = FFmpegWriter(
        out_path=output_path,
        fps=fps,
        width=out_w,
        height=out_h,
        ffmpeg_path=ffmpeg_path,
        codec=ffmpeg_codec,
        preset=ffmpeg_preset,
        quality=int(ffmpeg_crf),
    )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total_frames = min(max(0, total_frames), max(0, int(max_frames)))
    pbar = tqdm(total=max(0, total_frames), desc=f"CNN vis [{model_kind}] ({layer}, {mode})")
    frame_idx = 0
    previous_frame_rgb: Optional[torch.Tensor] = None
    policy_state: Optional[TemporalState] = None
    trajectory_prev_action: Optional[torch.Tensor] = None
    trajectory_cfg = model.cfg if isinstance(model, DrivingVideoPolicy) else None
    trajectory_enabled = (
        bool(draw_trajectory)
        and model_kind == "policy"
        and isinstance(model, DrivingVideoPolicy)
        and trajectory_cfg is not None
        and int(getattr(trajectory_cfg, "prediction_horizon", 0)) >= 10
    )
    real_buttons: Optional[np.ndarray] = None
    if trajectory_enabled and bool(draw_real_trajectory) and trajectory_cfg is not None:
        resolved_csv_path = real_csv_path or _find_csv_for_video(
            input_path,
            csv_ext=str(getattr(trajectory_cfg, "csv_ext", ".csv")),
        )
        if resolved_csv_path is not None:
            real_buttons = _load_real_button_sequence(resolved_csv_path, trajectory_cfg)
        else:
            print("[CNN] No matching CSV found; left-side real trajectory will be omitted.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            trajectory_probs: Optional[np.ndarray] = None
            real_trajectory_probs: Optional[np.ndarray] = None
            if isinstance(model, DrivingVideoPolicy) and trajectory_cfg is not None:
                heat_color, previous_frame_rgb, policy_state, trajectory_probs, trajectory_prev_action = (
                    _policy_visuals_for_frame(
                        frame,
                        model,
                        trajectory_cfg,
                        device,
                        state=policy_state,
                        prev_action=trajectory_prev_action,
                        layer=layer,
                        resize_to=resize_to,
                        robust_norm=robust_norm,
                        q_low=q_low,
                        q_high=q_high,
                        gamma=gamma,
                        amp_dtype=amp_dtype,
                        use_autocast=use_autocast,
                        need_trajectory=trajectory_enabled,
                    )
                )
                if trajectory_enabled:
                    real_trajectory_probs = _real_trajectory_probs_for_frame(
                        real_buttons,
                        frame_idx,
                        trajectory_cfg,
                        action_label_offset,
                    )
            else:
                heat_color, previous_frame_rgb, policy_state = _feature_heatmap_for_frame(
                    frame,
                    model,
                    device,
                    previous_frame_rgb=previous_frame_rgb,
                    layer=layer,
                    resize_to=resize_to,
                    robust_norm=robust_norm,
                    q_low=q_low,
                    q_high=q_high,
                    gamma=gamma,
                    amp_dtype=amp_dtype,
                    use_autocast=use_autocast,
                    policy_state=policy_state,
                )

            overlay = cv2.addWeighted(frame, 1.0 - alpha, heat_color, alpha, 0.0)
            display_frame = frame

            if trajectory_enabled and trajectory_probs is not None and trajectory_cfg is not None:
                if mode in {"side_by_side", "triple"}:
                    display_frame = display_frame.copy()
                    if real_trajectory_probs is not None:
                        display_frame = _draw_policy_trajectory(
                            display_frame,
                            real_trajectory_probs,
                            trajectory_cfg,
                            alpha=trajectory_alpha,
                            color=(0, 190, 255),
                            x_offset_px=-float(width) * 0.045,
                        )
                    display_frame = _draw_policy_trajectory(
                        display_frame,
                        trajectory_probs,
                        trajectory_cfg,
                        alpha=trajectory_alpha,
                        x_offset_px=float(width) * 0.045,
                    )
                overlay = _draw_policy_trajectory(
                    overlay,
                    trajectory_probs,
                    trajectory_cfg,
                    alpha=trajectory_alpha,
                )

            if mode == "heat":
                out_frame = _draw_policy_trajectory(
                    heat_color,
                    trajectory_probs,
                    trajectory_cfg,
                    alpha=trajectory_alpha,
                ) if trajectory_probs is not None and trajectory_cfg is not None else heat_color
            elif mode == "overlay":
                out_frame = overlay
            elif mode == "side_by_side":
                out_frame = np.concatenate([display_frame, overlay], axis=1)
            elif mode == "triple":
                out_frame = np.concatenate([display_frame, heat_color, overlay], axis=1)
            else:
                raise ValueError(f"Unknown mode {mode!r}")

            writer.write(out_frame)
            frame_idx += 1
            pbar.update(1)
            if max_frames is not None and frame_idx >= int(max_frames):
                break
    finally:
        pbar.close()
        writer.release()
        cap.release()

    print(f"[CNN] Done. Wrote {frame_idx} frames to {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize feature-energy heatmaps from the current CNN encoders.")
    parser.add_argument("--input", default=r'C:\Users\Abhil\Desktop\Github_Projects\VideoAgent\data\greenville\run_20260527_134734.mp4', help="Input video path. Defaults to the newest run in cfg.data_root.")
    parser.add_argument("--output", default=None, help="Output video path (.mp4). Default auto-names next to input.")
    parser.add_argument("--csv", default=None, help="CSV labels for the input video. Default auto-detects next to --input.")
    parser.add_argument(
        "--model-kind",
        choices=["auto", "policy", "inverse"],
        default="auto",
        help="Checkpoint type. Default auto-detects from the checkpoint config/state dict.",
    )
    parser.add_argument(
        "--ckpt-path",
        default=r'C:\Users\Abhil\Desktop\Github_Projects\VideoAgent\checkpoints_rt\model_latest.pt',
        help="Checkpoint path. If omitted, auto-select from --ckpt-dir.",
    )
    parser.add_argument(
        "--ckpt-dir",
        default="./checkpoints_rt",
        help="Checkpoint directory used when --ckpt-path is omitted.",
    )
    parser.add_argument(
        "--layer",
        choices=["motion", "stage1", "stage2", "stage3", "stage4", "spatial", "tokens"],
        default="tokens",
        help=(
            "CNN signal to visualize. Policy checkpoints use masked RGB input for motion, stage1-stage4 for "
            "the spatial encoder blocks, final spatial features for spatial, and ConvGRU hidden features for tokens. "
            "Inverse checkpoints support "
            "motion and stage1-stage4."
        ),
    )
    parser.add_argument("--mode", choices=["heat", "overlay", "side_by_side", "triple"], default="side_by_side")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay blend factor.")
    parser.add_argument(
        "--resize-to",
        type=int,
        default=None,
        help="Resize frames to this square size before encoding. Defaults to model_size from checkpoint config.",
    )
    parser.add_argument("--q-low", type=float, default=0.05)
    parser.add_argument("--q-high", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.75)
    parser.add_argument("--no-robust-norm", action="store_true", help="Use min/max normalization instead of quantiles.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 autocast on CUDA instead of bf16.")
    parser.add_argument("--no-amp", action="store_true", help="Disable autocast during encoder inference.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    parser.add_argument("--max-frames", type=int, default=5000, help="Stop after this many frames.")
    parser.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Disable the FSD-style policy trajectory overlay.",
        default=False,
    )
    parser.add_argument(
        "--no-real-trajectory",
        action="store_true",
        help="Disable the left-side ground-truth trajectory in side-by-side output.",
    )
    parser.add_argument(
        "--action-label-offset",
        type=int,
        default=0,
        help="Frame offset applied when reading real future labels from CSV.",
    )
    parser.add_argument(
        "--trajectory-alpha",
        type=float,
        default=0.42,
        help="Blend strength for the policy trajectory ribbon.",
    )
    parser.add_argument("--ffmpeg-path", default=None, help="Path to ffmpeg binary. Default uses PATH lookup.")
    parser.add_argument("--ffmpeg-codec", default="hevc_nvenc")
    parser.add_argument("--ffmpeg-preset", default="p4")
    parser.add_argument("--ffmpeg-crf", type=int, default=20)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    ckpt_path = args.ckpt_path or _find_latest_checkpoint(args.ckpt_dir)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint found. Set --ckpt-path or provide a valid --ckpt-dir (got {args.ckpt_dir!r})."
        )
    print(f"Loading checkpoint: {ckpt_path}")

    model, cfg, model_kind = load_visual_model_from_checkpoint(
        ckpt_path,
        device=device,
        requested_kind=str(args.model_kind),
    )
    print(f"Resolved checkpoint type: {model_kind}")
    resize_to = int(args.resize_to) if args.resize_to is not None else int(cfg.model_size)

    if args.input is not None:
        input_path = args.input
    else:
        candidates = sorted(glob.glob(os.path.join(cfg.data_root, f"run_*{cfg.video_ext}")))
        if not candidates:
            raise FileNotFoundError("No input video provided and no runs found in cfg.data_root.")
        input_path = candidates[-1]

    output_path = args.output or _default_output_path(input_path, args.layer, args.mode, model_kind)
    amp_dtype = torch.float16 if bool(args.fp16) else torch.bfloat16
    use_autocast = (not bool(args.no_amp)) and device.type == "cuda"

    process_video_cnn(
        input_path=input_path,
        output_path=output_path,
        model=model,
        device=device,
        model_kind=model_kind,
        layer=args.layer,
        mode=args.mode,
        alpha=float(args.alpha),
        resize_to=resize_to,
        robust_norm=not bool(args.no_robust_norm),
        q_low=float(args.q_low),
        q_high=float(args.q_high),
        gamma=float(args.gamma),
        amp_dtype=amp_dtype,
        use_autocast=use_autocast,
        ffmpeg_path=args.ffmpeg_path,
        ffmpeg_codec=str(args.ffmpeg_codec),
        ffmpeg_preset=str(args.ffmpeg_preset),
        ffmpeg_crf=int(args.ffmpeg_crf),
        max_frames=None if args.max_frames is None else int(args.max_frames),
        draw_trajectory=not bool(args.no_trajectory),
        draw_real_trajectory=not bool(args.no_real_trajectory),
        real_csv_path=None if args.csv is None else str(args.csv),
        action_label_offset=int(args.action_label_offset),
        trajectory_alpha=float(args.trajectory_alpha),
    )


if __name__ == "__main__":
    main()
