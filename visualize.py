import argparse
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
    ActionConditionedVideoPolicy,
    ModelConfig,
    get_key_names as MODEL_GET_KEY_NAMES,
    get_mouse_button_names as MODEL_GET_MOUSE_BUTTON_NAMES,
)


FeatureLayer = Literal["motion", "stage1", "stage2", "stage3", "stage4", "spatial", "tokens"]
ModelKind = Literal["auto", "policy", "inverse"]
LoadedModelKind = Literal["policy", "inverse"]
VisualModel = ActionConditionedVideoPolicy | InverseDynamicsModel
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
    cfg.max_context = int(cfg.max_context)
    cfg.prediction_dt = float(cfg.prediction_dt)
    cfg.prediction_horizon = int(getattr(cfg, "prediction_horizon", 1))
    cfg.d_model = int(cfg.d_model)
    cfg.frame_spatial_pool = int(cfg.frame_spatial_pool)
    cfg.frame_spatial_channels = int(cfg.frame_spatial_channels)
    cfg.spatial_attention_tokens = int(cfg.spatial_attention_tokens)
    cfg.spatial_attention_heads = int(cfg.spatial_attention_heads)
    cfg.spatial_temporal_grid = int(cfg.spatial_temporal_grid)
    cfg.temporal_layers = int(cfg.temporal_layers)
    cfg.temporal_heads = int(cfg.temporal_heads)
    cfg.encode_chunk_size = int(cfg.encode_chunk_size)
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
    policy_markers = {
        "prediction_horizon",
        "frame_spatial_pool",
        "frame_spatial_channels",
        "spatial_attention_tokens",
        "spatial_temporal_grid",
        "max_context",
    }
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
    model: ActionConditionedVideoPolicy,
    cfg: ModelConfig,
    device: torch.device,
) -> None:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy_frames = torch.zeros(1, 1, 3, cfg.model_size, cfg.model_size, device=device)
        dummy_dt = torch.full((1, 1), float(cfg.prediction_dt), device=device)
        _ = model(dummy_frames, dt=dummy_dt)
    model.train(was_training)


def load_model_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
) -> Tuple[ActionConditionedVideoPolicy, ModelConfig]:
    state = _load_checkpoint_state(ckpt_path, device)
    cfg = ModelConfig()
    config_dict = _extract_checkpoint_config(state)
    if config_dict:
        cfg = _apply_config_overrides(cfg, config_dict)
    else:
        cfg = _coerce_config_types(cfg)

    model = ActionConditionedVideoPolicy(cfg=cfg).to(device)
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


def _activation_stages(net: torch.nn.Sequential, x: torch.Tensor) -> List[torch.Tensor]:
    stages: List[torch.Tensor] = []
    y = x
    saw_stage = False
    for module in net:
        conv = getattr(module, "conv", None)
        starts_new_stage = isinstance(conv, torch.nn.Conv2d) and tuple(conv.stride) == (2, 2)
        if starts_new_stage and saw_stage:
            stages.append(y)
        if starts_new_stage:
            saw_stage = True
        y = module(y)
    stages.append(y)
    return stages


def _policy_encoder_input(
    model: ActionConditionedVideoPolicy,
    frame_rgb: torch.Tensor,
    previous_frame_rgb: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if frame_rgb.dim() != 4 or frame_rgb.size(1) != 3:
        raise ValueError(f"Expected current frame [B,3,H,W], got {tuple(frame_rgb.shape)}.")
    b, _, h, w = frame_rgb.shape
    frame_rgb = model._normalize_frames(frame_rgb)
    if previous_frame_rgb is None:
        motion = torch.zeros_like(frame_rgb)
    else:
        motion = frame_rgb - previous_frame_rgb.to(device=frame_rgb.device, dtype=frame_rgb.dtype)
    coords = model._coord_channels(
        batch=b,
        steps=1,
        height=h,
        width=w,
        device=frame_rgb.device,
        dtype=frame_rgb.dtype,
    )[:, 0]
    x = torch.cat([frame_rgb, motion, coords], dim=1)
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(model, ActionConditionedVideoPolicy):
        x, current = _policy_encoder_input(model, frame_rgb, previous_frame_rgb)
        if layer == "motion":
            return x[:, 3:6].abs(), current
        stages = _activation_stages(model.frame_encoder.net, x)
        if layer.startswith("stage"):
            stage_idx = int(layer.removeprefix("stage")) - 1
            if stage_idx < 0 or stage_idx >= len(stages):
                raise ValueError(f"Policy CNN has {len(stages)} stages; cannot show {layer!r}.")
            return stages[stage_idx], current
        spatial = model.frame_encoder.spatial_proj(stages[-1])
        if layer == "spatial":
            return spatial, current
        if layer == "tokens":
            return torch.nn.functional.adaptive_avg_pool2d(
                spatial,
                output_size=(model.frame_encoder.pool_size, model.frame_encoder.pool_size),
            ), current
        raise ValueError(f"Unknown policy CNN layer {layer!r}.")

    x, current = _inverse_encoder_input(frame_rgb, previous_frame_rgb)
    if layer == "motion":
        return x[:, 3:6].abs(), current
    if layer in ("spatial", "tokens"):
        raise ValueError(f"Inverse dynamics CNN does not have a {layer!r} layer; use motion or stage1-stage4.")
    stages = _activation_stages(model.frame_encoder.net, x)
    if layer.startswith("stage"):
        stage_idx = int(layer.removeprefix("stage")) - 1
        if stage_idx < 0 or stage_idx >= len(stages):
            raise ValueError(f"Inverse dynamics CNN has {len(stages)} stages; cannot show {layer!r}.")
        return stages[stage_idx], current
    raise ValueError(f"Unknown inverse CNN layer {layer!r}.")


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
) -> Tuple[np.ndarray, torch.Tensor]:
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
            feat, current_frame_rgb = _compute_feature_map(model, x, previous_frame_rgb, layer=layer)

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
    heat_up = cv2.resize(heat_np, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    return cv2.applyColorMap(heat_up, cv2.COLORMAP_JET), current_frame_rgb.detach()


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
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            heat_color, previous_frame_rgb = _feature_heatmap_for_frame(
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
            )
            overlay = cv2.addWeighted(frame, 1.0 - alpha, heat_color, alpha, 0.0)

            if mode == "heat":
                out_frame = heat_color
            elif mode == "overlay":
                out_frame = overlay
            elif mode == "side_by_side":
                out_frame = np.concatenate([frame, overlay], axis=1)
            elif mode == "triple":
                out_frame = np.concatenate([frame, heat_color, overlay], axis=1)
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
    parser.add_argument("--input", default=r'C:\Users\Abhil\Desktop\Github_Projects\VideoAgent\data\greenville\run_20260519_214809.mp4', help="Input video path. Defaults to the newest run in cfg.data_root.")
    parser.add_argument("--output", default=None, help="Output video path (.mp4). Default auto-names next to input.")
    parser.add_argument(
        "--model-kind",
        choices=["auto", "policy", "inverse"],
        default="auto",
        help="Checkpoint type. Default auto-detects from the checkpoint config/state dict.",
    )
    parser.add_argument(
        "--ckpt-path",
        default=None,
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
        default="spatial",
        help=(
            "CNN signal to visualize. Policy checkpoints support all choices; inverse checkpoints support "
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
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after this many frames.")
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
    )


if __name__ == "__main__":
    main()
