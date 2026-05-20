import argparse
import csv
import os
import shutil
import subprocess
from dataclasses import fields
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from models import ActionConditionedVideoPolicy, ModelConfig


class FFmpegPipeWriter:
    def __init__(
        self,
        out_path: str,
        fps: float,
        width: int,
        height: int,
        *,
        ffmpeg_path: Optional[str],
        codec: str,
        quality: int,
        preset: str,
    ) -> None:
        self.out_path = out_path if out_path.lower().endswith(".mp4") else f"{out_path}.mp4"
        self.ffmpeg_path = ffmpeg_path or shutil.which("ffmpeg")
        if not self.ffmpeg_path:
            raise RuntimeError("FFmpeg not found. Add ffmpeg to PATH or pass --ffmpeg-path.")

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{int(width)}x{int(height)}",
            "-r",
            str(float(fps)),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            str(codec),
        ]
        if "nvenc" in str(codec):
            cmd.extend(["-cq", str(int(quality)), "-preset", str(preset)])
        else:
            cmd.extend(["-crf", str(int(quality)), "-preset", str(preset)])
        cmd.extend(["-pix_fmt", "yuv420p", self.out_path])

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write(self, frame_bgr: np.ndarray) -> None:
        if self.proc.stdin is None or self.proc.stdin.closed:
            raise RuntimeError("FFmpeg writer stdin is closed.")
        self.proc.stdin.write(frame_bgr.tobytes())

    def release(self) -> None:
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.flush()
                self.proc.stdin.close()
            except Exception:
                pass
        self.proc.wait(timeout=10)
        if self.proc.returncode not in (0, None):
            raise RuntimeError(f"FFmpeg writer failed with exit code {self.proc.returncode}.")


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return 1.0 / (1.0 + np.exp(-x))


def _parse_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _default_checkpoint(ckpt_dir: str) -> str:
    best_path = os.path.join(ckpt_dir, "model_best.pt")
    if os.path.exists(best_path):
        return best_path
    latest_path = os.path.join(ckpt_dir, "model_latest.pt")
    if os.path.exists(latest_path):
        return latest_path
    raise FileNotFoundError(f"Expected checkpoint at {best_path!r} or {latest_path!r}.")


def _default_video(data_root: str, video_ext: str) -> str:
    candidates = sorted(
        path
        for path in os.listdir(data_root)
        if path.startswith("run_") and path.endswith(video_ext) and not path.endswith(f"_keys{video_ext}")
    )
    if not candidates:
        raise FileNotFoundError(f"No run_*{video_ext} videos found under {data_root!r}.")
    return os.path.join(data_root, candidates[0])


def load_model_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    *,
    use_checkpoint_thresholds: bool,
) -> Tuple[torch.nn.Module, ModelConfig, Dict[str, object]]:
    state = torch.load(checkpoint_path, map_location=device)
    if not isinstance(state, dict) or not isinstance(state.get("config"), dict):
        raise RuntimeError("Checkpoint must contain a config dict.")
    if not isinstance(state.get("model_state"), dict):
        raise RuntimeError("Checkpoint must contain a model_state dict.")

    config_dict = dict(state["config"])
    model_state = state["model_state"]
    valid_keys = {field.name for field in fields(ModelConfig)}
    cfg_kwargs = {key: value for key, value in config_dict.items() if key in valid_keys}
    if not use_checkpoint_thresholds:
        cfg_kwargs.pop("button_state_thresholds", None)
    cfg = ModelConfig(**cfg_kwargs)
    if "mouse_velocity_scales" in state:
        cfg.mouse_velocity_scales = tuple(float(x) for x in state["mouse_velocity_scales"])
        cfg.__post_init__()

    model = ActionConditionedVideoPolicy(cfg).to(device)
    load_result = model.load_state_dict(model_state, strict=False)
    unexpected = [
        key
        for key in load_result.unexpected_keys
        if not key.startswith("mouse_active_head.") and not key.startswith("mouse_delta_head.")
    ]
    if load_result.missing_keys or unexpected:
        raise RuntimeError(
            "Checkpoint does not match the current policy model. "
            f"missing={load_result.missing_keys} unexpected={unexpected}"
        )
    model.eval()
    return model, cfg, config_dict


def load_ground_truth(csv_path: str, cfg: ModelConfig, total_frames: int) -> Dict[str, np.ndarray]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Ground-truth CSV not found: {csv_path}")

    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        fieldnames = set(reader.fieldnames or [])
        required = ["timestamp", *list(cfg.key_names), *list(cfg.mouse_button_names), "delta_x", "delta_y"]
        missing = [name for name in required if name not in fieldnames]
        if missing:
            raise RuntimeError(f"CSV schema mismatch for {csv_path}: missing columns={missing}")
        rows.extend(reader)

    count = min(int(total_frames), len(rows))
    buttons = np.zeros((total_frames, cfg.num_bin), dtype=np.float32)
    mouse_delta = np.zeros((total_frames, 2), dtype=np.float32)
    dt = np.full((total_frames,), float(cfg.prediction_dt), dtype=np.float32)
    valid = np.zeros((total_frames,), dtype=bool)

    for idx in range(count):
        row = rows[idx]
        col = 0
        for name in cfg.key_names:
            buttons[idx, col] = 1.0 if _parse_float(row.get(name)) > 0.5 else 0.0
            col += 1
        for name in cfg.mouse_button_names:
            buttons[idx, col] = 1.0 if _parse_float(row.get(name)) > 0.5 else 0.0
            col += 1
        mouse_delta[idx, 0] = _parse_float(row.get("delta_x"))
        mouse_delta[idx, 1] = _parse_float(row.get("delta_y"))
        if "dt" in row:
            dt[idx] = np.clip(_parse_float(row.get("dt"), cfg.prediction_dt), 1.0 / 240.0, 0.5)
        valid[idx] = True

    return {
        "button_state": buttons,
        "mouse_delta": mouse_delta,
        "dt": dt,
        "valid_mask": valid,
    }


def _resolve_thresholds(cfg: ModelConfig, threshold: Optional[float]) -> np.ndarray:
    if threshold is not None:
        return np.full((cfg.num_bin,), float(threshold), dtype=np.float32)
    return np.asarray(list(cfg.button_state_thresholds), dtype=np.float32)


def infer_video(
    video_path: str,
    model: torch.nn.Module,
    cfg: ModelConfig,
    gt: Dict[str, np.ndarray],
    device: torch.device,
    *,
    command_horizon: int,
    action_label_offset: int,
    max_frames: Optional[int],
    use_autocast: bool,
    inference_dtype: torch.dtype,
) -> Dict[str, np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        total_frames = int(gt["button_state"].shape[0])
    total_frames = min(total_frames, int(gt["button_state"].shape[0]))
    if max_frames is not None and max_frames > 0:
        total_frames = min(total_frames, int(max_frames))

    predictions = {
        "button_logits": np.zeros((total_frames, cfg.num_bin), dtype=np.float32),
        "press_logits": np.zeros((total_frames, cfg.num_bin), dtype=np.float32),
        "release_logits": np.zeros((total_frames, cfg.num_bin), dtype=np.float32),
        "mouse_active_logits": np.zeros((total_frames, 1), dtype=np.float32),
        "mouse_delta": np.zeros((total_frames, 2), dtype=np.float32),
        "source_frame": np.full((total_frames,), -1, dtype=np.int32),
        "valid_mask": np.zeros((total_frames,), dtype=bool),
    }

    state = model.init_state(batch_size=1, device=device, dtype=inference_dtype)
    h_idx = max(0, min(int(command_horizon) - 1, int(cfg.prediction_horizon) - 1))
    target_offset = h_idx + 1 + int(action_label_offset)

    try:
        pbar = tqdm(total=total_frames, desc="Infer", unit="frame")
        for source_idx in range(total_frames):
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (cfg.model_size, cfg.model_size), interpolation=cv2.INTER_AREA)
            frame = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous()
            frame = frame.to(device, non_blocking=True)
            if frame.dtype != inference_dtype:
                frame = frame.to(dtype=inference_dtype)
            dt_value = float(gt["dt"][source_idx]) if source_idx < gt["dt"].shape[0] else float(cfg.prediction_dt)
            dt = torch.tensor([dt_value], device=device, dtype=frame.dtype)

            with torch.inference_mode():
                with torch.amp.autocast(device_type=device.type, dtype=inference_dtype, enabled=use_autocast):
                    output, state = model.forward_step(frame, dt, state)

            target_idx = source_idx + target_offset
            if 0 <= target_idx < total_frames:
                predictions["button_logits"][target_idx] = output.horizon_button_logits[0, h_idx].detach().cpu().float().numpy()
                predictions["press_logits"][target_idx] = output.horizon_press_logits[0, h_idx].detach().cpu().float().numpy()
                predictions["release_logits"][target_idx] = output.horizon_release_logits[0, h_idx].detach().cpu().float().numpy()
                predictions["mouse_active_logits"][target_idx] = output.horizon_mouse_active_logits[0, h_idx].detach().cpu().float().numpy()
                predictions["mouse_delta"][target_idx] = output.horizon_mouse_delta[0, h_idx].detach().cpu().float().numpy()
                predictions["source_frame"][target_idx] = source_idx
                predictions["valid_mask"][target_idx] = True
            pbar.update(1)
        pbar.close()
    finally:
        cap.release()

    return predictions


def compute_summary_stats(
    predictions: Dict[str, np.ndarray],
    gt: Dict[str, np.ndarray],
    thresholds: np.ndarray,
) -> Dict[str, float]:
    valid = predictions["valid_mask"].astype(bool) & gt["valid_mask"].astype(bool)
    if not bool(valid.any()):
        return {"accuracy": 0.0, "macro_f1": 0.0, "precision": 0.0, "recall": 0.0, "mouse_mae": 0.0}

    pred = (sigmoid_np(predictions["button_logits"][valid]) >= thresholds.reshape(1, -1)).astype(np.float32)
    true = gt["button_state"][valid].astype(np.float32)
    tp = (pred * true).sum(axis=0)
    tn = ((1.0 - pred) * (1.0 - true)).sum(axis=0)
    fp = (pred * (1.0 - true)).sum(axis=0)
    fn = ((1.0 - pred) * true).sum(axis=0)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / np.maximum(tp + fn, 1.0)
    f1 = (2.0 * precision * recall) / np.maximum(precision + recall, 1e-8)
    active_or_predicted = (tp + fn + tp + fp) > 0.0
    if bool(active_or_predicted.any()):
        macro_f1 = float(f1[active_or_predicted].mean())
        mean_precision = float(precision[active_or_predicted].mean())
        mean_recall = float(recall[active_or_predicted].mean())
    else:
        macro_f1 = 0.0
        mean_precision = 0.0
        mean_recall = 0.0
    accuracy = float((tp + tn).sum() / np.maximum(tp + tn + fp + fn, 1.0).sum())
    mouse_mae = float(np.abs(predictions["mouse_delta"][valid] - gt["mouse_delta"][valid]).mean())
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "precision": mean_precision,
        "recall": mean_recall,
        "mouse_mae": mouse_mae,
    }


def compute_per_button_stats(
    predictions: Dict[str, np.ndarray],
    gt: Dict[str, np.ndarray],
    names: Sequence[str],
    thresholds: np.ndarray,
) -> List[Dict[str, float | int | str]]:
    valid = predictions["valid_mask"].astype(bool) & gt["valid_mask"].astype(bool)
    if not bool(valid.any()):
        return []
    pred = (sigmoid_np(predictions["button_logits"][valid]) >= thresholds.reshape(1, -1)).astype(np.float32)
    true = gt["button_state"][valid].astype(np.float32)
    items: List[Dict[str, float | int | str]] = []
    for idx, name in enumerate(names):
        pred_col = pred[:, idx]
        true_col = true[:, idx]
        tp = float((pred_col * true_col).sum())
        tn = float(((1.0 - pred_col) * (1.0 - true_col)).sum())
        fp = float((pred_col * (1.0 - true_col)).sum())
        fn = float(((1.0 - pred_col) * true_col).sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = (2.0 * precision * recall) / max(precision + recall, 1e-8)
        total = max(tp + tn + fp + fn, 1.0)
        items.append(
            {
                "name": str(name),
                "accuracy": float((tp + tn) / total),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(true_col.sum()),
                "predicted": int(pred_col.sum()),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
            }
        )
    return items


def _put_text(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    color: Tuple[int, int, int] = (230, 230, 230),
    scale: float = 0.52,
    thickness: int = 1,
) -> int:
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return y + int(24 * scale / 0.52)


def _draw_key_rows(
    panel: np.ndarray,
    x: int,
    y: int,
    names: Sequence[str],
    probs: np.ndarray,
    pred_state: np.ndarray,
    true_state: np.ndarray,
    thresholds: np.ndarray,
) -> int:
    for idx, name in enumerate(names):
        pred_on = bool(pred_state[idx])
        true_on = bool(true_state[idx])
        if pred_on and true_on:
            color = (120, 255, 140)
            status = "OK "
        elif pred_on and not true_on:
            color = (80, 170, 255)
            status = "FP "
        elif (not pred_on) and true_on:
            color = (70, 90, 255)
            status = "MISS"
        else:
            color = (150, 155, 160)
            status = "off"
        y = _put_text(
            panel,
            f"{name:>6s}  p={float(probs[idx]):.3f}  t={float(thresholds[idx]):.2f}  gt={int(true_on)}  {status}",
            x,
            y,
            color=color,
        )
    return y


def draw_overlay_frame(
    frame: np.ndarray,
    *,
    frame_idx: int,
    predictions: Dict[str, np.ndarray],
    gt: Dict[str, np.ndarray],
    cfg: ModelConfig,
    names: Sequence[str],
    thresholds: np.ndarray,
    summary: Dict[str, float],
    command_horizon: int,
    action_label_offset: int,
    panel_width: int,
) -> np.ndarray:
    height, width = frame.shape[:2]
    output_height = height + (height % 2)
    output_width = width + int(panel_width)
    output_width += output_width % 2
    canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    canvas[:height, :width] = frame
    panel = canvas[:, width:]
    panel[:] = (18, 20, 24)

    probs = sigmoid_np(predictions["button_logits"][frame_idx])
    pred_state = probs >= thresholds
    true_state = gt["button_state"][frame_idx].astype(bool)
    gt_mouse = gt["mouse_delta"][frame_idx]
    pred_mouse = predictions["mouse_delta"][frame_idx]
    source_idx = int(predictions["source_frame"][frame_idx])

    x = 18
    y = 30
    y = _put_text(panel, f"Frame {frame_idx}", x, y, color=(255, 255, 255), scale=0.72, thickness=2)
    y = _put_text(
        panel,
        f"pred source={source_idx}  horizon={int(command_horizon)}  offset={int(action_label_offset)}",
        x,
        y,
        color=(190, 200, 210),
    )
    y = _put_text(
        panel,
        f"valid pred={int(predictions['valid_mask'][frame_idx])} gt={int(gt['valid_mask'][frame_idx])}",
        x,
        y,
        color=(190, 200, 210),
    )
    y += 8
    y = _put_text(panel, "Run metrics", x, y, color=(255, 230, 170), scale=0.6, thickness=2)
    y = _put_text(
        panel,
        f"acc={summary['accuracy']:.3f} f1={summary['macro_f1']:.3f} "
        f"P={summary['precision']:.3f} R={summary['recall']:.3f}",
        x,
        y,
    )
    y = _put_text(panel, f"mouse MAE={summary['mouse_mae']:.2f}px", x, y)
    y += 10
    y = _put_text(panel, "Keys", x, y, color=(180, 255, 180), scale=0.6, thickness=2)
    y = _draw_key_rows(panel, x, y, names, probs, pred_state, true_state, thresholds)
    y += 10
    y = _put_text(panel, "Mouse", x, y, color=(180, 220, 255), scale=0.6, thickness=2)
    y = _put_text(panel, f"pred dx={pred_mouse[0]:.2f} dy={pred_mouse[1]:.2f}", x, y)
    y = _put_text(panel, f"gt   dx={gt_mouse[0]:.2f} dy={gt_mouse[1]:.2f}", x, y)

    wrong = pred_state != true_state
    if bool(wrong.any()):
        missed = [names[idx] for idx in np.flatnonzero((~pred_state) & true_state)]
        extra = [names[idx] for idx in np.flatnonzero(pred_state & (~true_state))]
        y += 8
        y = _put_text(panel, f"miss: {', '.join(missed) if missed else 'none'}"[:70], x, y, color=(70, 90, 255))
        _put_text(panel, f"extra: {', '.join(extra) if extra else 'none'}"[:70], x, y, color=(80, 170, 255))

    return canvas


def write_overlay_video(
    video_path: str,
    output_path: str,
    predictions: Dict[str, np.ndarray],
    gt: Dict[str, np.ndarray],
    cfg: ModelConfig,
    *,
    thresholds: np.ndarray,
    command_horizon: int,
    action_label_offset: int,
    max_frames: Optional[int],
    panel_width: int,
    ffmpeg_path: Optional[str],
    codec: str,
    quality: int,
    preset: str,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError("Could not read video dimensions.")

    output_width = width + int(panel_width)
    output_width += output_width % 2
    output_height = height + (height % 2)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    writer = FFmpegPipeWriter(
        output_path,
        fps,
        output_width,
        output_height,
        ffmpeg_path=ffmpeg_path,
        codec=codec,
        quality=quality,
        preset=preset,
    )

    names = list(cfg.key_names) + list(cfg.mouse_button_names)
    total_frames = int(predictions["button_logits"].shape[0])
    if max_frames is not None and max_frames > 0:
        total_frames = min(total_frames, int(max_frames))
    summary = compute_summary_stats(predictions, gt, thresholds)

    try:
        pbar = tqdm(total=total_frames, desc="Render", unit="frame")
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            out = draw_overlay_frame(
                frame,
                frame_idx=frame_idx,
                predictions=predictions,
                gt=gt,
                cfg=cfg,
                names=names,
                thresholds=thresholds,
                summary=summary,
                command_horizon=command_horizon,
                action_label_offset=action_label_offset,
                panel_width=panel_width,
            )
            writer.write(out)
            pbar.update(1)
        pbar.close()
    finally:
        writer.release()
        cap.release()


def print_final_stats(
    predictions: Dict[str, np.ndarray],
    gt: Dict[str, np.ndarray],
    cfg: ModelConfig,
    thresholds: np.ndarray,
) -> None:
    summary = compute_summary_stats(predictions, gt, thresholds)
    names = list(cfg.key_names) + list(cfg.mouse_button_names)
    per_key = compute_per_button_stats(predictions, gt, names, thresholds)
    print("\nPolicy test stats")
    print(
        "Overall:",
        f"acc={summary['accuracy']:.4f}",
        f"macro_f1={summary['macro_f1']:.4f}",
        f"precision={summary['precision']:.4f}",
        f"recall={summary['recall']:.4f}",
        f"mouse_mae={summary['mouse_mae']:.3f}px",
    )
    print("Per key/button:")
    print("  name       acc     f1    prec    rec  support  pred  tp  fp  fn")
    for item in per_key:
        print(
            f"  {str(item['name'])[:8]:8s} "
            f"{float(item['accuracy']):6.3f} "
            f"{float(item['f1']):6.3f} "
            f"{float(item['precision']):6.3f} "
            f"{float(item['recall']):6.3f} "
            f"{int(item['support']):8d} "
            f"{int(item['predicted']):5d} "
            f"{int(item['tp']):3d} "
            f"{int(item['fp']):3d} "
            f"{int(item['fn']):3d}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay policy predictions vs CSV ground truth on a dataset video.")
    parser.add_argument("--checkpoint", default=None, help="Path to policy checkpoint. Defaults to ckpt dir best/latest.")
    parser.add_argument("--ckpt-dir", default="./checkpoints_rt", help="Checkpoint directory used when --checkpoint is omitted.")
    parser.add_argument("--video", default=None, help="Input dataset video. Defaults to the first run_*.mp4 in --data-root.")
    parser.add_argument("--labels", default=None, help="Ground-truth CSV. Defaults to the video path with .csv extension.")
    parser.add_argument("--data-root", default="./data/greenville", help="Dataset root used when --video is omitted.")
    parser.add_argument("--output", default="./data/test_model.mp4", help="Output annotated MP4 path.")
    parser.add_argument("--command-horizon", type=int, default=1, help="1-based horizon index to visualize.")
    parser.add_argument("--action-label-offset", type=int, default=None, help="Override checkpoint action_label_offset.")
    parser.add_argument("--threshold", type=float, default=None, help="Override button threshold. Defaults to checkpoint thresholds.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for quick tests.")
    parser.add_argument("--panel-width", type=int, default=560, help="Width of the side stats panel.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument("--use-checkpoint-thresholds", action="store_true", help="Use saved button thresholds if present.")
    parser.add_argument("--ffmpeg-path", default=None, help="Optional explicit ffmpeg path.")
    parser.add_argument("--output-codec", default="hevc_nvenc", help="FFmpeg encoder for the stats MP4.")
    parser.add_argument("--output-quality", type=int, default=20, help="NVENC CQ or x264 CRF value.")
    parser.add_argument("--output-preset", default="p4", help="FFmpeg encoder preset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = os.path.abspath(args.checkpoint) if args.checkpoint else os.path.abspath(_default_checkpoint(args.ckpt_dir))
    video_path = os.path.abspath(args.video) if args.video else os.path.abspath(_default_video(args.data_root, ".mp4"))
    label_path = os.path.abspath(args.labels) if args.labels else os.path.splitext(video_path)[0] + ".csv"
    output_path = os.path.abspath(args.output)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    use_autocast = device.type == "cuda" and bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    inference_dtype = torch.bfloat16 if use_autocast else torch.float32
    model, cfg, raw_config = load_model_checkpoint(
        checkpoint_path,
        device,
        use_checkpoint_thresholds=bool(args.use_checkpoint_thresholds),
    )
    action_label_offset = int(raw_config.get("action_label_offset", 0) if args.action_label_offset is None else args.action_label_offset)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if args.max_frames is not None and args.max_frames > 0 and total_frames > 0:
        total_frames = min(total_frames, int(args.max_frames))

    gt = load_ground_truth(label_path, cfg, total_frames)
    thresholds = _resolve_thresholds(cfg, args.threshold)
    print(
        "Loaded policy:",
        f"checkpoint={checkpoint_path}",
        f"video={video_path}",
        f"game={cfg.selected_game}",
        "temporal=attention",
        f"heads={cfg.temporal_heads}",
        f"seq={cfg.seq_len}",
        f"horizon={cfg.prediction_horizon}",
        f"command_horizon={args.command_horizon}",
        f"offset={action_label_offset}",
        f"actions={','.join(cfg.key_names + cfg.mouse_button_names)}",
        f"device={device}",
    )

    predictions = infer_video(
        video_path,
        model,
        cfg,
        gt,
        device,
        command_horizon=int(args.command_horizon),
        action_label_offset=action_label_offset,
        max_frames=args.max_frames,
        use_autocast=use_autocast,
        inference_dtype=inference_dtype,
    )
    write_overlay_video(
        video_path,
        output_path,
        predictions,
        gt,
        cfg,
        thresholds=thresholds,
        command_horizon=int(args.command_horizon),
        action_label_offset=action_label_offset,
        max_frames=args.max_frames,
        panel_width=max(360, int(args.panel_width)),
        ffmpeg_path=args.ffmpeg_path,
        codec=str(args.output_codec),
        quality=int(args.output_quality),
        preset=str(args.output_preset),
    )
    print_final_stats(predictions, gt, cfg, thresholds)
    print(f"Wrote stats video: {output_path}")


if __name__ == "__main__":
    main()
