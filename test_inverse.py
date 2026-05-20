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

from inverse_dynamics import (
    InverseDynamicsConfig,
    InverseDynamicsModel,
    center_window_bounds,
    inverse_checkpoint_family_mismatch_reason,
)


class FFmpegPipeWriter:
    def __init__(
        self,
        out_path: str,
        fps: float,
        width: int,
        height: int,
        *,
        ffmpeg_path: Optional[str],
        codec: str = "hevc_nvenc",
        quality: int = 20,
        preset: str = "p4",
        pix_fmt_in: str = "bgr24",
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
            pix_fmt_in,
            "-s",
            f"{int(width)}x{int(height)}",
            "-r",
            str(float(fps)),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            codec,
        ]
        if "nvenc" in codec:
            cmd.extend(["-cq", str(int(quality)), "-preset", preset])
        else:
            cmd.extend(["-crf", str(int(quality)), "-preset", preset])
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


def _load_checkpoint_config(checkpoint_path: str) -> Tuple[InverseDynamicsConfig, Dict[str, torch.Tensor]]:
    state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state, dict):
        raise RuntimeError(f"Expected checkpoint dict, got {type(state).__name__}.")

    config_dict = state.get("config", {})
    family_reason = inverse_checkpoint_family_mismatch_reason(config_dict)
    if family_reason is not None:
        raise RuntimeError(f"Cannot load inverse checkpoint {checkpoint_path}: {family_reason}")

    valid_keys = {field.name for field in fields(InverseDynamicsConfig)}
    cfg = InverseDynamicsConfig(**{key: value for key, value in config_dict.items() if key in valid_keys})
    if "model" in state:
        model_state = state["model"]
    elif "model_state" in state:
        model_state = state["model_state"]
    else:
        raise RuntimeError("Checkpoint does not contain a 'model' state dict.")
    return cfg, model_state


def load_inverse_model(
    checkpoint_path: str,
    device: torch.device,
    *,
    compile_model: bool,
    compile_mode: str,
) -> Tuple[torch.nn.Module, InverseDynamicsConfig, torch.dtype, bool]:
    cfg, model_state = _load_checkpoint_config(checkpoint_path)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    model: torch.nn.Module = InverseDynamicsModel(cfg).to(device)
    model.load_state_dict(model_state, strict=True)
    if device.type == "cuda":
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass

    inference_dtype = torch.float32
    use_autocast = False
    if device.type == "cuda" and bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()):
        inference_dtype = torch.bfloat16
        use_autocast = True

    model.eval()
    if device.type == "cuda" and compile_model and hasattr(torch, "compile"):
        try:
            compile_kwargs = {"fullgraph": False, "dynamic": False}
            if compile_mode and compile_mode.lower() != "default":
                compile_kwargs["mode"] = compile_mode
            model = torch.compile(model, **compile_kwargs)
            model.eval()
        except Exception as exc:
            print(f"torch.compile disabled after failure: {exc}")
    return model, cfg, inference_dtype, use_autocast


def _frame_count(cap: cv2.VideoCapture, max_frames: Optional[int]) -> int:
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames is not None and max_frames > 0 and count > 0:
        return min(count, int(max_frames))
    if count > 0:
        return count

    counted = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        counted += 1
        if max_frames is not None and max_frames > 0 and counted >= int(max_frames):
            break
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return counted


def _aggregate_window(output, start: int, accum: Dict[str, np.ndarray], *, output_length: int, batch_index: int) -> None:
    length = min(int(output.button_logits.shape[1]), int(output_length))
    if length <= 0:
        return
    sl = slice(start, start + length)
    accum["button_logits"][sl] += output.button_logits[batch_index, :length].detach().cpu().float().numpy()
    accum["mouse_active_logits"][sl] += output.mouse_active_logits[batch_index, :length].detach().cpu().float().numpy()
    accum["mouse_delta"][sl] += output.mouse_delta[batch_index, :length].detach().cpu().float().numpy()
    if accum["scroll_delta"].shape[1] > 0:
        accum["scroll_delta"][sl] += output.scroll_delta[batch_index, :length].detach().cpu().float().numpy()
    accum["counts"][sl] += 1.0


def infer_video_predictions(
    video_path: str,
    model: torch.nn.Module,
    cfg: InverseDynamicsConfig,
    device: torch.device,
    *,
    inference_dtype: torch.dtype,
    use_autocast: bool,
    batch_size: int,
    max_frames: Optional[int],
) -> Dict[str, np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = _frame_count(cap, max_frames)
    if total_frames <= 0:
        raise RuntimeError("Video frame count is unavailable or zero.")

    accum: Dict[str, np.ndarray] = {
        "button_logits": np.zeros((total_frames, cfg.num_bin), dtype=np.float32),
        "mouse_active_logits": np.zeros((total_frames, 1), dtype=np.float32),
        "mouse_delta": np.zeros((total_frames, 2), dtype=np.float32),
        "scroll_delta": np.zeros((total_frames, cfg.num_scroll), dtype=np.float32),
        "counts": np.zeros((total_frames,), dtype=np.float32),
    }

    output_offset, output_end = center_window_bounds(cfg.seq_len, cfg.output_seq_len)
    output_span = output_end - output_offset
    output_stride = max(1, int(cfg.output_seq_len))
    batch_size = max(1, int(batch_size))

    buffer: List[torch.Tensor] = []
    buffer_start = 0
    first_frame: Optional[torch.Tensor] = None
    frames_seen = 0
    next_output_start = 0
    pending_clips: List[torch.Tensor] = []
    pending_starts: List[int] = []
    pending_lengths: List[int] = []

    def flush_pending(*, force: bool = False) -> None:
        if not pending_clips:
            return
        if not force and len(pending_clips) < batch_size:
            return
        clips = torch.stack(pending_clips, dim=0)
        if device.type == "cuda":
            try:
                clips = clips.pin_memory()
            except Exception:
                pass
        clips = clips.to(device, non_blocking=True)
        if clips.dtype != inference_dtype:
            clips = clips.to(dtype=inference_dtype)
        with torch.inference_mode():
            with torch.amp.autocast(device_type=device.type, dtype=inference_dtype, enabled=use_autocast):
                output = model(clips)
        for batch_idx, (start, length) in enumerate(zip(pending_starts, pending_lengths)):
            _aggregate_window(output, start, accum, output_length=length, batch_index=batch_idx)
        pending_clips.clear()
        pending_starts.clear()
        pending_lengths.clear()

    def maybe_process_ready_windows() -> None:
        nonlocal next_output_start, buffer_start
        while next_output_start < total_frames:
            window_start = next_output_start - output_offset
            window_end = window_start + cfg.seq_len
            required_frames = min(total_frames, max(0, window_end))
            if frames_seen < required_frames:
                break
            if first_frame is None or not buffer:
                raise RuntimeError("Inference buffer is unexpectedly empty.")
            last_frame = buffer[-1]
            clip_items: List[torch.Tensor] = []
            for frame_idx in range(window_start, window_end):
                if frame_idx < 0:
                    clip_items.append(first_frame)
                elif frame_idx >= total_frames:
                    clip_items.append(last_frame)
                else:
                    offset = frame_idx - buffer_start
                    if offset < 0 or offset >= len(buffer):
                        raise RuntimeError(
                            f"Frame {frame_idx} missing from inference buffer "
                            f"(buffer_start={buffer_start}, len={len(buffer)})."
                        )
                    clip_items.append(buffer[offset])
            pending_clips.append(torch.stack(clip_items, dim=0))
            pending_starts.append(next_output_start)
            pending_lengths.append(min(output_span, total_frames - next_output_start))
            flush_pending()
            next_output_start += output_stride
            drop_until = max(0, next_output_start - output_offset)
            while buffer_start < drop_until and buffer:
                buffer.pop(0)
                buffer_start += 1

    try:
        pbar = tqdm(total=total_frames, desc="Infer", unit="frame")
        while frames_seen < total_frames:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (cfg.model_size, cfg.model_size), interpolation=cv2.INTER_AREA)
            frame = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
            if first_frame is None:
                first_frame = frame
            buffer.append(frame)
            frames_seen += 1
            pbar.update(1)
            maybe_process_ready_windows()
        maybe_process_ready_windows()
        flush_pending(force=True)
        pbar.close()
    finally:
        cap.release()

    counts = np.clip(accum["counts"], 1.0, None)
    return {
        "button_logits": accum["button_logits"] / counts[:, None],
        "mouse_active_logits": accum["mouse_active_logits"] / counts[:, None],
        "mouse_delta": accum["mouse_delta"] / counts[:, None],
        "scroll_delta": accum["scroll_delta"] / counts[:, None],
        "valid_mask": accum["counts"] > 0.0,
    }


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float32)))


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


def load_ground_truth_labels(label_path: str, cfg: InverseDynamicsConfig, total_frames: int) -> Dict[str, np.ndarray]:
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Ground-truth label CSV not found: {label_path}")

    rows: List[Dict[str, str]] = []
    with open(label_path, "r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        fieldnames = set(reader.fieldnames or [])
        required = ["timestamp"] + list(cfg.key_names or []) + list(cfg.mouse_button_names or []) + ["delta_x", "delta_y"]
        missing = [name for name in required if name not in fieldnames]
        if missing:
            raise RuntimeError(f"CSV schema mismatch for {label_path}: missing columns={missing}")
        rows.extend(reader)

    count = min(int(total_frames), len(rows))
    buttons = np.zeros((total_frames, cfg.num_bin), dtype=np.float32)
    mouse_delta = np.zeros((total_frames, 2), dtype=np.float32)
    scroll_delta = np.zeros((total_frames, cfg.num_scroll), dtype=np.float32)
    valid = np.zeros((total_frames,), dtype=bool)

    for idx in range(count):
        row = rows[idx]
        col = 0
        for name in cfg.key_names or []:
            buttons[idx, col] = 1.0 if _parse_float(row.get(name)) > 0.5 else 0.0
            col += 1
        for name in cfg.binary_mouse_button_names or []:
            buttons[idx, col] = 1.0 if _parse_float(row.get(name)) > 0.5 else 0.0
            col += 1
        for scroll_idx, name in enumerate(cfg.scroll_action_names):
            scroll_delta[idx, scroll_idx] = max(0.0, _parse_float(row.get(name)))
        mouse_delta[idx, 0] = _parse_float(row.get("delta_x"))
        mouse_delta[idx, 1] = _parse_float(row.get("delta_y"))
        valid[idx] = True

    return {
        "button_state": buttons,
        "mouse_delta": mouse_delta,
        "scroll_delta": scroll_delta,
        "valid_mask": valid,
    }


def compute_summary_stats(
    predictions: Dict[str, np.ndarray],
    ground_truth: Dict[str, np.ndarray],
    cfg: InverseDynamicsConfig,
    threshold: float,
) -> Dict[str, float]:
    valid = predictions["valid_mask"].astype(bool) & ground_truth["valid_mask"].astype(bool)
    if not bool(valid.any()):
        return {"button_f1": 0.0, "button_precision": 0.0, "button_recall": 0.0, "mouse_mae": 0.0, "scroll_mae": 0.0}

    pred_buttons = (sigmoid_np(predictions["button_logits"][valid]) >= float(threshold)).astype(np.float32)
    true_buttons = ground_truth["button_state"][valid].astype(np.float32)
    tp = (pred_buttons * true_buttons).sum(axis=0)
    fp = (pred_buttons * (1.0 - true_buttons)).sum(axis=0)
    fn = ((1.0 - pred_buttons) * true_buttons).sum(axis=0)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / np.maximum(tp + fn, 1.0)
    f1 = (2.0 * precision * recall) / np.maximum(precision + recall, 1e-8)
    support = tp + fn
    predicted = tp + fp
    measured = (support + predicted) > 0.0
    if bool(measured.any()):
        button_f1 = float(f1[measured].mean())
        button_precision = float(precision[measured].mean())
        button_recall = float(recall[measured].mean())
    else:
        button_f1 = 0.0
        button_precision = 0.0
        button_recall = 0.0

    mouse_mae = np.abs(predictions["mouse_delta"][valid] - ground_truth["mouse_delta"][valid]).mean()
    if cfg.num_scroll > 0:
        scroll_mae = np.abs(predictions["scroll_delta"][valid] - ground_truth["scroll_delta"][valid]).mean()
    else:
        scroll_mae = 0.0
    return {
        "button_f1": button_f1,
        "button_precision": button_precision,
        "button_recall": button_recall,
        "mouse_mae": float(mouse_mae),
        "scroll_mae": float(scroll_mae),
    }


def compute_per_button_stats(
    predictions: Dict[str, np.ndarray],
    ground_truth: Dict[str, np.ndarray],
    names: Sequence[str],
    threshold: float,
) -> List[Dict[str, float | int | str]]:
    valid = predictions["valid_mask"].astype(bool) & ground_truth["valid_mask"].astype(bool)
    if not bool(valid.any()):
        return []

    pred = (sigmoid_np(predictions["button_logits"][valid]) >= float(threshold)).astype(np.float32)
    true = ground_truth["button_state"][valid].astype(np.float32)
    stats: List[Dict[str, float | int | str]] = []
    for idx, name in enumerate(names):
        pred_col = pred[:, idx]
        true_col = true[:, idx]
        tp = float((pred_col * true_col).sum())
        tn = float(((1.0 - pred_col) * (1.0 - true_col)).sum())
        fp = float((pred_col * (1.0 - true_col)).sum())
        fn = float(((1.0 - pred_col) * true_col).sum())
        total = max(1.0, tp + tn + fp + fn)
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        f1 = (2.0 * precision * recall) / max(1e-8, precision + recall)
        stats.append(
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
    return stats


def print_final_stats(
    predictions: Dict[str, np.ndarray],
    ground_truth: Dict[str, np.ndarray],
    cfg: InverseDynamicsConfig,
    threshold: float,
) -> None:
    names = list(cfg.key_names or []) + list(cfg.binary_mouse_button_names or [])
    summary = compute_summary_stats(predictions, ground_truth, cfg, threshold)
    per_button = compute_per_button_stats(predictions, ground_truth, names, threshold)
    print("\nInverse test stats")
    print(
        "Overall:",
        f"button_f1={summary['button_f1']:.4f}",
        f"precision={summary['button_precision']:.4f}",
        f"recall={summary['button_recall']:.4f}",
        f"mouse_mae={summary['mouse_mae']:.3f}px",
        f"scroll_mae={summary['scroll_mae']:.3f}",
    )
    print("Per key/button:")
    print("  name                 acc     f1    prec    rec  support  pred  tp  fp  fn")
    for item in per_button:
        print(
            f"  {str(item['name'])[:18]:18s} "
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


def active_button_names(
    logits: np.ndarray,
    names: Sequence[str],
    *,
    threshold: float,
    max_items: int,
) -> List[Tuple[str, float]]:
    probs = sigmoid_np(logits)
    active = [(names[idx], float(probs[idx])) for idx in range(len(names)) if probs[idx] >= threshold]
    active.sort(key=lambda item: item[1], reverse=True)
    return active[:max_items]


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


def _wrap_items(prefix: str, items: Sequence[str], *, max_chars: int) -> List[str]:
    if not items:
        return [f"{prefix}: none"]
    lines: List[str] = []
    current = f"{prefix}: "
    for item in items:
        next_text = item if current.endswith(": ") else f", {item}"
        if len(current) + len(next_text) > max_chars and not current.endswith(": "):
            lines.append(current)
            current = "  " + item
        else:
            current += next_text
    lines.append(current)
    return lines


def _draw_delta_arrow(
    panel: np.ndarray,
    origin: Tuple[int, int],
    delta: np.ndarray,
    *,
    color: Tuple[int, int, int],
    label: str,
) -> None:
    scale = 2.5
    end = (
        int(origin[0] + float(delta[0]) * scale),
        int(origin[1] + float(delta[1]) * scale),
    )
    cv2.circle(panel, origin, 4, (210, 210, 210), -1, cv2.LINE_AA)
    cv2.arrowedLine(panel, origin, end, color, 2, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(panel, label, (origin[0] + 12, origin[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_stats_panel(
    frame: np.ndarray,
    *,
    frame_idx: int,
    cfg: InverseDynamicsConfig,
    predictions: Dict[str, np.ndarray],
    ground_truth: Dict[str, np.ndarray],
    summary: Dict[str, float],
    button_names: Sequence[str],
    threshold: float,
    max_buttons: int,
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

    pred_probs = sigmoid_np(predictions["button_logits"][frame_idx])
    pred_state = pred_probs >= float(threshold)
    true_state = ground_truth["button_state"][frame_idx].astype(bool)
    pred_items = [f"{button_names[idx]}:{pred_probs[idx]:.2f}" for idx in np.flatnonzero(pred_state)[:max_buttons]]
    true_items = [button_names[idx] for idx in np.flatnonzero(true_state)[:max_buttons]]
    false_pos = [button_names[idx] for idx in np.flatnonzero(pred_state & ~true_state)[:max_buttons]]
    false_neg = [button_names[idx] for idx in np.flatnonzero(~pred_state & true_state)[:max_buttons]]

    pred_mouse = predictions["mouse_delta"][frame_idx]
    true_mouse = ground_truth["mouse_delta"][frame_idx]
    mouse_err = pred_mouse - true_mouse
    mouse_active = float(sigmoid_np(predictions["mouse_active_logits"][frame_idx])[0])
    pred_scroll = predictions["scroll_delta"][frame_idx]
    true_scroll = ground_truth["scroll_delta"][frame_idx]
    valid_pred = bool(predictions["valid_mask"][frame_idx])
    valid_gt = bool(ground_truth["valid_mask"][frame_idx])

    x = 18
    y = 30
    y = _put_text(panel, f"Frame {frame_idx}", x, y, color=(255, 255, 255), scale=0.72, thickness=2)
    y = _put_text(panel, f"valid pred={int(valid_pred)} gt={int(valid_gt)}", x, y, color=(190, 200, 210), scale=0.5)
    y += 8
    y = _put_text(panel, "Run metrics", x, y, color=(255, 230, 170), scale=0.6, thickness=2)
    y = _put_text(panel, f"button F1={summary['button_f1']:.3f}  P={summary['button_precision']:.3f}  R={summary['button_recall']:.3f}", x, y)
    y = _put_text(panel, f"mouse MAE={summary['mouse_mae']:.2f}px  scroll MAE={summary['scroll_mae']:.2f}", x, y)
    y += 10
    y = _put_text(panel, "Buttons", x, y, color=(180, 255, 180), scale=0.6, thickness=2)
    for line in _wrap_items("pred", pred_items, max_chars=58):
        y = _put_text(panel, line, x, y, color=(170, 245, 170))
    for line in _wrap_items("gt", true_items, max_chars=58):
        y = _put_text(panel, line, x, y, color=(150, 200, 255))
    for line in _wrap_items("extra", false_pos, max_chars=58):
        y = _put_text(panel, line, x, y, color=(120, 170, 255))
    for line in _wrap_items("miss", false_neg, max_chars=58):
        y = _put_text(panel, line, x, y, color=(80, 110, 255))

    y += 10
    y = _put_text(panel, "Mouse", x, y, color=(180, 220, 255), scale=0.6, thickness=2)
    y = _put_text(panel, f"pred dx={pred_mouse[0]:.2f} dy={pred_mouse[1]:.2f} active={mouse_active:.2f}", x, y)
    y = _put_text(panel, f"gt   dx={true_mouse[0]:.2f} dy={true_mouse[1]:.2f}", x, y)
    y = _put_text(panel, f"err  dx={mouse_err[0]:.2f} dy={mouse_err[1]:.2f}", x, y)
    arrow_y = min(output_height - 80, y + 55)
    _draw_delta_arrow(panel, (x + 80, arrow_y), true_mouse, color=(150, 200, 255), label="gt")
    _draw_delta_arrow(panel, (x + 240, arrow_y), pred_mouse, color=(0, 255, 255), label="pred")
    y = arrow_y + 55

    if cfg.num_scroll > 0 and y < output_height - 28:
        y = _put_text(panel, "Scroll", x, y, color=(255, 220, 160), scale=0.6, thickness=2)
        pred_text = "pred " + " ".join(f"{name}={float(pred_scroll[idx]):.2f}" for idx, name in enumerate(cfg.scroll_action_names))
        gt_text = "gt   " + " ".join(f"{name}={float(true_scroll[idx]):.2f}" for idx, name in enumerate(cfg.scroll_action_names))
        y = _put_text(panel, pred_text[:64], x, y, color=(255, 220, 160))
        _put_text(panel, gt_text[:64], x, y, color=(150, 200, 255))

    return canvas


def write_stats_video(
    video_path: str,
    output_path: str,
    cfg: InverseDynamicsConfig,
    predictions: Dict[str, np.ndarray],
    ground_truth: Dict[str, np.ndarray],
    *,
    threshold: float,
    max_buttons: int,
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

    key_names = list(cfg.key_names or [])
    button_names = key_names + list(cfg.binary_mouse_button_names or [])
    total_frames = int(predictions["button_logits"].shape[0])
    if max_frames is not None and max_frames > 0:
        total_frames = min(total_frames, int(max_frames))
    summary = compute_summary_stats(predictions, ground_truth, cfg, threshold)

    try:
        pbar = tqdm(total=total_frames, desc="Render stats", unit="frame")
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = draw_stats_panel(
                frame,
                frame_idx=frame_idx,
                cfg=cfg,
                predictions=predictions,
                ground_truth=ground_truth,
                summary=summary,
                button_names=button_names,
                threshold=threshold,
                max_buttons=max_buttons,
                panel_width=panel_width,
            )
            writer.write(frame)
            pbar.update(1)
        pbar.close()
    finally:
        writer.release()
        cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render inverse-dynamics predictions and ground truth beside a video.")
    parser.add_argument("--checkpoint", default=r'C:\Users\Abhil\Desktop\vs_code_stuff\python\ai\checkpoints_idm\model_epoch_9.pt', help="Path to inverse model checkpoint.")
    parser.add_argument("--video", default=r'C:\Users\Abhil\Desktop\vs_code_stuff\python\ai\data\arc_raiders\run_20260421_182551.mp4', help="Input video path.")
    parser.add_argument("--labels", default=None, help="Ground-truth CSV path. Defaults to the video path with .csv extension.")
    parser.add_argument("--output", default=r'C:\Users\Abhil\Desktop\vs_code_stuff\python\ai\data\test_inverse.mp4', help="Output annotated mp4 path.")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference windows per forward pass.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Button probability threshold for predicted active labels.")
    parser.add_argument("--max-buttons", type=int, default=25, help="Maximum active buttons to show.")
    parser.add_argument("--panel-width", type=int, default=720, help="Width of the side stats panel in pixels.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for quick tests.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument("--compile", dest="compile_model", action="store_true", default=False, help="Use torch.compile on CUDA.")
    parser.add_argument("--no-compile", dest="compile_model", action="store_false", help="Disable torch.compile.")
    parser.add_argument("--compile-mode", default="default", help="Optional torch.compile mode.")
    parser.add_argument("--ffmpeg-path", default=None, help="Optional explicit ffmpeg path.")
    parser.add_argument("--output-codec", default="hevc_nvenc", help="FFmpeg encoder for the stats MP4.")
    parser.add_argument("--output-quality", type=int, default=20, help="NVENC CQ or x264 CRF value.")
    parser.add_argument("--output-preset", default="p4", help="FFmpeg encoder preset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = os.path.abspath(args.checkpoint)
    video_path = os.path.abspath(args.video)
    label_path = os.path.abspath(args.labels) if args.labels else os.path.splitext(video_path)[0] + ".csv"
    if args.output is None:
        root, _ = os.path.splitext(video_path)
        output_path = f"{root}_idm_stats.mp4"
    else:
        output_path = os.path.abspath(args.output)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, cfg, inference_dtype, use_autocast = load_inverse_model(
        checkpoint_path,
        device,
        compile_model=bool(args.compile_model),
        compile_mode=str(args.compile_mode),
    )
    print(
        "Loaded IDM:",
        f"arch={cfg.visual_encoder_name}",
        f"game={cfg.selected_game}",
        f"seq={cfg.seq_len}",
        f"out={cfg.output_seq_len}",
        f"model_size={cfg.model_size}",
        f"device={device}",
        f"autocast={use_autocast}",
    )
    predictions = infer_video_predictions(
        video_path,
        model,
        cfg,
        device,
        inference_dtype=inference_dtype,
        use_autocast=use_autocast,
        batch_size=int(args.batch_size),
        max_frames=args.max_frames,
    )
    ground_truth = load_ground_truth_labels(
        label_path,
        cfg,
        int(predictions["button_logits"].shape[0]),
    )
    write_stats_video(
        video_path,
        output_path,
        cfg,
        predictions,
        ground_truth,
        threshold=float(args.threshold),
        max_buttons=int(args.max_buttons),
        max_frames=args.max_frames,
        panel_width=max(360, int(args.panel_width)),
        ffmpeg_path=args.ffmpeg_path,
        codec=args.output_codec,
        quality=int(args.output_quality),
        preset=str(args.output_preset),
    )
    print_final_stats(predictions, ground_truth, cfg, float(args.threshold))
    print(f"Wrote stats video: {output_path}")


if __name__ == "__main__":
    main()
