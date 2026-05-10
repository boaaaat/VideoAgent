import argparse
import csv
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from action_space import game_data_root, selected_game as ACTION_SELECTED_GAME
from inverse_dynamics import (
    InverseDynamicsConfig,
    InverseDynamicsModel,
    center_window_bounds,
    inverse_checkpoint_family_mismatch_reason,
)


@dataclass
class SourceVideo:
    source_url: str
    video_id: str
    title: str
    raw_path: str
    info_path: Optional[str] = None


@dataclass
class PseudoLabelConfig:
    selected_game: str = ACTION_SELECTED_GAME
    checkpoint_path: str = "./checkpoints_idm/model_latest.pt"
    pseudo_root: Optional[str] = None
    download_root: Optional[str] = None
    ffmpeg_path: Optional[str] = None

    target_size: int = 512
    target_fps: float = 20.0

    export_codec: str = "hevc_nvenc"
    export_quality: int = 20
    export_preset: str = "p4"
    inference_batch_size: int = 8
    compile_model: bool = True
    compile_mode: str = "default"

    def __post_init__(self) -> None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if self.pseudo_root is None:
            self.pseudo_root = game_data_root(self.selected_game, root="data_pseudo")
        if self.download_root is None:
            self.download_root = game_data_root(self.selected_game, root="downloads_youtube")
        if not os.path.isabs(self.checkpoint_path):
            self.checkpoint_path = os.path.abspath(os.path.join(base_dir, self.checkpoint_path))
        if not os.path.isabs(self.pseudo_root):
            self.pseudo_root = os.path.abspath(os.path.join(base_dir, self.pseudo_root))
        if not os.path.isabs(self.download_root):
            self.download_root = os.path.abspath(os.path.join(base_dir, self.download_root))
        self.target_size = int(self.target_size)
        self.target_fps = float(self.target_fps)
        if self.target_size <= 0:
            raise ValueError("target_size must be positive.")
        if self.target_fps <= 0.0:
            raise ValueError("target_fps must be positive.")
        self.export_quality = int(self.export_quality)
        self.inference_batch_size = max(1, int(self.inference_batch_size))
        self.compile_model = bool(self.compile_model)
        self.compile_mode = str(self.compile_mode)
        os.makedirs(self.pseudo_root, exist_ok=True)
        os.makedirs(self.download_root, exist_ok=True)


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
        self.out_path = out_path if out_path.endswith(".mp4") else f"{out_path}.mp4"
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
        if self.proc.stdin is None:
            raise RuntimeError("FFmpeg writer stdin is closed.")
        self.proc.stdin.write(frame_bgr.tobytes())

    def close(self) -> None:
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.flush()
                self.proc.stdin.close()
            except Exception:
                pass
        self.proc.wait(timeout=10)


def _run_checked(cmd: Sequence[str]) -> None:
    subprocess.run(list(cmd), check=True)


def _parse_ffmpeg_timestamp_seconds(raw_value: str) -> Optional[float]:
    value = str(raw_value).strip()
    if not value or value.upper() == "N/A":
        return None
    parts = value.split(":")
    if len(parts) != 3:
        return None
    try:
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
    except ValueError:
        return None
    return (hours * 3600.0) + (minutes * 60.0) + seconds


def _video_duration_seconds(video_path: str) -> Optional[float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    if frame_count <= 0.0 or fps <= 0.0:
        return None
    return frame_count / fps


def _run_ffmpeg_with_progress(
    cmd: Sequence[str],
    *,
    desc: str,
    total_seconds: Optional[float],
) -> None:
    proc = subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    pbar = tqdm(
        total=total_seconds if total_seconds is not None and total_seconds > 0.0 else None,
        desc=desc,
        unit="s",
        leave=False,
    )
    last_time = 0.0
    return_code: Optional[int] = None
    try:
        if proc.stdout is not None:
            for raw_line in proc.stdout:
                line = raw_line.strip()
                if line.startswith("out_time="):
                    current_time = _parse_ffmpeg_timestamp_seconds(line.partition("=")[2])
                    if current_time is None:
                        continue
                    if pbar.total is not None:
                        current_time = min(float(current_time), float(pbar.total))
                    if current_time > last_time:
                        pbar.update(current_time - last_time)
                        last_time = current_time
                elif line == "progress=end" and pbar.total is not None and last_time < float(pbar.total):
                    pbar.update(float(pbar.total) - last_time)
                    last_time = float(pbar.total)
        return_code = proc.wait()
    finally:
        pbar.close()
        if proc.stdout is not None:
            proc.stdout.close()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, list(cmd))


def _flatten_yt_entries(info: Dict) -> List[Dict]:
    if info.get("_type") == "playlist":
        entries = []
        for entry in info.get("entries", []):
            if entry:
                entries.append(entry)
        return entries
    return [info]


def _sanitize_token(value: str) -> str:
    chars = [ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value]
    token = "".join(chars).strip("_")
    return token or "video"


def _format_fps_token(fps: float) -> str:
    value = f"{float(fps):.3f}".rstrip("0").rstrip(".")
    return value.replace(".", "_") or "0"


def _resolve_download_path(ydl, info: Dict) -> str:
    candidate = info.get("_filename") or ydl.prepare_filename(info)
    if candidate and os.path.exists(candidate):
        return candidate
    if candidate:
        base, _ = os.path.splitext(candidate)
        for ext in (".mp4", ".mkv", ".webm", ".mov"):
            alt = base + ext
            if os.path.exists(alt):
                return alt
    requested = info.get("requested_downloads") or []
    for item in requested:
        path = item.get("filepath")
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not resolve downloaded path for {info.get('id')}")


def download_youtube(url: str, cfg: PseudoLabelConfig) -> List[SourceVideo]:
    try:
        import yt_dlp  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "yt-dlp is required for YouTube ingest. Install `yt-dlp` in the runtime environment."
        ) from exc

    raw_root = os.path.join(cfg.download_root, "raw")
    os.makedirs(raw_root, exist_ok=True)

    ydl_opts = {
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(raw_root, "%(id)s.%(ext)s"),
        "writeinfojson": True,
        "quiet": False,
        "ignoreerrors": True,
    }

    videos: List[SourceVideo] = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        for entry in _flatten_yt_entries(info):
            raw_path = _resolve_download_path(ydl, entry)
            base, _ = os.path.splitext(raw_path)
            info_path = base + ".info.json"
            videos.append(
                SourceVideo(
                    source_url=entry.get("webpage_url") or url,
                    video_id=str(entry.get("id") or os.path.basename(base)),
                    title=str(entry.get("title") or entry.get("id") or "video"),
                    raw_path=raw_path,
                    info_path=info_path if os.path.exists(info_path) else None,
                )
            )
    return videos


def transcode_to_domain(
    source: SourceVideo,
    cfg: PseudoLabelConfig,
    *,
    progress_desc: Optional[str] = None,
) -> str:
    processed_root = os.path.join(cfg.download_root, "processed")
    os.makedirs(processed_root, exist_ok=True)
    fps_token = _format_fps_token(cfg.target_fps)
    codec_token = _sanitize_token(cfg.export_codec)
    out_path = os.path.join(processed_root, f"{source.video_id}_{cfg.target_size}_{fps_token}fps_{codec_token}.mp4")
    if os.path.exists(out_path):
        return out_path

    ffmpeg_path = cfg.ffmpeg_path or shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("FFmpeg not found. Add ffmpeg to PATH or pass --ffmpeg-path.")

    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        source.raw_path,
        "-vf",
        f"fps={cfg.target_fps},scale={cfg.target_size}:{cfg.target_size}:flags=lanczos,setsar=1",
        "-an",
        "-c:v",
        cfg.export_codec,
    ]
    if "nvenc" in cfg.export_codec:
        cmd.extend(["-cq", str(int(cfg.export_quality)), "-preset", cfg.export_preset])
    else:
        cmd.extend(["-crf", str(int(cfg.export_quality)), "-preset", cfg.export_preset])
    cmd.extend(
        [
            "-r",
            str(float(cfg.target_fps)),
            "-pix_fmt",
            "yuv420p",
            "-nostats",
            "-progress",
            "pipe:1",
            out_path,
        ]
    )
    _run_ffmpeg_with_progress(
        cmd,
        desc=progress_desc or f"Transcode {source.video_id}",
        total_seconds=_video_duration_seconds(source.raw_path),
    )
    return out_path


def _load_checkpoint_config(checkpoint_path: str) -> Tuple[InverseDynamicsConfig, Dict]:
    state = torch.load(checkpoint_path, map_location="cpu")
    config_dict = state.get("config", {}) if isinstance(state, dict) else {}
    family_reason = inverse_checkpoint_family_mismatch_reason(config_dict)
    if family_reason is not None:
        raise RuntimeError(f"Cannot load inverse checkpoint {checkpoint_path}: {family_reason}")
    valid_keys = {field.name for field in fields(InverseDynamicsConfig)}
    filtered = {key: value for key, value in config_dict.items() if key in valid_keys}
    if "output_seq_len" not in filtered:
        legacy_center_only = config_dict.get("predict_center_frame_only")
        if legacy_center_only is True:
            filtered["output_seq_len"] = 1
        else:
            filtered["output_seq_len"] = int(filtered.get("seq_len", InverseDynamicsConfig().seq_len))
    cfg = InverseDynamicsConfig(**filtered)
    if isinstance(state, dict) and "model" in state:
        model_state = state["model"]
    elif isinstance(state, dict) and "model_state" in state:
        model_state = state["model_state"]
    else:
        model_state = state
    if isinstance(state, dict) and "mouse_delta_scales" in state:
        cfg.mouse_delta_scales = tuple(float(item) for item in state["mouse_delta_scales"])
        cfg.__post_init__()
    if isinstance(state, dict) and "scroll_delta_scales" in state:
        cfg.scroll_delta_scales = tuple(float(item) for item in state["scroll_delta_scales"])
        cfg.__post_init__()
    return cfg, model_state


def load_inverse_model(
    checkpoint_path: str,
    device: torch.device,
    *,
    compile_model: bool = True,
    compile_mode: str = "default",
) -> Tuple[torch.nn.Module, InverseDynamicsConfig, torch.dtype, bool]:
    cfg, model_state = _load_checkpoint_config(checkpoint_path)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    model = InverseDynamicsModel(cfg).to(device)
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
            tqdm.write(f"torch.compile disabled for IDM inference after failure: {exc}")
    return model, cfg, inference_dtype, use_autocast


def configure_idm_inference(idm_cfg: InverseDynamicsConfig, *, target_fps: float) -> None:
    center_window_bounds(idm_cfg.seq_len, idm_cfg.output_seq_len)
    idm_cfg.train_seq_stride = int(idm_cfg.output_seq_len)
    idm_cfg.prediction_dt = 1.0 / float(target_fps)


def _frame_count(video_path: str, max_frames: Optional[int] = None) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if count > 0:
        cap.release()
        if max_frames is not None and int(max_frames) > 0:
            return min(count, int(max_frames))
        return count
    count = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        count += 1
        if max_frames is not None and int(max_frames) > 0 and count >= int(max_frames):
            break
    cap.release()
    return count


def _aggregate_window(
    output,
    start: int,
    accum: Dict[str, np.ndarray],
    *,
    output_length: Optional[int] = None,
    batch_index: int = 0,
) -> None:
    length = int(output.button_logits.shape[1])
    if output_length is not None:
        length = min(length, int(output_length))
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
    idm_cfg: InverseDynamicsConfig,
    device: torch.device,
    *,
    inference_dtype: torch.dtype,
    use_autocast: bool,
    inference_batch_size: int = 8,
    progress_desc: Optional[str] = None,
    max_frames: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    total_frames = _frame_count(video_path, max_frames=max_frames)
    if total_frames <= 0:
        raise RuntimeError(f"Video has no readable frames for IDM inference: {video_path}")

    accum: Dict[str, np.ndarray] = {
        "button_logits": np.zeros((total_frames, idm_cfg.num_bin), dtype=np.float32),
        "mouse_active_logits": np.zeros((total_frames, 1), dtype=np.float32),
        "mouse_delta": np.zeros((total_frames, 2), dtype=np.float32),
        "scroll_delta": np.zeros((total_frames, idm_cfg.num_scroll), dtype=np.float32),
        "counts": np.zeros((total_frames,), dtype=np.float32),
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    buffer: List[torch.Tensor] = []
    buffer_start = 0
    first_frame: Optional[torch.Tensor] = None
    next_output_start = 0
    frames_seen = 0
    output_offset, output_end = center_window_bounds(idm_cfg.seq_len, idm_cfg.output_seq_len)
    output_span = output_end - output_offset
    output_stride = max(1, int(idm_cfg.train_seq_stride))
    inference_batch_size = max(1, int(inference_batch_size))
    pending_clips: List[torch.Tensor] = []
    pending_starts: List[int] = []
    pending_lengths: List[int] = []
    pbar = tqdm(total=total_frames, desc=progress_desc, unit="frame", leave=False) if progress_desc else None

    def flush_pending_windows(*, force: bool = False) -> None:
        if not pending_clips:
            return
        if not force and len(pending_clips) < inference_batch_size:
            return
        clip_frames = torch.stack(pending_clips, dim=0)
        if device.type == "cuda":
            try:
                clip_frames = clip_frames.pin_memory()
            except Exception:
                pass
        clip_frames = clip_frames.to(device, non_blocking=True)
        if clip_frames.dtype != inference_dtype:
            clip_frames = clip_frames.to(dtype=inference_dtype)
        with torch.inference_mode():
            with torch.amp.autocast(device_type=device.type, dtype=inference_dtype, enabled=use_autocast):
                output = model(clip_frames)
        for batch_idx, (start, length) in enumerate(zip(pending_starts, pending_lengths)):
            _aggregate_window(
                output,
                start,
                accum,
                output_length=length,
                batch_index=batch_idx,
            )
        pending_clips.clear()
        pending_starts.clear()
        pending_lengths.clear()

    def maybe_process_ready_windows() -> None:
        nonlocal next_output_start, buffer_start
        while next_output_start < total_frames:
            actual_window_start = next_output_start - output_offset
            actual_window_end = actual_window_start + idm_cfg.seq_len
            required_frames = min(total_frames, max(0, actual_window_end))
            if frames_seen < required_frames:
                break
            if first_frame is None:
                raise RuntimeError("IDM inference buffer initialized without a first frame.")
            if not buffer:
                raise RuntimeError("IDM inference buffer unexpectedly empty.")
            last_frame = buffer[-1]
            clip_items: List[torch.Tensor] = []
            for frame_idx in range(actual_window_start, actual_window_end):
                if frame_idx < 0:
                    clip_items.append(first_frame)
                elif frame_idx >= total_frames:
                    clip_items.append(last_frame)
                else:
                    offset = frame_idx - buffer_start
                    if offset < 0 or offset >= len(buffer):
                        raise RuntimeError(
                            f"Frame {frame_idx} missing from IDM buffer "
                            f"(buffer_start={buffer_start}, buffer_len={len(buffer)}, next_output_start={next_output_start})"
                    )
                    clip_items.append(buffer[offset])
            output_length = min(output_span, total_frames - next_output_start)
            pending_clips.append(torch.stack(clip_items, dim=0))
            pending_starts.append(next_output_start)
            pending_lengths.append(output_length)
            flush_pending_windows()
            next_output_start += output_stride
            drop_until = max(0, next_output_start - output_offset)
            while buffer_start < drop_until and buffer:
                buffer.pop(0)
                buffer_start += 1

    try:
        while True:
            if frames_seen >= total_frames:
                break
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
            if first_frame is None:
                first_frame = frame
            buffer.append(frame)
            frames_seen += 1
            if pbar is not None:
                pbar.update(1)
            maybe_process_ready_windows()
        maybe_process_ready_windows()
        flush_pending_windows(force=True)
    finally:
        if pbar is not None:
            pbar.close()
        cap.release()

    valid_mask = accum["counts"] > 0.0
    counts = np.clip(accum["counts"], 1.0, None)
    button_logits = accum["button_logits"] / counts[:, None]
    mouse_active_logits = accum["mouse_active_logits"] / counts[:, None]
    mouse_delta = accum["mouse_delta"] / counts[:, None]
    scroll_delta = accum["scroll_delta"] / counts[:, None]

    return {
        "button_logits": button_logits,
        "mouse_active_logits": mouse_active_logits,
        "mouse_delta": mouse_delta,
        "scroll_delta": scroll_delta,
        "valid_mask": valid_mask,
    }


def decode_button_states(
    button_logits: np.ndarray,
    idm_cfg: InverseDynamicsConfig,
    *,
    progress_desc: Optional[str] = None,
) -> np.ndarray:
    del progress_desc
    button_probs = 1.0 / (1.0 + np.exp(-button_logits.astype(np.float32)))
    return (button_probs >= float(idm_cfg.button_state_threshold)).astype(np.float32)


def apply_mouse_activity_gate(
    mouse_delta: np.ndarray,
    mouse_active_logits: np.ndarray,
    idm_cfg: InverseDynamicsConfig,
) -> np.ndarray:
    active = (1.0 / (1.0 + np.exp(-mouse_active_logits.astype(np.float32)))) >= float(idm_cfg.mouse_active_threshold)
    return mouse_delta.astype(np.float32) * active.astype(np.float32)


def _valid_prediction_span(predictions: Dict[str, np.ndarray]) -> Optional[Tuple[int, int]]:
    valid_mask = predictions.get("valid_mask")
    if valid_mask is None:
        total = int(predictions["button_logits"].shape[0])
        return (0, total) if total > 0 else None
    valid_indices = np.flatnonzero(valid_mask.astype(bool))
    if valid_indices.size == 0:
        return None
    start = int(valid_indices[0])
    end = int(valid_indices[-1]) + 1
    if (end - start) != int(valid_indices.size):
        raise RuntimeError("Expected contiguous valid IDM predictions for center-window inference.")
    return start, end


def decode_prediction_tracks(
    predictions: Dict[str, np.ndarray],
    idm_cfg: InverseDynamicsConfig,
    *,
    button_progress_desc: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_frames = int(predictions["button_logits"].shape[0])
    decoded_buttons = np.zeros((total_frames, idm_cfg.num_bin), dtype=np.float32)
    mouse_delta = np.zeros((total_frames, 2), dtype=np.float32)
    scroll_delta = np.zeros((total_frames, idm_cfg.num_scroll), dtype=np.float32)
    span = _valid_prediction_span(predictions)
    if span is None:
        return decoded_buttons, mouse_delta, scroll_delta
    start, end = span
    decoded_buttons[start:end] = decode_button_states(
        predictions["button_logits"][start:end],
        idm_cfg,
        progress_desc=button_progress_desc,
    ).astype(np.float32)
    mouse_delta[start:end] = apply_mouse_activity_gate(
        predictions["mouse_delta"][start:end],
        predictions["mouse_active_logits"][start:end],
        idm_cfg,
    ).astype(np.float32)
    if idm_cfg.num_scroll > 0:
        scroll_delta[start:end] = np.maximum(predictions["scroll_delta"][start:end], 0.0).astype(np.float32)
    return decoded_buttons, mouse_delta, scroll_delta


def write_label_csv(
    csv_path: str,
    decoded_buttons: np.ndarray,
    mouse_delta: np.ndarray,
    scroll_delta: np.ndarray,
    *,
    key_names: Sequence[str],
    mouse_button_names: Sequence[str],
    scroll_action_names: Sequence[str],
    dt: float,
) -> None:
    scroll_set = set(scroll_action_names)
    binary_mouse_names = [name for name in mouse_button_names if name not in scroll_set]
    with open(csv_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["timestamp", "dt"] + list(key_names) + list(mouse_button_names) + ["delta_x", "delta_y"])
        for idx in range(decoded_buttons.shape[0]):
            timestamp = float(idx + 1) * float(dt)
            row = [timestamp, float(dt)]
            row.extend(int(value >= 0.5) for value in decoded_buttons[idx, : len(key_names)])
            binary_values = decoded_buttons[idx, len(key_names) : len(key_names) + len(binary_mouse_names)]
            binary_by_name = {
                name: int(value >= 0.5)
                for name, value in zip(binary_mouse_names, binary_values)
            }
            scroll_by_name = {
                name: float(scroll_delta[idx, scroll_idx])
                for scroll_idx, name in enumerate(scroll_action_names)
                if scroll_idx < scroll_delta.shape[1]
            }
            for name in mouse_button_names:
                if name in scroll_set:
                    row.append(scroll_by_name.get(name, 0.0))
                else:
                    row.append(binary_by_name.get(name, 0))
            row.extend([float(mouse_delta[idx, 0]), float(mouse_delta[idx, 1])])
            writer.writerow(row)


def export_video_range(
    video_path: str,
    out_path: str,
    start_frame: int,
    end_frame: int,
    *,
    fps: float,
    ffmpeg_path: Optional[str],
    codec: str,
    quality: int,
    preset: str,
    progress_desc: Optional[str] = None,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for export: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Failed to seek to frame {start_frame} in {video_path}")
    writer = FFmpegPipeWriter(
        out_path,
        fps=fps,
        width=first_frame.shape[1],
        height=first_frame.shape[0],
        ffmpeg_path=ffmpeg_path,
        codec=codec,
        quality=quality,
        preset=preset,
    )
    total_frames = max(0, int(end_frame) - int(start_frame))
    pbar = tqdm(total=total_frames, desc=progress_desc, unit="frame", leave=False) if progress_desc else None
    try:
        writer.write(first_frame)
        if pbar is not None:
            pbar.update(1)
        for _ in range(start_frame + 1, end_frame):
            ret, frame_bgr = cap.read()
            if not ret:
                break
            writer.write(frame_bgr)
            if pbar is not None:
                pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()
        writer.close()
        cap.release()


def export_full_pseudo_run(
    source: SourceVideo,
    processed_video_path: str,
    predictions: Dict[str, np.ndarray],
    idm_cfg: InverseDynamicsConfig,
    cfg: PseudoLabelConfig,
    decoded_buttons: np.ndarray,
    mouse_delta: np.ndarray,
    scroll_delta: np.ndarray,
    *,
    progress_desc: Optional[str] = None,
) -> List[Dict]:
    os.makedirs(cfg.pseudo_root, exist_ok=True)
    total_frames = min(
        int(predictions["button_logits"].shape[0]),
        _frame_count(processed_video_path),
    )
    if total_frames <= 0:
        return []

    base_name = f"run_yt_{_sanitize_token(source.video_id)}"
    video_out = os.path.join(cfg.pseudo_root, f"{base_name}.mp4")
    csv_out = os.path.join(cfg.pseudo_root, f"{base_name}.csv")
    meta_out = os.path.join(cfg.pseudo_root, f"{base_name}.meta.json")

    export_video_range(
        processed_video_path,
        video_out,
        0,
        total_frames,
        fps=cfg.target_fps,
        ffmpeg_path=cfg.ffmpeg_path,
        codec=cfg.export_codec,
        quality=cfg.export_quality,
        preset=cfg.export_preset,
        progress_desc=progress_desc,
    )
    write_label_csv(
        csv_out,
        decoded_buttons[:total_frames],
        mouse_delta[:total_frames],
        scroll_delta[:total_frames],
        key_names=idm_cfg.key_names,
        mouse_button_names=idm_cfg.mouse_button_names,
        scroll_action_names=idm_cfg.scroll_action_names,
        dt=1.0 / float(cfg.target_fps),
    )

    valid_mask = predictions.get("valid_mask")
    if valid_mask is None:
        frame_confidence = np.ones((total_frames,), dtype=np.float32)
    else:
        frame_confidence = np.asarray(valid_mask[:total_frames], dtype=np.float32)

    metadata = {
        "source_url": source.source_url,
        "video_id": source.video_id,
        "title": source.title,
        "raw_path": source.raw_path,
        "processed_path": processed_video_path,
        "target_fps": float(cfg.target_fps),
        "start_frame": 0,
        "end_frame": int(total_frames),
        "start_seconds": 0.0,
        "end_seconds": float(total_frames / cfg.target_fps),
        "num_frames": int(total_frames),
        "mean_confidence": float(frame_confidence.mean()) if total_frames > 0 else 0.0,
        "frame_confidence": frame_confidence.astype(np.float32).tolist(),
    }
    with open(meta_out, "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2)
    return [metadata]


def process_source_video(
    source: SourceVideo,
    model: torch.nn.Module,
    idm_cfg: InverseDynamicsConfig,
    cfg: PseudoLabelConfig,
    device: torch.device,
    inference_dtype: torch.dtype,
    use_autocast: bool,
) -> List[Dict]:
    processed_video_path = transcode_to_domain(
        source,
        cfg,
        progress_desc=f"Transcode {source.video_id}",
    )
    predictions = infer_video_predictions(
        processed_video_path,
        model,
        idm_cfg,
        device,
        inference_dtype=inference_dtype,
        use_autocast=use_autocast,
        inference_batch_size=cfg.inference_batch_size,
        progress_desc=f"Infer {source.video_id}",
    )
    decoded_buttons, mouse_delta, scroll_delta = decode_prediction_tracks(
        predictions,
        idm_cfg,
        button_progress_desc=f"Decode buttons {source.video_id}",
    )
    exported = export_full_pseudo_run(
        source,
        processed_video_path,
        predictions,
        idm_cfg,
        cfg,
        decoded_buttons,
        mouse_delta,
        scroll_delta,
        progress_desc=f"Export full video {source.video_id}",
    )
    return exported


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download YouTube gameplay and export pseudo-labeled runs with the IDM.")
    parser.add_argument("url", nargs="?", default="https://www.youtube.com/live/1Bc5jfeoTYY?si=YILNoqHuAgh1qM_K", help="YouTube video or playlist URL.")
    parser.add_argument("--video-path", default=None, help="Optional local video path to skip downloading.")
    parser.add_argument("--checkpoint", default="./checkpoints_idm/model_latest.pt", help="Inverse-dynamics checkpoint path.")
    parser.add_argument("--pseudo-root", default=None, help="Output root for pseudo-labeled runs. Defaults to data_pseudo/<selected_game>.")
    parser.add_argument("--download-root", default=None, help="Root for raw and processed YouTube downloads. Defaults to downloads_youtube/<selected_game>.")
    parser.add_argument(
        "--target-fps",
        type=float,
        default=20.0,
        help="Resample videos to this FPS while preserving playback speed before labeling.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=None,
        help="Square resize for IDM input/export. Defaults to the checkpoint model_size.",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=8,
        help="Number of sliding IDM windows to run per forward pass.",
    )
    parser.add_argument("--compile-idm", dest="compile_model", action="store_true", default=True, help="Compile the IDM for faster long-video inference.")
    parser.add_argument("--no-compile-idm", dest="compile_model", action="store_false", help="Disable torch.compile for IDM inference.")
    parser.add_argument("--compile-mode", default="default", help="Optional torch.compile mode for IDM inference.")
    parser.add_argument("--export-codec", default="hevc_nvenc", help="FFmpeg video encoder for processed/exported MP4s.")
    parser.add_argument("--export-quality", type=int, default=20, help="NVENC CQ or x264 CRF value.")
    parser.add_argument("--export-preset", default="p4", help="FFmpeg encoder preset.")
    parser.add_argument("--ffmpeg-path", default=None, help="Optional explicit ffmpeg path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.url and not args.video_path:
        raise ValueError("Provide a YouTube URL or --video-path.")

    cfg = PseudoLabelConfig(
        checkpoint_path=args.checkpoint,
        pseudo_root=args.pseudo_root,
        download_root=args.download_root,
        target_size=args.target_size if args.target_size is not None else 512,
        target_fps=args.target_fps,
        ffmpeg_path=args.ffmpeg_path,
        inference_batch_size=args.inference_batch_size,
        compile_model=args.compile_model,
        compile_mode=args.compile_mode,
        export_codec=args.export_codec,
        export_quality=args.export_quality,
        export_preset=args.export_preset,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, idm_cfg, inference_dtype, use_autocast = load_inverse_model(
        cfg.checkpoint_path,
        device,
        compile_model=cfg.compile_model,
        compile_mode=cfg.compile_mode,
    )
    configure_idm_inference(idm_cfg, target_fps=cfg.target_fps)
    if args.target_size is None:
        cfg.target_size = int(idm_cfg.model_size)

    if args.video_path:
        source = SourceVideo(
            source_url=args.video_path,
            video_id=_sanitize_token(os.path.splitext(os.path.basename(args.video_path))[0]),
            title=os.path.basename(args.video_path),
            raw_path=os.path.abspath(args.video_path),
            info_path=None,
        )
        exported = process_source_video(source, model, idm_cfg, cfg, device, inference_dtype, use_autocast)
        print(f"Exported {len(exported)} full-video pseudo-labeled run(s) from local video.")
        return

    sources = download_youtube(args.url, cfg)
    if not sources:
        raise RuntimeError("No videos were downloaded from the provided URL.")

    total_exported = 0
    for source in tqdm(sources, desc="Source videos", unit="video"):
        tqdm.write(f"Processing {source.video_id}: {source.title}")
        exported = process_source_video(source, model, idm_cfg, cfg, device, inference_dtype, use_autocast)
        tqdm.write(f"  Exported {len(exported)} full-video pseudo-labeled run(s).")
        total_exported += len(exported)
    print(f"Finished. Exported {total_exported} full-video pseudo-labeled run(s) to {cfg.pseudo_root}")


if __name__ == "__main__":
    main()
