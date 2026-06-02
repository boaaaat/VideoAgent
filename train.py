import argparse
import csv
import glob
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

# WSL:
#   cd ~/ai
#   source venv/bin/activate
#   cd /mnt/c/Users/Abhil/Desktop/Github_Projects/VideoAgent/
#   python train.py

from nvidia.dali import pipeline_def, types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from action_space import game_data_root
from augmentations import augment_frames
from dataset_wsl_sync import sync_dataset_for_training
from models import (
    DrivingVideoPolicy,
    ModelConfig,
    PolicyOutput,
    TemporalState,
)


@dataclass
class TrainConfig(ModelConfig):
    batch_size: int = 1
    target_effective_batch: int = 16
    grad_accum: int = 16
    num_epochs: int = 100

    lr: float = 2e-4
    min_lr: float = 1e-5
    warmup_steps: int = 50
    weight_decay: float = 0.03
    grad_clip: float = 1.0

    amp_dtype: str = "bf16"
    compile_model: bool = True
    compile_mode: str = "default"

    train_split: float = 0.8
    split_seed: int = 1337
    pos_weight_power: float = 0.8
    pos_weight_clamp: float = 25
    button_threshold_from_pos_weight: bool = False
    button_threshold_min: float = 0.2
    button_threshold_max: float = 0.9

    button_loss_weight: float = 1.0
    streaming_state_training: bool = True
    action_label_offset: int = 0
    last_action_sequence_dropout: float = 0.25
    last_action_key_dropout: float = 0.8
    last_action_corruption_prob: float = 0.05
    skipped_key_names: Optional[Sequence[str]] = ("e", "q", "c", "z")
    button_label_smoothing: float = 0.02

    aug_brightness: float = 0.1
    aug_contrast: float = 0.15
    aug_noise_std: float = 0.006
    aug_gray_prob: float = 0.03
    aug_translate_frac: float = 0.0
    aug_scale_frac: float = 0.0
    aug_edges_crop_prob: float = 0.0
    aug_edges_crop_min_frac: float = 0.0
    aug_edges_crop_max_frac: float = 0.0
    aug_cutout_prob: float = 0.0
    aug_cutout_min_frac: float = 0.04
    aug_cutout_max_frac: float = 0.10
    aug_cutout_count: int = 1

    early_stop_patience: int = 0

    dali_num_threads: int = 6
    dali_prefetch_queue_depth: int = 4
    dali_reader_prefetch_queue_depth: int = 4
    dali_read_ahead: bool = False
    dali_dont_use_mmap: bool = False
    dali_resize_mode: str = "video_then_resize"
    dali_prepare_first_batch: bool = True
    dali_train_random_shuffle: bool = True
    dali_val_random_shuffle: bool = False
    dali_shuffle_seed: int = 1337
    sync_dataset: bool = True
    dataset_cache_root: Optional[str] = None
    dataset_sync_delete_stale: Optional[bool] = None
    dataset_sync_hash_same_size: bool = True

    resume: bool = True
    resume_path: Optional[str] = None
    ckpt_dir: str = "./checkpoints_rt"
    save_every: int = 1
    print_every: int = 20
    max_train_batches: Optional[int] = None
    max_val_batches: Optional[int] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        skipped = tuple(str(name) for name in (self.skipped_key_names or ()))
        if skipped:
            skip_set = set(skipped)
            self.key_names = [name for name in self.key_names if name not in skip_set]
            self.num_bin = len(self.key_names) + len(self.mouse_button_names)
            if self.num_bin <= 0:
                raise ValueError("At least one action key/button must remain after skipped_key_names filtering.")
            if self.button_state_thresholds is None or len(tuple(self.button_state_thresholds)) != self.num_bin:
                self.button_state_thresholds = tuple(float(self.button_state_threshold) for _ in range(self.num_bin))
        self.skipped_key_names = skipped
        self.batch_size = max(1, int(self.batch_size))
        self.target_effective_batch = max(1, int(self.target_effective_batch))
        self.num_epochs = max(1, int(self.num_epochs))
        self.dali_prefetch_queue_depth = max(1, int(self.dali_prefetch_queue_depth))
        self.dali_reader_prefetch_queue_depth = max(1, int(self.dali_reader_prefetch_queue_depth))
        self.dali_train_random_shuffle = bool(self.dali_train_random_shuffle)
        self.dali_val_random_shuffle = bool(self.dali_val_random_shuffle)
        self.save_every = max(1, int(self.save_every))
        self.print_every = max(1, int(self.print_every))
        self.grad_accum = max(1, int(math.ceil(self.target_effective_batch / float(self.batch_size))))
        self.warmup_steps = max(0, int(self.warmup_steps))
        self.train_split = float(min(max(self.train_split, 0.05), 0.95))
        self.split_seed = int(self.split_seed)
        self.dali_shuffle_seed = int(self.dali_shuffle_seed)
        self.pos_weight_power = max(0.0, float(self.pos_weight_power))
        self.pos_weight_clamp = max(1.0, float(self.pos_weight_clamp))
        self.button_threshold_from_pos_weight = bool(self.button_threshold_from_pos_weight)
        self.button_threshold_min = float(min(max(self.button_threshold_min, 0.0), 1.0))
        self.button_threshold_max = float(min(max(self.button_threshold_max, self.button_threshold_min), 1.0))
        self.streaming_state_training = bool(self.streaming_state_training)
        self.grad_clip = max(0.0, float(self.grad_clip))
        self.action_label_offset = int(self.action_label_offset)
        self.last_action_sequence_dropout = float(min(max(self.last_action_sequence_dropout, 0.0), 1.0))
        self.last_action_key_dropout = float(min(max(self.last_action_key_dropout, 0.0), 1.0))
        self.last_action_corruption_prob = float(min(max(self.last_action_corruption_prob, 0.0), 1.0))
        self.button_label_smoothing = float(min(max(self.button_label_smoothing, 0.0), 0.2))
        self.aug_brightness = float(min(max(self.aug_brightness, 0.0), 0.5))
        self.aug_contrast = float(min(max(self.aug_contrast, 0.0), 0.5))
        self.aug_noise_std = float(min(max(self.aug_noise_std, 0.0), 0.1))
        self.aug_gray_prob = float(min(max(self.aug_gray_prob, 0.0), 1.0))
        self.aug_translate_frac = float(min(max(self.aug_translate_frac, 0.0), 0.25))
        self.aug_scale_frac = float(min(max(self.aug_scale_frac, 0.0), 0.50))
        self.aug_edges_crop_prob = float(min(max(self.aug_edges_crop_prob, 0.0), 1.0))
        self.aug_edges_crop_min_frac = float(min(max(self.aug_edges_crop_min_frac, 0.0), 0.45))
        self.aug_edges_crop_max_frac = float(min(max(self.aug_edges_crop_max_frac, self.aug_edges_crop_min_frac), 0.45))
        self.aug_cutout_prob = float(min(max(self.aug_cutout_prob, 0.0), 1.0))
        self.aug_cutout_min_frac = float(min(max(self.aug_cutout_min_frac, 0.0), 0.75))
        self.aug_cutout_max_frac = float(min(max(self.aug_cutout_max_frac, self.aug_cutout_min_frac), 0.75))
        self.aug_cutout_count = max(1, min(int(self.aug_cutout_count), 16))
        self.early_stop_patience = max(0, int(self.early_stop_patience))
        self.dali_resize_mode = str(self.dali_resize_mode).strip().lower()
        if self.dali_resize_mode not in {"video_resize", "video_then_resize", "none"}:
            raise ValueError(
                "dali_resize_mode must be one of: video_resize, video_then_resize, none; "
                f"got {self.dali_resize_mode!r}."
            )
        self.dali_num_threads = max(1, int(self.dali_num_threads))
        self.max_train_batches = None if self.max_train_batches is None else max(1, int(self.max_train_batches))
        self.max_val_batches = None if self.max_val_batches is None else max(1, int(self.max_val_batches))
        if not os.path.isabs(self.ckpt_dir):
            self.ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.ckpt_dir))
        os.makedirs(self.ckpt_dir, exist_ok=True)


@dataclass
class WindowTargets:
    button_horizon: torch.Tensor
    horizon_valid: torch.Tensor
    last_action: torch.Tensor
    meta: Optional[List[Tuple[str, int, int]]] = None


StreamingStateCache = Dict[str, Dict[int, torch.Tensor]]


@dataclass
class BinaryStats:
    num_classes: int
    device: torch.device

    def __post_init__(self) -> None:
        self.tp = torch.zeros(self.num_classes, dtype=torch.float64, device=self.device)
        self.fp = torch.zeros(self.num_classes, dtype=torch.float64, device=self.device)
        self.fn = torch.zeros(self.num_classes, dtype=torch.float64, device=self.device)
        self.total = torch.zeros(self.num_classes, dtype=torch.float64, device=self.device)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, true: torch.Tensor, valid: torch.Tensor) -> None:
        mask = valid.to(dtype=torch.float64).unsqueeze(-1)
        pred = pred.to(dtype=torch.float64)
        true = true.to(dtype=torch.float64)
        self.tp += (pred * true * mask).sum(dim=(0, 1))
        self.fp += (pred * (1.0 - true) * mask).sum(dim=(0, 1))
        self.fn += ((1.0 - pred) * true * mask).sum(dim=(0, 1))
        self.total += mask.sum(dim=(0, 1))

    def compute(self) -> Dict[str, float]:
        precision = self.tp / (self.tp + self.fp).clamp(min=1.0)
        recall = self.tp / (self.tp + self.fn).clamp(min=1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp(min=1e-8)
        support = self.tp + self.fn
        predicted = self.tp + self.fp
        measured = (support + predicted) > 0.0
        if bool(measured.any().item()):
            precision_mean = precision[measured].mean()
            recall_mean = recall[measured].mean()
            f1_mean = f1[measured].mean()
        else:
            precision_mean = precision.new_zeros(())
            recall_mean = recall.new_zeros(())
            f1_mean = f1.new_zeros(())
        return {
            "macro_f1": float(f1_mean.item()),
            "macro_precision": float(precision_mean.item()),
            "macro_recall": float(recall_mean.item()),
        }

    def per_class_f1(self) -> torch.Tensor:
        precision = self.tp / (self.tp + self.fp).clamp(min=1.0)
        recall = self.tp / (self.tp + self.fn).clamp(min=1.0)
        return 2.0 * precision * recall / (precision + recall).clamp(min=1e-8)

    def per_class_metrics(self) -> Dict[str, torch.Tensor]:
        precision = self.tp / (self.tp + self.fp).clamp(min=1.0)
        recall = self.tp / (self.tp + self.fn).clamp(min=1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp(min=1e-8)
        tn = (self.total - self.tp - self.fp - self.fn).clamp(min=0.0)
        accuracy = (self.tp + tn) / self.total.clamp(min=1.0)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": self.tp + self.fn,
            "predicted": self.tp + self.fp,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


def find_runs(data_root: str, video_ext: str, csv_ext: str) -> List[Tuple[str, str]]:
    video_files = sorted(glob.glob(os.path.join(data_root, f"run_*{video_ext}")))
    pairs: List[Tuple[str, str]] = []
    for video_path in video_files:
        csv_path = os.path.splitext(video_path)[0] + csv_ext
        if os.path.exists(csv_path):
            pairs.append((video_path, csv_path))
    return pairs


def split_runs(
    pairs: Sequence[Tuple[str, str]],
    train_split: float,
    seed: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    pairs = list(pairs)
    rng = random.Random(int(seed))
    rng.shuffle(pairs)
    if len(pairs) <= 1:
        return pairs, []
    split_idx = max(1, min(len(pairs) - 1, int(round(len(pairs) * float(train_split)))))
    return pairs[:split_idx], pairs[split_idx:]


def _parse_float(value: object) -> float:
    if isinstance(value, str):
        value = value.strip()
    return float(value)


def load_run_arrays(csv_path: str, cfg: TrainConfig) -> Dict[str, np.ndarray]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {csv_path}")
        fieldnames = list(reader.fieldnames)
        required = ["timestamp"] + list(cfg.key_names) + list(cfg.mouse_button_names)
        missing = [name for name in required if name not in fieldnames]
        if missing:
            raise RuntimeError(f"CSV schema mismatch for {csv_path}: missing columns={missing}")
        rows.extend(reader)

    if not rows:
        raise RuntimeError(f"CSV has no rows: {csv_path}")

    t = len(rows)
    buttons = np.zeros((t, cfg.num_bin), dtype=np.float32)

    for idx, row in enumerate(rows):
        col = 0
        for name in cfg.key_names:
            buttons[idx, col] = 1.0 if _parse_float(row[name]) > 0.5 else 0.0
            col += 1
        for name in cfg.mouse_button_names:
            buttons[idx, col] = 1.0 if _parse_float(row[name]) > 0.5 else 0.0
            col += 1
    return {"buttons": buttons}


def build_window_targets(
    pairs: Sequence[Tuple[str, str]],
    cfg: TrainConfig,
    *,
    stride: int,
    return_meta: bool = False,
) -> WindowTargets:
    button_windows: List[np.ndarray] = []
    valid_windows: List[np.ndarray] = []
    last_action_windows: List[np.ndarray] = []
    meta: List[Tuple[str, int, int]] = []

    horizon_offsets = tuple(int(offset) for offset in cfg.prediction_horizon_offsets)
    horizon = len(horizon_offsets)
    for video_path, csv_path in pairs:
        run = load_run_arrays(csv_path, cfg)
        buttons = run["buttons"]
        if buttons.shape[0] < cfg.seq_len:
            continue

        max_start = buttons.shape[0] - cfg.seq_len
        for start in range(0, max_start + 1, max(1, int(stride))):
            end = start + cfg.seq_len
            button_target = np.zeros((cfg.seq_len, horizon, cfg.num_bin), dtype=np.float32)
            valid_target = np.zeros((cfg.seq_len, horizon), dtype=np.float32)
            last_action = np.zeros((cfg.seq_len, cfg.num_bin), dtype=np.float32)

            frame_indices = np.arange(start, end)
            context_indices = frame_indices + int(cfg.action_label_offset)
            context_valid = (context_indices >= 0) & (context_indices < buttons.shape[0])
            if np.any(context_valid):
                last_action[context_valid] = buttons[context_indices[context_valid]]
            for horizon_idx, horizon_offset in enumerate(horizon_offsets):
                target_indices = frame_indices + horizon_offset + int(cfg.action_label_offset)
                prev_indices = target_indices - 1
                valid = (target_indices >= 0) & (target_indices < buttons.shape[0]) & (prev_indices >= 0)
                if np.any(valid):
                    valid_target_indices = target_indices[valid]
                    button_target[valid, horizon_idx] = buttons[valid_target_indices]
                    valid_target[valid, horizon_idx] = 1.0

            button_windows.append(button_target)
            valid_windows.append(valid_target)
            last_action_windows.append(last_action)
            if return_meta:
                meta.append((video_path, start, end))

    if not button_windows:
        raise RuntimeError("No training windows found. Check data_root, seq_len, and stride.")

    def stack(items: List[np.ndarray]) -> torch.Tensor:
        return torch.from_numpy(np.stack(items, axis=0)).float()

    return WindowTargets(
        button_horizon=stack(button_windows),
        horizon_valid=stack(valid_windows),
        last_action=stack(last_action_windows),
        meta=meta if return_meta else None,
    )


def write_window_file_list(
    meta: List[Tuple[str, int, int]],
    file_path: str,
    indices: Optional[Sequence[int]] = None,
) -> None:
    use_indices = list(range(len(meta))) if indices is None else [int(i) for i in indices]
    with open(file_path, "w", encoding="utf-8") as file_obj:
        for idx in use_indices:
            video_path, start, end = meta[idx]
            file_obj.write(f"{video_path} {idx} {start} {end}\n")


@pipeline_def
def video_pipeline(
    file_list=None,
    seq_len=None,
    resize_size=None,
    random_shuffle=False,
    reader_prefetch_queue_depth=1,
    read_ahead=False,
    dont_use_mmap=False,
    resize_mode="video_resize",
    reader_step=None,
    enable_frame_num="none",
    normalize_frames=True,
    enable_augmentation=False,
    color_prob=0.0,
    brightness_range=(1.0, 1.0),
    contrast_range=(1.0, 1.0),
    saturation_range=(1.0, 1.0),
    hue_deg_max=0.0,
    blur_prob=0.0,
    blur_sigma_range=(0.3, 0.8),
    noise_prob=0.0,
    noise_std_max=0.0,
    filenames=None,
    labels=None,
):
    del enable_augmentation, color_prob, brightness_range, contrast_range, saturation_range, hue_deg_max
    del blur_prob, blur_sigma_range, noise_prob, noise_std_max
    seq_len = int(seq_len)
    resize_size = int(resize_size)
    resize_mode = str(resize_mode)
    reader_step = seq_len if reader_step is None else int(reader_step)
    resized_bytes = seq_len * resize_size * resize_size * 3
    normalized_bytes = resized_bytes * 2

    if file_list is None and filenames is None:
        raise ValueError("video_pipeline requires either file_list or filenames.")
    if file_list is not None and filenames is not None:
        raise ValueError("video_pipeline accepts only one of file_list or filenames.")

    use_reader_resize = resize_mode in {"video_resize", "video_then_resize"}
    reader_kwargs = {
        "device": "gpu",
        "name": "Reader",
        "sequence_length": seq_len,
        "step": reader_step,
        "stride": 1,
        "prefetch_queue_depth": int(reader_prefetch_queue_depth),
        "read_ahead": bool(read_ahead),
        "dont_use_mmap": bool(dont_use_mmap),
        "random_shuffle": bool(random_shuffle),
        "image_type": types.RGB,
        "bytes_per_sample_hint": resized_bytes,
        "tensor_init_bytes": resized_bytes,
    }
    if use_reader_resize:
        reader_kwargs.update(
            {
                "resize_x": resize_size,
                "resize_y": resize_size,
                "interp_type": types.INTERP_LINEAR,
                "temp_buffer_hint": resized_bytes,
            }
        )
    else:
        reader_kwargs["pad_mode"] = "none"
    if file_list is not None:
        reader_kwargs["file_list"] = file_list
        if use_reader_resize:
            reader_kwargs["file_list_frame_num"] = True
        else:
            reader_kwargs.update(
                {
                    "file_list_format": "frames",
                    "file_list_include_end": False,
                }
            )
    else:
        reader_kwargs["filenames"] = filenames
        if labels is not None:
            reader_kwargs["labels"] = labels
    if enable_frame_num not in (None, False, "none"):
        reader_kwargs["enable_frame_num"] = enable_frame_num

    if use_reader_resize:
        vids = fn.readers.video_resize(**reader_kwargs)
    else:
        vids = fn.experimental.readers.video(**reader_kwargs)

    labels = None
    frame_nums = None
    if isinstance(vids, (tuple, list)):
        reader_outputs = vids
        vids, labels = reader_outputs[0], reader_outputs[1]
        if len(reader_outputs) > 2:
            frame_nums = reader_outputs[2]
    if not use_reader_resize:
        if resize_mode in {"video_resize", "video_then_resize"}:
            vids = fn.resize(
                vids,
                resize_x=resize_size,
                resize_y=resize_size,
                interp_type=types.INTERP_LINEAR,
                bytes_per_sample_hint=resized_bytes,
                temp_buffer_hint=resized_bytes,
            )
        elif resize_mode != "none":
            raise ValueError(f"Unknown DALI resize_mode: {resize_mode}")

    if normalize_frames:
        frames = fn.crop_mirror_normalize(
            vids,
            dtype=types.FLOAT16,
            output_layout="FCHW",
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
            bytes_per_sample_hint=normalized_bytes,
        )
    else:
        frames = vids
    if frame_nums is not None:
        return frames, labels, frame_nums
    return frames, labels


def ensure_fchw_layout(frames: torch.Tensor) -> torch.Tensor:
    if frames.dim() != 5:
        raise RuntimeError(f"Unexpected frame shape from DALI: {tuple(frames.shape)}")
    if frames.shape[2] == 3:
        return frames
    if frames.shape[-1] == 3:
        return frames.permute(0, 1, 4, 2, 3).contiguous()
    raise RuntimeError(f"Unexpected frame shape from DALI: {tuple(frames.shape)}")


def normalize_dali_labels(labels: torch.Tensor) -> torch.Tensor:
    return labels.reshape(-1).long()


def compute_pos_weight(
    labels: torch.Tensor,
    power: float,
    clamp: float,
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    labels = labels.float()
    if valid is None:
        mask = torch.ones_like(labels[..., :1])
    else:
        mask = valid.float().unsqueeze(-1)
    pos = (labels * mask).sum(dim=tuple(range(labels.dim() - 1)))
    total = mask.sum(dim=tuple(range(mask.dim() - 1))).clamp(min=1.0).to(device=pos.device, dtype=pos.dtype)
    neg = total.expand_as(pos) - pos
    weight = (neg / pos.clamp(min=1.0)).pow(float(power))
    return weight.clamp(min=1.0, max=float(clamp)).float()


def supervised_start_frame(seq_len: int) -> int:
    return max(0, int(seq_len) // 4)


def supervised_frame_range(seq_len: int) -> Tuple[int, int]:
    return supervised_start_frame(seq_len), max(0, int(seq_len))


def second_half_only(valid: torch.Tensor) -> torch.Tensor:
    if valid.dim() < 2:
        raise ValueError(f"Expected a time dimension in valid mask, got shape {tuple(valid.shape)}")
    time = torch.arange(valid.size(1), device=valid.device) >= supervised_start_frame(valid.size(1))
    view_shape = [1] * valid.dim()
    view_shape[1] = valid.size(1)
    time = time.view(*view_shape)
    if valid.dtype == torch.bool:
        return valid & time
    return valid * time.to(dtype=valid.dtype)


def decision_thresholds_from_pos_weight(pos_weight: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    """Undo the logit prior introduced by weighted BCE, softened by an alpha parameter."""
    if not bool(cfg.button_threshold_from_pos_weight):
        return torch.full_like(pos_weight.float(), float(cfg.button_state_threshold))
    
    alpha = 0.2 
    effective_weight = 1.0 + (pos_weight.float() - 1.0) * alpha
    
    thresholds = 1.0 / (effective_weight + 1.0)
    
    return thresholds.clamp(min=float(cfg.button_threshold_min), max=float(cfg.button_threshold_max))


def button_threshold_tensor(cfg: TrainConfig, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    thresholds = getattr(cfg, "button_state_thresholds", None)
    if thresholds is None:
        return torch.full((int(cfg.num_bin),), float(cfg.button_state_threshold), device=device, dtype=dtype)
    return torch.tensor(list(thresholds), device=device, dtype=dtype)


def warmup_cosine_lr(step: int, *, total_steps: int, base_lr: float, min_lr: float, warmup_steps: int) -> float:
    step = int(step)
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))
    if warmup_steps > 0 and step < warmup_steps:
        return float(base_lr) * float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr) + (float(base_lr) - float(min_lr)) * cosine


def resolve_amp_settings(amp: str) -> Tuple[torch.dtype, bool, bool]:
    amp = str(amp).lower().strip()
    if amp in {"fp32", "float32", "none"}:
        return torch.float32, False, False
    if amp == "bf16":
        return torch.bfloat16, True, False
    raise ValueError("Only bf16 and fp32 are supported by this trainer.")


def bundle_index(bundle: WindowTargets, indices: torch.Tensor) -> WindowTargets:
    return WindowTargets(
        button_horizon=bundle.button_horizon[indices],
        horizon_valid=bundle.horizon_valid[indices],
        last_action=bundle.last_action[indices],
        meta=None,
    )


def move_bundle_to_device(bundle: WindowTargets, device: torch.device) -> WindowTargets:
    return WindowTargets(
        button_horizon=bundle.button_horizon.to(device, non_blocking=True),
        horizon_valid=bundle.horizon_valid.to(device, non_blocking=True),
        last_action=bundle.last_action.to(device, non_blocking=True),
        meta=bundle.meta,
    )


def pin_bundle(bundle: WindowTargets) -> WindowTargets:
    return WindowTargets(
        button_horizon=bundle.button_horizon.pin_memory(),
        horizon_valid=bundle.horizon_valid.pin_memory(),
        last_action=bundle.last_action.pin_memory(),
        meta=bundle.meta,
    )


def compute_losses(
    output: PolicyOutput,
    targets: WindowTargets,
    cfg: TrainConfig,
    *,
    button_pos_weight: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    valid = second_half_only(targets.horizon_valid).float()
    valid_4d = valid.unsqueeze(-1)
    button_target = targets.button_horizon.float()
    if float(cfg.button_label_smoothing) > 0.0:
        eps = float(cfg.button_label_smoothing)
        button_target = button_target * (1.0 - eps) + 0.5 * eps
    button_logits = output.horizon_button_logits.float()
    button_loss_raw = F.binary_cross_entropy_with_logits(
        button_logits,
        button_target,
        pos_weight=button_pos_weight.view(1, 1, 1, -1).float(),
        reduction="none",
    )
    loss_weight = valid_4d
    horizon_count = int(button_logits.size(2))
    if horizon_count > 1:
        horizon_weight = torch.linspace(
            0.5,
            1.5,
            horizon_count,
            device=button_logits.device,
            dtype=button_logits.dtype,
        ).view(1, 1, horizon_count, 1)
        loss_weight = loss_weight * horizon_weight
    button_loss = (button_loss_raw * loss_weight).sum() / (loss_weight.sum() * cfg.num_bin).clamp(min=1.0)

    total = cfg.button_loss_weight * button_loss
    return total, {
        "button": button_loss.detach(),
    }


def regularize_last_action_context(last_action: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    prev_action = last_action.float()
    if prev_action.dim() != 3:
        raise ValueError(f"Expected last_action [B,T,C], got {tuple(prev_action.shape)}.")

    batch_size = int(prev_action.size(0))
    if batch_size > 1 and float(cfg.last_action_corruption_prob) > 0.0:
        replace = (
            torch.rand((batch_size, 1, 1), device=prev_action.device)
            < float(cfg.last_action_corruption_prob)
        )
        permuted = prev_action[torch.randperm(batch_size, device=prev_action.device)]
        prev_action = torch.where(replace, permuted, prev_action)

    if float(cfg.last_action_sequence_dropout) > 0.0:
        keep_sequence = (
            torch.rand((batch_size, 1, 1), device=prev_action.device)
            >= float(cfg.last_action_sequence_dropout)
        ).to(dtype=prev_action.dtype)
        prev_action = prev_action * keep_sequence

    if float(cfg.last_action_key_dropout) > 0.0:
        keep_key = (
            torch.rand((batch_size, 1, int(prev_action.size(-1))), device=prev_action.device)
            >= float(cfg.last_action_key_dropout)
        ).to(dtype=prev_action.dtype)
        prev_action = prev_action * keep_key

    return prev_action


@torch.no_grad()
def update_metrics(
    output: PolicyOutput,
    targets: WindowTargets,
    cfg: TrainConfig,
    step1_stats: BinaryStats,
    final_stats: BinaryStats,
) -> None:
    valid = second_half_only(targets.horizon_valid)
    step1_valid = valid[:, :, 0] > 0.5
    final_idx = int(cfg.prediction_horizon) - 1
    final_valid = valid[:, :, final_idx] > 0.5
    thresholds = button_threshold_tensor(cfg, device=output.horizon_button_logits.device).view(1, 1, -1)
    step1_pred = torch.sigmoid(output.horizon_button_logits[:, :, 0].float()) >= thresholds
    final_pred = torch.sigmoid(output.horizon_button_logits[:, :, final_idx].float()) >= thresholds
    step1_stats.update(step1_pred, targets.button_horizon[:, :, 0] > 0.5, step1_valid)
    final_stats.update(final_pred, targets.button_horizon[:, :, final_idx] > 0.5, final_valid)


def per_class_f1_summary(stats: BinaryStats, names: Sequence[str], count: int = 8) -> str:
    f1 = stats.per_class_f1().detach().cpu()
    support = (stats.tp + stats.fn).detach().cpu()
    predicted = (stats.tp + stats.fp).detach().cpu()
    measured = (support + predicted) > 0.0
    if f1.numel() == 0 or not bool(measured.any().item()):
        return ""
    measured_indices = torch.nonzero(measured, as_tuple=False).flatten()
    order = measured_indices[torch.argsort(f1[measured_indices])[: max(1, min(int(count), measured_indices.numel()))]]
    return "worst_f1=" + ", ".join(f"{names[int(idx)]}:{float(f1[int(idx)]):.3f}" for idx in order)


def binary_stats_rows(stats: BinaryStats, names: Sequence[str]) -> List[Dict[str, float | int | str]]:
    metrics = {key: value.detach().cpu() for key, value in stats.per_class_metrics().items()}
    rows: List[Dict[str, float | int | str]] = []
    for idx, name in enumerate(names):
        rows.append(
            {
                "name": str(name),
                "accuracy": float(metrics["accuracy"][idx].item()),
                "f1": float(metrics["f1"][idx].item()),
                "precision": float(metrics["precision"][idx].item()),
                "recall": float(metrics["recall"][idx].item()),
                "support": int(round(float(metrics["support"][idx].item()))),
                "predicted": int(round(float(metrics["predicted"][idx].item()))),
                "tp": int(round(float(metrics["tp"][idx].item()))),
                "fp": int(round(float(metrics["fp"][idx].item()))),
                "fn": int(round(float(metrics["fn"][idx].item()))),
            }
        )
    return rows


def persistence_baseline_metrics(targets: WindowTargets, cfg: TrainConfig) -> Dict[str, float]:
    stats = BinaryStats(cfg.num_bin, torch.device("cpu"))
    current = targets.button_horizon[:, :, 0].new_zeros(targets.button_horizon[:, :, 0].shape)
    current[:, 1:] = targets.button_horizon[:, :-1, 0]
    valid = second_half_only(targets.horizon_valid[:, :, 0] > 0.5)
    valid[:, 0] = False
    stats.update(current > 0.5, targets.button_horizon[:, :, 0] > 0.5, valid)
    result = stats.compute()
    result["per_class_summary"] = per_class_f1_summary(stats, list(cfg.key_names) + list(cfg.mouse_button_names))
    return result


def driving_score(metrics: Dict[str, float], cfg: TrainConfig) -> float:
    def score_rows(rows: object, fallback: float) -> float:
        if not isinstance(rows, list):
            return float(fallback)
        by_name = {str(row.get("name")): row for row in rows if isinstance(row, dict)}
        weights = {"w": 3.0, "a": 1.5, "d": 1.5, "s": 0.75}
        state_total = 0.0
        total_weight = 0.0
        for name, weight in weights.items():
            row = by_name.get(name)
            state_total += float((row or {}).get("f1", 0.0)) * float(weight)
            total_weight += float(weight)
        return state_total / max(total_weight, 1.0)

    step1_score = score_rows(metrics.get("step1_button_rows", []), metrics["step1_button_macro_f1"])
    if int(cfg.prediction_horizon) <= 1:
        return step1_score
    final_score = score_rows(metrics.get("final_button_rows", []), metrics["final_button_macro_f1"])
    return 0.2 * step1_score + 0.8 * final_score


def print_button_stats_table(title: str, rows: Sequence[Dict[str, float | int | str]]) -> None:
    print(title)
    print("  name                 acc     f1    prec    rec  support  pred  tp  fp  fn")
    for item in rows:
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


def make_dali_iterator(
    file_list: str,
    cfg: TrainConfig,
    *,
    batch_size: int,
    random_shuffle: bool,
    last_batch_policy,
):
    pipe = video_pipeline(
        batch_size=batch_size,
        num_threads=int(cfg.dali_num_threads),
        device_id=0,
        seed=int(cfg.dali_shuffle_seed),
        file_list=file_list,
        seq_len=cfg.seq_len,
        resize_size=cfg.model_size,
        resize_mode=cfg.dali_resize_mode,
        random_shuffle=random_shuffle,
        reader_prefetch_queue_depth=cfg.dali_reader_prefetch_queue_depth,
        read_ahead=cfg.dali_read_ahead,
        dont_use_mmap=cfg.dali_dont_use_mmap,
        prefetch_queue_depth=cfg.dali_prefetch_queue_depth,
        exec_async=True,
        exec_pipelined=True,
    )
    pipe.build()
    return DALIGenericIterator(
        [pipe],
        output_map=["frames", "labels"],
        reader_name="Reader",
        auto_reset=False,
        last_batch_policy=last_batch_policy,
        prepare_first_batch=cfg.dali_prepare_first_batch,
    )


def load_batch(iterator, targets: WindowTargets, device: torch.device, cfg: TrainConfig) -> Tuple[torch.Tensor, WindowTargets, torch.Tensor]:
    batch = next(iterator)[0]
    frames = ensure_fchw_layout(batch["frames"])
    if frames.device != device:
        frames = frames.to(device, non_blocking=True)
    amp_name = str(cfg.amp_dtype).strip().lower()
    if amp_name == "bf16" and frames.dtype != torch.bfloat16:
        frames = frames.to(dtype=torch.bfloat16)
    elif amp_name in {"fp32", "float32", "none"} and frames.dtype != torch.float32:
        frames = frames.float()
    labels = normalize_dali_labels(batch["labels"]).to(device, non_blocking=True)
    target = WindowTargets(
        button_horizon=targets.button_horizon[labels],
        horizon_valid=targets.horizon_valid[labels],
        last_action=targets.last_action[labels],
        meta=None,
    )
    return frames, target, labels


def _labels_to_indices(labels: torch.Tensor) -> List[int]:
    return [int(item) for item in labels.detach().cpu().reshape(-1).tolist()]


def _streaming_initial_state(
    labels: torch.Tensor,
    targets: WindowTargets,
    cache: StreamingStateCache,
) -> Optional[TemporalState]:
    if targets.meta is None or not cache:
        return None

    sample_states: List[Optional[torch.Tensor]] = []
    template: Optional[torch.Tensor] = None
    for label_idx in _labels_to_indices(labels):
        video_path, start, _ = targets.meta[label_idx]
        hidden = cache.get(video_path, {}).pop(int(start), None)
        sample_states.append(hidden)
        if hidden is not None and template is None:
            template = hidden

    if template is None:
        return None

    stacked = torch.stack(
        [hidden if hidden is not None else torch.zeros_like(template) for hidden in sample_states],
        dim=1,
    )
    return TemporalState(hidden_state=stacked)


def _update_streaming_cache(
    labels: torch.Tensor,
    targets: WindowTargets,
    state: Optional[TemporalState],
    cache: StreamingStateCache,
    cfg: TrainConfig,
) -> None:
    if targets.meta is None or state is None or state.hidden_state is None:
        return

    hidden = state.hidden_state.detach()
    max_pending = max(2, int(math.ceil(float(cfg.seq_len) / float(max(1, cfg.train_seq_stride)))) + 2)
    for batch_idx, label_idx in enumerate(_labels_to_indices(labels)):
        video_path, _, end = targets.meta[label_idx]
        video_cache = cache.setdefault(video_path, {})
        video_cache[int(end)] = hidden[:, batch_idx].detach()
        while len(video_cache) > max_pending:
            video_cache.pop(min(video_cache))



def run_epoch(
    *,
    desc: str,
    model: torch.nn.Module,
    iterator,
    batches: int,
    targets: WindowTargets,
    cfg: TrainConfig,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_autocast: bool,
    button_pos_weight: torch.Tensor,
    optimizer: Optional[torch.optim.Optimizer] = None,
    total_steps: int = 1,
    global_step: int = 0,
) -> Tuple[Dict[str, float], int]:
    is_train = optimizer is not None
    model.train(is_train)
    if is_train:
        optimizer.zero_grad(set_to_none=True)

    loss_sum = torch.zeros((), device=device)
    detail_sums = {name: torch.zeros((), device=device) for name in ("button",)}
    step1_stats = BinaryStats(cfg.num_bin, device)
    final_stats = BinaryStats(cfg.num_bin, device)
    steps = 0
    streaming_cache: StreamingStateCache = {}
    use_streaming_state = bool(is_train and cfg.streaming_state_training)

    iterator_it = iter(iterator)
    pbar = tqdm(range(int(batches)), desc=desc, dynamic_ncols=True)
    for batch_idx in pbar:
        frames, batch_targets, labels = load_batch(iterator_it, targets, device, cfg)
        if is_train:
            frames = augment_frames(frames, cfg)
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_autocast):
                prev_action = (
                    regularize_last_action_context(batch_targets.last_action, cfg)
                    if is_train
                    else batch_targets.last_action
                )
                if use_streaming_state:
                    initial_state = _streaming_initial_state(labels, targets, streaming_cache)
                    output, next_state = model(
                        frames,
                        state=initial_state,
                        return_aux=True,
                        prev_action=prev_action,
                    )
                    _update_streaming_cache(labels, targets, next_state, streaming_cache, cfg)
                else:
                    output = model(frames, prev_action=prev_action)
                loss, details = compute_losses(
                    output,
                    batch_targets,
                    cfg,
                    button_pos_weight=button_pos_weight,
                )
                loss_div = loss / max(1, int(cfg.grad_accum))

            if is_train:
                loss_div.backward()
                if ((batch_idx + 1) % cfg.grad_accum == 0) or (batch_idx + 1 == batches):
                    if cfg.grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
                    lr = warmup_cosine_lr(
                        global_step,
                        total_steps=total_steps,
                        base_lr=cfg.lr,
                        min_lr=cfg.min_lr,
                        warmup_steps=cfg.warmup_steps,
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

        update_metrics(
            output,
            batch_targets,
            cfg,
            step1_stats,
            final_stats,
        )
        loss_sum += loss.detach().float()
        for name, value in details.items():
            detail_sums[name] += value.float()
        steps += 1
        if cfg.print_every and ((batch_idx + 1) % cfg.print_every == 0 or batch_idx + 1 == batches):
            pbar.set_postfix(
                {
                    "loss": float((loss_sum / max(1, steps)).item()),
                    "btn": float((detail_sums["button"] / max(1, steps)).item()),
                }
            )
    iterator.reset()

    step1 = step1_stats.compute()
    final = final_stats.compute()
    button_names = list(cfg.key_names) + list(cfg.mouse_button_names)
    metrics = {
        "loss": float((loss_sum / max(1, steps)).item()),
        "button_loss": float((detail_sums["button"] / max(1, steps)).item()),
        "step1_button_macro_f1": step1["macro_f1"],
        "step1_button_macro_precision": step1["macro_precision"],
        "step1_button_macro_recall": step1["macro_recall"],
        "final_button_macro_f1": final["macro_f1"],
        "final_button_macro_precision": final["macro_precision"],
        "final_button_macro_recall": final["macro_recall"],
        "per_class_summary": per_class_f1_summary(step1_stats, button_names),
        "step1_button_rows": binary_stats_rows(step1_stats, button_names),
        "final_button_rows": binary_stats_rows(final_stats, button_names),
    }
    return metrics, global_step


def latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    path = os.path.join(ckpt_dir, "model_latest.pt")
    if os.path.exists(path):
        return path
    return None


def save_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    epoch: int,
    global_step: int,
    best_score: float,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(cfg),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "best_score": float(best_score),
        },
        path,
    )


def maybe_resume(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    device: torch.device,
) -> Tuple[int, int, float]:
    if not cfg.resume:
        return 0, 0, -1e9
    if cfg.resume_path is not None:
        ckpt_path = cfg.resume_path
    else:
        ckpt_path = latest_checkpoint(cfg.ckpt_dir)
    if not ckpt_path:
        return 0, 0, -1e9
    print(f"Resuming from {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(state["model_state"])
    except RuntimeError as exc:
        if cfg.resume_path is None:
            print(
                f"Skipping incompatible checkpoint {ckpt_path}. "
                "Starting a fresh run for the current model/action configuration."
            )
            return 0, 0, -1e9
        raise RuntimeError(
            f"Cannot resume checkpoint {ckpt_path}: it does not match the current CNN+temporal model. "
            "Start a fresh run or pass --no-resume."
        ) from exc
    optimizer.load_state_dict(state["optimizer_state"])
    return int(state["epoch"]), int(state["global_step"]), float(state["best_score"])


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the CNN+temporal behavioral cloning policy.")
    add = parser.add_argument
    add("--data-root", default=None)
    add("--ckpt-dir", default=None)
    add("--resume", dest="resume", action="store_true", default=None)
    add("--no-resume", dest="resume", action="store_false")
    add("--resume-path", default=None)
    add("--num-epochs", type=int, default=None)
    add("--batch-size", type=int, default=None)
    add("--seq-len", type=int, default=None)
    add("--prediction-horizon", type=int, default=None)
    add("--prediction-horizon-offsets", default=None, help="Comma-separated future frame offsets for horizon heads.")
    add("--model-size", type=int, default=None)
    add("--d-model", type=int, default=None)
    add("--spatial-dropout", type=float, default=None)
    add("--head-dropout", type=float, default=None)
    add("--zoneout", type=float, default=None)
    add("--train-seq-stride", type=int, default=None)
    add("--val-seq-stride", type=int, default=None)
    add("--target-effective-batch", type=int, default=None)
    add("--lr", type=float, default=None)
    add("--min-lr", type=float, default=None)
    add("--warmup-steps", type=int, default=None)
    add("--weight-decay", type=float, default=None)
    add("--train-split", type=float, default=None)
    add("--pos-weight-power", type=float, default=None)
    add("--pos-weight-clamp", type=float, default=None)
    add("--button-threshold-from-pos-weight", dest="button_threshold_from_pos_weight", action="store_true", default=None)
    add("--flat-button-threshold", dest="button_threshold_from_pos_weight", action="store_false", default=None)
    add("--button-threshold-min", type=float, default=None)
    add("--button-threshold-max", type=float, default=None)
    add("--streaming-state-training", dest="streaming_state_training", action="store_true", default=None)
    add("--no-streaming-state-training", dest="streaming_state_training", action="store_false")
    add("--action-label-offset", type=int, default=None)
    add("--last-action-sequence-dropout", type=float, default=None)
    add("--last-action-key-dropout", type=float, default=None)
    add("--last-action-corruption-prob", type=float, default=None)
    add("--skip-key-names", default=None, help="Comma-separated key names to exclude from training labels.")
    add("--train-all-keys", action="store_true", help="Disable the default Greenville test filter for e,q,c,z.")
    add("--button-label-smoothing", type=float, default=None)
    add("--aug-brightness", type=float, default=None)
    add("--aug-contrast", type=float, default=None)
    add("--aug-noise-std", type=float, default=None)
    add("--aug-gray-prob", type=float, default=None)
    add("--aug-translate-frac", type=float, default=None)
    add("--aug-scale-frac", type=float, default=None)
    add("--aug-edges-crop-prob", type=float, default=None)
    add("--aug-edges-crop-min-frac", type=float, default=None)
    add("--aug-edges-crop-max-frac", type=float, default=None)
    add("--aug-cutout-prob", type=float, default=None)
    add("--aug-cutout-min-frac", type=float, default=None)
    add("--aug-cutout-max-frac", type=float, default=None)
    add("--aug-cutout-count", type=int, default=None)
    add("--early-stop-patience", type=int, default=None)
    add("--max-train-batches", type=int, default=None)
    add("--max-val-batches", type=int, default=None)
    add("--dali-num-threads", type=int, default=None)
    add("--dali-prefetch-queue-depth", type=int, default=None)
    add("--dali-reader-prefetch-queue-depth", type=int, default=None)
    add("--amp-dtype", choices=["bf16", "fp32", "float32"], default=None)
    add("--compile", dest="compile_model", action="store_true", default=None)
    add("--no-compile", dest="compile_model", action="store_false", default=None)
    add("--compile-mode", default=None)
    add("--dali-resize-mode", choices=["video_resize", "video_then_resize", "none"], default=None)
    add("--dali-train-random-shuffle", dest="dali_train_random_shuffle", action="store_true", default=None)
    add("--no-dali-train-random-shuffle", dest="dali_train_random_shuffle", action="store_false")
    add("--dali-val-random-shuffle", dest="dali_val_random_shuffle", action="store_true", default=None)
    add("--no-dali-val-random-shuffle", dest="dali_val_random_shuffle", action="store_false")
    add("--dali-read-ahead", dest="dali_read_ahead", action="store_true")
    add("--no-dali-read-ahead", dest="dali_read_ahead", action="store_false")
    add("--dali-dont-use-mmap", dest="dali_dont_use_mmap", action="store_true")
    add("--dali-use-mmap", dest="dali_dont_use_mmap", action="store_false")
    add("--dataset-cache-root", default=None, help="Local Linux dataset cache. Defaults to ~/ai/dataset in WSL, ./dataset elsewhere.")
    add("--sync-dataset", dest="sync_dataset", action="store_true")
    add("--no-sync-dataset", dest="sync_dataset", action="store_false")
    add("--dataset-sync-delete-stale", dest="dataset_sync_delete_stale", action="store_true")
    add("--dataset-sync-keep-stale", dest="dataset_sync_delete_stale", action="store_false")
    add("--dataset-sync-hash-same-size", dest="dataset_sync_hash_same_size", action="store_true")
    add("--dataset-sync-no-hash-same-size", dest="dataset_sync_hash_same_size", action="store_false")
    parser.set_defaults(
        dali_read_ahead=None,
        dali_dont_use_mmap=None,
        sync_dataset=None,
        dataset_sync_delete_stale=None,
        dataset_sync_hash_same_size=None,
    )
    args = parser.parse_args()

    args_by_name = vars(args)
    kwargs = {}
    for key in (
        "data_root",
        "ckpt_dir",
        "resume",
        "resume_path",
        "num_epochs",
        "batch_size",
        "seq_len",
        "prediction_horizon",
        "prediction_horizon_offsets",
        "model_size",
        "d_model",
        "spatial_dropout",
        "head_dropout",
        "zoneout",
        "train_seq_stride",
        "val_seq_stride",
        "target_effective_batch",
        "lr",
        "min_lr",
        "warmup_steps",
        "weight_decay",
        "train_split",
        "pos_weight_power",
        "pos_weight_clamp",
        "button_threshold_from_pos_weight",
        "button_threshold_min",
        "button_threshold_max",
        "streaming_state_training",
        "action_label_offset",
        "last_action_sequence_dropout",
        "last_action_key_dropout",
        "last_action_corruption_prob",
        "button_label_smoothing",
        "aug_brightness",
        "aug_contrast",
        "aug_noise_std",
        "aug_gray_prob",
        "aug_translate_frac",
        "aug_scale_frac",
        "aug_edges_crop_prob",
        "aug_edges_crop_min_frac",
        "aug_edges_crop_max_frac",
        "aug_cutout_prob",
        "aug_cutout_min_frac",
        "aug_cutout_max_frac",
        "aug_cutout_count",
        "early_stop_patience",
        "max_train_batches",
        "max_val_batches",
        "dali_num_threads",
        "dali_prefetch_queue_depth",
        "dali_reader_prefetch_queue_depth",
        "amp_dtype",
        "compile_model",
        "compile_mode",
        "dali_resize_mode",
        "dali_train_random_shuffle",
        "dali_val_random_shuffle",
        "dali_read_ahead",
        "dali_dont_use_mmap",
        "dataset_cache_root",
        "sync_dataset",
        "dataset_sync_delete_stale",
        "dataset_sync_hash_same_size",
    ):
        value = args_by_name[key]
        if value is not None:
            kwargs[key] = value
    if args.train_all_keys:
        kwargs["skipped_key_names"] = ()
    elif args.skip_key_names is not None:
        kwargs["skipped_key_names"] = tuple(
            name.strip() for name in str(args.skip_key_names).split(",") if name.strip()
        )
    if args.prediction_horizon_offsets is not None:
        kwargs["prediction_horizon_offsets"] = tuple(
            int(part.strip())
            for part in str(args.prediction_horizon_offsets).split(",")
            if part.strip()
        )
    return TrainConfig(**kwargs)


def train() -> None:
    cfg = parse_args()
    if cfg.data_root is None:
        cfg.data_root = game_data_root(cfg.selected_game)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DALI video training.")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    if cfg.sync_dataset:
        cfg.data_root = sync_dataset_for_training(
            data_root=cfg.data_root,
            target_root=cfg.dataset_cache_root,
            video_ext=cfg.video_ext,
            csv_ext=cfg.csv_ext,
            delete_stale=cfg.dataset_sync_delete_stale,
            hash_same_size=bool(cfg.dataset_sync_hash_same_size),
        )

    pairs = find_runs(cfg.data_root, cfg.video_ext, cfg.csv_ext)
    if not pairs:
        raise RuntimeError(f"No runs found under {cfg.data_root!r}.")
    train_pairs, val_pairs = split_runs(pairs, cfg.train_split, cfg.split_seed)
    print(f"Runs: total={len(pairs)} train={len(train_pairs)} val={len(val_pairs)}")
    if cfg.skipped_key_names:
        print(f"Skipping action keys for this training run: {', '.join(cfg.skipped_key_names)}")
    print(f"Training action keys: {', '.join(cfg.key_names + cfg.mouse_button_names)}")
    horizon_offsets = tuple(int(offset) for offset in cfg.prediction_horizon_offsets)
    first_horizon_offset = int(horizon_offsets[0])
    print(f"Horizon frame offsets: {', '.join(str(offset) for offset in horizon_offsets)}")

    train_targets = build_window_targets(train_pairs, cfg, stride=cfg.train_seq_stride, return_meta=True)
    val_targets = build_window_targets(val_pairs, cfg, stride=cfg.val_seq_stride, return_meta=True) if val_pairs else None
    print(
        "Windows:",
        f"train={tuple(train_targets.button_horizon.shape)}",
        f"val={(tuple(val_targets.button_horizon.shape) if val_targets is not None else None)}",
        f"action_label_offset={cfg.action_label_offset}",
        f"horizon_offsets={horizon_offsets}",
    )
    supervised_start, supervised_end = supervised_frame_range(cfg.seq_len)
    print(
        "Loss supervision:",
        f"frames={supervised_start}-{supervised_end}",
    )
    print(
        "Streaming state training:",
        f"enabled={bool(cfg.streaming_state_training)}",
        f"requires_contiguous_chunks=True",
        f"train_shuffle={bool(cfg.dali_train_random_shuffle)}",
        f"train_stride={int(cfg.train_seq_stride)}",
        f"seq_len={int(cfg.seq_len)}",
    )
    print(
        "Last action conditioning:",
        "enabled=True",
        f"seq_drop={cfg.last_action_sequence_dropout:.2f}",
        f"key_drop={cfg.last_action_key_dropout:.2f}",
        f"corrupt={cfg.last_action_corruption_prob:.2f}",
    )
    train_persist = persistence_baseline_metrics(train_targets, cfg)
    print(f"Persistence baseline: train_f1@+{first_horizon_offset}={train_persist['macro_f1']:.4f}")
    if val_targets is not None:
        val_persist = persistence_baseline_metrics(val_targets, cfg)
        print(f"Persistence baseline: val_f1@+{first_horizon_offset}={val_persist['macro_f1']:.4f} {val_persist['per_class_summary']}")

    train_file_list = os.path.join(cfg.ckpt_dir, "train_file_list.txt")
    val_file_list = os.path.join(cfg.ckpt_dir, "val_file_list.txt")
    if train_targets.meta is None:
        raise RuntimeError("Training window metadata is required for DALI file list generation.")
    write_window_file_list(train_targets.meta, train_file_list)
    if val_targets is not None:
        if val_targets.meta is None:
            raise RuntimeError("Validation window metadata is required for DALI file list generation.")
        write_window_file_list(val_targets.meta, val_file_list)

    button_pos_weight = compute_pos_weight(
        train_targets.button_horizon,
        cfg.pos_weight_power,
        cfg.pos_weight_clamp,
        valid=second_half_only(train_targets.horizon_valid),
    ).to(device)
    button_thresholds = decision_thresholds_from_pos_weight(button_pos_weight, cfg).detach().cpu()
    cfg.button_state_thresholds = tuple(float(x) for x in list(button_thresholds.tolist()))
    button_names = list(cfg.key_names) + list(cfg.mouse_button_names)
    print(
        "Class weighting:",
        '   '.join([f"{name}={float(weight):.3f}" for name, weight in zip(button_names, button_pos_weight.tolist())]),
    )
    threshold_parts = [
        f"{name}={float(threshold):.3f}"
        for name, threshold in zip(button_names, cfg.button_state_thresholds)
    ]
    print("Button decision thresholds:", " ".join(threshold_parts))

    train_targets = move_bundle_to_device(train_targets, device)
    if val_targets is not None:
        val_targets = move_bundle_to_device(val_targets, device)

    train_iter = make_dali_iterator(
        train_file_list,
        cfg,
        batch_size=cfg.batch_size,
        random_shuffle=cfg.dali_train_random_shuffle,
        last_batch_policy=LastBatchPolicy.DROP,
    )
    train_batches = int(train_targets.button_horizon.shape[0]) // int(cfg.batch_size)
    if cfg.max_train_batches is not None:
        train_batches = min(train_batches, cfg.max_train_batches)
    if train_batches <= 0:
        raise RuntimeError("No training batches available.")

    val_iter = None
    val_batches = 0
    if val_targets is not None:
        val_batch_size = min(max(1, cfg.batch_size), int(val_targets.button_horizon.shape[0]))
        val_iter = make_dali_iterator(
            val_file_list,
            cfg,
            batch_size=val_batch_size,
            random_shuffle=cfg.dali_val_random_shuffle,
            last_batch_policy=LastBatchPolicy.PARTIAL,
        )
        val_batches = int(math.ceil(int(val_targets.button_horizon.shape[0]) / float(val_batch_size)))
        if cfg.max_val_batches is not None:
            val_batches = min(val_batches, cfg.max_val_batches)

    amp_dtype, use_autocast, use_scaler = resolve_amp_settings(cfg.amp_dtype)
    if use_scaler:
        raise RuntimeError("This trainer supports bf16/fp32 only.")
    print(f"AMP: dtype={amp_dtype} autocast={use_autocast}")

    base_model: torch.nn.Module = DrivingVideoPolicy(cfg).to(device)
    base_model = base_model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, fused=True)
    print(f"Parameters: {sum(p.numel() for p in base_model.parameters()) / 1e6:.4f}M")
    start_epoch, global_step, best_score = maybe_resume(base_model, optimizer, cfg, device)

    if cfg.compile_model:
        compile_kwargs = {"fullgraph": False, "dynamic": False}
        if cfg.compile_mode and cfg.compile_mode.lower() != "default":
            compile_kwargs["mode"] = cfg.compile_mode
        model = torch.compile(base_model, **compile_kwargs)
        print(f"torch.compile enabled: mode={cfg.compile_mode}")
    else:
        model = base_model

    optimizer_steps_per_epoch = int(math.ceil(train_batches / float(cfg.grad_accum)))
    total_steps = max(1, optimizer_steps_per_epoch * cfg.num_epochs)
    epochs_without_improvement = 0
    final_horizon_offset = int(horizon_offsets[-1])

    for epoch in range(start_epoch, cfg.num_epochs):
        train_metrics, global_step = run_epoch(
            desc=f"Epoch {epoch + 1}/{cfg.num_epochs} [train]",
            model=model,
            iterator=train_iter,
            batches=train_batches,
            targets=train_targets,
            cfg=cfg,
            device=device,
            amp_dtype=amp_dtype,
            use_autocast=use_autocast,
            button_pos_weight=button_pos_weight,
            optimizer=optimizer,
            total_steps=total_steps,
            global_step=global_step,
        )

        val_metrics = None
        score = driving_score(train_metrics, cfg)
        if val_iter is not None and val_targets is not None and val_batches > 0:
            with torch.inference_mode():
                val_metrics, _ = run_epoch(
                    desc=f"Epoch {epoch + 1}/{cfg.num_epochs} [val]",
                    model=model,
                    iterator=val_iter,
                    batches=val_batches,
                    targets=val_targets,
                    cfg=cfg,
                    device=device,
                    amp_dtype=amp_dtype,
                    use_autocast=use_autocast,
                    button_pos_weight=button_pos_weight,
                )
            score = driving_score(val_metrics, cfg)

        improved = score > best_score
        if improved:
            best_score = score
            epochs_without_improvement = 0
            save_checkpoint(
                os.path.join(cfg.ckpt_dir, "model_best.pt"),
                model=base_model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=epoch + 1,
                global_step=global_step,
                best_score=best_score,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            os.path.join(cfg.ckpt_dir, "model_latest.pt"),
            model=base_model,
            optimizer=optimizer,
            cfg=cfg,
            epoch=epoch + 1,
            global_step=global_step,
            best_score=best_score,
        )
        if (epoch + 1) % cfg.save_every == 0:
            save_checkpoint(
                os.path.join(cfg.ckpt_dir, f"model_epoch_{epoch + 1}.pt"),
                model=base_model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=epoch + 1,
                global_step=global_step,
                best_score=best_score,
            )

        parts = [
            f"Epoch {epoch + 1}/{cfg.num_epochs}",
            f"tr_loss={train_metrics['loss']:.4f}",
            f"tr_f1@+{first_horizon_offset}={train_metrics['step1_button_macro_f1']:.4f}",
        ]
        if int(cfg.prediction_horizon) > 1:
            parts.append(f"tr_f1@+{final_horizon_offset}={train_metrics['final_button_macro_f1']:.4f}")
        if val_metrics is not None:
            parts.extend(
                [
                    f"va_loss={val_metrics['loss']:.4f}",
                    f"va_f1@+{first_horizon_offset}={val_metrics['step1_button_macro_f1']:.4f}",
                    f"best={best_score:.4f}",
                    val_metrics["per_class_summary"],
                ]
            )
            if int(cfg.prediction_horizon) > 1:
                parts.insert(-3, f"va_f1@+{final_horizon_offset}={val_metrics['final_button_macro_f1']:.4f}")
        else:
            parts.append(train_metrics["per_class_summary"])
        print(" | ".join(part for part in parts if part))
        train_stats = (
            f"Epoch {epoch + 1} train stats: "
            f"f1@+{first_horizon_offset}={train_metrics['step1_button_macro_f1']:.4f} "
            f"prec@+{first_horizon_offset}={train_metrics['step1_button_macro_precision']:.4f} "
            f"rec@+{first_horizon_offset}={train_metrics['step1_button_macro_recall']:.4f}"
        )
        if int(cfg.prediction_horizon) > 1:
            train_stats += (
                f" f1@+{final_horizon_offset}={train_metrics['final_button_macro_f1']:.4f}"
            )
        print(train_stats)
        print_button_stats_table(
            f"Epoch {epoch + 1} train per-key/button @+{first_horizon_offset}:",
            train_metrics["step1_button_rows"],
        )
        if int(cfg.prediction_horizon) > 1:
            print_button_stats_table(
                f"Epoch {epoch + 1} train per-key/button @+{final_horizon_offset}:",
                train_metrics["final_button_rows"],
            )
        if val_metrics is not None:
            val_stats = (
                f"Epoch {epoch + 1} val stats: "
                f"f1@+{first_horizon_offset}={val_metrics['step1_button_macro_f1']:.4f} "
                f"prec@+{first_horizon_offset}={val_metrics['step1_button_macro_precision']:.4f} "
                f"rec@+{first_horizon_offset}={val_metrics['step1_button_macro_recall']:.4f}"
            )
            if int(cfg.prediction_horizon) > 1:
                val_stats += (
                    f" f1@+{final_horizon_offset}={val_metrics['final_button_macro_f1']:.4f}"
                )
            print(val_stats)
            print_button_stats_table(
                f"Epoch {epoch + 1} val per-key/button @+{first_horizon_offset}:",
                val_metrics["step1_button_rows"],
            )
            if int(cfg.prediction_horizon) > 1:
                print_button_stats_table(
                    f"Epoch {epoch + 1} val per-key/button @+{final_horizon_offset}:",
                    val_metrics["final_button_rows"],
                )
        if (
            val_metrics is not None
            and int(cfg.early_stop_patience) > 0
            and epochs_without_improvement >= int(cfg.early_stop_patience)
        ):
            print(f"Early stopping after {epoch + 1} epochs; best score={best_score:.4f}.")
            break


if __name__ == "__main__":
    train()
