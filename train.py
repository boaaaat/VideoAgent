"""Stateful DALI trainer for the dense-supervision FPN + ConvGRU driving policy.

The DALI pipeline owns GPU video decode and resize. CSV labels are parsed once
on the host, then joined to decoded 80-frame windows through the DALI sample
label written into the file list. Stateful mode processes complete videos as
contiguous chunks and detaches the ConvGRU state at every chunk boundary.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import math
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from nvidia.dali import pipeline_def, types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

# WSL:
#   cd ~/ai
#   source venv/bin/activate
#   cd /mnt/c/Users/Abhil/Desktop/Github_Projects/VideoAgent/
#   python train.py

from augmentations import augment_frames
from models import (
    ARCHITECTURE_VERSION,
    FUSED_CHANNELS,
    READOUT_CHANNELS,
    DrivingVideoPolicy,
    ModelConfig,
    PolicyOutput,
    TemporalState,
)


CONFLICTING_ACTION_PAIRS = (("w", "s"), ("a", "d"))


def _positive_int(name: str, value: object, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}.")
    return result


def _finite_float(name: str, value: object, minimum: float, maximum: Optional[float] = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, got {value!r}.")
    result = float(value)
    if not math.isfinite(result) or result < minimum or (maximum is not None and result > maximum):
        ceiling = "infinity" if maximum is None else str(maximum)
        raise ValueError(f"{name} must be in [{minimum}, {ceiling}], got {value!r}.")
    return result


@dataclass
class TrainConfig(ModelConfig):
    """Training settings. The inherited model config fixes the six action names."""

    train_seq_stride: int = 80
    val_seq_stride: int = 80
    action_offset: int = 1
    streaming_state_training: bool = True
    streaming_state_validation: bool = True
    dense_temporal_supervision: bool = True

    batch_size: int = 1
    target_effective_batch: int = 8
    grad_accum: int = field(init=False)
    num_epochs: int = 50
    lr: float = 2e-4
    min_lr: float = 1e-5
    warmup_steps: int = 100
    weight_decay: float = 0.03
    grad_clip: float = 1.0
    amp_dtype: str = "bf16"
    # Full-policy torch.compile has a large startup cost for the recurrent
    # sequence path, so keep it opt-in for training runs.
    compile_model: bool = True

    sync_dataset: bool = True
    train_data_root: Optional[str] = None
    val_data_root: Optional[str] = None
    train_split: float = 0.9
    split_seed: int = 1337

    # Keep rare controls visible without letting a handful of false positives
    # dominate the objective.  The previous 25x cap made z/c calibration very
    # unstable and encouraged the low-precision predictions seen in validation.
    pos_weight_power: float = 0.5
    pos_weight_clamp: float = 8.0
    conflict_penalty_weight: float = 0.20
    # The current architecture has no previous-action readout branch, so
    # policy logits and vision logits are identical. Keep the auxiliary term
    # off to avoid silently scaling BCE.
    vision_aux_loss_weight: float = 0.0
    autoregressive_feedback_stream_prob: float = 0.0
    recorded_feedback_stream_prob: float = 0.0
    fit_thresholds_from_val: bool = True
    threshold_min: float = 0.10
    threshold_max: float = 0.90

    aug_brightness: float = 0.08
    aug_contrast: float = 0.10
    aug_noise_std: float = 0.006
    aug_gray_prob: float = 0.02
    aug_translate_frac: float = 0.0
    aug_scale_frac: float = 0.0
    aug_edges_crop_prob: float = 0.0
    aug_edges_crop_min_frac: float = 0.02
    aug_edges_crop_max_frac: float = 0.06
    aug_cutout_prob: float = 0.0
    aug_cutout_min_frac: float = 0.04
    aug_cutout_max_frac: float = 0.12
    aug_cutout_count: int = 1

    dali_num_threads: int = 6
    dali_prefetch_queue_depth: int = 4
    dali_reader_prefetch_queue_depth: int = 4
    dali_read_ahead: bool = False
    dali_dont_use_mmap: bool = False
    dali_train_random_shuffle: bool = True
    dali_val_random_shuffle: bool = False
    dali_shuffle_seed: int = 1337

    resume: bool = True
    resume_path: Optional[str] = None
    ckpt_dir: str = "./checkpoints_rt"
    save_every: int = 1
    print_every: int = 20
    max_train_batches: Optional[int] = None
    max_val_batches: Optional[int] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.batch_size = _positive_int("batch_size", self.batch_size)
        self.target_effective_batch = _positive_int("target_effective_batch", self.target_effective_batch)
        self.grad_accum = int(math.ceil(self.target_effective_batch / float(self.batch_size)))
        self.num_epochs = _positive_int("num_epochs", self.num_epochs)
        self.lr = _finite_float("lr", self.lr, 0.0)
        if self.lr <= 0.0:
            raise ValueError("lr must be positive.")
        self.min_lr = _finite_float("min_lr", self.min_lr, 0.0)
        if self.min_lr > self.lr:
            raise ValueError(f"min_lr must be <= lr, got {self.min_lr} > {self.lr}.")
        self.warmup_steps = _positive_int("warmup_steps", self.warmup_steps, 0)
        self.weight_decay = _finite_float("weight_decay", self.weight_decay, 0.0)
        self.grad_clip = _finite_float("grad_clip", self.grad_clip, 0.0)
        self.train_split = _finite_float("train_split", self.train_split, 0.05, 0.95)
        self.pos_weight_power = _finite_float("pos_weight_power", self.pos_weight_power, 0.0)
        self.pos_weight_clamp = _finite_float("pos_weight_clamp", self.pos_weight_clamp, 1.0)
        self.conflict_penalty_weight = _finite_float(
            "conflict_penalty_weight", self.conflict_penalty_weight, 0.0
        )
        self.vision_aux_loss_weight = _finite_float("vision_aux_loss_weight", self.vision_aux_loss_weight, 0.0)
        self.autoregressive_feedback_stream_prob = _finite_float(
            "autoregressive_feedback_stream_prob", self.autoregressive_feedback_stream_prob, 0.0, 1.0
        )
        self.recorded_feedback_stream_prob = _finite_float(
            "recorded_feedback_stream_prob", self.recorded_feedback_stream_prob, 0.0, 1.0
        )
        if not self.last_action_conditioning:
            self.vision_aux_loss_weight = 0.0
            self.autoregressive_feedback_stream_prob = 0.0
            self.recorded_feedback_stream_prob = 0.0
        else:
            feedback_probability = (
                self.autoregressive_feedback_stream_prob + self.recorded_feedback_stream_prob
            )
            if not math.isclose(feedback_probability, 1.0, rel_tol=0.0, abs_tol=1e-8):
                raise ValueError(
                    "autoregressive_feedback_stream_prob and recorded_feedback_stream_prob must sum to 1.0, "
                    f"got {feedback_probability:.6f}."
                )
        self.threshold_min = _finite_float("threshold_min", self.threshold_min, 0.0, 1.0)
        self.threshold_max = _finite_float("threshold_max", self.threshold_max, 0.0, 1.0)
        if self.threshold_max < self.threshold_min:
            raise ValueError("threshold_max must be >= threshold_min.")
        self.amp_dtype = str(self.amp_dtype).lower().strip()
        if self.amp_dtype not in {"bf16", "fp32", "float32"}:
            raise ValueError("amp_dtype must be bf16 or fp32.")
        self.compile_model = bool(self.compile_model)
        self.sync_dataset = bool(self.sync_dataset)
        self.fit_thresholds_from_val = bool(self.fit_thresholds_from_val)
        self.streaming_state_training = bool(self.streaming_state_training)
        self.streaming_state_validation = bool(self.streaming_state_validation)
        if self.last_action_conditioning and (
            not self.streaming_state_training or not self.streaming_state_validation
        ):
            raise ValueError(
                "Previous-action conditioning requires streaming state for both training and validation."
            )
        if not bool(self.dense_temporal_supervision):
            raise ValueError("This trainer requires dense_temporal_supervision=True.")
        self.dense_temporal_supervision = True

        for name in (
            "aug_brightness",
            "aug_contrast",
            "aug_gray_prob",
            "aug_translate_frac",
            "aug_scale_frac",
            "aug_edges_crop_prob",
            "aug_edges_crop_min_frac",
            "aug_edges_crop_max_frac",
            "aug_cutout_prob",
            "aug_cutout_min_frac",
            "aug_cutout_max_frac",
        ):
            setattr(self, name, _finite_float(name, getattr(self, name), 0.0, 1.0))
        self.aug_noise_std = _finite_float("aug_noise_std", self.aug_noise_std, 0.0, 0.1)
        if self.aug_edges_crop_max_frac < self.aug_edges_crop_min_frac:
            raise ValueError("aug_edges_crop_max_frac must be >= aug_edges_crop_min_frac.")
        if self.aug_cutout_max_frac < self.aug_cutout_min_frac:
            raise ValueError("aug_cutout_max_frac must be >= aug_cutout_min_frac.")
        self.aug_cutout_count = _positive_int("aug_cutout_count", self.aug_cutout_count)

        self.dali_num_threads = _positive_int("dali_num_threads", self.dali_num_threads)
        self.dali_prefetch_queue_depth = _positive_int(
            "dali_prefetch_queue_depth", self.dali_prefetch_queue_depth
        )
        self.dali_reader_prefetch_queue_depth = _positive_int(
            "dali_reader_prefetch_queue_depth", self.dali_reader_prefetch_queue_depth
        )
        self.dali_train_random_shuffle = bool(self.dali_train_random_shuffle)
        self.dali_val_random_shuffle = bool(self.dali_val_random_shuffle)
        self.dali_read_ahead = bool(self.dali_read_ahead)
        self.dali_dont_use_mmap = bool(self.dali_dont_use_mmap)
        self.dali_shuffle_seed = int(self.dali_shuffle_seed)
        self.split_seed = int(self.split_seed)
        if self.streaming_state_training:
            if self.batch_size != 1:
                raise ValueError("streaming_state_training requires batch_size=1.")
            if self.train_seq_stride != self.seq_len:
                raise ValueError(
                    "streaming_state_training requires train_seq_stride to equal seq_len "
                    f"({self.seq_len}), got {self.train_seq_stride}."
                )
            self.dali_train_random_shuffle = False
        if self.streaming_state_validation:
            if self.batch_size != 1:
                raise ValueError("streaming_state_validation requires batch_size=1.")
            if self.val_seq_stride != self.seq_len:
                raise ValueError(
                    "streaming_state_validation requires val_seq_stride to equal seq_len "
                    f"({self.seq_len}), got {self.val_seq_stride}."
                )
            self.dali_val_random_shuffle = False

        if (self.train_data_root is None) != (self.val_data_root is None):
            raise ValueError("train_data_root and val_data_root must be supplied together.")
        if self.train_data_root is not None:
            self.train_data_root = str(self.train_data_root)
            self.val_data_root = str(self.val_data_root)
        self.resume = bool(self.resume)
        self.save_every = _positive_int("save_every", self.save_every)
        self.print_every = _positive_int("print_every", self.print_every)
        if self.max_train_batches is not None:
            self.max_train_batches = _positive_int("max_train_batches", self.max_train_batches)
            if self.streaming_state_training:
                raise ValueError("max_train_batches is incompatible with complete-video streaming training.")
        if self.max_val_batches is not None:
            self.max_val_batches = _positive_int("max_val_batches", self.max_val_batches)
            if self.streaming_state_validation:
                raise ValueError("max_val_batches is incompatible with complete-video streaming validation.")
        self.ckpt_dir = os.path.abspath(self.ckpt_dir)


@dataclass(frozen=True)
class WindowMeta:
    video_path: str
    start_frame: int
    end_frame: int


@dataclass
class WindowTargets:
    labels: torch.Tensor
    previous_actions: torch.Tensor
    meta: List[WindowMeta]


@dataclass
class BinaryStats:
    num_actions: int
    device: torch.device

    def __post_init__(self) -> None:
        self.tp = torch.zeros(self.num_actions, dtype=torch.float64, device=self.device)
        self.fp = torch.zeros(self.num_actions, dtype=torch.float64, device=self.device)
        self.fn = torch.zeros(self.num_actions, dtype=torch.float64, device=self.device)

    @torch.no_grad()
    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        pred = prediction.bool().to(dtype=torch.float64)
        true = target.bool().to(dtype=torch.float64)
        self.tp += (pred * true).sum(dim=0)
        self.fp += (pred * (1.0 - true)).sum(dim=0)
        self.fn += ((1.0 - pred) * true).sum(dim=0)

    def rows(self, names: Sequence[str]) -> List[Dict[str, float | str | int]]:
        precision = self.tp / (self.tp + self.fp).clamp(min=1.0)
        recall = self.tp / (self.tp + self.fn).clamp(min=1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp(min=1e-8)
        support = self.tp + self.fn
        return [
            {
                "name": str(name),
                "precision": float(precision[index].item()),
                "recall": float(recall[index].item()),
                "f1": float(f1[index].item()),
                "support": int(support[index].item()),
            }
            for index, name in enumerate(names)
        ]

    def macro_f1(self) -> float:
        precision = self.tp / (self.tp + self.fp).clamp(min=1.0)
        recall = self.tp / (self.tp + self.fn).clamp(min=1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp(min=1e-8)
        observed = (self.tp + self.fp + self.fn) > 0
        return float(f1[observed].mean().item()) if bool(observed.any()) else 0.0


@dataclass
class ThresholdFitter:
    """Collect validation predictions and choose per-action F1 thresholds."""

    probabilities: List[torch.Tensor] = field(default_factory=list)
    targets: List[torch.Tensor] = field(default_factory=list)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        self.probabilities.append(torch.sigmoid(logits.float()).detach().cpu())
        self.targets.append(target.bool().detach().cpu())

    def fit(self, low: float, high: float, num_actions: int) -> Tuple[float, ...]:
        if not self.probabilities:
            return tuple(0.5 for _ in range(num_actions))
        probabilities = torch.cat(self.probabilities, dim=0)
        targets = torch.cat(self.targets, dim=0)
        candidates = torch.linspace(float(low), float(high), 161)
        fitted: List[float] = []
        for action in range(num_actions):
            truth = targets[:, action]
            if not bool(truth.any()):
                fitted.append(0.5)
                continue
            predictions = probabilities[:, action].unsqueeze(1) >= candidates.unsqueeze(0)
            true = truth.unsqueeze(1)
            tp = (predictions & true).sum(dim=0).float()
            fp = (predictions & ~true).sum(dim=0).float()
            fn = (~predictions & true).sum(dim=0).float()
            f1 = 2.0 * tp / (2.0 * tp + fp + fn).clamp(min=1.0)
            fitted.append(float(candidates[int(f1.argmax().item())].item()))
        return tuple(fitted)

    def statistics(self, thresholds: Sequence[float], num_actions: int) -> BinaryStats:
        """Evaluate the collected validation set using its fitted thresholds."""

        if len(thresholds) != num_actions:
            raise ValueError(f"Expected {num_actions} thresholds, got {len(thresholds)}.")
        stats = BinaryStats(num_actions, torch.device("cpu"))
        if not self.probabilities:
            return stats
        probabilities = torch.cat(self.probabilities, dim=0)
        targets = torch.cat(self.targets, dim=0)
        threshold_tensor = torch.tensor(list(thresholds), dtype=probabilities.dtype).view(1, -1)
        stats.update(probabilities >= threshold_tensor, targets)
        return stats


def find_runs(data_root: str, video_ext: str, csv_ext: str) -> List[Tuple[str, str]]:
    video_paths = sorted(glob.glob(os.path.join(data_root, f"*{video_ext}")))
    pairs: List[Tuple[str, str]] = []
    for video_path in video_paths:
        csv_path = os.path.splitext(video_path)[0] + csv_ext
        if os.path.isfile(csv_path):
            pairs.append((video_path, csv_path))
    return pairs


def split_runs(
    pairs: Sequence[Tuple[str, str]], train_split: float, seed: int
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    ordered = list(pairs)
    random.Random(int(seed)).shuffle(ordered)
    if len(ordered) <= 1:
        return ordered, []
    split = max(1, min(len(ordered) - 1, round(len(ordered) * float(train_split))))
    return ordered[:split], ordered[split:]


def sync_dataset_roots_for_training(cfg: TrainConfig) -> None:
    """Mirror the configured dataset to the local WSL cache before DALI reads it."""

    if not cfg.sync_dataset:
        return

    from dataset_wsl_sync import default_target_root, sync_dataset_for_training

    target_root = default_target_root()

    if cfg.train_data_root is not None and cfg.val_data_root is not None:
        cfg.train_data_root = sync_dataset_for_training(
            data_root=str(cfg.train_data_root),
            target_root=str(target_root / "train"),
            video_ext=cfg.video_ext,
            csv_ext=cfg.csv_ext,
        )
        cfg.val_data_root = sync_dataset_for_training(
            data_root=str(cfg.val_data_root),
            target_root=str(target_root / "val"),
            video_ext=cfg.video_ext,
            csv_ext=cfg.csv_ext,
        )
        cfg.data_root = str(target_root)
        return

    data_root = str(cfg.data_root)
    detected_train = os.path.join(data_root, "train")
    detected_val = os.path.join(data_root, "val")
    if os.path.isdir(detected_train) and os.path.isdir(detected_val):
        sync_dataset_for_training(
            data_root=detected_train,
            target_root=str(target_root / "train"),
            video_ext=cfg.video_ext,
            csv_ext=cfg.csv_ext,
        )
        sync_dataset_for_training(
            data_root=detected_val,
            target_root=str(target_root / "val"),
            video_ext=cfg.video_ext,
            csv_ext=cfg.csv_ext,
        )
        cfg.data_root = str(target_root)
        return

    cfg.data_root = sync_dataset_for_training(
        data_root=data_root,
        target_root=None,
        video_ext=cfg.video_ext,
        csv_ext=cfg.csv_ext,
    )


def resolve_run_pairs(cfg: TrainConfig) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    if cfg.train_data_root is not None and cfg.val_data_root is not None:
        train_pairs = find_runs(cfg.train_data_root, cfg.video_ext, cfg.csv_ext)
        val_pairs = find_runs(cfg.val_data_root, cfg.video_ext, cfg.csv_ext)
    else:
        detected_train = os.path.join(str(cfg.data_root), "train")
        detected_val = os.path.join(str(cfg.data_root), "val")
        if os.path.isdir(detected_train) and os.path.isdir(detected_val):
            train_pairs = find_runs(detected_train, cfg.video_ext, cfg.csv_ext)
            val_pairs = find_runs(detected_val, cfg.video_ext, cfg.csv_ext)
        else:
            pairs = find_runs(str(cfg.data_root), cfg.video_ext, cfg.csv_ext)
            train_pairs, val_pairs = split_runs(pairs, cfg.train_split, cfg.split_seed)
    if not train_pairs:
        raise RuntimeError(f"No video/CSV pairs found for training under {cfg.data_root!r}.")
    if not val_pairs:
        raise RuntimeError(
            "No validation runs were found. Provide train/val folders or at least two complete runs."
        )
    return train_pairs, val_pairs


def _parse_float(value: object) -> float:
    if isinstance(value, str):
        value = value.strip()
    return float(value)


def load_run_labels(csv_path: str, cfg: TrainConfig) -> torch.Tensor:
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {csv_path}")
        required = ["timestamp", *cfg.key_names]
        missing = [name for name in required if name not in reader.fieldnames]
        if missing:
            raise RuntimeError(f"CSV schema mismatch for {csv_path}: missing columns={missing}")
        rows.extend(reader)
    if not rows:
        raise RuntimeError(f"CSV has no rows: {csv_path}")
    labels = torch.zeros((len(rows), cfg.num_bin), dtype=torch.float32)
    for index, row in enumerate(rows):
        for action, name in enumerate(cfg.key_names):
            try:
                labels[index, action] = 1.0 if _parse_float(row[name]) > 0.5 else 0.0
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Invalid value in {csv_path} row {index + 2}, column {name!r}.") from exc
    return labels


def build_window_targets(
    pairs: Sequence[Tuple[str, str]], cfg: TrainConfig, *, stride: int
) -> WindowTargets:
    """Create one causal action target for every frame in each input window."""

    stride = _positive_int("stride", stride)
    label_windows: List[torch.Tensor] = []
    previous_action_windows: List[torch.Tensor] = []
    meta: List[WindowMeta] = []
    action_offset = int(cfg.action_offset)
    for video_path, csv_path in pairs:
        labels = load_run_labels(csv_path, cfg)
        # For input frame start + t, predict the future action at
        # start + t + action_offset. This preserves causality while
        # supervising every frame instead of only the final frame.
        max_start = int(labels.size(0)) - int(cfg.seq_len) - action_offset
        for start in range(0, max_start + 1, stride):
            target_start = start + action_offset
            target_end = target_start + int(cfg.seq_len)
            label_windows.append(labels[target_start:target_end])
            # Retained for dataloader compatibility. The current model does
            # not consume previous-action labels.
            previous_start = start + action_offset - 1
            previous = labels[previous_start : previous_start + int(cfg.seq_len)]
            previous_action_windows.append(previous)
            meta.append(WindowMeta(video_path, start, start + int(cfg.seq_len)))
    if not label_windows:
        raise RuntimeError(
            "No valid 80-frame windows were created. Check CSV lengths, seq_len, and prediction_horizon."
        )
    return WindowTargets(
        labels=torch.stack(label_windows, dim=0),
        previous_actions=torch.stack(previous_action_windows, dim=0),
        meta=meta,
    )


def video_streams(meta: Sequence[WindowMeta]) -> List[List[int]]:
    """Return sorted, contiguous window-index streams, one stream per video."""

    by_video: Dict[str, List[int]] = {}
    for index, item in enumerate(meta):
        by_video.setdefault(item.video_path, []).append(index)

    streams: List[List[int]] = []
    for video_path in sorted(by_video):
        indices = sorted(by_video[video_path], key=lambda index: meta[index].start_frame)
        if not indices:
            continue
        first = meta[indices[0]]
        if first.start_frame != 0:
            raise RuntimeError(
                f"Streaming video {video_path!r} must begin at frame 0, got {first.start_frame}."
            )
        for previous_index, current_index in zip(indices, indices[1:]):
            previous = meta[previous_index]
            current = meta[current_index]
            if current.start_frame != previous.end_frame:
                raise RuntimeError(
                    "Streaming windows must be contiguous: "
                    f"{video_path!r} has [{previous.start_frame}, {previous.end_frame}) followed by "
                    f"[{current.start_frame}, {current.end_frame})."
                )
        streams.append(indices)
    if not streams:
        raise RuntimeError("No video streams were built from the training windows.")
    return streams


def streaming_window_order(
    meta: Sequence[WindowMeta],
    *,
    seed: int,
    shuffle_streams: bool,
) -> List[int]:
    """Shuffle complete video streams while preserving order inside every run."""

    streams = video_streams(meta)
    if shuffle_streams:
        random.Random(int(seed)).shuffle(streams)
    return [index for stream in streams for index in stream]


def write_window_file_list(
    meta: Sequence[WindowMeta],
    path: str,
    *,
    indices: Optional[Sequence[int]] = None,
) -> None:
    """Write DALI rows in a requested order while preserving original sample IDs."""

    ordered_indices = list(range(len(meta))) if indices is None else [int(index) for index in indices]
    with open(path, "w", encoding="utf-8") as handle:
        for sample_id in ordered_indices:
            if sample_id < 0 or sample_id >= len(meta):
                raise IndexError(f"Window index {sample_id} is out of range for {len(meta)} entries.")
            item = meta[sample_id]
            handle.write(f"{item.video_path} {sample_id} {item.start_frame} {item.end_frame}\n")


@pipeline_def
def video_window_pipeline(
    file_list: str,
    seq_len: int,
    resize_size: int,
    random_shuffle: bool,
    reader_prefetch_queue_depth: int,
    read_ahead: bool,
    dont_use_mmap: bool,
):
    decode_bytes = int(seq_len) * int(resize_size) * int(resize_size) * 3
    normalized_bytes = decode_bytes * 2
    frames, sample_ids = fn.readers.video_resize(
        device="gpu",
        name="Reader",
        file_list=file_list,
        file_list_frame_num=True,
        file_list_include_preceding_frame=False,
        sequence_length=int(seq_len),
        step=int(seq_len),
        stride=1,
        random_shuffle=bool(random_shuffle),
        prefetch_queue_depth=int(reader_prefetch_queue_depth),
        read_ahead=bool(read_ahead),
        dont_use_mmap=bool(dont_use_mmap),
        image_type=types.RGB,
        resize_x=int(resize_size),
        resize_y=int(resize_size),
        interp_type=types.INTERP_LINEAR,
        bytes_per_sample_hint=decode_bytes,
        tensor_init_bytes=decode_bytes,
        temp_buffer_hint=decode_bytes,
    )
    frames = fn.crop_mirror_normalize(
        frames,
        # Keep DALI's prefetched resized sequences in FP16. The trainer casts
        # each current batch to BF16 or FP32 immediately before augmentation.
        dtype=types.FLOAT16,
        output_layout="FCHW",
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        bytes_per_sample_hint=normalized_bytes,
    )
    return frames, sample_ids


def make_dali_iterator(
    file_list: str,
    cfg: TrainConfig,
    *,
    batch_size: int,
    random_shuffle: bool,
    last_batch_policy,
):
    pipeline = video_window_pipeline(
        batch_size=int(batch_size),
        num_threads=int(cfg.dali_num_threads),
        device_id=0,
        seed=int(cfg.dali_shuffle_seed),
        file_list=file_list,
        seq_len=int(cfg.seq_len),
        resize_size=int(cfg.model_size),
        random_shuffle=bool(random_shuffle),
        reader_prefetch_queue_depth=int(cfg.dali_reader_prefetch_queue_depth),
        read_ahead=bool(cfg.dali_read_ahead),
        dont_use_mmap=bool(cfg.dali_dont_use_mmap),
        prefetch_queue_depth=int(cfg.dali_prefetch_queue_depth),
        exec_async=True,
        exec_pipelined=True,
    )
    pipeline.build()
    return DALIGenericIterator(
        [pipeline],
        output_map=["frames", "sample_ids"],
        reader_name="Reader",
        auto_reset=False,
        last_batch_policy=last_batch_policy,
        prepare_first_batch=True,
    )


def _ensure_fchw(frames: torch.Tensor) -> torch.Tensor:
    if frames.dim() != 5:
        raise RuntimeError(f"Unexpected DALI frame shape: {tuple(frames.shape)}.")
    if frames.size(2) == 3:
        return frames
    if frames.size(-1) == 3:
        return frames.permute(0, 1, 4, 2, 3).contiguous()
    raise RuntimeError(f"DALI did not produce RGB frames: {tuple(frames.shape)}.")


def load_batch(
    iterator,
    targets: WindowTargets,
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    batch = next(iterator)[0]
    if "frames" not in batch or "sample_ids" not in batch:
        raise RuntimeError(f"Unexpected DALI outputs: {sorted(batch)}.")
    frames = _ensure_fchw(batch["frames"])
    if frames.device != device:
        frames = frames.to(device, non_blocking=True)
    if frames.dtype != amp_dtype:
        frames = frames.to(dtype=amp_dtype)

    sample_ids = batch["sample_ids"].reshape(-1).long().cpu()
    if sample_ids.numel() != frames.size(0):
        raise RuntimeError(
            f"DALI batch has {frames.size(0)} frame windows but {sample_ids.numel()} sample IDs."
        )
    if bool((sample_ids < 0).any()) or bool((sample_ids >= len(targets.meta)).any()):
        raise RuntimeError(f"DALI returned an out-of-range sample ID: {sample_ids.tolist()}.")
    labels = targets.labels[sample_ids].to(device, non_blocking=True)
    previous_actions = targets.previous_actions[sample_ids].to(device, non_blocking=True)
    return frames, labels, previous_actions, [int(sample_id) for sample_id in sample_ids.tolist()]


def compute_pos_weight(labels: torch.Tensor, power: float, clamp: float) -> torch.Tensor:
    labels = labels.reshape(-1, labels.size(-1)).float()
    positive = labels.sum(dim=0)
    negative = float(labels.size(0)) - positive
    return (negative / positive.clamp(min=1.0)).pow(float(power)).clamp(1.0, float(clamp))


def conflicting_action_penalty(logits: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    """Penalize probability mass assigned to mutually exclusive driving chords."""

    if cfg.conflict_penalty_weight <= 0.0:
        return logits.new_zeros(())
    action_index = {name: index for index, name in enumerate(cfg.key_names)}
    pair_penalties: List[torch.Tensor] = []
    probabilities = torch.sigmoid(logits.float())
    for first, second in CONFLICTING_ACTION_PAIRS:
        first_index = action_index.get(first)
        second_index = action_index.get(second)
        if first_index is not None and second_index is not None:
            pair_penalties.append(probabilities[..., first_index] * probabilities[..., second_index])
    if not pair_penalties:
        return logits.new_zeros(())
    return torch.stack(pair_penalties, dim=1).mean()


def warmup_cosine_lr(
    step: int, *, total_steps: int, base_lr: float, min_lr: float, warmup_steps: int
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(base_lr) * float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr) + (float(base_lr) - float(min_lr)) * cosine


def threshold_tensor(cfg: TrainConfig, *, device: torch.device) -> torch.Tensor:
    return torch.tensor(list(cfg.button_state_thresholds), device=device, dtype=torch.float32)


def print_metric_rows(prefix: str, rows: Iterable[Dict[str, float | str | int]]) -> None:
    print(prefix)
    for row in rows:
        print(
            f"  {str(row['name']):>2}: "
            f"f1={float(row['f1']):.3f} "
            f"precision={float(row['precision']):.3f} "
            f"recall={float(row['recall']):.3f} "
            f"support={int(row['support'])}"
        )


@dataclass
class StreamingStateEntry:
    next_frame: int
    temporal_state: TemporalState
    feedback_mode: str
    feedback_action: Optional[torch.Tensor] = None


StreamingStateCache = Dict[str, StreamingStateEntry]


def stream_feedback_mode(item: WindowMeta, *, epoch_index: int, cfg: TrainConfig) -> str:
    """Pick one feedback distribution for a full stream, reproducibly per epoch."""

    source = f"{cfg.dali_shuffle_seed}:{epoch_index}:{item.video_path}:action-feedback".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(source).digest()[:8], "little") / float(2**64)
    return "autoregressive" if value < float(cfg.autoregressive_feedback_stream_prob) else "recorded"


def stream_initial_state(
    sample_ids: Sequence[int],
    targets: WindowTargets,
    cache: StreamingStateCache,
) -> Tuple[Optional[StreamingStateEntry], WindowMeta, bool]:
    """Return the state for the next contiguous chunk and whether it was carried."""

    if len(sample_ids) != 1:
        raise RuntimeError(
            "Stateful ConvGRU training requires one DALI sample per batch; "
            f"received {len(sample_ids)}."
        )
    item = targets.meta[int(sample_ids[0])]
    cached = cache.get(item.video_path)
    if cached is None:
        if item.start_frame != 0:
            raise RuntimeError(
                f"Streaming state for {item.video_path!r} is missing before frame {item.start_frame}; "
                "DALI window order is not a complete video stream."
            )
        return None, item, False

    if item.start_frame != cached.next_frame:
        raise RuntimeError(
            f"Non-contiguous DALI stream for {item.video_path!r}: "
            f"expected frame {cached.next_frame}, got {item.start_frame}."
        )
    return cached, item, True


def stream_augmentation_generator(
    item: WindowMeta,
    *,
    epoch_index: int,
    cfg: TrainConfig,
    device: torch.device,
) -> torch.Generator:
    """Return the same transform seed for every chunk in one video/epoch."""

    source = f"{cfg.dali_shuffle_seed}:{epoch_index}:{item.video_path}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(source).digest()[:8], "little") % (2**63 - 1)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def run_epoch(
    *,
    model: torch.nn.Module,
    iterator,
    num_batches: int,
    targets: WindowTargets,
    cfg: TrainConfig,
    device: torch.device,
    amp_dtype: torch.dtype,
    pos_weight: torch.Tensor,
    optimizer: Optional[torch.optim.Optimizer],
    total_steps: int,
    global_step: int,
    description: str,
    streaming_state: bool,
    epoch_index: int,
) -> Tuple[Dict[str, object], int]:
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)

    loss_sum = 0.0
    bce_loss_sum = 0.0
    vision_bce_loss_sum = 0.0
    conflict_loss_sum = 0.0
    stats = BinaryStats(cfg.num_bin, device)
    vision_stats = BinaryStats(cfg.num_bin, device)
    fitter = ThresholdFitter() if (not training and cfg.fit_thresholds_from_val) else None
    thresholds = threshold_tensor(cfg, device=device)
    progress = tqdm(range(num_batches), desc=description, dynamic_ncols=True)
    iterator_it = iter(iterator)
    state_cache: StreamingStateCache = {}
    stream_resets = 0
    stream_carried = 0
    stream_total = 0

    for batch_index in progress:
        frames, labels, previous_actions, sample_ids = load_batch(
            iterator_it,
            targets,
            device=device,
            amp_dtype=amp_dtype,
        )
        initial_state: Optional[TemporalState] = None
        stream_item: Optional[WindowMeta] = None
        stream_entry: Optional[StreamingStateEntry] = None
        feedback_mode = "disabled"
        carried = False
        if streaming_state:
            stream_entry, stream_item, carried = stream_initial_state(sample_ids, targets, state_cache)
            if stream_entry is not None:
                initial_state = stream_entry.temporal_state
                feedback_mode = stream_entry.feedback_mode
            elif stream_item is not None:
                feedback_mode = "disabled"
            stream_total += 1
            stream_carried += int(carried)
            stream_resets += int(not carried)
        if training:
            generator = (
                stream_augmentation_generator(
                    stream_item,
                    epoch_index=epoch_index,
                    cfg=cfg,
                    device=frames.device,
                )
                if streaming_state and stream_item is not None
                else None
            )
            frames = augment_frames(frames, cfg, same_over_time=True, generator=generator)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=amp_dtype == torch.bfloat16,
            ):
                if streaming_state:
                    output, next_state = model(
                        frames,
                        state=initial_state,
                        return_aux=True,
                        return_sequence_logits=True,
                        prev_action=None,
                        feedback_thresholds=None,
                        autoregressive_feedback=False,
                    )
                    if stream_item is None:
                        raise RuntimeError("Streaming state metadata was not resolved.")
                    state_cache[stream_item.video_path] = StreamingStateEntry(
                        next_frame=stream_item.end_frame,
                        temporal_state=next_state,
                        feedback_mode=feedback_mode,
                        feedback_action=None,
                    )
                else:
                    output = model(
                        frames,
                        return_sequence_logits=True,
                        prev_action=None,
                    )
                logits = output.sequence_button_logits
                vision_logits = output.sequence_vision_button_logits
                if logits is None:
                    raise RuntimeError("Dense temporal supervision requires per-timestep policy logits.")
                if vision_logits is None:
                    raise RuntimeError("Dense temporal supervision requires per-timestep vision logits.")
                if tuple(logits.shape) != tuple(labels.shape):
                    raise RuntimeError(
                        f"Model logits shape {tuple(logits.shape)} does not match labels {tuple(labels.shape)}."
                    )
                bce_loss = F.binary_cross_entropy_with_logits(
                    logits.float(), labels.float(), pos_weight=pos_weight.float()
                )
                vision_bce_loss = F.binary_cross_entropy_with_logits(
                    vision_logits.float(), labels.float(), pos_weight=pos_weight.float()
                )
                conflict_loss = conflicting_action_penalty(logits, cfg)
                loss = (
                    bce_loss
                    + float(cfg.vision_aux_loss_weight) * vision_bce_loss
                    + float(cfg.conflict_penalty_weight) * conflict_loss
                )
                scaled_loss = loss / int(cfg.grad_accum)
            if training:
                scaled_loss.backward()
                is_last = batch_index + 1 == num_batches
                if ((batch_index + 1) % cfg.grad_accum == 0) or is_last:
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

        with torch.no_grad():
            flat_logits = logits.reshape(-1, cfg.num_bin)
            flat_vision_logits = vision_logits.reshape(-1, cfg.num_bin)
            flat_labels = labels.reshape(-1, cfg.num_bin)
            prediction = torch.sigmoid(flat_logits.float()) >= thresholds.view(1, -1)
            vision_prediction = torch.sigmoid(flat_vision_logits.float()) >= thresholds.view(1, -1)
            stats.update(prediction, flat_labels > 0.5)
            vision_stats.update(vision_prediction, flat_labels > 0.5)
            if fitter is not None:
                fitter.update(flat_logits, flat_labels)
        loss_sum += float(loss.detach().item())
        bce_loss_sum += float(bce_loss.detach().item())
        vision_bce_loss_sum += float(vision_bce_loss.detach().item())
        conflict_loss_sum += float(conflict_loss.detach().item())
        if (batch_index + 1) % cfg.print_every == 0 or batch_index + 1 == num_batches:
            progress.set_postfix(
                loss=f"{loss_sum / float(batch_index + 1):.4f}",
                bce=f"{bce_loss_sum / float(batch_index + 1):.4f}",
                vision_bce=f"{vision_bce_loss_sum / float(batch_index + 1):.4f}",
                conflict=f"{conflict_loss_sum / float(batch_index + 1):.4f}",
                f1=f"{stats.macro_f1():.3f}",
                vision_f1=f"{vision_stats.macro_f1():.3f}",
                carry=f"{stream_carried / max(1, stream_total):.3f}" if streaming_state else "n/a",
            )
    iterator.reset()

    metrics: Dict[str, object] = {
        "loss": loss_sum / max(1, num_batches),
        "bce_loss": bce_loss_sum / max(1, num_batches),
        "vision_bce_loss": vision_bce_loss_sum / max(1, num_batches),
        "conflict_penalty": conflict_loss_sum / max(1, num_batches),
        "macro_f1": stats.macro_f1(),
        "rows": stats.rows(cfg.key_names),
        "vision_macro_f1": vision_stats.macro_f1(),
        "vision_rows": vision_stats.rows(cfg.key_names),
    }
    if streaming_state:
        metrics["stream_chunks"] = stream_total
        metrics["stream_resets"] = stream_resets
        metrics["stream_carried"] = stream_carried
        metrics["stream_carry_rate"] = stream_carried / max(1, stream_total)
    if fitter is not None:
        fitted_thresholds = fitter.fit(cfg.threshold_min, cfg.threshold_max, cfg.num_bin)
        calibrated_stats = fitter.statistics(fitted_thresholds, cfg.num_bin)
        # The previous code reported F1 using the prior epoch's thresholds, then
        # printed thresholds fitted from the current epoch.  Report the latter
        # so per-class validation metrics describe the checkpoint being saved.
        metrics["fitted_thresholds"] = fitted_thresholds
        metrics["previous_threshold_macro_f1"] = metrics["macro_f1"]
        metrics["previous_threshold_rows"] = metrics["rows"]
        metrics["macro_f1"] = calibrated_stats.macro_f1()
        metrics["rows"] = calibrated_stats.rows(cfg.key_names)
    return metrics, global_step


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    *,
    epoch: int,
    global_step: int,
    best_validation_bce: float,
) -> Dict[str, object]:
    return {
        "architecture_version": ARCHITECTURE_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": asdict(cfg),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_validation_bce": float(best_validation_bce),
    }


def maybe_resume(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, cfg: TrainConfig, device: torch.device
) -> Tuple[int, int, float]:
    if not cfg.resume:
        return 0, 0, math.inf
    path = cfg.resume_path or os.path.join(cfg.ckpt_dir, "model_latest.pt")
    if not os.path.isfile(path):
        return 0, 0, math.inf
    state = torch.load(path, map_location=device)
    if not isinstance(state, dict) or state.get("architecture_version") != ARCHITECTURE_VERSION:
        if cfg.resume_path is not None:
            raise RuntimeError(f"Checkpoint {path!r} does not use {ARCHITECTURE_VERSION}.")
        print(f"Skipping incompatible checkpoint: {path}")
        return 0, 0, math.inf
    config = state.get("config")
    if not isinstance(config, dict) or list(config.get("key_names", [])) != list(cfg.key_names):
        raise RuntimeError(f"Checkpoint {path!r} has a different action schema.")
    if not bool(config.get("dense_temporal_supervision", False)):
        if cfg.resume_path is not None:
            raise RuntimeError(
                f"Checkpoint {path!r} was trained with final-frame-only supervision and cannot be resumed "
                "for dense temporal supervision. Start from scratch instead."
            )
        print(f"Skipping final-frame-only checkpoint: {path}")
        return 0, 0, math.inf
    model.load_state_dict(state["model_state"], strict=True)
    optimizer.load_state_dict(state["optimizer_state"])
    print(f"Resumed from {path} at epoch {state.get('epoch', 0)}.")
    # Older checkpoints selected by macro F1 have no comparable best BCE.  On
    # resume, treat the next validation pass as the first candidate instead of
    # comparing a loss to an old F1 value.
    return (
        int(state.get("epoch", 0)),
        int(state.get("global_step", 0)),
        float(state.get("best_validation_bce", math.inf)),
    )


def _batch_count(targets: WindowTargets, batch_size: int, *, partial: bool) -> int:
    if partial:
        return int(math.ceil(len(targets.meta) / float(batch_size)))
    return len(targets.meta) // int(batch_size)


def print_startup_stats(
    cfg: TrainConfig,
    *,
    train_pairs: Sequence[Tuple[str, str]],
    val_pairs: Sequence[Tuple[str, str]],
    train_targets: WindowTargets,
    val_targets: WindowTargets,
    train_batches: int,
    val_batches: int,
    parameter_count: int,
    pos_weight: torch.Tensor,
) -> None:
    """Print the fixed data/model contract before the first DALI batch is read."""

    action_names = list(cfg.key_names)
    train_positive_rate = train_targets.labels.reshape(-1, cfg.num_bin).float().mean(dim=0)
    val_positive_rate = val_targets.labels.reshape(-1, cfg.num_bin).float().mean(dim=0)
    total_videos = len(train_pairs) + len(val_pairs)
    print("Startup configuration:")
    print(
        "  Model:",
        f"architecture={ARCHITECTURE_VERSION}",
        f"parameters={parameter_count / 1_000_000:.4f}M",
        f"input=[B,{cfg.seq_len},3,{cfg.model_size},{cfg.model_size}]",
        f"temporal=2xConvGRU16({FUSED_CHANNELS}x{cfg.model_size // 16}x{cfg.model_size // 16})",
        f"readout=6x{READOUT_CHANNELS}",
    )
    print(
        "  Actions:",
        f"order={action_names}",
        f"action_offset=+{int(cfg.action_offset)}",
        f"thresholds={list(cfg.button_state_thresholds)}",
    )
    print(
        "  Data:",
        f"sync_dataset={cfg.sync_dataset}",
        f"data_root={cfg.data_root}",
        f"train_data_root={cfg.train_data_root}",
        f"val_data_root={cfg.val_data_root}",
        f"videos_total={total_videos}",
        f"train_videos={len(train_pairs)}",
        f"val_videos={len(val_pairs)}",
        f"train_windows={len(train_targets.meta)}",
        f"val_windows={len(val_targets.meta)}",
        f"train_stride={cfg.train_seq_stride}",
        f"val_stride={cfg.val_seq_stride}",
    )
    print(
        "  Batches:",
        f"batch_size={cfg.batch_size}",
        f"grad_accum={cfg.grad_accum}",
        f"effective_batch={cfg.batch_size * cfg.grad_accum}",
        f"train_batches={train_batches}",
        f"val_batches={val_batches}",
    )
    print(
        "  Optimizer:",
        "AdamW",
        f"lr={cfg.lr:.6g}",
        f"min_lr={cfg.min_lr:.6g}",
        f"warmup_steps={cfg.warmup_steps}",
        f"weight_decay={cfg.weight_decay:.6g}",
        f"grad_clip={cfg.grad_clip:.6g}",
        f"amp={cfg.amp_dtype}",
        f"compile={cfg.compile_model}",
    )
    print(
        "  DALI:",
        "gpu_decode_resize",
        f"output=fp16",
        f"reader_threads={cfg.dali_num_threads}",
        f"pipeline_prefetch={cfg.dali_prefetch_queue_depth}",
        f"reader_prefetch={cfg.dali_reader_prefetch_queue_depth}",
        f"train_shuffle={cfg.dali_train_random_shuffle}",
    )
    print(
        "  Streaming:",
        f"train={cfg.streaming_state_training}",
        f"validation={cfg.streaming_state_validation}",
        f"chunk_stride={cfg.seq_len}",
        "state=detached_at_chunk_boundary",
        "stream_order=shuffled_videos/ordered_chunks"
        if cfg.streaming_state_training
        else "stream_order=fresh_windows",
    )
    print(
        "  Supervision:",
        f"dense_causal={cfg.dense_temporal_supervision}",
        f"targets_per_window={cfg.seq_len}",
        f"action_offset=+{int(cfg.action_offset)}",
    )
    print(
        "  Loss:",
        "BCEWithLogits",
        f"pos_weight_power={cfg.pos_weight_power:.3f}",
        f"pos_weight_clamp={cfg.pos_weight_clamp:.3f}",
        f"conflict_weight={cfg.conflict_penalty_weight:.3f}",
        f"vision_aux_weight={cfg.vision_aux_loss_weight:.3f}",
    )
    print(
        "  Previous action:",
        f"conditioning={cfg.last_action_conditioning}",
        "train_feedback=disabled",
        "validation_feedback=disabled",
    )
    print(
        "  Train class stats:",
        " ".join(
            f"{name}:pos={float(rate):.4f},w={float(weight):.3f}"
            for name, rate, weight in zip(action_names, train_positive_rate, pos_weight.detach().cpu())
        ),
    )
    print(
        "  Val class stats:",
        " ".join(f"{name}:pos={float(rate):.4f}" for name, rate in zip(action_names, val_positive_rate)),
    )


def train(cfg: Optional[TrainConfig] = None) -> None:
    cfg = TrainConfig() if cfg is None else cfg
    if not torch.cuda.is_available():
        raise RuntimeError("DALI GPU video decode requires a CUDA-visible PyTorch device.")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    torch.manual_seed(cfg.split_seed)
    random.seed(cfg.split_seed)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    sync_dataset_roots_for_training(cfg)
    train_pairs, val_pairs = resolve_run_pairs(cfg)
    train_targets = build_window_targets(train_pairs, cfg, stride=cfg.train_seq_stride)
    val_targets = build_window_targets(val_pairs, cfg, stride=cfg.val_seq_stride)
    train_file_list = os.path.join(cfg.ckpt_dir, "train_windows.txt")
    val_file_list = os.path.join(cfg.ckpt_dir, "val_windows.txt")
    val_order = (
        streaming_window_order(
            val_targets.meta,
            seed=cfg.dali_shuffle_seed,
            shuffle_streams=False,
        )
        if cfg.streaming_state_validation
        else list(range(len(val_targets.meta)))
    )
    write_window_file_list(val_targets.meta, val_file_list, indices=val_order)

    train_batches = _batch_count(train_targets, cfg.batch_size, partial=False)
    val_batch_size = 1 if cfg.streaming_state_validation else min(cfg.batch_size, len(val_targets.meta))
    val_batches = _batch_count(val_targets, val_batch_size, partial=True)
    if cfg.max_train_batches is not None:
        train_batches = min(train_batches, cfg.max_train_batches)
    if cfg.max_val_batches is not None:
        val_batches = min(val_batches, cfg.max_val_batches)
    if train_batches <= 0 or val_batches <= 0:
        raise RuntimeError("Insufficient windows for the configured batch size.")

    val_iterator = make_dali_iterator(
        val_file_list,
        cfg,
        batch_size=val_batch_size,
        random_shuffle=cfg.dali_val_random_shuffle,
        last_batch_policy=LastBatchPolicy.PARTIAL,
    )

    base_model: torch.nn.Module = DrivingVideoPolicy(cfg).to(device)
    base_model = base_model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    start_epoch, global_step, best_validation_bce = maybe_resume(base_model, optimizer, cfg, device)
    model = torch.compile(base_model, mode="reduce-overhead") if cfg.compile_model else base_model
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float32
    pos_weight = compute_pos_weight(train_targets.labels, cfg.pos_weight_power, cfg.pos_weight_clamp).to(device)
    parameter_count = sum(parameter.numel() for parameter in base_model.parameters())
    print_startup_stats(
        cfg,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        train_targets=train_targets,
        val_targets=val_targets,
        train_batches=train_batches,
        val_batches=val_batches,
        parameter_count=parameter_count,
        pos_weight=pos_weight,
    )
    total_steps = max(1, int(math.ceil(train_batches / float(cfg.grad_accum))) * cfg.num_epochs)

    for epoch in range(start_epoch, cfg.num_epochs):
        train_order = (
            streaming_window_order(
                train_targets.meta,
                seed=cfg.dali_shuffle_seed + epoch,
                shuffle_streams=True,
            )
            if cfg.streaming_state_training
            else list(range(len(train_targets.meta)))
        )
        write_window_file_list(train_targets.meta, train_file_list, indices=train_order)
        train_iterator = make_dali_iterator(
            train_file_list,
            cfg,
            batch_size=cfg.batch_size,
            random_shuffle=cfg.dali_train_random_shuffle,
            last_batch_policy=LastBatchPolicy.DROP,
        )
        train_metrics, global_step = run_epoch(
            model=model,
            iterator=train_iterator,
            num_batches=train_batches,
            targets=train_targets,
            cfg=cfg,
            device=device,
            amp_dtype=amp_dtype,
            pos_weight=pos_weight,
            optimizer=optimizer,
            total_steps=total_steps,
            global_step=global_step,
            description=f"Epoch {epoch + 1}/{cfg.num_epochs} train",
            streaming_state=cfg.streaming_state_training,
            epoch_index=epoch,
        )
        del train_iterator
        with torch.inference_mode():
            val_metrics, _ = run_epoch(
                model=model,
                iterator=val_iterator,
                num_batches=val_batches,
                targets=val_targets,
                cfg=cfg,
                device=device,
                amp_dtype=amp_dtype,
                pos_weight=pos_weight,
                optimizer=None,
                total_steps=total_steps,
                global_step=global_step,
                description=f"Epoch {epoch + 1}/{cfg.num_epochs} val",
                streaming_state=cfg.streaming_state_validation,
                epoch_index=epoch,
            )

        fitted = val_metrics.get("fitted_thresholds")
        if isinstance(fitted, tuple) and len(fitted) == cfg.num_bin:
            cfg.button_state_thresholds = tuple(float(item) for item in fitted)
        score = float(val_metrics["macro_f1"])
        validation_bce = float(val_metrics["bce_loss"])
        is_best = validation_bce < best_validation_bce
        if is_best:
            best_validation_bce = validation_bce
        print(
            f"Epoch {epoch + 1}/{cfg.num_epochs}: "
            f"train_loss={float(train_metrics['loss']):.4f} "
            f"train_conflict={float(train_metrics['conflict_penalty']):.4f} "
            f"train_f1={float(train_metrics['macro_f1']):.4f} "
            f"train_vision_f1={float(train_metrics['vision_macro_f1']):.4f} "
            f"train_stream={int(train_metrics.get('stream_resets', 0))}reset/"
            f"{int(train_metrics.get('stream_carried', 0))}carried/"
            f"{float(train_metrics.get('stream_carry_rate', 0.0)):.3f} "
            f"val_loss={float(val_metrics['loss']):.4f} "
            f"val_conflict={float(val_metrics['conflict_penalty']):.4f} "
            f"val_f1={score:.4f} "
            f"val_vision_f1={float(val_metrics['vision_macro_f1']):.4f} "
            f"val_stream={int(val_metrics.get('stream_resets', 0))}reset/"
            f"{int(val_metrics.get('stream_carried', 0))}carried/"
            f"{float(val_metrics.get('stream_carry_rate', 0.0)):.3f}"
        )
        print(
            f"Validation best_bce={best_validation_bce:.4f} "
            f"current_bce={validation_bce:.4f} "
            f"calibrated_macro_f1={score:.4f} "
            f"new_best={is_best}"
        )
        print_metric_rows("Validation controls:", val_metrics["rows"])
        print_metric_rows("Validation vision-only controls:", val_metrics["vision_rows"])
        if fitted is not None:
            print("Validation thresholds:", dict(zip(cfg.key_names, cfg.button_state_thresholds)))

        payload = checkpoint_payload(
            base_model,
            optimizer,
            cfg,
            epoch=epoch + 1,
            global_step=global_step,
            best_validation_bce=best_validation_bce,
        )
        torch.save(payload, os.path.join(cfg.ckpt_dir, "model_latest.pt"))
        if is_best:
            torch.save(payload, os.path.join(cfg.ckpt_dir, "model_best.pt"))
        if (epoch + 1) % cfg.save_every == 0:
            torch.save(payload, os.path.join(cfg.ckpt_dir, f"model_epoch_{epoch + 1}.pt"))


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the six-action FPN/ConvGRU driving policy with DALI.")
    add = parser.add_argument
    add("--data-root", default=None)
    add("--train-data-root", default=None)
    add("--val-data-root", default=None)
    add("--ckpt-dir", default=None)
    add("--resume", dest="resume", action="store_true", default=None)
    add("--no-resume", dest="resume", action="store_false")
    add("--resume-path", default=None)
    add("--sync-dataset", dest="sync_dataset", action="store_true", default=None)
    add("--no-sync-dataset", dest="sync_dataset", action="store_false")
    add("--num-epochs", type=int, default=None)
    add("--batch-size", type=int, default=None)
    add("--target-effective-batch", type=int, default=None)
    add("--seq-len", type=int, default=None)
    add("--train-seq-stride", type=int, default=None)
    add("--val-seq-stride", type=int, default=None)
    add("--streaming-state-training", dest="streaming_state_training", action="store_true", default=None)
    add("--no-streaming-state-training", dest="streaming_state_training", action="store_false")
    add("--streaming-state-validation", dest="streaming_state_validation", action="store_true", default=None)
    add("--no-streaming-state-validation", dest="streaming_state_validation", action="store_false")
    add(
        "--action-offset",
        "--prediction-horizon",
        dest="action_offset",
        type=int,
        default=None,
        help="Future label shift: frame[i] predicts action[i + action_offset]. Default: 1.",
    )
    add("--model-size", type=int, default=None)
    add("--lr", type=float, default=None)
    add("--min-lr", type=float, default=None)
    add("--warmup-steps", type=int, default=None)
    add("--weight-decay", type=float, default=None)
    add("--grad-clip", type=float, default=None)
    add("--train-split", type=float, default=None)
    add("--pos-weight-power", type=float, default=None)
    add("--pos-weight-clamp", type=float, default=None)
    add("--conflict-penalty-weight", type=float, default=None)
    add("--vision-aux-loss-weight", type=float, default=None)
    add("--last-action-residual-cap", type=float, default=None)
    add("--autoregressive-feedback-stream-prob", type=float, default=None)
    add("--recorded-feedback-stream-prob", type=float, default=None)
    add("--threshold-min", type=float, default=None)
    add("--threshold-max", type=float, default=None)
    add("--fit-thresholds-from-val", dest="fit_thresholds_from_val", action="store_true", default=None)
    add("--no-fit-thresholds-from-val", dest="fit_thresholds_from_val", action="store_false")
    add("--amp-dtype", choices=["bf16", "fp32"], default=None)
    add("--compile", dest="compile_model", action="store_true", default=None)
    add("--no-compile", dest="compile_model", action="store_false")
    add("--dali-num-threads", type=int, default=None)
    add("--dali-prefetch-queue-depth", type=int, default=None)
    add("--dali-reader-prefetch-queue-depth", type=int, default=None)
    add("--dali-train-random-shuffle", dest="dali_train_random_shuffle", action="store_true", default=None)
    add("--no-dali-train-random-shuffle", dest="dali_train_random_shuffle", action="store_false")
    add("--dali-val-random-shuffle", dest="dali_val_random_shuffle", action="store_true", default=None)
    add("--no-dali-val-random-shuffle", dest="dali_val_random_shuffle", action="store_false")
    add("--max-train-batches", type=int, default=None)
    add("--max-val-batches", type=int, default=None)
    args = vars(parser.parse_args())
    return TrainConfig(**{name: value for name, value in args.items() if value is not None})


if __name__ == "__main__":
    train(parse_args())
