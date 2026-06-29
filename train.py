"""DALI trainer for the dense-supervision CNN grid-token transformer policy.

The DALI pipeline owns GPU video decode and resize. CSV labels are parsed once
on the host, then joined to decoded fixed-length windows through the DALI sample
label written into the file list. Each window is an independent causal
transformer context with observed previous actions used only for late residual
conditioning.
"""

from __future__ import annotations

import argparse
import csv
import glob
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
    DEFAULT_SEQUENCE_LENGTH,
    FUSED_CHANNELS,
    FUSED_SPATIAL_SIZE,
    NUM_VISUAL_TOKENS,
    READOUT_CHANNELS,
    TEMPORAL_HEADS,
    TEMPORAL_LAYERS,
    TOKENS_PER_STEP,
    DrivingVideoPolicy,
    ModelConfig,
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


def _parse_threshold_sequence(value: str) -> Tuple[float, ...]:
    parts = [item.strip() for item in str(value).split(",")]
    if not parts or any(not item for item in parts):
        raise argparse.ArgumentTypeError("threshold list must be comma-separated floats")
    try:
        return tuple(float(item) for item in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("threshold list must be comma-separated floats") from exc


@dataclass
class TrainConfig(ModelConfig):
    """Training settings. The inherited model config fixes the six action names."""

    train_seq_stride: int = DEFAULT_SEQUENCE_LENGTH
    val_seq_stride: int = DEFAULT_SEQUENCE_LENGTH
    action_offset: int = 4
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
    main_policy_loss_weight: float = 0.5
    conflict_penalty_weight: float = 0.20
    vision_aux_loss_weight: float = 1.0
    # Drop previous-action residual inputs during teacher forcing so the policy
    # cannot solve the task by copying action persistence alone.
    action_token_dropout_prob: float = 0.2
    # Optional closed-loop scheduled sampling over independent fixed-length windows.
    # Keep it off by default for V5 so early training optimizes the vision path;
    # closed-loop validation still runs as a secondary signal.
    autoregressive_feedback_prob: float = 0.0
    autoregressive_validation: bool = True
    autoregressive_feedback_threshold: float = 0.5
    autoregressive_feedback_thresholds: Optional[Sequence[float]] = None
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
    dali_prefetch_queue_depth: int = 1
    dali_reader_prefetch_queue_depth: int = 1
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
        self.main_policy_loss_weight = _finite_float(
            "main_policy_loss_weight", self.main_policy_loss_weight, 0.0
        )
        self.conflict_penalty_weight = _finite_float(
            "conflict_penalty_weight", self.conflict_penalty_weight, 0.0
        )
        self.vision_aux_loss_weight = _finite_float("vision_aux_loss_weight", self.vision_aux_loss_weight, 0.0)
        self.action_token_dropout_prob = _finite_float(
            "action_token_dropout_prob", self.action_token_dropout_prob, 0.0, 1.0
        )
        self.autoregressive_feedback_prob = _finite_float(
            "autoregressive_feedback_prob", self.autoregressive_feedback_prob, 0.0, 1.0
        )
        self.autoregressive_validation = bool(self.autoregressive_validation)
        self.autoregressive_feedback_threshold = _finite_float(
            "autoregressive_feedback_threshold", self.autoregressive_feedback_threshold, 0.0, 1.0
        )
        if self.autoregressive_feedback_thresholds is None:
            self.autoregressive_feedback_thresholds = tuple(
                0.35 if name in {"z", "c"} else float(self.autoregressive_feedback_threshold)
                for name in self.key_names
            )
        else:
            feedback_thresholds = tuple(float(item) for item in self.autoregressive_feedback_thresholds)
            if len(feedback_thresholds) != self.num_bin:
                raise ValueError(
                    "autoregressive_feedback_thresholds must contain "
                    f"{self.num_bin} values, got {len(feedback_thresholds)}."
                )
            for idx, threshold in enumerate(feedback_thresholds):
                _finite_float(f"autoregressive_feedback_thresholds[{idx}]", threshold, 0.0, 1.0)
            self.autoregressive_feedback_thresholds = feedback_thresholds
        if not self.last_action_conditioning:
            self.vision_aux_loss_weight = 0.0
            self.action_token_dropout_prob = 0.0
            self.autoregressive_feedback_prob = 0.0
            self.autoregressive_validation = False
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
        if self.max_val_batches is not None:
            self.max_val_batches = _positive_int("max_val_batches", self.max_val_batches)
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
            # The target at timestep t is labels[start + t + action_offset].
            # Previous action must stay causal, so it is the observed input-time
            # action labels[start + t], not the frame before the future target.
            previous_start = start
            previous = labels[previous_start : previous_start + int(cfg.seq_len)]
            previous_action_windows.append(previous)
            meta.append(WindowMeta(video_path, start, start + int(cfg.seq_len)))
    if not label_windows:
        raise RuntimeError(
            "No valid sequence windows were created. Check CSV lengths, seq_len, and prediction_horizon."
        )
    return WindowTargets(
        labels=torch.stack(label_windows, dim=0),
        previous_actions=torch.stack(previous_action_windows, dim=0),
        meta=meta,
    )


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
    seed: Optional[int] = None,
):
    pipeline = video_window_pipeline(
        batch_size=int(batch_size),
        num_threads=int(cfg.dali_num_threads),
        device_id=0,
        seed=int(cfg.dali_shuffle_seed if seed is None else seed),
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


def action_threshold_tensor(thresholds: Sequence[float], *, device: torch.device) -> torch.Tensor:
    return torch.tensor(list(thresholds), device=device, dtype=torch.float32)


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


def apply_action_residual_dropout(
    previous_actions: torch.Tensor,
    cfg: TrainConfig,
    *,
    training: bool,
) -> Tuple[torch.Tensor, float]:
    """Randomly zero complete previous-action residual inputs during teacher forcing."""

    probability = float(cfg.action_token_dropout_prob)
    if not training or probability <= 0.0:
        return previous_actions, 0.0
    if previous_actions.dim() != 3:
        raise ValueError(f"Expected previous actions [B,T,A], got {tuple(previous_actions.shape)}.")
    keep = torch.rand(
        previous_actions.size(0),
        previous_actions.size(1),
        1,
        device=previous_actions.device,
    ) >= probability
    # Preserve the first previous-action input so each independent window still
    # has a truthful initial controller state.
    keep[:, 0] = True
    dropped = 1.0 - float(keep.float().mean().item())
    return previous_actions * keep.to(dtype=previous_actions.dtype), dropped


def use_autoregressive_feedback(*, training: bool, cfg: TrainConfig) -> bool:
    if not training or float(cfg.autoregressive_feedback_prob) <= 0.0:
        return False
    return random.random() < float(cfg.autoregressive_feedback_prob)


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
    force_autoregressive_feedback: bool = False,
    fit_thresholds: Optional[bool] = None,
    metric_thresholds: Optional[Sequence[float]] = None,
    feedback_thresholds: Optional[Sequence[float]] = None,
) -> Tuple[Dict[str, object], int]:
    training = optimizer is not None
    if force_autoregressive_feedback and training:
        raise ValueError("Forced autoregressive feedback is only supported for validation/eval epochs.")
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    if fit_thresholds is None:
        fit_thresholds = (not training and cfg.fit_thresholds_from_val)

    loss_sum = 0.0
    bce_loss_sum = 0.0
    vision_bce_loss_sum = 0.0
    conflict_loss_sum = 0.0
    stats = BinaryStats(cfg.num_bin, device)
    vision_stats = BinaryStats(cfg.num_bin, device)
    fitter = ThresholdFitter() if bool(fit_thresholds) else None
    vision_fitter = ThresholdFitter() if bool(fit_thresholds) else None
    metric_threshold_values = (
        tuple(float(item) for item in cfg.button_state_thresholds)
        if metric_thresholds is None
        else tuple(float(item) for item in metric_thresholds)
    )
    feedback_threshold_values = (
        tuple(float(item) for item in cfg.autoregressive_feedback_thresholds)
        if feedback_thresholds is None
        else tuple(float(item) for item in feedback_thresholds)
    )
    if len(metric_threshold_values) != cfg.num_bin:
        raise ValueError(f"Expected {cfg.num_bin} metric thresholds, got {len(metric_threshold_values)}.")
    if len(feedback_threshold_values) != cfg.num_bin:
        raise ValueError(f"Expected {cfg.num_bin} feedback thresholds, got {len(feedback_threshold_values)}.")
    thresholds = action_threshold_tensor(metric_threshold_values, device=device)
    feedback_thresholds_tensor = action_threshold_tensor(feedback_threshold_values, device=device)
    progress = tqdm(range(num_batches), desc=description, dynamic_ncols=True)
    iterator_it = iter(iterator)
    autoregressive_batches = 0
    action_dropout_sum = 0.0

    for batch_index in progress:
        frames, labels, previous_actions, _sample_ids = load_batch(
            iterator_it,
            targets,
            device=device,
            amp_dtype=amp_dtype,
        )
        if training:
            frames = augment_frames(frames, cfg, same_over_time=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=amp_dtype == torch.bfloat16,
            ):
                closed_loop = force_autoregressive_feedback or use_autoregressive_feedback(training=training, cfg=cfg)
                if closed_loop:
                    autoregressive_batches += 1
                    output = model(
                        frames,
                        return_sequence_logits=True,
                        prev_action=previous_actions[:, 0],
                        feedback_thresholds=feedback_thresholds_tensor,
                        autoregressive_feedback=True,
                    )
                else:
                    model_prev_actions, action_dropout = apply_action_residual_dropout(
                        previous_actions,
                        cfg,
                        training=training,
                    )
                    action_dropout_sum += action_dropout
                    output = model(
                        frames,
                        return_sequence_logits=True,
                        prev_action=model_prev_actions,
                    )
                logits = output.sequence_button_logits
                vision_logits = output.sequence_vision_button_logits
                if logits is None:
                    raise RuntimeError("Dense temporal supervision requires per-timestep policy logits.")
                if vision_logits is None:
                    raise RuntimeError("Previous-action training requires per-timestep vision-only logits.")
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
                    float(cfg.main_policy_loss_weight) * bce_loss
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
            if vision_fitter is not None:
                vision_fitter.update(flat_vision_logits, flat_labels)
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
                residual_drop=f"{action_dropout_sum / float(batch_index + 1):.3f}",
                closed_loop=f"{autoregressive_batches}/{batch_index + 1}",
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
        "action_token_dropout_rate": action_dropout_sum / max(1, num_batches),
        "autoregressive_batches": autoregressive_batches,
        "metric_thresholds": metric_threshold_values,
        "feedback_thresholds": feedback_threshold_values,
    }
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
    if vision_fitter is not None:
        vision_fitted_thresholds = vision_fitter.fit(cfg.threshold_min, cfg.threshold_max, cfg.num_bin)
        calibrated_vision_stats = vision_fitter.statistics(vision_fitted_thresholds, cfg.num_bin)
        metrics["vision_fitted_thresholds"] = vision_fitted_thresholds
        metrics["previous_threshold_vision_macro_f1"] = metrics["vision_macro_f1"]
        metrics["previous_threshold_vision_rows"] = metrics["vision_rows"]
        metrics["vision_macro_f1"] = calibrated_vision_stats.macro_f1()
        metrics["vision_rows"] = calibrated_vision_stats.rows(cfg.key_names)
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


def fill_missing_vision_head_state(model: torch.nn.Module, model_state: Dict[str, torch.Tensor]) -> bool:
    current_state = model.state_dict()
    missing = [
        key
        for key in current_state
        if key.startswith("policy.vision_head.") and key not in model_state
    ]
    if not missing:
        return False
    for key in missing:
        source_key = key.replace("policy.vision_head.", "policy.head.", 1)
        source = model_state.get(source_key)
        if source is not None and tuple(source.shape) == tuple(current_state[key].shape):
            model_state[key] = source.detach().clone()
        else:
            model_state[key] = current_state[key].detach().clone()
    return True


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
    model_state = state["model_state"]
    initialized_vision_head = fill_missing_vision_head_state(model, model_state)
    model.load_state_dict(model_state, strict=True)
    try:
        optimizer.load_state_dict(state["optimizer_state"])
    except ValueError as exc:
        if not initialized_vision_head:
            raise
        print(f"Skipping optimizer state after adding vision_head: {exc}")
    print(f"Resumed from {path} at epoch {state.get('epoch', 0)}.")
    if initialized_vision_head:
        print("Initialized missing vision_head weights from the checkpoint main policy head.")
    # V5 checkpoints use vision BCE as the model-selection score.
    best_validation_bce = float(state.get("best_validation_bce", math.inf))

    def checkpoint_int(name: str, fallback: int) -> int:
        value = config.get(name, fallback)
        return int(fallback if value is None else value)

    checkpoint_validation_contract = (
        checkpoint_int("seq_len", int(cfg.seq_len)),
        checkpoint_int("val_seq_stride", int(cfg.val_seq_stride)),
        checkpoint_int("action_offset", int(cfg.action_offset)),
    )
    current_validation_contract = (
        int(cfg.seq_len),
        int(cfg.val_seq_stride),
        int(cfg.action_offset),
    )
    if checkpoint_validation_contract != current_validation_contract:
        print(
            "Validation contract changed since checkpoint; resetting best validation vision BCE "
            f"from {best_validation_bce:.4f}."
        )
        best_validation_bce = math.inf
    return (
        int(state.get("epoch", 0)),
        int(state.get("global_step", 0)),
        best_validation_bce,
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
        f"frame_fusion={FUSED_CHANNELS}x{FUSED_SPATIAL_SIZE}x{FUSED_SPATIAL_SIZE}",
        f"tokens={cfg.seq_len}x{TOKENS_PER_STEP}={cfg.seq_len * TOKENS_PER_STEP}",
        f"visual_tokens_per_frame={NUM_VISUAL_TOKENS}",
        f"transformer={TEMPORAL_LAYERS}x{READOUT_CHANNELS}/heads={TEMPORAL_HEADS}",
    )
    print(
        "  Actions:",
        f"order={action_names}",
        f"action_offset=+{int(cfg.action_offset)}",
        f"thresholds={list(cfg.button_state_thresholds)}",
        f"feedback_thresholds={list(cfg.autoregressive_feedback_thresholds)}",
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
        f"activation_checkpointing={cfg.use_activation_checkpointing}",
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
        "  Supervision:",
        f"dense_causal={cfg.dense_temporal_supervision}",
        f"targets_per_window={cfg.seq_len}",
        f"action_offset=+{int(cfg.action_offset)}",
        "window_state=independent_transformer_context",
    )
    print(
        "  Loss:",
        "BCEWithLogits",
        f"pos_weight_power={cfg.pos_weight_power:.3f}",
        f"pos_weight_clamp={cfg.pos_weight_clamp:.3f}",
        f"main_policy_weight={cfg.main_policy_loss_weight:.3f}",
        f"conflict_weight={cfg.conflict_penalty_weight:.3f}",
        f"vision_aux_weight={cfg.vision_aux_loss_weight:.3f}",
    )
    print(
        "  Previous action:",
        f"late_residual={cfg.last_action_conditioning}",
        f"residual_cap={cfg.last_action_residual_cap:.3f}",
        f"teacher_forcing={1.0 - float(cfg.autoregressive_feedback_prob):.2f}",
        f"closed_loop_train={float(cfg.autoregressive_feedback_prob):.2f}",
        f"residual_dropout={float(cfg.action_token_dropout_prob):.3f}",
        f"validation={'teacher_forced+closed_loop' if cfg.autoregressive_validation else 'teacher_forced'}",
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
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfg.split_seed)
    random.seed(cfg.split_seed)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    sync_dataset_roots_for_training(cfg)
    train_pairs, val_pairs = resolve_run_pairs(cfg)
    train_targets = build_window_targets(train_pairs, cfg, stride=cfg.train_seq_stride)
    val_targets = build_window_targets(val_pairs, cfg, stride=cfg.val_seq_stride)
    train_file_list = os.path.join(cfg.ckpt_dir, "train_windows.txt")
    val_file_list = os.path.join(cfg.ckpt_dir, "val_windows.txt")
    write_window_file_list(train_targets.meta, train_file_list)
    write_window_file_list(val_targets.meta, val_file_list)

    train_batches = _batch_count(train_targets, cfg.batch_size, partial=False)
    val_batch_size = min(cfg.batch_size, len(val_targets.meta))
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
    try:
        optimizer = torch.optim.AdamW(
            base_model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            fused=True,
        )
    except (RuntimeError, TypeError):
        optimizer = torch.optim.AdamW(base_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    start_epoch, global_step, best_validation_bce = maybe_resume(base_model, optimizer, cfg, device)
    model = torch.compile(base_model) if cfg.compile_model else base_model
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
        train_iterator = make_dali_iterator(
            train_file_list,
            cfg,
            batch_size=cfg.batch_size,
            random_shuffle=cfg.dali_train_random_shuffle,
            last_batch_policy=LastBatchPolicy.DROP,
            seed=cfg.dali_shuffle_seed + epoch,
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
            )

        fitted = val_metrics.get("fitted_thresholds")
        if isinstance(fitted, tuple) and len(fitted) == cfg.num_bin:
            cfg.button_state_thresholds = tuple(float(item) for item in fitted)
        closed_loop_val_metrics: Optional[Dict[str, object]] = None
        if cfg.autoregressive_validation:
            with torch.inference_mode():
                closed_loop_val_metrics, _ = run_epoch(
                    # Forced autoregressive validation unrolls through a different
                    # graph shape; keep it eager to avoid torch.compile overhead.
                    model=base_model,
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
                    description=f"Epoch {epoch + 1}/{cfg.num_epochs} val closed-loop",
                    force_autoregressive_feedback=True,
                    fit_thresholds=cfg.fit_thresholds_from_val,
                    metric_thresholds=cfg.autoregressive_feedback_thresholds,
                    feedback_thresholds=cfg.autoregressive_feedback_thresholds,
                )
        score = float(val_metrics["macro_f1"])
        validation_bce = float(val_metrics["bce_loss"])
        validation_vision_bce = float(val_metrics["vision_bce_loss"])
        is_best = validation_vision_bce < best_validation_bce
        if is_best:
            best_validation_bce = validation_vision_bce
        closed_loop_summary = ""
        if closed_loop_val_metrics is not None:
            closed_loop_summary = (
                f" val_closed_loop_bce={float(closed_loop_val_metrics['bce_loss']):.4f} "
                f"val_closed_loop_f1={float(closed_loop_val_metrics['macro_f1']):.4f}"
            )
        print(
            f"Epoch {epoch + 1}/{cfg.num_epochs}: "
            f"train_loss={float(train_metrics['loss']):.4f} "
            f"train_conflict={float(train_metrics['conflict_penalty']):.4f} "
            f"train_f1={float(train_metrics['macro_f1']):.4f} "
            f"train_vision_f1={float(train_metrics['vision_macro_f1']):.4f} "
            f"train_residual_drop={float(train_metrics['action_token_dropout_rate']):.3f} "
            f"train_closed_loop={int(train_metrics['autoregressive_batches'])} "
            f"val_loss={float(val_metrics['loss']):.4f} "
            f"val_conflict={float(val_metrics['conflict_penalty']):.4f} "
            f"val_f1={score:.4f} "
            f"val_bce={validation_bce:.4f} "
            f"val_vision_bce={validation_vision_bce:.4f} "
            f"val_vision_f1={float(val_metrics['vision_macro_f1']):.4f}"
            f"{closed_loop_summary}"
        )
        print(
            f"Validation best_vision_bce={best_validation_bce:.4f} "
            f"current_vision_bce={validation_vision_bce:.4f} "
            f"calibrated_macro_f1={score:.4f} "
            f"new_best={is_best}"
        )
        print_metric_rows("Validation controls:", val_metrics["rows"])
        print_metric_rows("Validation vision-only controls:", val_metrics["vision_rows"])
        if closed_loop_val_metrics is not None:
            print(
                f"Closed-loop validation: "
                f"loss={float(closed_loop_val_metrics['loss']):.4f} "
                f"bce={float(closed_loop_val_metrics['bce_loss']):.4f} "
                f"conflict={float(closed_loop_val_metrics['conflict_penalty']):.4f} "
                f"f1={float(closed_loop_val_metrics['macro_f1']):.4f} "
                f"closed_loop={int(closed_loop_val_metrics['autoregressive_batches'])}/{val_batches}"
            )
            print_metric_rows("Closed-loop validation controls:", closed_loop_val_metrics["rows"])
        if fitted is not None:
            print("Validation thresholds:", dict(zip(cfg.key_names, cfg.button_state_thresholds)))
        vision_fitted = val_metrics.get("vision_fitted_thresholds")
        if isinstance(vision_fitted, tuple) and len(vision_fitted) == cfg.num_bin:
            print("Vision validation thresholds:", dict(zip(cfg.key_names, vision_fitted)))
        if closed_loop_val_metrics is not None:
            print(
                "Autoregressive feedback thresholds:",
                dict(zip(cfg.key_names, cfg.autoregressive_feedback_thresholds)),
            )
            closed_loop_fitted = closed_loop_val_metrics.get("fitted_thresholds")
            if isinstance(closed_loop_fitted, tuple) and len(closed_loop_fitted) == cfg.num_bin:
                print("Closed-loop validation thresholds:", dict(zip(cfg.key_names, closed_loop_fitted)))
            closed_loop_vision_fitted = closed_loop_val_metrics.get("vision_fitted_thresholds")
            if isinstance(closed_loop_vision_fitted, tuple) and len(closed_loop_vision_fitted) == cfg.num_bin:
                print(
                    "Closed-loop vision validation thresholds:",
                    dict(zip(cfg.key_names, closed_loop_vision_fitted)),
                )

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
    parser = argparse.ArgumentParser(description="Train the six-action CNN grid-token transformer driving policy with DALI.")
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
    add(
        "--action-offset",
        "--prediction-horizon",
        dest="action_offset",
        type=int,
        default=None,
        help="Future label shift: frame[i] predicts action[i + action_offset]. Default: 4.",
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
    add("--main-policy-loss-weight", type=float, default=None)
    add("--conflict-penalty-weight", type=float, default=None)
    add("--vision-aux-loss-weight", type=float, default=None)
    add(
        "--action-token-dropout-prob",
        type=float,
        default=None,
        help="Drop observed previous-action residual inputs during teacher forcing. Legacy argument name.",
    )
    add("--autoregressive-feedback-prob", type=float, default=None)
    add("--autoregressive-feedback-threshold", type=float, default=None)
    add("--autoregressive-feedback-thresholds", type=_parse_threshold_sequence, default=None)
    add("--autoregressive-validation", dest="autoregressive_validation", action="store_true", default=None)
    add("--no-autoregressive-validation", dest="autoregressive_validation", action="store_false")
    add("--threshold-min", type=float, default=None)
    add("--threshold-max", type=float, default=None)
    add("--fit-thresholds-from-val", dest="fit_thresholds_from_val", action="store_true", default=None)
    add("--no-fit-thresholds-from-val", dest="fit_thresholds_from_val", action="store_false")
    add("--amp-dtype", choices=["bf16", "fp32"], default=None)
    add("--compile", dest="compile_model", action="store_true", default=None)
    add("--no-compile", dest="compile_model", action="store_false")
    add("--activation-checkpointing", dest="use_activation_checkpointing", action="store_true", default=None)
    add("--no-activation-checkpointing", dest="use_activation_checkpointing", action="store_false")
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
