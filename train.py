import argparse
import csv
import glob
import hashlib
import math
import os
import random
import shutil
from dataclasses import asdict, dataclass, field
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
from dataset_wsl_sync import default_target_root, sync_dataset_for_training
from models import (
    ACTION_DECODER_ACTION_QUERY,
    ACTION_DECODER_MLP,
    ACTION_DECODERS,
    DrivingVideoPolicy,
    LAST_ACTION_EMBEDDING_DROPOUT,
    ModelConfig,
    PolicyOutput,
    TemporalState,
    apply_static_masks,
)


CONFLICTING_BUTTON_PAIRS = (("w", "s"), ("a", "d"))


def _require_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {value!r}.")
    return value


def _require_optional_bool(name: str, value: object) -> Optional[bool]:
    if value is None:
        return None
    return _require_bool(name, value)


def _require_int_at_least(name: str, value: object, minimum: int) -> int:
    result = _require_int(name, value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}.")
    return result


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc
    try:
        if float(result) != float(value):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc
    return result


def _require_int_between(name: str, value: object, minimum: int, maximum: int) -> int:
    result = _require_int_at_least(name, value, minimum)
    if result < minimum or result > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {result}.")
    return result


def _require_float_at_least(name: str, value: object, minimum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}.")
    return result


def _require_float_range(name: str, value: object, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result) or result < minimum or result > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {value!r}.")
    return result


def _normalize_optional_path(name: str, value: object) -> Optional[str]:
    if value is None:
        return None
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must not be empty.")
    return result


@dataclass
class TrainConfig(ModelConfig):
    action_decoder: str = ACTION_DECODER_ACTION_QUERY
    zoneout: float = 0.05

    batch_size: int = 1
    target_effective_batch: int = 8
    grad_accum: int = field(init=False)
    num_epochs: int = 50

    lr: float = 2e-4
    min_lr: float = 1e-5
    warmup_steps: int = 50
    weight_decay: float = 0.03
    grad_clip: float = 1.0

    amp_dtype: str = "bf16"
    compile_model: bool = True
    compile_mode: str = "default"

    train_data_root: Optional[str] = None
    val_data_root: Optional[str] = None
    train_split: float = 0.9
    split_seed: int = 1337
    pos_weight_power: float = 0.6
    pos_weight_clamp: float = 25
    button_threshold_from_pos_weight: bool = True
    button_threshold_min: float = 0.2
    button_threshold_max: float = 0.85
    fit_thresholds_from_val: bool = True
    eval_only: bool = False
    eval_ckpt: Optional[str] = None

    button_loss_weight: float = 1.0
    # Keep binary heads, but make impossible driving chords costly enough to
    # matter during training.
    conflicting_button_loss_weight: float = 0.20
    streaming_state_training: bool = True
    streaming_state_validation: bool = True
    streaming_segment_min_chunks: int = 2
    streaming_segment_max_chunks: int = 20
    action_label_offset: int = 0
    # Last-action conditioning is useful, but teacher-forced ground truth can
    # hide runtime latching. Scheduled feedback below trains recovery from the
    # model's own previous predictions while keeping ground-truth labels.
    last_action_conditioning: bool = True
    last_action_feedback_train_prob: float = 1.0
    last_action_feedback_warmup_epochs: int = 4
    last_action_feedback_ramp_epochs: int = 8
    last_action_feedback_validation: bool = True
    last_action_feedback_soft: bool = True
    skipped_key_names: Optional[Sequence[str]] = ("e", "q", "c", "z")
    # Extra BCE weight on frames where a key changes state. Transitions are
    # ~5% of labels but are all that matters for control.
    transition_loss_weight: float = 5.0

    aug_brightness: float = 0.15
    aug_contrast: float = 0.20
    aug_noise_std: float = 0.01
    aug_gray_prob: float = 0.05
    aug_translate_frac: float = 0.0
    aug_scale_frac: float = 0.0
    aug_edges_crop_prob: float = 0.0
    aug_edges_crop_min_frac: float = 0.02
    aug_edges_crop_max_frac: float = 0.06
    aug_cutout_prob: float = 0.0
    aug_cutout_min_frac: float = 0.04
    aug_cutout_max_frac: float = 0.12
    aug_cutout_count: int = 2

    early_stop_patience: int = 0

    dali_num_threads: int = 6
    dali_prefetch_queue_depth: int = 4
    dali_reader_prefetch_queue_depth: int = 4
    dali_read_ahead: bool = False
    dali_dont_use_mmap: bool = False
    dali_resize_mode: str = "video_resize"
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
        had_custom_thresholds = self.button_state_thresholds is not None
        super().__post_init__()
        skipped = tuple(str(name) for name in (self.skipped_key_names or ()))
        if skipped:
            unknown = sorted(set(skipped) - set(self.key_names))
            if unknown:
                raise ValueError(f"skipped_key_names contains unknown keys: {unknown}.")
            if had_custom_thresholds:
                raise ValueError(
                    "button_state_thresholds cannot be combined with skipped_key_names because "
                    "filtering changes the action count."
                )
            skip_set = set(skipped)
            self.key_names = [name for name in self.key_names if name not in skip_set]
            self.num_bin = len(self.key_names) + len(self.mouse_button_names)
            if self.num_bin <= 0:
                raise ValueError("At least one action key/button must remain after skipped_key_names filtering.")
            self.button_state_thresholds = tuple(float(self.button_state_threshold) for _ in range(self.num_bin))
        self.skipped_key_names = skipped

        self.batch_size = _require_int_at_least("batch_size", self.batch_size, 1)
        self.target_effective_batch = _require_int_at_least("target_effective_batch", self.target_effective_batch, 1)
        self.grad_accum = int(math.ceil(self.target_effective_batch / float(self.batch_size)))
        self.num_epochs = _require_int_at_least("num_epochs", self.num_epochs, 1)
        self.train_data_root = _normalize_optional_path("train_data_root", self.train_data_root)
        self.val_data_root = _normalize_optional_path("val_data_root", self.val_data_root)
        if (self.train_data_root is None) != (self.val_data_root is None):
            raise ValueError("train_data_root and val_data_root must be provided together.")
        self.lr = _require_float_at_least("lr", self.lr, 0.0)
        if self.lr <= 0.0:
            raise ValueError(f"lr must be > 0.0, got {self.lr}.")
        self.min_lr = _require_float_at_least("min_lr", self.min_lr, 0.0)
        if self.min_lr > self.lr:
            raise ValueError(f"min_lr must be <= lr, got min_lr={self.min_lr} lr={self.lr}.")
        self.weight_decay = _require_float_at_least("weight_decay", self.weight_decay, 0.0)
        self.dali_prefetch_queue_depth = _require_int_at_least(
            "dali_prefetch_queue_depth",
            self.dali_prefetch_queue_depth,
            1,
        )
        self.dali_reader_prefetch_queue_depth = _require_int_at_least(
            "dali_reader_prefetch_queue_depth",
            self.dali_reader_prefetch_queue_depth,
            1,
        )
        self.dali_train_random_shuffle = _require_bool("dali_train_random_shuffle", self.dali_train_random_shuffle)
        self.dali_val_random_shuffle = _require_bool("dali_val_random_shuffle", self.dali_val_random_shuffle)
        self.dali_prepare_first_batch = _require_bool("dali_prepare_first_batch", self.dali_prepare_first_batch)
        self.save_every = _require_int_at_least("save_every", self.save_every, 1)
        self.print_every = _require_int_at_least("print_every", self.print_every, 1)
        self.warmup_steps = _require_int_at_least("warmup_steps", self.warmup_steps, 0)
        self.train_split = _require_float_range("train_split", self.train_split, 0.05, 0.95)
        self.split_seed = _require_int("split_seed", self.split_seed)
        self.dali_shuffle_seed = _require_int("dali_shuffle_seed", self.dali_shuffle_seed)
        self.pos_weight_power = _require_float_at_least("pos_weight_power", self.pos_weight_power, 0.0)
        self.pos_weight_clamp = _require_float_at_least("pos_weight_clamp", self.pos_weight_clamp, 1.0)
        self.button_threshold_from_pos_weight = _require_bool(
            "button_threshold_from_pos_weight",
            self.button_threshold_from_pos_weight,
        )
        self.fit_thresholds_from_val = _require_bool("fit_thresholds_from_val", self.fit_thresholds_from_val)
        self.eval_only = _require_bool("eval_only", self.eval_only)
        if self.eval_ckpt is not None:
            self.eval_ckpt = str(self.eval_ckpt)
        self.transition_loss_weight = max(1.0, float(self.transition_loss_weight))
        self.button_threshold_min = _require_float_range("button_threshold_min", self.button_threshold_min, 0.0, 1.0)
        self.button_threshold_max = _require_float_range("button_threshold_max", self.button_threshold_max, 0.0, 1.0)
        if self.button_threshold_max < self.button_threshold_min:
            raise ValueError(
                "button_threshold_max must be >= button_threshold_min, "
                f"got max={self.button_threshold_max} min={self.button_threshold_min}."
            )
        self.button_loss_weight = _require_float_at_least("button_loss_weight", self.button_loss_weight, 0.0)
        self.conflicting_button_loss_weight = _require_float_at_least(
            "conflicting_button_loss_weight",
            self.conflicting_button_loss_weight,
            0.0,
        )
        self.streaming_state_training = _require_bool("streaming_state_training", self.streaming_state_training)
        self.streaming_state_validation = _require_bool("streaming_state_validation", self.streaming_state_validation)
        self.streaming_segment_min_chunks = _require_int_at_least(
            "streaming_segment_min_chunks",
            self.streaming_segment_min_chunks,
            1,
        )
        self.streaming_segment_max_chunks = _require_int_at_least(
            "streaming_segment_max_chunks",
            self.streaming_segment_max_chunks,
            1,
        )
        if self.streaming_segment_max_chunks < self.streaming_segment_min_chunks:
            raise ValueError(
                "streaming_segment_max_chunks must be >= streaming_segment_min_chunks, "
                f"got max={self.streaming_segment_max_chunks} min={self.streaming_segment_min_chunks}."
            )
        self.grad_clip = _require_float_at_least("grad_clip", self.grad_clip, 0.0)
        self.action_label_offset = _require_int("action_label_offset", self.action_label_offset)
        self.last_action_feedback_train_prob = _require_float_range(
            "last_action_feedback_train_prob",
            self.last_action_feedback_train_prob,
            0.0,
            1.0,
        )
        self.last_action_feedback_warmup_epochs = _require_int_at_least(
            "last_action_feedback_warmup_epochs",
            self.last_action_feedback_warmup_epochs,
            0,
        )
        self.last_action_feedback_ramp_epochs = _require_int_at_least(
            "last_action_feedback_ramp_epochs",
            self.last_action_feedback_ramp_epochs,
            1,
        )
        self.last_action_feedback_validation = _require_bool(
            "last_action_feedback_validation",
            self.last_action_feedback_validation,
        )
        self.last_action_feedback_soft = _require_bool("last_action_feedback_soft", self.last_action_feedback_soft)
        self.aug_brightness = _require_float_range("aug_brightness", self.aug_brightness, 0.0, 0.5)
        self.aug_contrast = _require_float_range("aug_contrast", self.aug_contrast, 0.0, 0.5)
        self.aug_noise_std = _require_float_range("aug_noise_std", self.aug_noise_std, 0.0, 0.1)
        self.aug_gray_prob = _require_float_range("aug_gray_prob", self.aug_gray_prob, 0.0, 1.0)
        self.aug_translate_frac = _require_float_range("aug_translate_frac", self.aug_translate_frac, 0.0, 0.25)
        self.aug_scale_frac = _require_float_range("aug_scale_frac", self.aug_scale_frac, 0.0, 0.50)
        self.aug_edges_crop_prob = _require_float_range("aug_edges_crop_prob", self.aug_edges_crop_prob, 0.0, 1.0)
        self.aug_edges_crop_min_frac = _require_float_range(
            "aug_edges_crop_min_frac",
            self.aug_edges_crop_min_frac,
            0.0,
            0.45,
        )
        self.aug_edges_crop_max_frac = _require_float_range(
            "aug_edges_crop_max_frac",
            self.aug_edges_crop_max_frac,
            0.0,
            0.45,
        )
        if self.aug_edges_crop_max_frac < self.aug_edges_crop_min_frac:
            raise ValueError(
                "aug_edges_crop_max_frac must be >= aug_edges_crop_min_frac, "
                f"got max={self.aug_edges_crop_max_frac} min={self.aug_edges_crop_min_frac}."
            )
        self.aug_cutout_prob = _require_float_range("aug_cutout_prob", self.aug_cutout_prob, 0.0, 1.0)
        self.aug_cutout_min_frac = _require_float_range("aug_cutout_min_frac", self.aug_cutout_min_frac, 0.0, 0.75)
        self.aug_cutout_max_frac = _require_float_range("aug_cutout_max_frac", self.aug_cutout_max_frac, 0.0, 0.75)
        if self.aug_cutout_max_frac < self.aug_cutout_min_frac:
            raise ValueError(
                "aug_cutout_max_frac must be >= aug_cutout_min_frac, "
                f"got max={self.aug_cutout_max_frac} min={self.aug_cutout_min_frac}."
            )
        self.aug_cutout_count = _require_int_between("aug_cutout_count", self.aug_cutout_count, 1, 16)
        self.early_stop_patience = _require_int_at_least("early_stop_patience", self.early_stop_patience, 0)
        self.dali_resize_mode = str(self.dali_resize_mode).strip().lower()
        if self.dali_resize_mode not in {"video_resize", "none"}:
            raise ValueError(
                "dali_resize_mode must be one of: video_resize, none; "
                f"got {self.dali_resize_mode!r}."
            )
        self.dali_num_threads = _require_int_at_least("dali_num_threads", self.dali_num_threads, 1)
        self.dali_read_ahead = _require_bool("dali_read_ahead", self.dali_read_ahead)
        self.dali_dont_use_mmap = _require_bool("dali_dont_use_mmap", self.dali_dont_use_mmap)
        self.sync_dataset = _require_bool("sync_dataset", self.sync_dataset)
        self.dataset_sync_delete_stale = _require_optional_bool(
            "dataset_sync_delete_stale",
            self.dataset_sync_delete_stale,
        )
        self.dataset_sync_hash_same_size = _require_bool(
            "dataset_sync_hash_same_size",
            self.dataset_sync_hash_same_size,
        )
        self.resume = _require_bool("resume", self.resume)
        self.max_train_batches = (
            None
            if self.max_train_batches is None
            else _require_int_at_least("max_train_batches", self.max_train_batches, 1)
        )
        self.max_val_batches = (
            None
            if self.max_val_batches is None
            else _require_int_at_least("max_val_batches", self.max_val_batches, 1)
        )
        if not os.path.isabs(self.ckpt_dir):
            self.ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.ckpt_dir))


@dataclass
class WindowTargets:
    button_target: torch.Tensor
    target_valid: torch.Tensor
    last_action: torch.Tensor
    # 1.0 where the labelled key state differs from the true previous frame
    # (computed from the full CSV, so chunk-boundary flips at frame 0 count).
    button_transition: Optional[torch.Tensor] = None
    meta: Optional[List[Tuple[str, int, int]]] = None


StreamingStateCache = Dict[str, Dict[int, TemporalState]]


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


@dataclass
class ThresholdFitter:
    """Per-key histograms of predicted probabilities on validation data,
    used to pick the F1-optimal decision threshold for each key."""

    num_classes: int
    device: torch.device
    bins: int = 512

    def __post_init__(self) -> None:
        self.pos = torch.zeros(self.num_classes, self.bins, dtype=torch.float64, device=self.device)
        self.neg = torch.zeros(self.num_classes, self.bins, dtype=torch.float64, device=self.device)

    @torch.no_grad()
    def update(self, probs: torch.Tensor, true: torch.Tensor, valid: torch.Tensor) -> None:
        if probs.shape != true.shape:
            raise ValueError(f"probs shape {tuple(probs.shape)} must match true shape {tuple(true.shape)}.")
        idx = (probs.float().clamp(0.0, 1.0) * (self.bins - 1)).long()
        class_offset = torch.arange(self.num_classes, device=idx.device).view(*([1] * (idx.dim() - 1)), -1)
        flat = idx + class_offset * self.bins
        sel = valid.unsqueeze(-1).expand_as(true)
        flat = flat[sel]
        is_pos = true[sel]
        if flat.numel() == 0:
            return
        size = self.num_classes * self.bins
        self.pos += torch.bincount(flat[is_pos], minlength=size).reshape(self.num_classes, self.bins).to(self.pos.dtype)
        self.neg += torch.bincount(flat[~is_pos], minlength=size).reshape(self.num_classes, self.bins).to(self.neg.dtype)

    def fit(self, min_threshold: float, max_threshold: float) -> Tuple[List[Optional[float]], List[float]]:
        """Return (per-key F1-optimal threshold or None when unsupported, per-key F1 at it).

        A sample is predicted positive when prob >= threshold; bin i collects
        probs in [i/(bins-1), (i+1)/(bins-1)), so the count of bins >= i equals
        the positives at threshold i/(bins-1).
        """
        tp = self.pos.flip(-1).cumsum(-1).flip(-1)
        fp = self.neg.flip(-1).cumsum(-1).flip(-1)
        fn = self.pos.sum(dim=-1, keepdim=True) - tp
        f1 = (2.0 * tp) / (2.0 * tp + fp + fn).clamp(min=1e-9)
        best_idx = f1.argmax(dim=-1)
        best_f1 = f1.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)
        support = self.pos.sum(dim=-1)
        thresholds = (best_idx.to(torch.float64) / float(self.bins - 1)).clamp(
            min=float(min_threshold), max=float(max_threshold)
        )
        fitted = [
            float(threshold) if float(count) > 0.0 else None
            for threshold, count in zip(thresholds.tolist(), support.tolist())
        ]
        return fitted, [float(x) for x in best_f1.tolist()]


def find_runs(
    data_root: str,
    video_ext: str,
    csv_ext: str,
    *,
    video_name_glob: str = "run_*",
) -> List[Tuple[str, str]]:
    video_files = sorted(glob.glob(os.path.join(data_root, f"{video_name_glob}{video_ext}")))
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


def split_cache_root(cfg: TrainConfig, split_name: str) -> str:
    base_root = str(cfg.dataset_cache_root) if cfg.dataset_cache_root is not None else str(default_target_root())
    return os.path.join(base_root, split_name)


def detected_split_roots(data_root: str) -> Optional[Tuple[str, str]]:
    train_root = os.path.join(data_root, "train")
    val_root = os.path.join(data_root, "val")
    if os.path.isdir(train_root) and os.path.isdir(val_root):
        return train_root, val_root
    return None


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_cached_file(source_path: str, target_path: str, *, hash_same_size: bool) -> bool:
    if not os.path.exists(target_path):
        return False
    source_stat = os.stat(source_path)
    target_stat = os.stat(target_path)
    if source_stat.st_size != target_stat.st_size:
        return False
    if source_stat.st_mtime_ns == target_stat.st_mtime_ns:
        return True
    if not hash_same_size:
        return False
    return _sha256_file(source_path) == _sha256_file(target_path)


def sync_clip_folder_for_training(
    *,
    data_root: str,
    target_root: str,
    split_name: str,
    cfg: TrainConfig,
) -> str:
    source_root = os.path.abspath(data_root)
    cache_root = os.path.abspath(target_root)
    try:
        if os.path.samefile(source_root, cache_root):
            print(f"Dataset cache already selected for {split_name}: {cache_root}")
            return cache_root
    except FileNotFoundError:
        if source_root == cache_root:
            print(f"Dataset cache already selected for {split_name}: {cache_root}")
            return cache_root

    pairs = find_runs(source_root, cfg.video_ext, cfg.csv_ext, video_name_glob="*")
    if not pairs:
        raise RuntimeError(
            f"No {split_name} clips found under {source_root!r} "
            f"(expected *{cfg.video_ext} with matching {cfg.csv_ext})."
        )

    os.makedirs(cache_root, exist_ok=True)
    expected_paths = set()
    files_seen = 0
    unchanged = 0
    copied = 0
    bytes_copied = 0
    for video_path, csv_path in pairs:
        for source_path in (video_path, csv_path):
            files_seen += 1
            target_path = os.path.join(cache_root, os.path.basename(source_path))
            expected_paths.add(os.path.abspath(target_path))
            if _same_cached_file(source_path, target_path, hash_same_size=bool(cfg.dataset_sync_hash_same_size)):
                unchanged += 1
                continue
            shutil.copy2(source_path, target_path)
            copied += 1
            bytes_copied += int(os.path.getsize(source_path))

    deleted = 0
    should_delete_stale = True if cfg.dataset_sync_delete_stale is None else bool(cfg.dataset_sync_delete_stale)
    if should_delete_stale:
        stale_patterns = [f"*{cfg.video_ext}"]
        if cfg.csv_ext != cfg.video_ext:
            stale_patterns.append(f"*{cfg.csv_ext}")
        for pattern in stale_patterns:
            for path in glob.glob(os.path.join(cache_root, pattern)):
                if os.path.isfile(path) and os.path.abspath(path) not in expected_paths:
                    os.remove(path)
                    deleted += 1

    print(f"Dataset cache ready for {split_name}:")
    print(f"  source={source_root}")
    print(f"  target={cache_root}")
    print(
        "  "
        f"files={files_seen} unchanged={unchanged} copied={copied} "
        f"bytes_copied={bytes_copied} deleted_stale={deleted}"
    )
    return cache_root


def resolve_train_val_pairs(cfg: TrainConfig) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    split_source = "manual"
    if cfg.train_data_root is not None and cfg.val_data_root is not None:
        train_root = cfg.train_data_root
        val_root = cfg.val_data_root
    else:
        detected_roots = detected_split_roots(cfg.data_root)
        if detected_roots is None:
            train_root = None
            val_root = None
        else:
            train_root, val_root = detected_roots
            split_source = "detected"

    if train_root is not None and val_root is not None:
        if cfg.sync_dataset:
            train_root = sync_clip_folder_for_training(
                data_root=train_root,
                target_root=split_cache_root(cfg, "train"),
                split_name="train",
                cfg=cfg,
            )
            val_root = sync_clip_folder_for_training(
                data_root=val_root,
                target_root=split_cache_root(cfg, "val"),
                split_name="val",
                cfg=cfg,
            )
        cfg.train_data_root = train_root
        cfg.val_data_root = val_root

        train_pairs = find_runs(train_root, cfg.video_ext, cfg.csv_ext, video_name_glob="*")
        val_pairs = find_runs(val_root, cfg.video_ext, cfg.csv_ext, video_name_glob="*")
        if not train_pairs:
            raise RuntimeError(
                f"No train clips found under {train_root!r} "
                f"(expected *{cfg.video_ext} with matching {cfg.csv_ext})."
            )
        if not val_pairs:
            raise RuntimeError(
                f"No validation clips found under {val_root!r} "
                f"(expected *{cfg.video_ext} with matching {cfg.csv_ext})."
            )
        print(f"{split_source.title()} split roots: train={train_root} val={val_root}")
        print(f"Runs: train={len(train_pairs)} val={len(val_pairs)}")
        return train_pairs, val_pairs

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
    return train_pairs, val_pairs


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
    stride = _require_int_at_least("stride", stride, 1)
    button_windows: List[np.ndarray] = []
    valid_windows: List[np.ndarray] = []
    last_action_windows: List[np.ndarray] = []
    transition_windows: List[np.ndarray] = []
    meta: List[Tuple[str, int, int]] = []

    prediction_offset = int(tuple(cfg.prediction_horizon_offsets)[0])
    for video_path, csv_path in pairs:
        run = load_run_arrays(csv_path, cfg)
        buttons = run["buttons"]
        if buttons.shape[0] < cfg.seq_len:
            continue

        max_start = buttons.shape[0] - cfg.seq_len
        for start in range(0, max_start + 1, stride):
            end = start + cfg.seq_len
            button_target = np.zeros((cfg.seq_len, cfg.num_bin), dtype=np.float32)
            valid_target = np.zeros((cfg.seq_len,), dtype=np.float32)
            last_action = np.zeros((cfg.seq_len, cfg.num_bin), dtype=np.float32)
            transition_target = np.zeros((cfg.seq_len, cfg.num_bin), dtype=np.float32)

            frame_indices = np.arange(start, end)
            context_indices = frame_indices + int(cfg.action_label_offset)
            context_valid = (context_indices >= 0) & (context_indices < buttons.shape[0])
            if np.any(context_valid):
                last_action[context_valid] = buttons[context_indices[context_valid]]
            target_indices = frame_indices + prediction_offset + int(cfg.action_label_offset)
            prev_indices = target_indices - 1
            valid = (target_indices >= 0) & (target_indices < buttons.shape[0]) & (prev_indices >= 0)
            if np.any(valid):
                valid_target_indices = target_indices[valid]
                button_target[valid] = buttons[valid_target_indices]
                transition_target[valid] = (
                    buttons[valid_target_indices] != buttons[prev_indices[valid]]
                ).astype(np.float32)
                valid_target[valid] = 1.0

            button_windows.append(button_target)
            valid_windows.append(valid_target)
            last_action_windows.append(last_action)
            transition_windows.append(transition_target)
            if return_meta:
                meta.append((video_path, start, end))

    if not button_windows:
        raise RuntimeError("No training windows found. Check data_root, seq_len, and stride.")

    def stack(items: List[np.ndarray]) -> torch.Tensor:
        return torch.from_numpy(np.stack(items, axis=0)).float()

    return WindowTargets(
        button_target=stack(button_windows),
        target_valid=stack(valid_windows),
        last_action=stack(last_action_windows),
        button_transition=stack(transition_windows),
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


def _contiguous_window_streams(meta: List[Tuple[str, int, int]]) -> List[List[int]]:
    by_video: Dict[str, Dict[int, int]] = {}
    for idx, (video_path, start, _) in enumerate(meta):
        by_video.setdefault(video_path, {})[int(start)] = idx

    streams: List[List[int]] = []
    for start_to_idx in by_video.values():
        starts = sorted(start_to_idx)
        start_set = set(starts)
        end_set = {int(meta[idx][2]) for idx in start_to_idx.values()}
        for start in starts:
            if int(start) in end_set:
                continue

            stream: List[int] = []
            cursor = int(start)
            while cursor in start_set:
                stream_idx = start_to_idx[cursor]
                stream.append(stream_idx)
                _, _, cursor = meta[stream_idx]
            if stream:
                streams.append(stream)
    return streams


def _split_stream_random_segments(
    stream: Sequence[int],
    *,
    min_chunks: int,
    max_chunks: int,
    rng: random.Random,
) -> List[List[int]]:
    stream = list(stream)
    if not stream:
        return []

    min_chunks = int(min_chunks)
    max_chunks = int(max_chunks)
    if min_chunks < 1:
        raise ValueError(f"min_chunks must be >= 1, got {min_chunks}.")
    if max_chunks < min_chunks:
        raise ValueError(f"max_chunks must be >= min_chunks, got max={max_chunks} min={min_chunks}.")
    segments: List[List[int]] = []
    cursor = 0
    while cursor < len(stream):
        remaining = len(stream) - cursor
        if remaining < min_chunks and segments:
            segments[-1].extend(stream[cursor:])
            break
        if remaining <= max_chunks:
            length = remaining
        else:
            length = rng.randint(min_chunks, max_chunks)
            trailing = remaining - length
            if 0 < trailing < min_chunks:
                length = remaining - min_chunks
        segments.append(stream[cursor : cursor + length])
        cursor += length
    return segments


def streaming_window_order(
    meta: List[Tuple[str, int, int]],
    cfg: TrainConfig,
    *,
    seed: int,
) -> Tuple[List[int], set[int]]:
    rng = random.Random(int(seed))
    segments: List[List[int]] = []
    for stream in _contiguous_window_streams(meta):
        segments.extend(
            _split_stream_random_segments(
                stream,
                min_chunks=cfg.streaming_segment_min_chunks,
                max_chunks=cfg.streaming_segment_max_chunks,
                rng=rng,
            )
        )
    rng.shuffle(segments)
    order = [idx for segment in segments for idx in segment]
    reset_indices = {segment[0] for segment in segments if segment}
    return order, reset_indices


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
    filenames=None,
    labels=None,
):
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

    if resize_mode not in {"video_resize", "none"}:
        raise ValueError(f"Unknown DALI resize_mode: {resize_mode}")
    use_reader_resize = resize_mode == "video_resize"
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
            # Pin the current default so a future DALI upgrade cannot silently
            # shift every window's frames by one.
            reader_kwargs["file_list_include_preceding_frame"] = False
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
    _require_int_at_least("seq_len", seq_len, 1)
    return 0


def supervised_frame_range(seq_len: int) -> Tuple[int, int]:
    seq_len = _require_int_at_least("seq_len", seq_len, 1)
    return supervised_start_frame(seq_len), seq_len


def supervised_frames_only(valid: torch.Tensor) -> torch.Tensor:
    if valid.dim() < 2:
        raise ValueError(f"Expected a time dimension in valid mask, got shape {tuple(valid.shape)}")
    time = torch.arange(valid.size(1), device=valid.device) >= supervised_start_frame(valid.size(1))
    view_shape = [1] * valid.dim()
    view_shape[1] = valid.size(1)
    time = time.view(*view_shape)
    if valid.dtype == torch.bool:
        return valid & time
    return valid * time.to(dtype=valid.dtype)


def warmup_masked_valid(valid: torch.Tensor, fresh_state: Optional[torch.Tensor]) -> torch.Tensor:
    """Return the valid mask for supervised frames.

    All frames are supervised; fresh_state is retained for shape validation and
    compatibility with the streaming-state call sites.
    """
    if fresh_state is None:
        return supervised_frames_only(valid)
    if fresh_state.dim() != 1 or int(fresh_state.size(0)) != int(valid.size(0)):
        raise ValueError(
            f"Expected fresh_state shape [{int(valid.size(0))}], got {tuple(fresh_state.shape)}."
        )
    fresh = fresh_state.to(device=valid.device, dtype=torch.bool).view(-1, *([1] * (valid.dim() - 1)))
    return torch.where(fresh, supervised_frames_only(valid), valid)


def decision_thresholds_from_pos_weight(pos_weight: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    """Undo the probability overshoot introduced by weighted BCE.

    Training with pos_weight rho drives the optimal output to
    p = rho*q / (rho*q + 1 - q) for true conditional q, so deciding q > 0.5
    requires thresholding p at rho/(rho+1), not 0.5. alpha < 1 softens the
    correction back toward 0.5 (trading precision for recall on rare keys).
    These act as priors for epoch 1; val-fitted thresholds replace them.
    """
    if not bool(cfg.button_threshold_from_pos_weight):
        return torch.full_like(pos_weight.float(), float(cfg.button_state_threshold))

    alpha = 0.35
    effective_weight = 1.0 + (pos_weight.float() - 1.0) * alpha

    thresholds = effective_weight / (effective_weight + 1.0)

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


def move_bundle_to_device(bundle: WindowTargets, device: torch.device) -> WindowTargets:
    return WindowTargets(
        button_target=bundle.button_target.to(device, non_blocking=True),
        target_valid=bundle.target_valid.to(device, non_blocking=True),
        last_action=bundle.last_action.to(device, non_blocking=True),
        button_transition=(
            None if bundle.button_transition is None else bundle.button_transition.to(device, non_blocking=True)
        ),
        meta=bundle.meta,
    )


def action_index(cfg: TrainConfig, name: str) -> Optional[int]:
    name_to_idx = {str(action_name).lower(): idx for idx, action_name in enumerate(cfg.key_names)}
    return name_to_idx.get(str(name).lower())


def conflicting_button_loss(
    button_logits: torch.Tensor,
    loss_weight: torch.Tensor,
    cfg: TrainConfig,
) -> torch.Tensor:
    if float(cfg.conflicting_button_loss_weight) <= 0.0:
        return button_logits.new_zeros(())
    probs = torch.sigmoid(button_logits)
    conflicts: List[torch.Tensor] = []
    for first, second in CONFLICTING_BUTTON_PAIRS:
        first_idx = action_index(cfg, first)
        second_idx = action_index(cfg, second)
        if first_idx is not None and second_idx is not None:
            conflicts.append(probs[..., first_idx] * probs[..., second_idx])
    if not conflicts:
        return button_logits.new_zeros(())

    pair_conflict = torch.stack(conflicts, dim=-1).sum(dim=-1)
    weight = loss_weight.squeeze(-1)
    return (pair_conflict * weight).sum() / weight.sum().clamp(min=1.0)


def compute_losses(
    output: PolicyOutput,
    targets: WindowTargets,
    cfg: TrainConfig,
    *,
    button_pos_weight: torch.Tensor,
    fresh_state_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    valid = warmup_masked_valid(targets.target_valid, fresh_state_mask).float()
    valid_weight = valid.unsqueeze(-1)
    button_target = targets.button_target.float()
    button_logits = output.button_logits.float()
    button_loss_raw = F.binary_cross_entropy_with_logits(
        button_logits,
        button_target,
        pos_weight=button_pos_weight.view(1, 1, -1).float(),
        reduction="none",
    )
    loss_weight = valid_weight
    if float(cfg.transition_loss_weight) > 1.0:
        # Per-key upweight where the label flips vs the previous frame; the
        # weighted mean keeps the loss scale stable as the weight changes.
        if targets.button_transition is not None:
            # Built from the full CSV, so flips at frame 0 (chunk boundaries,
            # now supervised on carried-state chunks) are upweighted too.
            transition = targets.button_transition > 0.5
        else:
            raw_labels = targets.button_target > 0.5
            transition = torch.zeros_like(raw_labels)
            transition[:, 1:] = raw_labels[:, 1:] != raw_labels[:, :-1]
        button_weight = loss_weight * (
            1.0 + (float(cfg.transition_loss_weight) - 1.0) * transition.float()
        )
        button_loss = (button_loss_raw * button_weight).sum() / button_weight.sum().clamp(min=1.0)
    else:
        button_loss = (button_loss_raw * loss_weight).sum() / (loss_weight.sum() * cfg.num_bin).clamp(min=1.0)
    conflict_loss = conflicting_button_loss(button_logits, loss_weight, cfg)

    total = cfg.button_loss_weight * button_loss + cfg.conflicting_button_loss_weight * conflict_loss
    return total, {
        "button": button_loss.detach(),
        "conflict": conflict_loss.detach(),
    }


def prepare_last_action_context(last_action: torch.Tensor) -> torch.Tensor:
    prev_action = last_action.float()
    if prev_action.dim() != 3:
        raise ValueError(f"Expected last_action [B,T,C], got {tuple(prev_action.shape)}.")
    return prev_action


def scheduled_last_action_feedback_prob(cfg: TrainConfig, epoch_index: int) -> float:
    if not bool(cfg.last_action_conditioning):
        return 0.0
    max_prob = float(cfg.last_action_feedback_train_prob)
    if max_prob <= 0.0:
        return 0.0
    warmup = int(cfg.last_action_feedback_warmup_epochs)
    if int(epoch_index) < warmup:
        return 0.0
    ramp = max(1, int(cfg.last_action_feedback_ramp_epochs))
    progress = min(1.0, max(0.0, (int(epoch_index) - warmup + 1) / float(ramp)))
    return float(max_prob) * progress


def make_last_action_feedback_mask(prev_action: torch.Tensor, feedback_prob: float) -> Optional[torch.Tensor]:
    prob = float(feedback_prob)
    if prob <= 0.0:
        return None
    if prev_action.dim() != 3:
        raise ValueError(f"Expected prev_action [B,T,C], got {tuple(prev_action.shape)}.")
    batch_size, time_steps, _ = prev_action.shape
    # Sample feedback at the chunk level so chunks not selected for closed-loop
    # training keep the fast vectorized action-head path.
    if prob < 1.0 and random.random() >= prob:
        return None
    mask_shape = (int(batch_size), int(time_steps), 1)
    mask = torch.ones(mask_shape, device=prev_action.device, dtype=torch.bool)
    if time_steps > 0:
        mask[:, 0] = False
    return mask


def forward_policy(
    model: torch.nn.Module,
    frames: torch.Tensor,
    *,
    cfg: TrainConfig,
    state: Optional[TemporalState],
    return_aux: bool,
    prev_action: torch.Tensor,
    feedback_mask: Optional[torch.Tensor],
) -> PolicyOutput | Tuple[PolicyOutput, TemporalState]:
    if feedback_mask is None or not bool(cfg.last_action_conditioning):
        return model(
            frames,
            state=state,
            return_aux=return_aux,
            prev_action=prev_action,
        )

    return model(
        frames,
        state=state,
        return_aux=return_aux,
        prev_action=prev_action,
        feedback_mask=feedback_mask,
        feedback_thresholds=button_threshold_tensor(cfg, device=prev_action.device, dtype=torch.float32),
        soft_feedback=bool(cfg.last_action_feedback_soft),
    )


@torch.no_grad()
def update_metrics(
    output: PolicyOutput,
    targets: WindowTargets,
    cfg: TrainConfig,
    button_stats: BinaryStats,
    fresh_state_mask: Optional[torch.Tensor] = None,
) -> None:
    valid = warmup_masked_valid(targets.target_valid, fresh_state_mask)
    thresholds = button_threshold_tensor(cfg, device=output.button_logits.device).view(1, 1, -1)
    button_pred = torch.sigmoid(output.button_logits.float()) >= thresholds
    button_stats.update(button_pred, targets.button_target > 0.5, valid > 0.5)


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
    current = targets.button_target.new_zeros(targets.button_target.shape)
    current[:, 1:] = targets.button_target[:, :-1]
    valid = supervised_frames_only(targets.target_valid > 0.5)
    valid[:, 0] = False
    stats.update(current > 0.5, targets.button_target > 0.5, valid)
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

    return score_rows(metrics.get("button_rows", []), metrics["button_macro_f1"])


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


def load_batch(
    iterator,
    targets: WindowTargets,
    device: torch.device,
    cfg: TrainConfig,
    *,
    need_label_indices: bool = False,
) -> Tuple[torch.Tensor, WindowTargets, Optional[List[int]]]:
    batch = next(iterator)[0]
    if "frames" not in batch or "labels" not in batch:
        raise RuntimeError(f"DALI batch must contain 'frames' and 'labels', got keys={sorted(batch)}.")
    frames = ensure_fchw_layout(batch["frames"])
    if frames.device != device:
        frames = frames.to(device, non_blocking=True)
    amp_name = str(cfg.amp_dtype).strip().lower()
    if amp_name == "bf16" and frames.dtype != torch.bfloat16:
        frames = frames.to(dtype=torch.bfloat16)
    elif amp_name in {"fp32", "float32", "none"} and frames.dtype != torch.float32:
        frames = frames.float()
    labels = normalize_dali_labels(batch["labels"])
    if labels.numel() == 0:
        raise RuntimeError("DALI batch returned no labels.")
    target_count = int(targets.button_target.size(0))
    label_indices = (
        [int(item) for item in labels.detach().cpu().reshape(-1).tolist()]
        if need_label_indices
        else None
    )
    if label_indices is not None:
        bad = [idx for idx in label_indices if idx < 0 or idx >= target_count][:8]
    else:
        bad_mask = (labels < 0) | (labels >= target_count)
        bad = labels[bad_mask][:8].detach().cpu().tolist() if bool(bad_mask.any().item()) else []
    if bad:
        raise RuntimeError(f"DALI labels out of range for {target_count} windows: {bad}.")
    labels_for_index = labels.to(device, non_blocking=True)
    target = WindowTargets(
        button_target=targets.button_target[labels_for_index],
        target_valid=targets.target_valid[labels_for_index],
        last_action=targets.last_action[labels_for_index],
        button_transition=(
            None if targets.button_transition is None else targets.button_transition[labels_for_index]
        ),
        meta=None,
    )
    return frames, target, label_indices


def _streaming_initial_state(
    label_indices: Sequence[int],
    targets: WindowTargets,
    cache: StreamingStateCache,
    reset_indices: Optional[set[int]] = None,
) -> Tuple[Optional[TemporalState], List[bool]]:
    """Return the initial temporal state plus a per-sample carried flag."""
    if targets.meta is None or not cache:
        return None, [False] * len(label_indices)

    reset_indices = reset_indices or set()
    sample_states: List[Optional[TemporalState]] = []
    template_hidden: Optional[torch.Tensor] = None
    for label_idx in label_indices:
        video_path, start, _ = targets.meta[label_idx]
        video_cache = cache.get(video_path, {})
        cached = video_cache.pop(int(start), None)
        if label_idx in reset_indices:
            cached = None
        sample_states.append(cached)
        if cached is not None and cached.hidden_state is not None and template_hidden is None:
            template_hidden = cached.hidden_state

    carried_flags = [cached is not None and cached.hidden_state is not None for cached in sample_states]
    if template_hidden is None:
        return None, carried_flags

    stacked_hidden = torch.stack(
        [
            cached.hidden_state
            if cached is not None and cached.hidden_state is not None
            else torch.zeros_like(template_hidden)
            for cached in sample_states
        ],
        dim=1,
    )
    return TemporalState(hidden_state=stacked_hidden), carried_flags


def _update_streaming_cache(
    label_indices: Sequence[int],
    targets: WindowTargets,
    state: Optional[TemporalState],
    cache: StreamingStateCache,
    cfg: TrainConfig,
) -> None:
    if targets.meta is None or state is None or state.hidden_state is None:
        return

    hidden = state.hidden_state.detach()
    max_pending = max(2, int(math.ceil(float(cfg.seq_len) / float(cfg.train_seq_stride))) + 2)
    for batch_idx, label_idx in enumerate(label_indices):
        video_path, _, end = targets.meta[label_idx]
        video_cache = cache.setdefault(video_path, {})
        video_cache[int(end)] = TemporalState(
            hidden_state=hidden[:, batch_idx].detach(),
        )
        while len(video_cache) > max_pending:
            # Evict the oldest-inserted entry (stale leftovers from out-of-order
            # segments), never the freshly stored state for the active segment.
            video_cache.pop(next(iter(video_cache)))



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
    streaming_reset_indices: Optional[set[int]] = None,
    last_action_feedback_prob: float = 0.0,
) -> Tuple[Dict[str, float], int]:
    is_train = optimizer is not None
    model.train(is_train)
    if is_train:
        optimizer.zero_grad(set_to_none=True)

    loss_sum = torch.zeros((), device=device)
    detail_sums = {name: torch.zeros((), device=device) for name in ("button", "conflict")}
    button_stats = BinaryStats(cfg.num_bin, device)
    threshold_fitter = (
        ThresholdFitter(cfg.num_bin, device)
        if (not is_train and bool(getattr(cfg, "fit_thresholds_from_val", False)))
        else None
    )
    steps = 0
    streaming_cache: StreamingStateCache = {}
    segment_aug_seeds: Dict[str, int] = {}
    use_streaming_state = bool(
        cfg.streaming_state_training and (is_train or cfg.streaming_state_validation)
    )
    pre_mask_before_augmentation = bool(
        float(cfg.aug_translate_frac) > 0.0 or float(cfg.aug_scale_frac) > 0.0
    )
    stream_carried = 0
    stream_total = 0
    feedback_selected = torch.zeros((), device=device)
    feedback_total = 0

    iterator_it = iter(iterator)
    pbar = tqdm(range(int(batches)), desc=desc, dynamic_ncols=True)
    for batch_idx in pbar:
        frames, batch_targets, label_indices = load_batch(
            iterator_it,
            targets,
            device,
            cfg,
            need_label_indices=use_streaming_state,
        )

        initial_state: Optional[TemporalState] = None
        fresh_state_mask: Optional[torch.Tensor] = None
        carried_flags: List[bool] = []
        if use_streaming_state:
            if label_indices is None:
                raise RuntimeError("Streaming state training requires DALI label indices.")
            initial_state, carried_flags = _streaming_initial_state(
                label_indices,
                targets,
                streaming_cache,
                streaming_reset_indices if is_train else None,
            )
            stream_carried += sum(1 for flag in carried_flags if flag)
            stream_total += len(carried_flags)
            fresh_state_mask = torch.tensor(
                [not flag for flag in carried_flags],
                device=device,
                dtype=torch.bool,
            )

        if is_train:
            # Pre-mask so geometric augmentation cannot shift HUD/minimap
            # pixels out from under the model's fixed masks (train-only
            # leakage the model could learn to read).
            if pre_mask_before_augmentation:
                frames = apply_static_masks(frames, clone=False)
            aug_generator: Optional[torch.Generator] = None
            if use_streaming_state and targets.meta is not None and len(carried_flags) == 1:
                # One transform per streaming segment: re-seeding the generator
                # with the segment's seed makes every chunk draw identical
                # augmentation params, so the carried ConvGRU state never sees
                # the scene appearance jump mid-episode.
                if label_indices is None:
                    raise RuntimeError("Streaming augmentation requires DALI label indices.")
                video_path = targets.meta[label_indices[0]][0]
                if not carried_flags[0] or video_path not in segment_aug_seeds:
                    segment_aug_seeds[video_path] = random.getrandbits(63)
                aug_generator = torch.Generator(device=frames.device)
                aug_generator.manual_seed(segment_aug_seeds[video_path])
            frames = augment_frames(frames, cfg, generator=aug_generator)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_autocast):
                prev_action = prepare_last_action_context(batch_targets.last_action)
                feedback_mask = make_last_action_feedback_mask(prev_action, last_action_feedback_prob)
                if float(last_action_feedback_prob) > 0.0:
                    feedback_total += int(prev_action[:, 1:, :1].numel())
                if feedback_mask is not None:
                    feedback_selected += feedback_mask.sum().to(dtype=feedback_selected.dtype)
                if use_streaming_state:
                    output, next_state = forward_policy(
                        model,
                        frames,
                        cfg=cfg,
                        state=initial_state,
                        return_aux=True,
                        prev_action=prev_action,
                        feedback_mask=feedback_mask,
                    )
                    if label_indices is None:
                        raise RuntimeError("Streaming state cache update requires DALI label indices.")
                    _update_streaming_cache(label_indices, targets, next_state, streaming_cache, cfg)
                else:
                    output = forward_policy(
                        model,
                        frames,
                        cfg=cfg,
                        state=None,
                        return_aux=False,
                        prev_action=prev_action,
                        feedback_mask=feedback_mask,
                    )
                loss, details = compute_losses(
                    output,
                    batch_targets,
                    cfg,
                    button_pos_weight=button_pos_weight,
                    fresh_state_mask=fresh_state_mask,
                )
                loss_div = loss / int(cfg.grad_accum)

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
            button_stats,
            fresh_state_mask=fresh_state_mask,
        )
        if threshold_fitter is not None:
            threshold_fitter.update(
                torch.sigmoid(output.button_logits.float()),
                batch_targets.button_target > 0.5,
                warmup_masked_valid(batch_targets.target_valid, fresh_state_mask) > 0.5,
            )
        loss_sum += loss.detach().float()
        for name, value in details.items():
            detail_sums[name] += value.float()
        steps += 1
        if cfg.print_every and ((batch_idx + 1) % cfg.print_every == 0 or batch_idx + 1 == batches):
            postfix = {
                "loss": float((loss_sum / max(1, steps)).item()),
                "btn": float((detail_sums["button"] / max(1, steps)).item()),
                "conf": float((detail_sums["conflict"] / max(1, steps)).item()),
            }
            if use_streaming_state:
                postfix["carry"] = float(stream_carried / max(1, stream_total))
            if feedback_total > 0:
                postfix["fb"] = float((feedback_selected / max(1, feedback_total)).item())
            pbar.set_postfix(postfix)
    iterator.reset()

    button = button_stats.compute()
    button_names = list(cfg.key_names) + list(cfg.mouse_button_names)
    metrics = {
        "loss": float((loss_sum / max(1, steps)).item()),
        "button_loss": float((detail_sums["button"] / max(1, steps)).item()),
        "conflicting_button_loss": float((detail_sums["conflict"] / max(1, steps)).item()),
        "button_macro_f1": button["macro_f1"],
        "button_macro_precision": button["macro_precision"],
        "button_macro_recall": button["macro_recall"],
        "per_class_summary": per_class_f1_summary(button_stats, button_names),
        "button_rows": binary_stats_rows(button_stats, button_names),
    }
    if use_streaming_state:
        metrics["stream_state_carry_rate"] = float(stream_carried / max(1, stream_total))
    if feedback_total > 0:
        metrics["last_action_feedback_rate"] = float((feedback_selected / max(1, feedback_total)).item())
    if threshold_fitter is not None:
        fitted_thresholds, fitted_f1 = threshold_fitter.fit(cfg.button_threshold_min, cfg.button_threshold_max)
        metrics["fitted_button_thresholds"] = fitted_thresholds
        metrics["fitted_button_f1"] = fitted_f1
    return metrics, global_step


def latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    path = os.path.join(ckpt_dir, "model_latest.pt")
    if os.path.exists(path):
        return path
    return None


def uses_removed_vector_gru(checkpoint_config: object, model_state: object) -> bool:
    if isinstance(checkpoint_config, dict):
        if str(checkpoint_config.get("temporal_architecture", "")).strip().lower() == "vector_gru":
            return True
    if isinstance(model_state, dict):
        vector_prefixes = (
            "vector_pool.",
            "frame_fc1.",
            "frame_fc2.",
            "temporal_rnn.",
            "temporal_fusion.",
        )
        return any(str(key).startswith(vector_prefixes) for key in model_state)
    return False


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
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": asdict(cfg),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_score": float(best_score),
    }
    torch.save(payload, path)


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
    if not isinstance(state, dict):
        raise RuntimeError(f"Cannot resume checkpoint {ckpt_path}: expected a checkpoint dict.")
    missing_keys = [key for key in ("model_state", "optimizer_state", "epoch", "global_step", "best_score") if key not in state]
    if missing_keys:
        raise RuntimeError(f"Cannot resume checkpoint {ckpt_path}: missing keys {missing_keys}.")
    ckpt_config = state.get("config") if isinstance(state, dict) else None
    if uses_removed_vector_gru(ckpt_config, state.get("model_state")):
        message = "checkpoint was trained with the removed vector_gru architecture"
        if cfg.resume_path is not None:
            raise RuntimeError(f"Cannot resume checkpoint {ckpt_path}: {message}.")
        print(f"Skipping incompatible checkpoint {ckpt_path}: {message}. Starting a fresh run.")
        return 0, 0, -1e9
    ckpt_decoder = ACTION_DECODER_MLP
    if isinstance(ckpt_config, dict):
        ckpt_decoder = str(ckpt_config.get("action_decoder", ACTION_DECODER_MLP)).strip().lower()
    current_decoder = str(cfg.action_decoder).strip().lower()
    if ckpt_decoder != current_decoder:
        message = (
            f"Checkpoint decoder {ckpt_decoder!r} does not match current decoder "
            f"{current_decoder!r}."
        )
        if cfg.resume_path is not None:
            raise RuntimeError(f"Cannot resume checkpoint {ckpt_path}: {message}")
        print(f"Skipping incompatible checkpoint {ckpt_path}: {message} Starting a fresh run.")
        return 0, 0, -1e9
    try:
        model.load_state_dict(state["model_state"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"Cannot resume checkpoint {ckpt_path}: model_state does not match the current "
            f"policy config (num_bin={cfg.num_bin}, d_model={cfg.d_model}, "
            f"horizon={cfg.prediction_horizon}). Start a fresh run with --no-resume "
            "or choose a compatible --resume-path."
        ) from exc
    try:
        optimizer.load_state_dict(state["optimizer_state"])
    except (RuntimeError, ValueError, KeyError) as exc:
        raise RuntimeError(
            f"Cannot resume checkpoint {ckpt_path}: optimizer_state is incompatible with "
            "the current optimizer/model parameters. Start a fresh run with --no-resume "
            "or choose a compatible --resume-path."
        ) from exc
    return int(state["epoch"]), int(state["global_step"]), float(state["best_score"])


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the CNN+temporal behavioral cloning policy.")
    add = parser.add_argument
    add("--data-root", default=None)
    add(
        "--train-data-root",
        "--train-folder",
        "--train-dir",
        dest="train_data_root",
        default=None,
        help="Explicit training clip folder. Scans *<video-ext> files with same-stem CSVs.",
    )
    add(
        "--val-data-root",
        "--val-folder",
        "--val-dir",
        dest="val_data_root",
        default=None,
        help="Explicit validation clip folder. Must be provided with --train-data-root.",
    )
    add("--ckpt-dir", default=None)
    add("--resume", dest="resume", action="store_true", default=None)
    add("--no-resume", dest="resume", action="store_false")
    add("--resume-path", default=None)
    add("--num-epochs", type=int, default=None)
    add("--batch-size", type=int, default=None)
    add("--seq-len", type=int, default=None)
    add("--prediction-horizon", type=int, default=None, help="Legacy option; single-horizon policy requires 1.")
    add("--prediction-horizon-offsets", default=None, help="Single future frame offset, e.g. 1.")
    add("--model-size", type=int, default=None)
    add("--d-model", type=int, default=None)
    add("--action-decoder", choices=list(ACTION_DECODERS), default=None)
    add("--action-query-heads", type=int, default=None)
    add("--action-query-layers", type=int, default=None)
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
    add("--fit-thresholds-from-val", dest="fit_thresholds_from_val", action="store_true", default=None)
    add("--no-fit-thresholds-from-val", dest="fit_thresholds_from_val", action="store_false")
    add("--eval-only", dest="eval_only", action="store_true", default=None, help="Run a single validation pass on a checkpoint and exit (no training).")
    add("--eval-ckpt", default=None, help="Checkpoint for --eval-only. Defaults to <ckpt-dir>/model_best.pt.")
    add("--conflicting-button-loss-weight", type=float, default=None)
    add("--streaming-state-training", dest="streaming_state_training", action="store_true", default=None)
    add("--no-streaming-state-training", dest="streaming_state_training", action="store_false")
    add("--streaming-state-validation", dest="streaming_state_validation", action="store_true", default=None)
    add("--no-streaming-state-validation", dest="streaming_state_validation", action="store_false")
    add("--streaming-segment-min-chunks", type=int, default=None)
    add("--streaming-segment-max-chunks", type=int, default=None)
    add("--action-label-offset", type=int, default=None)
    add("--last-action-conditioning", dest="last_action_conditioning", action="store_true", default=None)
    add("--no-last-action-conditioning", dest="last_action_conditioning", action="store_false")
    add("--transition-loss-weight", type=float, default=None, help="Extra BCE weight on key state changes; 1 disables.")
    add("--last-action-feedback-train-prob", type=float, default=None, help="Max probability of feeding detached model predictions back as prev_action during training.")
    add("--last-action-feedback-warmup-epochs", type=int, default=None)
    add("--last-action-feedback-ramp-epochs", type=int, default=None)
    add("--last-action-feedback-validation", dest="last_action_feedback_validation", action="store_true", default=None)
    add("--no-last-action-feedback-validation", dest="last_action_feedback_validation", action="store_false")
    add("--last-action-feedback-soft", dest="last_action_feedback_soft", action="store_true", default=None)
    add("--last-action-feedback-hard", dest="last_action_feedback_soft", action="store_false")
    add("--skip-key-names", default=None, help="Comma-separated key names to exclude from training labels.")
    add("--train-all-keys", action="store_true", help="Disable the default Greenville test filter for e,q,c,z.")
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
    add("--dali-resize-mode", choices=["video_resize", "none"], default=None)
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
        "train_data_root",
        "val_data_root",
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
        "action_decoder",
        "action_query_heads",
        "action_query_layers",
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
        "fit_thresholds_from_val",
        "eval_only",
        "eval_ckpt",
        "conflicting_button_loss_weight",
        "streaming_state_training",
        "streaming_state_validation",
        "streaming_segment_min_chunks",
        "streaming_segment_max_chunks",
        "action_label_offset",
        "last_action_conditioning",
        "transition_loss_weight",
        "last_action_feedback_train_prob",
        "last_action_feedback_warmup_epochs",
        "last_action_feedback_ramp_epochs",
        "last_action_feedback_validation",
        "last_action_feedback_soft",
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
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DALI video training.")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    train_pairs, val_pairs = resolve_train_val_pairs(cfg)
    if cfg.skipped_key_names:
        print(f"Skipping action keys for this training run: {', '.join(cfg.skipped_key_names)}")
    print(f"Training action keys: {', '.join(cfg.key_names + cfg.mouse_button_names)}")
    prediction_offset = int(tuple(cfg.prediction_horizon_offsets)[0])
    print(f"Prediction frame offset: +{prediction_offset}")

    train_targets = build_window_targets(train_pairs, cfg, stride=cfg.train_seq_stride, return_meta=True)
    val_targets = build_window_targets(val_pairs, cfg, stride=cfg.val_seq_stride, return_meta=True) if val_pairs else None
    print(
        "Windows:",
        f"train={tuple(train_targets.button_target.shape)}",
        f"val={(tuple(val_targets.button_target.shape) if val_targets is not None else None)}",
        f"action_label_offset={cfg.action_label_offset}",
        f"prediction_offset=+{prediction_offset}",
    )
    supervised_start, supervised_end = supervised_frame_range(cfg.seq_len)
    print(
        "Loss supervision:",
        f"frames={supervised_start}-{supervised_end}",
        f"conflict_weight={float(cfg.conflicting_button_loss_weight):.4f}",
        f"transition_weight={float(cfg.transition_loss_weight):.1f}",
    )
    decoder_parts = ["Policy decoder:", f"type={cfg.action_decoder}"]
    if cfg.action_decoder == ACTION_DECODER_ACTION_QUERY:
        decoder_parts.extend(
            [
                f"queries={int(cfg.num_bin)}",
                f"heads={int(cfg.action_query_heads)}",
                f"layers={int(cfg.action_query_layers)}",
            ]
        )
    print(" ".join(decoder_parts))
    print("Temporal regularization:", f"zoneout={float(cfg.zoneout):.2f}")
    print(
        "Streaming state training:",
        f"enabled={bool(cfg.streaming_state_training)}",
        f"val_streaming={bool(cfg.streaming_state_validation)}",
        "shuffle=stream_segments" if cfg.streaming_state_training else f"train_shuffle={bool(cfg.dali_train_random_shuffle)}",
        f"segment_chunks={int(cfg.streaming_segment_min_chunks)}-{int(cfg.streaming_segment_max_chunks)}",
        f"train_stride={int(cfg.train_seq_stride)}",
        f"seq_len={int(cfg.seq_len)}",
    )
    print(
        "Last action conditioning:",
        f"enabled={bool(cfg.last_action_conditioning)}",
        f"embedding_dropout={LAST_ACTION_EMBEDDING_DROPOUT:.2f}",
    )
    print(
        "Last action feedback:",
        f"train_max={cfg.last_action_feedback_train_prob:.2f}",
        f"warmup_epochs={int(cfg.last_action_feedback_warmup_epochs)}",
        f"ramp_epochs={int(cfg.last_action_feedback_ramp_epochs)}",
        f"val_closed_loop={bool(cfg.last_action_feedback_validation)}",
        f"mode={'soft' if cfg.last_action_feedback_soft else 'hard'}",
    )
    train_persist = persistence_baseline_metrics(train_targets, cfg)
    print(f"Persistence baseline: train_f1@+{prediction_offset}={train_persist['macro_f1']:.4f}")
    if val_targets is not None:
        val_persist = persistence_baseline_metrics(val_targets, cfg)
        print(f"Persistence baseline: val_f1@+{prediction_offset}={val_persist['macro_f1']:.4f} {val_persist['per_class_summary']}")

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
        train_targets.button_target,
        cfg.pos_weight_power,
        cfg.pos_weight_clamp,
        valid=supervised_frames_only(train_targets.target_valid),
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

    if cfg.streaming_state_training and int(cfg.batch_size) != 1:
        raise ValueError("streaming_state_training currently requires batch_size=1 so chunks can carry state sequentially.")

    train_targets = move_bundle_to_device(train_targets, device)
    if val_targets is not None:
        val_targets = move_bundle_to_device(val_targets, device)

    train_iter = None
    if not cfg.streaming_state_training:
        train_iter = make_dali_iterator(
            train_file_list,
            cfg,
            batch_size=cfg.batch_size,
            random_shuffle=cfg.dali_train_random_shuffle,
            last_batch_policy=LastBatchPolicy.DROP,
        )
    train_batches = int(train_targets.button_target.shape[0]) // int(cfg.batch_size)
    if cfg.max_train_batches is not None:
        train_batches = min(train_batches, cfg.max_train_batches)
    if train_batches <= 0:
        raise RuntimeError("No training batches available.")

    val_iter = None
    val_batches = 0
    if val_targets is not None:
        val_batch_size = min(max(1, cfg.batch_size), int(val_targets.button_target.shape[0]))
        val_iter = make_dali_iterator(
            val_file_list,
            cfg,
            batch_size=val_batch_size,
            random_shuffle=cfg.dali_val_random_shuffle,
            last_batch_policy=LastBatchPolicy.PARTIAL,
        )
        val_batches = int(math.ceil(int(val_targets.button_target.shape[0]) / float(val_batch_size)))
        if cfg.max_val_batches is not None:
            val_batches = min(val_batches, cfg.max_val_batches)

    amp_dtype, use_autocast, use_scaler = resolve_amp_settings(cfg.amp_dtype)
    if use_scaler:
        raise RuntimeError("This trainer supports bf16/fp32 only.")
    print(f"AMP: dtype={amp_dtype} autocast={use_autocast}")

    eval_state = None
    if cfg.eval_only:
        eval_ckpt_path = cfg.eval_ckpt or os.path.join(cfg.ckpt_dir, "model_best.pt")
        eval_state = torch.load(eval_ckpt_path, map_location=device)
        ckpt_config = eval_state.get("config") if isinstance(eval_state, dict) else None
        if isinstance(ckpt_config, dict):
            # architecture-affecting flag must match the checkpoint under eval
            # (legacy checkpoints predate the flag and were trained with it on)
            if uses_removed_vector_gru(ckpt_config, eval_state.get("model_state")):
                raise RuntimeError(
                    f"Cannot eval checkpoint {eval_ckpt_path}: it was trained with the removed "
                    "vector_gru architecture."
                )
            cfg.last_action_conditioning = bool(ckpt_config.get("last_action_conditioning", True))
            cfg.action_decoder = str(ckpt_config.get("action_decoder", ACTION_DECODER_MLP)).strip().lower()
            cfg.action_query_heads = int(ckpt_config.get("action_query_heads", cfg.action_query_heads))
            cfg.action_query_layers = int(ckpt_config.get("action_query_layers", cfg.action_query_layers))

    base_model: torch.nn.Module = DrivingVideoPolicy(cfg).to(device)
    base_model = base_model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, fused=True)
    print(f"Parameters: {sum(p.numel() for p in base_model.parameters()) / 1e6:.4f}M")
    if cfg.eval_only:
        eval_ckpt = cfg.eval_ckpt or os.path.join(cfg.ckpt_dir, "model_best.pt")
        base_model.load_state_dict(eval_state["model_state"])
        print(
            f"Eval-only: loaded {eval_ckpt} (epoch={eval_state.get('epoch')}"
            f" best_score={eval_state.get('best_score')}"
            f" last_action_conditioning={bool(cfg.last_action_conditioning)})"
        )
        start_epoch, global_step, best_score = 0, 0, float("-inf")
    else:
        start_epoch, global_step, best_score = maybe_resume(base_model, optimizer, cfg, device)

    if cfg.compile_model:
        compile_kwargs = {"fullgraph": False, "dynamic": False}
        if cfg.compile_mode and cfg.compile_mode.lower() != "default":
            compile_kwargs["mode"] = cfg.compile_mode
        model = torch.compile(base_model, **compile_kwargs)
        print(f"torch.compile enabled: mode={cfg.compile_mode}")
    else:
        model = base_model

    if cfg.eval_only:
        if val_iter is None or val_targets is None or val_batches <= 0:
            raise RuntimeError("eval_only requires a validation split.")
        with torch.inference_mode():
            val_metrics, _ = run_epoch(
                desc="Eval [val]",
                model=model,
                iterator=val_iter,
                batches=val_batches,
                targets=val_targets,
                cfg=cfg,
                device=device,
                amp_dtype=amp_dtype,
                use_autocast=use_autocast,
                button_pos_weight=button_pos_weight,
                last_action_feedback_prob=1.0 if bool(cfg.last_action_feedback_validation) else 0.0,
            )
        print(
            f"Eval: va_loss={val_metrics['loss']:.4f}"
            f" va_f1@+{prediction_offset}={val_metrics['button_macro_f1']:.4f}"
            f" va_carry={val_metrics.get('stream_state_carry_rate', 0.0):.3f}"
        )
        print_button_stats_table(
            f"Eval val per-key/button @+{prediction_offset}:",
            val_metrics["button_rows"],
        )
        fitted = val_metrics.get("fitted_button_thresholds")
        if fitted:
            fitted_f1 = val_metrics.get("fitted_button_f1") or [0.0] * len(button_names)
            print(
                f"Fitted val thresholds @+{prediction_offset}: "
                + " ".join(
                    f"{name}={'n/a' if threshold is None else f'{threshold:.3f}'}(f1={f1:.3f})"
                    for name, threshold, f1 in zip(button_names, fitted, fitted_f1)
                )
            )
        return

    optimizer_steps_per_epoch = int(math.ceil(train_batches / float(cfg.grad_accum)))
    total_steps = max(1, optimizer_steps_per_epoch * cfg.num_epochs)
    epochs_without_improvement = 0

    for epoch in range(start_epoch, cfg.num_epochs):
        streaming_reset_indices: Optional[set[int]] = None
        epoch_train_iter = train_iter
        if cfg.streaming_state_training:
            if train_targets.meta is None:
                raise RuntimeError("Streaming state training requires training window metadata.")
            stream_order, streaming_reset_indices = streaming_window_order(
                train_targets.meta,
                cfg,
                seed=int(cfg.dali_shuffle_seed) + int(epoch),
            )
            if not stream_order:
                raise RuntimeError("No streaming train windows found.")
            write_window_file_list(train_targets.meta, train_file_list, indices=stream_order)
            epoch_train_iter = make_dali_iterator(
                train_file_list,
                cfg,
                batch_size=cfg.batch_size,
                random_shuffle=False,
                last_batch_policy=LastBatchPolicy.DROP,
            )
        if epoch_train_iter is None:
            raise RuntimeError("Training iterator was not initialized.")
        train_feedback_prob = scheduled_last_action_feedback_prob(cfg, epoch)
        train_metrics, global_step = run_epoch(
            desc=f"Epoch {epoch + 1}/{cfg.num_epochs} [train]",
            model=model,
            iterator=epoch_train_iter,
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
            streaming_reset_indices=streaming_reset_indices,
            last_action_feedback_prob=train_feedback_prob,
        )

        val_metrics = None
        score = driving_score(train_metrics, cfg)
        if val_iter is not None and val_targets is not None and val_batches > 0:
            val_feedback_prob = 1.0 if bool(cfg.last_action_feedback_validation) else 0.0
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
                    last_action_feedback_prob=val_feedback_prob,
                )
            score = driving_score(val_metrics, cfg)
            fitted_thresholds = val_metrics.get("fitted_button_thresholds")
            if fitted_thresholds:
                current_thresholds = list(cfg.button_state_thresholds)
                cfg.button_state_thresholds = tuple(
                    current if new is None else float(new)
                    for current, new in zip(current_thresholds, fitted_thresholds)
                )
                fitted_f1 = val_metrics.get("fitted_button_f1") or [0.0] * len(button_names)
                print(
                    f"Fitted val thresholds @+{prediction_offset} (deployed): "
                    + " ".join(
                        f"{name}={threshold:.3f}(f1={f1:.3f})"
                        for name, threshold, f1 in zip(button_names, cfg.button_state_thresholds, fitted_f1)
                    )
                )

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
            f"tr_f1@+{prediction_offset}={train_metrics['button_macro_f1']:.4f}",
        ]
        if "stream_state_carry_rate" in train_metrics:
            parts.append(f"tr_carry={train_metrics['stream_state_carry_rate']:.3f}")
        if "last_action_feedback_rate" in train_metrics:
            parts.append(f"tr_fb={train_metrics['last_action_feedback_rate']:.3f}")
        if val_metrics is not None:
            parts.extend(
                [
                    f"va_loss={val_metrics['loss']:.4f}",
                    f"va_f1@+{prediction_offset}={val_metrics['button_macro_f1']:.4f}",
                    (
                        f"va_carry={val_metrics['stream_state_carry_rate']:.3f}"
                        if "stream_state_carry_rate" in val_metrics
                        else ""
                    ),
                    (
                        f"va_fb={val_metrics['last_action_feedback_rate']:.3f}"
                        if "last_action_feedback_rate" in val_metrics
                        else ""
                    ),
                    f"best={best_score:.4f}",
                    val_metrics["per_class_summary"],
                ]
            )
        else:
            parts.append(train_metrics["per_class_summary"])
        print(" | ".join(part for part in parts if part))
        train_stats = (
            f"Epoch {epoch + 1} train stats: "
            f"f1@+{prediction_offset}={train_metrics['button_macro_f1']:.4f} "
            f"prec@+{prediction_offset}={train_metrics['button_macro_precision']:.4f} "
            f"rec@+{prediction_offset}={train_metrics['button_macro_recall']:.4f}"
        )
        print(train_stats)
        print_button_stats_table(
            f"Epoch {epoch + 1} train per-key/button @+{prediction_offset}:",
            train_metrics["button_rows"],
        )
        if val_metrics is not None:
            val_stats = (
                f"Epoch {epoch + 1} val stats: "
                f"f1@+{prediction_offset}={val_metrics['button_macro_f1']:.4f} "
                f"prec@+{prediction_offset}={val_metrics['button_macro_precision']:.4f} "
                f"rec@+{prediction_offset}={val_metrics['button_macro_recall']:.4f}"
            )
            print(val_stats)
            print_button_stats_table(
                f"Epoch {epoch + 1} val per-key/button @+{prediction_offset}:",
                val_metrics["button_rows"],
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
