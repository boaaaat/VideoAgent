import argparse
import csv
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
#   cd /mnt/c/Users/Abhil/Desktop/vs_code_stuff/python/ai
#   python train_inverse.py

from inverse_dynamics import (
    INVERSE_VISUAL_ENCODER_NAME,
    InverseDynamicsConfig,
    InverseDynamicsModel,
    center_window_bounds,
    inverse_checkpoint_family_mismatch_reason,
    initialize_inverse_lazy_layers,
)
from train import (
    BinaryStats,
    MouseStats,
    binary_stats_rows,
    configure_attention_backend,
    compute_pos_weight,
    ensure_fchw_layout,
    find_runs,
    maybe_channels_last_seq,
    normalize_dali_labels,
    resolve_amp_settings,
    print_button_stats_table,
    video_pipeline,
    warmup_cosine_lr,
)

from dataset_wsl_sync import sync_dataset_for_training
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

# WSL:
#   cd ~/ai
#   source venv/bin/activate
#   cd /mnt/c/Users/Abhil/Desktop/vs_code_stuff/python/ai
#   python train_inverse.py

@dataclass
class InverseTrainConfig(InverseDynamicsConfig):
    batch_size: int = 4
    target_effective_batch: int = 32
    grad_accum: int = 8
    num_epochs: int = 40

    lr: float = 3e-4
    min_lr: float = 3e-6
    warmup_steps: int = 500
    weight_decay: float = 0.05
    grad_clip: float = 1.0

    amp_dtype: str = "bf16"
    compile_model: bool = True
    compile_mode: str = "default"
    attention_backend: str = "flash"

    train_split: float = 0.85
    split_seed: int = 1337
    pos_weight_power: float = 0.5
    pos_weight_clamp: float = 12.0
    scale_percentile: float = 95.0

    button_loss_weight: float = 1.0
    mouse_delta_loss_weight: float = 1.0
    mouse_active_loss_weight: float = 0.10
    scroll_delta_loss_weight: float = 0.10
    active_mouse_loss_mult: float = 8.0
    mouse_delta_loss_type: str = "l1"
    mouse_large_delta_loss_mult: float = 2.0

    dali_num_threads: int = 8
    dali_prefetch_queue_depth: int = 8
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
    ckpt_dir: str = "./checkpoints_idm"
    save_every: int = 1
    print_every: int = 20
    max_train_batches: Optional[int] = None
    max_val_batches: Optional[int] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.batch_size = max(1, int(self.batch_size))
        self.target_effective_batch = max(1, int(self.target_effective_batch))
        self.num_epochs = max(1, int(self.num_epochs))
        self.dali_prefetch_queue_depth = max(1, int(self.dali_prefetch_queue_depth))
        self.dali_reader_prefetch_queue_depth = max(1, int(self.dali_reader_prefetch_queue_depth))
        self.save_every = max(1, int(self.save_every))
        self.print_every = max(1, int(self.print_every))
        self.grad_accum = max(1, int(math.ceil(self.target_effective_batch / float(self.batch_size))))
        self.warmup_steps = max(0, int(self.warmup_steps))
        self.split_seed = int(self.split_seed)
        self.dali_shuffle_seed = int(self.dali_shuffle_seed)
        self.train_split = float(min(max(self.train_split, 0.05), 0.95))
        self.scale_percentile = float(min(max(self.scale_percentile, 50.0), 99.9))
        self.pos_weight_power = max(0.0, float(self.pos_weight_power))
        self.pos_weight_clamp = max(1.0, float(self.pos_weight_clamp))
        self.attention_backend = str(self.attention_backend).strip().lower()
        if self.attention_backend not in {"auto", "flash", "mem_efficient", "math"}:
            raise ValueError(
                "attention_backend must be one of: auto, flash, mem_efficient, math; "
                f"got {self.attention_backend!r}."
            )
        self.dali_resize_mode = str(self.dali_resize_mode).strip().lower()
        if self.dali_resize_mode not in {"video_resize", "video_then_resize", "none"}:
            raise ValueError(
                "dali_resize_mode must be one of: video_resize, video_then_resize, none; "
                f"got {self.dali_resize_mode!r}."
            )
        self.grad_clip = max(0.0, float(self.grad_clip))
        self.active_mouse_loss_mult = max(1.0, float(self.active_mouse_loss_mult))
        self.mouse_delta_loss_type = str(self.mouse_delta_loss_type).strip().lower()
        if self.mouse_delta_loss_type not in {"l1", "smooth_l1"}:
            raise ValueError(f"mouse_delta_loss_type must be 'l1' or 'smooth_l1', got {self.mouse_delta_loss_type!r}.")
        self.mouse_large_delta_loss_mult = max(0.0, float(self.mouse_large_delta_loss_mult))
        self.button_loss_weight = max(0.0, float(self.button_loss_weight))
        self.mouse_delta_loss_weight = max(0.0, float(self.mouse_delta_loss_weight))
        self.mouse_active_loss_weight = max(0.0, float(self.mouse_active_loss_weight))
        self.scroll_delta_loss_weight = max(0.0, float(self.scroll_delta_loss_weight))
        self.max_train_batches = None if self.max_train_batches is None else max(1, int(self.max_train_batches))
        self.max_val_batches = None if self.max_val_batches is None else max(1, int(self.max_val_batches))
        self.dali_num_threads = max(1, int(self.dali_num_threads))
        if not os.path.isabs(self.ckpt_dir):
            self.ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.ckpt_dir))
        os.makedirs(self.ckpt_dir, exist_ok=True)


@dataclass
class InverseWindowTargets:
    button_state: torch.Tensor
    mouse_active: torch.Tensor
    mouse_delta: torch.Tensor
    scroll_delta: torch.Tensor
    valid: torch.Tensor
    meta: Optional[List[Tuple[str, int, int]]] = None


def inverse_button_names(cfg: InverseDynamicsConfig) -> List[str]:
    return list(cfg.key_names) + list(cfg.binary_mouse_button_names)


def _parse_csv_float(value: object) -> float:
    if isinstance(value, str):
        value = value.strip()
    return float(value)


def load_run_arrays(csv_path: str, cfg: InverseDynamicsConfig) -> Dict[str, np.ndarray]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {csv_path}")
        fieldnames = list(reader.fieldnames)
        required = ["timestamp"] + list(cfg.key_names) + list(cfg.mouse_button_names) + ["delta_x", "delta_y"]
        missing = [name for name in required if name not in fieldnames]
        if missing:
            raise RuntimeError(f"CSV schema mismatch for {csv_path}: missing columns={missing}")
        rows.extend(reader)

    if not rows:
        raise RuntimeError(f"CSV has no rows: {csv_path}")

    num_frames = len(rows)
    buttons = np.zeros((num_frames, cfg.num_bin), dtype=np.float32)
    mouse_delta = np.zeros((num_frames, 2), dtype=np.float32)
    scroll_delta = np.zeros((num_frames, cfg.num_scroll), dtype=np.float32)

    for row_idx, row in enumerate(rows):
        col = 0
        for name in cfg.key_names:
            buttons[row_idx, col] = 1.0 if _parse_csv_float(row[name]) > 0.5 else 0.0
            col += 1
        for name in cfg.binary_mouse_button_names:
            buttons[row_idx, col] = 1.0 if _parse_csv_float(row[name]) > 0.5 else 0.0
            col += 1
        for scroll_idx, name in enumerate(cfg.scroll_action_names):
            scroll_delta[row_idx, scroll_idx] = max(0.0, _parse_csv_float(row[name]))
        mouse_delta[row_idx, 0] = _parse_csv_float(row["delta_x"])
        mouse_delta[row_idx, 1] = _parse_csv_float(row["delta_y"])

    return {"buttons": buttons, "mouse_delta": mouse_delta, "scroll_delta": scroll_delta}


def split_inverse_runs(
    pairs: Sequence[Tuple[str, str]],
    cfg: InverseTrainConfig,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    pairs = list(pairs)
    rng = random.Random(int(cfg.split_seed))
    rng.shuffle(pairs)
    if len(pairs) <= 1:
        return pairs, []
    split_idx = max(1, min(len(pairs) - 1, int(round(len(pairs) * float(cfg.train_split)))))
    return pairs[:split_idx], pairs[split_idx:]


def compute_delta_scales(
    pairs: Sequence[Tuple[str, str]],
    cfg: InverseTrainConfig,
) -> Tuple[Tuple[float, float], Tuple[float, ...]]:
    mouse_values = [[], []]
    scroll_values: List[List[float]] = [[] for _ in range(cfg.num_scroll)]
    for _, csv_path in pairs:
        run = load_run_arrays(csv_path, cfg)
        mouse = run["mouse_delta"]
        active = np.linalg.norm(mouse, axis=-1) >= float(cfg.mouse_active_epsilon)
        for axis in range(2):
            values = np.abs(mouse[active, axis])
            mouse_values[axis].extend(values[values > 0.0].tolist())
        scroll = run["scroll_delta"]
        for idx in range(cfg.num_scroll):
            values = scroll[:, idx]
            scroll_values[idx].extend(values[values > 0.0].tolist())

    mouse_scales: List[float] = []
    for values in mouse_values:
        scale = float(np.percentile(np.asarray(values, dtype=np.float32), cfg.scale_percentile)) if values else 1.0
        mouse_scales.append(max(scale, 1.0))

    scroll_scales: List[float] = []
    for values in scroll_values:
        scale = float(np.percentile(np.asarray(values, dtype=np.float32), cfg.scale_percentile)) if values else 1.0
        scroll_scales.append(max(scale, 1.0))
    return (mouse_scales[0], mouse_scales[1]), tuple(scroll_scales)


def build_inverse_window_targets(
    pairs: Sequence[Tuple[str, str]],
    cfg: InverseTrainConfig,
    *,
    stride: int,
    return_meta: bool = False,
) -> InverseWindowTargets:
    output_start, output_end = center_window_bounds(cfg.seq_len, cfg.output_seq_len)
    button_windows: List[np.ndarray] = []
    mouse_active_windows: List[np.ndarray] = []
    mouse_delta_windows: List[np.ndarray] = []
    scroll_delta_windows: List[np.ndarray] = []
    valid_windows: List[np.ndarray] = []
    meta: List[Tuple[str, int, int]] = []

    for video_path, csv_path in pairs:
        run = load_run_arrays(csv_path, cfg)
        buttons = run["buttons"]
        if buttons.shape[0] < cfg.seq_len:
            continue
        mouse_delta = run["mouse_delta"]
        scroll_delta = run["scroll_delta"]
        mouse_active = (np.linalg.norm(mouse_delta, axis=-1, keepdims=True) >= cfg.mouse_active_epsilon).astype(np.float32)
        valid = np.ones((cfg.output_seq_len,), dtype=np.float32)

        for start in range(0, buttons.shape[0] - cfg.seq_len + 1, max(1, int(stride))):
            end = start + cfg.seq_len
            sl = slice(start + output_start, start + output_end)
            button_windows.append(buttons[sl].astype(np.float32))
            mouse_active_windows.append(mouse_active[sl].astype(np.float32))
            mouse_delta_windows.append(mouse_delta[sl].astype(np.float32))
            scroll_delta_windows.append(scroll_delta[sl].astype(np.float32))
            valid_windows.append(valid)
            if return_meta:
                meta.append((video_path, start, end))

    if not button_windows:
        raise RuntimeError("No inverse-dynamics windows found. Check data_root, seq_len, and stride.")

    def stack(items: List[np.ndarray]) -> torch.Tensor:
        return torch.from_numpy(np.stack(items, axis=0)).float()

    return InverseWindowTargets(
        button_state=stack(button_windows),
        mouse_active=stack(mouse_active_windows),
        mouse_delta=stack(mouse_delta_windows),
        scroll_delta=stack(scroll_delta_windows),
        valid=stack(valid_windows),
        meta=meta if return_meta else None,
    )


def bundle_index(bundle: InverseWindowTargets, indices: torch.Tensor) -> InverseWindowTargets:
    return InverseWindowTargets(
        button_state=bundle.button_state[indices],
        mouse_active=bundle.mouse_active[indices],
        mouse_delta=bundle.mouse_delta[indices],
        scroll_delta=bundle.scroll_delta[indices],
        valid=bundle.valid[indices],
        meta=None,
    )


def move_bundle_to_device(bundle: InverseWindowTargets, device: torch.device) -> InverseWindowTargets:
    return InverseWindowTargets(
        button_state=bundle.button_state.to(device, non_blocking=True),
        mouse_active=bundle.mouse_active.to(device, non_blocking=True),
        mouse_delta=bundle.mouse_delta.to(device, non_blocking=True),
        scroll_delta=bundle.scroll_delta.to(device, non_blocking=True),
        valid=bundle.valid.to(device, non_blocking=True),
        meta=bundle.meta,
    )


def pin_bundle(bundle: InverseWindowTargets) -> InverseWindowTargets:
    return InverseWindowTargets(
        button_state=bundle.button_state.pin_memory(),
        mouse_active=bundle.mouse_active.pin_memory(),
        mouse_delta=bundle.mouse_delta.pin_memory(),
        scroll_delta=bundle.scroll_delta.pin_memory(),
        valid=bundle.valid.pin_memory(),
        meta=bundle.meta,
    )


def compute_inverse_losses(
    output,
    targets: InverseWindowTargets,
    cfg: InverseTrainConfig,
    *,
    button_pos_weight: torch.Tensor,
    mouse_active_pos_weight: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    valid = targets.valid.float()
    valid_3d = valid.unsqueeze(-1)

    button_loss_raw = F.binary_cross_entropy_with_logits(
        output.button_logits.float(),
        targets.button_state.float(),
        pos_weight=button_pos_weight.view(1, 1, -1).float(),
        reduction="none",
    )
    button_loss = (button_loss_raw * valid_3d).sum() / (valid_3d.sum() * cfg.num_bin).clamp(min=1.0)

    mouse_active_loss_raw = F.binary_cross_entropy_with_logits(
        output.mouse_active_logits.float(),
        targets.mouse_active.float(),
        pos_weight=mouse_active_pos_weight.view(1, 1, 1).float(),
        reduction="none",
    )
    mouse_active_loss = (mouse_active_loss_raw * valid_3d).sum() / valid_3d.sum().clamp(min=1.0)

    mouse_scale = torch.tensor(cfg.mouse_delta_scales, device=targets.mouse_delta.device, dtype=torch.float32).view(1, 1, 2)
    pred_mouse_norm = output.mouse_delta.float() / mouse_scale
    true_mouse_norm = targets.mouse_delta.float() / mouse_scale
    target_norm_mag = true_mouse_norm.norm(dim=-1, keepdim=True)
    mouse_weight = (
        1.0
        + ((float(cfg.active_mouse_loss_mult) - 1.0) * targets.mouse_active.float())
        + (float(cfg.mouse_large_delta_loss_mult) * target_norm_mag.clamp(max=1.0))
    )
    if cfg.mouse_delta_loss_type == "l1":
        mouse_loss_raw = (pred_mouse_norm - true_mouse_norm).abs().mean(dim=-1, keepdim=True)
    else:
        mouse_loss_raw = F.smooth_l1_loss(pred_mouse_norm, true_mouse_norm, reduction="none").mean(dim=-1, keepdim=True)
    mouse_loss = (mouse_loss_raw * mouse_weight * valid_3d).sum() / (mouse_weight * valid_3d).sum().clamp(min=1.0)

    if cfg.num_scroll > 0:
        scroll_scale = torch.tensor(
            cfg.scroll_delta_scales,
            device=targets.scroll_delta.device,
            dtype=torch.float32,
        ).view(1, 1, cfg.num_scroll)
        pred_scroll_norm = output.scroll_delta.float() / scroll_scale
        true_scroll_norm = targets.scroll_delta.float() / scroll_scale
        scroll_loss_raw = F.smooth_l1_loss(pred_scroll_norm, true_scroll_norm, reduction="none").mean(dim=-1)
        scroll_loss = (scroll_loss_raw * valid).sum() / valid.sum().clamp(min=1.0)
    else:
        scroll_loss = output.button_logits.float().sum() * 0.0

    total = (
        (cfg.button_loss_weight * button_loss)
        + (cfg.mouse_delta_loss_weight * mouse_loss)
        + (cfg.mouse_active_loss_weight * mouse_active_loss)
        + (cfg.scroll_delta_loss_weight * scroll_loss)
    )
    return total, {
        "button": button_loss.detach(),
        "mouse_delta": mouse_loss.detach(),
        "mouse_active": mouse_active_loss.detach(),
        "scroll_delta": scroll_loss.detach(),
    }


@torch.no_grad()
def update_metrics(
    output,
    targets: InverseWindowTargets,
    cfg: InverseTrainConfig,
    button_stats: BinaryStats,
    mouse_stats: MouseStats,
    scroll_error: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    pred_buttons = (torch.sigmoid(output.button_logits.float()) >= cfg.button_state_threshold).to(targets.button_state.dtype)
    button_stats.update(pred_buttons, targets.button_state, targets.valid)
    mouse_stats.update(output.mouse_delta.float(), targets.mouse_delta.float(), targets.valid, targets.mouse_active)
    if scroll_error is not None and cfg.num_scroll > 0:
        err = (output.scroll_delta.float() - targets.scroll_delta.float()).abs().mean(dim=-1)
        scroll_error["sum"] += (err * targets.valid).sum().detach()
        scroll_error["count"] += targets.valid.sum().detach()


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


def load_inverse_dali_batch(
    iterator,
    targets: InverseWindowTargets,
    device: torch.device,
    *,
    channels_last_inputs: bool = False,
    target_lookup: Optional[Dict[Tuple[int, int], int]] = None,
) -> Tuple[torch.Tensor, InverseWindowTargets]:
    batch = next(iterator)[0]
    frames = ensure_fchw_layout(batch["frames"])
    if channels_last_inputs:
        frames = maybe_channels_last_seq(frames)
    labels = normalize_dali_labels(batch["labels"]).detach().cpu()
    if target_lookup is None:
        target_indices = labels
    else:
        starts = normalize_dali_labels(batch["frame_num"]).detach().cpu()
        mapped = []
        for video_id, start in zip(labels.tolist(), starts.tolist()):
            key = (int(video_id), int(start))
            if key not in target_lookup:
                raise RuntimeError(f"DALI returned stream window with no target: video_id={key[0]} start={key[1]}.")
            mapped.append(target_lookup[key])
        target_indices = torch.tensor(mapped, dtype=torch.long)
    target_batch = move_bundle_to_device(bundle_index(targets, target_indices), device)
    return frames, target_batch


def build_video_stream_inputs(
    pairs: Sequence[Tuple[str, str]],
    cfg: InverseTrainConfig,
) -> Tuple[List[str], List[int], Dict[str, int]]:
    filenames: List[str] = []
    labels: List[int] = []
    video_to_id: Dict[str, int] = {}
    for video_id, (video_path, csv_path) in enumerate(pairs):
        run = load_run_arrays(csv_path, cfg)
        frame_count = int(run["buttons"].shape[0])
        if frame_count < cfg.seq_len:
            continue
        video_to_id[video_path] = int(video_id)
        filenames.append(video_path)
        labels.append(int(video_id))
    if not video_to_id:
        raise RuntimeError(f"No videos with at least {cfg.seq_len} frames found.")
    return filenames, labels, video_to_id


def build_target_lookup(
    targets: InverseWindowTargets,
    video_to_id: Dict[str, int],
) -> Dict[Tuple[int, int], int]:
    if targets.meta is None:
        raise RuntimeError("Window metadata is required to map DALI stream labels to target windows.")
    lookup: Dict[Tuple[int, int], int] = {}
    for target_idx, (video_path, start, _end) in enumerate(targets.meta):
        if video_path not in video_to_id:
            raise RuntimeError(f"Target video {video_path!r} was not registered with DALI inputs.")
        video_id = video_to_id[video_path]
        lookup[(int(video_id), int(start))] = int(target_idx)
    if not lookup:
        raise RuntimeError("Built an empty DALI stream target lookup.")
    return lookup


def make_dali_iterator(
    filenames: Sequence[str],
    labels: Sequence[int],
    cfg: InverseTrainConfig,
    *,
    batch_size: int,
    random_shuffle: bool,
    last_batch_policy,
    reader_step: Optional[int] = None,
    return_frame_num: bool = False,
):
    output_bytes_per_sample = int(cfg.seq_len) * int(cfg.model_size) * int(cfg.model_size) * 3
    pipe = video_pipeline(
        batch_size=batch_size,
        num_threads=int(cfg.dali_num_threads),
        device_id=0,
        seed=int(cfg.dali_shuffle_seed),
        filenames=list(filenames),
        labels=list(labels),
        seq_len=cfg.seq_len,
        resize_size=cfg.model_size,
        resize_mode=cfg.dali_resize_mode,
        reader_step=reader_step,
        enable_frame_num="scalar" if return_frame_num else "none",
        random_shuffle=random_shuffle,
        reader_prefetch_queue_depth=cfg.dali_reader_prefetch_queue_depth,
        read_ahead=cfg.dali_read_ahead,
        dont_use_mmap=cfg.dali_dont_use_mmap,
        normalize_frames=False,
        enable_augmentation=False,
        prefetch_queue_depth=cfg.dali_prefetch_queue_depth,
        exec_async=True,
        exec_pipelined=True,
        bytes_per_sample=output_bytes_per_sample,
    )
    pipe.build()
    return DALIGenericIterator(
        [pipe],
        output_map=["frames", "labels", "frame_num"] if return_frame_num else ["frames", "labels"],
        reader_name="Reader",
        auto_reset=False,
        last_batch_policy=last_batch_policy,
        prepare_first_batch=cfg.dali_prepare_first_batch,
    )


def latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    path = os.path.join(ckpt_dir, "model_latest.pt")
    if os.path.exists(path):
        return path
    return None


def checkpoint_mismatch_reason(state: Dict, cfg: InverseTrainConfig) -> Optional[str]:
    config = state["config"]
    family_reason = inverse_checkpoint_family_mismatch_reason(config)
    if family_reason:
        return family_reason
    if not isinstance(config, dict):
        return "Checkpoint config is not a dict."
    checks = {
        "selected_game": cfg.selected_game,
        "seq_len": cfg.seq_len,
        "output_seq_len": cfg.output_seq_len,
        "model_size": cfg.model_size,
        "num_bin": cfg.num_bin,
        "num_scroll": cfg.num_scroll,
        "d_model": cfg.d_model,
        "cnn_width": cfg.cnn_width,
        "cnn_depth": cfg.cnn_depth,
        "transformer_layers": cfg.transformer_layers,
        "transformer_heads": cfg.transformer_heads,
    }
    for key, expected in checks.items():
        if key not in config:
            return f"Checkpoint config mismatch for {key}: ckpt=None current={expected!r}."
        if config[key] != expected:
            return f"Checkpoint config mismatch for {key}: ckpt={config[key]!r} current={expected!r}."

    sequence_checks = {
        "key_names": list(cfg.key_names),
        "mouse_button_names": list(cfg.mouse_button_names),
        "binary_mouse_button_names": list(cfg.binary_mouse_button_names),
        "scroll_action_names": list(cfg.scroll_action_names),
    }
    for key, expected in sequence_checks.items():
        if key not in config:
            return f"Checkpoint config mismatch for {key}: ckpt=None current={expected!r}."
        actual = config[key]
        if list(actual) != expected:
            return f"Checkpoint config mismatch for {key}: ckpt={actual!r} current={expected!r}."

    scale_checks = {
        "mouse_delta_scales": tuple(float(value) for value in cfg.mouse_delta_scales),
        "scroll_delta_scales": tuple(float(value) for value in cfg.scroll_delta_scales),
    }
    for key, expected in scale_checks.items():
        if key not in config:
            return f"Checkpoint config mismatch for {key}: ckpt=None current={expected!r}."
        actual_raw = config[key]
        actual = tuple(float(value) for value in actual_raw)
        if len(actual) != len(expected) or any(abs(a - b) > 1e-6 for a, b in zip(actual, expected)):
            return f"Checkpoint config mismatch for {key}: ckpt={actual!r} current={expected!r}."
    return None


def save_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: InverseTrainConfig,
    epoch: int,
    global_step: int,
    best_val_f1: float,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(cfg),
            "visual_encoder_name": INVERSE_VISUAL_ENCODER_NAME,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "best_val_f1": float(best_val_f1),
        },
        path,
    )


def maybe_resume(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: InverseTrainConfig,
    device: torch.device,
) -> Tuple[int, int, float]:
    if not cfg.resume:
        return 0, 0, -1.0
    if cfg.resume_path is not None:
        path = cfg.resume_path
    else:
        path = latest_checkpoint(cfg.ckpt_dir)
    if not path:
        return 0, 0, -1.0
    state = torch.load(path, map_location=device)
    reason = checkpoint_mismatch_reason(state, cfg)
    if reason:
        raise RuntimeError(f"Refusing to resume incompatible checkpoint {path}: {reason}")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    print(f"Resumed inverse checkpoint: {path}")
    return int(state["epoch"]), int(state["global_step"]), float(state["best_val_f1"])


def run_epoch(
    *,
    desc: str,
    model: torch.nn.Module,
    iterator,
    batches: int,
    targets: InverseWindowTargets,
    cfg: InverseTrainConfig,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_autocast: bool,
    button_pos_weight: torch.Tensor,
    mouse_active_pos_weight: torch.Tensor,
    target_lookup: Optional[Dict[Tuple[int, int], int]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    total_steps: int = 1,
    global_step: int = 0,
) -> Tuple[Dict[str, float], int]:
    train_mode = optimizer is not None
    model.train(train_mode)
    button_stats = BinaryStats(cfg.num_bin, device)
    mouse_stats = MouseStats(device)
    scroll_error = {
        "sum": torch.zeros((), dtype=torch.float64, device=device),
        "count": torch.zeros((), dtype=torch.float64, device=device),
    }

    loss_sums = {
        "loss": torch.zeros((), dtype=torch.float64, device=device),
        "button": torch.zeros((), dtype=torch.float64, device=device),
        "mouse_delta": torch.zeros((), dtype=torch.float64, device=device),
        "mouse_active": torch.zeros((), dtype=torch.float64, device=device),
        "scroll_delta": torch.zeros((), dtype=torch.float64, device=device),
    }
    steps = 0
    if train_mode:
        optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(total=batches, desc=desc)
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch_idx in range(batches):
            frames, target_batch = load_inverse_dali_batch(iterator, targets, device, target_lookup=target_lookup)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_autocast):
                output = model(frames)
                loss, details = compute_inverse_losses(
                    output,
                    target_batch,
                    cfg,
                    button_pos_weight=button_pos_weight,
                    mouse_active_pos_weight=mouse_active_pos_weight,
                )
            if train_mode:
                (loss / cfg.grad_accum).backward()
                if ((batch_idx + 1) % cfg.grad_accum == 0) or (batch_idx + 1 == batches):
                    if cfg.grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
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

            update_metrics(output, target_batch, cfg, button_stats, mouse_stats, scroll_error)
            loss_sums["loss"] += loss.detach().double()
            for key, value in details.items():
                loss_sums[key] += value.double()
            steps += 1

            if ((batch_idx + 1) % cfg.print_every == 0) or (batch_idx + 1 == batches):
                pbar.set_postfix(
                    {
                        "loss": float((loss_sums["loss"] / max(1, steps)).item()),
                        "btn": float((loss_sums["button"] / max(1, steps)).item()),
                        "mouse": float((loss_sums["mouse_delta"] / max(1, steps)).item()),
                    }
                )
            pbar.update(1)
    pbar.close()
    iterator.reset()

    button_metrics = button_stats.compute()
    mouse_metrics = mouse_stats.compute()
    metrics = {
        "loss": float((loss_sums["loss"] / max(1, steps)).item()),
        "button_loss": float((loss_sums["button"] / max(1, steps)).item()),
        "mouse_delta_loss": float((loss_sums["mouse_delta"] / max(1, steps)).item()),
        "mouse_active_loss": float((loss_sums["mouse_active"] / max(1, steps)).item()),
        "scroll_delta_loss": float((loss_sums["scroll_delta"] / max(1, steps)).item()),
        "button_macro_f1": button_metrics["macro_f1"],
        "button_macro_precision": button_metrics["macro_precision"],
        "button_macro_recall": button_metrics["macro_recall"],
        "mouse_mae": mouse_metrics["mae"],
        "mouse_active_mae": mouse_metrics["active_mae"],
        "scroll_mae": float((scroll_error["sum"] / scroll_error["count"].clamp(min=1.0)).item()),
        "per_class_summary": per_class_f1_summary(button_stats, inverse_button_names(cfg)),
        "button_rows": binary_stats_rows(button_stats, inverse_button_names(cfg)),
    }
    return metrics, global_step


def train(cfg: Optional[InverseTrainConfig] = None) -> None:
    cfg = cfg if cfg is not None else InverseTrainConfig()

    if not torch.cuda.is_available():
        raise RuntimeError("Inverse-dynamics training requires CUDA.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    configure_attention_backend(cfg.attention_backend, device=device)

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
        raise RuntimeError(f"No video/csv pairs found under {cfg.data_root}")
    train_pairs, val_pairs = split_inverse_runs(pairs, cfg)
    print(f"Inverse runs: total={len(pairs)} train={len(train_pairs)} val={len(val_pairs)}")

    mouse_scales, scroll_scales = compute_delta_scales(train_pairs, cfg)
    cfg.mouse_delta_scales = mouse_scales
    cfg.scroll_delta_scales = scroll_scales
    cfg.__post_init__()
    print(f"Mouse delta scales p{cfg.scale_percentile:.1f}: x={mouse_scales[0]:.3f} y={mouse_scales[1]:.3f}")
    if cfg.num_scroll > 0:
        print(
            "Scroll delta scales:",
            ", ".join(f"{name}={scale:.3f}" for name, scale in zip(cfg.scroll_action_names, scroll_scales)),
        )

    train_targets = build_inverse_window_targets(train_pairs, cfg, stride=cfg.train_seq_stride, return_meta=True)
    val_targets = (
        build_inverse_window_targets(val_pairs, cfg, stride=cfg.val_seq_stride, return_meta=True) if val_pairs else None
    )
    print(
        "Inverse windows:",
        f"train={tuple(train_targets.button_state.shape)}",
        f"val={(tuple(val_targets.button_state.shape) if val_targets is not None else None)}",
    )

    button_names = inverse_button_names(cfg)
    support = train_targets.button_state.sum(dim=(0, 1))
    zero_support = [button_names[idx] for idx, value in enumerate(support.tolist()) if value <= 0.0]
    if zero_support:
        print("Zero-support train actions:", zero_support)

    train_targets = pin_bundle(train_targets)
    if val_targets is not None:
        val_targets = pin_bundle(val_targets)

    button_pos_weight = compute_pos_weight(
        train_targets.button_state,
        cfg.pos_weight_power,
        cfg.pos_weight_clamp,
    ).to(device)
    mouse_active_pos_weight = compute_pos_weight(
        train_targets.mouse_active,
        cfg.pos_weight_power,
        cfg.pos_weight_clamp,
    ).to(device)
    print(
        "Class weighting:",
        f"pos_weight_power={cfg.pos_weight_power:.2f}",
        f"pos_weight_clamp={cfg.pos_weight_clamp:.1f}",
        f"button_pos_weight_mean={float(button_pos_weight.mean().item()):.3f}",
        f"mouse_active_pos_weight={float(mouse_active_pos_weight.mean().item()):.3f}",
    )
    print(
        "Mouse loss:",
        f"type={cfg.mouse_delta_loss_type}",
        f"delta_weight={cfg.mouse_delta_loss_weight:.3f}",
        f"active_mult={cfg.active_mouse_loss_mult:.3f}",
        f"large_delta_mult={cfg.mouse_large_delta_loss_mult:.3f}",
        f"active_head_weight={cfg.mouse_active_loss_weight:.3f}",
    )
    print(
        "DALI stream:",
        f"seq_len={cfg.seq_len}",
        f"output_seq_len={cfg.output_seq_len}",
        f"train_step={cfg.train_seq_stride}",
        f"train_shuffle={cfg.dali_train_random_shuffle}",
    )
    train_targets_gpu = move_bundle_to_device(train_targets, device)
    val_targets_gpu = move_bundle_to_device(val_targets, device) if val_targets is not None else None

    train_batches = int(train_targets.button_state.shape[0]) // cfg.batch_size
    train_filenames, train_labels, train_video_to_id = build_video_stream_inputs(train_pairs, cfg)
    train_target_lookup = build_target_lookup(train_targets, train_video_to_id)
    train_iter = make_dali_iterator(
        train_filenames,
        train_labels,
        cfg,
        batch_size=cfg.batch_size,
        random_shuffle=cfg.dali_train_random_shuffle,
        last_batch_policy=LastBatchPolicy.DROP,
        reader_step=cfg.train_seq_stride,
        return_frame_num=True,
    )
    if cfg.max_train_batches is not None:
        train_batches = min(train_batches, cfg.max_train_batches)
    if train_batches <= 0:
        raise RuntimeError("No inverse-dynamics training batches available.")

    val_batches = 0
    val_iter = None
    val_target_lookup = None
    if val_targets is not None:
        val_batch_size = min(max(cfg.batch_size, 1), int(val_targets.button_state.shape[0]))
        val_batches = int(math.ceil(int(val_targets.button_state.shape[0]) / float(val_batch_size)))
        val_filenames, val_labels, val_video_to_id = build_video_stream_inputs(val_pairs, cfg)
        val_target_lookup = build_target_lookup(val_targets, val_video_to_id)
        val_iter = make_dali_iterator(
            val_filenames,
            val_labels,
            cfg,
            batch_size=val_batch_size,
            random_shuffle=cfg.dali_val_random_shuffle,
            last_batch_policy=LastBatchPolicy.PARTIAL,
            reader_step=cfg.val_seq_stride,
            return_frame_num=True,
        )
        if cfg.max_val_batches is not None:
            val_batches = min(val_batches, cfg.max_val_batches)

    amp_dtype, use_autocast, use_scaler = resolve_amp_settings(cfg.amp_dtype)
    if use_scaler:
        raise RuntimeError("This simplified inverse trainer supports bf16/fp32 only; use --amp-dtype bf16 or fp32.")
    print(f"AMP: dtype={amp_dtype} autocast={use_autocast}")

    base_model: torch.nn.Module = InverseDynamicsModel(cfg).to(device)
    initialize_inverse_lazy_layers(base_model, cfg, device)
    base_model = base_model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, fused=True)
    print(f"Total parameters: {sum(p.numel() for p in base_model.parameters()) / 1e6:.4f}M")

    start_epoch, global_step, best_val_f1 = maybe_resume(base_model, optimizer, cfg, device)

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

    for epoch in range(start_epoch, cfg.num_epochs):
        train_metrics, global_step = run_epoch(
            desc=f"Epoch {epoch + 1}/{cfg.num_epochs} [train]",
            model=model,
            iterator=train_iter,
            batches=train_batches,
            targets=train_targets_gpu,
            cfg=cfg,
            device=device,
            amp_dtype=amp_dtype,
            use_autocast=use_autocast,
            button_pos_weight=button_pos_weight,
            mouse_active_pos_weight=mouse_active_pos_weight,
            target_lookup=train_target_lookup,
            optimizer=optimizer,
            total_steps=total_steps,
            global_step=global_step,
        )

        val_metrics = None
        if val_iter is not None and val_targets_gpu is not None and val_batches > 0:
            val_metrics, _ = run_epoch(
                desc=f"Epoch {epoch + 1}/{cfg.num_epochs} [val]",
                model=model,
                iterator=val_iter,
                batches=val_batches,
                targets=val_targets_gpu,
                cfg=cfg,
                device=device,
                amp_dtype=amp_dtype,
                use_autocast=use_autocast,
                button_pos_weight=button_pos_weight,
                mouse_active_pos_weight=mouse_active_pos_weight,
                target_lookup=val_target_lookup,
            )
            if val_metrics["button_macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["button_macro_f1"]
                save_checkpoint(
                    os.path.join(cfg.ckpt_dir, "model_best.pt"),
                    model=base_model,
                    optimizer=optimizer,
                    cfg=cfg,
                    epoch=epoch + 1,
                    global_step=global_step,
                    best_val_f1=best_val_f1,
                )
        elif train_metrics["button_macro_f1"] > best_val_f1:
            best_val_f1 = train_metrics["button_macro_f1"]
            save_checkpoint(
                os.path.join(cfg.ckpt_dir, "model_best.pt"),
                model=base_model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=epoch + 1,
                global_step=global_step,
                best_val_f1=best_val_f1,
            )

        save_checkpoint(
            os.path.join(cfg.ckpt_dir, "model_latest.pt"),
            model=base_model,
            optimizer=optimizer,
            cfg=cfg,
            epoch=epoch + 1,
            global_step=global_step,
            best_val_f1=best_val_f1,
        )
        if (epoch + 1) % cfg.save_every == 0:
            save_checkpoint(
                os.path.join(cfg.ckpt_dir, f"model_epoch_{epoch + 1}.pt"),
                model=base_model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=epoch + 1,
                global_step=global_step,
                best_val_f1=best_val_f1,
            )

        parts = [
            f"Epoch {epoch + 1}/{cfg.num_epochs}",
            f"tr_loss={train_metrics['loss']:.4f}",
            f"tr_f1={train_metrics['button_macro_f1']:.4f}",
            f"tr_mouse_mae={train_metrics['mouse_mae']:.3f}",
            f"tr_mouse_active_mae={train_metrics['mouse_active_mae']:.3f}",
        ]
        if cfg.num_scroll > 0:
            parts.append(f"tr_scroll_mae={train_metrics['scroll_mae']:.3f}")
        if val_metrics is not None:
            parts.extend(
                [
                    f"va_loss={val_metrics['loss']:.4f}",
                    f"va_f1={val_metrics['button_macro_f1']:.4f}",
                    f"va_prec={val_metrics['button_macro_precision']:.4f}",
                    f"va_rec={val_metrics['button_macro_recall']:.4f}",
                    f"va_mouse_mae={val_metrics['mouse_mae']:.3f}",
                    f"best_f1={best_val_f1:.4f}",
                ]
            )
            if val_metrics["per_class_summary"]:
                parts.append(val_metrics["per_class_summary"])
        else:
            parts.append(train_metrics["per_class_summary"])
        print(" | ".join(part for part in parts if part))
        print(
            f"Epoch {epoch + 1} inverse train stats: "
            f"f1={train_metrics['button_macro_f1']:.4f} "
            f"prec={train_metrics['button_macro_precision']:.4f} "
            f"rec={train_metrics['button_macro_recall']:.4f} "
            f"mouse_mae={train_metrics['mouse_mae']:.3f} "
            f"active_mouse_mae={train_metrics['mouse_active_mae']:.3f} "
            f"scroll_mae={train_metrics['scroll_mae']:.3f}"
        )
        print_button_stats_table(
            f"Epoch {epoch + 1} inverse train per-key/button:",
            train_metrics["button_rows"],
        )
        if val_metrics is not None:
            print(
                f"Epoch {epoch + 1} inverse val stats: "
                f"f1={val_metrics['button_macro_f1']:.4f} "
                f"prec={val_metrics['button_macro_precision']:.4f} "
                f"rec={val_metrics['button_macro_recall']:.4f} "
                f"mouse_mae={val_metrics['mouse_mae']:.3f} "
                f"active_mouse_mae={val_metrics['mouse_active_mae']:.3f} "
                f"scroll_mae={val_metrics['scroll_mae']:.3f}"
            )
            print_button_stats_table(
                f"Epoch {epoch + 1} inverse val per-key/button:",
                val_metrics["button_rows"],
            )


def parse_args() -> InverseTrainConfig:
    parser = argparse.ArgumentParser(description="Train the simplified inverse-dynamics model.")
    add = parser.add_argument
    add("--data-root", default=None)
    add("--ckpt-dir", default=None)
    add("--resume", dest="resume", action="store_true", default=True)
    add("--no-resume", dest="resume", action="store_false")
    add("--resume-path", default=None)
    add("--amp-dtype", default=None, choices=["bf16", "fp32", "float32"])
    add("--compile", dest="compile_model", action="store_true")
    add("--no-compile", dest="compile_model", action="store_false")
    add("--compile-mode", default=None)
    add("--attention-backend", default=None, choices=["auto", "flash", "mem_efficient", "math"])
    add("--dali-resize-mode", default=None, choices=["video_resize", "video_then_resize", "none"])
    parser.set_defaults(compile_model=None)

    for flag in (
        "model-size",
        "seq-len",
        "output-seq-len",
        "train-seq-stride",
        "val-seq-stride",
        "batch-size",
        "target-effective-batch",
        "epochs",
        "warmup-steps",
        "save-every",
        "print-every",
        "max-train-batches",
        "max-val-batches",
        "d-model",
        "cnn-width",
        "cnn-depth",
        "transformer-layers",
        "transformer-heads",
        "dali-num-threads",
        "dali-prefetch-queue-depth",
        "dali-reader-prefetch-queue-depth",
    ):
        add(f"--{flag}", type=int, default=None)
    for flag in (
        "lr",
        "min-lr",
        "weight-decay",
        "grad-clip",
        "dropout",
        "train-split",
        "pos-weight-power",
        "pos-weight-clamp",
        "scale-percentile",
        "button-loss-weight",
        "mouse-delta-loss-weight",
        "mouse-active-loss-weight",
        "scroll-delta-loss-weight",
        "active-mouse-loss-mult",
        "mouse-large-delta-loss-mult",
        "mouse-active-epsilon",
    ):
        add(f"--{flag}", type=float, default=None)
    add("--mouse-delta-loss-type", default=None, choices=["l1", "smooth_l1"])
    add("--dali-train-random-shuffle", dest="dali_train_random_shuffle", action="store_true")
    add("--no-dali-train-random-shuffle", dest="dali_train_random_shuffle", action="store_false")
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
        dali_train_random_shuffle=None,
        dali_read_ahead=None,
        dali_dont_use_mmap=None,
        sync_dataset=None,
        dataset_sync_delete_stale=None,
        dataset_sync_hash_same_size=None,
    )
    args = parser.parse_args()

    cfg = InverseTrainConfig()
    direct_names = {
        "model_size",
        "seq_len",
        "output_seq_len",
        "train_seq_stride",
        "val_seq_stride",
        "batch_size",
        "target_effective_batch",
        "warmup_steps",
        "save_every",
        "print_every",
        "max_train_batches",
        "max_val_batches",
        "d_model",
        "cnn_width",
        "cnn_depth",
        "transformer_layers",
        "transformer_heads",
        "dali_num_threads",
        "dali_prefetch_queue_depth",
        "dali_reader_prefetch_queue_depth",
        "lr",
        "min_lr",
        "weight_decay",
        "grad_clip",
        "dropout",
        "train_split",
        "pos_weight_power",
        "pos_weight_clamp",
        "scale_percentile",
        "button_loss_weight",
        "mouse_delta_loss_weight",
        "mouse_active_loss_weight",
        "scroll_delta_loss_weight",
        "active_mouse_loss_mult",
        "mouse_large_delta_loss_mult",
        "mouse_active_epsilon",
    }
    args_by_name = vars(args)
    for name in direct_names:
        value = args_by_name[name]
        if value is not None:
            setattr(cfg, name, value)
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.data_root:
        cfg.data_root = os.path.abspath(args.data_root)
    if args.dataset_cache_root:
        cfg.dataset_cache_root = args.dataset_cache_root
    if args.sync_dataset is not None:
        cfg.sync_dataset = bool(args.sync_dataset)
    if args.dataset_sync_delete_stale is not None:
        cfg.dataset_sync_delete_stale = bool(args.dataset_sync_delete_stale)
    if args.dataset_sync_hash_same_size is not None:
        cfg.dataset_sync_hash_same_size = bool(args.dataset_sync_hash_same_size)
    if args.ckpt_dir:
        cfg.ckpt_dir = os.path.abspath(args.ckpt_dir)
    if args.resume_path:
        cfg.resume_path = args.resume_path
    cfg.resume = bool(args.resume)
    if args.amp_dtype is not None:
        cfg.amp_dtype = "fp32" if args.amp_dtype == "float32" else args.amp_dtype
    if args.compile_model is not None:
        cfg.compile_model = bool(args.compile_model)
    if args.compile_mode is not None:
        cfg.compile_mode = str(args.compile_mode)
    if args.attention_backend is not None:
        cfg.attention_backend = str(args.attention_backend)
    if args.dali_resize_mode is not None:
        cfg.dali_resize_mode = str(args.dali_resize_mode)
    if args.mouse_delta_loss_type is not None:
        cfg.mouse_delta_loss_type = str(args.mouse_delta_loss_type)
    if args.dali_train_random_shuffle is not None:
        cfg.dali_train_random_shuffle = bool(args.dali_train_random_shuffle)
    if args.dali_read_ahead is not None:
        cfg.dali_read_ahead = bool(args.dali_read_ahead)
    if args.dali_dont_use_mmap is not None:
        cfg.dali_dont_use_mmap = bool(args.dali_dont_use_mmap)
    cfg.__post_init__()
    return cfg


if __name__ == "__main__":
    train(parse_args())
