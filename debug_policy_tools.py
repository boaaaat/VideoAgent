import argparse
import csv
import glob
import math
import os
import re
from dataclasses import fields
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from models import ARCHITECTURE_VERSION, DrivingVideoPolicy, ModelConfig, TemporalState


CONFLICTING_BUTTON_PAIRS = (("w", "s"), ("a", "d"))


def _try_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:
        print(f"Matplotlib is unavailable ({type(exc).__name__}: {exc}); using OpenCV plot fallback.")
        return None


CV_COLORS = [
    (168, 120, 76),
    (178, 183, 114),
    (24, 133, 245),
    (86, 87, 228),
    (75, 162, 84),
    (127, 92, 114),
]


def _cv_text(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    scale: float = 0.55,
    color: Tuple[int, int, int] = (35, 35, 35),
    thickness: int = 1,
) -> None:
    cv2.putText(image, str(text), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _short_label(text: object, limit: int = 18) -> str:
    value = str(text)
    return value if len(value) <= limit else value[: max(1, limit - 1)] + "~"


def _save_cv(path: str, image: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not cv2.imwrite(path, image):
        raise RuntimeError(f"Failed to write plot image: {path}")


def _cv_bar_chart(
    image: np.ndarray,
    rect: Tuple[int, int, int, int],
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    *,
    colors: Optional[Sequence[Tuple[int, int, int]]] = None,
) -> None:
    x0, y0, width, height = rect
    _cv_text(image, title, x0, y0 - 12, scale=0.7, thickness=2)
    values_np = np.asarray(values, dtype=np.float32)
    max_value = float(np.max(values_np)) if values_np.size else 1.0
    max_value = max(max_value, 1e-6)
    axis_left = x0 + 150
    axis_right = x0 + width - 20
    axis_top = y0 + 10
    row_h = max(22, int((height - 20) / max(1, len(values))))
    cv2.line(image, (axis_left, axis_top), (axis_left, y0 + height), (150, 150, 150), 1)
    for idx, (label, value) in enumerate(zip(labels, values)):
        y = axis_top + idx * row_h + 4
        bar_w = int((axis_right - axis_left) * max(0.0, float(value)) / max_value)
        color = (colors or CV_COLORS)[idx % len(colors or CV_COLORS)]
        _cv_text(image, _short_label(label, 20), x0, y + 15, scale=0.5)
        cv2.rectangle(image, (axis_left, y), (axis_left + bar_w, y + row_h - 7), color, -1)
        _cv_text(image, f"{float(value):.4f}", axis_left + bar_w + 8, y + 15, scale=0.48)


def _cv_line_chart(
    image: np.ndarray,
    rect: Tuple[int, int, int, int],
    series: Dict[str, Sequence[float]],
    title: str,
    *,
    y_min: float = 0.0,
    y_max: float = 1.0,
    x_labels: Optional[Sequence[str]] = None,
) -> None:
    x0, y0, width, height = rect
    _cv_text(image, title, x0, y0 - 10, scale=0.62, thickness=2)
    left = x0 + 40
    right = x0 + width - 15
    top = y0 + 8
    bottom = y0 + height - 35
    cv2.rectangle(image, (left, top), (right, bottom), (180, 180, 180), 1)
    _cv_text(image, f"{y_max:.2f}", x0, top + 4, scale=0.38)
    _cv_text(image, f"{y_min:.2f}", x0, bottom, scale=0.38)

    def point(index: int, count: int, value: float) -> Tuple[int, int]:
        px = int((left + right) / 2) if count <= 1 else int(left + (right - left) * index / (count - 1))
        frac = (float(value) - y_min) / max(y_max - y_min, 1e-8)
        py = int(bottom - np.clip(frac, 0.0, 1.0) * (bottom - top))
        return px, py

    legend_x = left + 8
    for series_idx, (name, values) in enumerate(series.items()):
        color = CV_COLORS[series_idx % len(CV_COLORS)]
        values_list = [float(v) for v in values]
        pts = [point(i, len(values_list), v) for i, v in enumerate(values_list)]
        for p0, p1 in zip(pts, pts[1:]):
            cv2.line(image, p0, p1, color, 2)
        for pt in pts:
            cv2.circle(image, pt, 3, color, -1)
        cv2.rectangle(image, (legend_x, bottom + 9 + 15 * series_idx), (legend_x + 10, bottom + 19 + 15 * series_idx), color, -1)
        _cv_text(image, name, legend_x + 15, bottom + 19 + 15 * series_idx, scale=0.4)
    if x_labels:
        for idx, label in enumerate(x_labels):
            if idx % max(1, int(math.ceil(len(x_labels) / 8))) == 0:
                px, _ = point(idx, len(x_labels), y_min)
                _cv_text(image, _short_label(label, 10), px - 20, bottom + 32, scale=0.35)


def _cv_histograms(path: str, label: str, probs: np.ndarray, names: Sequence[str]) -> None:
    cols = min(4, len(names))
    rows = int(math.ceil(len(names) / float(cols)))
    panel_w, panel_h = 360, 260
    image = np.full((rows * panel_h + 70, cols * panel_w + 30, 3), 255, dtype=np.uint8)
    _cv_text(image, f"Probability Histograms: {label}", 18, 32, scale=0.8, thickness=2)
    bins = np.linspace(0.0, 1.0, 41)
    for idx, name in enumerate(names):
        row, col = divmod(idx, cols)
        x0 = 20 + col * panel_w
        y0 = 60 + row * panel_h
        left, top, right, bottom = x0 + 40, y0 + 35, x0 + panel_w - 18, y0 + panel_h - 42
        counts, _ = np.histogram(probs[:, idx], bins=bins)
        max_count = max(1, int(counts.max()))
        _cv_text(image, str(name), x0 + 8, y0 + 22, scale=0.6, thickness=2)
        cv2.rectangle(image, (left, top), (right, bottom), (180, 180, 180), 1)
        bin_w = max(1, int((right - left) / len(counts)))
        for bin_idx, count in enumerate(counts):
            bar_h = int((bottom - top) * int(count) / max_count)
            x = left + bin_idx * bin_w
            cv2.rectangle(image, (x, bottom - bar_h), (x + bin_w - 1, bottom), CV_COLORS[0], -1)
        _cv_text(image, "0", left - 4, bottom + 18, scale=0.4)
        _cv_text(image, "1", right - 8, bottom + 18, scale=0.4)
    _save_cv(path, image)


def sigmoid_np(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    return 1.0 / (1.0 + np.exp(-values))


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


def _checkpoint_label(path: str) -> str:
    name = os.path.basename(path)
    match = re.search(r"model_epoch_(\d+)\.pt$", name)
    if match:
        return f"epoch_{int(match.group(1)):04d}"
    return os.path.splitext(name)[0]


def _checkpoint_sort_key(path: str) -> Tuple[int, int, str]:
    name = os.path.basename(path)
    match = re.search(r"model_epoch_(\d+)\.pt$", name)
    if match:
        return (0, int(match.group(1)), name)
    if name == "model_best.pt":
        return (1, 0, name)
    if name == "model_latest.pt":
        return (2, 0, name)
    return (3, 0, name)


def discover_checkpoints(ckpt_dir: str, ckpt_path: str, checkpoint_glob: Optional[str]) -> List[str]:
    if not checkpoint_glob:
        return [ckpt_path]
    pattern = checkpoint_glob
    if not os.path.isabs(pattern):
        pattern = os.path.join(ckpt_dir, pattern)
    paths = sorted(set(glob.glob(pattern)), key=_checkpoint_sort_key)
    if not paths:
        raise FileNotFoundError(f"No checkpoints matched {pattern!r}.")
    return paths


def load_model_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    *,
    use_checkpoint_thresholds: bool = True,
) -> Tuple[DrivingVideoPolicy, ModelConfig, Dict[str, object]]:
    state = torch.load(checkpoint_path, map_location=device)
    if not isinstance(state, dict) or not isinstance(state.get("config"), dict):
        raise RuntimeError(f"Checkpoint {checkpoint_path!r} must contain a config dict.")
    if not isinstance(state.get("model_state"), dict):
        raise RuntimeError(f"Checkpoint {checkpoint_path!r} must contain a model_state dict.")

    config_dict = dict(state["config"])
    if str(config_dict.get("architecture_version", "")).strip() != ARCHITECTURE_VERSION:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path!r} is incompatible with {ARCHITECTURE_VERSION!r}; retrain it "
            "with previous-action conditioning."
        )
    valid_keys = {field.name for field in fields(ModelConfig)}
    cfg_kwargs = {key: value for key, value in config_dict.items() if key in valid_keys}
    if not use_checkpoint_thresholds:
        cfg_kwargs.pop("button_state_thresholds", None)
    cfg = ModelConfig(**cfg_kwargs)

    model = DrivingVideoPolicy(cfg).to(device)
    load_result = model.load_state_dict(state["model_state"], strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path!r} does not match the current policy model. "
            f"missing={load_result.missing_keys} unexpected={load_result.unexpected_keys}"
        )
    model.eval()
    return model, cfg, config_dict


def find_run_pairs(data_root: str, video_ext: str, csv_ext: str, max_runs: Optional[int]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for video_path in sorted(glob.glob(os.path.join(data_root, f"*{video_ext}"))):
        if video_path.endswith(f"_keys{video_ext}"):
            continue
        base, _ = os.path.splitext(video_path)
        csv_path = base + csv_ext
        if os.path.exists(csv_path):
            pairs.append((video_path, csv_path))
    if max_runs is not None and int(max_runs) > 0:
        pairs = pairs[: int(max_runs)]
    if not pairs:
        raise FileNotFoundError(f"No video/CSV pairs found under {data_root!r}.")
    return pairs


def load_buttons(csv_path: str, cfg: ModelConfig) -> np.ndarray:
    names = list(cfg.key_names) + list(cfg.mouse_button_names)
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        fieldnames = set(reader.fieldnames or [])
        missing = [name for name in names if name not in fieldnames]
        if missing:
            raise RuntimeError(f"CSV schema mismatch for {csv_path}: missing action columns={missing}")
        rows.extend(reader)

    buttons = np.zeros((len(rows), int(cfg.num_bin)), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        for col_idx, name in enumerate(names):
            buttons[row_idx, col_idx] = 1.0 if _parse_float(row.get(name)) > 0.5 else 0.0
    return buttons


def compute_pos_weight(
    button_sets: Sequence[np.ndarray],
    power: float,
    clamp: float,
) -> torch.Tensor:
    total = 0.0
    pos: Optional[np.ndarray] = None
    for buttons in button_sets:
        if pos is None:
            pos = np.zeros((buttons.shape[1],), dtype=np.float64)
        pos += buttons.astype(np.float64).sum(axis=0)
        total += float(buttons.shape[0])
    if pos is None:
        raise ValueError("No buttons were provided for pos_weight computation.")
    neg = np.maximum(total - pos, 0.0)
    weights = np.power(neg / np.maximum(pos, 1.0), float(power))
    weights = np.clip(weights, 1.0, float(clamp))
    return torch.from_numpy(weights.astype(np.float32))


def resolve_thresholds(cfg: ModelConfig, threshold: Optional[float]) -> np.ndarray:
    if threshold is not None:
        return np.full((int(cfg.num_bin),), float(threshold), dtype=np.float32)
    values = getattr(cfg, "button_state_thresholds", None)
    if values is None or len(tuple(values)) != int(cfg.num_bin):
        return np.full((int(cfg.num_bin),), float(cfg.button_state_threshold), dtype=np.float32)
    return np.asarray(tuple(float(x) for x in values), dtype=np.float32)


def make_targets(
    buttons: np.ndarray,
    source_indices: np.ndarray,
    target_offset: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_indices = source_indices.astype(np.int64) + int(target_offset)
    previous_indices = target_indices - 1
    valid = (
        (target_indices >= 0)
        & (target_indices < buttons.shape[0])
        & (previous_indices >= 0)
        & (previous_indices < buttons.shape[0])
    )
    labels = np.zeros((source_indices.shape[0], buttons.shape[1]), dtype=np.float32)
    transitions = np.zeros_like(labels)
    if bool(valid.any()):
        labels[valid] = buttons[target_indices[valid]]
        transitions[valid] = (buttons[target_indices[valid]] != buttons[previous_indices[valid]]).astype(np.float32)
    return labels, transitions, valid


def infer_run(
    video_path: str,
    model: DrivingVideoPolicy,
    cfg: ModelConfig,
    device: torch.device,
    *,
    max_frames: Optional[int],
    frame_stride: int,
    reset_mode: str,
    reset_every: int,
    thresholds: np.ndarray,
    use_autocast: bool,
    inference_dtype: torch.dtype,
) -> Tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames is not None and int(max_frames) > 0:
        total_frames = min(total_frames, int(max_frames)) if total_frames > 0 else int(max_frames)
    frame_stride = max(1, int(frame_stride))
    reset_every = max(1, int(reset_every))
    threshold_tensor = torch.tensor(thresholds, device=device, dtype=torch.float32).view(1, -1)

    logits: List[np.ndarray] = []
    source_indices: List[int] = []
    state = TemporalState()
    prev_action = torch.zeros((1, int(cfg.num_bin)), device=device, dtype=inference_dtype)

    try:
        pbar = tqdm(total=total_frames if total_frames > 0 else None, desc=os.path.basename(video_path), unit="frame")
        source_idx = 0
        while True:
            if total_frames > 0 and source_idx >= total_frames:
                break
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if source_idx % frame_stride == 0:
                if reset_mode == "frame" or (reset_mode == "clip" and source_idx % reset_every == 0):
                    state = TemporalState()
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (int(cfg.model_size), int(cfg.model_size)), interpolation=cv2.INTER_AREA)
                frame = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
                frame = frame.to(device=device, dtype=inference_dtype, non_blocking=True)
                with torch.inference_mode():
                    with torch.amp.autocast(device_type=device.type, dtype=inference_dtype, enabled=use_autocast and device.type == "cuda"):
                        output, state = model.forward_step(frame, state, prev_action=prev_action)
                    current_logits = output.button_logits.detach().float()
                    predicted = (torch.sigmoid(current_logits) >= threshold_tensor).to(dtype=inference_dtype)
                    prev_action = predicted.detach()
                logits.append(current_logits[0].cpu().numpy().astype(np.float32))
                source_indices.append(source_idx)
            source_idx += 1
            pbar.update(1)
        pbar.close()
    finally:
        cap.release()

    if not logits:
        return np.zeros((0, int(cfg.num_bin)), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(logits, axis=0), np.asarray(source_indices, dtype=np.int64)


def infer_pairs(
    pairs: Sequence[Tuple[str, str]],
    model: DrivingVideoPolicy,
    cfg: ModelConfig,
    device: torch.device,
    *,
    max_frames: Optional[int],
    frame_stride: int,
    reset_mode: str,
    reset_every: int,
    thresholds: np.ndarray,
    use_autocast: bool,
    inference_dtype: torch.dtype,
) -> Dict[str, Dict[str, np.ndarray]]:
    results: Dict[str, Dict[str, np.ndarray]] = {}
    for video_path, csv_path in pairs:
        logits, source_indices = infer_run(
            video_path,
            model,
            cfg,
            device,
            max_frames=max_frames,
            frame_stride=frame_stride,
            reset_mode=reset_mode,
            reset_every=reset_every,
            thresholds=thresholds,
            use_autocast=use_autocast,
            inference_dtype=inference_dtype,
        )
        buttons = load_buttons(csv_path, cfg)
        results[video_path] = {
            "logits": logits,
            "source_indices": source_indices,
            "buttons": buttons,
        }
    return results


def flatten_results(
    results: Dict[str, Dict[str, np.ndarray]],
    target_offset: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_transitions: List[np.ndarray] = []
    all_valid: List[np.ndarray] = []
    for item in results.values():
        logits = item["logits"]
        labels, transitions, valid = make_targets(item["buttons"], item["source_indices"], target_offset)
        all_logits.append(logits)
        all_labels.append(labels)
        all_transitions.append(transitions)
        all_valid.append(valid)
    return (
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_transitions, axis=0),
        np.concatenate(all_valid, axis=0),
    )


def conflict_loss_np(logits: np.ndarray, valid: np.ndarray, names: Sequence[str]) -> float:
    if not bool(valid.any()):
        return 0.0
    name_to_idx = {str(name).lower(): idx for idx, name in enumerate(names)}
    probs = sigmoid_np(logits[valid])
    losses: List[np.ndarray] = []
    for first, second in CONFLICTING_BUTTON_PAIRS:
        first_idx = name_to_idx.get(first)
        second_idx = name_to_idx.get(second)
        if first_idx is not None and second_idx is not None:
            losses.append(probs[:, first_idx] * probs[:, second_idx])
    if not losses:
        return 0.0
    return float(np.stack(losses, axis=-1).sum(axis=-1).mean())


def binary_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    thresholds: np.ndarray,
) -> Dict[str, float]:
    if not bool(valid.any()):
        return {"accuracy": 0.0, "macro_f1": 0.0, "precision": 0.0, "recall": 0.0}
    pred = (sigmoid_np(logits[valid]) >= thresholds.reshape(1, -1)).astype(np.float32)
    true = labels[valid].astype(np.float32)
    tp = (pred * true).sum(axis=0)
    tn = ((1.0 - pred) * (1.0 - true)).sum(axis=0)
    fp = (pred * (1.0 - true)).sum(axis=0)
    fn = ((1.0 - pred) * true).sum(axis=0)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / np.maximum(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-8)
    measured = (tp + fp + fn) > 0.0
    return {
        "accuracy": float((tp + tn).sum() / np.maximum(tp + tn + fp + fn, 1.0).sum()),
        "macro_f1": float(f1[measured].mean()) if bool(measured.any()) else 0.0,
        "precision": float(precision[measured].mean()) if bool(measured.any()) else 0.0,
        "recall": float(recall[measured].mean()) if bool(measured.any()) else 0.0,
    }


def loss_breakdown(
    logits: np.ndarray,
    labels: np.ndarray,
    transitions: np.ndarray,
    valid: np.ndarray,
    pos_weight: torch.Tensor,
    *,
    transition_weight: float,
    conflict_weight: float,
    names: Sequence[str],
) -> Tuple[Dict[str, float], List[Dict[str, float | str]]]:
    if not bool(valid.any()):
        raise RuntimeError("No valid aligned predictions were available for loss breakdown.")

    logits_t = torch.from_numpy(logits[valid]).float()
    labels_t = torch.from_numpy(labels[valid]).float()
    transitions_t = torch.from_numpy(transitions[valid]).float()

    plain_bce = F.binary_cross_entropy_with_logits(logits_t, labels_t, reduction="none")
    weighted_bce = F.binary_cross_entropy_with_logits(
        logits_t,
        labels_t,
        pos_weight=pos_weight.float().view(1, -1),
        reduction="none",
    )
    transition_weights = 1.0 + (float(transition_weight) - 1.0) * transitions_t
    transition_bce = weighted_bce * transition_weights

    normal_mask = transitions_t <= 0.5
    transition_mask = transitions_t > 0.5
    conflict = conflict_loss_np(logits, valid, names)
    summary = {
        "plain_bce": float(plain_bce.mean().item()),
        "class_weighted_bce": float(weighted_bce.mean().item()),
        "transition_weighted_bce": float(transition_bce.sum().item() / transition_weights.sum().clamp(min=1.0).item()),
        "normal_frame_weighted_bce": float(weighted_bce[normal_mask].mean().item()) if bool(normal_mask.any()) else 0.0,
        "transition_frame_weighted_bce": float(weighted_bce[transition_mask].mean().item()) if bool(transition_mask.any()) else 0.0,
        "conflict_loss": conflict,
        "conflict_weighted": float(conflict_weight) * conflict,
        "total_like_train": float(transition_bce.sum().item() / transition_weights.sum().clamp(min=1.0).item())
        + float(conflict_weight) * conflict,
    }

    rows: List[Dict[str, float | str]] = []
    for idx, name in enumerate(names):
        key_transition = transitions_t[:, idx] > 0.5
        key_normal = ~key_transition
        denom = transition_weights[:, idx].sum().clamp(min=1.0)
        rows.append(
            {
                "name": str(name),
                "plain_bce": float(plain_bce[:, idx].mean().item()),
                "class_weighted_bce": float(weighted_bce[:, idx].mean().item()),
                "transition_weighted_bce": float(transition_bce[:, idx].sum().item() / denom.item()),
                "normal_frame_weighted_bce": (
                    float(weighted_bce[key_normal, idx].mean().item()) if bool(key_normal.any()) else 0.0
                ),
                "transition_frame_weighted_bce": (
                    float(weighted_bce[key_transition, idx].mean().item()) if bool(key_transition.any()) else 0.0
                ),
                "transition_rate": float(transitions_t[:, idx].mean().item()),
                "active_rate": float(labels[valid, idx].mean()),
                "pos_weight": float(pos_weight[idx].item()),
            }
        )
    return summary, rows


def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_loss_breakdown(
    out_dir: str,
    summary: Dict[str, float],
    per_key: Sequence[Dict[str, float | str]],
) -> None:
    plt = _try_matplotlib()
    os.makedirs(out_dir, exist_ok=True)
    summary_keys = [
        "plain_bce",
        "class_weighted_bce",
        "transition_weighted_bce",
        "conflict_weighted",
        "total_like_train",
    ]
    if plt is None:
        image = np.full((820, 1600, 3), 255, dtype=np.uint8)
        _cv_bar_chart(
            image,
            (30, 70, 760, 680),
            summary_keys,
            [summary[key] for key in summary_keys],
            "Loss Breakdown",
        )
        names = [str(row["name"]) for row in per_key]
        normal = [float(row["normal_frame_weighted_bce"]) for row in per_key]
        transition = [float(row["transition_frame_weighted_bce"]) for row in per_key]
        _cv_bar_chart(
            image,
            (835, 70, 350, 680),
            [f"{name} normal" for name in names],
            normal,
            "Per-Key Normal",
            colors=[CV_COLORS[1]],
        )
        _cv_bar_chart(
            image,
            (1210, 70, 350, 680),
            [f"{name} transition" for name in names],
            transition,
            "Per-Key Transition",
            colors=[CV_COLORS[2]],
        )
        _save_cv(os.path.join(out_dir, "loss_breakdown.png"), image)
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].bar(summary_keys, [summary[key] for key in summary_keys], color=["#4c78a8", "#72b7b2", "#f58518", "#e45756", "#54a24b"])
    axes[0].set_title("Loss Breakdown")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("loss")

    names = [str(row["name"]) for row in per_key]
    x = np.arange(len(names))
    width = 0.38
    axes[1].bar(x - width / 2, [float(row["normal_frame_weighted_bce"]) for row in per_key], width, label="normal")
    axes[1].bar(x + width / 2, [float(row["transition_frame_weighted_bce"]) for row in per_key], width, label="transition")
    axes[1].set_xticks(x, names)
    axes[1].set_title("Per-Key Weighted BCE")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_breakdown.png"), dpi=160)
    plt.close(fig)


def plot_logit_tracker(
    out_dir: str,
    checkpoint_stats: Sequence[Dict[str, object]],
    names: Sequence[str],
) -> None:
    plt = _try_matplotlib()
    os.makedirs(out_dir, exist_ok=True)

    labels = [str(item["checkpoint"]) for item in checkpoint_stats]
    if plt is None:
        panel_h = 250
        image = np.full((80 + panel_h * len(names), 1400, 3), 255, dtype=np.uint8)
        _cv_text(image, "Probability Distribution Tracker", 20, 35, scale=0.85, thickness=2)
        for key_idx, name in enumerate(names):
            means = [float(item[f"{name}_prob_mean"]) for item in checkpoint_stats]
            lows = [float(item[f"{name}_prob_p05"]) for item in checkpoint_stats]
            highs = [float(item[f"{name}_prob_p95"]) for item in checkpoint_stats]
            top = 80 + key_idx * panel_h
            _cv_line_chart(
                image,
                (30, top + 20, 1320, 205),
                {"p05": lows, "mean": means, "p95": highs},
                str(name),
                y_min=0.0,
                y_max=1.0,
                x_labels=labels,
            )
        _save_cv(os.path.join(out_dir, "logit_distribution_tracker.png"), image)
        return

    x = np.arange(len(labels))
    fig, axes = plt.subplots(len(names), 1, figsize=(max(10, len(labels) * 1.4), 3 * len(names)), sharex=True)
    axes_list = axes if isinstance(axes, np.ndarray) else np.asarray([axes])
    for key_idx, name in enumerate(names):
        means = [float(item[f"{name}_prob_mean"]) for item in checkpoint_stats]
        lows = [float(item[f"{name}_prob_p05"]) for item in checkpoint_stats]
        highs = [float(item[f"{name}_prob_p95"]) for item in checkpoint_stats]
        axes_list[key_idx].plot(x, means, marker="o", label="mean prob")
        axes_list[key_idx].fill_between(x, lows, highs, alpha=0.2, label="p05-p95")
        axes_list[key_idx].set_ylim(0.0, 1.0)
        axes_list[key_idx].set_ylabel(str(name))
        axes_list[key_idx].legend(loc="upper right")
    axes_list[-1].set_xticks(x, labels, rotation=35, ha="right")
    fig.suptitle("Probability Distribution Tracker")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "logit_distribution_tracker.png"), dpi=160)
    plt.close(fig)


def plot_probability_histograms(
    out_dir: str,
    label: str,
    logits: np.ndarray,
    names: Sequence[str],
) -> None:
    plt = _try_matplotlib()
    probs = sigmoid_np(logits)
    if plt is None:
        _cv_histograms(os.path.join(out_dir, f"logit_histograms_{label}.png"), label, probs, names)
        return

    cols = min(4, len(names))
    rows = int(math.ceil(len(names) / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    bins = np.linspace(0.0, 1.0, 41)
    for idx, name in enumerate(names):
        ax = axes[idx // cols][idx % cols]
        ax.hist(probs[:, idx], bins=bins, color="#4c78a8", alpha=0.9)
        ax.set_title(str(name))
        ax.set_xlim(0.0, 1.0)
    for idx in range(len(names), rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle(f"Probability Histograms: {label}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"logit_histograms_{label}.png"), dpi=160)
    plt.close(fig)


def logit_distribution_rows(label: str, logits: np.ndarray, names: Sequence[str]) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    probs = sigmoid_np(logits)
    row: Dict[str, object] = {"checkpoint": label, "samples": int(logits.shape[0])}
    rows: List[Dict[str, object]] = []
    for idx, name in enumerate(names):
        logit_col = logits[:, idx]
        prob_col = probs[:, idx]
        stats = {
            "checkpoint": label,
            "name": str(name),
            "samples": int(logits.shape[0]),
            "logit_mean": float(np.mean(logit_col)),
            "logit_std": float(np.std(logit_col)),
            "logit_p05": float(np.quantile(logit_col, 0.05)),
            "logit_p50": float(np.quantile(logit_col, 0.50)),
            "logit_p95": float(np.quantile(logit_col, 0.95)),
            "prob_mean": float(np.mean(prob_col)),
            "prob_std": float(np.std(prob_col)),
            "prob_p05": float(np.quantile(prob_col, 0.05)),
            "prob_p50": float(np.quantile(prob_col, 0.50)),
            "prob_p95": float(np.quantile(prob_col, 0.95)),
            "prob_lt_005": float(np.mean(prob_col < 0.05)),
            "prob_gt_095": float(np.mean(prob_col > 0.95)),
        }
        rows.append(stats)
        for key, value in stats.items():
            if key not in {"checkpoint", "name", "samples"}:
                row[f"{name}_{key}"] = value
    return row, rows


def plot_state_ablation(
    out_dir: str,
    first_run_name: str,
    mode_results: Dict[str, Dict[str, np.ndarray]],
    labels: np.ndarray,
    valid: np.ndarray,
    thresholds: np.ndarray,
    names: Sequence[str],
    max_points: int,
) -> None:
    plt = _try_matplotlib()
    os.makedirs(out_dir, exist_ok=True)
    valid_indices = np.flatnonzero(valid)[: int(max_points)]
    if valid_indices.size == 0:
        return
    if plt is None:
        panel_h = 260
        image = np.full((80 + panel_h * len(names), 1500, 3), 255, dtype=np.uint8)
        _cv_text(image, f"State Reset Ablation: {first_run_name}", 20, 35, scale=0.85, thickness=2)
        for key_idx, name in enumerate(names):
            series: Dict[str, Sequence[float]] = {
                "truth": labels[valid_indices, key_idx].astype(np.float32).tolist(),
            }
            for mode, item in mode_results.items():
                probs = sigmoid_np(item["logits"])
                series[mode] = probs[valid_indices, key_idx].astype(np.float32).tolist()
            top = 80 + key_idx * panel_h
            _cv_line_chart(
                image,
                (30, top + 20, 1420, 215),
                series,
                f"{name}  threshold={float(thresholds[key_idx]):.3f}",
                y_min=0.0,
                y_max=1.0,
            )
        _save_cv(os.path.join(out_dir, "state_reset_ablation_timeline.png"), image)
        return

    rows = len(names)
    fig, axes = plt.subplots(rows, 1, figsize=(16, 2.5 * rows), sharex=True)
    axes_list = axes if isinstance(axes, np.ndarray) else np.asarray([axes])
    x = valid_indices
    for key_idx, name in enumerate(names):
        ax = axes_list[key_idx]
        ax.step(x, labels[valid_indices, key_idx], where="post", color="black", linewidth=1.2, label="truth")
        for mode, item in mode_results.items():
            probs = sigmoid_np(item["logits"])
            ax.plot(x, probs[valid_indices, key_idx], linewidth=1.0, label=mode)
        ax.axhline(float(thresholds[key_idx]), color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(str(name))
        ax.legend(loc="upper right")
    axes_list[-1].set_xlabel("sample index")
    fig.suptitle(f"State Reset Ablation: {first_run_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "state_reset_ablation_timeline.png"), dpi=160)
    plt.close(fig)


def run_loss_viewer(args, model, cfg, raw_config, pairs, device, thresholds, dtype, use_autocast) -> None:
    results = infer_pairs(
        pairs,
        model,
        cfg,
        device,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        reset_mode="carry",
        reset_every=args.reset_every,
        thresholds=thresholds,
        use_autocast=use_autocast,
        inference_dtype=dtype,
    )
    target_offset = int(tuple(cfg.prediction_horizon_offsets)[0]) + int(args.action_label_offset)
    logits, labels, transitions, valid = flatten_results(results, target_offset)
    pos_weight = compute_pos_weight(
        [item["buttons"] for item in results.values()],
        float(raw_config.get("pos_weight_power", 0.65)),
        float(raw_config.get("pos_weight_clamp", 25.0)),
    )
    names = list(cfg.key_names) + list(cfg.mouse_button_names)
    summary, per_key = loss_breakdown(
        logits,
        labels,
        transitions,
        valid,
        pos_weight,
        transition_weight=float(raw_config.get("transition_loss_weight", 4.0)),
        conflict_weight=float(raw_config.get("conflicting_button_loss_weight", 0.20)),
        names=names,
    )
    out_dir = os.path.join(args.out_dir, "loss_breakdown")
    write_csv(os.path.join(out_dir, "loss_summary.csv"), [summary])
    write_csv(os.path.join(out_dir, "per_key_loss.csv"), per_key)
    plot_loss_breakdown(out_dir, summary, per_key)


def run_logit_tracker(args, cfg, pairs, device, thresholds, dtype, use_autocast, checkpoints: Sequence[str]) -> None:
    names = list(cfg.key_names) + list(cfg.mouse_button_names)
    out_dir = os.path.join(args.out_dir, "logit_distribution")
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_rows: List[Dict[str, object]] = []
    per_key_rows: List[Dict[str, object]] = []
    for checkpoint_path in checkpoints:
        label = _checkpoint_label(checkpoint_path)
        model, checkpoint_cfg, _ = load_model_checkpoint(checkpoint_path, device)
        if list(checkpoint_cfg.key_names) + list(checkpoint_cfg.mouse_button_names) != names:
            raise RuntimeError(f"Checkpoint {checkpoint_path!r} action names do not match the first checkpoint.")
        results = infer_pairs(
            pairs,
            model,
            checkpoint_cfg,
            device,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
            reset_mode="carry",
            reset_every=args.reset_every,
            thresholds=thresholds,
            use_autocast=use_autocast,
            inference_dtype=dtype,
        )
        target_offset = int(tuple(checkpoint_cfg.prediction_horizon_offsets)[0]) + int(args.action_label_offset)
        logits, _, _, valid = flatten_results(results, target_offset)
        logits = logits[valid]
        checkpoint_row, rows = logit_distribution_rows(label, logits, names)
        checkpoint_rows.append(checkpoint_row)
        per_key_rows.extend(rows)
        plot_probability_histograms(out_dir, label, logits, names)
    write_csv(os.path.join(out_dir, "logit_distribution_summary.csv"), checkpoint_rows)
    write_csv(os.path.join(out_dir, "logit_distribution_per_key.csv"), per_key_rows)
    plot_logit_tracker(out_dir, checkpoint_rows, names)


def run_state_ablation(args, model, cfg, pairs, device, thresholds, dtype, use_autocast) -> None:
    if not pairs:
        raise RuntimeError("State ablation requires at least one run pair.")
    video_path, csv_path = pairs[0]
    buttons = load_buttons(csv_path, cfg)
    modes = ("carry", "frame", "clip")
    mode_results: Dict[str, Dict[str, np.ndarray]] = {}
    target_offset = int(tuple(cfg.prediction_horizon_offsets)[0]) + int(args.action_label_offset)
    for mode in modes:
        logits, source_indices = infer_run(
            video_path,
            model,
            cfg,
            device,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
            reset_mode=mode,
            reset_every=args.reset_every,
            thresholds=thresholds,
            use_autocast=use_autocast,
            inference_dtype=dtype,
        )
        labels, transitions, valid = make_targets(buttons, source_indices, target_offset)
        mode_results[mode] = {
            "logits": logits,
            "source_indices": source_indices,
            "labels": labels,
            "transitions": transitions,
            "valid": valid,
        }

    names = list(cfg.key_names) + list(cfg.mouse_button_names)
    rows: List[Dict[str, object]] = []
    carry_probs = sigmoid_np(mode_results["carry"]["logits"])
    for mode, item in mode_results.items():
        metrics = binary_metrics(item["logits"], item["labels"], item["valid"], thresholds)
        probs = sigmoid_np(item["logits"])
        delta = np.abs(carry_probs - probs)
        row: Dict[str, object] = {"mode": mode, **metrics, "mean_abs_prob_delta_vs_carry": float(delta.mean())}
        for idx, name in enumerate(names):
            row[f"{name}_mean_abs_prob_delta_vs_carry"] = float(delta[:, idx].mean())
        rows.append(row)

    out_dir = os.path.join(args.out_dir, "state_reset_ablation")
    write_csv(os.path.join(out_dir, "state_reset_ablation_summary.csv"), rows)
    plot_state_ablation(
        out_dir,
        os.path.basename(video_path),
        mode_results,
        mode_results["carry"]["labels"],
        mode_results["carry"]["valid"],
        thresholds,
        names,
        args.timeline_points,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Debug policy loss, logits, and ConvGRU state behavior.")
    parser.add_argument("--mode", choices=["loss", "logits", "state", "all"], default="all")
    parser.add_argument("--ckpt-dir", default="./checkpoints_rt")
    parser.add_argument("--ckpt-path", default=r'C:\Users\Abhil\Desktop\Github_Projects\VideoAgent\checkpoints_rt\model_latest.pt')
    parser.add_argument("--checkpoint-glob", default=None, help="For logit tracking, e.g. model_epoch_*.pt")
    parser.add_argument("--data-root", default=r'C:\Users\Abhil\Desktop\Github_Projects\VideoAgent\data\greenville')
    parser.add_argument("--out-dir", default="./debug_policy_reports")
    parser.add_argument("--max-runs", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=2000)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--action-label-offset", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--reset-every", type=int, default=800, help="Frame interval for the clip-reset ablation mode.")
    parser.add_argument("--timeline-points", type=int, default=800)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--no-autocast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.amp_dtype == "bf16" and device.type == "cuda" else torch.float32
    use_autocast = bool(not args.no_autocast and dtype == torch.bfloat16 and device.type == "cuda")

    ckpt_path = args.ckpt_path or _default_checkpoint(args.ckpt_dir)
    model, cfg, raw_config = load_model_checkpoint(ckpt_path, device)
    if args.action_label_offset is None:
        args.action_label_offset = int(raw_config.get("action_label_offset", 0) or 0)
    data_root = args.data_root or str(cfg.data_root)
    pairs = find_run_pairs(data_root, cfg.video_ext, cfg.csv_ext, args.max_runs)
    thresholds = resolve_thresholds(cfg, args.threshold)
    checkpoints = discover_checkpoints(args.ckpt_dir, ckpt_path, args.checkpoint_glob)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "README.txt"), "w", encoding="utf-8") as file_obj:
        file_obj.write(
            "Policy debug reports\n"
            f"checkpoint={ckpt_path}\n"
            f"data_root={data_root}\n"
            f"runs={len(pairs)}\n"
            f"max_frames={args.max_frames}\n"
            f"frame_stride={args.frame_stride}\n"
            f"action_label_offset={args.action_label_offset}\n"
            f"prediction_offset={tuple(cfg.prediction_horizon_offsets)[0]}\n"
        )

    if args.mode in {"loss", "all"}:
        run_loss_viewer(args, model, cfg, raw_config, pairs, device, thresholds, dtype, use_autocast)
    if args.mode in {"logits", "all"}:
        run_logit_tracker(args, cfg, pairs, device, thresholds, dtype, use_autocast, checkpoints)
    if args.mode in {"state", "all"}:
        run_state_ablation(args, model, cfg, pairs, device, thresholds, dtype, use_autocast)

    print(f"Wrote debug reports to {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
