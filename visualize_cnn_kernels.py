"""Export CNN kernel visualizations from a policy checkpoint or fresh model.

The default view creates one image grid per 4-D convolution weight tensor, with
one tile per output filter. RGB input kernels are shown as signed RGB maps.
Other kernels are collapsed across input channels into signed-mean and energy
maps, which are easier to inspect than thousands of individual channel pairs.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
from dataclasses import fields
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from models import ARCHITECTURE_VERSION, DrivingVideoPolicy, ModelConfig


SummaryRow = Dict[str, object]


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return cleaned.strip("._") or "unnamed"


def _find_checkpoint(ckpt_dir: str) -> Optional[str]:
    candidates = [
        os.path.join(ckpt_dir, "model_best.pt"),
        os.path.join(ckpt_dir, "model_latest.pt"),
    ]
    epoch_paths = glob.glob(os.path.join(ckpt_dir, "model_epoch_*.pt"))
    if epoch_paths:
        epoch_paths.sort(key=os.path.getmtime, reverse=True)
        candidates.extend(epoch_paths)
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _model_config_from_checkpoint(config_dict: Dict[str, object]) -> ModelConfig:
    values: Dict[str, object] = {}
    for field in fields(ModelConfig):
        value = config_dict.get(field.name)
        if value is not None:
            values[field.name] = value
    values["architecture_version"] = ARCHITECTURE_VERSION
    return ModelConfig(**values)


def _fresh_model_state() -> Tuple[Dict[str, torch.Tensor], str]:
    model = DrivingVideoPolicy(ModelConfig())
    return model.state_dict(), "fresh_random_init"


def _checkpoint_state(path: str, *, partial_model_load: bool) -> Tuple[Dict[str, torch.Tensor], str]:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and isinstance(state.get("model_state"), dict):
        model_state = state["model_state"]
        if partial_model_load:
            config_dict = dict(state.get("config", {})) if isinstance(state.get("config"), dict) else {}
            model = DrivingVideoPolicy(_model_config_from_checkpoint(config_dict))
            load_result = model.load_state_dict(model_state, strict=False)
            if load_result.missing_keys:
                print(f"Partial load missing {len(load_result.missing_keys)} keys.")
            if load_result.unexpected_keys:
                print(f"Partial load ignored {len(load_result.unexpected_keys)} unexpected keys.")
            model_state = model.state_dict()
        return model_state, path
    if isinstance(state, dict):
        return state, path
    raise RuntimeError(f"Checkpoint {path!r} is not a state dict or checkpoint payload.")


def _iter_conv_weights(state_dict: Dict[str, torch.Tensor]) -> Iterable[Tuple[str, torch.Tensor]]:
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.dim() == 4 and tensor.numel() > 0:
            yield name, tensor.detach().cpu().float()


def _signed_to_bwr(values: np.ndarray) -> np.ndarray:
    max_abs = float(np.max(np.abs(values)))
    if max_abs <= 0.0 or not math.isfinite(max_abs):
        scaled = np.zeros_like(values, dtype=np.float32)
    else:
        scaled = np.clip(values.astype(np.float32) / max_abs, -1.0, 1.0)
    image = np.empty((*scaled.shape, 3), dtype=np.float32)
    positive = scaled >= 0.0
    image[..., 0] = np.where(positive, 255.0, 255.0 * (1.0 + scaled))
    image[..., 1] = np.where(positive, 255.0 * (1.0 - scaled), 255.0 * (1.0 + scaled))
    image[..., 2] = np.where(positive, 255.0 * (1.0 - scaled), 255.0)
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def _energy_to_gray(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    low = float(np.min(values))
    high = float(np.max(values))
    if high <= low or not math.isfinite(high - low):
        gray = np.zeros_like(values, dtype=np.uint8)
    else:
        gray = np.clip((values - low) / (high - low) * 255.0, 0.0, 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def _rgb_kernel_to_image(kernel: np.ndarray) -> np.ndarray:
    max_abs = float(np.max(np.abs(kernel)))
    if max_abs <= 0.0 or not math.isfinite(max_abs):
        rgb = np.full((kernel.shape[1], kernel.shape[2], 3), 127, dtype=np.uint8)
    else:
        rgb = np.clip((kernel / max_abs + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
        rgb = np.transpose(rgb, (1, 2, 0))
    return rgb


def _kernel_to_image(kernel: torch.Tensor, mode: str) -> np.ndarray:
    array = kernel.numpy()
    if mode == "rgb" and array.shape[0] == 3:
        return _rgb_kernel_to_image(array)
    if mode == "energy":
        return _energy_to_gray(np.sqrt(np.mean(np.square(array), axis=0)))
    if mode == "signed":
        return _signed_to_bwr(np.mean(array, axis=0))
    raise ValueError(f"Unsupported kernel mode {mode!r} for shape {array.shape}.")


def _channel_kernel_to_image(kernel: torch.Tensor) -> np.ndarray:
    return _signed_to_bwr(kernel.numpy())


def _grid_dims(count: int, columns: int) -> Tuple[int, int]:
    if count <= 0:
        return 0, 0
    if columns <= 0:
        columns = int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(count / float(columns)))
    return rows, columns


def _draw_grid(
    images: Sequence[np.ndarray],
    labels: Sequence[str],
    output_path: Path,
    *,
    tile_size: int,
    columns: int,
    gutter: int = 4,
    label_height: int = 14,
) -> None:
    if not images:
        return
    rows, cols = _grid_dims(len(images), columns)
    cell_h = tile_size + label_height
    height = rows * cell_h + (rows + 1) * gutter
    width = cols * tile_size + (cols + 1) * gutter
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    for index, image in enumerate(images):
        row = index // cols
        col = index % cols
        x = gutter + col * (tile_size + gutter)
        y = gutter + row * (cell_h + gutter)
        resized = cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)
        canvas[y : y + tile_size, x : x + tile_size] = resized
        cv2.putText(
            canvas,
            labels[index][:18],
            (x, y + tile_size + 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def _visualize_output_filters(
    name: str,
    weight: torch.Tensor,
    output_dir: Path,
    *,
    layer_index: int,
    modes: Sequence[str],
    max_filters: int,
    tile_size: int,
    columns: int,
) -> List[SummaryRow]:
    out_channels, in_channels, _kernel_h, _kernel_w = weight.shape
    count = out_channels if max_filters <= 0 else min(out_channels, max_filters)
    rows: List[SummaryRow] = []
    safe = _safe_name(name)
    for mode in modes:
        effective_mode = "rgb" if mode == "rgb" and in_channels == 3 else mode
        if effective_mode == "rgb" and in_channels != 3:
            continue
        images = [_kernel_to_image(weight[index], effective_mode) for index in range(count)]
        labels = [f"o{index}" for index in range(count)]
        filename = f"{layer_index:03d}_{safe}_{effective_mode}.png"
        path = output_dir / filename
        _draw_grid(images, labels, path, tile_size=tile_size, columns=columns)
        rows.append(
            {
                "layer": name,
                "shape": "x".join(str(item) for item in weight.shape),
                "view": effective_mode,
                "filters_written": count,
                "file": str(path),
            }
        )
    return rows


def _visualize_per_input(
    name: str,
    weight: torch.Tensor,
    output_dir: Path,
    *,
    layer_index: int,
    max_outputs: int,
    max_inputs: int,
    tile_size: int,
    columns: int,
) -> List[SummaryRow]:
    if max_outputs <= 0 or max_inputs <= 0:
        return []
    out_channels, in_channels, _kernel_h, _kernel_w = weight.shape
    output_count = min(out_channels, max_outputs)
    input_count = min(in_channels, max_inputs)
    rows: List[SummaryRow] = []
    safe = _safe_name(name)
    layer_dir = output_dir / "per_input" / f"{layer_index:03d}_{safe}"
    for out_index in range(output_count):
        images = [_channel_kernel_to_image(weight[out_index, in_index]) for in_index in range(input_count)]
        labels = [f"i{index}" for index in range(input_count)]
        path = layer_dir / f"out_{out_index:04d}.png"
        _draw_grid(images, labels, path, tile_size=tile_size, columns=columns)
        rows.append(
            {
                "layer": name,
                "shape": "x".join(str(item) for item in weight.shape),
                "view": "per_input_signed",
                "filters_written": input_count,
                "file": str(path),
            }
        )
    return rows


def _write_summary(rows: Sequence[SummaryRow], output_dir: Path) -> None:
    if not rows:
        return
    path = output_dir / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["layer", "shape", "view", "filters_written", "file"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize CNN kernels from the driving policy.")
    parser.add_argument("--ckpt-path", default=None, help="Checkpoint to visualize. Defaults to ckpt-dir latest.")
    parser.add_argument("--ckpt-dir", default="./checkpoints_rt", help="Directory used when ckpt-path is omitted.")
    parser.add_argument("--random-init", action="store_true", help="Visualize a freshly initialized current model.")
    parser.add_argument(
        "--partial-model-load",
        action="store_true",
        help="Load matching checkpoint keys into the current model and ignore old/incompatible keys.",
    )
    parser.add_argument("--out-dir", default="./kernel_visualizations")
    parser.add_argument("--max-filters", type=int, default=0, help="Max output filters per layer; 0 writes all.")
    parser.add_argument("--tile-size", type=int, default=56)
    parser.add_argument("--columns", type=int, default=0, help="Grid columns; 0 chooses sqrt(count).")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["signed", "energy", "rgb"],
        choices=["signed", "energy", "rgb"],
        help="Output-filter views to write.",
    )
    parser.add_argument("--per-input-max-outputs", type=int, default=0)
    parser.add_argument("--per-input-max-inputs", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.random_init:
        state_dict, source = _fresh_model_state()
    else:
        ckpt_path = args.ckpt_path or _find_checkpoint(args.ckpt_dir)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"No checkpoint found in {args.ckpt_dir!r}. Pass --ckpt-path or use --random-init."
            )
        state_dict, source = _checkpoint_state(ckpt_path, partial_model_load=bool(args.partial_model_load))

    rows: List[SummaryRow] = []
    conv_weights = list(_iter_conv_weights(state_dict))
    if not conv_weights:
        raise RuntimeError(f"No 4-D convolution weights found in source {source!r}.")

    for layer_index, (name, weight) in enumerate(conv_weights):
        rows.extend(
            _visualize_output_filters(
                name,
                weight,
                output_dir,
                layer_index=layer_index,
                modes=args.modes,
                max_filters=int(args.max_filters),
                tile_size=int(args.tile_size),
                columns=int(args.columns),
            )
        )
        rows.extend(
            _visualize_per_input(
                name,
                weight,
                output_dir,
                layer_index=layer_index,
                max_outputs=int(args.per_input_max_outputs),
                max_inputs=int(args.per_input_max_inputs),
                tile_size=int(args.tile_size),
                columns=int(args.columns),
            )
        )

    _write_summary(rows, output_dir)
    print(f"Source: {source}")
    print(f"Convolution tensors: {len(conv_weights)}")
    print(f"Wrote {len(rows)} visualization files under {output_dir}")
    print(f"Summary: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
