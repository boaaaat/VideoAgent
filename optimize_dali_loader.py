import argparse
import csv
import glob
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from tqdm.auto import tqdm

import torch

from nvidia.dali import pipeline_def, types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from dataset_wsl_sync import sync_dataset_for_training
from inverse_dynamics import InverseDynamicsConfig


@dataclass(frozen=True)
class LoaderVariant:
    name: str
    batch_size: int = 4                  # Keep this equal to the model training batch size.
    num_threads: int = 8                 # CPU container/parser/loader threads.
    prefetch_queue_depth: int = 4         # Pipeline prefetch depth.
    reader_prefetch_queue_depth: int = 4  # Reader-side prefetch depth.
    read_ahead: bool = False
    dont_use_mmap: bool = False
    prepare_first_batch: bool = True
    random_shuffle: bool = False


@pipeline_def
def video_pipeline(
    *,
    filenames: List[str],
    labels: List[int],
    seq_len: int,
    source_size: int,
    model_size: int,
    step: int,
    reader_prefetch_queue_depth: int,
    read_ahead: bool,
    dont_use_mmap: bool,
    random_shuffle: bool,
    output_layout: str,
    normalize: bool,
):
    """Reads H.264/H.265 video on the GPU, resizes 512 -> 256, and optionally returns [T,C,H,W]."""
    seq_len = int(seq_len)
    source_size = int(source_size)
    model_size = int(model_size)
    step = int(step)

    # Reader output is still the decoded source resolution, so this hint should match source_size.
    reader_bytes = seq_len * source_size * source_size * 3
    model_bytes = seq_len * model_size * model_size * 3

    reader_outputs = fn.experimental.readers.video(
        device="gpu",
        name="Reader",
        filenames=filenames,
        labels=labels,
        sequence_length=seq_len,
        step=step,
        stride=1,
        random_shuffle=bool(random_shuffle),
        prefetch_queue_depth=int(reader_prefetch_queue_depth),
        read_ahead=bool(read_ahead),
        dont_use_mmap=bool(dont_use_mmap),
        image_type=types.RGB,
        enable_frame_num="scalar",
        bytes_per_sample_hint=reader_bytes,
        tensor_init_bytes=reader_bytes,
        pad_mode="none",
    )
    frames, labels_out, frame_num = reader_outputs

    # Your videos are 512, but the model wants 256. Do this before PyTorch sees the tensor.
    frames = fn.resize(
        frames,
        resize_x=model_size,
        resize_y=model_size,
        interp_type=types.INTERP_LINEAR,
        bytes_per_sample_hint=model_bytes,
        temp_buffer_hint=model_bytes,
    )

    if normalize:
        frames = fn.cast(frames, dtype=types.FLOAT) / 255.0

    # DALI video samples are [T,H,W,C]. Most PyTorch video models want [T,C,H,W].
    # Doing the transpose inside DALI avoids a PyTorch permute(...).contiguous() in the training loop.
    if output_layout == "B_T_C_H_W":
        frames = fn.transpose(frames, perm=[0, 3, 1, 2])

    return frames, labels_out, frame_num


def find_videos_with_csv(data_root: str, video_ext: str, csv_ext: str, max_videos: Optional[int]) -> Tuple[List[str], List[str]]:
    videos = sorted(glob.glob(os.path.join(data_root, f"run_*{video_ext}")))
    video_paths: List[str] = []
    csv_paths: List[str] = []

    for video_path in videos:
        csv_path = os.path.splitext(video_path)[0] + csv_ext
        if os.path.exists(csv_path):
            video_paths.append(video_path)
            csv_paths.append(csv_path)
            if max_videos is not None and len(video_paths) >= max_videos:
                break

    if not video_paths:
        raise FileNotFoundError(f"No run_*{video_ext} with matching {csv_ext} found under {data_root!r}.")

    return video_paths, csv_paths


def count_csv_rows(csv_path: str) -> int:
    with open(csv_path, "r", newline="") as file_obj:
        reader = csv.reader(file_obj)
        next(reader, None)
        return sum(1 for _ in reader)


def sample_count(frame_count: int, seq_len: int, step: int) -> int:
    if frame_count < seq_len:
        return 0
    return ((frame_count - seq_len) // step) + 1


def build_iterator(
    variant: LoaderVariant,
    *,
    video_paths: List[str],
    seq_len: int,
    source_size: int,
    model_size: int,
    step: int,
    output_layout: str,
    normalize: bool,
):
    labels = list(range(len(video_paths)))

    pipe = video_pipeline(
        batch_size=int(variant.batch_size),
        num_threads=int(variant.num_threads),
        device_id=0,
        seed=1337,
        prefetch_queue_depth=int(variant.prefetch_queue_depth),
        exec_async=True,
        exec_pipelined=True,
        filenames=video_paths,
        labels=labels,
        seq_len=seq_len,
        source_size=source_size,
        model_size=model_size,
        step=step,
        reader_prefetch_queue_depth=variant.reader_prefetch_queue_depth,
        read_ahead=variant.read_ahead,
        dont_use_mmap=variant.dont_use_mmap,
        random_shuffle=variant.random_shuffle,
        output_layout=output_layout,
        normalize=normalize,
    )
    pipe.build()

    return DALIGenericIterator(
        [pipe],
        output_map=["frames", "labels", "frame_num"],
        reader_name="Reader",
        auto_reset=False,
        last_batch_policy=LastBatchPolicy.PARTIAL,
        prepare_first_batch=variant.prepare_first_batch,
    )


def benchmark_variant(
    variant: LoaderVariant,
    *,
    video_paths: List[str],
    seq_len: int,
    source_size: int,
    model_size: int,
    step: int,
    total_samples: int,
    output_layout: str,
    normalize: bool,
    warmup_batches: int,
) -> Dict[str, float | str | int | Tuple[int, ...]]:
    iterator = build_iterator(
        variant,
        video_paths=video_paths,
        seq_len=seq_len,
        source_size=source_size,
        model_size=model_size,
        step=step,
        output_layout=output_layout,
        normalize=normalize,
    )

    total_batches = int(math.ceil(total_samples / float(variant.batch_size)))
    timed_batches = max(1, total_batches - warmup_batches)

    first_shape: Optional[Tuple[int, ...]] = None
    timed_samples = 0

    # Warmup: lets DALI fill its queues and avoids timing one-time setup.
    for _ in range(min(warmup_batches, total_batches)):
        batch = next(iterator)[0]
        frames = batch["frames"]
        if first_shape is None:
            first_shape = tuple(int(dim) for dim in frames.shape)

    torch.cuda.synchronize()
    start = time.perf_counter()

    with tqdm(total=timed_batches, desc=variant.name, unit="batch", dynamic_ncols=True) as pbar:
        for batch_idx in range(timed_batches):
            batch = next(iterator)[0]
            frames = batch["frames"]

            if first_shape is None:
                first_shape = tuple(int(dim) for dim in frames.shape)

            batch_samples = int(frames.shape[0])
            timed_samples += batch_samples

            # Do not synchronize every batch. That kills async overlap and makes DALI look slower.
            if batch_idx == 0 or (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == timed_batches:
                elapsed_now = max(time.perf_counter() - start, 1e-9)
                pbar.set_postfix(
                    {
                        "samples/s": f"{timed_samples / elapsed_now:.1f}",
                        "decoded_frames/s": f"{timed_samples * seq_len / elapsed_now:.1f}",
                    }
                )
            pbar.update(1)

    torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - start, 1e-9)
    iterator.reset()

    return {
        "name": variant.name,
        "batch_size": int(variant.batch_size),
        "samples": int(timed_samples),
        "decoded_frames": int(timed_samples * seq_len),
        "seconds": float(elapsed),
        "samples_per_second": float(timed_samples / elapsed),
        "decoded_frames_per_second": float(timed_samples * seq_len / elapsed),
        "first_shape": first_shape if first_shape is not None else (),
    }


def default_variants(batch_size: int) -> List[LoaderVariant]:
    # Keep batch_size aligned with training. These variants test overlap and prefetch settings only.
    return [
        LoaderVariant(
            "experimental_b4_prefetch4_threads8",
            batch_size=batch_size,
            num_threads=8,
            prefetch_queue_depth=4,
            reader_prefetch_queue_depth=4,
            read_ahead=True,
        ),
    ]


def parse_args() -> argparse.Namespace:
    cfg = InverseDynamicsConfig()
    parser = argparse.ArgumentParser(description="Benchmark DALI video loading for 24-frame inverse dynamics batches.")
    parser.add_argument("--data-root", default=cfg.data_root)
    parser.add_argument("--dataset-cache-root", default=None, help="Local Linux dataset cache. Defaults to ~/ai/dataset in WSL, ./dataset elsewhere.")
    parser.add_argument("--sync-dataset", dest="sync_dataset", action="store_true", default=True)
    parser.add_argument("--no-sync-dataset", dest="sync_dataset", action="store_false")
    parser.add_argument("--dataset-sync-delete-stale", dest="dataset_sync_delete_stale", action="store_true", default=None)
    parser.add_argument("--dataset-sync-keep-stale", dest="dataset_sync_delete_stale", action="store_false")
    parser.add_argument("--dataset-sync-hash-same-size", dest="dataset_sync_hash_same_size", action="store_true", default=True)
    parser.add_argument("--dataset-sync-no-hash-same-size", dest="dataset_sync_hash_same_size", action="store_false")
    parser.add_argument("--video-ext", default=cfg.video_ext)
    parser.add_argument("--csv-ext", default=cfg.csv_ext)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--source-size", type=int, default=512, help="Decoded video resolution. Your source videos are 512x512.")
    parser.add_argument("--model-size", type=int, default=256, help="Output resolution for the model. Your model wants 256x256.")
    parser.add_argument("--batch-size", type=int, default=32, help="Keep this at 4 to match training.")
    parser.add_argument("--step", type=int, default=cfg.train_seq_stride, help="Frame step between sequence starts. Use 1 for overlapping clips or 24 for non-overlap speed test.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-videos", type=int, default=None, help="Use more than one video if possible. Single-video tests often underfeed NVDEC.")
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--normalize", action="store_true", help="Return float32 [0,1] instead of uint8. Benchmark separately because this adds GPU work.")
    parser.add_argument(
        "--output-layout",
        choices=["B_T_H_W_C", "B_T_C_H_W"],
        default="B_T_C_H_W",
        help="B_T_C_H_W matches most PyTorch video models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to benchmark DALI GPU video decoding.")

    if args.sync_dataset:
        args.data_root = sync_dataset_for_training(
            data_root=args.data_root,
            target_root=args.dataset_cache_root,
            video_ext=args.video_ext,
            csv_ext=args.csv_ext,
            max_videos=args.max_videos,
            delete_stale=args.dataset_sync_delete_stale,
            hash_same_size=bool(args.dataset_sync_hash_same_size),
        )

    video_paths, csv_paths = find_videos_with_csv(args.data_root, args.video_ext, args.csv_ext, args.max_videos)
    frame_counts = [count_csv_rows(path) for path in csv_paths]
    sample_counts = [sample_count(count, int(args.seq_len), int(args.step)) for count in frame_counts]
    total_samples = sum(sample_counts)

    if total_samples <= 0:
        raise RuntimeError(
            f"No usable samples for seq_len={args.seq_len}, step={args.step}. "
            f"videos={len(video_paths)} total_csv_rows={sum(frame_counts)}"
        )

    if args.max_samples is not None:
        total_samples = min(total_samples, max(1, int(args.max_samples)))

    print("Benchmark input:")
    print(f"  data_root={args.data_root}")
    print(f"  videos={len(video_paths)}")
    print(f"  first_video={video_paths[0]}")
    print(f"  total_csv_rows={sum(frame_counts)}")
    print(f"  seq_len={args.seq_len} source_size={args.source_size} model_size={args.model_size} step={args.step}")
    print(f"  batch_size={args.batch_size}")
    print(f"  samples={total_samples}")
    print(f"  output_layout={args.output_layout} dtype={'float32' if args.normalize else 'uint8'}")
    print("  decoded_frames/s = samples/s * seq_len")

    if len(video_paths) == 1:
        print("\nWARNING: You are benchmarking one video file. That often underfeeds NVDEC. Use many run_*.mp4 files for a realistic loader test.\n")

    results: List[Dict[str, float | str | int | Tuple[int, ...]]] = []
    for variant in default_variants(int(args.batch_size)):
        result = benchmark_variant(
            variant,
            video_paths=video_paths,
            seq_len=int(args.seq_len),
            source_size=int(args.source_size),
            model_size=int(args.model_size),
            step=int(args.step),
            total_samples=int(total_samples),
            output_layout=str(args.output_layout),
            normalize=bool(args.normalize),
            warmup_batches=int(args.warmup_batches),
        )
        results.append(result)

    results = sorted(results, key=lambda item: float(item["decoded_frames_per_second"]), reverse=True)
    print("\nResults:")
    for idx, item in enumerate(results, start=1):
        print(
            f"{idx:2d}. {item['name']:34s} "
            f"{float(item['samples_per_second']):9.2f} samples/s "
            f"{float(item['decoded_frames_per_second']):11.2f} decoded_frames/s "
            f"{float(item['seconds']):7.3f}s "
            f"shape={item['first_shape']}"
        )

    print("\nFor training with seq_len=24 and middle 8 targets:")
    print("  frames shape should be [B,24,C,256,256] when --output-layout B_T_C_H_W")
    print("  middle_8 = frames[:, 8:16]")


if __name__ == "__main__":
    main()
