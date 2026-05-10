import argparse
import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from tqdm.auto import tqdm

from action_space import game_data_root, normalize_game_name, selected_game as ACTION_SELECTED_GAME


def is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False


def default_target_root() -> Path:
    if is_wsl():
        return Path.home() / "ai" / "dataset"
    return Path("./dataset")


@dataclass
class SyncStats:
    files_seen: int = 0
    unchanged: int = 0
    copied: int = 0
    metadata_refreshed: int = 0
    deleted: int = 0
    bytes_copied: int = 0


def find_run_pairs(data_root: Path, video_ext: str, csv_ext: str, max_videos: Optional[int]) -> List[Tuple[Path, Path]]:
    video_paths = sorted(data_root.glob(f"run_*{video_ext}"))
    pairs: List[Tuple[Path, Path]] = []

    for video_path in video_paths:
        csv_path = video_path.with_suffix(csv_ext)
        if csv_path.exists():
            pairs.append((video_path, csv_path))
            if max_videos is not None and len(pairs) >= max_videos:
                break

    if not pairs:
        raise FileNotFoundError(f"No run_*{video_ext} with matching {csv_ext} found under {str(data_root)!r}.")

    return pairs


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_are_same(source_path: Path, target_path: Path, *, hash_same_size: bool) -> bool:
    if not target_path.exists():
        return False

    source_stat = source_path.stat()
    target_stat = target_path.stat()
    if source_stat.st_size != target_stat.st_size:
        return False
    if source_stat.st_mtime_ns == target_stat.st_mtime_ns:
        return True
    if not hash_same_size:
        return False

    return sha256_file(source_path) == sha256_file(target_path)


def copy_file(source_path: Path, target_path: Path) -> int:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f".{target_path.name}.tmp")
    try:
        shutil.copy2(source_path, temp_path)
        os.replace(temp_path, target_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return int(source_path.stat().st_size)


def sync_dataset(
    *,
    source_root: Path,
    target_root: Path,
    video_ext: str,
    csv_ext: str,
    max_videos: Optional[int],
    delete_stale: bool,
    hash_same_size: bool,
    dry_run: bool,
) -> SyncStats:
    source_root = source_root.resolve()
    target_root = target_root.resolve()

    try:
        if source_root.samefile(target_root):
            raise ValueError(f"Source and target are the same directory: {source_root}")
    except FileNotFoundError:
        pass

    pairs = find_run_pairs(source_root, video_ext, csv_ext, max_videos)
    files_to_sync: List[Tuple[Path, Path]] = []
    expected_targets = set()

    for video_path, csv_path in pairs:
        for source_path in (video_path, csv_path):
            target_path = target_root / source_path.name
            files_to_sync.append((source_path, target_path))
            expected_targets.add(target_path)

    stats = SyncStats(files_seen=len(files_to_sync))
    desc = f"Syncing {source_root} -> {target_root}"
    for source_path, target_path in tqdm(files_to_sync, desc=desc, unit="file", dynamic_ncols=True):
        if files_are_same(source_path, target_path, hash_same_size=hash_same_size):
            if target_path.exists() and source_path.stat().st_mtime_ns != target_path.stat().st_mtime_ns:
                stats.metadata_refreshed += 1
                if not dry_run:
                    shutil.copystat(source_path, target_path)
            stats.unchanged += 1
            continue

        stats.copied += 1
        stats.bytes_copied += int(source_path.stat().st_size)
        if not dry_run:
            copy_file(source_path, target_path)

    if delete_stale and target_root.exists():
        stale_paths = stale_cached_paths(target_root, video_ext, csv_ext, expected_targets)
        for stale_path in stale_paths:
            stats.deleted += 1
            if not dry_run:
                stale_path.unlink()

    return stats


def sync_dataset_for_training(
    *,
    data_root: str,
    target_root: Optional[str],
    video_ext: str,
    csv_ext: str,
    max_videos: Optional[int] = None,
    delete_stale: Optional[bool] = None,
    hash_same_size: bool = True,
) -> str:
    source_root = Path(data_root)
    cache_root = Path(target_root) if target_root is not None else default_target_root()
    should_delete_stale = delete_stale if delete_stale is not None else max_videos is None

    try:
        if source_root.resolve() == cache_root.resolve() or source_root.samefile(cache_root):
            print(f"Dataset cache already selected: {cache_root.resolve()}")
            return str(cache_root)
    except FileNotFoundError:
        if source_root.resolve() == cache_root.resolve():
            print(f"Dataset cache already selected: {cache_root.resolve()}")
            return str(cache_root)

    stats = sync_dataset(
        source_root=source_root,
        target_root=cache_root,
        video_ext=video_ext,
        csv_ext=csv_ext,
        max_videos=max_videos,
        delete_stale=bool(should_delete_stale),
        hash_same_size=bool(hash_same_size),
        dry_run=False,
    )

    print("Dataset cache ready:")
    print(f"  source={source_root.resolve()}")
    print(f"  target={cache_root.resolve()}")
    print(
        "  "
        f"files={stats.files_seen} unchanged={stats.unchanged} copied={stats.copied} "
        f"metadata_refreshed={stats.metadata_refreshed} deleted_stale={stats.deleted}"
    )
    return str(cache_root)


def stale_cached_paths(target_root: Path, video_ext: str, csv_ext: str, expected_targets: Sequence[Path]) -> List[Path]:
    expected = {path.resolve() for path in expected_targets}
    stale: List[Path] = []
    patterns = [f"run_*{video_ext}"]
    if csv_ext != video_ext:
        patterns.append(f"run_*{csv_ext}")

    for pattern in patterns:
        for path in target_root.glob(pattern):
            if path.is_file() and path.resolve() not in expected:
                stale.append(path)

    return sorted(set(stale))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mirror run_*.mp4/csv dataset pairs from the Windows-mounted dataset into a "
            "WSL-local cache directory for optimize_dali_loader.py, train.py, and train_inverse.py."
        )
    )
    parser.add_argument("--game", default=ACTION_SELECTED_GAME, help="Game name used when --data-root is omitted.")
    parser.add_argument("--data-root", default=None, help="Source dataset directory. Defaults to data/<game>.")
    parser.add_argument(
        "--target-root",
        default=None,
        help="Cached dataset directory to pass as --data-root. Defaults to ~/ai/dataset in WSL, ./dataset elsewhere.",
    )
    parser.add_argument("--video-ext", default=".mp4")
    parser.add_argument("--csv-ext", default=".csv")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument(
        "--delete-stale",
        dest="delete_stale",
        action="store_true",
        default=None,
        help="Delete cached run files that no longer exist in the source. Default: on for full syncs, off with --max-videos.",
    )
    parser.add_argument("--keep-stale", dest="delete_stale", action="store_false")
    parser.add_argument("--hash-same-size", dest="hash_same_size", action="store_true", default=True)
    parser.add_argument("--no-hash-same-size", dest="hash_same_size", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    game_name = normalize_game_name(args.game)
    source_root = Path(args.data_root if args.data_root is not None else game_data_root(game_name))
    target_root = Path(args.target_root) if args.target_root is not None else default_target_root()
    delete_stale = args.delete_stale if args.delete_stale is not None else args.max_videos is None

    stats = sync_dataset(
        source_root=source_root,
        target_root=target_root,
        video_ext=args.video_ext,
        csv_ext=args.csv_ext,
        max_videos=args.max_videos,
        delete_stale=bool(delete_stale),
        hash_same_size=bool(args.hash_same_size),
        dry_run=bool(args.dry_run),
    )

    print("Dataset sync complete:")
    print(f"  source={source_root.resolve()}")
    print(f"  target={target_root.resolve()}")
    print(f"  files_seen={stats.files_seen}")
    print(f"  unchanged={stats.unchanged}")
    print(f"  copied={stats.copied}")
    print(f"  metadata_refreshed={stats.metadata_refreshed}")
    print(f"  deleted_stale={stats.deleted}")
    print(f"  bytes_copied={stats.bytes_copied}")
    print(f"\nUse this for training/benchmarking: --data-root {target_root}")


if __name__ == "__main__":
    main()
