import argparse
import os
import glob
import torch
import cv2

from dataset_wsl_sync import sync_dataset_for_training
from models import DrivingVideoPolicy, ModelConfig

# Force CPU/GPU optimizations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen ResNet18 backbone features next to WSL-cached videos.")
    parser.add_argument("--data-root", default=None, help="Source dataset root. Defaults to ModelConfig.data_root.")
    parser.add_argument("--dataset-cache-root", default=None, help="WSL-local dataset cache root.")
    parser.add_argument("--feature-suffix", default="_features.pt")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--sync-dataset", dest="sync_dataset", action="store_true", default=True)
    parser.add_argument("--no-sync-dataset", dest="sync_dataset", action="store_false")
    parser.add_argument("--dataset-sync-delete-stale", dest="dataset_sync_delete_stale", action="store_true", default=None)
    parser.add_argument("--dataset-sync-keep-stale", dest="dataset_sync_delete_stale", action="store_false")
    parser.add_argument("--dataset-sync-hash-same-size", dest="dataset_sync_hash_same_size", action="store_true", default=True)
    parser.add_argument("--dataset-sync-no-hash-same-size", dest="dataset_sync_hash_same_size", action="store_false")
    parser.add_argument("--force", action="store_true", help="Recompute feature files that already exist in the WSL cache.")
    return parser.parse_args()


def cache_dataset_features():
    args = parse_args()

    # 1. Initialize configuration and load your model architecture
    cfg = ModelConfig(data_root=args.data_root)
    if args.sync_dataset:
        cfg.data_root = sync_dataset_for_training(
            data_root=cfg.data_root,
            target_root=args.dataset_cache_root,
            video_ext=cfg.video_ext,
            csv_ext=cfg.csv_ext,
            feature_suffix=str(args.feature_suffix),
            max_videos=args.max_videos,
            delete_stale=args.dataset_sync_delete_stale,
            hash_same_size=bool(args.dataset_sync_hash_same_size),
        )

    policy = DrivingVideoPolicy(cfg).to(device)
    policy.eval() # Ensure dropout is disabled during feature extraction
    
    # Locate all run videos in the local cache directory returned by dataset sync.
    video_files = sorted(glob.glob(os.path.join(cfg.data_root, f"run_*{cfg.video_ext}")))
    if args.max_videos is not None:
        video_files = video_files[: max(0, int(args.max_videos))]
    print(f"Reading videos from: {os.path.abspath(cfg.data_root)}")
    print(f"Found {len(video_files)} runs to compress into features.")

    for video_path in video_files:
        feature_path = os.path.splitext(video_path)[0] + str(args.feature_suffix)
        if os.path.exists(feature_path) and not args.force:
            print(f"Skipping existing cache: {os.path.basename(feature_path)}")
            continue
        print(f"Processing: {os.path.basename(video_path)}...")
        
        cap = cv2.VideoCapture(video_path)
        features_list = []
        
        with torch.inference_mode():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (cfg.model_size, cfg.model_size), interpolation=cv2.INTER_AREA)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor_frame = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

                norm_frame = policy._normalize_frames(tensor_frame)
                masked_frame = policy._apply_masks(norm_frame)
                feat = policy.spatial_encoder.extract_backbone_features(masked_frame)

                features_list.append(feat.cpu())
            
        cap.release()
        
        if features_list:
            # Stack all frames along time dimension: [Total_Frames, 512, H/32, W/32]
            all_features = torch.cat(features_list, dim=0)
            torch.save(all_features, feature_path)
            print(f"Saved: {os.path.basename(feature_path)} (Shape: {tuple(all_features.shape)})")

if __name__ == "__main__":
    cache_dataset_features()
