import cv2
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


def extract_frames(video_output_pair):
    video_path, output_dir = video_output_pair
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Skip if all frames exist
    if all((output_dir / f"frame_{i:05d}.jpg").exists() for i in range(total_frames)):
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (224, 224))
        save_path = output_dir / f"frame_{frame_idx:05d}.jpg"
        cv2.imwrite(str(save_path), frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        frame_idx += 1

    cap.release()


if __name__ == "__main__":
    video_folder = Path("/data/audio-video-deepfake-2/AV1M_preprocessed/val")
    output_root = Path("/data/audio-video-deepfake-3/av1m_RGB_mouth/test")
    metadata_path = Path("/data/av-deepfake-1m/av_deepfake_1m/test_labels.csv")
    # "/data/av-deepfake-1m/real_data_features/45k+5k_split/real_train_data.csv"
    # "/data/av-deepfake-1m/av_deepfake_1m/train_labels.csv"

    metadata = pd.read_csv(metadata_path)
    paths = metadata['path'].tolist()

    tasks = []
    for video_file in tqdm(paths):
        video_path = (video_folder / video_file.replace(".mp4", "_roi.mp4")).resolve()
        relative_path = Path(video_file).parent
        output_dir = output_root / relative_path / video_path.stem.removesuffix('_roi')
        tasks.append((video_path, output_dir))

    # Use number of available CPU cores
    num_workers = min(multiprocessing.cpu_count(), 16)  # You can tune this number

    print(f"Running with {num_workers} workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(extract_frames, tasks), total=len(tasks)))
