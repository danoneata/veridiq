import argparse

import torch
import numpy as np
import pandas as pd
import os
import clip
from tqdm import tqdm
import cv2 
from PIL import Image


def load_video(path, preprocess):
    for i in range(3):
        try:
            cap = cv2.VideoCapture(path)
            frames = []
            while True:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    frames.append(preprocess(frame))
                else:
                    break
            frames = torch.stack(frames)
            return frames
        except Exception as e:
            print(f"failed loading {path} ({i} / 3), {e}")
            if i == 2:
                raise ValueError(f"Unable to load {path}, {e}")


def load_clip(src, model, preprocess):
    frames = load_video(src, preprocess)
    frames = frames.to(device)

    chunk_size = 32
    features_list = []
    for i in range(0, len(frames), chunk_size):
        frames_i = frames[i: i + chunk_size]  # Take a chunk of 32 frames (or less if at the end)
        features_i = model.encode_image(frames_i)
        features_i = features_i.cpu().detach()
        features_list.append(features_i)

    features = torch.cat(features_list).numpy()
    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Video feats extraction'
    )

    parser.add_argument('--in_root_path', required=True)
    parser.add_argument('--out_root_path', required=True)
    parser.add_argument('--csv_file', default=None)
    parser.add_argument('--feats_extracted', default='clip')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    if args.csv_file is None:
        files = os.listdir(args.in_root_path)
        df = pd.DataFrame({
            "path": np.array(files)
        })
    else:
        df = pd.read_csv(args.csv_file)
        if 'path' not in df.columns:
            df['path'] = df['full_file_path'].apply(lambda x: x.replace("/feats/", "/videos/"))
            # df['path'] = df['full_path'].apply(lambda x: x.replace("FakeAVCeleb/", ""))

    if args.feats_extracted == "clip":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model, preprocess = clip.load("ViT-L/14", device=device)
        except:
            # pdb.set_trace()
            model, preprocess = clip.load("/root/av-fake-detection/landmark_prediction/preprocessing/clip/ViT-L-14.pt", device=device)
        model.eval()

    if args.test:
        paths = []
        videos = []

    for idx, row in tqdm(df.iterrows()):
        src = os.path.join(args.in_root_path, row['path'][:-4] + ".mp4")
        dst = os.path.join(args.out_root_path, row['path'][:-4] + ".npy")

        if args.feats_extracted == "clip":
            try:
                video_feats = load_clip(src, model, preprocess)
            except Exception as e:
                print(e)
                print(f"ERROR OOM: {src}")
                continue
            if video_feats is None:
                continue
            if args.test:
                paths.append(row['path'])
                videos.append(video_feats)
            else:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                np.save(dst, video_feats)

    if args.test:
        os.makedirs(args.out_root_path, exist_ok=True)
        np.save(os.path.join(args.out_root_path, "paths.npy"), np.array(paths, dtype=object))
        np.save(os.path.join(args.out_root_path, "video.npy"), np.array(videos, dtype=object)) 
