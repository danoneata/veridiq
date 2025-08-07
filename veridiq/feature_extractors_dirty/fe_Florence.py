import random

import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import cv2 

from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from transformers import AutoProcessor, AutoModelForCausalLM

class VideoDataset(Dataset):
    def __init__(self, video_paths, preprocess, frame_interval=1):
        self.video_paths = video_paths
        self.frame_interval = frame_interval
        self.preprocess = preprocess

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % self.frame_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                frames.append(self.preprocess(image))
            frame_id += 1
        
        cap.release()
        return torch.stack(frames) if frames else torch.empty(0), video_path  # Shape: (num_frames, 3, 224, 224)

def load_video(path, preprocess=None):
    for i in range(3):
        try:
            cap = cv2.VideoCapture(path)
            frames = []
            while True:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    if preprocess:
                        frames.append(preprocess(frame))
                    else:
                        frames.append(transforms.ToTensor()(frame))
                else:
                    break
            frames = torch.stack(frames)
            return frames
        except Exception as e:
            print(f"failed loading {path} ({i} / 3), {e}")
            if i == 2:
                raise ValueError(f"Unable to load {path}, {e}")

def main(
    split: str,
):
    model_id = "microsoft/Florence-2-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    #get data
    if split == "test":
        file_paths_root = "/data/av-deepfake-1m/av_deepfake_1m/val/"
        file_paths = pd.read_csv(f"/data/av-deepfake-1m/av_deepfake_1m/test_labels.csv")
        file_paths = file_paths["path"].tolist()
    else:
        file_paths_root = "/data/av-deepfake-1m/av_deepfake_1m/train/"
        file_paths = pd.read_csv(f"/data/av-deepfake-1m/real_data_features/45k+5k_split/real_{split}_data.csv")
        num_frames = file_paths["num_frames"].tolist()
        file_paths = file_paths["path"].tolist()

    if split == "test":
        save_path_root = f"/data/avlips/Florence/test_roi/"
    else:
        save_path_root = f"/data/avlips/Florence/real_data_roi/{split}/"

    all_video_features = []
    # dataset = VideoDataset(file_paths, preprocess)
    # dataloader = DataLoader(dataset, batch_size=1)
    # with torch.no_grad():
    #     for frames, file_path in dataloader:
    #         save_path = save_path_root + file_path[0].replace(".mp4", ".npy")
    #         if len(frames) == 0:
    #             continue  # Skip empty batches
    #         frames = frames.to(device)
    #         print(frames)
    #         features = model.encode_image(frames)

    #         if split != "test":
    #             print(features.size())
    #             # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #             # np.save(save_path, features)
    #         else:
    #             all_video_features.append(features.cpu().numpy())


    for i, file_path in enumerate(tqdm(file_paths)):
        file_path = file_path.replace(".mp4", "_roi.mp4")
        save_path = save_path_root + file_path.replace("_roi.mp4", ".npz")
        if os.path.exists(save_path):
            continue
        
        in_file_path = file_paths_root + file_path
        frames = load_video(in_file_path)
        frames = frames.to(device)

        # features_list = []
        # for i in range(10):
        #     frames_i = frames[i*len(frames)//10: (i+1)*len(frames)//10]
        #     features_i = model.encode_image(frames_i)
        #     features_i = features_i.cpu().detach()
        #     features_list.append(features_i)
        
        chunk_size = 4
        features_list = []
        for i in range(0, len(frames), chunk_size):
            frames_i = frames[i: i+chunk_size]  # Take a chunk of 64 frames (or less if at the end)
            images = [transforms.ToPILImage()(frame) for frame in frames_i]

            inputs = processor(images=images, return_tensors="pt").to(device)

            features_i = model._encode_image(inputs["pixel_values"])[:,0,:]
            features_i = features_i.cpu().detach()
            features_list.append(features_i)

        # features = torch.stack(features_list, dim=0).numpy()
        features = torch.cat(features_list).numpy()
        print(features.shape)
        # features = torch.cat(features_list).numpy()

        if split != "test":
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path, features)
        else:
            all_video_features.append(features)

    if split == 'test':
        os.makedirs(os.path.dirname(save_path_root), exist_ok=True)
        all_video_features = np.array(all_video_features, dtype=object)
        np.save(os.path.join(save_path_root, "video.npy"), all_video_features)
        
if __name__ == "__main__":
    main("val")