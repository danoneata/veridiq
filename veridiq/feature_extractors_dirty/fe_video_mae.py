
from transformers import VideoMAEImageProcessor, VideoMAEModel
from decord import VideoReader, cpu
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model & processor
processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-large")
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-large").to(device)
model.eval()

def extract_framewise_videomae_features(video_path, processor, model, chunk_size=16, stride=16):
    # Read video frames
    vr = VideoReader(video_path, ctx=cpu())
    frames = vr.get_batch(range(len(vr))).asnumpy()  # shape: (T, H, W, 3)
    num_frames = len(frames)

    feature_dim = model.config.hidden_size  
    num_chunks = (num_frames - chunk_size) // stride + 1
    frame_features = np.zeros((num_chunks, 8, feature_dim))

    frame_idx = 0
    print(f"Extracting features from {num_frames} frames...")

    for i in range(0, num_frames - chunk_size + 1, stride):
        chunk = frames[i : i + chunk_size]
        frame_list = [Image.fromarray(f) for f in chunk]

        inputs = processor(frame_list, return_tensors="pt").to(device)
        pixel_values = inputs["pixel_values"]  # (1, 16, 3, 224, 224)

        with torch.no_grad():
            outputs = model(pixel_values)
            vv = outputs.last_hidden_state.unflatten(1, [8, 14, 14])

        frame_features[frame_idx, :, :] = vv.mean([2, 3]).cpu().squeeze().numpy()
        frame_idx += 1

    # frame_features = frame_features.reshape(-1, frame_features.size(-1)) # to output features in shape [seq_len, feat_dim]
    return frame_features

def replace_ethnicity(path: str) -> str:
    replacements = {
        "Caucasian_American": "Caucasian (American)",
        "Asian_East": "Asian (East)",
        "Asian_South": "Asian (South)",
        "Caucasian_European": "Caucasian (European)"
    }
    
    for old, new in replacements.items():
        if old in path:
            path = path.replace(old, new)
    return path

def main(split=None):
    #frame_features, num_frames = extract_framewise_videomae_features(video_path, processor, model)
    
    # AV1M
    # csv_path = f'/av1m/av_deepfake_1m/{split}_labels.csv'
    # output_root = f'/audio_video_deepfake_2/Video_MAE_large/{split}/'
    # input_root = f'/av1m/av_deepfake_1m/train/'

    # FAVC
    # csv_path = f'/data/av-datasets/datasets/FakeAVCeleb_preprocessed/all_splits/{split}_split.csv'
    # output_root = f'/data/av-extracted-features/favc_video_mae/'
    # input_root = f'/data/av-datasets/datasets/FakeAVCeleb/'

    # BitDF
    # csv_path = "/data/veridiq-shared-pg/dataset/filtered_tracks_processed/metadata.csv"
    # output_root = "/data/av-extracted-features/bitdf_videomae/"
    # input_root = "/data/veridiq-shared-pg/dataset/filtered_tracks/"

    # AVLips
    csv_path = "/data/avlips/AVLips/test_labels.csv"
    output_root = "/data/av-extracted-features/avlips_videomae/"
    input_root = "/data/avlips/AVLips/"

    csv_file = f"{csv_path}"
    df = pd.read_csv(csv_file)

    if split=="test":
        all_features = []
        paths = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            path_original = row["path"]
        except KeyError:
            try:
                path_original = row["full_path"].replace("FakeAVCeleb/", "")
            except KeyError:
                path_original = row["full_file_path"].replace("/feats/", "/videos/")
                
        # path = replace_ethnicity(path_original)
        out_path = os.path.join(output_root,  path_original[:-4]) 
        # if os.path.exists(out_path):
        #     continue

        # if "socialmedia" not in path_original:
        #     continue
        # else:
        #     input_root = "/data/veridiq-shared-pg/dataset/filtered_tracks_processed/"


        video_path_input = input_root + path_original # or path for FAVC
        try:
            features = extract_framewise_videomae_features(video_path_input, processor, model)
        except Exception as e:
            print(e)
            continue
            
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)

        np.savez_compressed(out_path + '.npz', visual=features)
        if split=="test":
            all_features.append(features)
            paths.append(path_original)

    if split=="test":
        all_features = np.array(all_features, dtype=object)
        np.save(os.path.join(output_root,  "test", "video.npy"),  all_features, allow_pickle=True)
        paths = np.array(paths, dtype=object)
        np.save(os.path.join(output_root,  "test", "paths.npy"),  paths, allow_pickle=True)

if __name__ == "__main__":
    # main(None)
    # main("train")
    # main("val")
    main("test")