
from transformers import VideoMAEImageProcessor, VideoMAEModel
from decord import VideoReader, cpu
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import os


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
    # csv_path = f'/data/av-deepfake-1m/av_deepfake_1m/{split}_labels.csv'
    # output_root = f'/data/av-extracted-features/av1m_auto_avsr_multimodal/'
    # input_root = f'/data/audio-video-deepfake-4/av-extracted-features-2/av1m_auto_avsr_multimodal/test/'

    # FAVC
    csv_path = f'/data/av-datasets/datasets/FakeAVCeleb_preprocessed/all_splits/{split}_split.csv'
    output_root = f'/data/av-extracted-features/favc_auto_avsr/audiovisual/'
    input_root = f'/data/av-extracted-features/favc_auto_avsr/audiovisual/'

    # BitDF
    # csv_path = "/data/veridiq-shared-pg/dataset/filtered_tracks_processed/metadata.csv"
    # output_root = "/data/av-extracted-features/bitdf_videomae/"
    # input_root = "/data/veridiq-shared-pg/dataset/filtered_tracks/"

    # AVLips
    # csv_path = "/data/avlips/AVLips/test_labels.csv"
    # output_root = "/data/av-extracted-features/avlips_videomae/"
    # input_root = "/data/avlips/AVLips/"

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
        # out_path = os.path.join(output_root,  path_original[:-4], ".npz") 
        
        # if os.path.exists(out_path):
        #     continue

        # if "socialmedia" not in path_original:
        #     continue
        # else:
        #     input_root = "/data/veridiq-shared-pg/dataset/filtered_tracks_processed/"


        video_path_input = input_root + path_original[:-4] + ".wav.npz"# or path for FAVC
    
        features = np.load(video_path_input, allow_pickle=True)['arr_0']
        
        # DONT NEED THESE FOR TEST
        # out_dir = os.path.dirname(out_path)
        # os.makedirs(out_dir, exist_ok=True)
        # np.savez_compressed(out_path + '.npz', visual=features)

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