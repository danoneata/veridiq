import cv2
import os
import torchvision
from pipelines.data.data_module import AVSRDataLoader
from pipelines.detectors.mediapipe.detector import LandmarksDetector
from fractions import Fraction
import torch
from pipelines.model import AVSR
from tqdm import tqdm
import csv
import numpy as np
import shutil

def save2vid(filename, vid, frames_per_second):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # fps = float(frames_per_second)
    fps = Fraction(frames_per_second).limit_denominator()
    torchvision.io.write_video(filename, vid, fps)

def preprocess_video(src_filename, dst_filename):
    landmarks = landmarks_detector(src_filename)
    data = dataloader.load_data(src_filename, landmarks)
    
    fps_raw = cv2.VideoCapture(src_filename).get(cv2.CAP_PROP_FPS)
    fps = float(fps_raw) if fps_raw is not None else 25.0  # Default fallback
    # print("FPS:", fps, "Type:", type(fps))  # Debugging
    
    save2vid(dst_filename, data, fps)

def load_csv_paths(file_path):
    paths = set()
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                paths.add(row['path'])  # assuming 'path' is the column name
            except KeyError:
                try:
                    paths.add(row['full_path'])  # assuming 'full_path' is the column name
                except KeyError:
                    paths.add(row['full_file_path'])  # assuming 'full_file_path' is the column name

    return paths

def get_machine_indices(machine_id, start_idx=0, end_idx=50000, num_machines=1):
    total_elements = end_idx - start_idx
    elements_per_machine = total_elements // num_machines
    
    start_index = start_idx + machine_id * elements_per_machine
    end_index = start_idx + (machine_id + 1) * elements_per_machine if machine_id < num_machines - 1 else end_idx
    
    return start_index, end_index

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

if __name__ == "__main__":
    machine_id = 0
    print(f"Machine id {machine_id}")
    SPLIT = "train"
    print("Split: ", SPLIT)
    None_counter = 0

    modality = "video"
    # AV1M
    # features_root_path = "/data/av-deepfake-1m/av_deepfake_1m/train/"
    # save_path = f"/data/audio-video-deepfake-3/ASR/preprocessed_audio/real/{SPLIT}/"
    # csv1_paths = load_csv_paths(f"/data/av-deepfake-1m/real_data_features/45k+5k_split/real_{SPLIT}_data.csv")
    
    # FAVC
    # features_root_path = "/data/av-datasets/datasets/FakeAVCeleb/"
    # save_path = f"/data/av-extracted-features/favc_auto_avsr_preprocessed/"
    # csv1_paths = load_csv_paths(f"/data/av-datasets/datasets/FakeAVCeleb_preprocessed/all_splits/train_split.csv")
    # csv2_paths = load_csv_paths(f"/data/av-datasets/datasets/FakeAVCeleb_preprocessed/all_splits/val_split.csv")
    # csv3_paths = load_csv_paths(f"/data/av-datasets/datasets/FakeAVCeleb_preprocessed/all_splits/test_split.csv")

    # BitDF
    # features_root_path = "/data/veridiq-shared-pg/dataset/filtered_tracks/"
    # save_path = f"/data/av-extracted-features/bitdf_auto_avsr_preprocessed/"
    # csv1_paths = load_csv_paths(f"/data/veridiq-shared-pg/dataset/filtered_tracks_processed/metadata.csv")

    # AVLips
    features_root_path = "/data/avlips/AVLips/"
    save_path = f"/data/av-extracted-features/avlips_auto_avsr_preprocessed/"
    csv1_paths = load_csv_paths("/data/avlips/AVLips/test_labels.csv")

    # csv1_paths = load_csv_paths(f"/data/av-deepfake-1m/av_deepfake_1m/{SPLIT}_labels.csv")
    # csv2_paths = load_csv_paths(f'/data/av-deepfake-1m/real_data_features/45k+5k_split/real_{SPLIT}_data.csv')
    
    files = sorted(set(csv1_paths)) #.union(csv2_paths, csv3_paths))
    print(f"Saving at {save_path}")

    start_index, end_index = get_machine_indices(machine_id, end_idx = len(files)+2, num_machines=1)
    print(start_index, end_index)

    landmarks_detector = LandmarksDetector()
    dataloader = AVSRDataLoader(modality=modality, speed_rate=1, transform=False, detector="mediapipe", convert_gray=False)

    for i, file_path in enumerate(tqdm(files[start_index:end_index])):
        # file_path = file_path.replace("FakeAVCeleb/", "")
        # file_path = file_path.replace("/feats/", "/videos/")
        save_video_path = save_path + file_path
        
        # file_path = replace_ethnicity(file_path)
        
        # if "socialmedia" not in file_path:
        #     continue
        # else:
        #     features_root_path = "/data/veridiq-shared-pg/dataset/filtered_tracks_processed/"


        original_video_path = features_root_path + file_path
        if os.path.isfile(save_video_path):
            continue
        try:
            preprocess_video(original_video_path, save_video_path)
        except Exception as e:
            print(e)
            try:
                os.makedirs(os.path.dirname(save_video_path), exist_ok=True)
                shutil.copy(original_video_path, save_video_path)
            except Exception as e:
                print(e)
                continue