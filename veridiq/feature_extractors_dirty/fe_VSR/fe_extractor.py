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
import traceback
import ffmpeg

class InferencePipeline(torch.nn.Module):
    def __init__(self, modality, model_path, model_conf, detector="mediapipe", face_track=False, device="cuda:0"):
        super(InferencePipeline, self).__init__()
        self.device = device
        # modality configuration
        self.modality = modality
        self.dataloader = AVSRDataLoader(modality, detector=detector)
        self.model = AVSR(modality, model_path, model_conf, rnnlm=None, rnnlm_conf=None, penalty=0.0, ctc_weight=0.1, lm_weight=0.0, beam_size=40, device=device)
        if face_track and self.modality in ["video", "audiovisual"]:
            self.landmarks_detector = LandmarksDetector()
        else:
            self.landmarks_detector = None


    def process_landmarks(self, data_filename, landmarks_filename):
        if self.modality == "audio":
            return None
        if self.modality in ["video", "audiovisual"]:
            landmarks = self.landmarks_detector(data_filename)
            return landmarks


    def forward(self, data_filename, landmarks_filename=None):
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."
        landmarks = self.process_landmarks(data_filename, landmarks_filename)
        data = self.dataloader.load_data(data_filename, landmarks)
        transcript = self.model.infer(data)
        return transcript

    def extract_features(self, data_filename, landmarks_filename=None, extract_resnet_feats=False):
        if type(data_filename) is list:    
            # 0 = audio; 1 = video
            assert os.path.isfile(data_filename[0]), f"data_filename: {data_filename[0]} does not exist."
            assert os.path.isfile(data_filename[1]), f"data_filename: {data_filename[1]} does not exist."
        else:
            assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."
        
        try:
            if type(data_filename) is list: 
                landmarks = self.process_landmarks(data_filename[1], landmarks_filename)
            else:
                landmarks = self.process_landmarks(data_filename, landmarks_filename)

        except Exception as e:
            print(e, " passing landmarks as None")
            landmarks = None
        data = self.dataloader.load_data(data_filename, landmarks)
        
        with torch.no_grad():
            if isinstance(data, tuple):
                enc_feats = self.model.model.encode(data[0].to(self.device), data[1].to(self.device), extract_resnet_feats

                # in case of OOM, inference by chunks
                # n = data[0].size(1)  # number of frames (time dimension)
                # chunks = []
                # for i in range(100):
                #     start = i * n // 100
                #     end = (i + 1) * n // 100
                #     v_chunk = data[0][:, start:end].to(self.device)
                #     a_chunk = data[1][start*640:end*640].to(self.device)  # adjust audio stride if needed
                #     chunk = self.model.model.encode(v_chunk, a_chunk, extract_resnet_feats)
                #     chunks.append(chunk.cpu())  # free GPU
                # enc_feats = torch.cat(chunks, dim=0)
            else:
                enc_feats = self.model.model.encode(data.to(self.device), extract_resnet_feats)
        
                # in case of OOM, inference by chunks
                # n = data.size()[1] #len(data)
                # chunks = []
                # # Split into 10 parts
                # for i in range(10):
                #     start = i * n // 10
                #     end = (i + 1) * n // 10
                #     chunk = self.model.model.encode(data[:, start:end].to(self.device), extract_resnet_feats)
                #     chunks.append(chunk.cpu())  # move to CPU to free GPU
                # # Concatenate all chunks along batch dimension
                # enc_feats = torch.cat(chunks, dim=0)
        return enc_feats     

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
                    paths.add(row['full_file_path'])  # assuming 'full_path' is the column name
    return paths

def get_machine_indices(machine_id, start_idx=0, end_idx=50000, num_machines=1):
    total_elements = end_idx - start_idx
    elements_per_machine = total_elements // num_machines
    
    start_index = start_idx + machine_id * elements_per_machine
    end_index = start_idx + (machine_id + 1) * elements_per_machine if machine_id < num_machines - 1 else end_idx
    
    return start_index, end_index

def main(split):
    SPLIT = split
    print("Split: ", SPLIT)
    None_counter = 0
    dataset = "AV1M"

    modality = "audiovisual"
    model_conf = "LRS3_AV_WER0.9/model.json"  
    model_path = "LRS3_AV_WER0.9/model.pth"
    pipeline = InferencePipeline(modality, model_path, model_conf, face_track=True)

    path_to_features_root = f"/data/av-extracted-features/{dataset}_auto_avsr/{modality}/"
    print(f"Saving at {path_to_features_root}")

    if dataset == "AV1M":
        path_to_crops = f"/data/audio-video-deepfake/ASR/preprocessed/real+fake/{SPLIT}"
        csv_paths = load_csv_paths(f"/data/av-deepfake-1m/av_deepfake_1m/{SPLIT}_labels.csv")
        # csv_paths = load_csv_paths(f'/data/av-deepfake-1m/real_data_features/45k+5k_split/real_{SPLIT}_data.csv') # real-only data
    elif dataset == "FAVC":
        if modality == "audio":
            path_to_crops = f"/data/av-datasets/datasets/FakeAVCeleb/" # get audio (.wav)
        elif modality == "video":
            path_to_crops = f"/data/av-extracted-features/favc_auto_avsr_preprocessed/"
        elif modality == "audiovisual":
            path_to_crops = (f"/data/av-datasets/datasets/FakeAVCeleb_preprocessed/",  f"/data/av-extracted-features/favc_auto_avsr_preprocessed/")
        csv_paths = load_csv_paths(f"/data/av-datasets/datasets/FakeAVCeleb_preprocessed/all_splits/{SPLIT}_split.csv")
    elif dataset == "BitDF":
        if modality == "audio":
            path_to_crops = f"/data/veridiq-shared-pg/dataset/filtered_tracks_processed/"
        elif modality == "video":
            path_to_crops = f"/data/av-extracted-features/bitdf_auto_avsr_preprocessed/"
        elif modality == "audiovisual":
            path_to_crops = (f"/data/veridiq-shared-pg/dataset/filtered_tracks_processed/", f"/data/av-extracted-features/bitdf_auto_avsr_preprocessed/")
        csv_paths = load_csv_paths(f"/data/veridiq-shared-pg/dataset/filtered_tracks_processed/metadata.csv")
    elif dataset == "AVLips":
        path_to_crops = "/data/avlips/AVLips/"
        csv_paths = load_csv_paths(f"/data/avlips/AVLips/test_labels.csv")

    files = sorted(list(csv_paths))

    for i, file_path in enumerate(tqdm(files)):
        if dataset == "FAVC":
            file_path = file_path.replace("FakeAVCeleb/", "")
        elif dataset == "AVLips": #sau BitDF
            file_path = file_path.replace("/feats/", "/videos/")
        
        if modality == "audio":
            file_path = file_path.replace(".mp4", ".wav").replace("/videos/", "/processed/")
            mouth_roi_path = os.path.join(path_to_crops, file_path)
            save_path = os.path.join(path_to_features_root, file_path.replace(".wav", ".npz"))
        elif modality == "video":
            mouth_roi_path = os.path.join(path_to_crops, file_path)
            save_path = os.path.join(path_to_features_root, file_path.replace(".mp4", ".npz"))
        elif modality == "audiovisual":
            mouth_roi_path = [None, None]
            mouth_roi_path[1] = os.path.join(path_to_crops[1], file_path)
            file_path = file_path.replace(".mp4", ".wav").replace("/videos/", "/processed/")
            mouth_roi_path[0] = os.path.join(path_to_crops[0], file_path)
            
            save_path = os.path.join(path_to_features_root, file_path.replace(".mp4", ".npz"))

        if os.path.isfile(save_path):
            continue

        # if "socialmedia" not in file_path:
        #     continue
        # else:
        #     file_paths_root = "/data/veridiq-shared-pg/dataset/filtered_tracks_processed/"

        try:
            # if there are not .wav extracted yet (or a few missing)
            # if not os.path.exists(mouth_roi_path) and modality=="audio":
            #     og_path = os.path.join(f"/data/veridiq-shared-pg/dataset/filtered_tracks/", file_path.replace("/processed/", "/videos/").replace(".wav", ".mp4"))
            #     ffmpeg.input(og_path).output(mouth_roi_path).run()
            #     print("used ffmpeg ", mouth_roi_path)
            feature = pipeline.extract_features(mouth_roi_path)
            feature = feature.cpu().detach().numpy()

        except Exception as e:
            print(e)
            None_counter += 1
            traceback.print_exc()
            continue

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, feature)

    print("none counter: ", None_counter)

if __name__ == "__main__":
    # main(None)
    main("train")
    main("val")
    main("test")
    