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
    print("FPS:", fps, "Type:", type(fps))  # Debugging
    
    save2vid(dst_filename, data, fps)

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
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."
        landmarks = self.process_landmarks(data_filename, landmarks_filename)
        data = self.dataloader.load_data(data_filename, landmarks)
        with torch.no_grad():
            if isinstance(data, tuple):
                enc_feats = self.model.model.encode(data[0].to(self.device), data[1].to(self.device), extract_resnet_feats)
            else:
                enc_feats = self.model.model.encode(data.to(self.device), extract_resnet_feats)
        return enc_feats   

def load_csv_paths(file_path):
    paths = set()
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            paths.add(row['path'])  # assuming 'path' is the column name
    return paths

def get_machine_indices(machine_id, start_idx=0, end_idx=50000, num_machines=1):
    total_elements = end_idx - start_idx
    elements_per_machine = total_elements // num_machines
    
    start_index = start_idx + machine_id * elements_per_machine
    end_index = start_idx + (machine_id + 1) * elements_per_machine if machine_id < num_machines - 1 else end_idx
    
    return start_index, end_index

if __name__ == "__main__":
    machine_id = 0
    print(f"Machine id {machine_id}")
    SPLIT = "train"
    print("Split: ", SPLIT)
    None_counter = 0

    modality = "audio"
    # model_conf = "LRS3_V_WER19.1/model.json"  
    # model_path = "LRS3_V_WER19.1/model.pth"
    # pipeline = InferencePipeline(modality, model_path, model_conf, face_track=True)

    path_to_features_root = f"/data/audio-video-deepfake-3/ASR/preprocessed_audio/real/{SPLIT}/"
    print(f"Saving at {path_to_features_root}")

    csv1_paths = load_csv_paths(f"/data/av-deepfake-1m/real_data_features/45k+5k_split/real_{SPLIT}_data.csv")
    # csv1_paths = load_csv_paths(f"/data/av-deepfake-1m/av_deepfake_1m/{SPLIT}_labels.csv")
    # csv2_paths = load_csv_paths(f'/data/av-deepfake-1m/real_data_features/45k+5k_split/real_{SPLIT}_data.csv')
    files = sorted(list(csv1_paths))#.union(csv2_paths)))

    # files = ["id00017/OLguY5ofUrY/00043/real.mp4", "id00064/MnBv-hDLPWo/00259/real.mp4"]

    # files = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(VIDEOS_PATH) for f in filenames])
    # print(len(files))
    start_index, end_index = get_machine_indices(machine_id, end_idx = len(files)+2, num_machines=1)
    print(start_index, end_index)

    landmarks_detector = LandmarksDetector()
    dataloader = AVSRDataLoader(modality="video", speed_rate=1, transform=False, detector="mediapipe", convert_gray=False)

    for i, file_path in enumerate(tqdm(files[start_index:end_index])):
        original_video_path = "/data/av-deepfake-1m/av_deepfake_1m/train/" + file_path
        save_video_path = path_to_features_root + file_path
        if os.path.isfile(save_video_path):
            continue
        try:
            preprocess_video(original_video_path, save_video_path)
        except Exception as e:
            print(e)
            os.makedirs(os.path.dirname(save_video_path), exist_ok=True)
            shutil.copy(original_video_path, save_video_path)