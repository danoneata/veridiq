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
import pandas as pd

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
    def __init__(self, modality, model_path, model_conf, detector="mediapipe", face_track=False, device="cuda"):
        super(InferencePipeline, self).__init__()
        self.device = device
        print(self.device)
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
    None_counter = 0

    PATH_TO_DIRECTORY = "/data/av-deepfake-1m/av_deepfake_1m/"
    PATH_TO_FEATURES = f"/data/avlips/ASR/real_data/test/"
    PATH_TO_LABELS = "/data/av-deepfake-1m/av_deepfake_1m/"
    split = "test" #args.split

    labels_paths_file_path = os.path.join(PATH_TO_LABELS, f"{split}_labels.csv")
    path_to_images_root = os.path.join(PATH_TO_DIRECTORY, f"val") # test videos are in val folder
    # path_to_features_root = os.path.join(PATH_TO_FEATURES, f"{split}_features")
    # print(f"Saving at {path_to_features_root}")

    av_info = pd.read_csv(labels_paths_file_path)

    modality = "video"
    model_conf = "LRS3_V_WER19.1/model.json"  
    model_path = "LRS3_V_WER19.1/model.pth"
    pipeline = InferencePipeline(modality, model_path, model_conf, face_track=True)

    path_to_features_root = f"/data/audio-video-deepfake-2/ASR_features/LRS3_V_WER19.1/real+fake/{split}/"
    print(f"Saving at {path_to_features_root}")

    # start_index, end_index = get_machine_indices(machine_id, end_idx = len(files)+2, num_machines=5)

    all_video, paths = [], []

    # for i, file_path in enumerate(tqdm(files[start_index:end_index])):
    for idx in tqdm(av_info.index):
        original_video_path = os.path.join(path_to_images_root, av_info["path"][idx])
        mouth_roi_path = original_video_path[:-4] + '_roi.mp4'

        try:
            feature_vid = pipeline.extract_features(mouth_roi_path)
            feature_vid = feature_vid.cpu().detach().numpy()
        except:
            feature_vid = None
            None_counter += 1

        all_video.append(feature_vid)
    #     save_dict = {
    #         "visual": feature_vid,
    #         "path": file_path,
    #     }
    #     save_path = os.path.join(path_to_features_root, file_path.replace(".mp4", ".npz"))
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)

    #     np.savez(save_path, **save_dict)
    # print("none counter: ", None_counter)


        # if counter >= 5000:
        #     break

    os.makedirs(path_to_features_root, exist_ok=True)
    all_video = np.array(all_video, dtype=object)

    # Save features and paths
    np.save(f'{path_to_features_root}/video.npy', all_video)

    # Print for more info
    print(f'{split} number of videos: {len(av_info.index)}')
    print(f'{split} shape of video features: {all_video.shape}')