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
        try:
            landmarks = self.process_landmarks(data_filename, landmarks_filename)
        except Exception as e:
            print(e, " passing landmarks as None")
            landmarks = None
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
    # machine_id = 4
    # print(f"Machine id {machine_id}")
    SPLIT = "train"
    print("Split: ", SPLIT)
    None_counter = 0

    modality = "audio"
    model_conf = "LRS3_A_WER1.0/model.json"  
    model_path = "LRS3_A_WER1.0/model.pth"
    pipeline = InferencePipeline(modality, model_path, model_conf, face_track=True)

    path_to_features_root = f"/data/audio-video-deepfake/ASR_features/LRS3_A_WER1.0/real+fake/{SPLIT}/"
    print(f"Saving at {path_to_features_root}")
    path_to_crops = f"/data/audio-video-deepfake/ASR/preprocessed/real+fake/{SPLIT}/"

    # csv1_paths = load_csv_paths(f"/data/av-deepfake-1m/av_deepfake_1m/{SPLIT}_labels.csv")
    csv2_paths = load_csv_paths(f'/data/av-deepfake-1m/real_data_features/45k+5k_split/real_{SPLIT}_data.csv')
    files = sorted(list(csv2_paths))#.union(csv2_paths)))

    # files = ["id00017/OLguY5ofUrY/00043/real.mp4", "id00064/MnBv-hDLPWo/00259/real.mp4"]

    # files = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(VIDEOS_PATH) for f in filenames])
    # print(len(files))
    # start_index, end_index = get_machine_indices(machine_id, end_idx = len(files)+2, num_machines=5)
    # print(start_index, end_index)

    # landmarks_detector = LandmarksDetector()
    # dataloader = AVSRDataLoader(modality="video", speed_rate=1, transform=False, detector="mediapipe", convert_gray=False)
    for i, file_path in enumerate(tqdm(files)): #[start_index:end_index])):
        original_video_path = "/data/av-deepfake-1m/av_deepfake_1m/train/" + file_path
        # mouth_roi_path = original_video_path[:-4] + '_roi.mp4'

        save_path = os.path.join(path_to_features_root, file_path.replace(".mp4", ".npz"))
        if os.path.isfile(save_path):
            # continue
            x = np.load(save_path, allow_pickle=True)
            if x['audio'].any() != None:
                continue

        mouth_roi_path = path_to_crops + file_path
        # audio_path = original_video_path[:-4] + '.wav'
        # mouth_roi_path = mouth_roi_path.replace("/data/av1m-test/unzipped/test/", "/data/av1m-test-feats/preprocessed/")
        # audio_path = audio_path.replace("/data/av1m-test/unzipped/test/", "/data/av1m-test-feats/preprocessed/")

        try:
            # feature_vid = pipeline.extract_features(mouth_roi_path)
            # feature_vid = feature_vid.cpu().detach().numpy()
            feature_audio = pipeline.extract_features(original_video_path)
            feature_audio = feature_audio.cpu().detach().numpy()
        except Exception as e:
            print(e)
            # feature_vid = None
            None_counter += 1
            feature_audio = None
            # None_counter += 1
            # print(feature_vid)
            print(feature_audio)
            traceback.print_exc()
        
        save_dict = {
            # "visual": feature_vid,
            "audio": feature_audio,
            "path": file_path,
        }
        # save_path = os.path.join(path_to_features_root, file_path.replace(".mp4", ".npz"))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        np.savez(save_path, **save_dict)
        # with open(os.path.join(PATH_TO_FEATURES, os.path.join(PATH_TO_FEATURES, os.path.basename(original_video_path).replace(".mp4", ".pkl"))), 'wb') as f:
        #     pickle.dump(save_dict, f)
    
    print("none counter: ", None_counter)