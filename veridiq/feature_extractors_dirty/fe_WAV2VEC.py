# aletheia
import pdb
import random

import h5py
import torch
import numpy as np
import pandas as pd
import os

from transformers import AutoFeatureExtractor, WavLMModel, Wav2Vec2Model
from tqdm import tqdm
import soundfile as sf
import json

class HuggingFaceFeatureExtractor:
    def __init__(self, model_class, name):
        self.device = "cuda"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(name)
        self.model = model_class.from_pretrained(name)
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, audio, sr):
        inputs = self.feature_extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            # padding=True,
            # max_length=16_000,
            # truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                # output_attentions=True,
                # output_hidden_states=False,
            )
        return outputs.last_hidden_state


FEATURE_EXTRACTORS = {
    "wav2vec2-base": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-base"
    ),
    "wav2vec2-large": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-large"
    ),
    "wav2vec2-large-lv60": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-large-lv60"
    ),
    "wav2vec2-large-robust": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-large-robust"
    ),
    "wav2vec2-large-xlsr-53": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-large-xlsr-53"
    ),
    "wav2vec2-xls-r-300m": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-xls-r-300m"
    ),
    "wav2vec2-xls-r-1b": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-xls-r-1b"
    ),
    "wav2vec2-xls-r-2b": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-xls-r-2b"
    ),
    "wavlm-base": lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "microsoft/wavlm-base"
    ),
    "wavlm-base-sv": lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "microsoft/wavlm-base-sv"
    ),
    "wavlm-base-plus": lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "microsoft/wavlm-base-plus"
    ),
    "wavlm-large": lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "microsoft/wavlm-large"
    ),
}
SAMPLING_RATE = 16_000

def main(
    split: str,
    feature_type: str
):
    def extract1(audio):
        feature = feature_extractor(audio, sr=SAMPLING_RATE)
        feature = feature.cpu().numpy().squeeze()
        return feature

    feature_extractor = FEATURE_EXTRACTORS[feature_type]()
    feature_type = feature_type.replace("wav2vec2-", "")

    #get data
    if split == "test":
        file_paths_root = "/data/av-deepfake-1m/av_deepfake_1m/val/"
        file_paths = pd.read_csv(f"/data/av-deepfake-1m/av_deepfake_1m/test_labels.csv")
        file_paths = file_paths["path"].tolist()
    else:
        # real only
        # file_paths_root = "/data/av-deepfake-1m/av_deepfake_1m/train/"
        # file_paths = pd.read_csv(f"/data/av-deepfake-1m/real_data_features/45k+5k_split/real_{split}_data.csv")
        # num_frames = file_paths["num_frames"].tolist()
        # file_paths = file_paths["path"].tolist()
        # real + fake
        file_paths_root = "/data/av-deepfake-1m/av_deepfake_1m/train/"
        file_paths = pd.read_csv(f"/data/av-deepfake-1m/av_deepfake_1m/{split}_labels.csv")
        # num_frames = file_paths["num_frames"].tolist()
        file_paths = file_paths["path"].tolist()

    # save_path_root = f"/data/av-deepfake-1m-features/real_data_features/45k+5k_split/WAV2VEC_features/{feature_type}/{split}/"
    save_path_root = f"/data/av1m-test/other/WAV2VEC_features/real+fake/{feature_type}/{split}/"
    if split == "test":
        with open("/data/av-deepfake-1m/av_deepfake_1m/val_metadata.json", "r") as f:
            data = json.load(f)
    else:
        with open("/data/av-deepfake-1m/av_deepfake_1m/train_metadata.json", "r") as f:
            data = json.load(f)
        
    all_audio_features = []
    for i, file_path in enumerate(tqdm(file_paths)):
        save_path = save_path_root + file_path.replace(".mp4", ".npy")
        audio, sr = sf.read(file_paths_root + file_path.replace(".mp4", ".wav"))
        assert sr == SAMPLING_RATE
        num_frames = next((item['video_frames'] for item in data if item['file'] == file_path), None)

        feature = extract1(audio)

        if split == "test":
            if len(feature) % 2 != 0:
                feature = np.vstack([feature, feature[-1]])
        else:
            if len(feature) % 2 != 0:
                # if num_frames[i] * 2 > len(feature):
                if num_frames * 2 > len(feature):
                    # Duplicate the last feature if needed to make the length even
                    feature = np.vstack([feature, feature[-1]])
                else:
                    feature = feature[:-1]  # Cut the last feature if it's uneven
                
        feature_new = feature.reshape(len(feature) // 2, 2, feature.shape[1]).reshape(len(feature) // 2, feature.shape[1]*2)
        if split != "test":
            # feature_new = feature_new[:num_frames[i]]
            feature_new = feature_new[:num_frames]
        
        if split != "test":
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, feature_new)
        else:
            all_audio_features.append(feature_new)

    if split == 'test':
        os.makedirs(os.path.dirname(save_path_root), exist_ok=True)
        all_audio_features = np.array(all_audio_features, dtype=object)
        np.save(os.path.join(save_path_root, "audio.npy"), all_audio_features)
        


if __name__ == "__main__":
    main("train", "wav2vec2-xls-r-2b")