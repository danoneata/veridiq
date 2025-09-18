import argparse
import os

import cv2
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import tqdm
import torch
from python_speech_features import logfbank
from transformers import AutoFeatureExtractor, WavLMModel, Wav2Vec2Model
import torchaudio


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
WAV2VEC_MODEL_NAME = "wav2vec2-xls-r-2b"


"""
    Taken from FACTOR
"""
def load_logfbank(path, vid_frames_no):
    def stacker(feats, stack_order):
        """
        Concatenating consecutive audio frames
        Args:
        feats - numpy.ndarray of shape [T, F]
        stack_order - int (number of neighboring frames to concatenate
        Returns:
        feats - numpy.ndarray of shape [T', F']
        """
        feat_dim = feats.shape[1]
        if len(feats) % stack_order != 0:
            res = stack_order - len(feats) % stack_order
            res = np.zeros([res, feat_dim]).astype(feats.dtype)
            feats = np.concatenate([feats, res], axis=0)
        feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order * feat_dim)
        return feats

    stack_order_audio: int=4
    wav_data, sample_rate = librosa.load(path, sr=16_000)
    assert sample_rate == 16_000 and len(wav_data.shape) == 1

    audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32)  # [T, F]
    audio_feats = stacker(audio_feats, stack_order_audio)  # [T/stack_order_audio, F*stack_order_audio]
    audio_feats = audio_feats.astype(np.float32)

    print(f"Audio: {audio_feats.shape[0]}; Video: {vid_frames_no}", flush=True)
    # residual = audio_feats.shape[0] - vid_frames_no
    # if residual > 0:
    #     audio_feats = audio_feats[:-residual, :]
    #     print(f'For video: {path}; no audio frames removed: {residual}')
    # elif residual < 0:
    #     raise ValueError("Residual is negative: more video frames compared to audio frames!")

    return audio_feats


def load_wav2vec(path, vid_frames_no, feature_extractor):
    def extract1(audio, feature_extractor):
        feature = feature_extractor(audio, sr=SAMPLING_RATE)
        feature = feature.cpu().numpy().squeeze()
        return feature

    # audio, sr = sf.read(path)
    # assert sr == SAMPLING_RATE
    # try:
    #     audio, sr = librosa.load(path, sr=SAMPLING_RATE)
    # except:
    #     print(f"ERROR: {path}")
    #     return None
    # assert sr == SAMPLING_RATE

    try:
        audio, sr = torchaudio.load(path)
    except:
        print(f"ERROR: {path}")
        return None

    # Convert to mono (if stereo)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    # Resample if needed
    if sr != SAMPLING_RATE:
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLING_RATE)(audio)
    audio = audio.squeeze(0)

    try:
        feature = extract1(audio, feature_extractor)
    except Exception as e:
        print(e)
        print(f"ERROR OOM: {path}")
        return None

    # if split == "test":
    #     if len(feature) % 2 != 0:
    #         feature = np.vstack([feature, feature[-1]])
    # else:
    if len(feature) % 2 != 0:
        # if vid_frames_no * 2 > len(feature):
            # Duplicate the last feature if needed to make the length even
        feature = np.vstack([feature, feature[-1]])
        # else:
        #     feature = feature[:-1]  # Cut the last feature if it's uneven

    feature_new = feature.reshape(len(feature) // 2, 2, feature.shape[1]).reshape(len(feature) // 2, feature.shape[1] * 2)
    feature_new = feature_new[:vid_frames_no]
    print(f"Audio: {feature_new.shape[0]}; Video: {vid_frames_no}; Sampling rate: {sr}", flush=True)
    if feature_new.shape[0] - vid_frames_no < 0:
        print(feature_new.shape[0] - vid_frames_no)
        print(path)

    return feature_new


def get_frames_no(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return total_frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Audio feats extraction'
    )

    parser.add_argument('--in_root_path', required=True)
    parser.add_argument('--vid_root_path', required=True)
    parser.add_argument('--out_root_path', required=True)
    parser.add_argument('--csv_file', default=None)
    parser.add_argument('--feats_extracted', default='logfbank')
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
            df['path'] = df['full_file_path'].apply(lambda x: x.replace("/feats/", "/processed/"))
            # df['path'] = df['full_path'].apply(lambda x: x.replace("FakeAVCeleb/", ""))

    if args.feats_extracted == "wav2vec":
        feature_type = WAV2VEC_MODEL_NAME
        feature_extractor = FEATURE_EXTRACTORS[feature_type]()
        feature_type = feature_type.replace("wav2vec2-", "")

    if args.test:
        paths = []
        audios = []

    for idx, row in tqdm.tqdm(df.iterrows()):
        src = os.path.join(args.in_root_path, row['path'][:-4] + ".wav")
        vid_src = os.path.join(args.vid_root_path, row['path'].replace('/processed/', '/videos/')[:-4] + ".mp4")
        dst = os.path.join(args.out_root_path, row['path'][:-4].replace('/processed/', '/videos/') + ".npy")

        frames_no = get_frames_no(vid_src)

        if args.feats_extracted == "logfbank":
            audio_feats = load_logfbank(src, frames_no)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            np.save(dst, audio_feats)
        elif args.feats_extracted == "wav2vec":
            audio_feats = load_wav2vec(src, frames_no, feature_extractor)
            if audio_feats is None:
                continue
            if args.test:
                paths.append(row['path'])
                audios.append(audio_feats)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            np.save(dst, audio_feats)

    if args.test:
        os.makedirs(args.out_root_path, exist_ok=True)
        np.save(os.path.join(args.out_root_path, "paths.npy"), np.array(paths, dtype=object))
        np.save(os.path.join(args.out_root_path, "audio.npy"), np.array(audios, dtype=object)) 
