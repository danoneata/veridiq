import sys

from pathlib import Path

from torch.utils.data import DataLoader

from veridiq.linear_probing.train_test import test1, get_checkpoint_path_from_folder
from veridiq.linear_probing.datasets import AV1M_trainval_dataset


DIR_CKPTS = Path("/data/av-datasets/ckpts_linear_probing/ckpts")
DIR_OUTPUT = Path("output/calbiration/av1m/valid")

ROOT_PATHS = {
    "clip": "/data/av1m-test/other/CLIP_features/real+fake/",
    "wav2vec": "/data/av1m-test/other/WAV2VEC_features/real+fake/xls-r-2b/",
    "video_mae": "/data/audio-video-deepfake-2/Video_MAE_large/",
    # "avh_audio": "/data/av-deepfake-1m/av_deepfake_1m/avhubert_checkpoints/self_large_vox_433h/val_features",
    # "avh_video": "/data/av-deepfake-1m/av_deepfake_1m/avhubert_checkpoints/self_large_vox_433h/val_features",
}

INPUT_TYPES = {
    "clip": "video",
    "wav2vec": "audio",
    "video_mae": "video",
    # "avh_audio": "audio",
    # "avh_video": "video",
}

CONFIG = {
    "csv_root_path": "/data/av-deepfake-1m/av_deepfake_1m/",
    "metadata_path": "/data/av-deepfake-1m/av_deepfake_1m/val_metadata.json",
    "fvfa_rvra_only": True,
    "apply_l2": True,
    "input_type": "video",
    "dataset_name": "AV1M",
    "trimmed": False,
    "frame_level": False,
}


def get_config(feature_type):
    config = CONFIG.copy()
    config["root_path"] = ROOT_PATHS[feature_type]
    config["input_type"] = INPUT_TYPES[feature_type]
    return config


def predict_on_validation(feature_type):
    config = get_config(feature_type)
    dataset = AV1M_trainval_dataset(config, split="val")
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1)
    path_checkpoint = get_checkpoint_path_from_folder(DIR_CKPTS / feature_type)
    path_output = DIR_OUTPUT / feature_type
    test1(dataloader, path_checkpoint, path_output)


feature_type = sys.argv[1]
predict_on_validation(feature_type)