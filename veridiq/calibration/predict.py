import sys

from pathlib import Path

from torch.utils.data import DataLoader

from veridiq.linear_probing.train_test import test1, get_checkpoint_path_from_folder
from veridiq.linear_probing.datasets import AV1M_test_dataset, PerFileDataset


DIR_CKPTS = Path("output/training-linear-probing")

DATASET_TR = "av1m"
FEATURE_TYPE = "clip"


DATA_PATHS = {
    "av1m": {
        "csv_root_path": "/data/av-deepfake-1m/av_deepfake_1m/",
        "metadata_path": "/data/av-deepfake-1m/av_deepfake_1m/val_metadata.json",
    },
    "favc": {
        "csv_root_path": "/data/av-datasets/datasets/FakeAVCeleb_preprocessed/all_splits",
    },
    "avlips": {
        "csv_root_path": "/data/avlips/AVLips",
    },
    "bitdf": {
        "csv_root_path": "/data/veridiq-shared-pg/dataset/filtered_tracks_processed",
    }
}


FEAT_PATHS = {
    "clip": {
        "av1m": "/data/av1m-test/other/CLIP_features/test",
        "favc": "/data/av-extracted-features/favc_clip/test",
        "bitdf": "/data/av-extracted-features/bitdf_clip",
        "avlips": "/data/av-extracted-features/avlips_clip",
    },
}


CONFIG = {
    # "csv_root_path": "/data/av-deepfake-1m/av_deepfake_1m/",
    # "metadata_path": "/data/av-deepfake-1m/av_deepfake_1m/val_metadata.json",
    "input_type": "video",
    "fvfa_rvra_only": True,
    "apply_l2": True,
    "trimmed": False,
    "frame_level": False,
}


DATASETS = {
    "av1m": AV1M_test_dataset,
    "favc": AV1M_test_dataset,
    "bitdf": PerFileDataset,
    "avlips": PerFileDataset,
}


DATASET_NAMES = {
    "av1m": "AV1M",
    "favc": "FAVC",
    "bitdf": "BitDF",
    "avlips": "AVLips",
}


OTHER_CONFIG = {
    "av1m": {},
    "favc": {},
    "bitdf": {
        "files_to_remove": "veridiq/linear_probing/files_to_remove.txt",
    },
    "avlips": {},
}


def get_config(dataset_name):
    return {
        **CONFIG,
        "root_path": FEAT_PATHS[FEATURE_TYPE][dataset_name],
        "dataset_name": DATASET_NAMES[dataset_name],
        **DATA_PATHS[dataset_name],
        **OTHER_CONFIG[dataset_name],
    }


def predict(config_name, dataset_te):
    config = get_config(dataset_te)
    dataset = DATASETS[dataset_te](config)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1)
    folder = DIR_CKPTS / config_name
    path_checkpoint = get_checkpoint_path_from_folder(folder / "ckpts")
    path_output = folder / "predict" / "{}".format(dataset_te)
    test1(dataloader, path_checkpoint, path_output)


config_name = sys.argv[1]
dataset_te = sys.argv[2]
predict(config_name, dataset_te)
