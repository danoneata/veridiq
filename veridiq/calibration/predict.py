import sys
import click

from pathlib import Path

from torch.utils.data import DataLoader

from veridiq.linear_probing.train_test import test1, get_checkpoint_path_from_folder
from veridiq.linear_probing.datasets import AV1M_test_dataset, PerFileDataset
from veridiq.utils import implies


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
    },
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


def get_config(dataset_name, frame_level):
    return {
        **CONFIG,
        "root_path": FEAT_PATHS[FEATURE_TYPE][dataset_name],
        "dataset_name": DATASET_NAMES[dataset_name],
        "frame_level": frame_level,
        **DATA_PATHS[dataset_name],
        **OTHER_CONFIG[dataset_name],
    }


def predict(config_name, dataset_te, frame_level):
    config = get_config(dataset_te, frame_level)
    dataset = DATASETS[dataset_te](config)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1)
    folder = DIR_CKPTS / config_name
    path_checkpoint = get_checkpoint_path_from_folder(folder / "ckpts")
    path_output = folder / "predict" / "{}".format(dataset_te)
    test1(dataloader, path_checkpoint, path_output, frame_level)


@click.command()
@click.option(
    "-c",
    "--config-name",
    "config_name",
    type=str,
    required=True,
    help="Name of the config (folder in output/training-linear-probing).",
)
@click.option(
    "-d",
    "--dataset-te",
    "dataset_te",
    type=str,
    required=True,
    help="Name of the dataset to test (e.g., av1m, favc, avlips, bitdf).",
)
@click.option(
    "--frame-level",
    "frame_level",
    is_flag=True,
    help="Whether to do frame-level prediction (only supported for av1m).",
)
def main(config_name, dataset_te, frame_level: bool):
    assert implies(
        frame_level, dataset_te == "av1m"
    ), "Frame-level only supported for av1m."
    predict(config_name, dataset_te, frame_level)


if __name__ == "__main__":
    main()
