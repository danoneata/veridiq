import pdb

from functools import partial
from pathlib import Path
from typing import Optional

import click
import cv2
import h5py
import numpy as np
import torch
import yaml

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from transformers.image_transforms import get_resize_output_image_size

from torch import nn
from toolz import partition_all

from tqdm import tqdm

from veridiq.data import ExDDV
from veridiq.extract_features import (
    FeatureExtractor,
    DEVICE,
    FEATURE_EXTRACTORS,
    PerFrameFeatureExtractor,
    load_video_frames,
)
from veridiq.linear_probing.train_test import get_checkpoint_path


def reshape_transform(tensor, height=16, width=16):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # Bring the channels to the first dimension, # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


class LinearModel(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        return self.head(x)


def load_model_classifier(feature_extractor_type):
    PATHS = {
        "av-hubert-a": "/data/av-datasets/ckpts_linear_probing/ckpts/avh_audio/model-epoch=30.ckpt",
        "av-hubert-v": "/data/av-datasets/ckpts_linear_probing/ckpts/avh_video/model-epoch=71.ckpt",
        "clip": "output/clip-linear/model-epoch=98.ckpt",
        "fsfm": "output/fsfm-linear/model-epoch=98.ckpt",
        # "videomae": "output/videomae-linear/model-epoch=99.ckpt",
        "wav2vec": "/data/av-datasets/ckpts_linear_probing/ckpts/wav2vec/model-epoch=68.ckpt",
        "videomae": "/data/av-datasets/ckpts_linear_probing/ckpts/video_mae/model-epoch=99.ckpt",
    }
    DIMS = {
        "av-hubert-a": 1024,
        "av-hubert-v": 1024,
        "clip": 768,
        "fsfm": 768,
        "wav2vec": 2 * 1920,
        "videomae": 1024,
    }

    path = PATHS[feature_extractor_type]
    checkpoint = torch.load(path)

    dim = DIMS[feature_extractor_type]
    model = LinearModel(dim)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(DEVICE)
    return model


def load_model_classifier_from_config(config):
    path_checkpoint = get_checkpoint_path(config)
    checkpoint = torch.load(path_checkpoint)

    dim = config["model_hparams"]["feats_dim"]
    model = LinearModel(dim)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(DEVICE)
    return model


class FullModel(nn.Module):
    def __init__(self, feature_extractor: FeatureExtractor, model_linear):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.model_linear = model_linear

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x / torch.linalg.norm(x, axis=1, keepdims=True)
        x = self.model_linear(x)
        return x


# Dictionary mapping feature extractor types to their target layers
TARGET_LAYERS = {
    "clip": lambda model: [
        model.feature_extractor.model.vision_model.encoder.layers[-1].layer_norm1
    ],
    "fsfm": lambda model: [model.feature_extractor.model.blocks[-1]],
}

SIZES = {
    "clip": {
        "height": 16,
        "width": 16,
    },
    "fsfm": {
        "height": 14,
        "width": 14,
    },
}


def load_config(config_name):
    config_dir = Path("veridiq/linear_probing/configs")
    config_path = config_dir / (config_name + ".yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model_full(config):
    feature_extractor_type = config["data_info"]["feature_name"]
    feature_extractor = FEATURE_EXTRACTORS[feature_extractor_type]()
    assert isinstance(
        feature_extractor, PerFrameFeatureExtractor
    ), "Only PerFrameFeatureExtractor is currently supported."
    feature_extractor = feature_extractor.model
    model_classifier = load_model_classifier_from_config(config)

    model_full = FullModel(feature_extractor, model_classifier)
    model_full.eval()
    model_full.to(DEVICE)

    return model_full


class MyGradCAM:
    def __init__(self, config):
        feature_extractor_type = config["data_info"]["feature_name"]
        model_full = load_model_full(config)

        self.transform_images = model_full.feature_extractor.transform
        self.feature_extractor_type = feature_extractor_type

        target_layers = TARGET_LAYERS[feature_extractor_type](model_full)
        self.target = BinaryClassifierOutputTarget(1)

        self.grad_cam = GradCAM(
            model=model_full,
            target_layers=target_layers,
            reshape_transform=partial(
                reshape_transform,
                **SIZES[feature_extractor_type],
            ),
        )

    @classmethod
    def from_config_name(cls, config_name):
        config = load_config(config_name)
        return cls(config)

    def get_explanation_batch(self, images):
        n_images = len(images)

        images = self.transform_images(images)
        images = images.to(DEVICE)

        targets = [self.target for _ in range(n_images)]

        return self.grad_cam(
            input_tensor=images,
            targets=targets,
            eigen_smooth=None,
            aug_smooth=None,
        )

    def get_explanation_video(self, video):
        B = 16
        output = [
            self.get_explanation_batch(frames) for frames in partition_all(B, video)
        ]
        output = np.concatenate(output, axis=0)
        return output

    @staticmethod
    def show_cam_on_image(
        img: np.ndarray,
        mask: np.ndarray,
        use_rgb: bool = False,
        colormap: int = cv2.COLORMAP_JET,
        image_weight: float = 0.5,
        max_value: Optional[float] = None,
    ) -> np.ndarray:
        # Modified version from pytorch_grad_cam.utils.image.show_cam_on_image to allow for arbitrary maximum value.
        # https://github.com/jacobgil/pytorch-grad-cam/blob/781dbc0d16ffa95b6d18b96b7b829840a82d93d1/pytorch_grad_cam/utils/image.py#L35C1-L66C31
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
        if use_rgb:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255

        if np.max(img) > 1:
            raise Exception("The input image should np.float32 in the range [0, 1]")

        if image_weight < 0 or image_weight > 1:
            raise Exception(
                f"image_weight should be in the range [0, 1].\
                    Got: {image_weight}"
            )

        cam = (1 - image_weight) * heatmap + image_weight * img
        max_value = np.max(cam) if max_value is None else max_value
        cam = cam / max_value
        return np.uint8(255 * cam)


def get_exddv_videos():
    """Returns a subset of the video on ExDDV that we use for explainability
    purposes: fake-only videos from the test split.

    """
    return [
        video
        for video in ExDDV.get_videos()
        if video["split"] == "test" and video["label"] == "fake"
    ]


def get_exddv_images():
    for video in get_exddv_videos():
        frames = load_video_frames(video["path"])
        frame_idxs = [click["frame-idx"] for click in video["clicks"]]

        for i, frame in enumerate(frames):
            if i not in frame_idxs:
                continue

            yield {
                "frame": frame,
                "frame-idx": i,
                **video,
            }


def undo_image_transform_clip(frame, explanation):
    S = 224
    H, W = frame.shape[:2]
    size1 = get_resize_output_image_size(
        frame,
        size=S,
        default_to_square=False,
    )
    S1, S2 = explanation.shape
    assert S == S1 == S2
    dx = (size1[1] - S) // 2
    dy = (size1[0] - S) // 2
    explanation_out = np.zeros(size1, dtype=explanation.dtype)
    explanation_out[dy : dy + S, dx : dx + S] = explanation
    return cv2.resize(explanation_out, (W, H))


@click.command()
@click.option("-c", "--config", "config_name")
def main(config_name):
    my_grad_cam = MyGradCAM.from_config_name(config_name)
    path = "output/exddv/explanations/gradcam-{}.h5".format(config_name)
    with h5py.File(path, "w") as f:
        for image in tqdm(get_exddv_images()):
            explanation = my_grad_cam.get_explanation_batch([image["frame"]])
            explanation = explanation[0]
            group_name = "{}-{:05d}".format(image["name"], image["frame-idx"])
            group = f.create_group(group_name)
            group.create_dataset("explanation", data=explanation)


if __name__ == "__main__":
    main()
