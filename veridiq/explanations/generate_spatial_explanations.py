import pdb

from pathlib import Path
from typing import Optional

import click
import cv2
import numpy as np
import torch
import yaml

from functools import partial

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from torch import nn
from toolz import partition_all

from veridiq.data import ExDDV
from veridiq.extract_features import FeatureExtractor, DEVICE, FEATURE_EXTRACTORS, PerFrameFeatureExtractor, load_video_frames
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
        "clip": "output/clip-linear/model-epoch=98.ckpt",
        "fsfm": "output/fsfm-linear/model-epoch=98.ckpt",
        "videomae": "output/videomae-linear/model-epoch=99.ckpt",
    }
    DIMS = {
        "clip": 768,
        "fsfm": 768,
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
    "clip": lambda model: [model.feature_extractor.model.vision_model.encoder.layers[-1].layer_norm1],
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


class MyGradCAM:
    def __init__(self, config):
        feature_extractor_type = config["data_info"]["feature_name"]
        feature_extractor = FEATURE_EXTRACTORS[feature_extractor_type]()
        assert isinstance(feature_extractor, PerFrameFeatureExtractor), "Only PerFrameFeatureExtractor is currently supported."
        feature_extractor = feature_extractor.model
        model_classifier = load_model_classifier_from_config(config)

        model_full = FullModel(feature_extractor, model_classifier)
        model_full.eval()
        model_full.to(DEVICE)

        self.transform_images = feature_extractor.transform
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
        config_dir = Path("veridiq/linear_probing/configs")
        config_path = config_dir / (config_name + ".yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
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


@click.command()
@click.option("-c", "--config", "config_name")
def main(config_name):
    my_grad_cam = MyGradCAM.from_config_name(config_name)
    for image_data in get_exddv_images():
        explanation = my_grad_cam.get_explanation_batch([image_data["frame"]])
        break


if __name__ == "__main__":
    main()
