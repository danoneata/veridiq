import os

from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path

import click
import cv2
import h5py
import torch

from huggingface_hub import hf_hub_download
from toolz import partition_all
from torch import nn
from torchvision import transforms
from tqdm import tqdm
from transformers import (
    CLIPModel,
    CLIPProcessor,
    VideoMAEImageProcessor,
    VideoMAEModel,
)

import veridiq.fsfm.models_vit

from veridiq.data import DATASETS
from veridiq.utils import implies


DEVICE = "cuda"
FEATURE_DIR = Path("/data/deepfake-features")


def load_video_frames(video_path: str):
    capture = cv2.VideoCapture(video_path)
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame
    capture.release()


class FeatureExtractor(ABC, nn.Module):
    """Abstract base class for feature extractors."""

    def __init__(self):
        super().__init__()
        self.feature_dim = None

    @abstractmethod
    def transform(self, x):
        """Transform input images for the model."""
        pass

    @abstractmethod
    def get_features(self, x):
        """Extract features from transformed images."""
        pass

    def forward(self, x):
        return self.get_features(x)


class CLIP(FeatureExtractor):
    def __init__(self, model_id, tokens, layer):
        super().__init__()

        assert tokens in ["CLS", "patches"]
        assert implies(layer == "post-projection", tokens == "CLS")

        self.model = CLIPModel.from_pretrained(model_id).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

        self.model.to(DEVICE)
        self.model.eval()

        if layer == "post-projection":
            self.feature_dim = self.model.config.projection_dim
            self.get_features = self.get_image_features_post_projection
        else:
            self.feature_dim = self.model.config.vision_config.hidden_size
            self.get_features = partial(
                self.get_image_features_pre_projection,
                tokens=tokens,
                layer=layer,
            )

    def transform(self, x):
        output = self.processor(images=x, return_tensors="pt")
        output = output["pixel_values"]
        # output = output.squeeze(0)
        return output

    def get_features(self, x):
        # This method is required to satisfy the abstract base class.
        # It will be replaced by the appropriate method in __init__.
        raise NotImplementedError("get_features is set dynamically in __init__")

    def get_image_features_pre_projection(self, x, tokens, layer):
        outputs = self.model.vision_model(x, output_hidden_states=True)
        outputs = outputs.hidden_states[layer]
        outputs = outputs[:, :1] if tokens == "CLS" else outputs[:, 1:]
        outputs = outputs.mean(1)
        return outputs

    def get_image_features_post_projection(self, x):
        return self.model.get_image_features(x)

    def forward(self, x):
        return self.get_features(x)


class FSFM(FeatureExtractor):
    def __init__(self):
        super().__init__()
        self.model = self.load_model()
        self.transform1 = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def load_model(self):
        CKPT_NAME = "finetuned_models/FF++_c23_32frames/checkpoint-min_val_loss.pth"
        CKPT_SAVE_PATH = "output/fsfm"
        hf_hub_download(
            repo_id="Wolowolo/fsfm-3c",
            filename=CKPT_NAME,
            local_dir=CKPT_SAVE_PATH,
        )

        model = veridiq.fsfm.models_vit.vit_base_patch16(
            global_pool="avg",
            num_classes=2,
        )
        checkpoint = torch.load(
            os.path.join(CKPT_SAVE_PATH, CKPT_NAME),
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])

        model.head = torch.nn.Identity()
        model.head_drop = torch.nn.Identity()

        model = model.to(DEVICE)
        model.eval()

        return model

    def transform(self, x):
        # Convert images to the format expected by the model.
        # The model expects images of shape (B, 3, 224, 224).
        x = [self.transform1(Image.fromarray(image)) for image in x]
        x = torch.stack(x, dim=0)
        return x.to(DEVICE)

    def get_features(self, x):
        return self.model.forward_features(x)


class VideoMAE(FeatureExtractor):
    def __init__(self, model_id):
        super().__init__()
        path = "MCG-NJU/videomae-large"
        self.processor = VideoMAEImageProcessor.from_pretrained(path)
        self.model = VideoMAEModel.from_pretrained(path)

        self.model.to(DEVICE)
        self.model.eval()

    def transform(self, x):
        # Convert images to the format expected by the model.
        x = [Image.fromarray(image) for image in x]
        return self.model.preprocess(x)

    def get_features(self, x):
        return self.model.forward_features(x)


class PerFrameFeatureExtractor:
    def __init__(self, get_model):
        self.batch_size = 16
        self.model = get_model()

    def get_features(self, video_path):
        def extract_batch(frames):
            frames = self.model.transform(frames)
            frames = frames.to(DEVICE)
            with torch.no_grad():
                return self.model.get_features(frames)

        features = [
            extract_batch(frames)
            for frames in partition_all(self.batch_size, load_video_frames(video_path))
        ]
        return torch.cat(features, dim=0)


FEATURE_EXTRACTORS = {
    "clip": partial(
        PerFrameFeatureExtractor,
        partial(
            CLIP, "openai/clip-vit-large-patch14", tokens="CLS", layer="post-projection"
        ),
    ),
    "fsfm": partial(PerFrameFeatureExtractor, FSFM),
    "video-mae": VideoMAE,
}


@click.command()
@click.option(
    "-d",
    "--dataset",
    "dataset_name",
    type=click.Choice(DATASETS.keys()),
)
@click.option(
    "-f",
    "--feature",
    "feature_name",
    type=click.Choice(FEATURE_EXTRACTORS.keys()),
)
def main(dataset_name, feature_name):
    dataset = DATASETS[dataset_name]
    feature_extractor = FEATURE_EXTRACTORS[feature_name]()

    path = FEATURE_DIR / dataset_name / (feature_name + ".h5")
    path.parent.mkdir(parents=True, exist_ok=True)

    videos = dataset.get_videos()

    with h5py.File(path, "w") as f:
        for video in tqdm(videos):
            features = feature_extractor.get_features(video["path"])
            features = features.cpu().numpy()

            group = f.create_group(video["name"])
            group.create_dataset("features", data=features)
            group.attrs["name"] = video["name"]
            group.attrs["path"] = video["path"]


if __name__ == "__main__":
    main()
