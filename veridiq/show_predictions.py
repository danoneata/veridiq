import gc
import pdb
import random

from functools import partial
from typing import Optional

from matplotlib import pyplot as plt
from pathlib import Path

from scipy.special import logsumexp
from sklearn.metrics import accuracy_score, roc_auc_score
from toolz import take, partition_all
from tqdm import tqdm

from transformers import (
    CLIPModel,
    CLIPProcessor,
)

import cv2
import numpy as np
import streamlit as st
import torch

from torch import nn
from torch.nn import Module

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from data import AV1M
from utils import cache_np, implies
import mymarkdown as md


DEVICE = "cuda"
TEST_DIR = Path("/av1m-test/other/CLIP_features/test")
DATASET = AV1M("val")


class CLIP(nn.Module):
    def __init__(self, model_id, tokens, layer):
        super(CLIP, self).__init__()

        assert tokens in ["CLS", "patches"]
        assert implies(layer == "post-projection", tokens == "CLS")

        self.model = CLIPModel.from_pretrained(model_id).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

        if layer == "post-projection":
            self.feature_dim = self.model.config.projection_dim
            self.get_image_features = self.get_image_features_post_projection
        else:
            self.feature_dim = self.model.config.vision_config.hidden_size
            self.get_image_features = partial(
                self.get_image_features_pre_projection,
                tokens=tokens,
                layer=layer,
            )

    def transform(self, x):
        output = self.processor(images=x, return_tensors="pt")
        output = output["pixel_values"]
        output = output.squeeze(0)
        return output

    def get_image_features_pre_projection(self, x, tokens, layer):
        outputs = self.model.vision_model(x, output_hidden_states=True)
        outputs = outputs.hidden_states[layer]
        outputs = outputs[:, :1] if tokens == "CLS" else outputs[:, 1:]
        outputs = outputs.mean(1)
        return outputs

    def get_image_features_post_projection(self, x):
        return self.model.get_image_features(x)

    def forward(self, x):
        return self.get_image_features(x)


class LinearModel(Module):
    def __init__(self, dim=768):
        super().__init__()
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        return self.head(x)


class FullModel(Module):
    def __init__(self, model_features, model_linear):
        super().__init__()
        self.model_features = model_features
        self.model_linear = model_linear

    def forward(self, x):
        x = self.model_features(x)
        x = x / torch.linalg.norm(x, axis=1, keepdims=True)
        x = self.model_linear(x)
        return x


def load_video_frames(datum):
    video_path = DATASET.get_video_path(datum["file"])
    capture = cv2.VideoCapture(str(video_path))
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame
    capture.release()


def pred_to_proba(score):
    # The model was trained with the cross-entropy loss.
    # See /root/work/avh-align-linear/avh_sup/mlp.py L77.
    # The logits for the two classes were `-score` and `score`, where `score` is `self.head(x)`.
    # This means that the probability of the positive class is given by: sigmoid(2 * score).
    return 1.0 / (1.0 + np.exp(-2 * score))


def load_model():
    checkpoint = torch.load("output/clip-linear/model-epoch=98.ckpt")
    model = LinearModel()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(DEVICE)
    return model


def load_test_metadata():
    metadata = DATASET.load_filelist()
    path_to_metadata = {m["file"]: m for m in metadata}

    path = TEST_DIR / "paths.npy"
    paths = np.load(path, allow_pickle=True)

    return [path_to_metadata[p] for p in paths]


def load_test_features():
    path = TEST_DIR / "video.npy"
    features = np.load(path, allow_pickle=True)
    return features


def l2_normalize(features):
    return features / np.linalg.norm(features, axis=1, keepdims=True)


def compute_predictions(model, features):
    def compute1(features):
        features = np.array(features, dtype=np.float32)
        features = l2_normalize(features)
        features = torch.tensor(features)
        features = features.to(DEVICE)
        with torch.no_grad():
            preds = model(features)
            preds = preds.squeeze().cpu().numpy()
            return preds

    preds = [compute1(f) for f in tqdm(features)]
    preds = np.array(preds, dtype=object)
    return preds


def get_label(datum):
    is_fake_audio = len(datum["audio_fake_segments"]) > 0
    is_fake_video = len(datum["visual_fake_segments"]) > 0
    if is_fake_audio and is_fake_video:
        # if datum["modify_type"] != "both_modified":
        #     st.warning(datum["modify_type"])
        return 1
    elif not is_fake_audio and not is_fake_video:
        assert datum["modify_type"] == "real"
        return 0
    else:
        return None


FPS = 25


def time_to_index(time):
    return int(time * FPS)


def index_to_time(index):
    return index / FPS


def eval_video_level(preds_video, metadata):
    pred = preds_video
    true = [get_label(m) for m in metadata]
    return roc_auc_score(true, pred)


def get_per_frame_labels(datum):
    n = datum["video_frames"]
    labels = np.zeros(n)
    for s, e in datum["fake_segments"]:
        s = time_to_index(s)
        e = time_to_index(e)
        labels[s:e] = 1
    return labels


def eval_per_video(preds, metadata):
    def eval1(pred, datum):
        pred = pred_to_proba(pred) > 0.5
        true = get_per_frame_labels(datum)
        true = true[:len(pred)]
        return roc_auc_score(true, pred)

    return [eval1(p, m) for p, m in zip(preds, metadata)]


def aggregate_preds(preds):
    return logsumexp(preds)


def select_by_labels(preds, metadata, labels):
    preds_metadata = [(p, m) for p, m in zip(preds, metadata) if get_label(m) in labels]
    return zip(*preds_metadata)


def select_fvfa(preds, metadata):
    return select_by_labels(preds, metadata, labels=[1])


def select_rvra_or_fvfa(preds, metadata):
    return select_by_labels(preds, metadata, labels=[0, 1])


def get_prediction_figure(preds, datum):
    def show_fake_segment(ax, fake_segment):
        s = fake_segment[0]
        e = fake_segment[1]
        ax.axvspan(s, e, color="red", alpha=0.3)

    fig, axs = plt.subplots(figsize=(10, 6), nrows=2, sharex=True)
    probas = pred_to_proba(preds)
    indices = np.arange(len(preds))
    times = index_to_time(indices)

    axs[0].plot(times, preds)
    axs[0].set_ylabel("logit")

    axs[1].plot(times, probas)
    axs[1].set_ylim(0, 1)
    axs[1].set_ylabel("proba")
    axs[1].set_xlabel("time")

    for fake_segment in datum["fake_segments"]:
        for ax in axs:
            show_fake_segment(ax, fake_segment)

    return fig


def show1(preds, datum):
    video_path = DATASET.get_video_path(datum["file"])
    fig = get_prediction_figure(preds, datum)
    label = get_label(datum)
    label_str = "fake" if label == 1 else "real"
    st.markdown(
        "`{}` · label: {} · modify type: {}".format(
            datum["file"], label_str, datum["modify_type"]
        )
    )
    st.video(video_path)
    st.pyplot(fig)


def reshape_transform(tensor, height=16, width=16):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # Bring the channels to the first dimension, # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


class MyGradCAM:
    def __init__(self):
        model_features = CLIP(
            "openai/clip-vit-large-patch14",
            tokens="CLS",
            layer="post-projection",
        )
        model_classifier = load_model()

        model_full = FullModel(model_features, model_classifier)
        model_full.eval()
        model_full.to(DEVICE)

        self.transform_images = model_features.transform

        target_layers = [
            model_full.model_features.model.vision_model.encoder.layers[-1].layer_norm1
        ]
        self.target = BinaryClassifierOutputTarget(1)

        self.grad_cam = GradCAM(
            model=model_full,
            target_layers=target_layers,
            reshape_transform=reshape_transform,
        )

    def get_explanation_batch(self, images):
        n_images = len(images)

        images = self.transform_images(images)
        images = images.to(DEVICE)

        if n_images == 1:
            # This is neede because transform_images squeezes the batch
            # dimension if there is only one image.
            images = images.unsqueeze(0)

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

    def make_video(self, datum):
        path = Path("output/grad-cam")
        path = path / datum["file"]
        path = path.with_suffix(".mp4")
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            return path

        path_npy = path.with_suffix(".npy")
        frames = list(load_video_frames(datum))
        output = cache_np(path_npy, my_grad_cam.get_explanation_video, frames)

        video_writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"H264"),
            25,
            (frames[0].shape[1], frames[0].shape[0]),
        )
        max_value = np.max(output)
        for frame, cam in zip(frames, output):
            frame = frame / 255
            image = my_grad_cam.show_cam_on_image(
                frame,
                cam,
                use_rgb=True,
                max_value=max_value,
            )
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            video_writer.write(image)

        video_writer.release()
        return path


def show_frames(preds, datum, my_grad_cam):
    frames = list(load_video_frames(datum))
    probas = pred_to_proba(preds)
    labels = get_per_frame_labels(datum)

    def get_info(i):
        label = labels[i]
        label_str = "fake" if label == 1 else "real"
        return md.ul([
            "frame: {:d} · time: {:.1f}s".format(i, index_to_time(i)),
            "label: {}".format(label_str),
            "proba: {:.2f}".format(probas[i]),
        ])

    for s, e in datum["fake_segments"]:
        idx_s = time_to_index(s)
        idx_e = time_to_index(e)

        segment_scores = probas[idx_s:idx_e]
        idx_1 = np.argmax(segment_scores) + idx_s

        # Find thw two adjacent frames to the fake segment.
        idx_0 = idx_s - 1 if idx_s > 0 else None
        idx_2 = idx_e if idx_e < len(frames) else None

        idxs = [idx_0, idx_1, idx_2]
        cols = st.columns(3)

        for i in range(3):
            idx = idxs[i]
            if idx is None:
                continue

            cols[i].markdown(get_info(idx))
            cols[i].image(frames[idx])

        cols = st.columns(3)
        for i in range(3):
            idx = idxs[i]
            if idx is None:
                continue

            explanation = my_grad_cam.get_explanation_batch([frames[idx]])
            cols[i].markdown("Grad-CAM · max score: {:.1f}".format(explanation[0].max()))
            explanation = my_grad_cam.show_cam_on_image(frames[idx] / 255, explanation[0], use_rgb=True)
            cols[i].image(explanation)


@st.cache_resource
def get_grad_cam_model():
    return MyGradCAM()


def show_temporal_explanations():
    model_classifier = load_model()

    metadata = load_test_metadata()
    features = load_test_features()

    path = "output/clip-linear/predictions.npy"
    preds = cache_np(path, compute_predictions, model_classifier, features)

    preds, metadata = select_rvra_or_fvfa(preds, metadata)
    preds_video = [aggregate_preds(p) for p in preds]

    auc = eval_video_level(preds_video, metadata)
    scores_video = eval_per_video(preds, metadata)

    num_videos = len(preds)
    st.markdown("num. of selected videos: {}".format(num_videos))
    st.markdown("AUC: {:.2f}%".format(100 * auc))
    st.markdown("---")

    SELECTIONS = {
        "first": lambda n: range(n),
        "random": lambda n: random.sample(range(num_videos), n),
        # videos with the highest or lowest per-frame scores
        "highest-preds": lambda n: np.argsort(preds_video)[-n:],
        "lowest-preds": lambda n: np.argsort(preds_video)[:n],
        # videos that are best or worst according to the video-level scores
        # "best": lambda n: np.argsort(scores_video)[-n:],
        # "worst": lambda n: np.argsort(scores_video)[:n],
    }

    with st.sidebar:
        selection = st.selectbox(
            "Video selection",
            options=list(SELECTIONS.keys()),
        )
        num_to_show = st.number_input(
            "Number of videos to show",
            min_value=1,
            value=16,
        )

    indices = SELECTIONS[selection](num_to_show)

    for i in indices:
        datum = metadata[i]
        pred = preds[i]
        show1(pred, datum)
        st.markdown("---")


def show_spatial_explanations():
    st.set_page_config(layout="wide")

    preds = np.load("output/clip-linear/predictions.npy", allow_pickle=True)
    metadata = load_test_metadata()

    preds, metadata = select_fvfa(preds, metadata)
    scores_video = eval_per_video(preds, metadata)

    num_videos = len(preds)

    SELECTIONS = {
        "random": lambda n: random.sample(range(num_videos), n),
        "first": lambda n: range(n),
        # videos that are best or worst according to the video-level scores
        "best": lambda n: np.argsort(scores_video)[::-1][:n],
        "worst": lambda n: np.argsort(scores_video)[:n],
    }

    with st.sidebar:
        selection = st.selectbox(
            "Video selection",
            options=list(SELECTIONS.keys()),
        )
        num_to_show = st.number_input(
            "Number of videos to show",
            min_value=1,
            value=16,
        )

    n_cols = 2
    indices = SELECTIONS[selection](num_to_show)
    my_grad_cam = get_grad_cam_model()

    for group in partition_all(n_cols, indices):
        cols = st.columns(n_cols)
        for i, col in zip(group, cols):
            datum = metadata[i]
            pred = preds[i]
            with col:
                show1(pred, datum)
                show_frames(pred, datum, my_grad_cam)
                st.markdown("---")


if __name__ == "__main__":
    # show_temporal_explanations()
    show_spatial_explanations()
