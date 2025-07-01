import pdb
import random

from matplotlib import pyplot as plt
from pathlib import Path

from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import numpy as np
import streamlit as st
import torch

from torch import nn
from torch.nn import Module

from data import AV1M
from utils import cache_np


DEVICE = "cuda"
TEST_DIR = Path("/av1m-test/other/CLIP_features/test")
DATASET = AV1M("val")


class LinearModel(Module):
    def __init__(self, dim=768):
        super().__init__()
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        return self.head(x)


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


def compute_predictions(features):
    def compute1(features):
        features = torch.tensor(features, dtype=torch.float32)
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


def eval_video_level(preds_video, metadata):
    pred = preds_video
    true = [get_label(m) for m in metadata]
    return roc_auc_score(true, pred)


def aggregate_preds(preds):
    return logsumexp(preds)


def select_rvra_or_fvfa(preds, metadata):
    preds_metadata = [
        (p, m)
        for p, m in zip(preds, metadata)
        if get_label(m) is not None
    ]
    return zip(*preds_metadata)


def get_prediction_figure(preds, datum):
    def time_to_index(time, fps):
        return int(time * fps)

    def index_to_time(index, fps):
        return index / fps

    # fps = get_fps(datum["file"])
    fps = 25

    def show_fake_segment(ax, fake_segment):
        s = fake_segment[0]
        # s = time_to_index(s, fps)
        e = fake_segment[1]
        # e = time_to_index(e, fps)
        ax.axvspan(s, e, color="red", alpha=0.3)

    fig, axs = plt.subplots(figsize=(10, 6), nrows=2, sharex=True)
    probas = pred_to_proba(preds)
    indices = np.arange(len(preds))
    times = index_to_time(indices, fps)

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
    st.markdown("`{}` · label: {} · modify type: {}".format(datum["file"], label_str, datum["modify_type"]))
    # st.write(datum)
    st.video(video_path)
    st.pyplot(fig)


model = load_model()
metadata = load_test_metadata()
features = load_test_features()

path = "output/clip-linear/predictions.npy"
preds = cache_np(path, compute_predictions, features)

preds, metadata = select_rvra_or_fvfa(preds, metadata)
preds_video = [aggregate_preds(p) for p in preds]

auc = eval_video_level(preds_video, metadata)

num_videos = len(preds)
st.markdown("num. of selected videos: {}".format(num_videos))
st.markdown("AUC: {:.2f}%".format(100 * auc))
st.markdown("---")

SELECTIONS = {
    "first": lambda n: range(n),
    "random": lambda n: random.sample(range(num_videos), n),
    "highest-scores": lambda n: np.argsort(preds_video)[-n:],
    "lowest-scores": lambda n: np.argsort(preds_video)[:n],
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


indices = SELECTIONS[selection](10)

for i in indices:
    datum = metadata[i]
    pred = preds[i]
    show1(pred, datum)
    st.markdown("---")