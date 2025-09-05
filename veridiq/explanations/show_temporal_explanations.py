import pdb
import random

import cv2
import h5py
import numpy as np
import pandas as pd
import streamlit as st
import torch

from matplotlib import pyplot as plt
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score
from pathlib import Path
from tqdm import tqdm
from toolz import partition_all

from veridiq.data import AV1M
from veridiq.extract_features import load_video_frames
from veridiq.explanations.generate_spatial_explanations import load_model_classifier
from veridiq.utils import cache_np
from veridiq.utils.markdown import ul
from veridiq.utils.streamlit import (
    NoSorter,
    RandomSorter,
    KeySorter,
)


DEVICE = "cuda"
DATASET = AV1M("val")

FEATURES_DIR = {
    "clip": Path("/data/av1m-test/other/CLIP_features/test"),
    "fsfm": Path("/data/audio-video-deepfake-3/FSFM_face_features/test_face_fix"),
    "videomae": Path("/data/audio-video-deepfake-2/Video_MAE_large/test"),
}

SUBSAMPLING_FACTORS = {
    "clip": 1,
    "fsfm": 1,
    "videomae": 2,
}


def load_video_frames_datum(datum):
    video_path = DATASET.get_video_path(datum["file"])
    return load_video_frames(video_path)


def pred_to_proba(score):
    # The model was trained with the cross-entropy loss.
    # See /root/work/avh-align-linear/avh_sup/mlp.py L77.
    # The logits for the two classes were `-score` and `score`, where `score` is `self.head(x)`.
    # This means that the probability of the positive class is given by: sigmoid(2 * score).
    return 1.0 / (1.0 + np.exp(-2 * score))


def load_test_paths(feature_extractor_type):
    path = FEATURES_DIR[feature_extractor_type] / "paths.npy"
    paths = np.load(path, allow_pickle=True)
    return paths


def load_test_features(feature_extractor_type):
    path = FEATURES_DIR[feature_extractor_type] / "video.npy"
    features = np.load(path, allow_pickle=True)
    return features


def load_test_metadata(feature_extractor_type):
    metadata = DATASET.load_filelist()
    path_to_metadata = {m["file"]: m for m in metadata}
    paths = load_test_paths(feature_extractor_type)
    return [path_to_metadata[p] for p in paths]


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


def time_to_index(time, subsampling_factor=1):
    return int(time * FPS / subsampling_factor)


def index_to_time(index, subsampling_factor=1):
    return index * subsampling_factor / FPS


def eval_video_level(preds_video, metadata):
    pred = preds_video
    true = [get_label(m) for m in metadata]
    return roc_auc_score(true, pred)


def get_per_frame_labels_default(datum):
    n = datum["video_frames"]
    labels = np.zeros(n)
    for s, e in datum["fake_segments"]:
        s = time_to_index(s)
        e = time_to_index(e)
        labels[s:e] = 1
    return labels


def get_per_frame_labels_video_mae(datum):
    chunk_size = 16
    stride_size = 16
    num_features = 8
    subsampling_factor = 2

    n = datum["video_frames"]
    n = (n - chunk_size) // stride_size + 1
    n = n * num_features
    labels = np.zeros(n)

    for s, e in datum["fake_segments"]:
        s = time_to_index(s, subsampling_factor)
        e = time_to_index(e, subsampling_factor)
        labels[s:e] = 1

    return labels


GET_PER_FRAME_LABELS = {
    "clip": get_per_frame_labels_default,
    "fsfm": get_per_frame_labels_default,
    "videomae": get_per_frame_labels_video_mae,
}


def eval_per_video(preds, metadata, feature_extractor_type, to_binarize=True):
    def get_pred(pred):
        if to_binarize:
            return pred_to_proba(pred) > 0.5
        else:
            return pred

    get_per_frame_labels = GET_PER_FRAME_LABELS[feature_extractor_type]

    def eval1(pred, datum):
        pred = get_pred(pred)
        true = get_per_frame_labels(datum)

        n_pred = len(pred)
        n_true = len(true)

        diff = abs(n_pred - n_true)

        if diff > 2:
            print(diff)
            # print(n_pred)
            # print(n_true)
            print(datum)
            print()
            return None

        n = min(n_pred, n_true)
        true1 = true[:n]
        pred1 = pred[:n]

        if sum(true1) == 0:
            return None

        return roc_auc_score(true1, pred1)

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


def get_predictions_path(feature_extractor_type):
    return "output/{}-linear/predictions.npy".format(feature_extractor_type.lower())


def get_prediction_figure(preds, datum, feature_extractor_type="CLIP"):
    subsampling_factor = SUBSAMPLING_FACTORS[feature_extractor_type]

    def show_fake_segment(ax, fake_segment):
        s = fake_segment[0]
        e = fake_segment[1]
        ax.axvspan(s, e, color="red", alpha=0.3)

    fig, axs = plt.subplots(figsize=(10, 6), nrows=2, sharex=True)
    probas = pred_to_proba(preds)
    indices = np.arange(len(preds))
    times = index_to_time(indices, subsampling_factor)

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


def show1(preds, datum, feature_extractor_type):
    video_path = DATASET.get_video_path(datum["file"])
    fig = get_prediction_figure(preds, datum, feature_extractor_type)
    label = get_label(datum)
    label_str = "fake" if label == 1 else "real"
    video_score_str = (
        "{:.2f}".format(datum["video-score"])
        if datum["video-score"] is not None
        else "N/A"
    )
    st.markdown(
        "`{}` · label: {} · modify type: {} · pred: {:.2f} · video score: {}".format(
            datum["file"],
            label_str,
            datum["modify_type"],
            datum["video-pred"],
            video_score_str,
        )
    )
    st.video(video_path)
    st.pyplot(fig)


def show_frames(preds, datum, feature_extractor_type):
    frames = list(load_video_frames_datum(datum))
    probas = pred_to_proba(preds)

    get_per_frame_labels = GET_PER_FRAME_LABELS[feature_extractor_type]
    labels = get_per_frame_labels(datum)

    subsampling_factor = SUBSAMPLING_FACTORS[feature_extractor_type]

    def get_info(i):
        label = labels[i]
        label_str = "fake" if label == 1 else "real"
        return ul(
            [
                "frame: {:d} · time: {:.1f}s".format(i, index_to_time(i)),
                "label: {}".format(label_str),
                "proba: {:.2f}".format(probas[i]),
            ]
        )

    for s, e in datum["fake_segments"]:
        idx_s = time_to_index(s, subsampling_factor)
        idx_e = time_to_index(e, subsampling_factor)

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


if __name__ == "__main__":
    st.set_page_config(layout="wide")

    def argsort_with_none(xs, none_value):
        return np.argsort([x if x is not None else none_value for x in xs])

    sorters = [
        NoSorter(),
        # videos with the highest or lowest per-frame scores
        KeySorter("video-pred", reverse=True),
        KeySorter("video-pred", reverse=False),
        # videos that are best or worst according to the video-level scores
        KeySorter("video-score", reverse=True),
        KeySorter("video-score", reverse=False),
        RandomSorter(),
    ]

    with st.sidebar:
        feature_extractor_type = st.selectbox(
            "Feature Extractor",
            options=["clip", "fsfm", "videomae"],
            index=2,
        )
        sorter = st.selectbox(
            "Sort by",
            options=sorters,
            index=1,
        )
        num_to_show = st.number_input(
            "Number of videos to show",
            min_value=1,
            value=16,
        )

    model_classifier = load_model_classifier(feature_extractor_type)

    metadata0 = load_test_metadata(feature_extractor_type)
    features0 = load_test_features(feature_extractor_type)

    path = get_predictions_path(feature_extractor_type)
    preds = cache_np(path, compute_predictions, model_classifier, features0)

    preds, metadata = select_rvra_or_fvfa(preds, metadata0)
    preds_video = [aggregate_preds(p) for p in preds]

    auc = eval_video_level(preds_video, metadata)
    # feats, _ = select_rvra_or_fvfa(features0, metadata0)

    scores_video = eval_per_video(
        preds,
        metadata,
        feature_extractor_type=feature_extractor_type,
        to_binarize=False,
    )
    scores_video_no_none = [s for s in scores_video if s is not None]

    num_videos = len(preds)
    st.markdown(
        ul(
            [
                "num. of selected videos: {}".format(num_videos),
                "Classification: {:.2f}% AUC".format(100 * auc),
                "Localization: {:.2f}% AUC (over {} videos)".format(
                    100 * np.mean(scores_video_no_none),
                    len(scores_video_no_none),
                ),
            ]
        )
    )
    st.markdown("---")

    data = [
        {
            "frame-preds": preds[i],
            "video-pred": preds_video[i],
            "video-score": scores_video[i],
            **metadata[i],
        }
        for i in range(num_videos)
    ]

    if str(sorter).startswith("video-score"):
        data = [datum for datum in data if datum["video-score"] is not None]

    data = sorter(data)
    data = data[:num_to_show]

    num_cols = 1

    for group in partition_all(num_cols, data):
        cols = st.columns(num_cols)
        for col, datum in zip(cols, group):
            with col:
                col1, col2 = st.columns([1, 3])
                with col1:
                    show1(
                        datum["frame-preds"],
                        datum,
                        feature_extractor_type,
                    )
                with col2:
                    show_frames(
                        datum["frame-preds"],
                        datum,
                        feature_extractor_type,
                    )
                st.markdown("---")
