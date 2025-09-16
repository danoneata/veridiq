import json
import pdb
import random

import cv2
import h5py
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import torch

from matplotlib import pyplot as plt
from scipy.signal import find_peaks
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
    "av-hubert-v": Path("/data/av-deepfake-1m/av_deepfake_1m/avhubert_checkpoints/self_large_vox_433h/test_features"),
    "clip": Path("/data/av1m-test/other/CLIP_features/test"),
    "fsfm": Path("/data/audio-video-deepfake-3/FSFM_face_features/test_face_fix"),
    "videomae": Path("/data/audio-video-deepfake-2/Video_MAE_large/test"),
}

SUBSAMPLING_FACTORS = {
    "av-hubert-v": 1,
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
        e = e + 1
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
        e = e + 1
        labels[s:e] = 1

    return labels


GET_PER_FRAME_LABELS = {
    "av-hubert-v": get_per_frame_labels_default,
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


def get_prediction_figure(datum, feature_extractor_type="CLIP"):
    preds = datum["frame-preds"]
    subsampling_factor = SUBSAMPLING_FACTORS[feature_extractor_type]

    def show_fake_segment(ax, fake_segment):
        s = fake_segment[0]
        e = fake_segment[1]
        ax.axvspan(s, e, color="red", alpha=0.3)

    sns.set(style="white", font="Arial", context="poster")

    fig = plt.figure(figsize=(8, 4))
    gs = fig.add_gridspec(2, 1, hspace=0)
    axs = gs.subplots(sharex="col")
    # fig, axs = plt.subplots(figsize=(8, 5), nrows=2, sharex=True)
    # plt.subplots_adjust(hspace=0.0)
    probas = pred_to_proba(preds)
    indices = np.arange(len(preds))
    times = index_to_time(indices, subsampling_factor)

    axs[0].plot(times, preds)
    axs[0].set_ylabel("score")
    axs[0].set_ylim(-7.5, 7.5)

    axs[1].plot(times, probas)
    axs[1].set_ylim(0, 1.1)
    axs[1].set_ylabel("proba")
    axs[1].set_xlabel("time")

    for fake_segment in datum["fake_segments"]:
        for ax in axs:
            show_fake_segment(ax, fake_segment)

    axs[0].axhline(0.0, linestyle="--", color="gray")
    axs[1].axhline(0.5, linestyle="--", color="gray")

    MARKERS = {
        "idx-l": "v",
        "idx-c": "v",
        "idx-r": "v",
    }

    for idxs in datum["frames-to-show"]:
        for k, idx in idxs.items():
            if not (0 <= idx < len(preds)):
                continue
            axs[1].plot(
                times[idx],
                # probas[idx],
                1.05,
                MARKERS[k],
                color="black",
            )

    # fig.tight_layout()
    return fig


def show1(datum, feature_extractor_type):
    video_path = DATASET.get_video_path(datum["file"])
    fig = get_prediction_figure(datum, feature_extractor_type)
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


def get_frame_info(datum, i, subsampling_factor):
    pred = datum["frame-preds"][i]
    proba = datum["frame-probas"][i]
    label = datum["frame-labels"][i]
    label_str = "fake" if label == 1 else "real"
    t = index_to_time(i, subsampling_factor)
    return ul(
        [
            "frame: {:d} · time: {:.1f}s".format(i, t),
            "label: {}".format(label_str),
            "proba: {:.2f} · pred: {:.1f}".format(proba, pred),
        ]
    )


def show_frames(datum, feature_extractor_type):
    frames = list(load_video_frames_datum(datum))
    ssf = SUBSAMPLING_FACTORS[feature_extractor_type]

    def undo_ss(f):
        return f * ssf

    for idxs in datum["frames-to-show"]:
        cols = st.columns(3)

        for col, i in zip(cols, idxs.values()):
            # - i is the index in the predictions
            # - f is the index in the frames
            f = undo_ss(i)

            if not (0 <= f < len(frames)):
                continue

            col.markdown(get_frame_info(datum, i, ssf))
            col.image(frames[f])


def get_frames_to_show_fake_segments(datum, feature_extractor_type):
    subsampling_factor = SUBSAMPLING_FACTORS[feature_extractor_type]

    def do1(s, e):
        idx_s = time_to_index(s, subsampling_factor)
        idx_e = time_to_index(e, subsampling_factor)

        segment_scores = datum["frame-probas"][idx_s:idx_e]
        try:
            idx_c = np.argmax(segment_scores) + idx_s
        except ValueError:
            return None

        # Find thw two adjacent frames to the fake segment.
        idx_l = idx_s - 1
        idx_r = idx_e + 1
        return {
            "idx-l": idx_l,
            "idx-c": idx_c,
            "idx-r": idx_r,
        }

    results = [do1(s, e) for s, e in datum["fake_segments"]]
    results = [r for r in results if r is not None]
    return results


def get_frames_to_show_peaks(datum, *args, **kwargs):
    peaks, peaks_info = find_peaks(datum["frame-probas"], height=0.999, prominence=0.7)
    return [
        {
            "idx-l": peaks_info["left_bases"][i],
            "idx-c": peaks[i],
            "idx-r": peaks_info["right_bases"][i],
        }
        for i in range(len(peaks))
    ]


GET_FRAMES_TO_SHOW = {
    "fake segments": get_frames_to_show_fake_segments,
    "peaks": get_frames_to_show_peaks,
}


@st.cache_data
def load_data(feature_extractor_type):
    model_classifier = load_model_classifier(feature_extractor_type)

    metadata0 = load_test_metadata(feature_extractor_type)
    features0 = load_test_features(feature_extractor_type)

    path = get_predictions_path(feature_extractor_type)
    preds = cache_np(path, compute_predictions, model_classifier, features0)

    preds, metadata = select_rvra_or_fvfa(preds, metadata0)
    preds_video = [aggregate_preds(p) for p in preds]

    # feats, _ = select_rvra_or_fvfa(features0, metadata0)

    scores_video = eval_per_video(
        preds,
        metadata,
        feature_extractor_type=feature_extractor_type,
        to_binarize=False,
    )

    data = [
        {
            "frame-preds": preds[i],
            "frame-probas": pred_to_proba(preds[i]),
            "frame-labels": GET_PER_FRAME_LABELS[feature_extractor_type](metadata[i]),
            "video-pred": preds_video[i],
            "video-score": scores_video[i],
            **metadata[i],
        }
        for i in range(len(preds))
    ]

    return data


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
            options=FEATURES_DIR.keys(),
            index=0,
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
        frames_to_show_type = st.selectbox(
            "Frames to show",
            options=["fake segments", "peaks"],
            index=0,
        )

    data = load_data(feature_extractor_type)
    num_videos = len(data)

    get_frames_to_show_fake_segments = GET_FRAMES_TO_SHOW[frames_to_show_type]

    for datum in data:
        datum["frames-to-show"] = get_frames_to_show_fake_segments(
            datum, feature_extractor_type
        )

    preds_video = [datum["video-pred"] for datum in data]
    auc_clf = eval_video_level(preds_video, data)
    auc_loc_no_none = [
        datum["video-score"] for datum in data if datum["video-score"] is not None
    ]

    st.markdown(
        ul(
            [
                "num. of selected videos: {}".format(num_videos),
                "Classification: {:.2f}% AUC".format(100 * auc_clf),
                "Localization: {:.2f}% AUC (over {} videos)".format(
                    100 * np.mean(auc_loc_no_none),
                    len(auc_loc_no_none),
                ),
            ]
        )
    )
    st.markdown("---")

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
                    show1(datum, feature_extractor_type)
                with col2:
                    show_frames(datum, feature_extractor_type)
                st.markdown("---")
