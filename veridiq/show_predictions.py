import gc
import pdb
import os
import random

from abc import ABC, abstractmethod
from functools import partial

from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image

from scipy.special import logsumexp
from sklearn.metrics import accuracy_score, roc_auc_score
from toolz import take, partition_all
from tqdm import tqdm


import cv2
import numpy as np
import streamlit as st
import torch

from veridiq.data import AV1M
from veridiq.extract_features import FeatureExtractor, load_video_frames
from veridiq.utils import cache_json, cache_np, implies

import veridiq.mymarkdown as md


DEVICE = "cuda"
DATASET = AV1M("val")

FEATURES_DIR = {
    "CLIP": Path("/data/av1m-test/other/CLIP_features/test"),
    "FSFM": Path("/data/audio-video-deepfake-3/FSFM_face_features/test_face_fix"),
    "VideoMAE": Path("/data/audio-video-deepfake-2/Video_MAE_large/test"),
}

SUBSAMPLING_FACTORS = {
    "CLIP": 1,
    "FSFM": 1,
    "VideoMAE": 2,
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


def load_test_metadata(feature_extractor_type):
    metadata = DATASET.load_filelist()
    path_to_metadata = {m["file"]: m for m in metadata}
    paths = load_test_paths(feature_extractor_type)
    return [path_to_metadata[p] for p in paths]


def load_test_features(feature_extractor_type):
    path = FEATURES_DIR[feature_extractor_type] / "video.npy"
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
    "CLIP": get_per_frame_labels_default,
    "FSFM": get_per_frame_labels_default,
    "VideoMAE": get_per_frame_labels_video_mae,
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
    st.markdown(
        "`{}` · label: {} · modify type: {}".format(
            datum["file"], label_str, datum["modify_type"]
        )
    )
    st.video(video_path)
    st.pyplot(fig)


def show_frames(preds, datum, my_grad_cam):
    frames = list(load_video_frames(datum))
    probas = pred_to_proba(preds)
    labels = get_per_frame_labels(datum)

    def get_info(i):
        label = labels[i]
        label_str = "fake" if label == 1 else "real"
        return md.ul(
            [
                "frame: {:d} · time: {:.1f}s".format(i, index_to_time(i)),
                "label: {}".format(label_str),
                "proba: {:.2f}".format(probas[i]),
            ]
        )

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
            explanation = explanation[0]
            # explanation = undo_image_transform_clip(frames[idx], explanation)

            cols[i].markdown("Grad-CAM · max score: {:.1f}".format(explanation.max()))
            explanation = my_grad_cam.show_cam_on_image(
                frames[idx] / 255,
                explanation,
                use_rgb=True,
            )
            cols[i].image(explanation)


@st.cache_resource
def get_grad_cam_model(feature_extractor_type="CLIP", **kwargs):
    return MyGradCAM(feature_extractor_type=feature_extractor_type, **kwargs)


def get_predictions_path(feature_extractor_type):
    return "output/{}-linear/predictions.npy".format(feature_extractor_type.lower())


def show_temporal_explanations():
    def argsort_with_none(xs, none_value):
        return np.argsort([x if x is not None else none_value for x in xs])

    SELECTIONS = {
        "first": lambda n: range(n),
        "random": lambda n: random.sample(range(num_videos), n),
        # videos with the highest or lowest per-frame scores
        "highest-preds": lambda n: np.argsort(preds_video)[-n:],
        "lowest-preds": lambda n: np.argsort(preds_video)[:n],
        # videos that are best or worst according to the video-level scores
        "best": lambda n: argsort_with_none(scores_video, -np.inf)[-n:],
        "worst": lambda n: argsort_with_none(scores_video, +np.inf)[:n],
    }

    with st.sidebar:
        feature_extractor_type = st.selectbox(
            "Feature Extractor",
            options=["CLIP", "FSFM", "VideoMAE"],
            index=2,
        )
        selection = st.selectbox(
            "Video selection",
            options=list(SELECTIONS.keys()),
            index=2,
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

    # Test feature extractor
    # feature_extractor = FEATURE_EXTRACTORS[feature_extractor_type]()
    # video = load_video_frames(metadata[0])
    # for frames in partition_all(16, video):
    #     frames = feature_extractor.transform(frames)
    #     frames = frames.to(DEVICE)
    #     with torch.no_grad():
    #         f = feature_extractor.get_image_features(frames)
    #     print(f)
    #     print(features0[0][:16])
    #     pdb.set_trace()

    auc = eval_video_level(preds_video, metadata)
    # feats, _ = select_rvra_or_fvfa(features0, metadata0)

    scores_video = eval_per_video(
        preds,
        metadata,
        feature_extractor_type=feature_extractor_type,
        to_binarize=False,
    )

    num_videos = len(preds)
    st.markdown("num. of selected videos: {}".format(num_videos))
    st.markdown("AUC: {:.2f}%".format(100 * auc))
    st.markdown("---")

    indices = SELECTIONS[selection](num_to_show)

    for i in indices:
        datum = metadata[i]
        pred = preds[i]
        show1(pred, datum, feature_extractor_type)
        st.markdown("---")


def show_spatial_explanations():
    st.set_page_config(layout="wide")

    SELECTIONS = {
        "random": lambda n: random.sample(range(num_videos), n),
        "first": lambda n: range(n),
        "best": lambda n: np.argsort(scores_video)[::-1][:n],
        "worst": lambda n: np.argsort(scores_video)[:n],
    }

    with st.sidebar:
        feature_extractor_type = st.selectbox(
            "Feature Extractor",
            options=["CLIP", "FSFM"],
            index=1,
        )
        selection = st.selectbox(
            "Video selection",
            options=list(SELECTIONS.keys()),
            index=2,
        )
        num_to_show = st.number_input(
            "Number of videos to show",
            min_value=1,
            value=16,
        )

    preds = np.load(get_predictions_path(feature_extractor_type), allow_pickle=True)
    metadata = load_test_metadata(feature_extractor_type)

    preds, metadata = select_fvfa(preds, metadata)
    scores_video = eval_per_video(preds, metadata)

    num_videos = len(preds)

    n_cols = 2
    indices = SELECTIONS[selection](num_to_show)
    my_grad_cam = get_grad_cam_model(feature_extractor_type=feature_extractor_type)

    for group in partition_all(n_cols, indices):
        cols = st.columns(n_cols)
        for i, col in zip(group, cols):
            datum = metadata[i]
            pred = preds[i]
            with col:
                show1(pred, datum, feature_extractor_type)
                show_frames(pred, datum, my_grad_cam)
                st.markdown("---")


def show_frames_classifier_maximization():
    st.set_page_config(layout="wide")

    with st.sidebar:
        feature_extractor_type = st.selectbox(
            "Feature Extractor",
            options=["CLIP", "FSFM"],
            index=0,
        )
        n_frames = st.number_input(
            "Number of frames to show",
            min_value=5,
            max_value=150,
            value=30,
            step=5,
        )
        maximize = st.selectbox(
            "Frame selection",
            options=["Highest scores", "Lowest scores"],
            index=0,
        )

    model_classifier = load_model_classifier(feature_extractor_type)

    metadata0 = load_test_metadata(feature_extractor_type)
    features0 = load_test_features(feature_extractor_type)
    path = get_predictions_path(feature_extractor_type)
    preds = cache_np(path, compute_predictions, model_classifier, features0)

    preds, metadata = select_rvra_or_fvfa(preds, metadata0)
    num_videos = len(preds)
    to_maximize = maximize == "Highest scores"
    subsampling_factor = SUBSAMPLING_FACTORS[feature_extractor_type]

    def get_frame_data_1(i):
        pred = preds[i]
        frame_idxs = np.argsort(pred)
        if to_maximize:
            frame_idxs = frame_idxs[::-1]
        frame_idxs = frame_idxs[:n_frames]
        frame_idxs = frame_idxs.tolist()
        return [
            {
                "video_idx": i,
                "frame_idx": frame_idx,
                "pred": pred[frame_idx].item(),
            }
            for frame_idx in frame_idxs
        ]

    def get_frame_data_all():
        return [e for i in tqdm(range(num_videos)) for e in get_frame_data_1(i)]

    def get_path():
        return "output/frames-classifier-maximization-{}-{}.json".format(
            feature_extractor_type.lower(),
            "highest" if maximize == "Highest scores" else "lowest",
        )

    all_frame_data = cache_json(get_path(), get_frame_data_all)
    all_frame_data = sorted(
        all_frame_data,
        key=lambda x: x["pred"],
        reverse=to_maximize,
    )
    selected_frames = all_frame_data[:n_frames]

    def get_info(frame_data):
        video_idx = frame_data["video_idx"]
        frame_idx = frame_data["frame_idx"]

        datum = metadata[video_idx]
        labels = get_per_frame_labels(datum, subsampling_factor)
        label = labels[frame_idx]
        label_str = "fake" if label == 1 else "real" if label == 0 else "unknown"

        return md.ul(
            [
                # "video: `{}`".format(datum["file"]),
                "frame: {:d} · time: {:.1f}s".format(
                    frame_data["frame_idx"], index_to_time(frame_data["frame_idx"])
                ),
                "label: {}".format(label_str),
                "pred: {:.3f}".format(frame_data["pred"]),
            ]
        )

    def get_frame(datum):
        video_idx = datum["video_idx"]
        frame_idx = datum["frame_idx"]
        frames = list(load_video_frames(metadata[video_idx]))
        return frames[frame_idx]

    # Display frames in a grid
    n_cols = 5
    for group in partition_all(n_cols, selected_frames):
        cols = st.columns(len(group))
        for j, frame_data in enumerate(group):
            cols[j].markdown(get_info(frame_data))
            cols[j].image(get_frame(frame_data))


if __name__ == "__main__":
    show_temporal_explanations()
    # show_spatial_explanations()
    # show_frames_classifier_maximization()
