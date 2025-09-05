import gc
import pdb
import os
import random

from abc import ABC, abstractmethod
from functools import partial

from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image

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

from veridiq.utils import mymarkdown as md


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
    show_frames_classifier_maximization()
