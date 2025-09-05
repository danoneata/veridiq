from abc import ABC, abstractmethod
import pdb
import random

import cv2
import h5py
import pandas as pd
import streamlit as st

from toolz import partition_all

from veridiq.data import ExDDV
from veridiq.explanations.evaluate_spatial_explanations import (
    GetPredictionGradCAM,
    evaluate1,
)
from veridiq.extract_features import load_video_frames
from veridiq.explanations.generate_spatial_explanations import (
    MyGradCAM,
    get_exddv_videos,
    undo_image_transform_clip,
)
from veridiq.explanations.utils import SCORE_LOADERS


CONFIG_NAME = "exddv-clip"


@st.cache_resource
def load_gradcam():
    return MyGradCAM.from_config_name(CONFIG_NAME)


@st.cache_resource
def load_gradcam_file():
    path = "output/exddv/explanations/gradcam-{}.h5".format(CONFIG_NAME)
    return h5py.File(path, "r")


def load_gradcam_from_file(file, name, i):
    group_name = "{}-{:05d}".format(name, i)
    return file[group_name]["explanation"][...]


def add_location(frame, loc, color=(255, 0, 0)):
    # Convert relative coordinates to absolute coordinates.
    h, w, _ = frame.shape
    x = int(loc["x"] * w)
    y = int(loc["y"] * h)

    # Use for radius and thickness relative to the frame size.
    radius = int(0.05 * (h + w) / 2)
    thickness = int(0.005 * (h + w) / 2)

    return cv2.circle(
        frame,
        (x, y),
        radius=radius,
        color=color,
        thickness=thickness,
    )


class Sorter(ABC):
    @abstractmethod
    def __call__(self, data):
        pass

    @abstractmethod
    def __str__(self):
        pass


class NoSorter(Sorter):
    def __call__(self, data):
        return data

    def __str__(self):
        return "none"


class RandomSorter(Sorter):
    def __call__(self, data):
        return random.sample(data, len(data))

    def __str__(self):
        return "random"


class KeySorter(Sorter):
    def __init__(self, key, reverse=False):
        self.key = key
        self.reverse = reverse

    def __call__(self, data):
        return sorted(data, key=lambda x: x[self.key], reverse=self.reverse)

    def __str__(self):
        return "{}/{}".format(self.key, "desc" if self.reverse else "asc")


# class Filter(ABC):
#     @abstractmethod
#     def __call__(self, data):
#         pass

#     @abstractmethod
#     def __str__(self):
#         pass


# class FilterByKey(Filter):
#     def __init__(self, key, value):
#         self.key = key
#         self.value = value

#     def __call__(self, data):
#         return [item for item in data if item[self.key] == self.value]

#     def __str__(self):
#         return "{}={}".format(self.key, self.value)


if __name__ == "__main___":
    st.set_page_config(layout="wide")
    NUM_COLS = 3

    videos = get_exddv_videos()
    num_videos = len(videos)

    video_score_loader = SCORE_LOADERS["video"](CONFIG_NAME)
    frame_score_loader = SCORE_LOADERS["frame"](CONFIG_NAME)

    for v in videos:
        v["pred-score"] = video_score_loader(v["name"])
        frame_scores = [frame_score_loader(v["name"], c["frame-idx"]) for c in v["clicks"]]
        v["pred-score-frame-max"] = max(frame_scores)

    with st.sidebar:
        sorters = [
            NoSorter(),
            KeySorter("pred-score", reverse=True),
            KeySorter("pred-score", reverse=False),
            KeySorter("pred-score-frame-max", reverse=True),
            RandomSorter(),
        ]
        sorter = st.selectbox("Sort by", options=sorters, index=1)
        num_to_show = st.number_input(
            "Number of videos to show",
            min_value=1,
            max_value=num_videos,
            value=10,
            step=5,
        )

        st.markdown("---")
        st.markdown(
            "Showing only fake videos from the test split. The total number of such videos is {}.".format(
                num_videos
            )
        )

    videos = sorter(videos)

    # gradcam = load_gradcam()
    gradcam_file = load_gradcam_file()

    for video in videos[:10]:
        clicks = video["clicks"]

        clicks = sorted(clicks, key=lambda x: x["frame-idx"])
        frame_idxs = [click["frame-idx"] for click in clicks]

        frames = load_video_frames(video["path"])
        frames = [frame for i, frame in enumerate(frames) if i in frame_idxs]
        frames = [add_location(frame, click) for frame, click in zip(frames, clicks)]

        st.markdown(
            "### `{}` · score: {:.1f}".format(video["name"], video["pred-score"])
        )

        col1, col2, col3 = st.columns(num_cols)

        with col1:
            st.markdown("Video")
            st.markdown("")
            st.video(video["path"])
            st.markdown("Textual annotation:\n > {}".format(video["text"]))
            # st.markdown("Frame indices:\n > {}".format(", ".join(map(str, frame_idxs))))

        with col2:
            st.markdown("Frame annotations")
            for frame_idx, frame in zip(frame_idxs, frames):
                st.markdown("Frame: {}".format(frame_idx))
                st.image(frame)

        with col3:
            st.markdown("Explanations")
            for i, frame in enumerate(frames):
                # explanations = gradcam.get_explanation_batch([frame])
                # explanation = explanations[0]

                click = clicks[i]
                frame_idx = click["frame-idx"]

                explanation = load_gradcam_from_file(
                    gradcam_file,
                    video["name"],
                    frame_idx,
                )
                explanation = undo_image_transform_clip(frame, explanation)
                pos = GetPredictionGradCAM.get_position(frame, explanation)

                frame = frame / 255
                frame = MyGradCAM.show_cam_on_image(frame, explanation, use_rgb=True)
                frame = add_location(frame, pos, color=(0, 255, 0))

                error = evaluate1(click, pos)
                score = frame_score_loader(video["name"], frame_idx)

                st.markdown("Error: {:.3f} · Score: {:.3f}".format(error, score))
                st.image(frame)

        st.markdown("---")
