import random

import cv2
import pandas as pd
import streamlit as st

from toolz import partition_all

from veridiq.data import ExDDV
from veridiq.extract_features import load_video_frames
from veridiq.explanations.generate_spatial_explanations import MyGradCAM

st.set_page_config(layout="wide")


def add_location(frame, click):
    # Convert relative coordinates to absolute coordinates.
    h, w, _ = frame.shape
    x = int(click["x"] * w)
    y = int(click["y"] * h)

    # Use for radius and thickness relative to the frame size.
    radius = int(0.05 * (h + w) / 2)
    thickness = int(0.005 * (h + w) / 2)

    return cv2.circle(
        frame,
        (x, y),
        radius=radius,
        color=(255, 0, 0),
        thickness=thickness,
    )


def load_predictions():
    df = pd.read_csv("output/training-linear-probing/exddv-clip/test/results.csv")
    df = df.rename(columns={"paths": "name", "scores": "pred-score", "labels": "label"})
    df = df.set_index("name")
    return df.to_dict(orient="index")


class Sorter:
    def __call__(self, videos):
        pass

    def __str__(self):
        pass


class NoSorter(Sorter):
    def __call__(self, videos):
        return videos

    def __str__(self):
        return "none"


class RandomSorter(Sorter):
    def __call__(self, videos):
        return random.sample(videos, len(videos))

    def __str__(self):
        return "random"


class KeySorter(Sorter):
    def __init__(self, key, reverse=False):
        self.key = key
        self.reverse = reverse

    def __call__(self, videos):
        return sorted(videos, key=lambda x: x[self.key], reverse=self.reverse)

    def __str__(self):
        return "{}/{}".format(self.key, "desc" if self.reverse else "asc")


videos = ExDDV.get_videos()
videos = [v for v in videos if v["split"] == "test" and v["label"] == "fake"]
num_videos = len(videos)
preds = load_predictions()

for v in videos:
    v["pred-score"] = preds[v["name"]]["pred-score"]

with st.sidebar:
    sorters = [
        NoSorter(),
        KeySorter("pred-score", reverse=True),
        KeySorter("pred-score", reverse=False),
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


@st.cache_resource
def load_gradcam():
    return MyGradCAM.from_config_name("exddv-clip")

num_cols = 3
videos = sorter(videos)
gradcam = load_gradcam()

for video in videos[:10]:
    clicks = video["clicks"]

    frame_idxs = [click["frame-idx"] for click in clicks]
    frames = load_video_frames(video["path"])
    frames = [add_location(frame, click) for frame, click in zip(frames, clicks)]

    st.markdown("### `{}` · score: {:.1f}".format(video["name"], video["pred-score"]))

    col1, col2, col3 = st.columns(num_cols)

    with col1:
        st.markdown("Video")
        st.video(video["path"])
        st.markdown("Textual annotation:\n > {}".format(video["text"]))

    with col2:
        st.markdown("Frame annotations")
        for frame in frames:
            st.image(frame)

    with col3:
        st.markdown("Explanations")
        for frame in frames:
            explanation, = gradcam.get_explanation_batch([frame])
            explanation = cv2.resize(explanation, (frame.shape[1], frame.shape[0]))
            frame = frame / 255
            frame = gradcam.show_cam_on_image(frame, explanation, use_rgb=True)
            st.image(frame)

    st.markdown("---")
