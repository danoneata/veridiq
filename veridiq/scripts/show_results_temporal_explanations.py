import pdb
import random
import sys

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from PIL import Image

from veridiq.utils.markdown import ul
from veridiq.utils.latex import image, macro, multicol, tabular
from veridiq.explanations.show_temporal_explanations import (
    get_frames_to_show_fake_segments,
    get_prediction_figure,
    index_to_time,
    load_data,
    load_video_frames_datum,
    SUBSAMPLING_FACTORS,
)


OUTDIR = Path("output/plots/explanations-temporal")


def make_table_examples():

    def do1(datum):
        feature_extractor_type = datum["feature-extractor-type"]
        ssf = SUBSAMPLING_FACTORS[feature_extractor_type]

        def undo_ss(f):
            return f * ssf

        def get_frames_info(f):
            i = undo_ss(f)
            label = datum["frame-labels"][f]
            label_str = "fake" if label == 1 else "real"
            proba = datum["frame-probas"][f]
            proba_str = "{:.1f}".format(proba)
            time = index_to_time(i, ssf)
            time_str = "{:.1f}s".format(time)
            return {
                "i": i,
                "t": time_str,
                "label": label_str,
                "proba": proba_str,
            }

        name = datum["file"].replace("/", "-")
        name = name.split(".")[0]
        path_plot = OUTDIR / "plot-scores-{}-{}.png".format(
            name, feature_extractor_type
        )
        path_plot = str(path_plot)

        fig = get_prediction_figure(datum, feature_extractor_type)
        fig.tight_layout()
        fig.savefig(str(path_plot), pad_inches=0)

        frames_to_show_ss = datum["frames-to-show"][0].values()
        frames_to_show = [undo_ss(f) for f in frames_to_show_ss]

        frames = list(load_video_frames_datum(datum))
        frames = [frames[i] for i in frames_to_show]
        path_frames = [
            str(OUTDIR / "frames-{}-{}-{}.png".format(name, feature_extractor_type, i))
            for i in range(len(frames))
        ]

        for frame, path in zip(frames, path_frames):
            Image.fromarray(frame).save(path)

        frames_info = [get_frames_info(f) for f in frames_to_show_ss]
        video_pred_str = "{:.2f}".format(datum["video-pred"])

        return {
            "plot-scores": path_plot,
            "frames": path_frames,
            "frames-info": frames_info,
            "feature-extractor-type": feature_extractor_type,
            "video-pred": video_pred_str,
        }

    FEATURE_EXTRACTORS = [
        "av-hubert-v",
        "clip",
        # "fsfm",
        "videomae",
    ]

    random.seed(44)

    data = [
        {**datum, "feature-extractor-type": f}
        for f in FEATURE_EXTRACTORS
        for datum in load_data(f)
    ]
    data = [datum for datum in data if datum["video-score"] is not None]

    for datum in data:
        datum["frames-to-show"] = get_frames_to_show_fake_segments(
            datum,
            datum["feature-extractor-type"],
        )

    def subset(data, f):
        return [d for d in data if d["feature-extractor-type"] == f]

    def pick(data):
        return max(data, key=lambda x: x["video-pred"])

    selected_videos = [
        {
            "file": "id01096/RPR5JA7_PVM/00088/fake_video_fake_audio.mp4",
            "feature-extractor-type": "av-hubert-v",
            "fake-segment-idx": 0,
        },
        {
            "file": "id08993/NMtZSqGb4DQ/00431/fake_video_fake_audio.mp4",
            "feature-extractor-type": "clip",
            "fake-segment-idx": 1,
        },
        {
            "file": "id07558/c7LpfmgkjXw/00044/fake_video_fake_audio.mp4",
            "feature-extractor-type": "videomae",
            "fake-segment-idx": 0,
        },
    ]

    # configs = [pick(subset(data, f)) for f in FEATURE_EXTRACTORS]
    # data_to_show = [do1(datum) for datum in configs]

    def matches(datum, selected):
        matches_file = datum["file"] == selected["file"]
        matches_feat = datum["feature-extractor-type"] == selected["feature-extractor-type"]
        return matches_file and matches_feat

    data_selected = [
        datum
        for datum in data
        if any(matches(datum, selected) for selected in selected_videos)
    ]

    for datum in data_selected:
        fake_segment_idx = next(
            selected["fake-segment-idx"]
            for selected in selected_videos
            if matches(datum, selected)
        )
        frames_to_show = datum["frames-to-show"][fake_segment_idx]
        datum["frames-to-show"] = [frames_to_show]

    data_to_show = [do1(datum) for datum in data_selected]

    def show_st(data_to_show):
        cols = st.columns(len(data_to_show))
        for col, datum in zip(cols, data_to_show):
            with col:
                st.markdown(datum["feature-extractor-type"])
                st.markdown(datum["video-pred"])
                st.image(datum["plot-scores"])
                colss = st.columns(3)
                for i in range(3):
                    with colss[i]:
                        st.image(datum["frames"][i])
                        st.markdown(
                            ul(
                                [
                                    datum["frames-info"][i]["t"],
                                    datum["frames-info"][i]["label"],
                                    datum["frames-info"][i]["proba"],
                                ]
                            )
                        )

    def show_latex(data_to_show):
        def get_path_rel(path):
            path = Path(path)
            path = path.relative_to("output/plots")
            path = "imgs/" + str(path)
            return path

        FEAT_NAMES = {
            "av-hubert-v": "AV-HuBERT (V)",
            "clip": "CLIP",
            "fsfm": "FSFM",
            "videomae": "VideoMAE",
        }

        def make_row_header_feature():
            return [
                multicol(
                    FEAT_NAMES[datum["feature-extractor-type"]],
                    3,
                    "c",
                )
                for datum in data_to_show
            ]

        def make_row_header_score():
            return [
                multicol(
                    "score: {}".format(datum["video-pred"]),
                    3,
                    "c",
                )
                for datum in data_to_show
            ]

        def make_row_header_feature_and_score():
            return [
                multicol(
                    macro(
                        "mycombp",
                        FEAT_NAMES[datum["feature-extractor-type"]],
                        datum["video-pred"],
                    ),
                    3,
                    "c",
                )
                for datum in data_to_show
            ]

        def make_row_plots():
            return [
                multicol(macro("myplot", get_path_rel(datum["plot-scores"])), 3, "c")
                for datum in data_to_show
            ]

        def make_row_frames():
            return [
                macro("myimg", get_path_rel(elem))
                for datum in data_to_show
                for elem in datum["frames"]
            ]

        def make_row_frame_infos(key):
            return [
                macro("my" + key, elem[key])
                for datum in data_to_show
                for elem in datum["frames-info"]
            ]

        def make_row_frame_infos_comb():
            return [
                macro(
                    "mycombf",
                    elem["label"],
                    elem["proba"],
                )
                for datum in data_to_show
                for elem in datum["frames-info"]
            ]

        table = [
            make_row_header_feature_and_score(),
            # make_row_header_feature(),
            # make_row_header_score(),
            make_row_plots(),
            make_row_frame_infos("t"),
            make_row_frames(),
            make_row_frame_infos_comb(),
        ]
        table = tabular(table)
        st.markdown("```\n" + table + "\n```")

    st.set_page_config(layout="wide")
    show_st(data_to_show)
    show_latex(data_to_show)


def main():
    make_table_examples()


if __name__ == "__main__":
    main()
