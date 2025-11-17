import pdb
import random
import sys

from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from matplotlib.patches import ConnectionPatch

from PIL import Image

from veridiq.utils.markdown import ul
from veridiq.utils.latex import image, macro, multicol, tabular
from veridiq.explanations.show_temporal_explanations import (
    get_frames_to_show_fake_segments,
    get_prediction_figure,
    index_to_time,
    load_data,
    load_video_frames_datum,
    pred_to_proba,
    SUBSAMPLING_FACTORS,
)


OUTDIR = Path("output/plots/explanations-temporal")
ROOT = Path("/data/av-deepfake-1m/av_deepfake_1m/val")

SR = 16_000
HOP_LENGTH = 512


def make_table_examples_a():

    def extract_melspectrogram(datum):
        path = ROOT / datum["file"]
        audio, _ = librosa.load(path, sr=SR)
        audio = audio.astype(np.float32)
        S = librosa.feature.melspectrogram(
            y=audio,
            sr=SR,
            n_fft=1024,
            hop_length=HOP_LENGTH,
            n_mels=128,
        )
        return librosa.power_to_db(S, ref=np.max)

    def add_melspectrogram_segment(ax, melspec, segment, segments_fake):
        def to_frame(t):
            return int(t * SR / HOP_LENGTH)

        def to_time(f):
            return f * HOP_LENGTH / SR

        s, e = segment
        s = to_frame(s)
        e = to_frame(e)
        ax.imshow(melspec[:, s:e], aspect="auto", origin="lower")
        xticks = ax.get_xticks()
        xticklabels = ["{:.1f}".format(to_time(s + t)) for t in xticks]
        ax.set_xticklabels(xticklabels)
        for seg_fake in segments_fake:
            s_fake = to_frame(seg_fake[0])
            e_fake = to_frame(seg_fake[1])
            if s_fake >= s and e_fake <= e:
                ax.axvline(s_fake - s, color="black", linestyle="--")
                ax.axvline(e_fake - s, color="black", linestyle="--")
                ax.plot(s_fake - s, 135, marker="v", color="black", clip_on=False)
                ax.plot(e_fake - s, 135, marker="v", color="black", clip_on=False)
        # ax.set_xlim(0, e - s)
        ax.set_ylim(0, 127)

    def add_melspectrogram_segments(fig, gs, datum):
        segments = datum["segments"]
        num_segments = len(segments)

        seg_durations = [e - s for s, e in segments]
        tot_duration = sum(seg_durations)
        width_ratios = [d / tot_duration for d in seg_durations]

        gs1 = gs.subgridspec(1, num_segments, width_ratios=width_ratios, wspace=0.05)
        axs = gs1.subplots(sharey=True, squeeze=False)

        melspec = extract_melspectrogram(datum)
        for i, segment in enumerate(segments):
            add_melspectrogram_segment(
                axs[0, i], melspec, segment, datum["fake_segments"]
            )

        return axs

    def get_prediction_figure(datum, feature_extractor_type="CLIP"):
        preds = datum["frame-preds"]
        subsampling_factor = SUBSAMPLING_FACTORS[feature_extractor_type]

        def show_fake_segment(ax, fake_segment):
            s = fake_segment[0]
            e = fake_segment[1]
            ax.axvspan(s, e, color="red", alpha=0.3)

        sns.set(style="white", font="Arial", context="poster")

        fig = plt.figure(figsize=(8, 6))
        gs_outer = fig.add_gridspec(2, 1, wspace=0, height_ratios=[4, 2])

        axs_bot = add_melspectrogram_segments(fig, gs_outer[1], datum)

        gs0 = gs_outer[0].subgridspec(2, 1, hspace=0)

        # gs = fig.add_gridspec(2, 1, hspace=0)
        axs_top = gs0.subplots(sharex="col")
        # fig, axs = plt.subplots(figsize=(8, 5), nrows=2, sharex=True)
        # plt.subplots_adjust(hspace=0.0)
        probas = pred_to_proba(preds)
        indices = np.arange(len(preds))
        times = index_to_time(indices, subsampling_factor)

        axs_top[0].plot(times, preds)
        axs_top[0].set_ylabel("score")
        axs_top[0].set_ylim(-7.5, 7.5)

        axs_top[1].plot(times, probas)
        axs_top[1].set_ylim(0, 1.1)
        axs_top[1].set_ylabel("proba")
        axs_top[1].set_xlabel("time")

        for fake_segment in datum["fake_segments"]:
            for ax in axs_top:
                show_fake_segment(ax, fake_segment)

        axs_top[0].axhline(0.0, linestyle="--", color="gray")
        axs_top[1].axhline(0.5, linestyle="--", color="gray")

        for seg in datum["fake_segments"]:
            for t in seg:
                axs_top[1].plot(
                    t,
                    1.05,
                    "v",
                    color="black",
                )

        # connect axes
        for i, segment in enumerate(datum["segments"]):
            xlims = axs_bot[0, i].get_xlim()
            con1 = ConnectionPatch(
                xyA=(segment[0], 0),
                coordsA=axs_top[1].transData,
                xyB=(xlims[0], 127),
                coordsB=axs_bot[0, i].transData,
                color="black",
            )
            con2 = ConnectionPatch(
                xyA=(segment[1], 0),
                coordsA=axs_top[1].transData,
                xyB=(xlims[1], 127),
                coordsB=axs_bot[0, i].transData,
                color="black",
            )
            fig.add_artist(con1)
            fig.add_artist(con2)

        # fig.tight_layout()
        return fig

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
        path_plot = OUTDIR / "plot-scores-melspec-{}-{}.png".format(
            name,
            feature_extractor_type,
        )
        path_plot = str(path_plot)

        fig = get_prediction_figure(datum, feature_extractor_type)
        fig.tight_layout()
        fig.savefig(str(path_plot), pad_inches=0)

        # frames_to_show_ss = datum["frames-to-show"][0].values()
        # frames_to_show = [undo_ss(f) for f in frames_to_show_ss]

        # frames_info = [get_frames_info(f) for f in frames_to_show_ss]
        video_pred_str = "{:.2f}".format(datum["video-pred"])

        return {
            "plot-scores": path_plot,
            # "frames-info": frames_info,
            "feature-extractor-type": feature_extractor_type,
            "video-pred": video_pred_str,
        }

    FEATURE_EXTRACTORS = [
        "av-hubert-a",
        "wav2vec",
    ]

    random.seed(44)

    data = [
        {**datum, "feature-extractor-type": f}
        for f in FEATURE_EXTRACTORS
        for datum in load_data(f)
    ]
    data = [datum for datum in data if datum["video-score"] is not None]

    selected_videos = [
        {
            "file": "id05256/BVlAlH9JHBw/00073/fake_video_fake_audio.mp4",
            "feature-extractor-type": "av-hubert-a",
            "segments": [
                (0.0, 0.25),
                (1.3, 1.8),
            ],
        },
        {
            "file": "id03854/vWv5GCtZMqE/00146/fake_video_fake_audio.mp4",
            "feature-extractor-type": "wav2vec",
            "segments": [
                (0, 0.75),
            ],
        },
    ]

    def matches(datum, selected):
        matches_file = datum["file"] == selected["file"]
        matches_feat = (
            datum["feature-extractor-type"] == selected["feature-extractor-type"]
        )
        return matches_file and matches_feat

    def find_match(datum, selected_videos):
        for selected in selected_videos:
            if matches(datum, selected):
                return selected
        return None

    data_selected = []
    for datum in data:
        match = find_match(datum, selected_videos)
        if match is not None:
            datum_new = {**datum, "segments": match["segments"]}
            data_selected.append(datum_new)

    data_to_show = [do1(datum) for datum in data_selected]

    def show_st(data_to_show):
        cols = st.columns(len(data_to_show))
        for col, datum in zip(cols, data_to_show):
            with col:
                st.markdown(datum["feature-extractor-type"])
                st.markdown(datum["video-pred"])
                st.image(datum["plot-scores"])

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

        def make_row_header_feature_and_score():
            return [
                multicol(
                    macro(
                        "mycombp",
                        FEAT_NAMES[datum["feature-extractor-type"]],
                        datum["video-pred"],
                    ),
                    3,
                    r"c@{\htc}",
                )
                for datum in data_to_show
            ]

        def make_row_plots():
            return [
                multicol(
                    macro("myplot", get_path_rel(datum["plot-scores"])),
                    3,
                    r"c@{\htc}",
                )
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

        table = [
            make_row_header_feature_and_score(),
            # make_row_header_feature(),
            # make_row_header_score(),
            make_row_plots(),
            make_row_frame_infos("t"),
            make_row_frames(),
            make_row_frame_infos("label"),
            make_row_frame_infos("proba"),
        ]
        table = tabular(table)
        st.markdown("```\n" + table + "\n```")

    st.set_page_config(layout="wide")
    show_st(data_to_show)
    # show_latex(data_to_show)


def make_table_examples_v():

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
        # "videomae",
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
        matches_feat = (
            datum["feature-extractor-type"] == selected["feature-extractor-type"]
        )
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

        def make_row_header_feature_and_score():
            return [
                multicol(
                    macro(
                        "mycombp",
                        FEAT_NAMES[datum["feature-extractor-type"]],
                        datum["video-pred"],
                    ),
                    3,
                    r"c@{\htc}",
                )
                for datum in data_to_show
            ]

        def make_row_plots():
            return [
                multicol(
                    macro("myplot", get_path_rel(datum["plot-scores"])),
                    3,
                    r"c@{\htc}",
                )
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

        table = [
            make_row_header_feature_and_score(),
            # make_row_header_feature(),
            # make_row_header_score(),
            make_row_plots(),
            make_row_frame_infos("t"),
            make_row_frames(),
            make_row_frame_infos("label"),
            make_row_frame_infos("proba"),
        ]
        table = tabular(table)
        st.markdown("```\n" + table + "\n```")

    st.set_page_config(layout="wide")
    show_st(data_to_show)
    show_latex(data_to_show)


def main():
    make_table_examples_a()
    # make_table_examples_v()


if __name__ == "__main__":
    main()
