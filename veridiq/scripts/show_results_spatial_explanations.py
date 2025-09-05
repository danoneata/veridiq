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
from tqdm import tqdm

from veridiq import mylatex
from veridiq.explanations.generate_spatial_explanations import (
    MyGradCAM,
    get_exddv_videos,
    undo_image_transform_clip,
)
from veridiq.explanations.evaluate_spatial_explanations import (
    evaluate1,
    GET_PREDICTORS,
    GetPredictionGradCAM,
    evaluate,
)
from veridiq.explanations.show_spatial_explanations import (
    add_location,
    load_gradcam_file,
    load_gradcam_from_file,
    load_video_frames,
)
from veridiq.explanations.utils import SCORE_LOADERS
from veridiq import mylatex
from veridiq.utils import cache_json, cache_image, transpose


CONFIG_NAME = "exddv-clip"


def load_results_gradcam(score_type):
    get_prediction_video_gradcam = GET_PREDICTORS["gradcam"]()
    score_loader = SCORE_LOADERS[score_type](CONFIG_NAME)

    return [
        {
            "loc-error": evaluate1(click, get_prediction_video_gradcam(video, click)),
            "pred-score": score_loader(video["name"], click["frame-idx"]),
        }
        for video in get_exddv_videos()
        for click in video["clicks"]
    ]


def get_loc_error_confident(results, thresh):
    results_confident = [r for r in results if r["pred-score"] >= thresh]
    errors = [r["loc-error"] for r in results_confident]
    return errors


def make_plot_mae_vs_thresh(score_type):
    assert score_type in SCORE_LOADERS

    path_cache = (
        "output/cache/results-spatial-explanations-gradcam-scores-{}.json".format(
            score_type
        )
    )
    results_gradcam = cache_json(path_cache, load_results_gradcam, score_type)
    results_baselines = {
        "center": evaluate("center"),
        "center-face": evaluate("center-face"),
        # from paper: https://arxiv.org/pdf/2503.14421 (Table 6, ViT)
        "click-model": 0.0553,
    }
    LABELS = {
        "center": "Frame center",
        "center-face": "Face center",
        "click-model": "Click model",
        "gradcam": "Explanations",
    }
    LINESTYLES = {
        "center": ":",
        "center-face": "-.",
        "click-model": "--",
    }

    thresholds = sorted(set([r["pred-score"] for r in results_gradcam]))
    results_gradcam_cum = [
        {
            "thresh": thresh,
            "loc-error": error,
        }
        for thresh in tqdm(thresholds)
        for error in get_loc_error_confident(results_gradcam, thresh)
    ]

    df = pd.DataFrame(results_gradcam_cum)

    sns.set(context="poster", style="whitegrid", font="Arial")
    fig, ax = plt.subplots(figsize=(6.4, 5.2))

    sns.lineplot(
        data=df,
        x="thresh",
        y="loc-error",
        label=LABELS["gradcam"],
        ax=ax,
    )
    palette = sns.color_palette()

    xmin = df["thresh"].min()
    xmax = df["thresh"].max()

    for i, (label, value) in enumerate(results_baselines.items(), start=1):
        color = palette[i]
        ax.axhline(
            value,
            # xmin=xmin,
            # xmax=xmax,
            color=color,
            linestyle=LINESTYLES[label],
            label=LABELS[label],
        )
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        title="",
        ncol=2,
        frameon=False,
        fontsize="small",
    )
    # ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Minimum fakeness score")
    ax.set_ylabel("MAE")

    fig.tight_layout()

    st.markdown("### Alignment to human annotations")
    st.pyplot(fig)

    fig.savefig(
        "output/plots/explainability-spatial-mae-vs-thresh.pdf",
        transparent=True,
    )


def make_table_examples():
    data_videos = get_exddv_videos()
    data_frames = [
        {
            **video,
            **click,
        }
        for video in data_videos
        for click in video["clicks"]
    ]

    random.seed(17)
    data_frames = random.sample(data_frames, 4)
    score_loader = SCORE_LOADERS["video"](CONFIG_NAME)
    gradcam_file = load_gradcam_file()

    for datum in data_frames:
        datum["score"] = score_loader(datum["name"])

    data_frames = sorted(data_frames, key=lambda x: x["score"], reverse=True)

    def get_image(datum, to_add_annotation):
        for i, frame in enumerate(load_video_frames(datum["path"])):
            if i == datum["frame-idx"]:
                if to_add_annotation:
                    frame = add_location(frame, datum)
                return frame
        assert False

    def get_explanation(frame, datum):
        explanation = load_gradcam_from_file(
            gradcam_file,
            datum["name"],
            datum["frame-idx"],
        )
        explanation = undo_image_transform_clip(frame, explanation)

        frame_inp = frame / 255
        frame_out = MyGradCAM.show_cam_on_image(frame_inp, explanation, use_rgb=True)

        pos = GetPredictionGradCAM.get_position(frame, explanation)
        frame_out = add_location(frame_out, pos, color=(0, 255, 0))

        true = dict(x=datum["x"], y=datum["y"])
        pred = pos
        error = evaluate1(true, pred)

        return frame_out, error

    def get_path_image(datum, suffix):
        return "output/plots/explanations/{}-{}-{}.png".format(
            datum["name"].replace("/", "_"),
            datum["frame-idx"],
            suffix,
        )

    def do1(datum):
        image_orig_path = get_path_image(datum, "orig")
        image_anno_path = get_path_image(datum, "anno")
        image_expl_path = get_path_image(datum, "expl")

        image_orig = cache_image(
            image_orig_path, get_image, datum, to_add_annotation=False
        )
        image_anno = cache_image(
            image_anno_path, get_image, datum, to_add_annotation=True
        )

        image_expl, error = get_explanation(image_orig, datum)
        Image.fromarray(image_expl).save(image_expl_path)

        name = datum["name"]
        score = score_loader(datum["name"])

        return {
            "name": name,
            "image-anno": image_anno_path,
            "image-expl": image_expl_path,
            "error": error,
            "score": score,
        }

    def show_st(data_to_show):
        st.set_page_config(layout="wide")
        cols = st.columns(len(data_frames))
        for col, datum in zip(cols, data_to_show):
            with col:
                st.markdown(
                    "`{}...` · score: {:.3f}".format(datum["name"][:20], datum["score"])
                )
                st.image(datum["image-anno"])
                st.markdown("error: {:.3f}".format(datum["error"]))
                st.image(datum["image-expl"])

    def show_latex(data_to_show):
        image_kwargs = {
            "relative_to": "output/plots",
            "options": ["myimg"],
        }

        def get_path_rel(path):
            path = Path(path)
            path = path.relative_to("output/plots")
            path = "imgs/" + str(path)
            return path

        def make_col(datum):
            return [
                "score: {:.3f}".format(datum["score"]),
                mylatex.macro("myimg", get_path_rel(datum["image-anno"])),
                "error: {:.3f}".format(datum["error"]),
                mylatex.macro("myimg", get_path_rel(datum["image-expl"])),
            ]

        table = [make_col(datum) for datum in data_to_show]
        table = transpose(table)
        table = mylatex.tabular(table)
        st.markdown("```\n" + table + "\n```")

    data_to_show = [do1(datum) for datum in data_frames]
    show_st(data_to_show)
    show_latex(data_to_show)


TODO = {
    "plot-mae-vs-thresh": make_plot_mae_vs_thresh,
    "table-examples": make_table_examples,
}


def main():
    what, *args = sys.argv[1:]
    TODO[what](*args)


if __name__ == "__main__":
    main()
