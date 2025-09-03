import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from tqdm import tqdm

from veridiq.explanations.generate_spatial_explanations import get_exddv_videos
from veridiq.explanations.evaluate_spatial_explanations import (
    evaluate1,
    GET_PREDICTORS,
)
from veridiq.explanations.utils import load_predictions
from veridiq.utils import cache_json


def load_results_gradcam():
    get_prediction_video_gradcam = GET_PREDICTORS["gradcam"]()
    scores = load_predictions("exddv-clip")
    return [
        {
            "loc-error": evaluate1(click, get_prediction_video_gradcam(video, click)),
            "pred-score": scores[video["name"]]["pred-score"],
        }
        for video in get_exddv_videos()
        for click in video["clicks"]
    ]


def get_loc_error_confident(results, thresh):
    results_confident = [r for r in results if r["pred-score"] >= thresh]
    errors = [r["loc-error"] for r in results_confident]
    return errors


path_cache = "output/cache/results-spatial-explanations-gradcam.json"
results_gradcam = cache_json(path_cache, load_results_gradcam)
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
fig, ax = plt.subplots()
sns.lineplot(data=df, x="thresh", y="loc-error", ax=ax)
ax.set_xlabel("Minimum fakeness score")
ax.set_ylabel("MAE")
st.markdown("### Alignment to human annotations")
st.pyplot(fig)