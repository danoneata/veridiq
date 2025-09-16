from pathlib import Path

import glob
import os
import pdb
import sys
import streamlit as st

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, roc_auc_score


BASE_DIR = Path("/data/av-datasets/results_uni_multi_modal/outputs")
MODELS = [
    "wav2vec",
    "avh_audio",
    "avh_video",
    "video_mae",
    "clip",
]
MODEL_TO_NAME = {
    # "clip": "CLIP",
    # "wav2vec": "Wav2Vec2",
    # "video_mae": "Video-MAE",
    # "avh_audio": "AV-HuBERT (A)",
    # "avh_video": "AV-HuBERT (V)",
    "clip": "CLIP",
    "wav2vec": "W2V2",
    "video_mae": "V-MAE",
    "avh_audio": "AV-H (A)",
    "avh_video": "AV-H (V)",
}


class GetResults:
    def __init__(self, dataset_tr, dataset_te):
        self.dataset_tr = dataset_tr
        self.dataset_te = dataset_te

    def load_df(self, model):
        path = (
            BASE_DIR
            / ("results_" + self.dataset_tr)
            / ("tested_on_" + self.dataset_te)
            / model
            / "results.csv"
        )
        return pd.read_csv(path)

    def evaluate(self, df):
        scores = df["scores"].to_numpy()
        labels = df["labels"].to_numpy()
        return 100 * roc_auc_score(y_score=scores, y_true=labels)

    def get_results_single(self, model):
        df = self.load_df(model)
        return self.evaluate(df)

    def get_results_combination(self, model1, model2):
        df1 = self.load_df(model1)
        df2 = self.load_df(model2)
        scores1 = df1["scores"].to_numpy()
        scores2 = df2["scores"].to_numpy()
        labels1 = df1["labels"].to_numpy()
        labels2 = df2["labels"].to_numpy()
        assert (labels1 == labels2).all()
        df = pd.DataFrame(
            {
                "labels": labels1,
                "scores": 0.5 * (scores1 + scores2),
            }
        )
        return self.evaluate(df)

    def get_correlation(self, model1, model2):
        df1 = self.load_df(model1)
        df2 = self.load_df(model2)
        scores1 = df1["scores"].to_numpy()
        scores2 = df2["scores"].to_numpy()
        return 100 * np.corrcoef(scores1, scores2)[0, 1]

    def __call__(self, model1, model2):
        if model1 == model2:
            return self.get_results_single(model1)
        else:
            return self.get_results_combination(model1, model2)


def make_plot_correlation(df, ax):
    sns.heatmap(
        df,
        vmin=0.0,
        vmax=100.0,
        annot=True,
        fmt=".1f",
        square=True,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Model correlation")


def make_plot_absolute(df, ax):
    sns.heatmap(
        df,
        vmin=50.0,
        vmax=100.0,
        annot=True,
        fmt=".1f",
        square=True,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")


def make_plot_improvement(df, ax1, ax2):
    values_diag = np.diag(df.values)[:, np.newaxis]
    values_improvement = 100 * (df.values - values_diag) / values_diag

    df_values_diag = pd.DataFrame(
        values_diag,
        index=df.index,
        columns=[""],
    )
    df_values_improvement = pd.DataFrame(
        values_improvement,
        index=df.index,
        columns=df.columns,
    )

    sns.heatmap(
        df_values_diag,
        fmt=".1f",
        vmin=50.0,
        vmax=100.0,
        ax=ax1,
        cbar=False,
        annot=True,
        square=True,
    )
    sns.heatmap(
        df_values_improvement,
        vmin=-50,
        vmax=+50,
        fmt=".1f",
        cmap=sns.diverging_palette(10, 220, as_cmap=True, center="light"),
        ax=ax2,
        cbar=False,
        annot=True,
        square=True,
    )

    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.set_title("AUC")

    ax2.set_xlabel("")
    ax2.set_ylabel("")
    # ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
    # ax2.set_title("Improvement relative to unimodal")
    ax2.set_title("Relative improvement (%)")


def make_plots():
    def make_df(results):
        MODEL_ORDER = [MODEL_TO_NAME[m] for m in MODELS]
        df = pd.DataFrame(results)
        df["model1"] = df["model1"].map(MODEL_TO_NAME)
        df["model2"] = df["model2"].map(MODEL_TO_NAME)
        df = df.pivot(index="model1", columns="model2", values="value")
        df = df.reindex(index=MODEL_ORDER, columns=MODEL_ORDER)
        return df

    get_results = GetResults("av1m", "favc")

    results_perf = [
        {
            "model1": m1,
            "model2": m2,
            "value": get_results(m1, m2),
        }
        for m1 in MODELS
        for m2 in MODELS
    ]
    df_perf = make_df(results_perf)

    results_corr = [
        {
            "model1": m1,
            "model2": m2,
            "value": get_results.get_correlation(m1, m2),
        }
        for m1 in MODELS
        for m2 in MODELS
    ]
    df_corr = make_df(results_corr)


    sns.set(context="poster", font="Arial")

    fig = plt.figure(figsize=(14, 6))
    spec = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[5, 6], wspace=0.1)
    spec1 = spec[0].subgridspec(nrows=1, ncols=1)
    spec2 = spec[1].subgridspec(nrows=1, ncols=2, width_ratios=[1, 5], wspace=0.05)

    ax = fig.add_subplot(spec1[0, 0])
    axs = [
        fig.add_subplot(spec2[0, 0]),
        fig.add_subplot(spec2[0, 1]),
    ]

    make_plot_correlation(df_corr, ax)
    make_plot_improvement(df_perf, *axs)
    axs[0].set_yticklabels([])
    axs[1].set_yticklabels([])

    fig.tight_layout()
    st.pyplot(fig)

    path = "output/plots/fusion-plot-{}-{}.pdf".format(
        get_results.dataset_tr,
        get_results.dataset_te,
    )
    fig.savefig(path, bbox_inches="tight")


make_plots()
