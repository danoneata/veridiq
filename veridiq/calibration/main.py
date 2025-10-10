from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

from sklearn.calibration import CalibrationDisplay, _SigmoidCalibration
from sklearn.metrics import roc_auc_score

from veridiq.explanations.show_temporal_explanations import pred_to_proba


FEATURE_TYPE = "clip"
FEATURE_TYPES = ["clip", "wav2vec", "video_mae", "avh_audio", "avh_video"]
DATASET_TRAIN = "av1m"
DATASETS_TEST = ["av1m", "favc", "bitdf", "dfeval"]


BASE_DIR = Path("/data/av-datasets/results_uni_multi_modal/outputs")
sns.set(style="whitegrid", context="poster", font="Arial")


def get_folder_results_test(dataset_tr, dataset_te):
    return BASE_DIR / ("results_" + dataset_tr) / ("tested_on_" + dataset_te)


def get_folder_results_valid(dataset_tr, dataset_te):
    assert dataset_tr == dataset_te == "av1m"
    return Path("output/calibration/av1m/valid")


GET_FOLDER_RESULTS = {
    "test": get_folder_results_test,
    "valid": get_folder_results_valid,
}


def load_df(dataset_tr, dataset_te, model, split="test"):
    folder = GET_FOLDER_RESULTS[split](dataset_tr, dataset_te)
    path = folder / model / "results.csv"
    return pd.read_csv(path)


def show_calibration_plot(ax, df):
    labels = df["labels"].to_numpy()
    probas = df["probas"].to_numpy()
    CalibrationDisplay.from_predictions(labels, probas, n_bins=10, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Frac. pos.")
    legend = ax.legend()
    legend.remove()


def show_probas_histogram(ax, df):
    labels = df["labels"].to_numpy()
    probas = df["probas"].to_numpy()
    df = pd.DataFrame({"probas": probas, "labels": labels})
    sns.histplot(df, x="probas", hue="labels", bins=20, ax=ax)
    ax.set_xlabel("Probabilty")


δ = 0.01


def entropy_normed(x):
    if x == 0 or x == 1:
        return 1.0
    else:
        max_entropy = -np.log(0.5)
        H = -x * np.log(x) - (1 - x) * np.log(1 - x)
        return 1.0 - H / max_entropy


def max_normed(x):
    return 2 * (np.maximum(x, 1 - x) - 0.5)


def evaluate_reliability(df, τ=0.5):
    idxs = df["reliability"] >= τ

    pred_binary = df[idxs]["pred-binary"]
    pred = df[idxs]["probas"]
    true = df[idxs]["labels"]
    accuracy = np.mean(true == pred_binary)

    num_kept = sum(idxs)
    num_samples = len(df)
    frac_kept = num_kept / num_samples

    if num_kept == 0 or true.nunique() == 1:
        auc_roc = np.nan
    else:
        auc_roc = roc_auc_score(true, pred)

    return {
        "accuracy": 100 * accuracy,
        "frac-kept": 100 * frac_kept,
        "auc-roc": 100 * auc_roc,
    }


def get_reliability_metrics(df, func):
    df["reliability"] = df["probas"].map(func)
    df["pred-binary"] = df["probas"] > 0.5

    return [
        {
            "τ": τ,
            **evaluate_reliability(df, τ),
        }
        for τ in np.arange(0.0, 1.0 + δ, δ)
    ]


def show_accuracy_rejection_curve(ax, df):
    metrics_entropy = get_reliability_metrics(df, entropy_normed)
    df = pd.DataFrame(metrics_entropy)
    sns.lineplot(data=df, x="frac-kept", y="auc-roc", ax=ax)
    ax.set_ylim(0, 100)


def show_results_all():
    n_datasets = len(DATASETS_TEST)
    n_rows = 3
    W = 7
    H = 5
    for f in FEATURE_TYPES:
        fig, axs = plt.subplots(
            n_rows,
            n_datasets,
            figsize=(W * n_datasets, H * n_rows),
            # sharex=True,
        )
        for row in axs:
            for ax in row[1:]:
                ax.sharey(row[0])
        for i, dataset_test in enumerate(DATASETS_TEST):
            df = load_df(DATASET_TRAIN, dataset_test, f)
            df["probas"] = pred_to_proba(df["scores"])
            show_calibration_plot(axs[0, i], df)
            show_probas_histogram(axs[1, i], df)
            show_accuracy_rejection_curve(axs[2, i], df)
            axs[0, i].set_title(f"Test: {dataset_test.upper()}")
        fig.tight_layout()
        st.write(f"## Feature type: {f.upper()}")
        st.pyplot(fig)


def show_calibration_before_after():
    feature_type = "clip"

    df = load_df(DATASET_TRAIN, DATASET_TRAIN, feature_type, split="valid")
    scores = df["scores"].to_numpy()
    labels = df["labels"].to_numpy()
    probas = pred_to_proba(scores)
    calib = _SigmoidCalibration()
    calib.fit(probas, labels)
    st.write(calib.a_, calib.b_)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    df0 = df.copy()
    df1 = df.copy()
    df0["probas"] = probas
    df1["probas"] = calib.predict(probas)
    show_calibration_plot(axs[0], df0)
    show_calibration_plot(axs[1], df1)

    fig.tight_layout()
    col, _ = st.columns(2)
    col.pyplot(fig)

    n_rows = 2
    n_datasets = len(DATASETS_TEST)
    W = 7
    H = 5
    fig, axs = plt.subplots(
        n_rows,
        n_datasets,
        figsize=(W * n_datasets, H * n_rows),
        sharex=True,
    )

    for i, dataset_test in enumerate(DATASETS_TEST):
        df0 = load_df("av1m", dataset_test, feature_type, split="test")
        df0["probas"] = pred_to_proba(df0["scores"])

        df1 = df0.copy()
        df1["probas"] = calib.predict(df1["probas"].to_numpy())

        show_calibration_plot(axs[0, i], df0)
        show_calibration_plot(axs[1, i], df1)

        axs[0, i].set_title(f"Test: {dataset_test.upper()}")
        axs[1, i].set_title(f"Test: {dataset_test.upper()}")

    fig.tight_layout()
    st.pyplot(fig)


def show_calibration_feature_combination():
    FUNCS = {
        "calibration": show_calibration_plot,
        "histogram": show_probas_histogram,
        "accuracy-rejection": show_accuracy_rejection_curve,
    }

    with st.sidebar:
        dataset_te = st.selectbox("Test dataset:", DATASETS_TEST, index=1)
        to_show = st.selectbox("Show:", list(FUNCS.keys()), index=0)

    to_show_func = FUNCS[to_show]
    dataset_tr = "av1m"
    feature_types = ["clip", "wav2vec", "video_mae", "avh_audio", "avh_video"]
    n_rows = len(feature_types)
    n_cols = len(feature_types)
    W = 7
    H = 5
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(W * n_cols, H * n_rows),
        sharex=True,
    )

    for i, f1 in enumerate(feature_types):
        for j, f2 in enumerate(feature_types):
            df1 = load_df(dataset_tr, dataset_te, f1, split="test")
            df2 = load_df(dataset_tr, dataset_te, f2, split="test")
            df1["probas"] = pred_to_proba(df1["scores"])
            df2["probas"] = pred_to_proba(df2["scores"])
            df = df1.copy()
            df["probas"] = (df1["probas"] + df2["probas"]) / 2
            to_show_func(axs[i, j], df)
            if i == 0:
                axs[i, j].set_title(f2)
            if j == 0:
                axs[i, j].set_ylabel(f1)

    fig.tight_layout()
    st.pyplot(fig)


def main():
    st.set_page_config(layout="wide")
    # show_results_all()
    # show_calibration_before_after()
    show_calibration_feature_combination()


if __name__ == "__main__":
    main()
