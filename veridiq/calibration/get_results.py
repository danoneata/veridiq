import json

from pathlib import Path

import click
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve

from veridiq.linear_probing.train_test import evaluate_frame_level
from veridiq.explanations.show_temporal_explanations import pred_to_proba


DATASETS = [
    "av1m",
    "favc",
    "avlips",
    "dfeval",
]

BASE_DIR = Path("output/training-linear-probing")


def compute_ece(probs, labels, n_bins=10):
    prob_true, prob_pred = calibration_curve(labels, probs, n_bins=n_bins)
    B, _ = np.histogram(probs, bins=n_bins)
    N = len(probs)
    ece = np.sum(np.abs(prob_true - prob_pred) * B / N)
    return ece


def load_results_path(path):
    return pd.read_csv(path)


def load_results_single(config_name, dataset):
    path = BASE_DIR / config_name / "predict" / dataset / "results.csv"
    return load_results_path(path)


def ensemble_results(dfs):
    scores = np.stack([df["scores"].to_numpy() for df in dfs], axis=0)
    scores = np.mean(scores, axis=0)
    labels = dfs[0]["labels"]
    df = pd.DataFrame({"scores": scores, "labels": labels})
    return df


def evaluate_auc(df):
    scores = df["scores"].to_numpy()
    labels = df["labels"].to_numpy()
    return 100 * roc_auc_score(y_score=scores, y_true=labels)


def evaluate_ece(df):
    scores = df["scores"].to_numpy()
    labels = df["labels"].to_numpy()
    probas = pred_to_proba(scores)
    ece = compute_ece(probas, labels, n_bins=10)
    return 100 * ece


EVAL_FUNCS = {
    "auc": evaluate_auc,
    "auc-frame-level": evaluate_frame_level,
    "ece": evaluate_ece,
}


def get_results_default(config_names, dataset, evaluate):
    dfs = [load_results_single(config_name, dataset) for config_name in config_names]
    df = ensemble_results(dfs)
    return evaluate(df)


def get_results_dfeval(config_names, dataset, evaluate):
    def get_path(config_name):
        return BASE_DIR / config_name / "predict" / "bitdf" / "results.csv"

    def filter_results(df):
        return df[df["paths"].str.startswith("Deepfake-Eval-2024/")]

    dfs = [load_results_path(get_path(config_name)) for config_name in config_names]
    dfs = [filter_results(df) for df in dfs]
    df = ensemble_results(dfs)
    return evaluate(df)


def get_results_localization(config_names, dataset, evaluate):
    def load_frame_level_results(config_name):
        path = BASE_DIR / config_name / "predict" / dataset / "results-frame-level.json"
        with open(path, "r") as f:
            return json.load(f)

    def ensemble_scores(data):
        scores = np.stack([d["scores-local"] for d in data], axis=0)
        scores = np.mean(scores, axis=0)
        return scores

    def ensemble_datum(data):
        return {
            "path": data[0]["path"],
            "scores": data[0]["scores"],
            "labels": data[0]["labels"],
            "scores-local": ensemble_scores(data),
            "labels-local": data[0]["labels-local"],
        }

    data_all = [load_frame_level_results(c) for c in config_names]
    ensembled_data = [ensemble_datum(data) for data in zip(*data_all)]
    return 100 * evaluate(ensembled_data)


GET_RESULTS_FUNCS = {
    "av1m": get_results_default,
    "av1m-loc": get_results_localization,
    "favc": get_results_default,
    "avlips": get_results_default,
    "dfeval": get_results_dfeval,
}


def get_results(config_names):
    RESULTS_CONFIGS = [
        ("av1m", "av1m", evaluate_auc),
        ("av1m-loc", "av1m", evaluate_frame_level),
        ("favc", "favc", evaluate_auc),
        ("avlips", "avlips", evaluate_auc),
        ("dfeval", "dfeval", evaluate_auc),
    ]
    results1 = [GET_RESULTS_FUNCS[r](config_names, d, e) for r, d, e in RESULTS_CONFIGS]
    results2 = [GET_RESULTS_FUNCS[d](config_names, d, evaluate_ece) for d in DATASETS]
    return results1 + results2


@click.command()
@click.option(
    "-c",
    "--config-names",
    "config_names",
    type=str,
    required=True,
    multiple=True,
    help="Name of the config (folder in output/training-linear-probing). Repeat to ensemble multiple configs.",
)
def main(config_names):
    assert len(config_names) >= 1, "At least one config name must be provided."
    results = get_results(config_names)
    print(",".join(["{:.2f}".format(r) for r in results]))


if __name__ == "__main__":
    main()
