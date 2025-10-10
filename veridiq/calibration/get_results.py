import sys

from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score


DATASETS = [
    "av1m",
    "favc",
    "avlips",
    "dfeval",
]

BASE_DIR = Path("output/training-linear-probing")

def load_results_path(path):
    return pd.read_csv(path)


def load_results(config_name, dataset):
    path = BASE_DIR / config_name / "predict" / dataset / "results.csv"
    return load_results_path(path)


def evaluate(df):
    scores = df["scores"].to_numpy()
    labels = df["labels"].to_numpy()
    return 100 * roc_auc_score(y_score=scores, y_true=labels)


def get_results_default(config_name, dataset):
    df = load_results(config_name, dataset)
    return evaluate(df)


def get_results_dfeval(config_name, dataset):
    def filter_results(df):
        return df[df["paths"].str.startswith("Deepfake-Eval-2024/")]

    path = BASE_DIR / config_name / "predict" / "bitdf" / "results.csv"
    df = load_results_path(path)
    df = filter_results(df)
    return evaluate(df)


GET_RESULTS_FUNCS = {
    "av1m": get_results_default,
    "favc": get_results_default,
    "avlips": get_results_default,
    "dfeval": get_results_dfeval,
}


def get_results(config_name):
    return [
        GET_RESULTS_FUNCS[dataset](config_name, dataset)
        for dataset in DATASETS
    ]


config_name = sys.argv[1]
results = get_results(config_name)
results = [results[0], np.nan, *results[1:]]
print(",".join(["{:.2f}".format(r) for r in results]))