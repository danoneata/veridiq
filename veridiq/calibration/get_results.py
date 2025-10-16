from pathlib import Path

import click
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


def load_results_single(config_name, dataset):
    path = BASE_DIR / config_name / "predict" / dataset / "results.csv"
    return load_results_path(path)


def ensemble_results(dfs):
    scores = np.stack([df["scores"].to_numpy() for df in dfs], axis=0)
    scores = np.mean(scores, axis=0)
    labels = dfs[0]["labels"]
    df = pd.DataFrame({"scores": scores, "labels": labels})
    return df


def evaluate(df):
    scores = df["scores"].to_numpy()
    labels = df["labels"].to_numpy()
    return 100 * roc_auc_score(y_score=scores, y_true=labels)


def get_results_default(config_names, dataset):
    dfs = [load_results_single(config_name, dataset) for config_name in config_names]
    df = ensemble_results(dfs)
    return evaluate(df)


def get_results_dfeval(config_names, dataset):
    def get_path(config_name):
        return BASE_DIR / config_name / "predict" / "bitdf" / "results.csv"

    def filter_results(df):
        return df[df["paths"].str.startswith("Deepfake-Eval-2024/")]

    dfs = [load_results_path(get_path(config_name)) for config_name in config_names]
    dfs = [filter_results(df) for df in dfs]
    df = ensemble_results(dfs)
    return evaluate(df)


GET_RESULTS_FUNCS = {
    "av1m": get_results_default,
    "favc": get_results_default,
    "avlips": get_results_default,
    "dfeval": get_results_dfeval,
}


def get_results(config_names):
    return [GET_RESULTS_FUNCS[dataset](config_names, dataset) for dataset in DATASETS]


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
    results = [results[0], np.nan, *results[1:]]
    print(",".join(["{:.2f}".format(r) for r in results]))


if __name__ == "__main__":
    main()
