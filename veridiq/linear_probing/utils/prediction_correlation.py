import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def plot_correlations(corr_metric, p_value, corr_name, labels_models, dst_path, midpath):
    plt.figure(figsize=(9, 9))
    sns.set_theme(font_scale=1.2)
    sns.heatmap(corr_metric, fmt=".2f", vmin=0.0, vmax=1.0, cbar=False, xticklabels=labels_models, yticklabels=labels_models, annot=True, square=True)
    plt.title(f"{corr_name} correlation")
    plt.savefig(os.path.join(dst_path, midpath, f"{corr_name.lower()}_corr.png"))
    plt.tight_layout()
    plt.clf()
    plt.close()

    plt.figure(figsize=(9, 9))
    sns.set_theme(font_scale=1.2)
    sns.heatmap(p_value, fmt=".2f", vmin=0.0, vmax=1.0, cbar=False, xticklabels=labels_models, yticklabels=labels_models, annot=True, square=True)
    plt.title(f"{corr_name} P-value")
    plt.savefig(os.path.join(dst_path, midpath, f"{corr_name.lower()}_pvalue.png"))
    plt.tight_layout()
    plt.clf()
    plt.close()


if __name__ == "__main__":
    models_to_run = ["wav2vec", "avh_audio", "avh_video", "video_mae", "clip"]
    labels_models = ["Wav2Vec2", "AV-HuBERT(A)", "AV-HuBERT(V)", "Video-MAE", "CLIP"]
    root_path = "/mnt/results_uni_multi_modal/outputs/"
    dst_path = "/root/veridiq/veridiq/linear_probing/outputs/rq4/results_correlation/"

    path_list = glob.glob(os.path.join(root_path, "**", models_to_run[0]), recursive=True)
    path_list = [os.path.dirname(x).replace(root_path, "") for x in path_list]
    path_list = [v for v in path_list if not v.startswith("results_dfeval")]
    assert len(path_list) == 8

    for midpath in path_list:
        os.makedirs(os.path.join(dst_path, midpath), exist_ok=True)
        with open(os.path.join(dst_path, midpath, "eval_concatenated_results.txt"), "w") as f:
            pearson_corr = np.zeros((len(models_to_run), len(models_to_run)))
            pearson_p = np.zeros((len(models_to_run), len(models_to_run)))
            spearman_corr = np.zeros((len(models_to_run), len(models_to_run)))
            spearman_p = np.zeros((len(models_to_run), len(models_to_run)))
            kendall_corr = np.zeros((len(models_to_run), len(models_to_run)))
            kendall_p = np.zeros((len(models_to_run), len(models_to_run)))

            for i in range(len(models_to_run)):
                for j in range(i + 1):
                    first_name = models_to_run[i]
                    second_name = models_to_run[j]

                    first = pd.read_csv(os.path.join(root_path, midpath, first_name, "results.csv"))
                    second = pd.read_csv(os.path.join(root_path, midpath, second_name, "results.csv"))
                    first["paths"] = first["paths"].apply(lambda x: x.replace("/feats/", "/videos/"))
                    second["paths"] = second["paths"].apply(lambda x: x.replace("/feats/", "/videos/"))
                    first = first.rename(columns={"scores": "scores_first"})
                    second = second.rename(columns={"scores": "scores_second"})

                    merged_df = pd.merge(first, second, on=["paths", "labels"], how="inner")
                    assert len(merged_df.index) == len(first.index)
                    assert len(merged_df.index) == len(second.index)

                    f.write(f"{first_name} <-> {second_name}\n")
                    pearson_r, p_value = stats.pearsonr(merged_df['scores_first'].to_numpy(), merged_df['scores_second'].to_numpy())
                    f.write(f"Pearson: {pearson_r}; P-value: {p_value}; Correlated?: {p_value <= 0.05}\n")

                    pearson_corr[i, j] = pearson_r
                    pearson_corr[j, i] = pearson_r
                    pearson_p[i, j] = p_value
                    pearson_p[j, i] = p_value

                    spearman_r, p_value = stats.spearmanr(merged_df['scores_first'].to_numpy(), merged_df['scores_second'].to_numpy())
                    f.write(f"Spearman: {spearman_r}; P-value: {p_value}; Correlated?: {p_value <= 0.05}\n")

                    spearman_corr[i, j] = spearman_r
                    spearman_corr[j, i] = spearman_r
                    spearman_p[i, j] = p_value
                    spearman_p[j, i] = p_value

                    kendall_tau, p_value = stats.kendalltau(merged_df['scores_first'].to_numpy(), merged_df['scores_second'].to_numpy())
                    f.write(f"Kendall: {kendall_tau}; P-value: {p_value}; Correlated?: {p_value <= 0.05}\n\n")

                    kendall_corr[i, j] = kendall_tau
                    kendall_corr[j, i] = kendall_tau
                    kendall_p[i, j] = p_value
                    kendall_p[j, i] = p_value

        plot_correlations(pearson_corr, pearson_p, "Pearson", labels_models, dst_path, midpath)
        plot_correlations(spearman_corr, spearman_p, "Spearman", labels_models, dst_path, midpath)
        plot_correlations(kendall_corr, kendall_p, "Kendall", labels_models, dst_path, midpath)
