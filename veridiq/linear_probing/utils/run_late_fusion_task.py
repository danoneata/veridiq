import glob
import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, roc_auc_score


if __name__ == "__main__":
    models_to_run = ["clip", "wav2vec", "video_mae", "avh_audio", "avh_video"]
    labels_models = ["CLIP", "Wav2Vec2", "Video-MAE", "AV-HuBERT(A)", "AV-HuBERT(V)"]
    root_path = "/mnt/results_uni_multi_modal/outputs/"
    dst_path = "/root/veridiq/veridiq/linear_probing/outputs/results_correlation/"

    path_list = glob.glob(os.path.join(root_path, "**", models_to_run[0]), recursive=True)
    path_list = [os.path.dirname(x).replace(root_path, "") for x in path_list]
    # pdb.set_trace()
    assert len(path_list) == 8

    for midpath in path_list:
        results_auc = np.zeros((len(models_to_run), len(models_to_run)))
        results_ap = np.zeros((len(models_to_run), len(models_to_run)))

        os.makedirs(os.path.join(dst_path, midpath), exist_ok=True)
        with open(os.path.join(dst_path, midpath, "eval_concatenated_results.txt"), "w") as f:
            for i in range(len(models_to_run)):
                for j in range(i + 1):
                    first_name = models_to_run[i]
                    second_name = models_to_run[j]
                    f.write(f"{first_name} <-> {second_name}\n")

                    first = pd.read_csv(os.path.join(root_path, midpath, first_name, "results.csv"))
                    second = pd.read_csv(os.path.join(root_path, midpath, second_name, "results.csv"))
                    first["paths"] = first["paths"].apply(lambda x: x.replace("/feats/", "/videos/"))
                    second["paths"] = second["paths"].apply(lambda x: x.replace("/feats/", "/videos/"))
                    first = first.rename(columns={"scores": "scores_first"})
                    second = second.rename(columns={"scores": "scores_second"})

                    # Merge and average
                    merged_df = pd.merge(first, second, on=["paths", "labels"], how="inner")
                    assert len(merged_df.index) == len(first.index)
                    assert len(merged_df.index) == len(second.index)
                    merged_df["scores"] = merged_df[["scores_first", "scores_second"]].mean(axis=1)

                    scores = merged_df["scores"].to_numpy()
                    labels = merged_df["labels"].to_numpy()
                    merged_df.to_csv(os.path.join(dst_path, midpath, f"results_{first_name}_{second_name}.csv"), index=False)

                    auc = roc_auc_score(y_score=scores, y_true=labels)
                    ap = average_precision_score(y_score=scores, y_true=labels)
                    results_auc[i, j] = auc
                    results_auc[j, i] = auc
                    results_ap[i, j] = ap
                    results_ap[j, i] = ap

                    f.write(f'No: {scores.shape}\n')
                    f.write(f"AUC: {auc}\n")
                    f.write(f"AP: {ap}\n\n")

        sns.heatmap(results_auc, vmin=0.0, vmax=1.0, xticklabels=labels_models, yticklabels=labels_models, annot=True)
        plt.savefig(os.path.join(dst_path, midpath, f"results_auc.png"))
        plt.clf()
        sns.heatmap(results_ap, vmin=0.0, vmax=1.0, xticklabels=labels_models, yticklabels=labels_models, annot=True)
        plt.savefig(os.path.join(dst_path, midpath, f"results_ap.png"))
        plt.clf()

        diag_auc = np.diag(results_auc)[:, np.newaxis]
        results_auc_diff = results_auc - diag_auc
        diag_ap = np.diag(results_ap)[:, np.newaxis]
        results_ap_diff = results_ap - diag_ap
        sns.heatmap(results_auc_diff, vmin=-0.5, vmax=0.5, xticklabels=labels_models, yticklabels=labels_models, annot=True)
        plt.savefig(os.path.join(dst_path, midpath, f"results_auc_diff.png"))
        plt.clf()
        sns.heatmap(results_ap_diff, vmin=-0.5, vmax=0.5, xticklabels=labels_models, yticklabels=labels_models, annot=True)
        plt.savefig(os.path.join(dst_path, midpath, f"results_ap_diff.png"))
        plt.clf()

        results_auc_diff_per = (results_auc_diff / diag_auc)
        results_ap_diff_per = (results_ap_diff / diag_ap)
        sns.heatmap(results_auc_diff_per, vmin=-0.5, vmax=0.5, fmt=".1%", xticklabels=labels_models, yticklabels=labels_models, annot=True)
        plt.savefig(os.path.join(dst_path, midpath, f"results_auc_diff_per.png"))
        plt.clf()
        sns.heatmap(results_ap_diff_per, vmin=-0.5, vmax=0.5, fmt=".1%", xticklabels=labels_models, yticklabels=labels_models, annot=True)
        plt.savefig(os.path.join(dst_path, midpath, f"results_ap_diff_per.png"))
        plt.clf()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 7.5), gridspec_kw={"width_ratios": [1.5, 7.5], "wspace": 0.01}, constrained_layout=True)
        sns.set_theme(font_scale=1.2)
        sns.heatmap(diag_auc*100, fmt=".2f", vmin=0.0, vmax=100.0, ax=ax1, cbar=False, xticklabels=[], yticklabels=labels_models, annot=True, square=True)
        # fig.colorbar(ax1.collections[0], ax=ax1,location="left", use_gridspec=False, pad=0.2)
        sns.heatmap(results_auc_diff_per, vmin=-0.5, vmax=0.5, fmt=".1%", cmap=sns.diverging_palette(10, 220, as_cmap=True, center="light"), ax=ax2, cbar=False, xticklabels=labels_models, yticklabels=[], annot=True, square=True)
        # fig.colorbar(ax2.collections[0], ax=ax2,location="right", use_gridspec=False, pad=0.2)
        ax2.yaxis.tick_right()
        ax2.tick_params(rotation=0)
        ax2.set_title("Improvement compared to baseline")
        plt.savefig(os.path.join(dst_path, midpath, f"results_auc_comparison.png"))
        plt.clf()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 7.5), gridspec_kw={"width_ratios": [1.5, 7.5], "wspace": 0.01}, constrained_layout=True)
        sns.set_theme(font_scale=1.2)
        sns.heatmap(diag_ap*100, fmt=".2f", vmin=0.0, vmax=100.0, ax=ax1, cbar=False, xticklabels=[], yticklabels=labels_models, annot=True, square=True)
        # fig.colorbar(ax1.collections[0], ax=ax1,location="left", use_gridspec=False, pad=0.2)
        sns.heatmap(results_ap_diff_per, vmin=-0.5, vmax=0.5, fmt=".1%", cmap=sns.diverging_palette(10, 220, as_cmap=True, center="light"), ax=ax2, cbar=False, xticklabels=labels_models, yticklabels=[], annot=True, square=True)
        # fig.colorbar(ax2.collections[0], ax=ax2,location="right", use_gridspec=False, pad=0.2)
        ax2.yaxis.tick_right()
        ax2.tick_params(rotation=0)
        ax2.set_title("Improvement compared to baseline")
        plt.savefig(os.path.join(dst_path, midpath, f"results_ap_comparison.png"))
        plt.clf()
