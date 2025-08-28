import glob
import os

import pandas as pd
from scipy import stats


if __name__ == "__main__":
    models_to_run = ["clip", "wav2vec", "video_mae", "avh_audio", "avh_video"]
    root_path = "/mnt/results_uni_multi_modal/outputs/"
    dst_path = "/root/veridiq/veridiq/linear_probing/outputs/results_correlation/"

    path_list = glob.glob(os.path.join(root_path, "**", models_to_run[0]), recursive=True)
    path_list = [os.path.dirname(x).replace(root_path, "") for x in path_list]
    assert len(path_list) == 8

    for midpath in path_list:
        os.makedirs(os.path.join(dst_path, midpath), exist_ok=True)
        with open(os.path.join(dst_path, midpath, "eval_concatenated_results.txt"), "w") as f:
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
                    spearman_r, p_value = stats.spearmanr(merged_df['scores_first'].to_numpy(), merged_df['scores_second'].to_numpy())
                    f.write(f"Spearman: {spearman_r}; P-value: {p_value}; Correlated?: {p_value <= 0.05}\n")
                    kendall_tau, p_value = stats.kendalltau(merged_df['scores_first'].to_numpy(), merged_df['scores_second'].to_numpy())
                    f.write(f"Kendall: {kendall_tau}; P-value: {p_value}; Correlated?: {p_value <= 0.05}\n\n")
