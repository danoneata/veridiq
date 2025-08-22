import argparse
import os

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Late fusion script'
    )

    parser.add_argument('--audio_csv_path')
    parser.add_argument('--video_csv_path')
    parser.add_argument('--output_path')
    args = parser.parse_args()

    aud_df = pd.read_csv(args.audio_csv_path)
    vid_df = pd.read_csv(args.video_csv_path)

    # Remove in order not to have duplicate columns
    if "labels" in vid_df.columns:
        vid_df = vid_df.drop(columns=["labels"])

    # Rename scores
    aud_df_renamed = aud_df.rename(columns={"scores": "scores_aud"})
    vid_renamed = vid_df.rename(columns={"scores": "scores_vid"})

    # Merge and average
    merged_df = pd.merge(aud_df_renamed, vid_renamed, on="paths", how="inner")
    merged_df["scores_avg"] = merged_df[["scores_aud", "scores_vid"]].mean(axis=1)

    os.makedirs(args.output_path, exist_ok=True)
    merged_df.to_csv(os.path.join(args.output_path, "results.csv"), index=False)
    with open(os.path.join(args.output_path, "eval_results.txt"), "w") as f:
        scores = merged_df["scores_avg"].to_numpy()
        labels = merged_df["labels"].to_numpy()
        f.write(f'No: {scores.shape}\n')
        f.write(f"AUC: {roc_auc_score(y_score=scores, y_true=labels)}\n")
        f.write(f"AP: {average_precision_score(y_score=scores, y_true=labels)}\n")
