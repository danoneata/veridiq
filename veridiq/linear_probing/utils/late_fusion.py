import argparse
import os

import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Late fusion script'
    )

    parser.add_argument('--config_path', default=None,
                        help='YAML with audio_csv_path, video_csv_path, output_path')
    parser.add_argument('--audio_csv_path', default=None)
    parser.add_argument('--video_csv_path', default=None)
    parser.add_argument('--output_path', default=None)
    args = parser.parse_args()

    if args.config_path is not None:
        with open(args.config_path, "r") as f:
            cfg = yaml.safe_load(f)
        audio_csv_path = args.audio_csv_path or cfg["audio_csv_path"]
        video_csv_path = args.video_csv_path or cfg["video_csv_path"]
        output_path = args.output_path or cfg["output_path"]
    else:
        audio_csv_path = args.audio_csv_path
        video_csv_path = args.video_csv_path
        output_path = args.output_path

    if not audio_csv_path or not video_csv_path or not output_path:
        raise ValueError(
            "Need audio_csv_path, video_csv_path, and output_path "
            "(via --config_path and/or CLI flags)"
        )

    aud_df = pd.read_csv(audio_csv_path)
    vid_df = pd.read_csv(video_csv_path)

    # Remove in order not to have duplicate columns
    if "labels" in vid_df.columns:
        vid_df = vid_df.drop(columns=["labels"])

    # Rename scores
    aud_df_renamed = aud_df.rename(columns={"scores": "scores_aud"})
    vid_renamed = vid_df.rename(columns={"scores": "scores_vid"})

    # Merge and average
    merged_df = pd.merge(aud_df_renamed, vid_renamed, on="paths", how="inner")
    merged_df["scores_avg"] = merged_df[["scores_aud", "scores_vid"]].mean(axis=1)

    os.makedirs(output_path, exist_ok=True)
    merged_df.to_csv(os.path.join(output_path, "results.csv"), index=False)
    with open(os.path.join(output_path, "eval_results.txt"), "w") as f:
        scores = merged_df["scores_avg"].to_numpy()
        labels = merged_df["labels"].to_numpy()
        f.write(f'No: {scores.shape}\n')
        f.write(f"AUC: {roc_auc_score(y_score=scores, y_true=labels)}\n")
        f.write(f"AP: {average_precision_score(y_score=scores, y_true=labels)}\n")
