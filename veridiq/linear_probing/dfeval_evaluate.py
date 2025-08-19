import argparse
import os

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Late fusion script'
    )

    parser.add_argument('--root_dir')
    args = parser.parse_args()

    src_root = os.path.join(os.path.dirname(args.root_dir), "tested_on_bitdf")
    dst_root = os.path.join(os.path.dirname(args.root_dir), "tested_on_dfeval")

    folders = os.listdir(src_root)
    for fname in folders:
        df = pd.read_csv(os.path.join(src_root, fname, "results.csv"))
        df = df[df["paths"].str.startswith("Deepfake-Eval-2024/")]

        os.makedirs(os.path.join(dst_root, fname), exist_ok=True)
        df.to_csv(os.path.join(dst_root, fname, "results.csv"), index=False)

        with open(os.path.join(dst_root, fname, "eval_results.txt"), "w") as f:
            try:
                scores = df["scores"].to_numpy()
            except:
                scores = df["scores_avg"].to_numpy()
            labels = df["labels"].to_numpy()
            f.write(f'No: {scores.shape}\n')
            f.write(f"AUC: {roc_auc_score(y_score=scores, y_true=labels)}\n")
            f.write(f"AP: {average_precision_score(y_score=scores, y_true=labels)}\n")
