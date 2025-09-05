import json

import numpy as np
import pandas as pd

from veridiq.explanations.generate_per_frame_scores import PATH as FRAME_SCORES_PATH


class VideoScoreLoader:
    def __init__(self, config_name):
        self.scores = self.load_predictions(config_name)

    @staticmethod
    def load_predictions(config_name):
        path = f"output/training-linear-probing/{config_name}/test/results.csv"
        df = pd.read_csv(path)
        df = df.rename(columns={"paths": "name", "scores": "pred-score", "labels": "label"})
        df = df.set_index("name")
        return df.to_dict(orient="index")

    def __call__(self, name, *args, **kwargs):
        return self.scores[name]["pred-score"]


class FrameScoreLoader:
    def __init__(self, config_name):
        path = FRAME_SCORES_PATH.format(config_name)
        with open(path, "r") as f:
            self.scores = json.load(f)

    def __call__(self, name, frame_idx):
        datum = [
            datum
            for datum in self.scores
            if datum["name"] == name and datum["frame-idx"] == frame_idx
        ]
        if len(datum) == 0:
            return -np.inf
        elif len(datum) == 1:
            return datum[0]["score"]
        else:
            assert False


SCORE_LOADERS = {
    "video": VideoScoreLoader,
    "frame": FrameScoreLoader,
}
