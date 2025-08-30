import pdb
import random

import click
import numpy as np

from sklearn.metrics import mean_absolute_error

from veridiq.data import ExDDV
from veridiq.explanations.generate_spatial_explanations import get_exddv_videos
from veridiq.utils import read_json



def get_prediction_random(*args, **kwargs):
    return {"x": np.random.rand(), "y": np.random.rand()}


def get_prediction_center(*args, **kwargs):
    return {"x": 0.5, "y": 0.5}


class GetPredictionCenterFace:
    def __init__(self):
        path = "output/exddv/extracted-faces.json"
        self.data = read_json(path)

    def __call__(self, video, click):
        data_ = [datum for datum in self.data if datum["name"] == video["name"] and click["frame-idx"] == datum["frame-idx"]]

        if len(data_) == 0:
            print("WARN no data for {}".format(video["name"]))
            return {"x": 0.5, "y": 0.5}

        datum = data_[0]
        faces = datum["face-locations"]

        if len(faces) == 0:
            print("WARN no faces detected for {}".format(video["name"]))
            return {"x": 0.5, "y": 0.5}

        face = random.choice(faces)
        y1, x1, y2, x2 = face
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        h, w = datum["frame-size"]
        return {"x": x / w, "y": y / h}


GET_PREDICTORS = {
    "random": lambda: get_prediction_random,
    "center": lambda: get_prediction_center,
    "center-face": GetPredictionCenterFace,
    # "gradcam": get_prediction_gradcam,
}


def evaluate1(click, prediction):
    keys = ["x", "y"]
    true = [click[k] for k in keys]
    pred = [prediction[k] for k in keys]
    return mean_absolute_error(true, pred)


@click.command()
@click.option("-p", "--pred", "pred_type", type=click.Choice(GET_PREDICTORS.keys()))
def main(pred_type):
    get_prediction_video = GET_PREDICTORS[pred_type]()
    results = [
        evaluate1(click, get_prediction_video(video, click))
        for video in get_exddv_videos()
        for click in video["clicks"]
    ]
    print(np.mean(results))


if __name__ == "__main__":
    main()
