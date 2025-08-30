import click
import numpy as np

from sklearn.metrics import mean_absolute_error

from veridiq.data import ExDDV
from veridiq.explanations.generate_spatial_explanations import get_exddv_videos



def get_prediction_random(*args, **kwargs):
    return {"x": np.random.rand(), "y": np.random.rand()}


def get_prediction_center(*args, **kwargs):
    return {"x": 0.5, "y": 0.5}


PREDICTORS = {
    "random": get_prediction_random,
    "center": get_prediction_center,
    "center-face": get_prediction_center_face,
}


def evaluate1(click, prediction):
    keys = ["x", "y"]
    true = [click[k] for k in keys]
    pred = [prediction[k] for k in keys]
    return mean_absolute_error(true, pred)


@click.command()
@click.option("-p", "--pred", "pred_type", type=click.Choice(PREDICTORS.keys()))
def main(pred_type):
    get_prediction_video = PREDICTORS[pred_type]
    results = [
        evaluate1(click, get_prediction_video(video, click))
        for video in get_exddv_videos()
        for click in video["clicks"]
    ]
    print(np.mean(results))


if __name__ == "__main__":
    main()
