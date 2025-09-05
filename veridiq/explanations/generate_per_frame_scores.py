import json
import pdb

from tqdm import tqdm
from toolz import partition_all, take

import click
import torch

from veridiq.explanations.generate_spatial_explanations import (
    DEVICE,
    load_config,
    load_model_full,
    get_exddv_images,
)


def predict_batch(model, batch):
    images = [datum["frame"] for datum in batch]
    images = model.feature_extractor.transform(images)
    images = images.to(DEVICE)
    with torch.no_grad():
        scores = model(images)
        return scores.cpu().numpy()


PATH = "output/exddv/explanations/scores-per-frame-{}.json"


@click.command()
@click.option("-c", "--config", "config_name")
def main(config_name):
    B = 32
    model = load_model_full(load_config(config_name))
    output = [
        {
            "score": score.item(),
            "name": datum["name"],
            "frame-idx": datum["frame-idx"],
        }
        for batch in tqdm(partition_all(B, get_exddv_images()))
        for score, datum in zip(predict_batch(model, batch), batch)
    ]
    path = PATH.format(config_name)
    with open(path, "w") as f:
        json.dump(output, f)



if __name__ == "__main__":
    main()