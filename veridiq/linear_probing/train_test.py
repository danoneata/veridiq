import argparse
import json
import random
import tqdm
import os
import yaml

from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from sklearn.metrics import average_precision_score, roc_auc_score
from toolz import dissoc, take

from veridiq.linear_probing.datasets import load_data
from veridiq.linear_probing.model import MyModel

# from datasets import load_data
# from model import LinearModel


def init_callbacks(config):
    # LOGGER
    logger_path = config["logger"]["log_path"]
    if config["logger"]["name"] == "tensorboard":
        logger = TensorBoardLogger(logger_path)
    elif config["logger"]["name"] == "csv":
        logger = CSVLogger(logger_path)
    else:
        raise ValueError(config["logger"]["name"] + " not yet implemented!")

    # CALLBACKS
    callbacks = []
    if "ckpt_args" in config:
        callbacks.append(
            ModelCheckpoint(
                monitor=config["ckpt_args"]["metric"],
                dirpath=config["ckpt_args"]["ckpt_dir"],
                filename="model-{epoch:02d}",
                mode=config["ckpt_args"]["mode"],
            )
        )

    if "early_stopping" in config:
        callbacks.append(
            EarlyStopping(
                monitor=config["early_stopping"]["metric"],
                mode=config["early_stopping"]["mode"],
                patience=config["early_stopping"]["patience"],
            )
        )

    return logger, callbacks


def set_seed(seed):
    print(f"Using seed: {seed}", flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_checkpoint_path_from_folder(folder):
    files = os.listdir(folder)
    files = [f for f in files if f.endswith(".ckpt")]
    assert len(files) == 1, "More than one checkpoint in the folder"
    return os.path.join(folder, files[0])


def get_checkpoint_path(config):
    """Allows to get the checkpoint path from a training config."""
    if "ckpt_path" in config:
        return config["ckpt_path"]
    elif "callbacks" in config:
        checkpoint_dir = config["callbacks"]["ckpt_args"]["ckpt_dir"]
        return get_checkpoint_path_from_folder(checkpoint_dir)
    else:
        raise ValueError("No checkpoint path found in config.")


def get_output_path(config):
    """Allows to get the output path from a training config."""
    if "output_path" in config:
        return config["output_path"]
    elif "callbacks" in config:
        log_dir = Path(config["callbacks"]["logger"]["log_path"])
        out_dir = log_dir.parent
        out_dir = out_dir / "test"
        return str(out_dir)
    else:
        raise ValueError("No output path found in config.")


def get_frame_level(config):
    return (
        "frame_level" in config["data_info"].keys()
        and config["data_info"]["frame_level"]
    )


def train(config):
    train_dl, val_dl = load_data(config=config["data_info"])
    model = MyModel(config=config)
    logger, callbacks = init_callbacks(config=config["callbacks"])

    trainer = L.Trainer(max_epochs=config["epochs"], logger=logger, callbacks=callbacks)
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


def evaluate_frame_level(data):
    def eval1(datum):
        return roc_auc_score(
            y_score=datum["scores-local"],
            y_true=datum["labels-local"],
        ) 

    scores = [eval1(d) for d in data if d["labels"] == 1]
    return np.mean(scores)


def test1(dataloader, path_checkpoint, path_output, is_frame_level=False):
    model = MyModel.load_from_checkpoint(path_checkpoint)
    model.to("cuda")
    model.eval()

    def test_frame_level_yes(batch):
        video_feats, audio_feats, local_labels, paths = batch
        video_feats, audio_feats = video_feats.to("cuda"), audio_feats.to("cuda")

        scores, local_scores = model.predict_scores_per_frame(video_feats, audio_feats)

        scores = scores.item()
        labels = local_labels.max()
        labels = labels.item()

        local_scores = local_scores[0].cpu().numpy()
        local_labels = local_labels.cpu().numpy().squeeze()
        length = min(len(local_scores), len(local_labels))
        local_scores = local_scores[:length]
        local_labels = local_labels[:length]

        return {
            "path": paths[0],
            "scores": scores,
            "labels": labels,
            "scores-local": local_scores.tolist(),
            "labels-local": local_labels.tolist(),
        }

    def test1_frame_level_no(batch):
        video_feats, audio_feats, labels, paths = batch
        video_feats, audio_feats = video_feats.to("cuda"), audio_feats.to("cuda")

        scores = model.predict_scores(video_feats, audio_feats)
        scores = scores.item()
        labels = labels.item()

        return {
            "path": paths[0],
            "scores": scores,
            "labels": labels,
        }

    TEST_FUNCS = {
        True: test_frame_level_yes,
        False: test1_frame_level_no,
    }

    with torch.no_grad():
        data = [TEST_FUNCS[is_frame_level](batch) for batch in tqdm.tqdm(dataloader)]

    os.makedirs(path_output, exist_ok=True)

    data_ss = [dissoc(d, "scores-local", "labels-local") for d in data]
    df = pd.DataFrame(data_ss)
    df.to_csv(os.path.join(path_output, "results.csv"), index=False)

    if is_frame_level:
        path = os.path.join(path_output, "results-frame-level.json")
        with open(path, "w") as f:
            json.dump(data, f)

    path = os.path.join(path_output, "eval_results.txt")
    with open(path, "w") as f:
        score_auc = roc_auc_score(y_score=df["scores"], y_true=df["labels"])
        score_ap = average_precision_score(y_score=df["scores"], y_true=df["labels"])
        f.write(f"AUC: {score_auc}\n")
        f.write(f"AP: {score_ap}\n")

    if is_frame_level:
        path = os.path.join(path_output, "eval_results-frame-level.txt")
        with open(path, "w") as f:
            score_auc = evaluate_frame_level(data)
            f.write(f"AUC-frame-level: {score_auc}\n")


def test(config):
    test_dl = load_data(config=config["data_info"], test=True)
    path_checkpoint = get_checkpoint_path(config)
    path_output = get_output_path(config)
    is_frame_level = get_frame_level(config)
    test1(
        dataloader=test_dl,
        path_checkpoint=path_checkpoint,
        path_output=path_output,
        is_frame_level=is_frame_level,
    )

    with open(os.path.join(path_output, "tested_config.yaml"), "w") as f:
    yaml.safe_dump(config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training and testing loop")

    parser.add_argument("--config_path")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    if args.test:
        test(config=config)
    else:
        train(config=config)
