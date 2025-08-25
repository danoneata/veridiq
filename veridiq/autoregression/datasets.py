import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


###### AV1M ######

class AV1M_RealOnly_Dataset(Dataset):
    def __init__(self, config, split="train"):
        self.config = config
        self.split = split

        self.csv_root_path = self.config["csv_root_path"]
        self.df = pd.read_csv(os.path.join(self.csv_root_path, f"real_{self.split}_data.csv"))

        self.audio_root_path = self.config["audio_root_path"]
        self.video_root_path = self.config["video_root_path"]
        self.audio_feats_dir = os.path.join(self.audio_root_path, self.split)
        self.video_feats_dir = os.path.join(self.video_root_path, "real_" + self.split)  # TODO: find a better workaround for CLIP

    def __len__(self):
        return len(self.df.index)

    def _get_feats(self, row, modality, feats_dir):
        try:
            feats = np.load(os.path.join(feats_dir, row["path"][:-4] + ".npz"), allow_pickle=True)
        except FileNotFoundError:
            try:
                feats = np.load(os.path.join(feats_dir, row["path"][:-4] + ".npy"), allow_pickle=True)
            except FileNotFoundError:
                feats = np.load(os.path.join(feats_dir, row["path"][:-4] + ".npz.npy"), allow_pickle=True)

        try:
            feats = feats[modality]
        except:
            try:
                feats = feats['arr_0']
            except:
                feats = feats

        return feats

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        audio = self._get_feats(row, "audio", self.audio_feats_dir)
        video = self._get_feats(row, "visual", self.video_feats_dir)

        label = int(row["label"])

        if "apply_l2" in self.config and self.config["apply_l2"]:
            video = video / (np.linalg.norm(video, ord=2, axis=-1, keepdims=True))
            audio = audio / (np.linalg.norm(audio, ord=2, axis=-1, keepdims=True))

        return torch.tensor(video), torch.tensor(audio), torch.ones(video.shape[0]), label, row["path"][:-4] + ".npz"  # video, audio, mask, label, path


class AV1M_FakeReal_Dataset(Dataset):
    def __init__(self, config):
        self.config = config

        self.paths_aud = np.load(os.path.join(self.config["audio_root_path"], "paths.npy"), allow_pickle=True)
        self.paths_vid = np.load(os.path.join(self.config["video_root_path"], "paths.npy"), allow_pickle=True)

        self.audio_feats = np.load(os.path.join(self.config["audio_root_path"], "audio.npy"), allow_pickle=True)
        self.video_feats = np.load(os.path.join(self.config["video_root_path"], "video.npy"), allow_pickle=True)

        self.csv_root_path = self.config["csv_root_path"]
        self.df = pd.read_csv(os.path.join(self.csv_root_path, "test_labels.csv"))

        removed_rows = []
        for idx, row in self.df.iterrows():
            if (row['path'] not in self.paths_aud) or (row['path'] not in self.paths_vid):
                removed_rows.append(idx)
        self.df = self.df.drop(removed_rows)
        self._get_indices()

    def _get_indices(self):
        self.indices_aud = {}
        self.indices_vid = {}

        for _, row in self.df.iterrows():
            path = row['path']

            indices = np.where(self.paths_aud == path)[0]
            if len(indices) != 1:
                raise ValueError(f"Multiple or no values for path (audio): {path}")
            self.indices_aud[path] = indices[0]

            indices = np.where(self.paths_vid == path)[0]
            if len(indices) != 1:
                raise ValueError(f"Multiple or no values for path (video): {path}")
            self.indices_vid[path] = indices[0]

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['path']
        labels = row['label']

        video = self.video_feats[self.indices_vid[path]]
        audio = self.audio_feats[self.indices_vid[path]]

        residual = video.shape[0] - audio.shape[0]
        if residual > 0:
            video = video[:-residual]
        elif residual < 0:
            audio = audio[:residual]

        if "apply_l2" in self.config and self.config["apply_l2"]:
            video = video / (np.linalg.norm(video, ord=2, axis=-1, keepdims=True))
            audio = audio / (np.linalg.norm(audio, ord=2, axis=-1, keepdims=True))

        return torch.tensor(video), torch.tensor(audio), torch.ones(video.shape[0]), labels, path  # video, audio, mask, labels, path


###### ######


def collate_fn(data):
    max_len = max([x.shape[0] for (x, _, _, _, _) in data])
    videos = []
    audios = []
    masks = []
    labels = []
    paths = []

    for sample in data:
        video, audio, mask, label, path = sample

        if video.shape[0] != max_len:
            video = torch.cat((video, torch.zeros(max_len - video.shape[0], video.shape[1])), dim=0)
            audio = torch.cat((audio, torch.zeros(max_len - audio.shape[0], audio.shape[1])), dim=0)
            mask = torch.cat((mask, torch.zeros(max_len - mask.shape[0])), dim=0)

        videos.append(video)
        audios.append(audio)
        masks.append(mask)
        labels.append(label)
        paths.append(path)

    return torch.stack(videos), torch.stack(audios), torch.stack(masks), torch.tensor(labels), paths


def load_data(config, test=False):
    if test:
        if config["name"] == "AV1M" or config["name"] == "FAVC":
            test_ds = AV1M_FakeReal_Dataset(config)
        else:
            raise ValueError("Dataset name error. Expected: AV1M, FAVC; Got: " + config["name"])

        test_dl = DataLoader(test_ds, shuffle=False, batch_size=1)  # For now batch_size=1

        return test_dl

    else:
        train_ds = AV1M_RealOnly_Dataset(config, split="train")
        val_ds = AV1M_RealOnly_Dataset(config, split="val")

        train_dl = DataLoader(train_ds, shuffle=True, batch_size=config["batch_size"], collate_fn=collate_fn, num_workers=config["num_workers"])
        val_dl = DataLoader(val_ds, shuffle=False, batch_size=config["batch_size"], collate_fn=collate_fn, num_workers=config["num_workers"])

        return train_dl, val_dl
