import os
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


###### AV1M ######

class AV1M_trainval_dataset(Dataset):
    def __init__(self, config, split="train"):
        self.config = config
        self.split = split

        self.root_path = self.config["root_path"]
        self.csv_root_path = self.config["csv_root_path"]

        self.df = pd.read_csv(os.path.join(self.csv_root_path, f"{self.split}_labels.csv"))
        self.feats_dir = os.path.join(self.root_path, self.split)

        if "fvfa_rvra_only" in config and config["fvfa_rvra_only"]:
            with open(config["metadata_path"], "r") as f:
                metadata = json.load(f)

            remove_paths = []
            set_paths = set(self.df['path'].to_list())
            for md in metadata:
                if md["file"] in set_paths:
                    if (len(md['audio_fake_segments']) > 0 and len(md['visual_fake_segments']) > 0) or len(md['fake_segments']) == 0:
                        continue
                    else:
                        remove_paths.append(md["file"])

            remove_paths = set(remove_paths)
            remove_indices = []
            for idx in self.df.index:
                if self.df.iloc[idx]['path'] in remove_paths:
                    remove_indices.append(idx)

            self.df.drop(remove_indices, inplace=True)


    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        try:
            feats = np.load(os.path.join(self.feats_dir, row["path"][:-4] + ".npz"), allow_pickle=True)
        except FileNotFoundError:
            try:
                feats = np.load(os.path.join(self.feats_dir, row["path"][:-4] + ".npy"), allow_pickle=True)
            except FileNotFoundError:
                feats = np.load(os.path.join(self.feats_dir, row["path"][:-4] + ".npz.npy"), allow_pickle=True)

        label = int(row["label"])

        if self.config["input_type"] == "both":
            video = feats['visual']
            audio = feats['audio']
        elif self.config["input_type"] == "audio":
            audio = feats['audio']
            video = -np.ones((video.shape[0], 1024)) * np.inf
        elif self.config["input_type"] == "video":
            video = feats['visual']
            audio = -np.ones((video.shape[0], 1024)) * np.inf
        else:
            raise ValueError(f"input_type should be both, video or audio! Got: " + self.config["input_type"])

        if "apply_l2" in self.config and self.config["apply_l2"]:
            video = video / (np.linalg.norm(video, ord=2, axis=-1, keepdims=True))
            audio = audio / (np.linalg.norm(audio, ord=2, axis=-1, keepdims=True))

        return torch.tensor(video), torch.tensor(audio), label, row["path"][:-4] + ".npz"  # video, audio, label, path


class AV1M_test_dataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.csv_root_path = self.config["csv_root_path"]
        self.root_path = config["root_path"]

        self.paths = np.load(os.path.join(self.root_path, "paths.npy"), allow_pickle=True)
        self.audio_feats = None
        self.video_feats = None

        if self.config["input_type"] == "both" or self.config["input_type"] == "audio":
            self.audio_feats = np.load(os.path.join(self.root_path, "audio.npy"), allow_pickle=True)
        if self.config["input_type"] == "both" or self.config["input_type"] == "video":
            self.video_feats = np.load(os.path.join(self.root_path, "video.npy"), allow_pickle=True)

        self.labels = self._get_labels()

    def _get_labels(self):
        df = pd.read_csv(os.path.join(self.csv_root_path, "test_labels.csv"))

        if "fvfa_rvra_only" in self.config and self.config["fvfa_rvra_only"]:
            with open(self.config["metadata_path"], "r") as f:
                metadata = json.load(f)

            remove_paths = []
            set_paths = set(df['path'].to_list())
            for md in metadata:
                if md["file"] in set_paths:
                    if (len(md['audio_fake_segments']) > 0 and len(md['visual_fake_segments']) > 0) or len(md['fake_segments']) == 0:
                        continue
                    else:
                        remove_paths.append(md["file"])

            remove_paths = set(remove_paths)
            remove_indices = []
            for idx, path in enumerate(self.paths):
                if path in remove_paths:
                    remove_indices.append(idx)

            self.paths = np.delete(self.paths, remove_indices)
            if self.audio_feats is not None:
                self.audio_feats = np.delete(self.audio_feats, remove_indices)
            if self.video_feats is not None:
                self.video_feats = np.delete(self.video_feats, remove_indices)

            remove_indices = []
            for idx in df.index:
                if df.iloc[idx]['path'] in remove_paths:
                    remove_indices.append(idx)

            df.drop(remove_indices, inplace=True)

        labels = {}
        for path in self.paths:
            row = df.loc[df['path'] == path]
            if len(row.index) != 1:
                raise ValueError("Multiple or no entries in test_labels.csv for a single path!")
            row = row.iloc[0]
            labels[path] = int(row['label'])

        return labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        if self.video_feats is None:
            video = -np.ones((self.audio_feats[idx].shape[0], 1024)) * np.inf
        else:
            video = self.video_feats[idx]
        if self.audio_feats is None:
            audio = -np.ones((self.video_feats[idx].shape[0], 1024)) * np.inf
        else:
            audio = self.audio_feats[idx]

        if "apply_l2" in self.config and self.config["apply_l2"]:
            video = video / (np.linalg.norm(video, ord=2, axis=-1, keepdims=True))
            audio = audio / (np.linalg.norm(audio, ord=2, axis=-1, keepdims=True))

        label = self.labels[path]

        return torch.tensor(video), torch.tensor(audio), label, path  # video, audio, label, path

###### ######

def load_data(config, test=False):
    if test:
        test_ds = AV1M_test_dataset(config)
        test_dl = DataLoader(test_ds, shuffle=False, batch_size=1)

        return test_dl

    else:
        train_ds = AV1M_trainval_dataset(config, split="train")
        val_ds = AV1M_trainval_dataset(config, split="val")

        train_dl = DataLoader(train_ds, shuffle=True, batch_size=1)
        val_dl = DataLoader(val_ds, shuffle=False, batch_size=1)

        return train_dl, val_dl
