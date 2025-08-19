import os
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

INVALID_VIDS = [
    "FakeVideo-FakeAudio/African/women/id02071/00195_id01661_SdP_Monh-_4_id04547_wavtolip.mp4",
    "FakeVideo-FakeAudio/African/women/id02071/00195_id04245_nQD_PRlBDyw_id03658_wavtolip.mp4",
    "FakeVideo-FakeAudio/African/men/id01170/00021_id01933_I5XXxgK7QpE_id00781_wavtolip.mp4",
    "RealVideo-RealAudio/Asian_East/men/id04789/002121.mp4",
]

###### AV1M ######

class AV1M_trainval_dataset(Dataset):
    def __init__(self, config, split="train"):
        self.config = config
        self.split = split

        self.root_path = self.config["root_path"]
        self.csv_root_path = self.config["csv_root_path"]

        if config["dataset_name"] == "AV1M":
            self.df = pd.read_csv(os.path.join(self.csv_root_path, f"{self.split}_labels.csv"))
            self.feats_dir = os.path.join(self.root_path, self.split)
        elif config["dataset_name"] == "FAVC":
            self.df = pd.read_csv(os.path.join(self.csv_root_path, f"{self.split}_split.csv"))
            self.feats_dir = self.root_path
            self.df['path'] = self.df['full_path'].apply(lambda x: x.replace("FakeAVCeleb/", ""))
            self.df['label'] = self.df['category'].map({'A': 0, 'D': 1})
            self.df = self.df[~self.df['path'].isin(INVALID_VIDS)]
        else:
            raise ValueError("Wrong dataset_name!")

        if "fvfa_rvra_only" in config and config["fvfa_rvra_only"]:
            if config["dataset_name"] == "AV1M":
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
            elif config["dataset_name"] == "FAVC":
                self.df = self.df[self.df['category'].isin(['A', 'D'])]
            else:
                raise ValueError("Wrong dataset_name!")

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
            try:
                audio = feats['audio']
            except:
                audio = feats
            video = -np.ones((audio.shape[0], 1024)) * np.inf
        elif self.config["input_type"] == "video":
            try:
                video = feats['visual']
                if len(video.shape) > 2:
                    video = video.reshape(-1, video.shape[-1])
            except:
                video = feats
            audio = -np.ones((video.shape[0], 1024)) * np.inf
        elif self.config["input_type"] == "multimodal":
            video = feats["multimodal"]
            audio = feats["multimodal"]
        else:
            raise ValueError(f"input_type should be both, multimodal, video or audio! Got: " + self.config["input_type"])

        if "apply_l2" in self.config and self.config["apply_l2"]:
            video = video / (np.linalg.norm(video, ord=2, axis=-1, keepdims=True))
            audio = audio / (np.linalg.norm(audio, ord=2, axis=-1, keepdims=True))

        return torch.tensor(video, dtype=torch.float32), torch.tensor(audio, dtype=torch.float32), label, row["path"][:-4] + ".npz"  # video, audio, label, path


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
        if self.config["input_type"] == "multimodal":
            self.video_feats = np.load(os.path.join(self.root_path, "multimodal.npy"), allow_pickle=True)
            self.audio_feats = np.load(os.path.join(self.root_path, "multimodal.npy"), allow_pickle=True)

        self.labels = self._get_labels()

    def _get_labels(self):
        if self.config["dataset_name"] == "AV1M":
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

        elif self.config["dataset_name"] == "FAVC":
            df = pd.read_csv(os.path.join(self.csv_root_path, "test_split.csv"))
            df['path'] = df['full_path'].apply(lambda x: x.replace("FakeAVCeleb/", ""))
            df['label'] = df['category'].map({'A': 0, 'D': 1})

            ### PIECE OF COD FOR FAVC VIDEO-MAE TEST
            # new_paths = []
            # for path in self.paths:
            #     path_cleaned = path.replace(" (", "_").replace(")", "")
            #     new_paths.append(path_cleaned)
            # self.paths = np.array(new_paths)

            remove_paths = []
            remove_paths.extend(INVALID_VIDS)
            if "fvfa_rvra_only" in self.config and self.config["fvfa_rvra_only"]:
                remove_paths.extend(df[~df['category'].isin(['A', 'D'])]['path'].to_list())
                df = df[df['category'].isin(['A', 'D'])]

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

        else:
            raise ValueError("Wrong dataset_name!")

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
        if len(video.shape) > 2:
            video = video.reshape(-1, video.shape[-1])

        return torch.tensor(video, dtype=torch.float32), torch.tensor(audio, dtype=torch.float32), label, path  # video, audio, label, path


class FakeAVCeleb_Dataset(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.split = split
        self.root_path = self.config["root_path"]
        self.csv_root_path = self.config["csv_root_path"]
        self.input_type = self.config["input_type"]

        labels = pd.read_csv(os.path.join(self.csv_root_path, f"{split}_split.csv"))
        labels['path'] = labels['full_path'].apply(lambda x: x.replace("FakeAVCeleb/", ""))
        labels = labels[~labels['path'].isin(INVALID_VIDS)]

        if "fvfa_rvra_only" in self.config and self.config["fvfa_rvra_only"]:
            labels = labels[labels['category'].isin(['A', 'D'])]

        self.features, self.paths = np.array([]), np.array([])

        for folder_name in os.listdir(self.root_path):
            feats = np.load(os.path.join(self.root_path, folder_name, f"{self.input_type}.npy"), allow_pickle=True)
            self.features = np.concatenate((self.features, feats))

            ps = np.load(os.path.join(self.root_path, folder_name, "paths.npy"), allow_pickle=True)
            self.paths = np.concatenate((self.paths, ps))

        self.useful_data = []
        for idx in labels.index:
            row = labels.loc[idx]
            path = row['full_path'].replace("FakeAVCeleb/", "")
            label = int(row['category'] != 'A')

            for id_path in range(len(self.paths)):
                if self.paths[id_path] == path:
                    self.useful_data.append((id_path, label))
                    break

    def __len__(self):
        return len(self.useful_data)

    def __getitem__(self, idx):
        id_path, label = self.useful_data[idx]
        feats = self.features[id_path]
        path = self.paths[id_path]

        if "apply_l2" in self.config and self.config["apply_l2"]:
            feats = feats / (np.linalg.norm(feats, ord=2, axis=-1, keepdims=True))
        if len(feats.shape) > 2:
            feats = feats.reshape(-1, feats.shape[-1])

        return torch.tensor(feats), torch.tensor(feats), label, path  # video, audio, label, path

###### ######

class PerFileDataset(Dataset):
    def __init__(self, config):
        self.config = config

        self.root_path = self.config["root_path"]
        self.csv_root_path = self.config["csv_root_path"]

        self.df = pd.read_csv(os.path.join(self.csv_root_path, "metadata.csv"))
        self.feats_dir = self.root_path
        self.df['path'] = self.df['full_file_path']

        self.df = self.df[self.df['original_split'] != "train"]
        self.df = self.df[self.df["label"] != "unknown"]

        if "files_to_remove" in config.keys():
            with open(config["files_to_remove"], "r") as f:
                self.files_to_remove = f.readlines()
                self.files_to_remove = [v.replace("\n", "") for v in self.files_to_remove]

            self.df = self.df[~self.df['path'].isin(self.files_to_remove)]

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].copy()
        row['path'] = row['path'].replace("/feats/", "/videos/")

        try:
            feats = np.load(os.path.join(self.feats_dir, row["path"][:-4] + ".npz"), allow_pickle=True)
        except FileNotFoundError:
            try:
                feats = np.load(os.path.join(self.feats_dir, row["path"][:-4] + ".npy"), allow_pickle=True)
            except FileNotFoundError:
                feats = np.load(os.path.join(self.feats_dir, row["path"][:-4] + ".npz.npy"), allow_pickle=True)

        if row["label"] == "real":
            label = 0
        elif row["label"] == "fake":
            label = 1
        else:
            raise ValueError("only real or fake!")

        if self.config["input_type"] == "both":
            video = feats['visual']
            audio = feats['audio']
        elif self.config["input_type"] == "audio":
            try:
                audio = feats['audio']
            except:
                audio = feats
            video = -np.ones((audio.shape[0], 1024)) * np.inf
        elif self.config["input_type"] == "video":
            try:
                video = feats['visual']
                if len(video.shape) > 2:
                    video = video.reshape(-1, video.shape[-1])
            except:
                video = feats
            audio = -np.ones((video.shape[0], 1024)) * np.inf
        elif self.config["input_type"] == "multimodal":
            video = feats["multimodal"]
            audio = feats["multimodal"]
        else:
            raise ValueError(f"input_type should be both, multimodal, video or audio! Got: " + self.config["input_type"])

        if "apply_l2" in self.config and self.config["apply_l2"]:
            video = video / (np.linalg.norm(video, ord=2, axis=-1, keepdims=True))
            audio = audio / (np.linalg.norm(audio, ord=2, axis=-1, keepdims=True))

        return torch.tensor(video, dtype=torch.float32), torch.tensor(audio, dtype=torch.float32), label, row["path"][:-4] + ".npz"  # video, audio, label, path


def load_data(config, test=False):
    if test:
        if config["dataset_name"] == "FAVC_old":
            test_ds = FakeAVCeleb_Dataset(config, split="test")
        elif config["dataset_name"] == "BitDF":
            test_ds = PerFileDataset(config)
        else:
            test_ds = AV1M_test_dataset(config)
        test_dl = DataLoader(test_ds, shuffle=False, batch_size=1)

        return test_dl

    else:
        if config["dataset_name"] == "FAVC_old":
            train_ds = FakeAVCeleb_Dataset(config, split="train")
            val_ds = FakeAVCeleb_Dataset(config, split="val")
        else:
            train_ds = AV1M_trainval_dataset(config, split="train")
            val_ds = AV1M_trainval_dataset(config, split="val")

        train_dl = DataLoader(train_ds, shuffle=True, batch_size=1)
        val_dl = DataLoader(val_ds, shuffle=False, batch_size=1)

        return train_dl, val_dl
