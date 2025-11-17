import json
import os

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

class AV1M_RealOnly_Dataset(Dataset):
    def __init__(self, config, split="train"):
        self.config = config
        self.split = split

        self.csv_root_path = self.config["csv_root_path"]
        self.df = pd.read_csv(os.path.join(self.csv_root_path, f"real_{self.split}_data.csv"))

        self.audio_root_path = self.config["audio_root_path"]
        self.video_root_path = self.config["video_root_path"]
        self.multimodal_root_path = self.config["multimodal_root_path"]
        self.audio_feats_dir = os.path.join(self.audio_root_path, self.split)
        self.video_feats_dir = os.path.join(self.video_root_path, self.split)  # TODO: find a better workaround for CLIP
        self.multimodal_feats_dir = os.path.join(self.multimodal_root_path, self.split)

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
        label = int(row["label"])

        if self.config["modality"] == "both":
            audio = self._get_feats(row, "audio", self.audio_feats_dir)
            video = self._get_feats(row, "visual", self.video_feats_dir)

            residual = video.shape[0] - audio.shape[0]
            if residual > 0:
                video = video[:-residual]
            elif residual < 0:
                audio = audio[:residual]
            multi = -np.ones((audio.shape[0], 1024))
        elif self.config["modality"] == "audio":
            audio = self._get_feats(row, "audio", self.audio_feats_dir)
            video = -np.ones((audio.shape[0], 1024))
            multi = -np.ones((audio.shape[0], 1024))
        elif self.config["modality"] == "video":
            video = self._get_feats(row, "visual", self.video_feats_dir)
            audio = -np.ones((video.shape[0], 1024))
            multi = -np.ones((video.shape[0], 1024))
        elif self.config["modality"] == "multimodal":
            multi = self._get_feats(row, "multimodal", self.multimodal_feats_dir)
            audio = -np.ones((multi.shape[0], 1024))
            video = -np.ones((multi.shape[0], 1024))

        if "apply_l2" in self.config and self.config["apply_l2"]:
            video = video / (np.linalg.norm(video, ord=2, axis=-1, keepdims=True))
            audio = audio / (np.linalg.norm(audio, ord=2, axis=-1, keepdims=True))
            multi = multi / (np.linalg.norm(multi, ord=2, axis=-1, keepdims=True))

        if self.config["modality"] == "both":
            features = np.concatenate((video, audio), axis=-1)
        elif self.config["modality"] == "audio":
            features = audio
        elif self.config["modality"] == "video":
            features = video
        elif self.config["modality"] == "multimodal":
            features = multi

        return torch.tensor(features), torch.ones(features.shape[0]), label, row["path"][:-4] + ".npz"  # features, mask, label, path


class CombinedForm_Dataset(Dataset):
    def __init__(self, config):
        self.config = config

        if self.config["modality"] == "both":
            self.paths_aud = np.load(os.path.join(self.config["audio_root_path"], "paths.npy"), allow_pickle=True)
            self.paths_vid = np.load(os.path.join(self.config["video_root_path"], "paths.npy"), allow_pickle=True)
            self.audio_feats = np.load(os.path.join(self.config["audio_root_path"], "audio.npy"), allow_pickle=True)
            self.video_feats = np.load(os.path.join(self.config["video_root_path"], "video.npy"), allow_pickle=True)
        elif self.config["modality"] == "audio":
            self.paths_aud = np.load(os.path.join(self.config["audio_root_path"], "paths.npy"), allow_pickle=True)
            self.paths_vid = None
            self.audio_feats = np.load(os.path.join(self.config["audio_root_path"], "audio.npy"), allow_pickle=True)
            self.video_feats = None
        elif self.config["modality"] == "video":
            self.paths_aud = None
            self.paths_vid = np.load(os.path.join(self.config["video_root_path"], "paths.npy"), allow_pickle=True)
            self.audio_feats = None
            self.video_feats = np.load(os.path.join(self.config["video_root_path"], "video.npy"), allow_pickle=True)
        elif self.config["modality"] == "multimodal":
            self.paths_multi = np.load(os.path.join(self.config["multimodal_root_path"], "paths.npy"), allow_pickle=True)
            self.multimodal_feats = np.load(os.path.join(self.config["multimodal_root_path"], "multimodal.npy"), allow_pickle=True)

        self.csv_root_path = self.config["csv_root_path"]
        if self.config["name"] == "AV1M":
            self.df = pd.read_csv(os.path.join(self.csv_root_path, "test_labels.csv"))
        elif self.config["name"] == "FAVC":
            self.df = pd.read_csv(os.path.join(self.csv_root_path, "test_split.csv"))
            self.df['path'] = self.df['full_path'].apply(lambda x: x.replace("FakeAVCeleb/", ""))
            self.df['label'] = self.df['category'].map({'A': 0, 'D': 1})
            self.df = self.df[~self.df['path'].isin(INVALID_VIDS)]
        else:
            raise ValueError("Wrong dataset name")

        if "rvra_fvfa_only" in self.config and self.config["rvra_fvfa_only"]:
            self._clean_df()

        removed_rows = []
        for idx, row in self.df.iterrows():
            if self.config["modality"] == "multimodal":
                if row['path'] not in self.paths_multi:
                    removed_rows.append(idx)
            elif (self.paths_aud is not None and row['path'] not in self.paths_aud) or (self.paths_vid is not None and row['path'] not in self.paths_vid):
                removed_rows.append(idx)
        self.df = self.df.drop(removed_rows)
        self._get_indices()

    def _clean_df(self):
        print(len(self.df.index), flush=True)
        if self.config["name"] == "AV1M":
            with open(self.config["metadata_path"], "r") as f:
                metadata = json.load(f)

            remove_indices = []
            for idx, row in self.df.iterrows():
                for md in metadata:
                    if md['file'] == row['path']:
                        if not ((len(md['fake_segments']) == 0) or (len(md['visual_fake_segments']) > 0 and len(md['audio_fake_segments']) > 0)):
                            remove_indices.append(idx)
                        break

            self.df.drop(remove_indices, inplace=True)
        elif self.config["name"] == "FAVC":
            self.df = self.df[self.df['category'].isin(['A', 'D'])]
        else:
            raise ValueError("Wrong dataset name")
        print(len(self.df.index), flush=True)

    def _get_indices(self):
        self.indices_aud = {}
        self.indices_vid = {}
        self.indices_multi = {}

        for _, row in self.df.iterrows():
            path = row['path']

            if self.config["modality"] == "both" or self.config["modality"] == "audio":
                indices = np.where(self.paths_aud == path)[0]
                if len(indices) != 1:
                    raise ValueError(f"Multiple or no values for path (audio): {path}")
                self.indices_aud[path] = indices[0]

            if self.config["modality"] == "both" or self.config["modality"] == "video":
                indices = np.where(self.paths_vid == path)[0]
                if len(indices) != 1:
                    raise ValueError(f"Multiple or no values for path (video): {path}")
                self.indices_vid[path] = indices[0]

            if self.config["modality"] == "multimodal":
                indices = np.where(self.paths_multi == path)[0]
                if len(indices) != 1:
                    raise ValueError(f"Multiple or no values for path (multimodal): {path}")
                self.indices_multi[path] = indices[0]

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['path']
        labels = row['label']

        if self.config["modality"] == "both":
            video = self.video_feats[self.indices_vid[path]]
            audio = self.audio_feats[self.indices_aud[path]]
            assert self.paths_vid[self.indices_vid[path]] == path
            assert self.paths_aud[self.indices_aud[path]] == path

            residual = video.shape[0] - audio.shape[0]
            if residual > 0:
                video = video[:-residual]
            elif residual < 0:
                audio = audio[:residual]
            multi = -np.ones((audio.shape[0], 1024))
        elif self.config["modality"] == "audio":
            audio = self.audio_feats[self.indices_aud[path]]
            video = -np.ones((audio.shape[0], 1024))
            multi = -np.ones((audio.shape[0], 1024))
            assert self.paths_aud[self.indices_aud[path]] == path
        elif self.config["modality"] == "video":
            video = self.video_feats[self.indices_vid[path]]
            audio = -np.ones((video.shape[0], 1024))
            multi = -np.ones((video.shape[0], 1024))
            assert self.paths_vid[self.indices_vid[path]] == path
        elif self.config["modality"] == "multimodal":
            multi = self.multimodal_feats[self.indices_multi[path]]
            audio = -np.ones((multi.shape[0], 1024))
            video = -np.ones((multi.shape[0], 1024))
            assert self.paths_multi[self.indices_multi[path]] == path

        if "apply_l2" in self.config and self.config["apply_l2"]:
            video = video / (np.linalg.norm(video, ord=2, axis=-1, keepdims=True))
            audio = audio / (np.linalg.norm(audio, ord=2, axis=-1, keepdims=True))
            multi = multi / (np.linalg.norm(multi, ord=2, axis=-1, keepdims=True))

        if self.config["modality"] == "both":
            features = np.concatenate((video, audio), axis=-1)
        elif self.config["modality"] == "audio":
            features = audio
        elif self.config["modality"] == "video":
            features = video
        elif self.config["modality"] == "multimodal":
            features = multi

        return torch.tensor(features, dtype=torch.float32), torch.ones(features.shape[0]), labels, path  # features, mask, labels, path


class PerFile_Dataset(Dataset):
    def __init__(self, config):
        self.config = config

        self.csv_root_path = self.config["csv_root_path"]
        self.df = pd.read_csv(os.path.join(self.csv_root_path, "metadata.csv"))

        self.audio_feats_dir = self.config["audio_root_path"]
        self.video_feats_dir = self.config["video_root_path"]
        self.multimodal_feats_dir = self.config["multimodal_root_path"]

        self.df['path'] = self.df['full_file_path']

        self.df = self.df[self.df['original_split'] != "train"]
        self.df = self.df[self.df["label"] != "unknown"]
        self.df = self.df[self.df["path"].str.startswith("Deepfake-Eval-2024/")]  # for dfeval only evaluation

        if "files_to_remove" in config.keys():
            with open(config["files_to_remove"], "r") as f:
                self.files_to_remove = f.readlines()
                self.files_to_remove = [v.replace("\n", "") for v in self.files_to_remove]

            self.df = self.df[~self.df['path'].isin(self.files_to_remove)]

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
        row = self.df.iloc[idx].copy()
        row['path'] = row['path'].replace("/feats/", "/videos/")

        if row["label"] == "real":
            label = 0
        elif row["label"] == "fake":
            label = 1
        else:
            raise ValueError("only real or fake!")

        if self.config["modality"] == "both":
            audio = self._get_feats(row, "audio", self.audio_feats_dir)
            video = self._get_feats(row, "visual", self.video_feats_dir)

            residual = video.shape[0] - audio.shape[0]
            if residual > 0:
                video = video[:-residual]
            elif residual < 0:
                audio = audio[:residual]
            multi = -np.ones((audio.shape[0], 1024))
        elif self.config["modality"] == "audio":
            audio = self._get_feats(row, "audio", self.audio_feats_dir)
            video = -np.ones((audio.shape[0], 1024))
            multi = -np.ones((audio.shape[0], 1024))
        elif self.config["modality"] == "video":
            video = self._get_feats(row, "visual", self.video_feats_dir)
            audio = -np.ones((video.shape[0], 1024))
            multi = -np.ones((video.shape[0], 1024))
        elif self.config["modality"] == "multimodal":
            multi = self._get_feats(row, "multimodal", self.multimodal_feats_dir)
            audio = -np.ones((multi.shape[0], 1024))
            video = -np.ones((multi.shape[0], 1024))

        if "apply_l2" in self.config and self.config["apply_l2"]:
            video = video / (np.linalg.norm(video, ord=2, axis=-1, keepdims=True))
            audio = audio / (np.linalg.norm(audio, ord=2, axis=-1, keepdims=True))
            multi = multi / (np.linalg.norm(multi, ord=2, axis=-1, keepdims=True))

        if self.config["modality"] == "both":
            features = np.concatenate((video, audio), axis=-1)
        elif self.config["modality"] == "audio":
            features = audio
        elif self.config["modality"] == "video":
            features = video
        elif self.config["modality"] == "multimodal":
            features = multi

        return torch.tensor(features, dtype=torch.float32), torch.ones(features.shape[0]), label, row["path"][:-4] + ".npz"  # features, mask, label, path


###### ######


def collate_fn(data):
    max_len = max([x.shape[0] for (x, _, _, _) in data])
    features = []
    masks = []
    labels = []
    paths = []

    for sample in data:
        feats, mask, label, path = sample

        if feats.shape[0] != max_len:
            feats = torch.cat((feats, torch.zeros(max_len - feats.shape[0], feats.shape[1])), dim=0)
            mask = torch.cat((mask, torch.zeros(max_len - mask.shape[0])), dim=0)

        features.append(feats)
        masks.append(mask)
        labels.append(label)
        paths.append(path)

    return torch.stack(features), torch.stack(masks), torch.tensor(labels), paths


def load_data(config, test=False):
    if test:
        if config["name"] == "AV1M" or config["name"] == "FAVC":
            test_ds = CombinedForm_Dataset(config)
        elif config["name"] == "DFEval":
            test_ds = PerFile_Dataset(config)
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
