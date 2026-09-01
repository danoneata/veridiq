"""On-the-fly Wav2Vec2 feature extraction with RandAugment-style online audio degradations.

Mirrors clip_online_randaugment.py for the audio pathway (Cubuk et al., CVPR 2020).

Degradation pool (online / streaming proxies on the waveform):
  compress (bit-depth / mu-law), reduce_sample_rate, change_format (PCM round-trip),
  bandlimit (low-pass / \"blur\"), change_tempo (speed / framerate analog).

Applied at train time: N ops sampled with replacement at magnitude M.

Feature post-process matches fe_WAV2VEC.py for xls-r-2b:
  last_hidden (T, 1920) → pair-reshape → (T//2, 3840).

Usage:
  export PYTHONPATH=.../veridiq_2026/veridiq
  python train_test.py --config_path configs/train_config_all_wav2vec_online.yaml
"""

from __future__ import annotations

import io
import json
import os
import random
import warnings
from typing import Callable, List, Optional, Sequence, Tuple

import lightning as L
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoFeatureExtractor, Wav2Vec2Model

from veridiq.linear_probing.datasets import INVALID_VIDS

SAMPLING_RATE = 16_000

OpFn = Callable[[np.ndarray, float, float], np.ndarray]
OpSpec = Tuple[str, OpFn]


def resolve_wav2vec_model_name(config: dict) -> str:
    """HF id or short key; default matches fe_WAV2VEC / disk features."""
    if config.get("wav2vec_model"):
        return str(config["wav2vec_model"])
    data_info = config.get("data_info") or {}
    return str(data_info.get("wav2vec_model", "facebook/wav2vec2-xls-r-2b"))


def _short_to_hf(name: str) -> str:
    aliases = {
        "wav2vec2-xls-r-2b": "facebook/wav2vec2-xls-r-2b",
        "xls-r-2b": "facebook/wav2vec2-xls-r-2b",
        "wav2vec2-xls-r-300m": "facebook/wav2vec2-xls-r-300m",
        "wav2vec2-base": "facebook/wav2vec2-base",
        "wav2vec2-large": "facebook/wav2vec2-large",
    }
    return aliases.get(name, name)


# ---------------------------------------------------------------------------
# Waveform helpers
# ---------------------------------------------------------------------------

def _as_mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    return audio.astype(np.float32, copy=False)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or len(audio) == 0:
        return audio.astype(np.float32, copy=False)
    n = max(1, int(round(len(audio) * float(target_sr) / float(orig_sr))))
    x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    xi = np.linspace(0.0, 1.0, num=n, endpoint=False)
    return np.interp(xi, x, audio.astype(np.float64)).astype(np.float32)


def _magnitude_01(M: float, M_max: float = 10.0) -> float:
    return float(np.clip(M / M_max, 0.0, 1.0))


def _peak_normalize(audio: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < eps:
        return audio.astype(np.float32, copy=False)
    return (audio / peak).astype(np.float32)


# ---------------------------------------------------------------------------
# RandAugment ops (waveform in/out @ SAMPLING_RATE)
# ---------------------------------------------------------------------------

def op_compress(audio: np.ndarray, M: float, M_max: float = 10.0) -> np.ndarray:
    """Lossy amplitude quantization / mu-law (codec-like compression proxy)."""
    t = _magnitude_01(M, M_max)
    # bits: 16 → 4 as M grows
    bits = int(round(16 - t * 12))
    bits = int(np.clip(bits, 4, 16))
    audio = _peak_normalize(audio)
    if t > 0.4:
        # mu-law encode/decode
        mu = float(2 ** bits - 1)
        x = np.clip(audio, -1.0, 1.0)
        compressed = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
        levels = mu
        q = np.round((compressed + 1.0) * 0.5 * levels) / levels * 2.0 - 1.0
        out = np.sign(q) * (1.0 / mu) * (np.expm1(np.abs(q) * np.log1p(mu)))
        return out.astype(np.float32)
    levels = float(2 ** bits)
    q = np.round(np.clip(audio, -1.0, 1.0) * (levels / 2.0)) / (levels / 2.0)
    return q.astype(np.float32)


def op_reduce_sample_rate(audio: np.ndarray, M: float, M_max: float = 10.0) -> np.ndarray:
    """Downsample then upsample back to 16 kHz (telephony / adaptive streaming)."""
    t = _magnitude_01(M, M_max)
    # target sr: 16k → 4k
    low_sr = int(round(SAMPLING_RATE * (1.0 - 0.75 * t)))
    low_sr = int(np.clip(low_sr, 4000, SAMPLING_RATE))
    if low_sr >= SAMPLING_RATE:
        return audio
    down = resample_audio(audio, SAMPLING_RATE, low_sr)
    return resample_audio(down, low_sr, SAMPLING_RATE)


def op_change_format(audio: np.ndarray, M: float, M_max: float = 10.0) -> np.ndarray:
    """PCM container round-trips (int16 / int8 WAV), not a true file-format change."""
    t = _magnitude_01(M, M_max)
    subtype = "PCM_16" if t < 0.5 else "PCM_U8"
    audio = np.clip(_as_mono_float32(audio), -1.0, 1.0)
    buf = io.BytesIO()
    try:
        sf.write(buf, audio, SAMPLING_RATE, format="WAV", subtype=subtype)
        buf.seek(0)
        out, sr = sf.read(buf, dtype="float32")
        out = _as_mono_float32(out)
        if sr != SAMPLING_RATE:
            out = resample_audio(out, sr, SAMPLING_RATE)
        return out
    except Exception:
        # fallback: int16 cast
        q = (audio * 32767.0).astype(np.int16).astype(np.float32) / 32767.0
        return q


def op_bandlimit(audio: np.ndarray, M: float, M_max: float = 10.0) -> np.ndarray:
    """Moving-average low-pass (band-limit / \"blur\" analog for audio)."""
    t = _magnitude_01(M, M_max)
    # kernel: 1 → ~101 samples (~6 ms at 16k)
    k = int(round(1 + t * 100))
    if k <= 1:
        return audio
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float64) / float(k)
    padded = np.pad(audio.astype(np.float64), (k // 2, k // 2), mode="edge")
    out = np.convolve(padded, kernel, mode="valid")
    return out.astype(np.float32)


def op_change_tempo(audio: np.ndarray, M: float, M_max: float = 10.0) -> np.ndarray:
    """Speed change via interpolate then crop/pad to original length (fps analog)."""
    if len(audio) < 2:
        return audio
    t = _magnitude_01(M, M_max)
    max_delta = 0.3 * t
    if max_delta < 1e-6:
        return audio
    factor = 1.0 + random.uniform(-max_delta, max_delta)
    factor = float(np.clip(factor, 0.7, 1.3))
    new_len = max(1, int(round(len(audio) / factor)))
    x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    xi = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
    stretched = np.interp(xi, x, audio.astype(np.float64)).astype(np.float32)
    n = len(audio)
    if len(stretched) >= n:
        return stretched[:n]
    out = np.zeros(n, dtype=np.float32)
    out[: len(stretched)] = stretched
    out[len(stretched) :] = stretched[-1]
    return out


TEMPORAL_OPS: Sequence[OpSpec] = (
    ("change_tempo", op_change_tempo),
)
SPATIAL_OPS: Sequence[OpSpec] = (
    ("compress", op_compress),
    ("reduce_sample_rate", op_reduce_sample_rate),
    ("change_format", op_change_format),
    ("bandlimit", op_bandlimit),
)
AUGMENT_OPS: Sequence[OpSpec] = tuple(TEMPORAL_OPS) + tuple(SPATIAL_OPS)
_TEMPORAL_NAMES = {name for name, _ in TEMPORAL_OPS}


class AudioRandAugment:
    """RandAugment over online-audio degradation ops (Cubuk et al., 2020)."""

    def __init__(
        self,
        N: int = 2,
        M: float = 9,
        M_max: float = 10.0,
        p: float = 1.0,
        ops: Optional[Sequence[OpSpec]] = None,
    ):
        self.N = int(N)
        self.M = float(M)
        self.M_max = float(M_max)
        self.p = float(p)
        self.ops = list(ops) if ops is not None else list(AUGMENT_OPS)

    def sample_policy(self) -> Optional[List[OpSpec]]:
        if self.N <= 0 or random.random() > self.p:
            return None
        return [random.choice(self.ops) for _ in range(self.N)]

    @staticmethod
    def split_policy(policy: List[OpSpec]) -> Tuple[List[OpSpec], List[OpSpec]]:
        temporal = [op for op in policy if op[0] in _TEMPORAL_NAMES]
        spatial = [op for op in policy if op[0] not in _TEMPORAL_NAMES]
        return temporal, spatial

    def apply_ops(self, audio: np.ndarray, ops: Sequence[OpSpec]) -> np.ndarray:
        for _name, op in ops:
            audio = op(audio, self.M, self.M_max)
            audio = _as_mono_float32(audio)
        return audio

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        policy = self.sample_policy()
        if policy is None:
            return audio
        temporal, spatial = self.split_policy(policy)
        audio = self.apply_ops(audio, temporal)
        audio = self.apply_ops(audio, spatial)
        return audio


# ---------------------------------------------------------------------------
# Audio IO
# ---------------------------------------------------------------------------

def load_audio_mono(path: str, target_sr: int = SAMPLING_RATE, max_seconds: Optional[float] = None) -> np.ndarray:
    """Load mono float32 audio at target_sr. Prefers .wav beside .mp4; else tries path."""
    candidates = []
    if path.endswith(".mp4"):
        candidates.append(path[:-4] + ".wav")
    candidates.append(path)
    if path.endswith(".wav"):
        candidates.append(path[:-4] + ".mp4")

    last_err = None
    audio = None
    sr = None
    for c in candidates:
        if not c or not os.path.isfile(c):
            continue
        try:
            if c.endswith(".wav"):
                audio, sr = sf.read(c, dtype="float32", always_2d=False)
            else:
                # optional torchaudio for video containers
                try:
                    import torchaudio

                    wav, file_sr = torchaudio.load(c)
                    audio = wav.mean(dim=0).numpy().astype(np.float32)
                    sr = int(file_sr)
                except Exception as e:
                    last_err = e
                    continue
            break
        except Exception as e:
            last_err = e
            continue

    if audio is None:
        raise FileNotFoundError(f"Cannot load audio for {path} (last error: {last_err})")

    audio = _as_mono_float32(audio)
    if sr != target_sr:
        audio = resample_audio(audio, int(sr), target_sr)

    if max_seconds is not None and max_seconds > 0:
        max_n = int(max_seconds * target_sr)
        if len(audio) > max_n:
            # take a centered crop (stable vs random for val)
            start = max(0, (len(audio) - max_n) // 2)
            audio = audio[start : start + max_n]
    return audio


def pair_reshape_wav2vec_feats(feats: torch.Tensor) -> torch.Tensor:
    """Match fe_WAV2VEC.py: (T, D) → even T → (T//2, 2*D)."""
    if feats.dim() != 2:
        raise ValueError(f"Expected (T, D) features, got {tuple(feats.shape)}")
    t, d = feats.shape
    if t == 0:
        return feats.new_zeros((0, d * 2))
    if t % 2 != 0:
        feats = torch.cat([feats, feats[-1:]], dim=0)
        t = feats.shape[0]
    return feats.view(t // 2, 2, d).reshape(t // 2, d * 2)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Wav2VecOnlineAudioDataset(Dataset):
    """Raw audio → optional AudioRandAugment → waveform tensor for frozen Wav2Vec2.

    Returns (video_dummy, audio_waveform, label, path) with audio shape (1, N).
    Encoding is done in OnlineWav2VecLinearModel (CUDA outside DataLoader workers).
    """

    def __init__(
        self,
        config: dict,
        split: str = "train",
        augment: Optional[AudioRandAugment] = None,
    ):
        self.config = config
        self.split = split
        self.augment = (
            augment if (split == "train" and config.get("use_randaugment", True)) else None
        )
        self.max_seconds = float(config.get("max_seconds", 10.0))
        # Oversample duration so tempo aug has room before final crop.
        self.decode_max_seconds = float(
            config.get("decode_max_seconds", max(self.max_seconds * 1.5, self.max_seconds))
        )
        self.load_retries = int(config.get("load_retries", 3))
        self.dataset_name = config["dataset_name"]
        self.audio_root = config.get("audio_root_path") or config.get("video_root_path")
        if not self.audio_root:
            raise KeyError("audio_root_path or video_root_path required")
        self.csv_root = config["csv_root_path"]
        self._build_index()

    def _build_index(self):
        name = self.dataset_name
        if name == "AV1M":
            self.df = pd.read_csv(os.path.join(self.csv_root, f"{self.split}_labels.csv"))
            self._path_prefix = self.audio_root
        elif name == "FAVC":
            self.df = pd.read_csv(os.path.join(self.csv_root, f"{self.split}_split.csv"))
            self.df["path"] = self.df["full_path"].apply(lambda x: x.replace("FakeAVCeleb/", ""))
            if self.config.get("fvfa_rvra_only", False):
                self.df["label"] = self.df["category"].map({"A": 0, "D": 1})
                self.df = self.df[self.df["category"].isin(["A", "D"])]
            else:
                self.df["label"] = self.df["category"].map({"A": 0, "B": 1, "C": 1, "D": 1})
            self.df = self.df[~self.df["path"].isin(INVALID_VIDS)]
            self._path_prefix = self.audio_root
        elif name == "AVLips":
            if self.split == "test":
                self.df = pd.read_csv(os.path.join(self.csv_root, "test_labels.csv"))
            else:
                self.df = pd.read_csv(os.path.join(self.csv_root, "metadata.csv"))
                self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
                split_idx = int(0.8 * len(self.df))
                self.df = (
                    self.df.iloc[:split_idx] if self.split == "train" else self.df.iloc[split_idx:]
                )
            self._path_prefix = self.audio_root
        elif name == "BitDF":
            self.df = pd.read_csv(os.path.join(self.csv_root, f"{self.split}_labels.csv"))
            self.df["path"] = self.df["full_file_path"].apply(
                lambda x: str(x).replace("/feats/", "/videos/")
            )
            self.df["label"] = self.df["label"].map({"real": 0, "fake": 1})
            self.df = self.df[self.df["label"].isin([0, 1])]
            self._path_prefix = self.audio_root
        else:
            raise ValueError(f"Unsupported dataset_name for wav2vec online: {name}")

        if name == "AV1M" and self.config.get("fvfa_rvra_only", False):
            meta_path = self.config.get("metadata_path")
            if meta_path and os.path.isfile(meta_path):
                with open(meta_path, "r") as f:
                    metadata = json.load(f)
                set_paths = set(self.df["path"].tolist())
                remove_paths = []
                for md in metadata:
                    if md["file"] in set_paths:
                        both = (
                            len(md["audio_fake_segments"]) > 0
                            and len(md["visual_fake_segments"]) > 0
                        )
                        real = len(md["fake_segments"]) == 0
                        if both or real:
                            continue
                        remove_paths.append(md["file"])
                self.df = self.df[~self.df["path"].isin(set(remove_paths))]

        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _resolve_media_path(self, rel_path: str) -> str:
        rel = str(rel_path)
        candidates = [
            os.path.join(self._path_prefix, rel),
            os.path.join(self.audio_root, rel),
            rel,
        ]
        if rel.endswith(".mp4"):
            candidates = (
                [os.path.join(self._path_prefix, rel[:-4] + ".wav")]
                + candidates
                + [os.path.join(self.audio_root, rel[:-4] + ".wav")]
            )
        for c in candidates:
            if c and os.path.isfile(c):
                return c
        return os.path.join(self._path_prefix, rel)

    def _crop_to_max(self, audio: np.ndarray) -> np.ndarray:
        max_n = int(self.max_seconds * SAMPLING_RATE)
        if max_n <= 0 or len(audio) <= max_n:
            return audio
        if self.split == "train":
            start = random.randint(0, len(audio) - max_n)
        else:
            start = max(0, (len(audio) - max_n) // 2)
        return audio[start : start + max_n]

    def _load_and_augment(self, idx: int):
        row = self.df.iloc[idx]
        rel = row["path"]
        media_path = self._resolve_media_path(rel)
        audio = load_audio_mono(
            media_path, target_sr=SAMPLING_RATE, max_seconds=self.decode_max_seconds
        )

        policy = None
        if self.augment is not None:
            policy = self.augment.sample_policy()
            if policy is not None:
                temporal, spatial = self.augment.split_policy(policy)
                audio = self.augment.apply_ops(audio, temporal)

        audio = self._crop_to_max(audio)

        if self.augment is not None and policy is not None:
            _, spatial = self.augment.split_policy(policy)
            audio = self.augment.apply_ops(audio, spatial)

        wav = torch.from_numpy(_as_mono_float32(audio)).unsqueeze(0)  # (1, N)
        video_dummy = torch.full((1, 1), float("-inf"))
        label = int(row["label"])
        rel_out = rel if str(rel).endswith(".mp4") else str(rel)[:-4] + ".mp4"
        return video_dummy, wav, label, rel_out

    def __getitem__(self, idx):
        tried = {idx}
        last_err = None
        for attempt in range(max(1, self.load_retries)):
            cur = idx if attempt == 0 else random.randrange(len(self.df))
            if attempt > 0 and cur in tried and len(tried) < len(self.df):
                continue
            tried.add(cur)
            try:
                return self._load_and_augment(cur)
            except Exception as e:
                last_err = e
                warnings.warn(f"Failed to load wav2vec sample {cur}: {e}")
        warnings.warn(f"All wav2vec load retries failed (last: {last_err})")
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _build_randaugment(config: dict) -> AudioRandAugment:
    ra = config.get("randaugment", {}) or {}
    return AudioRandAugment(
        N=int(ra.get("N", 2)),
        M=float(ra.get("M", 9)),
        M_max=float(ra.get("M_max", 10)),
        p=float(ra.get("p", 1.0)),
    )


def _fill_single_dataset_paths(cfg: dict, name: str) -> dict:
    cfg = dict(cfg)
    if "audio_root_path" not in cfg:
        key = f"audio_root_path_{name}"
        alt = f"video_root_path_{name}"
        if key in cfg:
            cfg["audio_root_path"] = cfg[key]
        elif "video_root_path" in cfg:
            cfg["audio_root_path"] = cfg["video_root_path"]
        elif alt in cfg:
            cfg["audio_root_path"] = cfg[alt]
    if "csv_root_path" not in cfg:
        cfg["csv_root_path"] = cfg[f"csv_root_path_{name}"]
    return cfg


def load_data_online_wav2vec(config: dict):
    """Build train/val loaders for on-the-fly Wav2Vec2 + RandAugment."""
    augment = _build_randaugment(config)
    dataset_map = {
        "AV1M": Wav2VecOnlineAudioDataset,
        "FAVC": Wav2VecOnlineAudioDataset,
        "AVLips": Wav2VecOnlineAudioDataset,
        "BitDF": Wav2VecOnlineAudioDataset,
    }

    train_sampler = None

    if config["dataset_name"] == "all":
        train_datasets, val_datasets = [], []
        for name, cls in dataset_map.items():
            audio_key = f"audio_root_path_{name}"
            video_key = f"video_root_path_{name}"
            csv_key = f"csv_root_path_{name}"
            if csv_key not in config:
                continue
            if audio_key not in config and video_key not in config:
                continue
            sub = dict(config)
            sub["audio_root_path"] = config.get(audio_key, config.get(video_key))
            sub["csv_root_path"] = config[csv_key]
            sub["dataset_name"] = name
            if f"metadata_path_{name}" in config:
                sub["metadata_path"] = config[f"metadata_path_{name}"]
            print(f"[online wav2vec] Adding {name} from {sub['audio_root_path']}")
            train_ds = cls(sub, split="train", augment=augment)
            val_ds = cls(sub, split="val", augment=None)
            print(f"  train={len(train_ds)} val={len(val_ds)}")
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)

        if not train_datasets:
            raise ValueError("No datasets were configured for dataset_name='all'")

        train_ds = ConcatDataset(train_datasets)
        val_ds = ConcatDataset(val_datasets)

        sizes = [len(d) for d in train_ds.datasets]
        weights = []
        for size in sizes:
            weights += [1.0 / size] * size
        train_sampler = WeightedRandomSampler(
            torch.DoubleTensor(weights), num_samples=len(weights), replacement=True
        )
    else:
        name = config["dataset_name"]
        if name not in dataset_map:
            raise ValueError(f"Unknown dataset_name: {name}")
        cfg = _fill_single_dataset_paths(config, name)
        train_ds = dataset_map[name](cfg, split="train", augment=augment)
        val_ds = dataset_map[name](cfg, split="val", augment=None)

    def collate_skip_none(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return torch.utils.data.default_collate(
                [(torch.empty(0), torch.empty(0), torch.tensor(0), "")]
            )
        # Variable-length waveforms: batch_size must be 1
        return torch.utils.data.default_collate(batch)

    num_workers = int(config.get("num_workers", 4))
    if train_sampler is not None:
        train_dl = DataLoader(
            train_ds,
            batch_size=1,
            sampler=train_sampler,
            collate_fn=collate_skip_none,
            num_workers=num_workers,
        )
    else:
        train_dl = DataLoader(
            train_ds,
            batch_size=1,
            shuffle=True,
            collate_fn=collate_skip_none,
            num_workers=num_workers,
        )
    val_dl = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_skip_none,
        num_workers=num_workers,
    )
    return train_dl, val_dl


# ---------------------------------------------------------------------------
# Lightning: frozen Wav2Vec2 + linear probe
# ---------------------------------------------------------------------------

class OnlineWav2VecLinearModel(L.LightningModule):
    """Frozen Wav2Vec2 encoder + linear head (audio-only).

    Expects batch audio waveforms (B=1, 1, N) @ 16 kHz.
    Applies fe_WAV2VEC pair-reshape so feats_dim matches disk features (3840 for xls-r-2b).
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.full_config = config
        mh = config["model_hparams"]
        self.feats_dim = int(mh["feats_dim"])
        self.apply_l2 = bool(config.get("data_info", {}).get("apply_l2", True))
        self.pair_reshape = bool(config.get("data_info", {}).get("pair_reshape", True))
        model_name = _short_to_hf(resolve_wav2vec_model_name(config))

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.wav2vec = Wav2Vec2Model.from_pretrained(model_name)
        self.wav2vec.eval()
        for p in self.wav2vec.parameters():
            p.requires_grad = False

        self.head = nn.Linear(self.feats_dim, 1)
        self._feats_dim_checked = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.wav2vec.eval()
        return self

    def encode_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (1, N) or (B, 1, N) with B=1 → (T, D_out)."""
        if waveform.dim() == 3:
            waveform = waveform[0]
        if waveform.dim() == 2 and waveform.shape[0] == 1:
            wav_np = waveform[0].detach().cpu().numpy()
        elif waveform.dim() == 1:
            wav_np = waveform.detach().cpu().numpy()
        else:
            raise ValueError(f"Unexpected waveform shape {tuple(waveform.shape)}")

        if wav_np.size == 0:
            return waveform.new_zeros((0, self.feats_dim))

        inputs = self.feature_extractor(
            wav_np,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self.wav2vec.eval()
        with torch.no_grad():
            outputs = self.wav2vec(**inputs)
            feats = outputs.last_hidden_state[0].float()  # (T, D)

        if self.pair_reshape:
            feats = pair_reshape_wav2vec_feats(feats)

        if not self._feats_dim_checked:
            if feats.shape[-1] != self.feats_dim:
                raise RuntimeError(
                    f"Wav2Vec output dimension {feats.shape[-1]} does not match "
                    f"feats_dim={self.feats_dim}"
                )
            self._feats_dim_checked = True

        if self.apply_l2:
            feats = F.normalize(feats, p=2, dim=-1)
        return feats

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        audio_feats = self.encode_audio(waveform)
        if audio_feats.shape[0] == 0:
            return waveform.new_zeros(())
        logits = self.head(audio_feats)[:, 0]
        # Same temporal pool as disk-feature LinearModel
        return torch.logsumexp(logits, dim=-1)

    def _step(self, batch, stage: str):
        _video, audio, labels, _paths = batch
        if audio.numel() == 0:
            return None
        score = self.forward(audio)
        score = score.view(1)
        logits = torch.stack((-score, score), dim=1)
        loss = F.cross_entropy(logits, labels.view(-1))
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=(stage == "val"))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.head.parameters(), lr=1e-3)

    def predict_scores(self, video_feats=None, audio_feats=None):
        # audio_feats here is raw waveform when used online
        return self.forward(audio_feats).view(-1)
