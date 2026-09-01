"""On-the-fly CLIP feature extraction with RandAugment-style online video degradations.

Simulates artifacts from online video distribution (Cubuk et al., CVPR 2020 RandAugment):
  JPEG/WebP image compression, resolution reduction, chroma degradation, blur,
  framerate change. (Frame-level proxies — not full H.264/H.265 video re-encodes.)

Applied dynamically at train time: N ops are sampled uniformly (with replacement)
from the pool and applied sequentially at magnitude M.

Pipeline (train):
  decode (bounded) → temporal augs → sample max_frames → spatial augs → CLIP preprocess
  → GPU frozen CLIP encode → linear head (logsumexp pool, same as disk-feature LinearModel)

Usage (from linear_probing/):
  export PYTHONPATH=.../veridiq_2026/veridiq
  python train_test.py --config_path configs/train_config_all_clip_online.yaml
"""

from __future__ import annotations

import io
import json
import os
import random
import warnings
from typing import Callable, List, Optional, Sequence, Tuple

import clip
import cv2
import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from veridiq.linear_probing.datasets import INVALID_VIDS

OpFn = Callable[[List[np.ndarray], float, float], List[np.ndarray]]
OpSpec = Tuple[str, OpFn]


def resolve_clip_model_name(config: dict) -> str:
    """Single source for CLIP name: data_info.clip_model or top-level clip_model."""
    if "clip_model" in config and config["clip_model"]:
        return str(config["clip_model"])
    data_info = config.get("data_info") or {}
    return str(data_info.get("clip_model", "ViT-L/14"))


# ---------------------------------------------------------------------------
# RandAugment-style online distribution degradations
# ---------------------------------------------------------------------------

def _magnitude_01(M: float, M_max: float = 10.0) -> float:
    """Map RandAugment magnitude M in [0, M_max] to [0, 1]."""
    return float(np.clip(M / M_max, 0.0, 1.0))


def op_compress(frames: List[np.ndarray], M: float, M_max: float = 10.0) -> List[np.ndarray]:
    """Per-frame JPEG recompression (image-level CDN/upload proxy, not H.264)."""
    t = _magnitude_01(M, M_max)
    quality = int(np.clip(round(95 - t * 80), 10, 95))
    out = []
    for frame in frames:
        ok, buf = cv2.imencode(
            ".jpg",
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if not ok:
            out.append(frame)
            continue
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        out.append(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
    return out


def op_reduce_resolution(frames: List[np.ndarray], M: float, M_max: float = 10.0) -> List[np.ndarray]:
    """Downsample then upsample back (adaptive streaming / thumbnail upscale)."""
    t = _magnitude_01(M, M_max)
    scale = float(np.clip(1.0 - t * 0.75, 0.2, 1.0))
    if scale >= 0.999:
        return frames
    out = []
    for frame in frames:
        h, w = frame.shape[:2]
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        out.append(restored)
    return out


def op_change_format(frames: List[np.ndarray], M: float, M_max: float = 10.0) -> List[np.ndarray]:
    """Image-format / chroma degradations (WebP re-encode or chroma subsample).

    Not a video-container conversion (MP4↔WebM); frames are already decoded.
    factor=2 ≈ 4:2:0; factor=4 is a stronger magnitude-dependent chroma loss.
    """
    t = _magnitude_01(M, M_max)
    mode = random.choice(["webp", "chroma420"])
    out = []
    for frame in frames:
        if mode == "webp":
            quality = int(np.clip(round(95 - t * 70), 10, 95))
            pil = Image.fromarray(frame)
            buf = io.BytesIO()
            try:
                pil.save(buf, format="WEBP", quality=quality)
                buf.seek(0)
                out.append(np.array(Image.open(buf).convert("RGB")))
            except Exception:
                out.extend(op_compress([frame], M, M_max))
        else:
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            factor = 4 if t > 0.5 else 2
            h, w = cr.shape
            cr_s = cv2.resize(
                cr, (max(1, w // factor), max(1, h // factor)), interpolation=cv2.INTER_AREA
            )
            cb_s = cv2.resize(
                cb, (max(1, w // factor), max(1, h // factor)), interpolation=cv2.INTER_AREA
            )
            cr_u = cv2.resize(cr_s, (w, h), interpolation=cv2.INTER_LINEAR)
            cb_u = cv2.resize(cb_s, (w, h), interpolation=cv2.INTER_LINEAR)
            merged = cv2.merge([y, cr_u, cb_u])
            out.append(cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB))
    return out


def op_blur(frames: List[np.ndarray], M: float, M_max: float = 10.0) -> List[np.ndarray]:
    """Gaussian blur (motion / focus / heavy re-encode softening)."""
    t = _magnitude_01(M, M_max)
    k = int(round(1 + t * 14))
    if k % 2 == 0:
        k += 1
    k = max(1, k)
    if k <= 1:
        return frames
    sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
    return [cv2.GaussianBlur(f, (k, k), sigmaX=sigma) for f in frames]


def op_change_framerate(frames: List[np.ndarray], M: float, M_max: float = 10.0) -> List[np.ndarray]:
    """Temporal downsample then hold-frame upsample (fps conversion proxy)."""
    if len(frames) <= 1:
        return frames
    t = _magnitude_01(M, M_max)
    keep_ratio = float(np.clip(1.0 - t * 0.75, 0.2, 1.0))
    n = len(frames)
    keep_n = max(1, int(round(n * keep_ratio)))
    if keep_n >= n:
        return frames
    indices = np.linspace(0, n - 1, keep_n).astype(int)
    sampled = [frames[i] for i in indices]
    out = []
    for i in range(n):
        src = int(round(i * (keep_n - 1) / max(n - 1, 1)))
        src = int(np.clip(src, 0, keep_n - 1))
        out.append(sampled[src])
    return out


TEMPORAL_OPS: Sequence[OpSpec] = (
    ("change_framerate", op_change_framerate),
)
SPATIAL_OPS: Sequence[OpSpec] = (
    ("compress", op_compress),
    ("reduce_resolution", op_reduce_resolution),
    ("change_format", op_change_format),
    ("blur", op_blur),
)
AUGMENT_OPS: Sequence[OpSpec] = tuple(TEMPORAL_OPS) + tuple(SPATIAL_OPS)
_TEMPORAL_NAMES = {name for name, _ in TEMPORAL_OPS}


class VideoRandAugment:
    """RandAugment (Cubuk et al., 2020) over online-video degradation ops.

    Parameters
    ----------
    N : int
        Number of sequential transforms (sampled with replacement from the pool).
    M : float
        Shared magnitude in [0, M_max].
    M_max : float
        Maximum magnitude used to normalize M.
    p : float
        Probability of applying the policy at all (else identity).
    """

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

    def apply_ops(self, frames: List[np.ndarray], ops: Sequence[OpSpec]) -> List[np.ndarray]:
        for _name, op in ops:
            frames = op(frames, self.M, self.M_max)
        return frames

    def __call__(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply a full random policy in one shot (temporal then spatial order)."""
        if not frames:
            return frames
        policy = self.sample_policy()
        if policy is None:
            return frames
        temporal, spatial = self.split_policy(policy)
        frames = self.apply_ops(frames, temporal)
        frames = self.apply_ops(frames, spatial)
        return frames


# ---------------------------------------------------------------------------
# Video IO + CLIP preprocess
# ---------------------------------------------------------------------------

def load_video_frames_rgb(path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """Load video as RGB uint8 frames; optionally stride-decode to at most max_frames."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    frames: List[np.ndarray] = []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if max_frames is not None and total > max_frames > 0:
            keep = set(np.linspace(0, total - 1, max_frames).astype(int).tolist())
            i = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if i in keep:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                i += 1
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    if max_frames is not None and len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in idx]
    return frames


def _subsample_frames(frames: List[np.ndarray], max_frames: int) -> List[np.ndarray]:
    if len(frames) <= max_frames:
        return frames
    sel = np.linspace(0, len(frames) - 1, max_frames).astype(int)
    return [frames[i] for i in sel]


def frames_to_clip_tensor(frames: List[np.ndarray], preprocess) -> torch.Tensor:
    """Apply CLIP preprocess to each RGB frame -> (T, 3, 224, 224)."""
    tensors = [preprocess(Image.fromarray(f)) for f in frames]
    return torch.stack(tensors, dim=0)


# ---------------------------------------------------------------------------
# Dataset: raw videos -> (optionally augmented) CLIP-ready frames
# ---------------------------------------------------------------------------

class CLIPOnlineVideoDataset(Dataset):
    """Load raw videos, apply RandAugment degradations (train), return CLIP tensors.

    Returns the same 4-tuple shape as disk-feature datasets:
      video_frames (T,3,H,W), audio_dummy, label, path

    CLIP encoding happens in OnlineCLIPLinearModel (keeps CUDA out of workers).
    """

    def __init__(
        self,
        config: dict,
        split: str = "train",
        preprocess=None,
        augment: Optional[VideoRandAugment] = None,
    ):
        self.config = config
        self.split = split
        self.preprocess = preprocess
        self.augment = (
            augment if (split == "train" and config.get("use_randaugment", True)) else None
        )
        self.max_frames = config.get("max_frames", 64)
        # Decode a bounded oversample so temporal augs have room, without full-video cost.
        default_decode = (
            max(int(self.max_frames) * 4, 256) if self.max_frames else None
        )
        self.decode_max_frames = config.get("decode_max_frames", default_decode)
        self.load_retries = int(config.get("load_retries", 3))
        self.dataset_name = config["dataset_name"]
        self.video_root = config["video_root_path"]
        self.csv_root = config["csv_root_path"]
        self._build_index()

    def _build_index(self):
        name = self.dataset_name
        if name == "AV1M":
            self.df = pd.read_csv(os.path.join(self.csv_root, f"{self.split}_labels.csv"))
            self._path_prefix = self.video_root
        elif name == "FAVC":
            self.df = pd.read_csv(os.path.join(self.csv_root, f"{self.split}_split.csv"))
            self.df["path"] = self.df["full_path"].apply(lambda x: x.replace("FakeAVCeleb/", ""))
            if self.config.get("fvfa_rvra_only", False):
                self.df["label"] = self.df["category"].map({"A": 0, "D": 1})
                self.df = self.df[self.df["category"].isin(["A", "D"])]
            else:
                self.df["label"] = self.df["category"].map({"A": 0, "B": 1, "C": 1, "D": 1})
            self.df = self.df[~self.df["path"].isin(INVALID_VIDS)]
            self._path_prefix = self.video_root
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
            self._path_prefix = self.video_root
        elif name == "BitDF":
            self.df = pd.read_csv(os.path.join(self.csv_root, f"{self.split}_labels.csv"))
            self.df["path"] = self.df["full_file_path"].apply(
                lambda x: str(x).replace("/feats/", "/videos/")
            )
            self.df["label"] = self.df["label"].map({"real": 0, "fake": 1})
            self.df = self.df[self.df["label"].isin([0, 1])]
            self._path_prefix = self.video_root
        else:
            raise ValueError(f"Unsupported dataset_name for CLIP online: {name}")

        if name == "AV1M" and self.config.get("fvfa_rvra_only", False):
            meta_path = self.config.get("metadata_path")
            if meta_path and os.path.isfile(meta_path):
                with open(meta_path, "r") as f:
                    metadata = json.load(f)
                set_paths = set(self.df["path"].tolist())
                remove_paths = []
                for md in metadata:
                    if md["file"] in set_paths:
                        both = len(md["audio_fake_segments"]) > 0 and len(md["visual_fake_segments"]) > 0
                        real = len(md["fake_segments"]) == 0
                        if both or real:
                            continue
                        remove_paths.append(md["file"])
                self.df = self.df[~self.df["path"].isin(set(remove_paths))]

        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _resolve_video_path(self, rel_path: str) -> str:
        rel = str(rel_path)
        candidates = [
            os.path.join(self._path_prefix, rel),
            os.path.join(self.video_root, rel),
            rel,
        ]
        if rel.endswith((".npy", ".npz")):
            candidates.insert(0, os.path.join(self._path_prefix, rel[:-4] + ".mp4"))
        for c in candidates:
            if c and os.path.isfile(c):
                return c
        return os.path.join(self._path_prefix, rel)

    def _load_and_augment(self, idx: int):
        row = self.df.iloc[idx]
        rel = row["path"]
        video_path = self._resolve_video_path(rel)
        frames = load_video_frames_rgb(video_path, max_frames=self.decode_max_frames)

        policy = None
        if self.augment is not None:
            policy = self.augment.sample_policy()
            if policy is not None:
                temporal, spatial = self.augment.split_policy(policy)
                frames = self.augment.apply_ops(frames, temporal)

        if self.max_frames is not None:
            frames = _subsample_frames(frames, int(self.max_frames))

        if self.augment is not None and policy is not None:
            _, spatial = self.augment.split_policy(policy)
            frames = self.augment.apply_ops(frames, spatial)

        if self.preprocess is None:
            raise RuntimeError("CLIP preprocess was not provided to CLIPOnlineVideoDataset")

        video = frames_to_clip_tensor(frames, self.preprocess)
        label = int(row["label"])
        audio = torch.full((video.shape[0], 1), float("-inf"))
        rel_out = rel if str(rel).endswith(".mp4") else str(rel)[:-4] + ".mp4"
        return video, audio, label, rel_out

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
                warnings.warn(f"Failed to load sample {cur}: {e}")
        warnings.warn(f"All load retries failed (last: {last_err}); returning None")
        return None


# ---------------------------------------------------------------------------
# Data loading (multi-dataset + train WeightedRandomSampler)
# ---------------------------------------------------------------------------

def _build_randaugment(config: dict) -> VideoRandAugment:
    ra = config.get("randaugment", {}) or {}
    return VideoRandAugment(
        N=int(ra.get("N", 2)),
        M=float(ra.get("M", 9)),
        M_max=float(ra.get("M_max", 10)),
        p=float(ra.get("p", 1.0)),
    )


def load_clip_preprocess(clip_model_name: str = "ViT-L/14", device: str = "cpu"):
    """Load CLIP solely for its preprocess transform (encoder loaded later on GPU)."""
    _, preprocess = clip.load(clip_model_name, device=device)
    return preprocess


def _fill_single_dataset_paths(cfg: dict, name: str) -> dict:
    cfg = dict(cfg)
    if "video_root_path" not in cfg:
        cfg["video_root_path"] = cfg[f"video_root_path_{name}"]
    if "csv_root_path" not in cfg:
        cfg["csv_root_path"] = cfg[f"csv_root_path_{name}"]
    return cfg


def load_data_online_clip(config: dict, preprocess=None):
    """Build train/val loaders for on-the-fly CLIP + RandAugment."""
    if preprocess is None:
        preprocess = load_clip_preprocess(resolve_clip_model_name(config), device="cpu")

    augment = _build_randaugment(config)
    dataset_map = {
        "AV1M": CLIPOnlineVideoDataset,
        "FAVC": CLIPOnlineVideoDataset,
        "AVLips": CLIPOnlineVideoDataset,
        "BitDF": CLIPOnlineVideoDataset,
    }

    train_sampler = None

    if config["dataset_name"] == "all":
        train_datasets, val_datasets = [], []
        for name, cls in dataset_map.items():
            root_key = f"video_root_path_{name}"
            csv_key = f"csv_root_path_{name}"
            if root_key not in config or csv_key not in config:
                continue
            sub = dict(config)
            sub["video_root_path"] = config[root_key]
            sub["csv_root_path"] = config[csv_key]
            sub["dataset_name"] = name
            if f"metadata_path_{name}" in config:
                sub["metadata_path"] = config[f"metadata_path_{name}"]
            print(f"[online CLIP] Adding {name} from {sub['video_root_path']}")
            train_ds = cls(sub, split="train", preprocess=preprocess, augment=augment)
            val_ds = cls(sub, split="val", preprocess=preprocess, augment=None)
            print(f"  train={len(train_ds)} val={len(val_ds)}")
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)

        if not train_datasets:
            raise ValueError("No datasets were configured for dataset_name='all'")

        train_ds = ConcatDataset(train_datasets)
        val_ds = ConcatDataset(val_datasets)

        # Balance datasets on train only; validate over the full set deterministically.
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
        train_ds = dataset_map[name](cfg, split="train", preprocess=preprocess, augment=augment)
        val_ds = dataset_map[name](cfg, split="val", preprocess=preprocess, augment=None)

    def collate_skip_none(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return torch.utils.data.default_collate(
                [(torch.empty(0), torch.empty(0), torch.tensor(0), "")]
            )
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
# Lightning module: frozen CLIP encode (on the fly) + linear probe
# ---------------------------------------------------------------------------

class OnlineCLIPLinearModel(L.LightningModule):
    """Frozen CLIP image encoder + linear deepfake head.

    Expects batches of preprocessed frames (B=1, T, 3, 224, 224).
    Features are L2-normalized to match the disk-feature training setup.
    Temporal pool is logsumexp (same as LinearModel on pre-extracted features).
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.full_config = config
        mh = config["model_hparams"]
        self.feats_dim = int(mh["feats_dim"])
        self.apply_l2 = bool(config.get("data_info", {}).get("apply_l2", True))
        self.chunk_size = int(config.get("data_info", {}).get("clip_chunk_size", 32))
        clip_name = resolve_clip_model_name(config)

        self.clip_model, _ = clip.load(clip_name, device="cpu")
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.head = nn.Linear(self.feats_dim, 1)
        self._feats_dim_checked = False

    def train(self, mode: bool = True):
        """Keep frozen CLIP in eval() even when Lightning sets the module to train."""
        super().train(mode)
        self.clip_model.eval()
        return self

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (T, 3, H, W) or (B, T, 3, H, W) with B=1 -> (T, D)."""
        if frames.dim() == 5:
            frames = frames[0]
        if frames.numel() == 0:
            return frames.new_zeros((0, self.feats_dim))

        self.clip_model.eval()
        feats = []
        with torch.no_grad():
            for i in range(0, frames.shape[0], self.chunk_size):
                chunk = frames[i : i + self.chunk_size]
                f = self.clip_model.encode_image(chunk).float()
                feats.append(f)
        video = torch.cat(feats, dim=0)

        if not self._feats_dim_checked:
            if video.shape[-1] != self.feats_dim:
                raise RuntimeError(
                    f"CLIP output dimension {video.shape[-1]} does not match "
                    f"feats_dim={self.feats_dim}"
                )
            self._feats_dim_checked = True

        if self.apply_l2:
            video = F.normalize(video, p=2, dim=-1)
        return video

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        video = self.encode_frames(frames)
        if video.shape[0] == 0:
            return frames.new_zeros(())
        logits = self.head(video)[:, 0]
        # Match disk-feature LinearModel pooling (length-dependent by design).
        return torch.logsumexp(logits, dim=-1)

    def _step(self, batch, stage: str):
        video_frames, _audio, labels, _paths = batch
        if video_frames.numel() == 0:
            return None
        score = self.forward(video_frames)
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

    def predict_scores(self, video_frames, audio_feats=None):
        return self.forward(video_frames).view(-1)
