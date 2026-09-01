#!/usr/bin/env python
"""Print AUC x 100 for the cross-dataset linear-probing table from outputs/table/."""

from __future__ import annotations

import re
from pathlib import Path

LP_DIR = Path(__file__).resolve().parents[1]
OUT = LP_DIR / "outputs" / "table"

# (row_label, train, feature_key) -> expected eval column order
EVALS = ["AV1M", "FAVC", "BitDF", "DFE", "MAVOS"]

ROWS = [
    ("BRAVEn-V", "AV1M", "braven_v"),
    ("BRAVEn-A", "AV1M", "braven_a"),
    ("BRAVEn-A+V (late fusion)", "AV1M", "braven_late"),
    ("BRAVEn-A+V (early fusion)", "AV1M", "braven_early"),
    ("BRAVEn-V", "FAVC", "braven_v"),
    ("BRAVEn-A", "FAVC", "braven_a"),
    ("BRAVEn-A+V (late fusion)", "FAVC", "braven_late"),
    ("BRAVEn-A+V (early fusion)", "FAVC", "braven_early"),
    ("CLIP", "FAVC", "clip"),
    ("WAV2VEC2", "FAVC", "wav2vec"),
    ("CLIP+WAV2VEC (late fusion)", "FAVC", "clip_wav2vec_late"),
    ("CLIP+WAV2VEC (early fusion)", "FAVC", "clip_wav2vec_early"),
]

# eval name in table -> folder suffix
EVAL_SUFFIX = {
    "AV1M": "av1m",
    "FAVC": "favc",
    "BitDF": "bitdf",
    "DFE": "dfe",
    "MAVOS": "mavos",
}

# Features that do not evaluate on MAVOS in the paper table
SKIP_MAVOS = {"clip", "wav2vec", "clip_wav2vec_late", "clip_wav2vec_early"}


def read_auc(path: Path) -> float | None:
    if not path.is_file():
        return None
    text = path.read_text()
    m = re.search(r"AUC:\s*([0-9.]+)", text)
    if not m:
        return None
    return float(m.group(1)) * 100.0


def cell(train: str, feature: str, eval_name: str) -> str:
    if eval_name == "MAVOS" and feature in SKIP_MAVOS:
        return "-"
    train_l = train.lower()
    eval_s = EVAL_SUFFIX[eval_name]
    folder = OUT / f"{train_l}_{eval_s}_{feature}"
    auc = read_auc(folder / "eval_results.txt")
    if auc is None:
        return "NA"
    return f"{auc:.1f}"


def main() -> None:
    header = ["Features", "Train set", *EVALS]
    rows = [header]
    for label, train, feature in ROWS:
        rows.append([label, train, *[cell(train, feature, e) for e in EVALS]])

    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    for r in rows:
        print("  ".join(val.ljust(widths[i]) for i, val in enumerate(r)))


if __name__ == "__main__":
    main()
