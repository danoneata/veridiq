from pathlib import Path

import json
import pdb

import pandas as pd


class AV1M:
    def __init__(self, split):
        assert split in {"train", "val"}
        self.split = split
        self.root = Path("/data/av-deepfake-1m/av_deepfake_1m")

    def load_filelist(self):
        path = self.root / f"{self.split}_metadata.json"
        with open(path, "r") as f:
            return json.load(fm)

    def get_video_path(self, file):
        path = self.root / self.split / file
        return str(path)


class ExDDV:
    def __init__(self):
        pass

    def load_metadata(self):
        df_reals = pd.read_csv("data/exddv/ExDDV_reals_FF++.csv")
        df_fakes = pd.read_csv("data/exddv/ExDDV_with_full_path.csv")
        return {"real": df_reals, "fake": df_fakes}

    def get_video_paths(self) -> list:
        metadata = self.load_metadata()
        paths_real = [row.full_path for row in metadata["real"].itertuples()]
        paths_fake = [row.full_path for row in metadata["fake"].itertuples()]
        paths = paths_real + paths_fake
        paths = ["/data" + path for path in paths]
        return paths


DATASETS = {
    # "fakeavceleb": FakeAVCeleb,
    "av1m": AV1M,
    "exddv": ExDDV,
}


if __name__ == "__main__":
    import streamlit as st

    # dataset = AV1M("val")
    # data = dataset.load_filelist()

    # st.write(data[0])
    # st.video(dataset.get_video_path(data[0]["file"]))

    def func(x):
        num_annotators = len(x.username.unique())
        if num_annotators >= 2:
            pdb.set_trace()
        else:
            return 1

    dataset = ExDDV("val")
    metadata = dataset.load_metadata()
    for row in metadata["real"].sample(10).itertuples():
        st.write(row)
        st.video(row.full_path)
    # print(metadata.columns)
    # metadata.groupby("movie_name").apply(func)
