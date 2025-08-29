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
            return json.load(f)

    def get_video_path(self, file):
        path = self.root / self.split / file
        return str(path)


class ExDDV:
    @staticmethod
    def load_metadata():
        df_reals = pd.read_csv("data/exddv/ExDDV_reals_FF++.csv")
        df_reals["name"] = "real/" + df_reals["movie_name"]
        df_reals["path"] = "/data" + df_reals["full_path"]
        df_reals["label"] = "real"

        df_fakes = pd.read_csv("data/exddv/ExDDV_with_full_path.csv")
        df_fakes["name"] = (
            df_fakes["dataset"]
            + "/"
            + df_fakes["manipulation"]
            + "/"
            + df_fakes["movie_name"]
        )
        df_fakes["path"] = "/data" + df_fakes["full_path"]
        df_fakes["label"] = "fake"

        return df_reals, df_fakes

    @staticmethod
    def get_videos() -> list[dict]:
        df_reals, df_fakes = ExDDV.load_metadata()

        cols = ["name", "path", "split", "label", "text", "click_locations"]
        videos_fake = df_fakes[cols]
        videos_fake = videos_fake.drop_duplicates(subset=["path"])
        videos_fake = videos_fake.to_dict(orient="records")
        for datum in videos_fake:
            clicks = datum.pop("click_locations")
            clicks = json.loads(clicks)
            datum["clicks"] = [{"frame-idx": int(k), **v} for k, v in clicks.items()]

        cols = ["name", "path", "split", "label"]
        videos_real = df_reals[cols]
        videos_real = videos_real.to_dict(orient="records")

        return videos_fake + videos_real


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
