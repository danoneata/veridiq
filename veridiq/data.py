from pathlib import Path
import json


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


DATASETS = {
    "av1m": AV1M,
}


if __name__ == "__main__":
    import streamlit as st

    dataset = AV1M("val")
    data = dataset.load_filelist()

    st.write(data[0])
    st.video(dataset.get_video_path(data[0]["file"]))
