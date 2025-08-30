import json
import pdb

from toolz import dissoc, take
from tqdm import tqdm

from veridiq.explanations.generate_spatial_explanations import get_exddv_images
import face_recognition


def extract_landmarks(data):
    image = data["frame"]
    locations = face_recognition.face_locations(image)
    landmarks = face_recognition.face_landmarks(image)
    return {
        "frame-size": image.shape[:2],
        "num-faces": len(locations),
        "face-locations": locations,
        "face-landmarks": landmarks,
    }


data = [
    {
        **extract_landmarks(image),
        **dissoc(image, "frame"),
    }
    for image in tqdm(get_exddv_images())
]

with open("output/exddv/extracted-faces.json", "w") as f:
    json.dump(data, f, indent=4)
