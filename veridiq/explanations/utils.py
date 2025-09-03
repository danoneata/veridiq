import pandas as pd

def load_predictions(config_name):
    path = f"output/training-linear-probing/{config_name}/test/results.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"paths": "name", "scores": "pred-score", "labels": "label"})
    df = df.set_index("name")
    return df.to_dict(orient="index")
