import random
import pandas as pd


def count_splits(df):
    split_counts = df["split"].value_counts()
    split_counts = split_counts.reset_index()
    split_counts["proportion"] = split_counts["count"] / split_counts["count"].sum()
    print(split_counts)


df_fakes = pd.read_csv("data/exddv/ExDDV_with_full_path.csv")
print("Split proportions for fakes")
count_splits(df_fakes)

df_reals = pd.read_csv("data/exddv/ExDDV_reals_FF++.csv")
num_reals = len(df_reals)
num_train = int(0.8 * num_reals)
num_valid = int(0.1 * num_reals)
num_test = num_reals - num_train - num_valid
split_labels = num_train * ["train"] + num_valid * ["val"] + num_test * ["test"]
random.shuffle(split_labels)

df_reals["split"] = split_labels
df_reals.to_csv("data/exddv/ExDDV_reals_FF++.csv", index=False)

print("Split proportions for reals")
count_splits(df_reals)