from sklearn.cluster import AgglomerativeClustering
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import ast
import os
import sys

tqdm.pandas()

actions_df_path = Path("./processed/actions_day_1.csv")
actions_df = pd.read_csv(actions_df_path)



def get_group_size(row):
    return max(ast.literal_eval(row))

gs = actions_df["groupSize"].progress_apply(get_group_size)


duration = list(actions_df["end_time"] - actions_df["time"])

points = np.array(list(zip(gs, duration)), dtype=np.int64)

ac = AgglomerativeClustering(n_clusters=100)

ac.fit(points)

import pickle as pkl

with open("ac_model.pkl", "wb") as handle:
    pkl.dump(ac, handle)
