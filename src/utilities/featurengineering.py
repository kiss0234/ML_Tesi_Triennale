import pandas as pd
from typing import List
import pickle

def get_features(dfs : List[pd.DataFrame]):

    all_features = set()

    for df in dfs:
        all_features.update(df.columns)
    
    with open("resources/features.pkl", "wb") as file:
        pickle.dump(sorted(all_features), file)

def get_all_features_set():
    with open("resources/features.pkl", "rb") as file:
        full_feature_set = pickle.load(file)
    return full_feature_set

def align_features(df : pd.DataFrame, all_features):
    df_aligned = df.reindex(columns=all_features, fill_value=0)
    return df_aligned