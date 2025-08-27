import pandas as pd
from typing import List
import pickle
from utilities.config_manager import ConfigManager
import logging

logger = logging.getLogger(__name__)

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

def load_feature_types():
    logger.info("Loading feature types..")

    config_manager = ConfigManager()
    config_manager.load_config("config/dataset.json")
    numerical_columns = config_manager.get_value("dataset", "numeric_columns")
    categorical_columns = config_manager.get_value("dataset", "categorical_columns")
    target_column = config_manager.get_value("dataset", "target_column")

    logger.info("Loading complete")

    return [numerical_columns, categorical_columns, target_column]