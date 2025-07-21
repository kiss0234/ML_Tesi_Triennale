import logging
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Dict
from utilities.config_manager import ConfigManager
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from data.scalers.LogMinMaxScaler import LogMinMaxScaler
from data.scalers.custom_imputer import CustomImputer
from data.scalers.frequency_encoder import FrequencyEncoder
 
logger = logging.getLogger(__name__)
MAX_DATASETS = 5

def remove_attacks_under_threshold(datasets, threshold) -> None:
    for host, dataset in datasets.items():
        attacks = dataset["Attack"].value_counts()
        attacks_under_threshold = attacks[attacks < threshold].index
        print(f"{host} : {attacks_under_threshold}")

        for attack in attacks_under_threshold:
            dataset.drop(dataset[dataset["Attack"] == attack].index, inplace=True)


def divide_dataset(df) -> Dict[str, pd.DataFrame]:

    #Identifico il numero di righe per traffico inviato per host
    rows_per_src_host = df["IPV4_SRC_ADDR"].value_counts()

    #Identifico il numero di righe per traffico ricevuto per host
    rows_per_dst_host = df["IPV4_DST_ADDR"].value_counts()

    #Ottengo il traffico per ogni host
    traffic_per_host = rows_per_dst_host.add(rows_per_src_host, fill_value=0).sort_values(ascending=False)

    #Divido il dataset in sotto dataset per indirizzo ip relativo a traffico in entrata ed in uscita
    logger.info("Dividing dataset in smaller datasets per ip..")

    datasets = {}
    for host in traffic_per_host.index[:MAX_DATASETS]:
        datasets[host] = df[df["IPV4_SRC_ADDR"].isin([host]) | df["IPV4_DST_ADDR"].isin([host])]
    logger.info("Created 5 datasets")

    # Stampa grandezza dataset
    for host in datasets.keys():
        print(f"Host:{host} Traffic rows: {int(traffic_per_host.loc[host])}")

    # Codice per mostrare figure traffic rows per single host
    traffic_per_host[:MAX_DATASETS].plot(kind="bar", xlabel="host", ylabel="traffic rows", title="traffic per host")
    plt.tight_layout()
    plt.close()
    
    return datasets

# Ritorna dict di df normalizzati con std scaler (x-u)/std
def scale_features(datasets : Dict[str, pd.DataFrame], numerical_features_names) -> Dict[str, pd.DataFrame]:
    logger.info("Normalizing..")

    scaler = LogMinMaxScaler()
    column_transformer = ColumnTransformer(
        transformers=[("normalize", scaler, numerical_features_names)], remainder="passthrough")

    datasetsScaled = {}

    for host, dataset in datasets.items():

        # Fit e transform sul dataset
        transformed_ndarray = column_transformer.fit_transform(dataset)
    
        # Nomi corretti nell'ordine giusto
        features = [name.split('__')[-1] for name in column_transformer.get_feature_names_out()]

        df = pd.DataFrame(data=transformed_ndarray, columns=features)

        # Fai downcast solamente di quelle numeriche, inizialmente sono float 64 dopo float 32
        for col in numerical_features_names:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], downcast="float")

        datasetsScaled[host] = df
        logger.info(f"{host} done")
    logger.info("Done")
    return datasetsScaled

# Ritorna dict di df divisi per valori di feature
def subdataset_per_feature_categories(datasets : Dict[str, pd.DataFrame], feature) -> Dict[str, pd.DataFrame]:
    subdatasets = {}
    logger.info("Dividing the datasets")
    for host, dataset in datasets.items():
        groups = dataset.groupby(feature)
        for category, group in groups:
            subdatasets[host + "_" + category] = group
    logger.info("Divided successfully")
    logger.info(f"Dataset: {len(subdatasets.values())}")
    return subdatasets

def preprocess(train_df : pd.DataFrame, validation_df : pd.DataFrame, test_df : pd.DataFrame):
    # Loading feature types from json
    logger.info("Loading feature types..")

    config_manager = ConfigManager()
    config_manager.load_config("config/dataset.json")
    numerical_columns = config_manager.get_value("dataset", "numeric_columns")
    categorical_columns = config_manager.get_value("dataset", "categorical_columns")
    target_column = config_manager.get_value("dataset", "target_column")

    logger.info("Loading complete")

    logger.info("Preprocessing...")

    # Pipeline for numerical features
    num_pipeline = Pipeline([
        ("Impute", CustomImputer()),
        ("Scaling", LogMinMaxScaler())
    ])

    # Pipeline per categorical features
    cat_pipeline = Pipeline([
        ("FrequencyEncoder", FrequencyEncoder(soglia=0.5)),
        ("1hot", OneHotEncoder(sparse_output = False, handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("num", num_pipeline, numerical_columns),
        ("cat", cat_pipeline, categorical_columns),
    ])

    # Training Set Fit
    logger.info("Fitting...")
    preprocessing.fit(train_df)
    logger.info("Fit")

    # Training Set Transform
    logger.info("Transforming Training Set...")
    train_preprocessed = preprocessing.transform(train_df)
    logger.info("Transformed")

    # Validation Set Transform
    logger.info("Transforming Validation Set...")
    validation_preprocessing = preprocessing.transform(validation_df)
    logger.info("Transformed")

    # Test Set Transform
    logger.info("Transforming Test Set...")
    test_preprocessed = preprocessing.transform(test_df)
    logger.info("Transformed")

    # Get the names from the fitted OneHotEncoder
    one_hot_column_names = preprocessing.named_transformers_["cat"].named_steps["1hot"].get_feature_names_out(categorical_columns)

    all_columns = numerical_columns + list(one_hot_column_names)

    train_df_preprocessed = pd.DataFrame(data=train_preprocessed, columns=all_columns, index=train_df.index)
    validation_df_preprocessed = pd.DataFrame(data=validation_preprocessing, columns=all_columns, index=validation_df.index)
    test_df_preprocessed = pd.DataFrame(data=test_preprocessed, columns=all_columns, index=test_df.index)

    train_df_preprocessed[target_column] = train_df[target_column]
    validation_df_preprocessed[target_column] = validation_df[target_column]
    test_df_preprocessed[target_column] = test_df[target_column]

    labels_train = train_df_preprocessed[target_column].value_counts()
    labels_validation = validation_df_preprocessed[target_column].value_counts()
    labels_test = test_df_preprocessed[target_column].value_counts()

    print("Training Set Labels:")
    print(labels_train)

    print("Validation Set Labels:")
    print(labels_validation)

    print("Test Set Labels:")
    print(labels_test)

    print(f"Features Train: {len(train_df_preprocessed.columns)}")
    print(f"Fatures Validation: {len(validation_df_preprocessed.columns)}")
    print(f"Features Test: {len(test_df_preprocessed.columns)}")

    assert all(train_df.index == train_df_preprocessed.index)
    assert all(validation_df.index == validation_df_preprocessed.index)
    assert all(test_df.index == test_df_preprocessed.index)


    return train_df_preprocessed, validation_df_preprocessed, test_df_preprocessed


def first_n_hosts_per_traffic(df, n):
    #Identifico il numero di righe per traffico inviato per host
    rows_per_src_host = df["IPV4_SRC_ADDR"].value_counts()

    #Identifico il numero di righe per traffico ricevuto per host
    rows_per_dst_host = df["IPV4_DST_ADDR"].value_counts()

    #Ottengo il traffico per ogni host
    traffic_per_host = rows_per_dst_host.add(rows_per_src_host, fill_value=0).sort_values(ascending=False)

    return traffic_per_host[:n]

def remove_one_sample(df, feature):
    values = df[feature].value_counts()
    only_one_sample = values[values == 1]
    print(only_one_sample)

    mask = df[feature].isin(only_one_sample.keys())
    logger.info(f"Removing records of {feature}  with only 1 sample per category..")
    df = df[~mask]
    removed = df[mask]
    logger.info(f"Rows: {len(df)}, Removed: {len(removed)}")

    return (df, removed)
