import logging
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import List, Dict
from data.scalers.LogMinMaxScaler import LogMinMaxScaler
from sklearn.compose import ColumnTransformer
 
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
    # plt.show()
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