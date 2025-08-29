import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from sklearn.model_selection import train_test_split
from utilities.config_manager import ConfigManager
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from data.scalers.LogMinMaxScaler import LogMinMaxScaler
from data.scalers.custom_imputer import CustomImputer
from data.scalers.frequency_encoder import FrequencyEncoder
import os
 
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
        print(len(dataset.columns))
        # Fit e transform sul dataset
        transformed_ndarray = column_transformer.fit_transform(dataset)
    
        # Nomi corretti nell'ordine giusto
        features = [name.split('__')[-1] for name in column_transformer.get_feature_names_out()]
        df = pd.DataFrame(data=transformed_ndarray, columns=features)
        del transformed_ndarray
        # Fai downcast solamente di quelle numeriche, inizialmente sono float 64 dopo float 32
        for col in numerical_features_names:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], downcast="float")

        df.to_csv(f"resources/datasets/temporary/{host}.csv" ,index=False)
        print(len(df.columns))
        del df
        datasetsScaled[host] = f"resources/datasets/temporary/{host}.csv"
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
        ("FrequencyEncoder", FrequencyEncoder(soglia=0.3)),
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

    mask = df[feature].isin(only_one_sample.keys())
    logger.info(f"Removing records of {feature}  with only 1 sample per category..")
    removed = df[mask]
    df = df[~mask]
    logger.info(f"Rows: {len(df)}, Removed: {len(removed)}")

    return (df, removed)

def create_attack_dataset(df : pd.DataFrame, attack_category : str) -> pd.DataFrame:
    logger.info(f"Creating a dataset containing Benign and {attack_category}..")
    mask = (df["Attack"] == "Benign") | (df["Attack"] == attack_category)
    df = df[mask].reset_index(drop=True)
    logger.info(f"Created a dataset with {len(df)} rows\n")

    return df

def preprocess_categorical(df : pd.DataFrame, categorical_features : List[str]) -> pd.DataFrame:
    # Pipeline for categorical features
    cat_pipeline = Pipeline([
        ("FrequencyEncoder", FrequencyEncoder(soglia=0.3)),
        ("1hot", OneHotEncoder(sparse_output = False, handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("cat", cat_pipeline, categorical_features)
    ], remainder="passthrough")

    logger.info("Preprocessing Categorical Features..")
    transformed_array = preprocessing.fit_transform(df)
    # Get the names from the fitted OneHotEncoder
    one_hot_column_names = preprocessing.named_transformers_["cat"].named_steps["1hot"].get_feature_names_out(categorical_features)

    remainder_columns = [col for col in df.columns if col not in categorical_features]
    all_columns = list(one_hot_column_names) + remainder_columns

    df = pd.DataFrame(data=transformed_array, columns=all_columns, index=df.index)
    logger.info("Done\n")

    return df

def divide_dataset_per_host(df : pd.DataFrame, hosts : List[str], datasets_dir, check_for_attack=None, drop_ip=False) -> Dict[str, str]:
    dfs = {}
    logger.info("Creating dataset per host")
    for host in hosts:
        logger.info(f"Host: {host}")
        temp_df = df[(df["IPV4_SRC_ADDR"] == host) | (df["IPV4_DST_ADDR"] == host)].copy()
        logger.info(f"Created a dataset of {len(temp_df)} rows\n")

        if drop_ip:
            temp_df.drop(["IPV4_SRC_ADDR", "IPV4_DST_ADDR"], axis=1, inplace=True)

        if check_for_attack == None:
            dataset_path = os.path.join(datasets_dir, f"{host}.csv")
            temp_df.to_csv(dataset_path, index=False)
            dfs[host] = dataset_path
        elif (temp_df["Attack"].value_counts()).get(check_for_attack) != None:
            dataset_path = os.path.join(datasets_dir, f"{host}_{check_for_attack}.csv")
            temp_df.to_csv(dataset_path, index=False)
            dfs[host] = dataset_path
        else:
            logger.info(f"Host {host} excluded because it does not contain any {check_for_attack} rows")

        print(temp_df["Attack"].value_counts())

    return dfs
            
def balance_attack_data(dfs : Dict[str, str], attack : str, seed, output_dir, drop_attack=True):
    dfs_sampled = {}
    for host, df in dfs.items():
        logger.info(f"Host: {host}, balancing attack data..")
        df_loaded = pd.read_csv(df)
        
        benign_mask = df_loaded["Attack"] == "Benign"
        attack_mask = df_loaded["Attack"] == attack

        attack_df = df_loaded[attack_mask]
        benign_df = df_loaded[benign_mask].sample(n=len(attack_df), random_state=seed)

        new_df = pd.concat([benign_df, attack_df], ignore_index=True)
        new_df = new_df.sample(frac=1, random_state=seed).reset_index(drop=True)

        # values = new_df["Attack"].value_counts()

        # logger.info(f"Created a dataset containing {len(new_df)} rows")
        # logger.info(f"Benign: {values['Benign']}")
        # logger.info(f"{attack}: {values[attack]}")

        if drop_attack:
            new_df.drop(["Attack"], axis=1, inplace=True)

        dataset_path = os.path.join(output_dir, f"{host}_{attack}_balanced.csv")

        new_df.to_csv(dataset_path, index=False)
        dfs_sampled[host] = dataset_path
        logger.info(f"Saved to {dataset_path}\n")
    return dfs_sampled

def prepare_training_validation_test(df : str, output_path, seed, train_ratio, drop=True, stratify=None):
    logger.info("Preparing Training, Validation and Test set..")
    df_loaded = pd.read_csv(df)

    data_dict = {}

    if stratify != None:
        train_df, temp_df = train_test_split(df_loaded, test_size=(1-train_ratio), stratify=df_loaded[stratify], random_state=seed)
        validation_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df[stratify], random_state=seed)
    else:
        train_df, temp_df = train_test_split(df_loaded, test_size=(1-train_ratio), random_state=seed)
        validation_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)

    if stratify != None and drop:
        train_df.drop(["Attack"], axis=1, inplace=True)
        validation_df.drop(["Attack"], axis=1, inplace=True)
        test_df.drop(["Attack"], axis=1, inplace=True)

    train_df.to_csv(output_path + f"_train.csv", index=False)
    data_dict["train"] = output_path + f"_train.csv"

    validation_df.to_csv(output_path + f"_validation.csv", index=False)
    data_dict["validation"] = output_path + f"_validation.csv"

    test_df.to_csv(output_path + f"_test.csv", index=False)
    data_dict["test"] = output_path + f"_test.csv"

    print(len(train_df.columns))
    print(len(validation_df.columns))
    print(len(test_df.columns))
    print(train_df.columns)

    logger.info("Done\n")

    return data_dict

def preprocess_numerical(train_df, validation_df, test_df, numerical_features, output_path):
    logger.info("Preprocessing Numerical Features...")

    train_df = pd.read_csv(train_df)
    validation_df = pd.read_csv(validation_df)
    test_df = pd.read_csv(test_df)

    # Pipeline for numerical features
    num_pipeline = Pipeline([
        ("Impute", CustomImputer()),
        ("Scaling", LogMinMaxScaler())
    ])

    preprocessing = ColumnTransformer([
        ("num", num_pipeline, numerical_features),
    ], remainder="passthrough")

    # Training Set Fit
    preprocessing.fit(train_df)

    # Training Set Transform
    train_preprocessed = preprocessing.transform(train_df)

    # Validation Set Transform
    validation_preprocessed = preprocessing.transform(validation_df)

    # Test Set Transform
    test_preprocessed = preprocessing.transform(test_df)

    remainder_passthrough = [col for col in train_df.columns if col not in numerical_features]

    all_columns = numerical_features + remainder_passthrough

    train_df_preprocessed = pd.DataFrame(data=train_preprocessed, columns=all_columns, index=train_df.index)
    validation_df_preprocessed = pd.DataFrame(data=validation_preprocessed, columns=all_columns, index=validation_df.index)
    test_df_preprocessed = pd.DataFrame(data=test_preprocessed, columns=all_columns, index=test_df.index)

    assert all(train_df.index == train_df_preprocessed.index)
    assert all(validation_df.index == validation_df_preprocessed.index)
    assert all(test_df.index == test_df_preprocessed.index)

    data_dict = {}
    train_df_preprocessed.to_csv(output_path + f"_train.csv", index=False)
    data_dict["train"] = output_path + f"_train.csv"

    validation_df_preprocessed.to_csv(output_path + f"_validation.csv", index=False)
    data_dict["validation"] = output_path + f"_validation.csv"

    test_df_preprocessed.to_csv(output_path + f"_test.csv", index=False)
    data_dict["test"] = output_path + f"_test.csv"

    logger.info("Done\n")

    return data_dict


def filter_hosts_by_attack(hosts, df, attack):
    logger.info(f"Checking if every host traffic contains {attack} labeled rows")
    filtered = []
    for host in hosts:
        temp_df = df[(df["IPV4_SRC_ADDR"] == host) | (df["IPV4_DST_ADDR"] == host)]

        if temp_df["Attack"].value_counts().get(attack) != None:
            filtered.append(host)
        else:
            logger.info(f"Host {host} excluded because it does not contain any {attack} labeled rows")
    logger.info("Hosts:")
    logger.info(f"{filtered}\n")

    return filtered

def concat_all(df_training_test, category, output_path):
        all = pd.concat([pd.read_csv(sets[category]) for _, sets in df_training_test.items()], ignore_index=True)
        output_path = os.path.join(output_path, f"combined_{category}.csv")
        print(len(all.columns))
        print(all.columns)
        logger.info(f"{category}: {len(all)}")
        all.to_csv(output_path, index=False)
        return output_path