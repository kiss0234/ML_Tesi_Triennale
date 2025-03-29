import sys
import logging

import pandas as pd
import matplotlib.pyplot as plt

from utilities.logging_config import setup_logging
from utilities.argument_parser import ArgumentParser
from utilities.config_manager import ConfigManager

setup_logging()
logger = logging.getLogger(__name__)


def prepare_data(input_path: str, output_path: str):
    logger.info("Preparing data...")

    df = pd.read_csv(input_path, nrows=10000)

    logger.info("Data loaded")

    #Caricamento tipo delle feature dal json in config
    logger.info("Loading feature types..")

    config_manager = ConfigManager()
    config_manager.load_config("config/dataset.json")
    numerical_columns = config_manager.get_value("dataset", "numeric_columns")
    categorical_columns = config_manager.get_value("dataset", "categorical_columns")

    logger.info("Loading completed")
    
    return (df, numerical_columns, categorical_columns)

def help_study():
    print("------------------------------------")
    print("Enter 0 to divide dataset per host")
    print("Enter 1 to show target label's graph")
    print("Enter 2 to show numerical features' graph")
    print("Enter 3 to show categorical features' graph")
    print("Enter q to leave this section")
    print("------------------------------------")

def divide_dataset(df):

    #Identifico il numero di righe per traffico inviato per host
    rows_per_src_host = df["IPV4_SRC_ADDR"].value_counts()

    #Identifico il numero di righe per traffico ricevuto per host
    rows_per_dst_host = df["IPV4_DST_ADDR"].value_counts()

    #Ottengo il traffico per ogni host
    traffic_per_host = rows_per_dst_host.add(rows_per_src_host, fill_value=0).sort_values(ascending=False)

    #Divido il dataset in sotto dataset per indirizzo ip relativo a traffico in entrata ed in uscita
    logger.info("Dividing datasets in smaller datasets per ip..")

    datasets = []
    for index in traffic_per_host.index:
        datasets.append(df[df["IPV4_SRC_ADDR"].isin([index]) | df["IPV4_DST_ADDR"].isin([index])].copy())

    logger.info(f"Created {len(traffic_per_host.index)} datasets")

    # Codice per mostrare figure traffic rows per single host
    print("Traffic rows per single host figure: ")
    traffic_per_host.plot(kind="bar", xlabel="host", ylabel="traffic rows", title="traffic per host")
    plt.show()
    
    return datasets


# Funzione per vedere eterogeneità delle label fra i dataset
def labels_study(df):

    # Raggruppo dataset per host e conto il numero di tipologie di attacco
    logger.info("Grouping dataset per host..")

    df_melted = df.melt(id_vars=df.columns.drop(["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]),value_vars=["IPV4_SRC_ADDR", "IPV4_DST_ADDR"])
    df_melted_grouped_by_ip = df_melted.groupby("value")

    type_of_attacks_per_host = df_melted_grouped_by_ip["Attack"].value_counts().unstack(fill_value=0).sort_values(by="Benign", ascending=False)
    logger.info("Dataset grouped correctly")

    # Creazione grafici per facilitare comparazione degli attacchi tra host
    i = 1
    for attack_type in type_of_attacks_per_host.columns:
        plt.figure(10)
        plt.subplot(3, 3, i)
        i = i + 1
        type_of_attacks_per_host[attack_type].plot(kind="bar", xlabel="host", ylabel="number of attack rows", title=str(attack_type), figsize=(15, 8))

    plt.show()    

# Funzione per lo studio delle feature numeriche
def numerical_features_study(df):

    importantNumericalFeatures = ["IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS", "DURATION_IN", 
                                 "DURATION_OUT", "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
                                 "NUM_PKTS_1024_TO_1514_BYTES", "SRC_TO_DST_IAT_AVG", "DST_TO_SRC_IAT_AVG"]
    
    

def study_data(input_path: str, output_path: str):

    df, numerical_columns, categorical_columns = prepare_data(input_path=input_path, output_path=output_path)

    datasets = []
    help_study()
    selection = input("->")
    while(True):
        match selection:
            case "0":       
                datasets = divide_dataset(df)
                print(len(datasets))
            case "1":
                labels_study(df)
            case "2":
                numerical_features_study(df)
            case "3":
                ...
            case "q":
                break
            case _:
                logger.warning("WRONG INPUT")
        help_study()
        selection = input("->")


if __name__ == "__main__":
    parser = ArgumentParser("parser")

    parser.register_subcommand(
        subcommand="prepare",
        arguments=["--input", "--output"],
        helps=[
            "The input path for the data.",
            "The output path for the prepared data.",
        ],
        defaults=["resources/datasets/dataset.csv", None],
    )

    parser.register_subcommand(
        subcommand="study",
        arguments=["--input", "--output"],
        helps=[
            "The input path for the data.",
            "The output path for the prepared data.",
        ],
        defaults=["resources/datasets/dataset.csv", None],
    )

    args = parser.parse_arguments(sys.argv[1:])

    if args.subcommand == "prepare":
        prepare_data(args.input, args.output)

    elif args.subcommand == "study":
        study_data(args.input, args.output)
