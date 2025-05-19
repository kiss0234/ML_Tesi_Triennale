import sys
import logging

import pandas as pd

from utilities.memoryusage import print_memory_usage
from utilities.logging_config import setup_logging
from utilities.argument_parser import ArgumentParser
from utilities.config_manager import ConfigManager
from data.study.study_labels import labels_study
from data.prepdata.prepare_dataset import divide_dataset, remove_attacks_under_threshold
from data.study.study_numerical import numerical_features_study
from data.study.study_categorical import categorical_features_study
from data.study.study_cat_num import categorical_numerical_features_study

setup_logging()
logger = logging.getLogger(__name__)
MAX_SPLIT = 1
THRESHOLD = 3000

def prepare_data(input_path: str, output_path: str) -> None:
    logger.info("Preparing data...")

    df = pd.read_csv(input_path, nrows=10000)

    logger.info(f"Loaded {df.shape[0]} rows")

    #Caricamento tipo delle feature dal json in config
    logger.info("Loading feature types..")

    config_manager = ConfigManager()
    config_manager.load_config("config/dataset.json")
    numerical_columns = config_manager.get_value("dataset", "numeric_columns")
    categorical_columns = config_manager.get_value("dataset", "categorical_columns")

    logger.info("Loading complete")

def help_study() -> None:
    print("------------------------------------")
    print("Enter 0 to divide dataset per host")
    print("Enter 1 to show target label's graph")
    print("Enter 2 to go to the numerical features section")
    print("Enter 3 to go to the categorical features section")
    print("Enter 4 to go to the all features section")
    print(f"Enter 5 to drop attacks with under {THRESHOLD} samples")
    print("Enter q to end the execution")
    print("------------------------------------")

def study_data(input_path: str):
    split = 0

    logger.info("Loading data...")
    df = pd.read_csv(input_path) 
    logger.info(f"Loaded {df.shape[0]} rows")

    logger.info("Loading feature types..")
    config_manager = ConfigManager()
    config_manager.load_config("config/dataset.json")
    numerical_features = config_manager.get_value("dataset", "numeric_columns")
    numerical_features.remove("FLOW_START_MILLISECONDS")
    numerical_features.remove("FLOW_END_MILLISECONDS")
    categorical_features = config_manager.get_value("dataset", "categorical_columns")
    logger.info("Loading complete")

    print_memory_usage()
    datasets = {}
    help_study()
    userInput = input("->")
    while(True):
        match userInput:
            case "0":       
                if(split != MAX_SPLIT):
                    datasets = divide_dataset(df)
                    split = split + 1
                    del df
                else:
                    print("ERROR: You can't split more than one time")
            case "1":
                if(split == 1):
                    labels_study(datasets)
                else:
                    print("ERROR: Split the dataset first")
            case "2":
                if(split == 1):
                    numerical_features_study(datasets, numerical_features)
                else:
                    print("ERROR: Split the dataset first")
            case "3":
                if(split == 1):
                    categorical_features_study(datasets, categorical_features)
                else:
                    print("ERROR: Split the dataset first")
            case "4":
                if(split == 1):
                    categorical_numerical_features_study(datasets, categorical_features, numerical_features)
                else:
                    print("ERROR: Split the dataset first")
            case "5":
                if(split==1):
                    remove_attacks_under_threshold(datasets, THRESHOLD)
                else:
                    print("ERROR: Split the dataset first")
            case "q":
                break
            case _:
                print("WRONG INPUT")
        print_memory_usage()
        help_study()
        userInput = input("->")


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
        arguments=["--input"],
        helps=[
            "The input path for the data."
        ],
        defaults=["resources/datasets/dataset.csv"],
    )

    args = parser.parse_arguments(sys.argv[1:])

    if args.subcommand == "prepare":
        prepare_data(args.input, args.output)

    elif args.subcommand == "study":
        study_data(args.input) 