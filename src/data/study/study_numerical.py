import logging
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import os
from typing import List, Dict
from data.prepdata.prepare_dataset import scale_features, subdataset_per_feature_categories

divided = 0
normalized = 0
logger = logging.getLogger(__name__)

def plot_feature_distribution(datasets : Dict[str, pd.DataFrame], numerical_feature) -> None:
    for host, dataset in datasets.items():
        plt.figure(1)
        # plt.gca().set_axis_off()
        fpath = f"graphs/hosts/{host}/numerical"
        #Controllo se il dataset è stato normalizzato
        if(normalized):
            fpath = os.path.join(fpath, "normalized/")
        os.makedirs(name=fpath, exist_ok=True)

        for feature in numerical_feature:
            dataset[feature].hist()
            plt.gca().set_title(label=host + " " + feature, fontsize=12, pad=20)
            plt.gca().tick_params(axis="x", labelsize=7)
            filename = os.path.join(fpath, f"{feature}.jpg")

            # Se il grafico non esiste già nella cartella graph allora aggiungilo 
            if(not os.path.exists(filename)):
                plt.savefig(filename)
            plt.clf()

def features_infos(datasets, numerical_feature) -> None:
    # Creo un df temporaneo con tutte le info delle feature numeriche per host
        logger.info("Processing features' infos") 
        for host, dataset in datasets.items():
            feature_infos = {}
            print("------------------------------------------------------------------")
            print(f"Host: {host}, Rows:{dataset.shape[0]}")
            # Creo un dizionario contenente le varie info per ogni feature
            for feature in numerical_feature:
                feature_infos[feature] = {"min" : dataset[feature].min(),
                                          "max" : dataset[feature].max(),
                                          "mean" : dataset[feature].mean(),
                                          "std" : dataset[feature].std(),
                                          "most_frequent" : dataset[feature].mode()[0],
                                          "median" : dataset[feature].median()
                                          }
            df_features_infos = pd.DataFrame(feature_infos).T
            print(df_features_infos, end="\n\n\n")
        logger.info("Processing done")


def help_numerical_features_study() -> None:
    print("\nEnter 0 to divide the datasets per attack")
    print("Enter 1 to normalize the given features")
    print("Enter 2 to go back to the given datasets")
    print("Enter 3 to show feature infos for all datasets")
    print("Enter 4 to show all the distributions plots")
    print("Enter q to quit")

# Funzione per lo studio delle feature numeriche
def numerical_features_study(datasets: Dict[str, pd.DataFrame], numerical_feature: List[str]) -> None:
    givenDatasets = datasets

    print("\nNUMERICAL FEATURES SECTION\nHere you can choose between different options to modify the given dataset", end="\n")
    help_numerical_features_study()
    userInput = input("->")

    global divided, normalized
    while(True):
        match userInput:
            case "0":
                if(divided != 1):
                    datasets = subdataset_per_feature_categories(datasets, feature="Attack")
                    divided = 1
                else:
                    print("ERROR: You can't divide the dataset more than once")
            case "1":
                if(normalized != 1):
                    datasets = scale_features(datasets, numerical_feature)
                    normalized = 1
                else:
                    print("ERROR: You can't normalize the dataset again")
            case "2":
                divided = 0
                normalized = 0
                datasets = givenDatasets
                logger.info("Done")
            case "3":
                features_infos(datasets, numerical_feature)
            case "4":
                plot_feature_distribution(datasets, numerical_feature)
            case "q":
                break
            case _:
                print("WRONG INPUT")
        help_numerical_features_study()
        userInput = input("->")
