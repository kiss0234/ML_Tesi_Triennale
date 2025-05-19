import logging
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import pickle
import os
from data.study.comparedist.cmpdist import  compare_distributions2, plot_feature_prob_dist, compare_plots
from typing import List, Dict
from data.prepdata.prepare_dataset import scale_features, subdataset_per_feature_categories
from data.prepdata.binning import binning
from collections import defaultdict

divided = 0
normalized = 0
logger = logging.getLogger(__name__)


def save_dict(df_dict : Dict[str, pd.DataFrame]):
    if(normalized and divided):
        with open("resources/feature_infosDN.pickle", "wb") as f:
            pickle.dump(df_dict, f)
    elif(divided):
        with open("resources/feature_infosD.pickle", "wb") as f:
            pickle.dump(df_dict, f)
    elif(normalized):
        with open("resources/feature_infosN.pickle", "wb") as f:
            pickle.dump(df_dict, f)
    else:
        with open("resources/feature_infos.pickle", "wb") as f:
            pickle.dump(df_dict, f)

def plot_feature_distribution(datasets : Dict[str, pd.DataFrame], numerical_feature : List[str]) -> None:
    for host, dataset in datasets.items():
        plt.figure(1)
        # plt.gca().set_axis_off()
        fpath = f"graphs/hostsProva/{host}/numerical"
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
        df_dict = {}
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
            if(normalized):
                df_dict[f"{host} normalized"] = pickle.dumps(df_features_infos)
            else:
                df_dict[host] = pickle.dumps(df_features_infos)
            print(df_features_infos, end="\n\n\n")
        
        save_dict(df_dict)
        logger.info("Processing done")

def print_feature_infos(filePath):
    with open(filePath, "rb") as f:
        try:
            df_dict = pickle.load(f)
        except Exception:
            logger.error("Errore nel caricamento dei dati")
            return
        # host - features info
        for host, df in df_dict.items():
            print("--------------------------------------")
            print(f"Host: {host}")
            data_loaded = pickle.loads(df)
            print(data_loaded)
        print("\n\n")

def help_show_computed_infos():
    print("Enter 1 to show not processed")
    print("Enter 2 to show only normalized")
    print("Enter 3 to show divided")
    print("Enter 4 to show divided and normalized")
    print("Enter q to leave")

# prendo dal binary file salvato
def show_computed_infos():
    help_show_computed_infos()
    userInput = input("->")
    while(True):
        match userInput:
            case "1":
                print_feature_infos("resources/feature_infos.pickle")
            case "2":
                print_feature_infos("resources/feature_infosN.pickle")
            case "3":
                print_feature_infos("resources/feature_infosD.pickle")
            case "4":
                print_feature_infos("resources/feature_infosDN.pickle")
            case "q":
                break

        help_show_computed_infos()
        userInput = input("->")

def help_numerical_features_study() -> None:
    print("Enter 0 to go back to the given datasets")
    print("Enter 1 to divide the datasets per attack")
    print("Enter 2 to normalize the given features")
    print("Enter 3 to show feature infos for all datasets")
    print("Enter 4 to plot the frequency distributions")
    print("Enter 5 to show the already computed infos")
    print("Enter 6 to compare distributions")
    print("Enter 7 to plot feature probability distributions")
    print("Enter 8 to bucketize the normalized datasets")
    print("Enter q to quit")

# Funzione per lo studio delle feature numeriche
def numerical_features_study(datasets: Dict[str, pd.DataFrame], numerical_features: List[str]) -> None:
    givenDatasets = datasets

    print("\nNUMERICAL FEATURES SECTION\nHere you can choose between different options to modify the given dataset", end="\n")
    help_numerical_features_study()
    userInput = input("->")
    global divided, normalized
    while(True):
        match userInput:
            case "0":
                divided = 0
                normalized = 0
                datasets = givenDatasets
                logger.info("Done")
            case "1":
                if(divided):
                    print("ERROR: You can't divide the dataset more than once")
                elif(normalized):
                    print("ERROR: You can't divide the dataset after normalizing")
                else:
                    datasets = subdataset_per_feature_categories(datasets, feature="Attack")
                    divided = 1
            case "2":
                if(not normalized):
                    datasets = scale_features(datasets, numerical_features)
                    normalized = 1
                else:
                    print("ERROR: You can't normalize the dataset again")
            case "3":
                features_infos(datasets, numerical_features)
            case "4":
                plot_feature_distribution(datasets, numerical_features)
            case "5":
                show_computed_infos()

            case "6":  
                if(divided and normalized):
                    grouped_dicts = defaultdict(dict)

                    # Dividi il dict in sotto dict contenenti datasets relativi allo stesso attacco
                    for name, df in datasets.items():
                        group_key = name.split('_')[-1] # La notazione della chiave è "host_attacco"
                        grouped_dicts[group_key][name] = df
                    
                    for attack, group in grouped_dicts.items():
                        print(f"Group: {attack}")
                        compare_distributions2(group, numerical_features, attack)
                        # plot_feature_over_time(group, numerical_feature)
                elif(normalized):
                    compare_distributions2(datasets, numerical_features)
                    # plot_feature_over_time(datasets, numerical_feature)
                else:
                    print("ERROR: Normalize the datasets first")

            case "7":
                if(divided and normalized):
                    grouped_dicts = defaultdict(dict)

                    # Dividi il dict in sotto dict contenenti datasets relativi allo stesso attacco
                    for name, df in datasets.items():
                        group_key = name.split('_')[-1] # La notazione della chiave è "host_attacco"
                        grouped_dicts[group_key][name] = df
                    
                    for attack, group in grouped_dicts.items():
                        print(f"Group: {attack}")
                        plot_feature_prob_dist(group, numerical_features, attack)
                        compare_plots(group, numerical_features, attack)
                elif(normalized):
                    plot_feature_prob_dist(datasets, numerical_features)
                    compare_plots(datasets, numerical_features)
                else:
                    print("ERROR: Normalize the datasets first")
            case "8":
                if(normalized):
                    buckets = int(input("Inserisci numero di buckets: "))
                    datasets = binning(datasets, buckets, numerical_features)
                else:
                    print("ERROR: Normalize the dataset first")
            case "q":
                divided = 0
                normalized = 0
                break
            case _:
                print("WRONG INPUT")

        help_numerical_features_study()
        userInput = input("->")
