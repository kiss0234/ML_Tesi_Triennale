import logging
import pandas as pd 
import numpy as np
from data.study.comparedist.cmpdist import  compare_distributions_all_features
from typing import List, Dict
from data.prepdata.prepare_dataset import scale_features, subdataset_per_feature_categories
from data.prepdata.binning import binning
from collections import defaultdict

divided = 0
normalized = 0
logger = logging.getLogger(__name__)


def help_study() -> None:
    print("\nEnter 0 to go back to the given datasets")
    print("Enter 1 to divide the datasets per attack")
    print("Enter 2 to normalize the numerical features")
    print("Enter 3 to bucketize the normalized datasets")
    print("Enter 4 to compare distributions")
    print("Enter q to quit")

# Funzione per lo studio delle feature numeriche
def categorical_numerical_features_study(datasets: Dict[str, pd.DataFrame], categorical_features, numerical_features: List[str]) -> None:
    givenDatasets = datasets
    compare_tables = {}
    print("\nALL FEATURES SECTION\nHere you can choose between different options to modify the given dataset", end="\n")
    help_study()
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
                if(normalized):
                    buckets = int(input("Inserisci numero di buckets: "))
                    datasets = binning(datasets, buckets, numerical_features)
                else:
                    print("ERROR: Normalize the dataset first")

            case "4":  
                if(divided and normalized):
                    grouped_dicts = defaultdict(dict)

                    # Dividi il dict in sotto dict contenenti datasets relativi allo stesso attacco
                    for name, dataset in datasets.items():
                        group_key = name.split('_')[-1] # La notazione della chiave è "host_attacco", la group_key è attacco
                        grouped_dicts[group_key][name] = dataset
                    
                    for attack, group in grouped_dicts.items():
                        print(f"Group: {attack}")
                        compare_distributions_all_features(group, (categorical_features+numerical_features), attack)
                        

                elif(normalized):
                    compare_distributions_all_features(datasets, (categorical_features+numerical_features))
                else:
                    print("ERROR: Normalize the datasets first")

            case "q":
                divided = 0
                normalized = 0
                break

            case _:
                print("WRONG INPUT")

        help_study()
        userInput = input("->")


