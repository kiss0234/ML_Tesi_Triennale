import logging
import pandas as pd
import matplotlib.pyplot as plt

import os
from data.prepdata.prepare_dataset import subdataset_per_feature_categories 
from typing import Dict, List

logger = logging.getLogger(__name__)
divided = 0

def feature_categories(datasets : Dict[str, pd.DataFrame], categorical_features) -> Dict[str, pd.DataFrame]:
    print("Only the first 30 feature's categories will be shown")

    for host, dataset in datasets.items():
        current = f"graphs/hosts/{host}/categorical" # Path per grafici di feature categoriche
        os.makedirs(name=current, exist_ok=True) # Se non esiste crea le cartelle

        for feature in categorical_features:
            filename = os.path.join(current, f"{feature}.jpg") # Il nome del file è composto dal nome della feature
            categoriesCount = dataset[feature].value_counts()
            categoriesCount[:30].plot(kind="bar", xlabel="categories") # Considero solo le prime 30 categorie della feature
            plt.gca().set_title(label=feature, fontsize=12, pad=20)
            if(not os.path.exists(filename)):
                plt.savefig(filename)
            plt.clf()

def help_categorical_features_study() -> None:
    print("Enter 0 to divide the datasets per attack")
    print("Enter 1 to plot a bar graph for every feature's category")
    print("Enter 2 to go back to the given datasets")
    print("Enter q to leave this section")

def categorical_features_study(datasets : Dict[str, pd.DataFrame], categorical_features : List[str]) -> None:
    givenDatasets = datasets
    print(categorical_features)
    print("\nCATEGORICAL FEATURES SECTION\nHere you can choose between different options to modify the given dataset", end="\n")
    help_categorical_features_study()
    userInput = input("->")

    global divided
    while(True):
        match userInput:
            case "0":
                if(divided != 1):
                    datasets = subdataset_per_feature_categories(datasets, feature="Attack")
                    divided = 1
                else:
                    print("ERROR: You can't divide the dataset more than once")
            case "1":
                feature_categories(datasets, categorical_features)
            case "2":
                datasets = givenDatasets
            case "q":
                break
            case _:
                print("WRONG INPUT")
        help_categorical_features_study()
        userInput = input("->")