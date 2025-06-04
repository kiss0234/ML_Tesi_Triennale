import sys
import logging
import torch
import os
from torch.utils.data import random_split
from torch import optim
from data.CSVDataset import CSVDataset
from model.MyNet import MyNet
import torch.nn as nn
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from data.scalers.LogMinMaxScaler import LogMinMaxScaler
from data.scalers.custom_binner import CustomBinner
from data.scalers.custom_imputer import CustomImputer
from data.scalers.frequency_encoder import FrequencyEncoder
from sklearn.model_selection import train_test_split
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

    df = pd.read_csv(input_path, nrows=100000)

    logger.info(f"Loaded {df.shape[0]} rows")

    #Caricamento tipo delle feature dal json in config
    logger.info("Loading feature types..")

    config_manager = ConfigManager()
    config_manager.load_config("config/dataset.json")
    numerical_columns = config_manager.get_value("dataset", "numeric_columns")
    categorical_columns = config_manager.get_value("dataset", "categorical_columns")
    target_column = config_manager.get_value("dataset", "target_column")

    logger.info("Loading complete")

    logger.info("Preprocessing...")
    # Pipeline per numeriche
    num_pipeline = Pipeline([
        ("Impute", CustomImputer()),
        ("Binning", CustomBinner()),
        ("Scaling", LogMinMaxScaler())
    ])

    # Pipeline per categoriche
    cat_pipeline = Pipeline([
        ("FrequencyEncoder", FrequencyEncoder(soglia=0.5)),
        ("1hot", OneHotEncoder(sparse_output = False)),
    ])

    # Numeriche passano per pipeline numerica, categoriche per pipeline categorica
    preprocessing = ColumnTransformer([
        ("num", num_pipeline, numerical_columns),
        ("cat", cat_pipeline, categorical_columns),
    ])

    array_preprocessed = preprocessing.fit_transform(df)

    one_hot_column_names = preprocessing.named_transformers_["cat"].named_steps["1hot"].get_feature_names_out(categorical_columns)

    all_columns = numerical_columns + list(one_hot_column_names)

    df_preprocessed = pd.DataFrame(data=array_preprocessed, columns=all_columns)

    df_preprocessed[target_column] = df[target_column]

    train_ratio = 0.8

    train_df, test_df = train_test_split(df_preprocessed, test_size=(1-train_ratio), random_state=None)

    if output_path != None:
        train_df.to_csv(output_path + "_train.csv", index=False)
        test_df.to_csv(output_path + "_test.csv", index=False)

    logger.info("Preprocessed and saved correctly")


def train_model(input_path: str, out_path_model: str) -> None:
    config_manager = ConfigManager()
    config_manager.load_config("config/dataset.json")
    target_column = config_manager.get_value("dataset", "target_column")
    os.makedirs(out_path_model, exist_ok=True)

    full_path = os.path.join(out_path_model, "best_model.pth")

    # Creo oggetto Dataset
    dataset = CSVDataset(input_path, target_column)

    # Definisco validation dataset e training dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=64,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=64,
        shuffle=False
    )

    device = ('cuda' if torch.cuda.is_available() else 'cpu') 
    model = MyNet(input, 1).to(device=device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    patience = 10
    counter = 0
    best_loss = float('inf')
    N_EPOCHS = 15
    for epoch in range(N_EPOCHS):
        train_loss = 0.0
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            labels = targets.to(device)

            # Evito gradient accumulation
            optimizer.zero_grad()

            # Predict
            outputs = model(inputs)

            # Calcolo loss con bce
            loss = criterion(outputs, labels)

            # Calcolo gradienti
            loss.backward() 

            # Gradient Descent
            optimizer.step()

            train_loss += loss.item()
        
        # Validation loss
        val_loss = 0.0
        all_preds = []
        all_labels = []

        # Modello in validation mode
        model.eval()
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            labels = targets.to(device)
            # Predict
            outputs = model(inputs)

            # Calcolo loss con bce
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).int()

            # Salvo tutte le labels e gli outputs per usare metriche
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu().int())
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), full_path)
            counter = 0
        else:
            counter += 1
            print("Patience +1")

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # Metriche
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        confusionM = confusion_matrix(all_labels, all_preds)
        
        print("Epoch: {} Patience: {} Validation Loss: {} Training Loss: {}".format(epoch, counter, val_loss/len(val_loader), train_loss/len(train_loader)))
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:\n", confusionM)

        if counter >= patience:
            print(f"Validation loss didn't improve for more than {patience} epochs")
            break




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
        defaults=["resources/datasets/dataset.csv", "resources/datasets"],
    )

    parser.register_subcommand(
        subcommand="study",
        arguments=["--input"],
        helps=[
            "The input path for the data."
        ],
        defaults=["resources/datasets/dataset.csv"],
    )

    parser.register_subcommand(
        subcommand="train",
        arguments=["--input", "--output"],
        helps=[
            "The input path for the data.", "The output path for the model"
        ],
        defaults=[],
    )


    args = parser.parse_arguments(sys.argv[1:])

    if args.subcommand == "prepare":
        prepare_data(args.input, args.output)

    elif args.subcommand == "study":
        study_data(args.input) 

    elif args.subcommand == "train":
        train_model(args.input, args.output)