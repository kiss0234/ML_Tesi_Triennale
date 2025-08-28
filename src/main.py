import sys
import logging
import torch
import os
import copy
from typing import Dict, List
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from data.CSVDataset import CSVDataset
from model.MyNet import MyNet
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from utilities.memoryusage import print_memory_usage
from utilities.logging_config import setup_logging
from utilities.argument_parser import ArgumentParser
from utilities.config_manager import ConfigManager
from data.study.study_labels import labels_study
import subprocess
from data.prepdata.prepare_dataset import (
    divide_dataset, 
    remove_attacks_under_threshold, 
    preprocess, first_n_hosts_per_traffic, 
    remove_one_sample, scale_features, 
    create_attack_dataset, 
    preprocess_categorical,
    divide_dataset_per_host,
    balance_attack_data,
    prepare_training_validation_test,
    filter_hosts_by_attack,
    preprocess_numerical,
    concat_all
    )
from data.study.study_numerical import numerical_features_study
from data.study.study_categorical import categorical_features_study
from data.study.study_cat_num import categorical_numerical_features_study
from utilities.featurengineering import get_all_features_set, get_features, align_features, load_feature_types
from trainer.training_test import train_local_model, test_local_model, append_results, mean_and_std, plot_results
from sklearn.model_selection import train_test_split
import random 


setup_logging()
logger = logging.getLogger(__name__)
MAX_SPLIT = 1
THRESHOLD = 3000

def align(input_path : str, output_path : str):
    df = pd.read_csv(input_path)
    logger.info("Aligning...")
    df_aligned = align_features(df, get_all_features_set())
    logger.info("Aligned")
    df_aligned.to_csv(output_path, index=False)

def save_all_features(df1_path : str, df2_path : str):
    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)

    get_features([df1, df2])

    print(f"Dataset1 Features ({len(df1.columns)}): ")
    print(df1.columns)

    print(f"Dataset2 Features ({len(df2.columns)}):")
    print(df2.columns)

    all_features = get_all_features_set()
    print(f"All Features ({len(all_features)}): ")
    print(all_features)

def divide_per_host(input_path : str, output_path : str) -> None:
    if(not os.path.exists(output_path)):
        os.makedirs(output_path, exist_ok=True)

    logger.info("Loading data...")
    df = pd.read_csv(input_path)
    logger.info("Loaded")
    datasets = divide_dataset(df)

    for host, dataset in datasets.items():
        # Attack Label 
        logger.info(f"{host} Attack Rows:")
        print(dataset["Attack"].value_counts())

        # Save to output_path as csv
        filename = os.path.join(output_path, f"{host}.csv")
        dataset.to_csv(filename, index=False)
        logger.info(f"Saved {host}'s dataset at {filename}")
    
def prepare_attack_specific_data(input_path : str, attack : str, attack_nrows : int, output_path : str) -> None:

    logger.info("Loading data...")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows")

    logger.info(f"Preparing a dataset containing Benign and {attack}...")
    benign_mask = (df["Attack"] == "Benign")
    attack_mask = (df["Attack"] == attack)

    benign_df = df[benign_mask]

    attack_df = df[attack_mask]

    attack_rows = len(attack_df)

    # Check if attack nrows wanted are more than the actual attack rows
    if attack_rows < attack_nrows:
        logger.info(f"The number of attack rows is lower than the parameter specified, proceeding with {attack_rows} as the actual number")
        attack_nrows = attack_rows

    # If more than 0 then splice
    if attack_nrows > 0:
        attack_df = attack_df.sample(n=attack_nrows, random_state=42)

    new_df = pd.concat([benign_df, attack_df], ignore_index=True)
    new_df = new_df.sample(frac=1, random_state=42).reset_index(drop=True)

    values = new_df["Attack"].value_counts()

    logger.info(f"Created a dataset containing {len(new_df)} rows")
    logger.info(f"Benign: {values['Benign']}")
    logger.info(f"{attack}: {values[attack]}")

    new_df.drop(["Attack"], axis=1, inplace=True)

    new_df.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")

def train_model(training_path : str, validation_path : str, model_name : str, epochs : int, section : str) -> None:
    config_manager = ConfigManager()
    config_manager.load_config("config/dataset.json")
    target_column = config_manager.get_value("dataset", "target_column")


    # Get current script dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Path the requested dir to save the model into
    save_models_dir = os.path.join(project_root, "saved_models")
    folder_path = os.path.join(save_models_dir, model_name)
    
    # Dir used to save the tensorboard logs
    save_tensorboard_logs = os.path.join(project_root, "tensorboard_logs/runs")

    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(save_tensorboard_logs, exist_ok=True)

    # Model path used for saving the parameters
    model_path = os.path.join(folder_path, "best_model.pth")

    logger.info("Loading data...")

    # Get Dataset Object
    train_data = CSVDataset(filepath=training_path, target_column=target_column)
    val_data = CSVDataset(filepath=validation_path, target_column=target_column)

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

    logger.info(f"Loading model info from config/model.json")
    logger.info(f"Type of NN: {section}")

    # NN Hyperparameters
    config_manager.load_config("config/model.json")
    hidden_layers = config_manager.get_value(section, "hidden_layers")
    output = config_manager.get_value(section, "output_layer")
    dropout = config_manager.get_value(section, "dropout")
    normalization = config_manager.get_value(section, "norm")

    logger.info("Loaded data")
    
    # Learning values
    lr = config_manager.get_value(section, "lr")
    momentum = config_manager.get_value(section, "momentum")
    patience = config_manager.get_value(section, "patience")
    counter = 0
    num_features = train_data.X.shape[1]

    # Model Creation
    device = ('cuda' if torch.cuda.is_available() else 'cpu') 
    model = MyNet(num_features, output, hidden_layers, dropout, normalization).to(device=device)

    logger.info(f"Prepared a model with:")
    logger.info(f"Input Layer: {num_features} features")
    logger.info(f"Hidden Layers: {len(hidden_layers)}")
    for i, n in enumerate(hidden_layers):
        logger.info(f"Layer {i}: {n} neurons")
    logger.info(f"Output Layer: {output} neurons")
    logger.info(f"Dropout: {dropout}")
    logger.info(f"Normalization: {normalization}")

    # Define Criterion and Optimization function
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # Writer for tensorboard logs
    log_dir = os.path.join(save_tensorboard_logs, f"{model_name}_val_lr{lr}_momentum{momentum}")
    writer = SummaryWriter(log_dir=log_dir)

    # Learning Values
    print("\n")
    logger.info("Training with:")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Momentum: {momentum}")
    logger.info(f"Device: {device}")
    logger.info(f"Patience: {patience}")

    # Best Loss for early stopping
    best_loss = float('inf')

    logger.info("Training...")
    print("----------------------------------------------------------------")
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            labels = targets.to(device).unsqueeze(1)

            # Avoid gradient accumulation
            optimizer.zero_grad()

            # Predict
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Calculate gradients
            loss.backward() 

            # Gradient Descent
            optimizer.step()

            train_loss += loss.item()
        
        # Validation loss
        val_loss = 0.0
        all_preds = []
        all_labels = []

        # Validation mode
        model.eval()
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            labels = targets.to(device).unsqueeze(1)
            # Predict
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).int()         

            # Save all labels and predictions
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu().int())
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_path)
            counter = 0
        else:
            counter += 1

        all_preds = torch.cat(all_preds).squeeze().numpy()
        all_labels = torch.cat(all_labels).squeeze().numpy()

        # Metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        print(f"Epoch: {epoch}")
        # Print metrics
        print("Metrics:")
        print("Patience: {}\nTraining Loss: {}\nValidation Loss: {}".format(counter, train_loss/len(train_loader), val_loss/len(val_loader)))
        print(f"Precision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}\nAccuracy: {accuracy:.4f}\n")

        # Log everything for tensorboard visualization
        writer.add_scalar("Loss/valid", val_loss/len(val_loader), epoch)
        writer.add_scalar("Loss/train", train_loss/len(train_loader), epoch)
        writer.add_scalar("Metrics/accuracy", accuracy, epoch)
        writer.add_scalar("Metrics/precision", precision, epoch)
        writer.add_scalar("Metrics/recall", recall, epoch)
        writer.add_scalar("Metrics/f1", f1, epoch)

        # Early stopping
        if counter >= patience:
            print(f"Validation loss didn't improve for more than {patience} epochs\n")
            break
        print("----------------------------------------------------------------")

    writer.close()
    logger.info("Training Completed")

def test_model(input_path : str, model_path : str, output_path : str, section : str):
    # Verify path existence
    if(not os.path.exists(input_path)):
        raise FileNotFoundError(f"Input path {input_path} not found.")
    if(not os.path.exists(model_path)):
        raise FileNotFoundError(f"Input path {model_path} not found.")


    config_manager = ConfigManager()
    config_manager.load_config("config/dataset.json")
    target_column = config_manager.get_value("dataset", "target_column")


    test_data = CSVDataset(filepath=input_path, target_column=target_column)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=64,
        shuffle=False
    )


    logger.info(f"Loading model info from config/model.json")

    # NN Hyperparameters
    config_manager.load_config("config/model.json")
    hidden_layers = config_manager.get_value(section, "hidden_layers")
    output = config_manager.get_value(section, "output_layer")
    dropout = config_manager.get_value(section, "dropout")
    normalization = config_manager.get_value(section, "norm")

    logger.info("Loaded data")

    num_features = test_data.X.shape[1]

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyNet(num_features, output, hidden_layers, dropout, normalization).to(device)

    logger.info(f"Prepared a model with:")
    logger.info(f"Input Layer: {num_features} features")
    logger.info(f"Hidden Layers: {len(hidden_layers)}")
    for i, n in enumerate(hidden_layers):
        logger.info(f"Layer {i}: {n} neurons")
    logger.info(f"Output Layer: {output} neurons")
    logger.info(f"Dropout: {dropout}")
    logger.info(f"Normalization: {normalization}")


    model.load_state_dict(torch.load(model_path))
    criterion = nn.BCEWithLogitsLoss().to(device)


    test_loss = 0.0
    all_preds = []
    all_labels = []
    
    print("\n")
    logger.info("Testing...")

    model.eval()
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        labels = targets.to(device).unsqueeze(1)

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        test_loss += loss.item()

        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).int()

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu().int())
    
    all_preds = torch.cat(all_preds).squeeze().numpy()
    all_labels = torch.cat(all_labels).squeeze().numpy()

    logger.info("Testing Completed \n")

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print("Metrics:")
    print("Test Loss: {}".format(test_loss/len(test_loader)))
    print(f"Precision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}\nAccuracy: {accuracy:.4f}")

    # Metrics Graph
    fig, ax = plt.subplots()

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    values = [accuracy, precision, recall, f1]

    bars = ax.bar(metrics, values, label=metrics, color=['tab:blue','tab:orange','tab:green','tab:red'])
    ax.bar_label(bars, fmt='%.4f', padding=-15)
    ax.set_title('Metrics')
    ax.set_yticks(np.arange(0, 1.05, 0.05))

    # Save Graph
    os.makedirs(output_path, exist_ok=True)
    metrics_path = os.path.join(output_path, "Metrics.png")
    plt.savefig(metrics_path)
    print(f"Metrics saved in {metrics_path}")

    # Confusion Matrix
    confusionM = confusion_matrix(all_labels, all_preds)
    display_labels = ["Benign", "Attack"]
    disp = ConfusionMatrixDisplay(confusion_matrix=confusionM, display_labels=display_labels)
    disp.plot(cmap="Blues")

    # Save Confusion Matrix Image
    matrix_path = os.path.join(output_path, "TestingCM.png")
    disp.figure_.savefig(matrix_path, dpi=300, bbox_inches="tight")
    print(f"Confusion Matrix saved in {matrix_path}\n")

def prepare_data(input_path : str, output_path : str, host_output_path : str):
    logger.info("Loading data...")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows")

    first_5_hosts_per_traffic = first_n_hosts_per_traffic(df, 5)
    logger.info("Top 5 hosts per traffic:")
    logger.info(first_5_hosts_per_traffic)

    logger.info("Filtering the dataset...")
    df = df[df["IPV4_SRC_ADDR"].isin(first_5_hosts_per_traffic.keys()) | df["IPV4_DST_ADDR"].isin(first_5_hosts_per_traffic.keys())]
    logger.info("Filtered")

    logger.info(f"Created a dataset containing {len(df)} rows")

    logger.info("Traffic SRC:")
    logger.info(df["IPV4_SRC_ADDR"].value_counts())

    train_ratio = 0.6

    df, removed1 = remove_one_sample(df, "IPV4_SRC_ADDR")

    train_df, temp_data = train_test_split(df, test_size=(1-train_ratio), stratify=df["IPV4_SRC_ADDR"], random_state=42)

    temp_data, removed2 = remove_one_sample(temp_data, "IPV4_SRC_ADDR")

    validation_df, test_df = train_test_split(temp_data, test_size=0.5, stratify=temp_data["IPV4_SRC_ADDR"], random_state=42)

    test_df = pd.concat([test_df, removed1, removed2])

    del df
    del temp_data

    print("\n")
    logger.info("Training Set Traffic SRC:")
    logger.info(train_df["IPV4_SRC_ADDR"].value_counts())   
    logger.info("Training Set Attack Categories:")
    logger.info(train_df["Attack"].value_counts())

    print("\n")
    logger.info("Validation Set Traffic SRC:")
    logger.info(validation_df["IPV4_SRC_ADDR"].value_counts())
    logger.info("Validation Set Attack Categories:")
    logger.info(validation_df["Attack"].value_counts())

    print("\n")
    logger.info("Test Set Traffic SRC:")
    logger.info(test_df["IPV4_SRC_ADDR"].value_counts())
    logger.info("Test Set Attack Categories:")
    logger.info(test_df["Attack"].value_counts())

    print("\n")
    logger.info("Initiating preprocessing...")
    train_df_preprocessed, validation_df_preprocessed, test_df_preprocessed = preprocess(train_df, validation_df, test_df)

    # Riaggiungo la colonna degli attacchi per poter successivamente dividere i dataset per attacco
    train_df_preprocessed["Attack"] = train_df["Attack"]
    validation_df_preprocessed["Attack"] = validation_df["Attack"]
    test_df_preprocessed["Attack"] = test_df["Attack"]


    logger.info("Creating and saving host preprocessed datasets...\n")
    for host in first_5_hosts_per_traffic.keys():
        print("\n")
        logger.info(f"Current Host: {host}")

        temp_df = train_df_preprocessed[(train_df["IPV4_SRC_ADDR"] == host) | (train_df["IPV4_DST_ADDR"] == host)]
        logger.info(f"Created a training dataset of {len(temp_df)} rows")
        logger.info("Train Dataset Attack Categories:")
        logger.info(temp_df["Attack"].value_counts())
        temp_df.to_csv(host_output_path + f"{host}_train.csv", index=False)

        print("\n")
        temp_df = validation_df_preprocessed[(validation_df["IPV4_SRC_ADDR"] == host) | (validation_df["IPV4_DST_ADDR"] == host)]
        logger.info(f"Created a validation dataset of {len(temp_df)} rows")
        logger.info("Validation Dataset Attack Categories:")
        logger.info(temp_df["Attack"].value_counts())
        temp_df.to_csv(host_output_path + f"{host}_validation.csv", index=False)

        print("\n")
        temp_df = test_df_preprocessed[(test_df["IPV4_SRC_ADDR"] == host) | (test_df["IPV4_DST_ADDR"] == host)]
        logger.info(f"Created a test dataset of {len(temp_df)} rows")
        logger.info("Test Dataset Attack Categories:")
        logger.info(temp_df["Attack"].value_counts())
        temp_df.to_csv(host_output_path + f"{host}_test.csv", index=False)
    logger.info("Done")

    logger.info("Saving general preprocessed datasets...")
    train_df_preprocessed.to_csv(output_path + "dataset_train.csv", index=False)
    validation_df_preprocessed.to_csv(output_path + "dataset_validation.csv", index=False)
    test_df_preprocessed.to_csv(output_path + "dataset_test.csv", index=False)
    logger.info("Saved")

def reduce_dataset(input_path : str, keep_fraction : float, output_path : str):
    df = pd.read_csv(input_path)

    df_reduced, _ =  train_test_split(df ,train_size=keep_fraction, stratify=df["Label"], random_state=42)

    logger.info("Label:")
    logger.info(f"{df_reduced['Label'].value_counts()}")

    df_reduced.to_csv(output_path, index=False)

def drop_feature (input_path : str, feature : str, output_path : str):

    df = pd.read_csv(input_path)

    df.drop(columns=[feature], inplace=True)

    df.to_csv(output_path, index=False)

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
            case "6":
                    first_5_hosts_per_traffic = first_n_hosts_per_traffic(df, 5)
                    logger.info("Top 5 hosts per traffic:")
                    logger.info(first_5_hosts_per_traffic)

                    logger.info("Filtering the dataset...")
                    df = df[df["IPV4_SRC_ADDR"].isin(first_5_hosts_per_traffic.keys()) | df["IPV4_DST_ADDR"].isin(first_5_hosts_per_traffic.keys())]
                    logger.info("Filtered")

                    logger.info(f"Created a dataset containing {len(df)} rows")

                    datasets = scale_features({"dataset_scaled" : df}, numerical_features)
            case "q":
                break
            case _:
                print("WRONG INPUT")
        print_memory_usage()
        help_study()
        userInput = input("->")

def start_test(input_path : str, attack_category : str, hyperparams : str):

    # Basic Loading
    logger.info("Loading data..")
    df = pd.read_csv(input_path, nrows=1000000)
    logger.info(f"Loaded {len(df)} rows, Features: {len(df.columns)}")

    numerical_features, categorical_features, target_column = load_feature_types()

    # Utilities
    datasets_dir = "resources/datasets/experiment"
    os.makedirs(f"results/{attack_category}/local/", exist_ok=True)
    os.makedirs(datasets_dir, exist_ok=True)
    config_manager = ConfigManager()
    random.seed(42)

    # Define first 5 hosts per traffic
    hosts = first_n_hosts_per_traffic(df, 2).keys()

    # Create dataset with Benign and Attack
    df = create_attack_dataset(df, attack_category)

    # Select only the hosts with traffic containing the attack category
    hosts = filter_hosts_by_attack(hosts, df, attack_category)

    logger.info("Filtering the dataset...")
    df = df[df["IPV4_SRC_ADDR"].isin(hosts) | df["IPV4_DST_ADDR"].isin(hosts)]
    logger.info("Filtered")
    logger.info(f"Created a dataset containing {len(df)} rows\n")

    # Preprocess only the categorical features
    df = preprocess_categorical(df, categorical_features)
    
    # Divide the dataset per host, considering only the ones with the specified attack
    dfs : Dict[str : str] = divide_dataset_per_host(df, hosts, datasets_dir, 
                                                    check_for_attack=attack_category, drop_ip=True) 
    # Not needed anymore
    del df

    # Prepare Test Results Dicts
    config_manager.load_config("config/results.json")
    results_local = {}
    for host in hosts:
        template = config_manager.get_section("local")
        results_local[host] = copy.deepcopy(template["host"])

    results_federated = {}
    for i in range(2, len(hosts)):
        template = config_manager.get_section("federated")
        results_federated[f"{i}_hosts"] = copy.deepcopy(template["N_hosts"])

    # Start Experiment Iteration
    random_seeds = [random.randint(1,300) for _ in range(2)]
    for seed in random_seeds:
        logger.info(f"New Iteration(seed:{seed})")
        # Balance Attack and Benign Proportions
        dfs_balanced : Dict[str, str] = balance_attack_data(dfs, attack=attack_category, seed=seed, output_dir=datasets_dir, drop_attack=False)         
        
        # Prepare data for training, validation and testing
        dict_training_test = {}
        for host, df in dfs_balanced.items():
            logger.info(f"Current Host: {host}")
            output_path = os.path.join(datasets_dir, f"{host}_{attack_category}")
            dict_training_test[host] = prepare_training_validation_test(df, train_ratio=0.6, stratify="Attack", drop=True,
                                                                       seed=seed, output_path=output_path)
            
        # Dict containing all training, validation and test sets concatenated
        training_test_all = {}
        training_test_all["train"] = concat_all(dict_training_test, "train", datasets_dir)
        training_test_all["validation"] = concat_all(dict_training_test, "validation", datasets_dir)
        training_test_all["test"] = concat_all(dict_training_test, "test", datasets_dir)

        # Preprocess Numerical Features
        for host, datasets in dict_training_test.items():
            logger.info(f"Current: {host}")
            output_path = os.path.join(datasets_dir, f"{host}_{attack_category}")
            dict_training_test[host] = preprocess_numerical(datasets["train"], datasets["validation"],
                                                           datasets["test"], numerical_features, output_path)
        
        # Preprocess Combined Datasets Numerical Features
        logger.info("Preprocessing the combined datasets..")
        output_path = os.path.join(datasets_dir, f"combined_{attack_category}")
        training_test_all = preprocess_numerical(training_test_all["train"], training_test_all["validation"],
                                                 training_test_all["test"], numerical_features,
                                                 output_path=output_path)
        logger.info("Done\n")

        logger.info("Training and Testing Phase:\n")
        for host, datasets in dict_training_test.items():
            logger.info(f"Training Model Host: {host}")
            logger.info("Current: Local Model")
            model_name = f"{host}_{attack_category.upper()}_{hyperparams}" 

            # Train Best Model
            best_model_path = train_local_model(training_path=datasets["train"], validation_path=datasets["validation"], seed=seed,
                                           target_column=target_column, save_model_dir=f"saved_models/experiment/{attack_category}/{seed}/local",
                                           model_name=model_name, tensorboard_logs_dir=f"tensorboard_logs/experiment/{attack_category}/{seed}/local/runs",
                                           section=hyperparams)
            

            # Test Best Model on the host test set
            logger.info("Testing on host test set")
            results = test_local_model(model_path=best_model_path, test_path=datasets["test"], 
                                       results_dir=f"graphs/experiment/{attack_category}/{seed}/local/test/{model_name}", 
                                       section=hyperparams, target_column=target_column)
            logger.info("Testing Done")
            append_results(results_local[host]["test"], results)
            logger.info(f"Results for local test saved correctly\n")

            # Test Best Model on the combined test sets
            logger.info("Testing on the combined test sets")
            results = test_local_model(model_path=best_model_path, test_path=training_test_all["test"], 
                                                               results_dir=f"graphs/experiment/{attack_category}/{seed}/local/test_combined/{model_name}",
                                                               section=hyperparams, target_column=target_column)
            logger.info("Testing Done\n")
            append_results(results_local[host]["test_combined"], results)
            logger.info("Results for combined test saved correctly\n")
        
        # FEDERATED
        # Federated Config
        save_model_dir = f"saved_models/experiment/{attack_category}/{seed}/federated/"
        log_dir = f"tensorboard_logs/experiment/{attack_category}/{seed}/federated/runs"
        os.makedirs(save_model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        federated_config = {}

        partition_ids = str(list(range(len(dict_training_test))))
        partition = dict(zip(partition_ids, dict_training_test.items()))
        federated_config["partition"] = partition
        federated_config["seed"] = seed
        federated_config["net"] = {
            "hyperparams" : hyperparams,
            "features_in" : len(training_test_all["train"].columns)
        }

        logger.info("Federated:")
        for i in range(2, len(partition_ids)):
            logger.info(f"Current: Federated Model with {i} hosts")

            current_log_dir = os.path.join(log_dir, f"{i}_hosts")
            current_save_model_dir = os.path.join(save_model_dir, f"{i}_hosts/best_model.pth")
            os.makedirs(current_log_dir, exist_ok=True)
            os.makedirs(current_save_model_dir, exist_ok=True)

            federated_config["log_dir"] = current_log_dir
            federated_config["save_model_dir"] = current_save_model_dir

            # Save everything to JSON
            with open(f"config/federated_config.json") as fp:
                json.dump(federated_config, fp)

            cmd = ["flwr", "run", ".", "--run-config", f"num-supernodes={i}"]
            # Run the command and capture output
            result = subprocess.run(cmd, capture_output=True, text=True)

            print("STDOUT:")
            print(result.stdout)

            print("STDERR:")
            print(result.stderr)

            results = test_local_model(model_path=current_save_model_dir, test_path=training_test_all["test"], 
                                                               results_dir=f"graphs/experiment/{attack_category}/{seed}/federated/test_combined/{i}_hosts",
                                                               section=hyperparams, target_column=target_column)
            logger.info("Testing Done\n")
            append_results(results_federated[f"{i}_hosts"]["test_combined"], results)
            logger.info("Results for combined test saved correctly\n")

    results = {
        "local" : results_local,
        "federated" : results_federated
    }

    for _, models in results.items():
        for model, tests in models:
            for _, test_result in tests.items():
                mean_and_std(test_result)

    with open(f"results/{attack_category}/results.json", "w") as fp:
        json.dump(results, fp, indent=4)

    plot_results(results, output_path=f"results/{attack_category}/plots", metrics=["mean_precision", "mean_recall", "mean_recall", "mean_accuracy"])

if __name__ == "__main__":
    parser = ArgumentParser("parser")

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
        arguments=["--train", "--valid", "--model_name", "--epochs", "--hyperparams"],
        helps=[
            "The input path for the training data.", 
            "The input path for the validation data",
            "The name of the model", 
            "The number of epochs", 
            "The json file section for the model hyperparameters"
        ],
        defaults=["resources/datasets", "resources/datasets", "Model", 15, "small"],
    )

    parser.register_subcommand(
        subcommand="test",
        arguments=["--input", "--model", "--output", "--hyperparams"],
        helps=[
            "The input path for the testing data.",
            "The path for the model",
            "The output path for the graphs",
            "The json file section for the model hyperparameters"
        ],
        defaults=["resources/datasets", "saved_models", "graphs/models", "small"],
    )

    parser.register_subcommand(
        subcommand="prepare_attack_data",
        arguments=["--input", "--attack", "--attack_nrows", "--output"],
        helps=[
            "The input path for the data.", "The attack name", "The number of attack rows wanted (< 0 if all of them)", "The output path for the data" 
        ],
        defaults=["resources/datasets", "Benign", -1, "resources/datasets"],
    )

    parser.register_subcommand(
        subcommand="divide_per_host",
        arguments=["--input", "--output"],
        helps=[
            "The input path for the data.", "The output path for the data" 
        ],
        defaults=["resources/datasets", "resources/datasets"],
    )
    
    parser.register_subcommand(
        subcommand="prepare_data",
        arguments=["--input", "--output", "--output_host"],
        helps=[
            "The input path for the first dataset.",
            "The output path for the preprocessed general dataset",
            "The output path for the preprocessed host datasets"
        ],
        defaults=["resources/datasets", "resources/datasets", "resources/datasets/hosts"],
    )

    parser.register_subcommand(
        subcommand="reduce_data",
        arguments=["--input", "--keep_fraction", "--output"],
        helps=[
            "The input path for the first dataset.",
            "A value between 0 and 1, representing the fraction of the dataset that you want to keep",
            "The output path for the dataset"
        ],
        defaults=["resources/datasets", 0.5, "resources/datasets"],
    )

    parser.register_subcommand(
        subcommand="drop_feature",
        arguments=["--input", "--feature", "--output"],
        helps=[
            "The input path for the first dataset.",
            "Feature's name",
            "The output path for the dataset"
        ],
        defaults=["resources/datasets", 0.5, "resources/datasets"],
    )

    parser.register_subcommand(
        subcommand="start_test",
        arguments=["--input", "--attack", "--hyperparams"],
        helps=[
            "The input path for the first dataset.",
            "The attack category",
            "The json file section for the model hyperparameters"
        ],
        defaults=["resources/datasets", None, "small"],
    )

    args = parser.parse_arguments(sys.argv[1:])

    if args.subcommand == "prepare":
      prepare_data(args.input, args.output, args.output_host)

    elif args.subcommand == "study":
        study_data(args.input) 

    elif args.subcommand == "train":
        train_model(args.train, args.valid, args.model_name, int(args.epochs), args.hyperparams)
    
    elif args.subcommand == "test":
        test_model(args.input, args.model, args.output, args.hyperparams)

    elif args.subcommand == "prepare_attack_data":
        prepare_attack_specific_data(args.input, args.attack, int(args.attack_nrows), args.output)

    elif args.subcommand == "divide_per_host":
        divide_per_host(args.input, args.output)
    
    elif args.subcommand == "reduce_data":
        reduce_dataset(args.input, float(args.keep_fraction), args.output)
    
    elif args.subcommand == "drop_feature":
        drop_feature(args.input, args.feature, args.output)
    
    elif args.subcommand == "start_test":
        start_test(args.input, args.attack, args.hyperparams)
    
# 1: Preprocessing su tutto il dataset (tenendo solamente i primi 5 host) garantendo che la % di host tra training set,
# validation set e test set rimanga la stessa. Successivamente fare anche la divisione per host dei 3 set
# 2: Fare un training su un dataset contenente benign e un attacco (partendo dal più presente) cercando almeno un f1 > 80% 
# (sfruttare i dataset preprocessati nel punto 1) 
# 3: Visualizzare f1 degli host singoli nel processo federato per avere una visione migliore dell'allenamento del modello globale
# e se sta underperformando su qualche modello
# 4: Allenare un modello federato con 2 host relativi a quell'attacco e confrontare il modello generale 
# col modello globale federato testandoli su un dataset di test di un host su cui si è fatto federated. 
# 5: Infine ripetere lo stesso processo considerando uno dei due host come sottocampionato. Verificare quindi se riusciamo
# ad ottenere un buon f1 nonostante si abbia un dataset sottocampionato. 

# Per il preprocessing, inizialmente faccio in modo che ci siano solamente i 5 host più importanti 
