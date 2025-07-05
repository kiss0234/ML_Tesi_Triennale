import sys
import logging
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torch import optim
from data.CSVDataset import CSVDataset
from model.MyNet import MyNet
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
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

    values = attack_df["Attack"].value_counts()
    attack_rows = values[attack]

    # Check if attack nrows wanted are more than the actual attack rows
    if attack_rows < attack_nrows:
        logger.info(f"There number of attack rows if lower than the parameter specified, proceeding with {attack_rows} as the actual number")
        attack_nrows = attack_rows

    # If more than 0 then splice
    if attack_nrows > 0:
        attack_df = attack_df.iloc[:attack_nrows]

    frames = [benign_df, attack_df]
    new_df = pd.concat(frames)

    values = new_df["Attack"].value_counts()

    logger.info(f"Created a dataset containing {len(new_df)} rows")
    logger.info(f"Benign: {values['Benign']}")
    logger.info(f"{attack}: {values[attack]}")

    logger.info("Labels:")
    print(new_df["Label"].value_counts())

    new_df.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")

def prepare_data(input_path: str, output_path: str) -> None:
    logger.info("Preparing data...")

    df = pd.read_csv(input_path)

    logger.info(f"Loaded {df.shape[0]} rows")

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
        # ("Binning", CustomBinner()),
        ("Scaling", LogMinMaxScaler())
    ])

    # Pipeline per categorical features
    cat_pipeline = Pipeline([
        ("FrequencyEncoder", FrequencyEncoder(soglia=0.5)),
        ("1hot", OneHotEncoder(sparse_output = False)),
    ])

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

    labels_train = train_df[target_column].value_counts()
    labels_test = test_df[target_column].value_counts()

    print("Training Set Labels:")
    print(labels_train)

    print("Test Set Labels:")
    print(labels_test)

    if output_path != None:
        train_df.to_csv(output_path + "_train.csv", index=False)
        test_df.to_csv(output_path + "_test.csv", index=False)

    logger.info("Preprocessed and saved correctly")

def train_model(input_path : str, model_name : str, epochs : int, section : str) -> None:
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
    dataset = CSVDataset(input_path, target_column)

    # Define training dataset and validation dataset
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
    num_features = dataset.X.shape[1]

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

    # num_positives = dataset.Y.sum()
    # num_negatives = len(dataset.Y) - num_positives
    # pos_weight = torch.tensor(num_negatives / num_positives, dtype=torch.float32, device=device)

    # Define Criterion and Optimization function
    criterion = nn.BCEWithLogitsLoss()
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
    # logger.info(f"Pos_Weight: {pos_weight:.4f}")

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


    test_data = CSVDataset(input_path, target_column)

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
    criterion = nn.BCEWithLogitsLoss()


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

    # Confusion Matrix
    confusionM = confusion_matrix(all_labels, all_preds)
    display_labels = ["Benign", "Attack"]
    disp = ConfusionMatrixDisplay(confusion_matrix=confusionM, display_labels=display_labels)
    disp.plot(cmap="Blues")

    # Save Confusion Matrix Image
    os.makedirs(output_path, exist_ok=True)
    full_path = os.path.join(output_path, "TestingCM.png")
    disp.figure_.savefig(full_path, dpi=300, bbox_inches="tight")
    print(f"Confusion Matrix saved in {full_path}\n")


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
    df = pd.read_csv(input_path, nrows=1000) 
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
        arguments=["--input", "--model_name", "--epochs", "--hyperparams"],
        helps=[
            "The input path for the data.", 
            "The name of the model", 
            "The number of epochs", 
            "The json file section for the model hyperparameters"
        ],
        defaults=["resources/datasets", "Model", 15, "small"],
    )

    parser.register_subcommand(
        subcommand="test",
        arguments=["--input", "--model", "--output", "--hyperparams"],
        helps=[
            "The input path for the test data.",
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


    args = parser.parse_arguments(sys.argv[1:])

    if args.subcommand == "prepare":
        prepare_data(args.input, args.output)

    elif args.subcommand == "study":
        study_data(args.input) 

    elif args.subcommand == "train":
        train_model(args.input, args.model_name, int(args.epochs), args.hyperparams)
    
    elif args.subcommand == "test":
        test_model(args.input, args.model, args.output, args.hyperparams)

    elif args.subcommand == "prepare_attack_data":
        prepare_attack_specific_data(args.input, args.attack, int(args.attack_nrows), args.output)

    elif args.subcommand == "divide_per_host":
        divide_per_host(args.input, args.output)


# Imparare flower + featered learning
# Test con 2 host con molti campioni, 1 solo attacco
# Test con stessi 2 host uno sotto campionato con attacco (togli anche fino ad arrivare a 10000/1000 sample attacco totali)
# Call il 15 alle 15.