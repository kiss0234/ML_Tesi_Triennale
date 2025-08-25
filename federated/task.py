from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import pandas as pd
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.data.CSVDataset import CSVDataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.data.scalers.LogMinMaxScaler import LogMinMaxScaler
from src.data.scalers.custom_imputer import CustomImputer
from src.data.scalers.frequency_encoder import FrequencyEncoder
from src.utilities.config_manager import ConfigManager
from flwr.common import Context

fds = None  # Cache FederatedDataset
logger = logging.getLogger(__name__)

def load_data(partition_id: int):
    # Load partition
    if partition_id == 0: 
        partition = {"train" : "resources/datasets/hosts/attack/192.168.1.31_train_dos.csv",
                     "validation" : "resources/datasets/hosts/attack/192.168.1.31_validation_dos.csv"}
    elif partition_id == 1:
        partition = {"train" : "resources/datasets/hosts/attack/192.168.1.30_train_dos.csv",
                     "validation" : "resources/datasets/hosts/attack/192.168.1.30_validation_dos.csv"}
    elif partition_id == 2:
        partition = {"train" : "resources/datasets/hosts/attack/192.168.1.1_train_dos.csv",
                     "validation" : "resources/datasets/hosts/attack/192.168.1.1_validation_dos.csv"}
    elif partition_id == 3:
        partition = {"train" : "resources/datasets/hosts/attack/192.168.1.184_train_dos.csv",
                     "validation" : "resources/datasets/hosts/attack/192.168.1.184_validation_dos.csv"}
    else:
        return

    # Create CSVDataset to use for DataLoader
    training_set = CSVDataset(target_column="Label", filepath=partition["train"])
    validation_set = CSVDataset(target_column="Label", filepath=partition["validation"])

    # Create the dataloaders
    trainloader = DataLoader(training_set, batch_size=64, shuffle=True)
    valloader = DataLoader(validation_set, batch_size=64)
    return trainloader, valloader


def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available

    # Define Criterion and Optimization function
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for inputs, targets in trainloader:
            inputs = inputs.to(device)
            labels = targets.to(device).unsqueeze(1)
            # Avoid gradient accumulation
            optimizer.zero_grad()
            # Predict
            outputs = net(inputs)
            # Calculate loss
            loss = criterion(outputs, labels)
            # Calculate gradients
            loss.backward() 
            # Gradient Descent
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)

    test_loss = 0.0
    all_preds = []
    all_labels = []

    net.eval()
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs = inputs.to(device)
            labels = targets.to(device).unsqueeze(1)

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).int()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu().int())
    
    all_preds = torch.cat(all_preds).squeeze().numpy()
    all_labels = torch.cat(all_labels).squeeze().numpy()

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    loss = test_loss / len(testloader)

    return loss, accuracy, precision, recall, f1

    


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def preprocess(df : pd.DataFrame):
    # Loading feature types from json
    config_manager = ConfigManager()
    config_manager.load_config("config/dataset.json")
    numerical_columns = config_manager.get_value("dataset", "numeric_columns")
    categorical_columns = config_manager.get_value("dataset", "categorical_columns")
    target_column = config_manager.get_value("dataset", "target_column")


    train_ratio = 0.6

    train_df, temp_data = train_test_split(df, test_size=(1-train_ratio), random_state=42)
    validation_df, test_df = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Pipeline for numerical features
    num_pipeline = Pipeline([
        ("Impute", CustomImputer()),
        ("Scaling", LogMinMaxScaler())
    ])

    # Pipeline per categorical features
    cat_pipeline = Pipeline([
        ("FrequencyEncoder", FrequencyEncoder(soglia=0.5)),
        ("1hot", OneHotEncoder(sparse_output = False, handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("num", num_pipeline, numerical_columns),
        ("cat", cat_pipeline, categorical_columns),
    ])

    # Training Set Fit
    preprocessing.fit(train_df)

    # Training Set Transform
    train_preprocessed = preprocessing.transform(train_df)
    # Validation Set Transform
    validation_preprocessing = preprocessing.transform(validation_df)
    # Test Set Transform
    test_preprocessed = preprocessing.transform(test_df)
    # Get the names from the fitted OneHotEncoder
    one_hot_column_names = preprocessing.named_transformers_["cat"].named_steps["1hot"].get_feature_names_out(categorical_columns)

    all_columns = numerical_columns + list(one_hot_column_names)

    train_df_preprocessed = pd.DataFrame(data=train_preprocessed, columns=all_columns, index=train_df.index)
    validation_df_preprocessed = pd.DataFrame(data=validation_preprocessing, columns=all_columns, index=validation_df.index)
    test_df_preprocessed = pd.DataFrame(data=test_preprocessed, columns=all_columns, index=test_df.index)

    train_df_preprocessed[target_column] = train_df[target_column]
    validation_df_preprocessed[target_column] = validation_df[target_column]
    test_df_preprocessed[target_column] = test_df[target_column]

    return train_df_preprocessed, validation_df_preprocessed, test_df_preprocessed

def load_hyperparameters(context : Context):
    # Loading Hyperparameters from JSON
    config_manager = ConfigManager()
    config_manager.load_config("config/model.json")
    section = context.run_config["hyper-parameters"] 
    layers_size = config_manager.get_value(section, "hidden_layers")
    output_size = config_manager.get_value(section, "output_layer")
    dropout = config_manager.get_value(section, "dropout")
    normalization = config_manager.get_value(section, "norm")
    input_dim = context.run_config["features_in"]

    hyperparams = {"input_dim" : input_dim,
                   "output_size" : output_size,
                   "layers_size" : layers_size,
                   "dropout" : dropout,
                   "layernorm" : normalization}

    return hyperparams