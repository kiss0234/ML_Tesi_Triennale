from logging import Logger
from utilities.config_manager import ConfigManager
import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from data.CSVDataset import CSVDataset
from model.MyNet import MyNet
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

logger = Logger(__name__)

def significant_decrement(best_loss, val_loss, delta):
    return ((best_loss - val_loss ) / best_loss) > delta

def train_local_model(training_path : str, validation_path : str, target_column : str, seed, save_model_dir : str, model_name : str, tensorboard_logs_dir : str, section : str) -> None:
    config_manager = ConfigManager()
    config_manager.load_config("config/dataset.json")

    # Path the requested dir to save the model into
    save_model_dir = os.path.join(save_model_dir, model_name)

    os.makedirs(save_model_dir, exist_ok=True)
    os.makedirs(tensorboard_logs_dir, exist_ok=True)

    # Model path used for saving the parameters
    model_path = os.path.join(save_model_dir, "best_model.pth")

    logger.info("Loading data...")

    # Get Dataset Object
    train_data = CSVDataset(filepath=training_path, target_column=target_column)
    val_data = CSVDataset(filepath=validation_path, target_column=target_column)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=32,
        generator=generator,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=32,
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
    model = MyNet(num_features, output, hidden_layers, dropout, normalization, seed=seed).to(device=device)

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

    # Dir used to save the tensorboard logs
    log_dir = os.path.join(tensorboard_logs_dir, f"{model_name}_val_lr{lr}_momentum{momentum}")

    # Writer for tensorboard logs
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
    epoch = 1

    logger.info("Training...")
    print("----------------------------------------------------------------")
    while True:
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
        
        if best_loss == float('inf'):
            best_loss = val_loss
            torch.save(model.state_dict(), model_path)
            counter = 0
        elif significant_decrement(best_loss, val_loss, delta=0.005):
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

        epoch += 1

        # Early stopping
        if counter >= patience:
            print(f"Validation loss didn't improve for more than {patience} epochs\n")
            break
        print("----------------------------------------------------------------")

    writer.close()
    logger.info("Training Completed")

    return model_path


def test_local_model(test_path : str, model_path : str, results_dir : str, section : str, target_column : str):
    # Verify path existence
    if(not os.path.exists(test_path)):
        raise FileNotFoundError(f"Input path {test_path} not found.")
    if(not os.path.exists(model_path)):
        raise FileNotFoundError(f"Input path {model_path} not found.")


    config_manager = ConfigManager()
    config_manager.load_config("config/dataset.json")


    test_data = CSVDataset(filepath=test_path, target_column=target_column)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=32,
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
    accuracy = round(accuracy_score(all_labels, all_preds), 4)
    precision = round(precision_score(all_labels, all_preds, zero_division=0), 4)
    recall = round(recall_score(all_labels, all_preds, zero_division=0), 4)
    f1 = round(f1_score(all_labels, all_preds, zero_division=0), 4)

    print("Metrics:")
    print("Test Loss: {}".format(test_loss/len(test_loader)))
    print(f"Precision: {precision}\nRecall: {recall}\nF1: {f1}\nAccuracy: {accuracy}")

    # Metrics Graph
    fig, ax = plt.subplots()

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    values = [accuracy, precision, recall, f1]

    bars = ax.bar(metrics, values, label=metrics, color=['tab:blue','tab:orange','tab:green','tab:red'])
    ax.bar_label(bars, fmt='%.4f', padding=-15)
    ax.set_title('Metrics')
    ax.set_yticks(np.arange(0, 1.05, 0.05))

    # Save Graph
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "Metrics.png")
    plt.savefig(metrics_path)

    # Confusion Matrix
    confusionM = confusion_matrix(all_labels, all_preds)
    display_labels = ["Benign", "Attack"]
    disp = ConfusionMatrixDisplay(confusion_matrix=confusionM, display_labels=display_labels)
    disp.plot(cmap="Blues")

    # Save Confusion Matrix Image
    matrix_path = os.path.join(results_dir, "TestingCM.png")
    disp.figure_.savefig(matrix_path, dpi=300, bbox_inches="tight")

    results = zip(metrics, values)

    return results

def append_results(results, new_results):
    for key, value in new_results: 
        results[key].append(value)