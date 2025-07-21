"""first-federated-experience: A Flower / PyTorch app."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from federated.task import get_weights, load_data, set_weights, test, train, load_hyperparameters
from src.model.MyNet import MyNet
from torch.utils.tensorboard import SummaryWriter

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, partition_id):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.partition_id = partition_id

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        current_round = config.get("current_round")
        print(f"Client: {self.partition_id} Current_round: {current_round}")
        loss, accuracy, precision, recall, f1 = test(self.net, self.valloader, self.device)

        writer = SummaryWriter(log_dir="tensorboard_logs/federated/runs/results")

        writer.add_scalar(f"Client_{self.partition_id}/Metrics/Accuracy/Valid", accuracy, global_step=current_round)
        writer.add_scalar(f"Client_{self.partition_id}/Metrics/Precision/Valid", precision, global_step=current_round)
        writer.add_scalar(f"Client_{self.partition_id}/Metrics/Recall/Valid", recall, global_step=current_round)
        writer.add_scalar(f"Client_{self.partition_id}/Metrics/F1Score/Valid", f1, global_step=current_round)

        writer.close()

        return loss, len(self.valloader.dataset), {"accuracy": accuracy,
                                                   "precision" : precision,
                                                   "recall" : recall,
                                                   "f1" : f1}

def client_fn(context: Context):
    # Load Net Hyperparameters
    hyperparams = load_hyperparameters(context)
    net = MyNet(**hyperparams)

    partition_id = context.node_config["partition-id"]
    trainloader, valloader = load_data(partition_id)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs, partition_id).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
