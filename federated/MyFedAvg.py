from flwr.server.strategy import FedAvg
from src.model.MyNet import MyNet
from flwr.common import parameters_to_ndarrays
from federated.task import set_weights
import torch
from torch.utils.tensorboard import SummaryWriter

class MyFedAvg(FedAvg):

    def __init__(self, hyperparams, *args, **kwargs):
        self.hyperparams = hyperparams
        return super().__init__(*args, **kwargs)
    
    def aggregate_fit(self, server_round, results, failures):
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        net = MyNet(**self.hyperparams)
        weights = parameters_to_ndarrays(parameters_aggregated)
        set_weights(net, weights)

        # Log every metric
        writer = SummaryWriter(log_dir="tensorboard_logs/federated/runs/results")

        writer.add_scalar("Loss/Train", metrics_aggregated["train_loss"], global_step=server_round)

        writer.close()

        torch.save(net.state_dict(), f"saved_models/federated/global_round_{server_round}")

        return parameters_aggregated, metrics_aggregated
    
    def aggregate_evaluate(self, server_round, results, failures):
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)

        # Log every metric
        writer = SummaryWriter(log_dir="tensorboard_logs/federated/runs/results")

        writer.add_scalar("Loss/Valid", loss_aggregated, global_step=server_round)
        writer.add_scalar("Metrics/Accuracy/Valid", metrics_aggregated["accuracy"], global_step=server_round)
        writer.add_scalar("Metrics/Precision/Valid", metrics_aggregated["precision"], global_step=server_round)
        writer.add_scalar("Metrics/Recall/Valid", metrics_aggregated["recall"], global_step=server_round)
        writer.add_scalar("Metrics/F1Score/Valid", metrics_aggregated["f1"], global_step=server_round)

        writer.close()

        return loss_aggregated, metrics_aggregated