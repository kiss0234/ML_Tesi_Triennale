"""first-federated-experience: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from federated.task import get_weights, load_hyperparameters
from src.model.MyNet import MyNet
from federated.MyFedAvg import MyFedAvg
from src.utilities.config_manager import ConfigManager

def weighted_average(metrics):
    num_examples = sum([num_example for num_example, _ in metrics])

    aggregated_metrics = {}

    for metric_name in metrics[0][1].keys():
        sum_weighted = sum([num_example * m[metric_name] for num_example, m in metrics])
        aggregated_metrics[metric_name] = sum_weighted / num_examples
    return aggregated_metrics

def config_fn(server_round):
    config = {"current_round" : server_round}

    return config

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    config_manager = ConfigManager()
    config_manager.load_config("config/federated_config.json")
    tensorboard_logs_dir = config_manager.get_value(section="utility", key="log_dir")
    save_model_dir = config_manager.get_value(section="utility", key="save_model_dir")

    # Load Net Hyperparameters
    hyperparams = load_hyperparameters(context)
    net = MyNet(**hyperparams)

    # Initialize model parameters
    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = MyFedAvg(
        hyperparams = hyperparams,
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
        on_evaluate_config_fn=config_fn,
        save_model_dir=save_model_dir,
        tensorboard_logs_dir=tensorboard_logs_dir,
        delta=0.005
    )
    
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
