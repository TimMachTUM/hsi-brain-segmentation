import wandb
import torch
import sys
from FADA.train_FADA_supervised import train_sweep
import yaml

log_file = open("./out/hyperparameter_logs.txt", "w")
sys.stdout = log_file
wandb.login()

with open("./config/learning_rate.yaml", "r") as file:
    sweep_config = yaml.safe_load(file)

sweep_config["parameters"]["device"] = {
    "value": "cuda:2" if torch.cuda.is_available() else "cpu"
}
sweep_id = wandb.sweep(sweep_config, project="supervised-domain-adaptation")
wandb.agent(sweep_id, function=train_sweep)

sys.stdout = sys.__stdout__
log_file.close()
