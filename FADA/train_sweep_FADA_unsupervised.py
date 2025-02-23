import wandb
import torch
import sys
from FADA.train_FADA_unsupervised import train_sweep
import yaml
import os

config_path = 'config/FeatureExtractorCycleGAN_CNN_RGB.yaml'
# Create the log file name by replacing the YAML extension with .txt.
log_file_name = os.path.basename(config_path).replace('.yaml', '.txt')
log_file_path = os.path.join("out", log_file_name)

# Open the log file and redirect stdout.
log_file = open(log_file_path, "w")
sys.stdout = log_file

wandb.login()

with open(config_path, "r") as file:
    sweep_config = yaml.safe_load(file)

sweep_config["parameters"]["device"] = {
    "value": "cuda:2" if torch.cuda.is_available() else "cpu"
}
sweep_id = wandb.sweep(sweep_config, project="domain-adaptation")
wandb.agent(sweep_id, function=train_sweep, count=20)

sys.stdout = sys.__stdout__
log_file.close()
