import wandb
import torch
import sys
from FADA.train_FADA_unsupervised import train_sweep

log_file = open("./out/hyperparameter_logs.txt", "w")
sys.stdout = log_file
wandb.login()

sweep_config = {
    "name": "FADA-Learning-Rate-Sweep",
    "method": "bayes",
    "parameters": {
        "architecture": {"value": "Linknet"},
        "encoder": {"value": "timm-regnetx_320"},
        "in_channels": {"value": 1},
        "optimizer": {"value": "Adam"},
        "batch_size_source": {"value": 16},
        "batch_size_target": {"value": 8},
        "dataset_path": {"value": "./data/FIVES_random_crops_threshold01"},
        "learning_rate_fea": {"min": 0.001, "max": 0.1},
        "learning_rate_cls": {"min": 0.001, "max": 0.1},
        "learning_rate_dis": {"min": 0.0001, "max": 0.1},
        "ndf": {"value": 512},
        "epochs": {"value": 10},
        "seg_loss": {"value": "BCE"},
        "pretrained": {
            "value": "./models/Linknet-timm-regnetx_320-512x512-augmented-with-random-crops-single-channel-focal.pth"
        },
        "augmented": {"value": True},
        "window": {"value": (500, 600)},
    },
    "metric": {"name": "dice_score/target", "goal": "maximize"},
}

sweep_config["parameters"]["device"] = {
    "value": "cuda:2" if torch.cuda.is_available() else "cpu"
}
sweep_id = wandb.sweep(sweep_config, project="domain-adaptation")
wandb.agent(sweep_id, function=train_sweep, count=20)

sys.stdout = sys.__stdout__
log_file.close()