import wandb
import torch
import sys
from FADA.train_FADA_supervised import train_sweep

log_file = open("./out/hyperparameter_logs.txt", "w")
sys.stdout = log_file
wandb.login()

sweep_config = {
    "name": "FADA-Supervised-Sweep",
    "method": "grid",
    "parameters": {
        "architecture": {"value": "Linknet"},
        "encoder": {"value": "timm-regnetx_320"},
        "in_channels": {"value": 1},
        "optimizer": {"value": "Adam"},
        "batch_size_source": {"value": 16},
        "batch_size_target": {"value": 8},
        "dataset_path": {"value": "./data/FIVES_random_crops_threshold01"},
        "learning_rate_fea": {"value": 0.001},
        "learning_rate_cls": {"value": 0.001},
        "learning_rate_dis": {"value": 0.0001},
        "ndf": {"value": 512},
        "epochs": {"value": 10},
        "seg_loss": {"value": "BCE"},
        "pretrained": {
            "value": "./models/Linknet-timm-regnetx_320-512x512-augmented-with-random-crops-single-channel-focal.pth"
        },
        "augmented": {"value": True},
        "window": {"value": (500, 600)},
        "train_indices": {
            "values": [
                [0],
                [1],
                [2],
                [3],
                [4],
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [1, 2],
                [1, 3],
                [1, 4],
                [2, 3],
                [2, 4],
                [3, 4],
                [0, 1, 2],
                [0, 1, 3],
                [0, 1, 4],
                [0, 2, 3],
                [0, 2, 4],
                [0, 3, 4],
                [1, 2, 3],
                [1, 2, 4],
                [1, 3, 4],
                [2, 3, 4],
                [0, 1, 2, 3],
                [0, 1, 2, 4],
                [0, 1, 3, 4],
                [0, 2, 3, 4],
                [1, 2, 3, 4],
            ]
        },
    },
    "metric": {"name": "dice_score/target", "goal": "maximize"},
}

sweep_config["parameters"]["device"] = {
    "value": "cuda:2" if torch.cuda.is_available() else "cpu"
}
sweep_id = wandb.sweep(sweep_config, project="supervised-domain-adaptation")
wandb.agent(sweep_id, function=train_sweep)

sys.stdout = sys.__stdout__
log_file.close()
