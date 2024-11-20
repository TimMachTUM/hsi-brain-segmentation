import wandb
import torch
from segmentation_util import train_sweep
import sys

log_file = open('./out/hyperparameter_logs.txt', 'w')
sys.stdout = log_file
wandb.login()

sweep_config = {
    'name': '3-Channels-Sweep',
    'method': 'bayes',
    'parameters': {
        'encoder': {
            'value': 'timm-regnetx_320'
        },
        'learning_rate': {
            "min": 0.01, 
            "max": 0.1
        },
        'epochs': {
            'value': 15
        },
        'loss': {
            'value': 'BCE' 
        },
        'optimizer': {
            'value': 'Adam'
        },
        'batch_size': {
            'value': 8
        },
        'proportion_augmented_data': {
            'value': 0.1
        },
        'architecture': {
            'value': 'Linknet',
        },
        'channels': {
            'value': 3
        },
        'gamma': {
            'value': [2,3,4]
        },
        'width': {
            'value': 512
        },
        'height': {
            'value': 512
        },
    },
    'metric': {
        'name': 'dice_score',
        'goal': 'maximize'
    }
}

sweep_config['parameters']['device'] = {'value':'cuda:2' if torch.cuda.is_available() else 'cpu'}
sweep_id = wandb.sweep(sweep_config, project='fundus-segmentation')
wandb.agent(sweep_id, function=train_sweep, count=10)

sys.stdout = sys.__stdout__
log_file.close()