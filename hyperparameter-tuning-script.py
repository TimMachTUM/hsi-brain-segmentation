import wandb
import torch
from segmentation_util import train_sweep
import sys

log_file = open('./out/hyperparameter_logs.txt', 'w')
sys.stdout = log_file
wandb.login()

sweep_config = {
    'name': 'Fundus-Segmentation-Sweep',
    'method': 'bayes',
    'parameters': {
        'encoder': {
            'values': ['resnet152', 'resnext101_32x8d', 'timm-resnest269e', 'timm-regnetx_320', 'timm-regnety_320', 'senet154']
        },
        'learning_rate': {
            "min": 0.00001, 
            "max": 0.1
        },
        'epochs': {
            'value': 10
        },
        'loss': {
            'values': ['Dice', 'BCE']
        },
        'optimizer': {
            'values': ['Adam', 'SGD']
        },
        'batch_size': {
            'value': 8
        },
        'proportion_augmented_data': {
            'values': [0.1, 0.2, 0.3]
        },
        'architecture': {
            'values': ['Unet', 'UnetPlusPlus']
        }
    },
    'metric': {
        'name': 'dice_score',
        'goal': 'maximize'
    }
}

sweep_config['parameters']['device'] = {'value':'cuda:0' if torch.cuda.is_available() else 'cpu'}
sweep_id = wandb.sweep(sweep_config, project='fundus-segmentation')
wandb.agent(sweep_id, function=train_sweep, count=10)

sys.stdout = sys.__stdout__
log_file.close()