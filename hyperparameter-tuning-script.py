import wandb
import torch
from segmentation_util import train_sweep
import sys

log_file = open('./out/hyperparameter_logs_3-channels.txt', 'w')
sys.stdout = log_file
wandb.login()

sweep_config = {
    'name': 'Fundus-Segmentation-Sweep-3-channels',
    'method': 'bayes',
    'parameters': {
        'encoder': {
            'values': ['timm-resnest269e', 'timm-regnetx_320', 'timm-regnety_320', 'senet154']
        },
        'learning_rate': {
            "min": 0.001, 
            "max": 0.04
        },
        'epochs': {
            'value': 10
        },
        'loss': {
            'values': ['Dice', 'BCE']
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
            'values': ['UnetPlusPlus', 'Linknet', 'Unet']
        },
        'channels': {
            'value': 3
        },
    },
    'metric': {
        'name': 'dice_score',
        'goal': 'maximize'
    }
}

sweep_config['parameters']['device'] = {'value':'cuda:2' if torch.cuda.is_available() else 'cpu'}
sweep_id = wandb.sweep(sweep_config, project='fundus-segmentation')
wandb.agent(sweep_id, function=train_sweep, count=20)

sys.stdout = sys.__stdout__
log_file.close()