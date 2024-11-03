import torch
from torch import nn

class FeatureExtractor(nn.Module):
    def __init__(self, base_model):
        super(FeatureExtractor, self).__init__()
        self.encoder = base_model.encoder

    def forward(self, x):
        features = self.encoder(x)
        return features