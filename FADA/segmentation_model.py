import torch.nn as nn

class SegmentationModelFADA(nn.Module):
    def __init__(self, feature_extractor, classifier):
        super(SegmentationModelFADA, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x