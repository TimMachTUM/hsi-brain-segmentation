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
    
class SegmentationWithChannelReducerFADA(nn.Module):
    def __init__(self, channel_reducer, feature_extractor, classifier):
        super(SegmentationWithChannelReducerFADA, self).__init__()
        self.channel_reducer = channel_reducer
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, x):
        if x.shape[1] > 3:
            x = self.channel_reducer(x)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x