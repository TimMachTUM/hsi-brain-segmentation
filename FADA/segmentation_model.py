import torch.nn as nn

from FADA.classifier import Classifier
from FADA.feature_extractor import FeatureExtractor
from dimensionality_reduction.gaussian import build_gaussian_channel_reducer
from segmentation_util import build_segmentation_model, load_model

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
    
def build_FADA_segmentation_model(
    architecture, encoder, in_channels, path, device, channel_reducer_path=None
):
    segmentation_model = build_segmentation_model(
        architecture=architecture,
        encoder=encoder,
        device=device,
        in_channels=in_channels,
    )
    feature_extractor = FeatureExtractor(segmentation_model).to(device)
    classifier = Classifier(segmentation_model).to(device)
    if channel_reducer_path is not None:
        gcr = build_gaussian_channel_reducer(
            num_input_channels=826,
            num_reduced_channels=3,
            load_from_path=channel_reducer_path,
            device=device,
        )
        model = SegmentationWithChannelReducerFADA(
            gcr, feature_extractor, classifier
        ).to(device)
    else:
        model = SegmentationModelFADA(feature_extractor, classifier).to(device)
        
    model = load_model(model, path, device)
    return model
