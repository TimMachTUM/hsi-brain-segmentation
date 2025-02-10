import torch
from torch import nn

class FeatureExtractor(nn.Module):
    def __init__(self, base_model):
        super(FeatureExtractor, self).__init__()
        self.encoder = base_model.encoder

    def forward(self, x):
        features = self.encoder(x)
        return features
    
class FeatureExtractorWithConvReducer(nn.Module):
    def __init__(self, base_model, hyperspectral_channels=826, freeze_encoder=False, encoder_in_channels=1):
        """
        Args:
            base_model (nn.Module): Pretrained segmentation model containing:
                                    - base_model.in_channels: expected input channels for the encoder.
                                    - base_model.encoder: the feature extraction (encoder) module.
            hyperspectral_channels (int): Number of channels for hyperspectral input data.
            freeze_encoder (bool): If True, the encoder's parameters will not be updated.
        """
        super(FeatureExtractorWithConvReducer, self).__init__()
        
        # Store the expected number of channels (e.g., 1 or 3)
        self.expected_channels = encoder_in_channels
        self.hyperspectral_channels = hyperspectral_channels
        
        # Create the 1x1 convolution for dimensionality reduction.
        self.dim_reduction = nn.Conv2d(
            in_channels=self.hyperspectral_channels,
            out_channels=self.expected_channels,
            kernel_size=1
        )
        
        # CNN transformation block that preserves spatial dimensions.
        # self.cnn_transform = nn.Sequential(
        #     nn.Conv2d(self.expected_channels, self.expected_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(self.expected_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.expected_channels, self.expected_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(self.expected_channels),
        #     nn.ReLU(inplace=True)
        # )
        
        # The pretrained encoder from the base model.
        self.encoder = base_model.encoder
        
        # Optionally freeze the encoder.
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x: shape (batch_size, channels, height, width)
        # If the input channel count does not match what the encoder expects, apply dimensionality reduction.
        if x.shape[1] != self.expected_channels:
            x = self.dim_reduction(x)
            # Apply the CNN transformation.
            # x = self.cnn_transform(x)
        # Extract high-level features using the encoder.
        features = self.encoder(x)
        return features
    
    def reduce_for_visualization(self, x):
        """
        Applies the dimensionality reduction and CNN transformation without passing the data through the encoder.
        Use this method to visualize the representation that the segmentation network would see.
        
        Args:
            x (Tensor): Input image tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tensor: The transformed image after reduction and CNN transformation.
        """
        if x.shape[1] != self.expected_channels:
            x = self.dim_reduction(x)
            # x = self.cnn_transform(x)
        return x
