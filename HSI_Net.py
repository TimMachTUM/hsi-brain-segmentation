import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import segmentation_models_pytorch as smp
from torch.autograd import Function
import torch.nn as nn
import torch


class HyperspectralToGrayscale(nn.Module):
    def __init__(self):
        super(HyperspectralToGrayscale, self).__init__()
        # Define a convolutional layer that takes 832 input channels and reduces them to 1 channel.
        # Kernel size, stride, and padding can be adjusted depending on the specific requirements and data characteristics.
        self.conv = nn.Conv2d(826, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Apply the convolutional layer
        x = self.conv(x)
        return x


class ConvolutionalReducer(nn.Module):
    def __init__(self):
        super(ConvolutionalReducer, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(826, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.layers(x)


class CombinedModel(nn.Module):
    def __init__(self, encoder, segmenter):
        super(CombinedModel, self).__init__()
        self.encoder = (
            encoder  # This could be an autoencoder, PCA layer, or convolutional layer
        )
        self.segmenter = segmenter  # This is the pre-trained segmentation model

    def forward(self, x):
        original_size = x.shape[2:]
        x = self.encoder(x)  # Reduce dimensionality
        x, pad = _pad_input(x)
        x = self.segmenter(x)  # Segment
        x = _crop_output(x, original_size)
        return x


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, encoder_name="resnet34"):
        super(Unet, self).__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name, in_channels=in_channels, classes=out_channels
        )

    def forward(self, x):
        original_size = x.shape[2:]
        x, pad = _pad_input(x)
        x = self.model(x)
        x = _crop_output(x, original_size)
        return x


class UnetPlusPlus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetPlusPlus, self).__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34", in_channels=in_channels, classes=out_channels
        )

    def forward(self, x):
        original_size = x.shape[2:]
        x, pad = _pad_input(x)
        x = self.model(x)
        x = _crop_output(x, original_size)
        return x


class MAnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MAnet, self).__init__()
        self.model = smp.MAnet(
            encoder_name="resnet34", in_channels=in_channels, classes=out_channels
        )

    def forward(self, x):
        original_size = x.shape[2:]
        x, pad = _pad_input(x)
        x = self.model(x)
        x = _crop_output(x, original_size)
        return x


class Linknet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Linknet, self).__init__()
        self.model = smp.Linknet(
            encoder_name="resnet34", in_channels=in_channels, classes=out_channels
        )

    def forward(self, x):
        original_size = x.shape[2:]
        x, pad = _pad_input(x)
        x = self.model(x)
        x = _crop_output(x, original_size)
        return x


def _pad_input(x):
    # Calculate padding
    _, _, h, w = x.size()
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad = (pad_left, pad_right, pad_top, pad_bottom)

    # Apply padding
    x = F.pad(x, pad, mode="replicate")
    return x, pad


def _crop_output(x, original_size):
    # Crop to the original size
    return x[:, :, : original_size[0], : original_size[1]]


class SegmentationModelWithWindowing(nn.Module):
    def __init__(self, pretrained_unet, window):
        super(SegmentationModelWithWindowing, self).__init__()
        self.pretrained_unet = pretrained_unet
        self.window = window

    def forward(self, hsi_image):
        # Apply windowing
        hsi_image_windowed = hsi_image[:, self.window[0] : self.window[1], :, :]

        # Calculate median along the windowed dimension
        hsi_image_median = torch.median(hsi_image_windowed, dim=1).values

        # Normalize the image
        hsi_image_min = (
            hsi_image_median.min(dim=1, keepdim=True)
            .values.min(dim=2, keepdim=True)
            .values
        )
        hsi_image_max = (
            hsi_image_median.max(dim=1, keepdim=True)
            .values.max(dim=2, keepdim=True)
            .values
        )
        hsi_image_normalized = (hsi_image_median - hsi_image_min) / (
            hsi_image_max - hsi_image_min
        )

        # Add channel dimension
        hsi_image_normalized = hsi_image_normalized.unsqueeze(1)

        # Forward pass through the U-Net model
        output = self.pretrained_unet(hsi_image_normalized)
        return output


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_param):
        ctx.lambda_param = lambda_param
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_param, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_param=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_param = lambda_param

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_param)


class DomainClassifierFC(nn.Module):
    def __init__(self, input_shape=(512, 7, 7), num_domains=2):
        super(DomainClassifierFC, self).__init__()
        # Flatten layer to reshape input from [8, 512, 7, 7] to [8, 512 * 7 * 7]
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(
                input_shape[0] * input_shape[1] * input_shape[2], 1024
            ),  # [8, 512 * 7 * 7] -> [8, 1024]
            nn.ReLU(),
            nn.Linear(1024, 512),  # [8, 1024] -> [8, 512]
            nn.ReLU(),
            nn.Linear(512, num_domains),  # [8, 512] -> [8, num_domains]
            nn.Sigmoid(),  # Sigmoid output to distinguish between the domains
        )

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input tensor
        x = self.fc_layers(x)
        return x


class ModelWithDomainAdaptation(nn.Module):
    def __init__(self, base_model, lambda_param, domain_classifier):
        super(ModelWithDomainAdaptation, self).__init__()
        self.encoder = base_model.encoder
        self.decoder = base_model.decoder
        self.segmentation_head = base_model.segmentation_head
        self.grl = GradientReversalLayer(lambda_param)
        self.domain_classifier = domain_classifier

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        segmentation_output = self.segmentation_head(decoder_output)
        if not self.training:
            return segmentation_output
        else:
            reversed_features = self.grl(features[-1])
            domain_output = self.domain_classifier(reversed_features)
            return segmentation_output, domain_output
