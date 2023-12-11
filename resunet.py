import torch
import torch.nn as nn
from torchvision.models import resnet50
from unet_parts import *


class PretrainedResNet(nn.Module):
    def __init__(self):
        super(PretrainedResNet, self).__init__()
        resnet = resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # Remove the last FC layer

    def forward(self, x):
        x = self.features(x)
        return x


class UNet_1(nn.Module):
    def __init__(self, n_channels, n_classes, dropout_rate=0.5):
        super(UNet_1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)
        self.down4 = DoubleConv(512, 512)
        self.up1 = DoubleConv(1024, 256)
        self.up2 = DoubleConv(512, 128)
        self.up3 = DoubleConv(256, 64)
        self.up4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.sigmoid_activation = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout_rate)



    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.dropout(x2)
        x3 = self.down2(x2)
        x3 = self.dropout(x3)
        x4 = self.down3(x3)
        x4 = self.dropout(x4)
        x5 = self.down4(x4)
        x5 = self.dropout(x5)
        x = self.up1(torch.cat([x4, x5], dim=1))
        x = self.up2(torch.cat([x3, x], dim=1))
        x = self.up3(torch.cat([x2, x], dim=1))
        x = self.up4(torch.cat([x1, x], dim=1))
        logits = self.outc(x)
        # logits = self.sigmoid_activation(x)

        return logits



class ResNetUNet(nn.Module):
    def __init__(self):
        super(ResNetUNet, self).__init__()
        self.resnet_model = PretrainedResNet()
        # Assuming ResNet outputs 2048 feature maps, and we have 6 images
        self.unet_model = UNet_1(input_channels=2048*6, output_channels=1)  # output_channels depends on your segmentation task

    def forward(self, *input_imgs):
        # Process each image through ResNet
        features = [self.resnet_model(img) for img in input_imgs]
        # Concatenate features along the channel dimension
        combined_features = torch.cat(features, dim=1)
        # Pass concatenated features through U-Net
        output = self.unet_model(combined_features)
        return output


