import torch.nn as nn
import torch
from torchinfo import summary
from network.models.EfficientnetConv2DT.utils import get_bounding_box_prediction
import clip
from torchvision.models import ResNet18_Weights


class EfficientnetConv2DT_RoIHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        layers = []

        layers.append(
            nn.Conv2d(
                in_channels=1,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,

            ))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(3))
        layers.append(nn.MaxPool2d(kernel_size=16,
                                   stride=4,
                                   ))

        layers.append(
            nn.Conv2d(
                in_channels=3,
                out_channels=256,
                kernel_size=3,
                stride=1,

            ))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.AvgPool2d(kernel_size=16,
                                   stride=4,
                                   padding=1
                                   ))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(256, 512))

        self.model = nn.Sequential(*layers)

    def forward(self, masked_heatmaps_features):
        return self.model(masked_heatmaps_features)

    def print_details(self):
        pass


class SMP_RoIHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        layers = []

        layers.append(
            nn.Conv2d(
                in_channels=int(cfg["smp"]["decoder_output_classes"]),
                out_channels=32,
                kernel_size=3,
                padding=1

            ))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(32))
        layers.append(
            nn.Conv2d(
                in_channels=32,
                out_channels=256,
                kernel_size=3,
                padding=1,

            ))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(256))
        layers.append(
            nn.Conv2d(
                in_channels=256,
                out_channels=1,
                kernel_size=3,
                padding=1,

            ))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model.forward(x)

    def print_details(self):
        batch_size = 32
        summary(self.model, input_size=(batch_size, 256, 96, 96))

class ResNet_RoIHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        layers = []

        super().__init__()
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0",
            "resnet18", weights=ResNet18_Weights.DEFAULT
        )
        self.model = nn.Sequential(*list(self.model.children())[:-2])

        layers = []

        layers.append(
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=32,
                kernel_size=2,
                stride=2

            ))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.BatchNorm2d(32))
        layers.append(
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=4,
                stride=4

            ))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(16))
        layers.append(
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=8,
                kernel_size=2,
                stride=2

            ))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(8))
        layers.append(
            nn.ConvTranspose2d(
                in_channels=8,
                out_channels=1,
                kernel_size=2,
                stride=2

            ))
        layers.append(nn.ReLU(inplace=True))
        self.conv_model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model.forward(x)
        x = self.conv_model.forward(x)
        return x

    def print_details(self):
        batch_size = 32
        summary(self.model, input_size=(batch_size, 256, 96, 96))
