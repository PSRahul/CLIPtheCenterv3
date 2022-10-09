import torch.nn as nn
import torch
from torchinfo import summary

from torchvision.models import ResNet18_Weights

class SMP_Embedder(nn.Module):
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
                out_channels=3,
                kernel_size=3,
                stride=1,

            ))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(3))
        layers.append(nn.AvgPool2d(kernel_size=16,
                                   stride=4,
                                   padding=1
                                   ))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(768, 512))

        self.model = nn.Sequential(*layers)

    def forward(self, masked_heatmaps_features):
        return self.model(masked_heatmaps_features)

    def print_details(self):
        pass

class ResNet_Embedder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0",
            "resnet18", weights=ResNet18_Weights.DEFAULT
        )
        self.model.conv1 = torch.nn.Conv1d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
        self.model = nn.Sequential(*list(self.model.children())[:-2])



        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=512,
                out_channels=64,
                kernel_size=3,
                padding=1

            ))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.BatchNorm2d(64))
        layers.append(
            nn.Conv2d(
                in_channels=64,
                out_channels=8,
                kernel_size=3,
                padding=1

            ))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.BatchNorm2d(8))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(800, 512))


        self.conv_model = nn.Sequential(*layers)

    def forward(self, masked_heatmaps_features):
        return self.conv_model(self.model(masked_heatmaps_features))

    def print_details(self):
        pass
