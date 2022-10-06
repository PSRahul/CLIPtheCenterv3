import torch
import torch.nn as nn

from torchinfo import summary


class SMP_BBoxHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bbox_w_model = self.get_model(cfg)
        self.bbox_h_model = self.get_model(cfg)
        self.cfg=cfg

    def get_model(self, cfg):
        layers = []

        layers.append(
            nn.Conv2d(
                in_channels=int(cfg["smp"]["decoder_output_classes"]),
                out_channels=8,
                kernel_size=3,
                padding=1,
            )
        )
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(8))

        layers.append(
            nn.Conv2d(
                in_channels=8,
                out_channels=4,
                kernel_size=3,
                padding=1,
            )
        )
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(4))
        layers.append(
            nn.Conv2d(
                in_channels=4,
                out_channels=1,
                kernel_size=3,
                padding=1,
            )
        )
        layers.append(nn.ReLU(inplace=True))
        model = nn.Sequential(*layers)
        return model

    def forward(self, x):
        w_heatmap = self.bbox_w_model.forward(x)
        h_heatmap = self.bbox_h_model.forward(x)
        heatmap = torch.cat([w_heatmap, h_heatmap], dim=1)
        heatmap=torch.clip(heatmap,0,self.cfg["heatmap"]["output_dimension"])
        return heatmap

    def print_details(self):
        batch_size = 32
        summary(self.model, input_size=(batch_size, 256, 96, 96))
