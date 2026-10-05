import torch 
from torch import nn
from torchvision.models import EfficientNet_B0_Weights,efficientnet_b0


class PlantNet(nn.Module):
    def __init__(self, output_shape):
        super().__init__()

        weights = EfficientNet_B0_Weights.DEFAULT
        self.model = efficientnet_b0(weights=weights)

        self.model.classifier[1] = nn.Linear(
            in_features=self.model.classifier[1].in_features,
            out_features=output_shape
        )

    def forward(self, x):
        return self.model(x)