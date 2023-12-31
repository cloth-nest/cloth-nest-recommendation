import logging
import torch.nn as nn
from torchvision.models import resnet18


class ImageEncoder(nn.Module):
    def __init__(self, output_embedding_dim=64):
        super().__init__()
        self.resnet18 = resnet18(pretrained=True)

        # We need a focal layer to transform an input (Resnet18's output) to our desired output (embedding with dim = output_embedding_dim). The input to this focal layer is 1000 because Resnset18's output's dimension is 1000
        self.fc_layer = nn.Linear(1000, output_embedding_dim)

    def forward(self, x):
        logging.debug(f"\n[IMAGE ENCODER] forward START")
        logging.debug(f"image_encoder.py - forward - [1] - input x's shape: {x.shape}")

        x = self.resnet18(x)
        logging.debug(
            f"image_encoder.py - forward - [2] - input x's shape after Resnet18: {x.shape}"
        )

        x = self.fc_layer(x)
        logging.debug(
            f"image_encoder.py - forward - [3] - input x's shape after fc_layer: {x.shape}"
        )
        logging.debug(f"[IMAGE ENCODER] forward END\n")

        return x
