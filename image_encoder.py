from regex import F
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torchvision.models import resnet18


# Define the Image Encoder using ResNet18
class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=64):
        super(ImageEncoder, self).__init__()
        self.resnet18 = resnet18(pretrained=True)
        self.fc_layer = nn.Linear(
            1000, embedding_dim
        )  # Assuming ResNet18 output size is 1000

    def forward(self, x):
        print("-" * 10)
        # print(f"ImageEncoder - initial x's shape: {x.shape}")
        # print(f"ImageEncoder - initial x: {x}")
        # Ensure x has at least 3 dimensions
        if len(x.shape) < 3:
            x = x.unsqueeze(0)

        x = self.resnet18(x)
        # print(f"ImageEncoder - after resnet18 x's shape: {x.shape}")
        # print(f"ImageEncoder - after resnet18 x: {x}")

        x = self.fc_layer(x)
        print(f"ImageEncoder - after fc_layer x's shape: {x.shape}")
        print("-" * 10)

        return x
