import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torchvision.models import resnet18


# Define the Image Encoder using ResNet18
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.resnet18 = resnet18(pretrained=True)
        self.embedding_layer = nn.Linear(
            1000, 64
        )  # Assuming ResNet18 output size is 1000

    def forward(self, x):
        x = self.resnet18(x)
        x = self.embedding_layer(x)
        return x
