import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torchvision.models import resnet18


# Image Encoder using ResNet18 initialized with pretrained weights from ImageNet
class ImageEncoder(nn.Module):
    def __init__(self, image_dim):
        super(ImageEncoder, self).__init__()
        resnet_model = resnet18(pretrained=True)
        # Remove the classification head
        resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])
        resnet_model[-1] = nn.Linear(512, image_dim)
        self.features = resnet_model

    def forward(self, x):
        return self.features(x)
