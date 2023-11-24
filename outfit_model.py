import torch.nn as nn
import torch
from image_encoder import ImageEncoder
from text_encoder import TextEncoder
from transformer_encoder import TransformerEncoder


# Define the Outfit Compatibility Model
class OutfitCompatibilityModel(nn.Module):
    def __init__(self):
        super(OutfitCompatibilityModel, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=16), num_layers=6
        )
        self.outfit_token = nn.Parameter(torch.rand(1, 1, 128))
        self.mlp = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, images, texts):
        image_embeddings = self.image_encoder(images)
        text_embeddings = self.text_encoder(texts)
        outfit_features = torch.cat([image_embeddings, text_embeddings], dim=1)

        # Calculate outfit token by prepending it to the outfit features
        outfit_token = self.outfit_token.repeat(outfit_features.size(0), 1).unsqueeze(1)
        outfit_features = torch.cat([outfit_token, outfit_features.unsqueeze(1)], dim=1)

        transformer_output = self.transformer_encoder(outfit_features)
        outfit_score = self.mlp(
            transformer_output[:, 0, :]
        )  # Use the output of the outfit token

        return outfit_score
