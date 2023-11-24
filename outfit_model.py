import torch.nn as nn
import torch
from image_encoder import ImageEncoder
from text_encoder import TextEncoder
from transformer_encoder import TransformerEncoder


# The core model to predict outfit compatibility
class OutfitCompatibilityModel(nn.Module):
    def __init__(
        self, image_size, text_size, transformer_input_size, num_layers=6, num_heads=16
    ):
        super(OutfitCompatibilityModel, self).__init__()
        self.image_encoder = ImageEncoder(image_dim=image_size)
        self.text_encoder = TextEncoder(text_dim=text_size)

        self.outfit_embedding = nn.Embedding(1, transformer_input_size)

        # Learnable outfit token
        self.transformer_encoder = TransformerEncoder(
            input_size=transformer_input_size,
            num_layers=num_layers,
            num_heads=num_heads,
        )

        self.mlp = nn.Sequential(
            nn.Linear(transformer_input_size, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, images, texts):
        item_embeddings = []
        image_embeddings = self.image_encoder(images)

        for i in range(images.size(0)):  # Iterate over items in the outfit
            image_embedding = self.image_encoder(image_embeddings[i : i + 1])
            text_embedding = self.text_encoder(texts[i : i + 1])
            item_feature_vector = torch.cat((image_embedding, text_embedding), dim=1)
            item_embeddings.append(item_feature_vector)

        # Stack item embeddings along a new dimension
        item_embeddings = torch.stack(item_embeddings, dim=0)

        # Prepend the outfit token to the set of feature vectors
        feature_vectors_with_token = torch.cat(
            (self.outfit_embedding(torch.zeros(1, dtype=torch.long)), item_embeddings),
            dim=1,
        )

        # Apply Transformer Encoder
        transformer_output = self.transformer_encoder(feature_vectors_with_token)

        # Feed the output through the MLP for compatibility prediction
        compatibility_score = self.mlp(transformer_output.squeeze(0))

        return compatibility_score
