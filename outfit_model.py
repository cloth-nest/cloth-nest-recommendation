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
        self.outfit_token = nn.Parameter(torch.rand(1, 128))
        self.mlp = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, images, texts):
        # Assuming images is of shape (batch_size, num_outfits, max_items_per_outfit, ...)
        # and texts is a 2D list where each sublist corresponds to text descriptions for the items within an outfit

        # Initialize empty list to store features for items in all outfits
        outfit_item_features = []

        # Process each outfit
        for outfit_index in range(images.size(1)):
            # Extract all items' images and texts for the current outfit
            outfit_images = images[:, outfit_index, ...]
            outfit_texts = [texts[j][outfit_index] for j in range(len(texts))]

            # Initialize empty list to store features for items in the current outfit
            outfit_item_features_current = []

            # Process each item in the current outfit
            for item_index in range(outfit_images.size(1)):
                # Extract individual item's image and text
                item_image = outfit_images[:, item_index, ...]
                item_text = outfit_texts[item_index]

                print(f"item_image: {item_image.shape} item_text: {item_text}")
                
                # Encode item's image using ResNet18
                item_image_embedding = self.image_encoder(item_image)

                # Encode item's text using TextEncoder
                item_text_embedding = self.text_encoder(item_text)

                # Concatenate image and text embeddings to form the item's embedding
                item_features = torch.cat(
                    [item_image_embedding, item_text_embedding], dim=-1
                )

                # Append item features to the list for the current outfit
                outfit_item_features_current.append(item_features)

            # Stack individual item embeddings to form the set of feature vectors F for the current outfit
            outfit_item_features_current = torch.stack(
                outfit_item_features_current, dim=1
            )

            # Append the set of feature vectors for the current outfit to the list for all outfits
            outfit_item_features.append(outfit_item_features_current)

        # Stack the set of feature vectors for all outfits along the second dimension
        outfit_item_features = torch.stack(outfit_item_features, dim=2)

        # Prepend the outfit token along the third dimension
        outfit_token = (
            self.outfit_token.unsqueeze(0).unsqueeze(1).repeat(images.size(0), 1, 1)
        )
        outfit_features = torch.cat([outfit_token, outfit_item_features], dim=2)

        # Apply transformer encoder
        transformer_output = self.transformer_encoder(outfit_features)

        # Use the output of the outfit token for MLP
        outfit_score = self.mlp(transformer_output[0, :, :])

        return outfit_score
