import torch.nn as nn
import torch
from image_encoder import ImageEncoder
from text_encoder import TextEncoder
from transformer_encoder import TransformerEncoder
from utils import get_dimensions_and_lengths
import logging

# Set the root logger level to WARNING to disable logs with levels INFO and DEBUG
logging.basicConfig(level=logging.DEBUG)



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
        # Assuming images is of shape (batch_size, max_items_per_outfit, ...)
        # and texts is a 2D list where each sublist corresponds to text descriptions for the items within an outfit

        # Initialize empty list to store features for items in all outfits
        all_outfits_item_features = []
        logging.debug("!" * 10)
        logging.debug("OutfitCompatibilityModel - START")
        logging.debug(f"OutfitCompatibilityModel - intial images.shape: {images.shape}")

        # Process each outfit
        for outfit_index in range(images.size(0)):
            logging.debug("@" * 10)
            logging.debug(f"[START LOOP] OUTFIT - {outfit_index}")

            # Extract all items' images and texts for the current outfit
            outfit_images = images[outfit_index, ...]
            outfit_texts = texts[outfit_index]
            logging.debug(
                f"OutfitCompatibilityModel - outfit_images.shape: {outfit_images.shape}"
            )

            # Initialize empty list to store features for items in the current outfit
            cur_outfit_item_features = []

            # Process each item in the current outfit
            for item_index in range(outfit_images.size(0)):
                logging.debug("#" * 10)
                logging.debug(f"[START LOOP] ITEM - {item_index}")

                # Extract individual item's image and text
                item_image = outfit_images[item_index, ...].unsqueeze(0)
                item_text = outfit_texts[item_index]

                logging.debug(
                    f"OutfitCompatibilityModel - item_index: {item_index} - item_image.shape: {item_image.shape} item_text.shape: {item_text}"
                )

                # Encode item's image using ResNet18
                item_image_embedding = self.image_encoder(item_image)

                # Encode item's text using TextEncoder
                item_text_embedding = self.text_encoder(item_text)

                # Concatenate image and text embeddings to form the item's embedding
                item_features = torch.cat(
                    [item_image_embedding, item_text_embedding], dim=-1
                )

                logging.debug(
                    f"OutfitCompatibilityModel - item_features.shape: {item_features.shape}"
                )

                # Append item features to the list for the current outfit
                cur_outfit_item_features.append(item_features)

                logging.debug(f"[END LOOP] ITEM - {item_index}")
                logging.debug("#" * 10)

            dimensions, lengths = get_dimensions_and_lengths(cur_outfit_item_features)
            logging.debug(
                f"OutfitCompatibilityModel - curren outfit's feature vectors dimensions: {dimensions} with lengths: {lengths}"
            )

            # Stack individual item embeddings to form the set of feature vectors F for the current outfit
            cur_outfit_item_features = torch.stack(cur_outfit_item_features, dim=1)
            logging.debug(
                f"OutfitCompatibilityModel - CUR OUTFIT's feature vectors after stack: {cur_outfit_item_features.shape}"
            )
            # Append the set of feature vectors for the current outfit to the list for all outfits
            all_outfits_item_features.append(cur_outfit_item_features)

            logging.debug(f"[END LOOP] OUTFIT - {outfit_index}")
            logging.debug("@" * 10)

        # Stack the set of feature vectors for all outfits along the second dimension. Format is (1, 15, 5, 128)
        all_outfits_item_features = torch.stack(
            all_outfits_item_features, dim=1
        ).squeeze(0)

        logging.debug(
            f"OutfitCompatibilityModel - ALL OUTFITS' feature vetors: {all_outfits_item_features.shape}"
        )

        logging.debug(
            f"OutfitCompatibilityModel - outfit_token.shape - init: {self.outfit_token.shape}"
        )

        # Assuming all_outfits_item_features has shape (15, 5, 128) and outfit_token has shape (1, 1, 128)
        # Expand outfit_token to match the dimensions of all_outfits_item_features
        outfit_token_expanded = self.outfit_token.expand(
            all_outfits_item_features.size(0), 1, 128
        )

        logging.debug(
            f"OutfitCompatibilityModel - outfit_token.shape - after expand: {self.outfit_token.shape}"
        )

        # Concatenate outfit_token_expanded and all_outfits_item_features along the last dimension
        # The resulting outfit_features tensor has shape (15, 6, 128)

        outfit_features = torch.cat(
            [outfit_token_expanded, all_outfits_item_features], dim=1
        )

        logging.debug(
            f"OutfitCompatibilityModel - outfit_features.shape: {outfit_features.shape}"
        )

        # Apply transformer encoder
        transformer_output = self.transformer_encoder(outfit_features)

        logging.debug(
            f"OutfitCompatibilityModel - transformer_output.shape: {transformer_output.shape}"
        )

        # Perform pooling or aggregation along the sequence dimension
        global_outfit_representation = torch.mean(transformer_output, dim=1)

        logging.debug(
            f"OutfitCompatibilityModel - global_outfit_representation: {global_outfit_representation}"
        )

        # Use the output of the outfit token for MLP
        outfit_score = self.mlp(global_outfit_representation)

        logging.debug(f"OutfitCompatibilityModel - outfit_score: {outfit_score.shape}")
        logging.debug("OutfitCompatibilityModel - END")
        logging.debug("!" * 10)

        return outfit_score
