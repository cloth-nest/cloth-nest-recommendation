import logging
import torch
import torch.nn as nn

from image_encoder import ImageEncoder
from text_encoder import TextEncoder
from transformer_encoder import TransformerEncoder


class OutfitCompatibilityModel(nn.Module):
    def __init__(self):
        super(OutfitCompatibilityModel, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.transformer_encoder = TransformerEncoder(input_size=128)
        self.outfit_token = nn.Embedding(1, 128)
        self.mlp = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, outfits_images, outfits_texts):
        """
        outfits_images should have shape like this:
        (batch_size, max_num_of_items_per_outfit, img_channel, img_width, img_height)

        outfit_texts should be 2D list with size (batch_size, max_num_of_items_per_outfit) with each sublist contains all
        items' text descriptions of one outfit

        batch_size is the number of outfits per batch. This forward method
        works with 1 batch of outfits at a time
        """
        logging.debug(f"\n[OUTFIT COMPATIBILITY MODEL] forward START")
        logging.debug(
            f"outfit_compatibility_model.py - forward - [1] - input outfits_images' shape: {outfits_images}"
        )

        all_outfits_features = []

        # For each outfit in this batch, ext
        for outfit_index in range(outfits_images[0]):
            # The "..." means getting the rest of the dimensions, in this case it gets the current outfit's all items' images
            outfit_item_images = outfits_images[outfit_index, ...]
            outfit_item_descriptions = outfits_texts[outfit_index]

            if outfit_index == 0:
                logging.debug(
                    f"outfit_compatibility_model.py - forward - [2] - 1st outfit: \n- outfit_item_images' shape: {outfit_item_images.shape}; \n- outfit_item_descriptions's length: {len(outfit_item_descriptions)}"
                )

            cur_outfit_features = []

            # For each item in this outfit, extract the item feature vector
            for item_index in range(outfit_item_images[0]):
                item_image = outfit_item_images[item_index, ...]
                item_description = outfit_item_descriptions[item_index]

                # We need to unsqueeze at the 0-th position because ImageEncoder expects input as a batch. Unsqueeze basically add another dimension => This helps create a 1-item batch
                item_image_embedding = self.image_encoder(item_image.unsqueeze(0))

                item_text_embedding = self.text_encoder(item_description)

                # Concatenate the image embedding and text embedding to get item feature vector just as in the paper
                item_features = torch.cat([item_image_embedding, item_text_embedding])

                if outfit_index == 0 and item_index == 0:
                    logging.debug(
                        f"outfit_compatibility_model.py - forward - [3] - 1st outfit's 1st item: \n- item_image_embedding' shape: {item_image_embedding.shape}; \n- item_text_embedding's shape: {item_text_embedding.shape}; \n- item_features' shape: {item_features.shape}"
                    )
                
                cur_outfit_features.append(item_features)
            
            # Stack individual item embeddings to form the set of feature vectors F for the current outfit
            cur_outfit_features = torch.stack(cur_outfit_features)
            
            if outfit_index == 0:
                logging.debug(
                    f"outfit_compatibility_model.py - forward - [4] - cur_outfit_features' shape: {cur_outfit_features.shape}"
                )
                
            all_outfits_features.append(cur_outfit_features)
            
            
        all_outfits_features = torch.stack(
            all_outfits_features
        )
        
        logging.debug(
            f"outfit_compatibility_model.py - forward - [5] - all_outfits_features' shape: {all_outfits_features.shape}"
        )
        return
