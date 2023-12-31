import logging
import torch
import torch.nn as nn

from image_encoder import ImageEncoder
from text_encoder import TextEncoder
from transformer_encoder import TransformerEncoder
import torch.nn.functional as F


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

    def forward(self, outfits_images, outfits_texts, outfits_items_nums):
        """
        outfits_images should have shape like this:
        (batch_size, max_num_of_items_per_outfit, img_channel, img_width, img_height)

        outfit_texts should be 2D list with size (batch_size, max_num_of_items_per_outfit) with each sublist contains all
        items' text descriptions of one outfit

        batch_size is the number of outfits per batch. This forward method
        works with 1 batch of outfits at a time
        """
        logging.info(f"\n[OUTFIT COMPATIBILITY MODEL] forward START")
        logging.info(
            f"outfit_compatibility_model.py - forward - [1] - input outfits_images' shape: {outfits_images.shape}"
        )

        all_outfits_features = []

        # For each outfit in this batch, ext
        for outfit_index in range(outfits_images.size(0)):
            # The "..." means getting the rest of the dimensions, in this case it gets the current outfit's all items' images
            outfit_item_images = outfits_images[outfit_index, ...]
            outfit_item_descriptions = outfits_texts[outfit_index]

            if outfit_index == 0:
                logging.debug(
                    f"outfit_compatibility_model.py - forward - [2] - 1st outfit: \n- outfit_item_images' shape: {outfit_item_images.shape}; \n- outfit_item_descriptions's length: {len(outfit_item_descriptions)}"
                )

            cur_outfit_features = []

            # For each item in this outfit, extract the item feature vector
            for item_index in range(outfit_item_images.size(0)):
                item_image = outfit_item_images[item_index, ...]
                item_description = outfit_item_descriptions[item_index]

                # We need to unsqueeze at the 0-th position because ImageEncoder expects input as a batch. Unsqueeze basically add another dimension => This helps create a 1-item batch. Then after we get the embedding we remove that first dimensio to later concate with text embedding
                item_image_embedding = self.image_encoder(
                    item_image.unsqueeze(0)
                ).squeeze(0)

                item_text_embedding = self.text_encoder(item_description)

                # Concatenate the image embedding and text embedding to get item feature vector just as in the paper
                item_features = torch.cat([item_image_embedding, item_text_embedding])

                if outfit_index == 0 and item_index == 0:
                    logging.debug(
                        f"outfit_compatibility_model.py - forward - [3] - 1st outfit's 1st item: \n- item_image_embedding' shape: {item_image_embedding.shape}; \n- item_text_embedding's shape: {item_text_embedding.shape}; \n- item_features' shape: {item_features.shape}"
                    )

                cur_outfit_features.append(item_features)

            # Stacking outfit's feature vectors to create a single tensor with shape (item_count, item feature vector's size)
            # item feature vector's size is 128 in this case
            cur_outfit_features = torch.stack(cur_outfit_features)

            # Prepend outfit token to set of outfit's feature vectors. The result will have a shape of (item_count + 1, item feature vector's size)
            outfit_token_value = self.outfit_token(torch.tensor([0]))
            cur_outfit_featuress_with_token = torch.cat(
                [
                    outfit_token_value,
                    cur_outfit_features,
                ],
                dim=0,
            )

            if outfit_index == 0:
                logging.debug(
                    f"outfit_compatibility_model.py - forward - [4A] - cur_outfit_features' shape: {cur_outfit_features.shape}"
                )
                logging.debug(
                    f"outfit_compatibility_model.py - forward - [4B] - cur_outfit_featuress_with_token' shape: {cur_outfit_featuress_with_token.shape}"
                )

            all_outfits_features.append(cur_outfit_featuress_with_token)

        # Stacking all outfit's feature vectors to create a single tensor with shape (outfit_count, item_count + 1, item feature vector's size). We have item_count + 1 because of the added outfit token
        all_outfits_features = torch.stack(all_outfits_features)

        logging.info(
            f"outfit_compatibility_model.py - forward - [5] - all_outfits_features' shape: {all_outfits_features.shape}"
        )

        # Generate mask to avoid attending to padded elements
        # Need to decrease by 1 because all_outfits_features.size(1) = item_count + 1
        max_item_count = all_outfits_features.size(1) - 1
        outfit_count = all_outfits_features.size(0)

        # Need to add 1 to max_item_count because we also need to create mask for the outfit token
        mask_tensor = torch.ones(outfit_count, max_item_count + 1)

        for i, original_length in enumerate(outfits_items_nums):
            mask_tensor[i, original_length:] = 0
            mask_tensor[i, max_item_count - 1] = 1

        logging.debug(
            f"outfit_compatibility_model.py - forward - [6] \n- mask_tensor's shape: {mask_tensor.shape} "
        )
        # Convert the mask tensor to boolean
        src_key_padding_mask = mask_tensor.bool()

        # We need to transpose all_outfits_features (transpose = switching 2 dimensions like you are rotating the matrix) since transformer_encoder takes input with shape (sequence_length/item_count, batch_size/outfit_count, embedding_dimension)
        # Then we will transpose the result back after we're done with the transformer
        transformer_output = self.transformer_encoder(
            all_outfits_features.transpose(0, 1),
            src_key_padding_mask=src_key_padding_mask,
        ).transpose(0, 1)

        logging.info(
            f"outfit_compatibility_model.py - forward - [6] \n- src_mask_key_padding's shape: {src_key_padding_mask.shape} \n- transformer_output' shape: {transformer_output.shape}"
        )

        return

    def generate_mask(self, item_count):
        # Create a square mask with dimensions (item_count + 1, item_count + 1)
        mask = torch.triu(torch.ones(item_count + 1, item_count + 1), diagonal=1).bool()
        return mask
