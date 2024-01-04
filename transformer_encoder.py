import logging
import torch.nn as nn


class TransformerEncoder(nn.Module):
    # Initialize a Transformer Encoder with 6 layers & 16 heads as in the journal
    def __init__(self, input_size, num_layers=6, num_heads=16):
        super(TransformerEncoder, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=num_heads
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer, num_layers=num_layers
        )

    def forward(self, x, src_key_padding_mask):
        logging.debug(f"\n[TRANSFORMER ENCODER] forward START")
        logging.debug(
            f"transformer_encoder.py - forward - [1] - input x's: {x.shape}")

        output = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask)

        logging.debug(
            f"transformer_encoder.py - forward - [2] - output's shape: {output.shape}"
        )

        return output
