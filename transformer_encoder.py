import torch.nn as nn

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, input_size, num_layers=6, num_heads=16):
        super(TransformerEncoder, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=num_heads
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer, num_layers=num_layers
        )

    def forward(self, x):
        return self.transformer_encoder(x)
