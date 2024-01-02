import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import logging
from sentence_transformers import SentenceTransformer

class TextEncoder(nn.Module):
    def __init__(self, output_embedding_dim=64):
        super(TextEncoder, self).__init__()

        self.sentence_bert = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # We need a focal layer to transform an SentenceBERT's output to our desired embedding dimension
        self.fc_layer = nn.Linear(
            self.sentence_bert.get_sentence_embedding_dimension(), output_embedding_dim
        )

    def forward(self, x):
        logging.debug(f"\n[TEXT ENCODER] forward START")
        logging.debug(f"text_encoder.py - forward - [1] - input x: {x}")

        sentence_embeddings = self.sentence_bert.encode(x, convert_to_tensor=True)

        logging.debug(
            f"text_encoder.py - forward - [2] - x as embeddings' shape: {sentence_embeddings}"
        )

        final_embeddings = self.fc_layer(sentence_embeddings)
        logging.debug(
            f"text_encoder.py - forward - [3] - final embeddings' shape after fc layer: {final_embeddings}"
        )
        logging.debug(f"[TEXT ENCODER] forward END")

        return final_embeddings
