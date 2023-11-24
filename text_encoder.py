import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


# Text Encoder using SentenceBERT
class TextEncoder(nn.Module):
    def __init__(self, text_dim):
        super(TextEncoder, self).__init__()
        self.sentence_transformer = SentenceTransformer("paraphrase-MiniLM-L6-v2")

        self.fc_layer = nn.Linear(
            self.sentence_transformer.get_sentence_embedding_dimension(), text_dim
        )

    def forward(self, x):
        # If x is a list of strings, encode each string individually
        if isinstance(x, list):
            text_embeddings = [self.sentence_transformer.encode(text) for text in x]
            text_embeddings = torch.tensor(text_embeddings)
        else:
            # Assuming x is a single string
            text_embeddings = self.sentence_transformer.encode(x)
            text_embeddings = torch.tensor(text_embeddings).unsqueeze(0)

        text_embedding_fc = self.fc_layer(text_embeddings)
        return text_embedding_fc
