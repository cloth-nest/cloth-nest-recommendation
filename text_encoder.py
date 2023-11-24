import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel


# Define the Text Encoder using SentenceBERT
class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/bert-base-nli-mean-tokens"
        )
        self.sentence_bert = AutoModel.from_pretrained(
            "sentence-transformers/bert-base-nli-mean-tokens"
        )
        self.fc_layer = nn.Linear(768, 64)  # Assuming SentenceBERT output size is 768

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]

        # Tokenize the text data
        tokens = self.tokenizer(
            x, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        embeddings = self.sentence_bert(**tokens).last_hidden_state.mean(dim=1)
        x = self.fc_layer(embeddings)
        return x
