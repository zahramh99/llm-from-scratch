import torch.nn as nn
from .embeddings import Embedding
from .positional_enc import PositionalEncoding
from .transformer import TransformerBlock

class SimpleLLM(nn.Module):
    """Complete language model with transformer architecture."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension for transformer blocks
            num_layers: Number of transformer blocks
        """
        super(SimpleLLM, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(embedding_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        """Forward pass through entire model."""
        x = self.embedding(x)  # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        x = x.transpose(0, 1)  # (seq_len, batch_size, embedding_dim) for positional encoding
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # Back to (batch_size, seq_len, embedding_dim)
        x = self.transformer_blocks(x)
        x = self.output(x)  # (batch_size, seq_len, vocab_size)
        return x