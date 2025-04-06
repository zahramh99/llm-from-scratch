import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    def __init__(self, embedding_dim, max_seq_len=5000):
        """
        Args:
            embedding_dim: Dimension of input embeddings
            max_seq_len: Maximum sequence length to support
        """
        super(PositionalEncoding, self).__init__()
        
        position = torch.arange(max_seq_len).unsqueeze(1)  # (max_seq_len, 1)
        
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * 
            (-math.log(10000.0) / embedding_dim)  # <- fixed here
        )

        pe = torch.zeros(max_seq_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_seq_len, 1, embedding_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor (seq_len, batch_size, embedding_dim)
            
        Returns:
            Position-aware embeddings
        """
        return x + self.pe[:x.size(0), :]
