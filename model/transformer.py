import torch.nn as nn
from .attention import SelfAttention

class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feed-forward layers."""
    def __init__(self, embedding_dim, hidden_dim):
        """
        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dim: Hidden dimension of feed-forward layer
        """
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        """Forward pass through transformer block."""
        attended = self.attention(x)
        x = self.norm1(x + attended)  # Add & norm
        forwarded = self.feed_forward(x)
        x = self.norm2(x + forwarded)  # Add & norm
        return x