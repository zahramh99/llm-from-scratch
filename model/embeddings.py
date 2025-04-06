import torch.nn as nn

class Embedding(nn.Module):
    """Embedding layer that maps token IDs to dense vectors."""
    def __init__(self, vocab_size, embedding_dim):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embedding vectors
        """
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of token IDs (batch_size, seq_len)
            
        Returns:
            Embedded vectors (batch_size, seq_len, embedding_dim)
        """
        return self.embedding(x)