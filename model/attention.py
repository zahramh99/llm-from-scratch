import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(x.size(-1))
        attention_weights = torch.softmax(scores, dim=-1)
        attended_values = torch.bmm(attention_weights, values)
        return attended_values
        