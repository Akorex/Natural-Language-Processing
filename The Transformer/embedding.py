import torch
import torch.nn as nn
import math

class PositionAndTokenEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len, vocab_size):
        super(PositionAndTokenEmbedding, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype = torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        
        x = self.embedding(x)
        
        return x + self.pe[:, :x.size(1)]