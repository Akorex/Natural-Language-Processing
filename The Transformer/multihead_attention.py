import torch.nn as nn
from attention import scaled_dot_product_attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        assert self.d_model % self.num_heads == 0
        self.depth = self.d_model//self.num_heads
        
        self.wk = nn.Linear(d_model, d_model)
        self.wq = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        self.dense = nn.Linear(d_model, d_model)
        
    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.depth).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_len, depth = x.size()    
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, query, key, value, mask = None):
        batch_size = query.size(0)
        
        # Dense on the query, key, value vectors
        q = self.wq(query)
        k = self.wk(key)
        v = self.wv(value)
        
        # split the heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        # compute attention
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        
        # combine heads, and add Dense layer
        output = self.combine_heads(scaled_attention)
        output = self.dense(output)
        
        return output, attention_weights