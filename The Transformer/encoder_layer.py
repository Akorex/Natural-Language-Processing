import torch.nn as nn
from multihead_attention import MultiHeadAttention
from ffn import pointwise_feedforward_network

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate = 0.1):
        """Initializes the encoder layer
        
        Args:
            d_model: Embedding dimension for the transformer
            num_heads: number of heads for the MultiHeadAttention
            dff: number of units for the ffn
            rate: dropout rate
        """
        
        super(EncoderLayer, self).__init__()
        self.multihead = MultiHeadAttention(d_model, num_heads)
        self.ffn = pointwise_feedforward_network(d_model, dff)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p = rate)
        self.dropout2 = nn.Dropout(p = rate)
        
        
    def forward(self, inputs, mask):
        attn_output, _ = self.multihead(inputs, inputs, inputs, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(attn_output + inputs)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(ffn_output + out1)
        
        return out2