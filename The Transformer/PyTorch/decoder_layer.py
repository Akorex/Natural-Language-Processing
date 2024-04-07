import torch.nn as nn
from multihead_attention import MultiHeadAttention
from ffn import pointwise_feedforward_network

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate = 0.1):
        super(DecoderLayer, self).__init__()
        
        self.multihead1 = MultiHeadAttention(d_model, num_heads)
        self.multihead2 = MultiHeadAttention(d_model, num_heads)
        
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        
        self.ffn = pointwise_feedforward_network(d_model, dff)
        
        self.dropout1 = nn.Dropout(p = rate)
        self.dropout2 = nn.Dropout(p = rate)
        self.dropout3 = nn.Dropout(p = rate)
        
    def forward(self, inputs, enc_output, src_mask, tgt_mask):
        attn1, attn_weights_block1 = self.multihead1(inputs, inputs, inputs, tgt_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + inputs) # becomes the query vector in the next step
        
        attn2, attn_weights_block2 = self.multihead2(out1, enc_output, enc_output, src_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)
        
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)
        
        return out3, attn_weights_block1, attn_weights_block2