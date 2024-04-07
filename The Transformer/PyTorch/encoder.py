import torch.nn as nn
from embedding import PositionAndTokenEmbedding
from encoder_layer import EncoderLayer

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate = 0.1):
        """Initialize the Encoder of the Transformer"""
        
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.pos_token_embedding = PositionAndTokenEmbedding(d_model, maximum_position_encoding, input_vocab_size)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = nn.Dropout(p = rate)
        
    def forward(self, inputs, mask):
        inputs_embedded = self.pos_token_embedding(inputs)
        inputs_embedded = self.dropout(inputs_embedded)
        
        enc_output = inputs_embedded
        for i in range(self.num_layers):
            enc_output = self.enc_layers[i](enc_output, mask)
            
        return enc_output