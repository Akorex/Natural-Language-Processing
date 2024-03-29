import torch.nn as nn
from embedding import PositionAndTokenEmbedding
from decoder_layer import DecoderLayer

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, tgt_vocab_size, maximum_position_encoding, rate = 0.1):
        super(Decoder, self).__init__()
        
        self.num_layers = num_layers
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        
        self.pos_token_embedding = PositionAndTokenEmbedding(d_model, maximum_position_encoding, tgt_vocab_size)
        self.dropout = nn.Dropout(p = rate)
        
    def forward(self, inputs, enc_output, src_mask, tgt_mask):
        inputs_embedded = self.pos_token_embedding(inputs)
        inputs_embedded = self.dropout(inputs_embedded)
        
        dec_output = inputs_embedded
        
        for i in range(self.num_layers):
            dec_output, _, _ = self.dec_layers[i](dec_output, enc_output, src_mask, tgt_mask)
        
        return dec_output