import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
                 tgt_vocab_size, maximum_position_encoding, rate = 0.1):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, tgt_vocab_size, maximum_position_encoding, rate)
        
        self.final_layer = nn.Linear(d_model, tgt_vocab_size)
        
    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask
        
    def forward(self, inputs, targets):
        src_mask, tgt_mask = self.generate_mask(inputs, targets)
        
        enc_output = self.encoder(inputs, src_mask)
        dec_output = self.decoder(targets, enc_output, src_mask, tgt_mask)
        final_output = self.final_layer(dec_output)
        
        return final_output