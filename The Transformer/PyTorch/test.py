import torch
from transformer import Transformer


temp_inputs = torch.randint(low=0, high=200, size=(64, 38), dtype=torch.int64)
temp_targets = torch.randint(low=0, high=200, size=(64, 36), dtype=torch.int64)

temp_transformer = Transformer(num_layers = 2, d_model = 512, num_heads = 8, dff = 2048,
                              input_vocab_size = 8500, tgt_vocab_size = 8000, 
                               maximum_position_encoding = 10000, rate = 0.1)

transformer_output = temp_transformer(temp_inputs, temp_targets)

print(transformer_output.shape)