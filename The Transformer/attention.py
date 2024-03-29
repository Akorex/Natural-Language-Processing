import torch
import torch.nn.functional as nn

def scaled_dot_product_attention(query, key, value, mask = None):
    """
    Returns the attention weights for the query, key, vectors.

    Args:
        query - (..., seq_len_q, depth)
        key - (..., seq_len_k, depth)
        value - (..., seq_len_v, depth_v)
    """

    dk = key.size()[-1]
    dk = torch.tensor(dk)
    key_transposed = key.transpose(-2, -1)
    
    matmul_qk = torch.matmul(query, key_transposed)
    scaled_attention_logits = matmul_qk/torch.sqrt(dk)

    if mask is not None:
        scaled_attention_logits = scaled_attention_logits.masked_fill(mask == 0, -1e9)

    attention_weights = nn.softmax(scaled_attention_logits, dim = -1)
    outputs = torch.matmul(attention_weights, value)

    return outputs, attention_weights