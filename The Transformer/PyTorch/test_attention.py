import torch
from attention import scaled_dot_product_attention

q = torch.randn(16, 32, 512)
k = torch.randn(16, 32, 512)
v = torch.randn(16, 32, 512)

print(scaled_dot_product_attention(q, k, v))