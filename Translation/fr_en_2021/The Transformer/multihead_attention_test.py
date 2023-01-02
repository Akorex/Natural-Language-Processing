# ensures multihead_attention works as intended

from multihead_attention import MultiHeadAttention
import tensorflow as tf


temp_multihead = MultiHeadAttention(512, 8)
y = tf.random.uniform((1, 60, 512))
out, attn = temp_multihead(y, k=y, q=y, mask=None)
    
print(out.shape, attn.shape)