import numpy as np
import tensorflow as tf
from attention import scaled_dot_product_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Computes the attention for several heads in the transformer"""
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model%num_heads == 0 # ensures d_model can be split evenly across heads
        self.depth = self.d_model // self.num_heads
        
        self.wk = tf.keras.layers.Dense(d_model)
        self.wq = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)
         Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def get_config(self):
        """Implement serialization so we can save the model"""
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
        })
    
    def __call__(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        # Dense layer on the q, k, v vectors
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        # split the heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # compute attention weights
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        
        # reshape and add Dense layer
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention) #(batch_size, seq_len_q, d_model)
        
        return output, attention_weights


if __name__ == '__main__':
    # let's check if this works as intended
    temp_multihead = MultiHeadAttention(512, 8)
    y = tf.random.uniform((1, 60, 512))
    out, attn = temp_multihead(y, k=y, q=y, mask=None)
    
    print(out.shape, attn.shape)