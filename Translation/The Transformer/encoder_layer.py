import numpy as np
import tensorflow as tf
from multihead_attention import MultiHeadAttention
from ffn import pointwise_feed_forward_network


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """Initializes the encoder layer
        
        Args: 
            d_model: depth of the transformer model
            num_heads: number of heads for multi-head attention
            dff: depth of the feed forward network
            rate: dropout rate for training
        """
        super(EncoderLayer, self).__init__()
        self.multihead = MultiHeadAttention(d_model, num_heads)
        self.ffn = pointwise_feed_forward_network(d_model, dff)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        
    def __call__(self, x, training, mask):
        attn_output, _ = self.multihead(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(attn_output + x)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(ffn_output + out1)
        
        return out2

if __name__ == '__main__':
    sample_encoder_layer = EncoderLayer(d_model=512, num_heads=8, dff=2048)
    sample_encoder_layer_output = sample_encoder_layer(tf.random.uniform((64, 43, 512)), False, mask=None)
    print(sample_encoder_layer_output.shape)