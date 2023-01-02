import numpy as np
import tensorflow as tf
from transformer.multihead_attention import MultiHeadAttention
from transformer.ffn import pointwise_feed_forward_network

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """Initialize a decoder layer"""
        super(DecoderLayer, self).__init__()
        self.multihead1 = MultiHeadAttention(d_model, num_heads)
        self.multihead2 = MultiHeadAttention(d_model, num_heads)
        
        self.ffn = pointwise_feed_forward_network(d_model, dff)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        
    def __call__(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """x - query vector for the Decoder
        enc_output - a set of attention vectors k and v from the top Encoder layer
        training - mode for Dropout
        look_ahead_mask/padding_mask - required for MultiHeadAttention
        """
        attn1, attn_weights_block1 = self.multihead1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x) # becomes the query vector for the next MultiHeadAttention
        
        attn2, attn_weights_block2 = self.multihead2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        
        return out3, attn_weights_block1, attn_weights_block2

