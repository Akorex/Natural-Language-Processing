from decoder_layer import DecoderLayer
from encoder_layer_test import sample_encoder_layer_output
import tensorflow as tf

# let's test this out
sample_decoder_layer = DecoderLayer(d_model=512, num_heads=8, dff=2048)
sample_decoder_layer_output, _, _ = sample_decoder_layer(x=tf.random.uniform((64, 50, 512)), enc_output=sample_encoder_layer_output, 
                                                        training=False, look_ahead_mask=None, padding_mask=None)

sample_decoder_layer_output.shape