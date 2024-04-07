# ensures encoder_layer.py works as intended
from transformer.encoder_layer import EncoderLayer
import tensorflow as tf



sample_encoder_layer = EncoderLayer(d_model=512, num_heads=8, dff=2048)
sample_encoder_layer_output = sample_encoder_layer(tf.random.uniform((64, 50, 512)), False, mask=None)
print(sample_encoder_layer_output.shape)