import numpy as np
import tensorflow as tf

def pointwise_feed_forward_network(d_model, dff):
    model = tf.keras.Sequential([tf.keras.layers.Dense(dff, activation='relu'), 
                                tf.keras.layers.Dense(d_model)])
    return model


if __name__ == '__main__':
    sample_ffn = pointwise_feed_forward_network(512, 2048)
    print(sample_ffn(tf.random.uniform((64, 50, 512))).shape)