import numpy as np
import tensorflow as tf

def pointwise_feed_forward_network(d_model, dff):
    model = tf.keras.Sequential([tf.keras.layers.Dense(dff, activation='relu'), 
                                tf.keras.layers.Dense(d_model)])
    return model
