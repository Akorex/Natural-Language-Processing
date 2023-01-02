import numpy as np
import tensorflow as tf

def get_angles(pos, i, d_model):
    """Function to compute the angles to embed positional encoding to the tokens to be fed to the transformer"""
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    """Adds  positional encoding to the Embeddings to be fed to the Transformer model
    
    Computes a sin and cos of the angles determined by the get_angles() function
    and adds the value computed to an axis of the embeddings
    """
    
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],d_model)
    
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


if __name__ == '__main__':
    # check if the code works as intended
    pos_encoding = positional_encoding(50, 512)
    print(pos_encoding)