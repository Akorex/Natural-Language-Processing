import numpy as np
import tensorflow as tf

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    
    # add extra dimension to add padding to the attention logits
    return seq[:, tf.newaxis, tf.newaxis, :] # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)
    
def create_masks(inp, tar):
    # encoder padding mask (also necessary for 2nd attention block in the decoder)
    enc_padding_mask = create_padding_mask(inp)
        
    # dec padding mask - used in the 2nd attention block in the decoder
    dec_padding_mask = create_padding_mask(inp)
        
    # used in the first attention block
    # used to pad and mask future tokens in the tokens received by the decoder
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        
    return enc_padding_mask, combined_mask, dec_padding_mask