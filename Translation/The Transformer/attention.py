import numpy as np
import tensorflow as tf
np.set_printoptions(suppress=True)

def scaled_dot_product_attention(q, k, v, mask):
    """Computes the attention weight for the q, k, v vectors
    
    Attention in the transformer is popularly known as self-attention because the q, k, v vectors are
    sourced from the same sequence. Self Attention, also called intra Attention, is an attention mechanism relating 
    different positions of a single sequence in order to compute a representation of the same sequence.
    
    q, k, v must have leading dimensions - same 'batch_size'
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v
    
    q - query vectors; shape == (..., seq_len_q, depth)
    k - key vectors; shape == (..., seq_len_k, depth)
    v - value vectors; shape == (..., seq_len_v, depth_v)
    
    Returns - attention weights, output
    """
    
    matmul_qk = tf.matmul(q, k, transpose_b = True)
    dk = tf.cast(tf.shape(k)[-1], dtype=tf.float32)
    scaled_dk = tf.math.sqrt(dk)
    
    scaled_attention_logits = matmul_qk/scaled_dk
    
    # add mask
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
        
    # softmax the attention logits so it adds up to 1
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    
    
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(q, k, v, None)
    print('Attention weights are:')
    print(temp_attn)
    print('\nOutput is:')
    print(temp_out)


if __name__ == '__main__':
    temp_k = tf.constant([[10, 0, 0],
                      [0, 10, 0],
                      [0, 0, 10],
                      [0, 0, 10]], dtype=tf.float32)  # (4, 3)
                      
    temp_v = tf.constant([[1, 0],
                      [10, 0],
                      [100, 5],
                      [1000, 6]], dtype=tf.float32) # (4, 2)
                      
    temp_q = tf.constant([[0, 0, 10],
                      [0, 10, 0],
                      [10, 10, 0]], dtype=tf.float32)
                      
    print_out(temp_q, temp_k, temp_v)