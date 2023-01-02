from transformer import Transformer
import tensorflow as tf

sample_transformer = Transformer(num_layers=2, d_model=512, num_heads=8, dff=2048, 
    input_vocab_size=8500, target_vocab_size=8000, pe_input=10000, pe_target=6000)
    
temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)
    
fn_out, _ = sample_transformer(temp_input, temp_target, training=False, enc_padding_mask=None, look_ahead_mask=None,
                               dec_padding_mask=None)
                               
print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)