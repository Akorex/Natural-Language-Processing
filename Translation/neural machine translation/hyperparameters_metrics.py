import tensorflow as tf
from tokenizers import tokenizer_en, tokenizer_fr


# hyperparameters
d_model = 128
dff = 512
num_layers = 4
num_heads = 8
dropout_rate = 0.1

input_vocab_size = len(tokenizer_en.word_index) + 2
target_vocab_size = len(tokenizer_fr.word_index) + 2

pe_input = input_vocab_size
pe_target = target_vocab_size

maxlen = 25 # max tokens for prediction

# metrics and loss

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))
    
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')