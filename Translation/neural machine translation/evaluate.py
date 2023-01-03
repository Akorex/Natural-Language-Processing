from transformer import Transformer
from hyperparameters_metrics import num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size
from hyperparameters_metrics import pe_input, pe_target, dropout_rate, maxlen
from tokenizers import tokenizer_en, tokenizer_fr
import tensorflow as tf
import numpy as np
from masking import create_masks
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings('ignore')

# create the transformer
model = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, dropout_rate)

# load the weights
path = r'model artifacts\exports\french_translator'
model.load_weights(path)

def evaluate(sentence):
    sentence = 'sos ' + sentence[0] + ' eos.'
    sentence = [sentence] # done because of the way TensorFlow's tokenizer
    
    # vectorize and pad the sentence
    sentence = tokenizer_fr.texts_to_sequences(sentence)
    sentence = pad_sequences(sentence, maxlen=30, padding='post', truncating='post')
    inp = tf.convert_to_tensor(np.array(sentence),dtype=tf.int64) # convert input to tensors
    
    # tokenize the start of the decoder input & convert to tensor
    decoder_input = tokenizer_en.texts_to_sequences(['sos'])
    decoder_input = tf.convert_to_tensor(np.array(decoder_input), dtype=tf.int64)
    
    for i in range(maxlen):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, decoder_input)
        predictions, _ = model(inp, decoder_input,False,enc_padding_mask,combined_mask, dec_padding_mask)
        
        # select the last word from the seq_len dimension
        predictions = predictions[: ,-1:, :] 
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int64)
        
        # return the result if the predicted_id is equal to the end token
        if predicted_id == tokenizer_en.texts_to_sequences(['eos']):
            return tf.squeeze(decoder_input, axis=0)
        
        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        decoder_input = tf.concat([decoder_input, predicted_id], axis=1)
    return tf.squeeze(decoder_input, axis=0)

def translate(sentence):
    sentence = [sentence] # our evaluate function requires lists
    print('Input: {}'.format(sentence[0]))
    print("Please wait while we translate: \n")
    result = (evaluate(sentence)).tolist()
    
    predicted_sentence = tokenizer_en.sequences_to_texts([[i for i in result if i != tokenizer_en.texts_to_sequences(['sos'])[0][0]
                                                          and i != tokenizer_en.texts_to_sequences(['eos.'])[0][0]]])  

    
    print('Predicted translation: {}'.format(predicted_sentence[0]))


if __name__ == '__main__':
    sentence = "Je vous aime."
    print(translate(sentence))