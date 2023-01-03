# accesses the tokenizers

import pickle

# for english
with open(r'model artifacts\tokenizer_en.pickle', 'rb') as handle:
    tokenizer_en = pickle.load(handle)
    
# for french
with open(r'model artifacts\tokenizer_fr.pickle', 'rb') as handle:
    tokenizer_fr = pickle.load(handle)


